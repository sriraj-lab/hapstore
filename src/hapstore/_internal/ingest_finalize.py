"""Streaming ingest finalization helpers."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from numcodecs import VLenUTF8, Zstd

from hapstore._internal.chrom_attrs import build_chrom_attrs
from hapstore._internal.chunking import resolve_matrix_chunk_layout
from hapstore._internal.genetic_map import interpolate_cm
from hapstore._internal.ingest_models import (
    _IngestArrays,
    _PartitionPlan,
    _ShardArtifactMeta,
    _ShardSpec,
    _StreamingFinalizeContext,
)
from hapstore.chrom import normalize_chrom
from hapstore.store import HapStore, _haplotypes_to_genotypes

logger = logging.getLogger(__name__)


def _to_unicode_array_safe(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    out = np.empty(arr.shape, dtype=object)
    it = np.nditer(arr, flags=["refs_ok", "multi_index", "zerosize_ok"])
    for item in it:
        value = item.item()
        out[it.multi_index] = "" if value is None else str(value)
    return out


def _validate_cross_chrom_samples(store: HapStore, sample_ids: np.ndarray, chrom_key: str) -> None:
    existing = [
        key for key in store.chromosomes
        if str(key) != str(chrom_key) and f"{key}/samples/ids" in store.root
    ]
    if not existing:
        return
    existing_ids = np.asarray(store.root[f"{existing[0]}/samples/ids"][:], dtype=object)
    existing_ids = _to_unicode_array_safe(existing_ids)
    if existing_ids.shape != sample_ids.shape or not np.array_equal(existing_ids, sample_ids):
        raise ValueError("sample IDs must match across chromosomes in the same HapStore")


def _create_store_from_ingested(
    output_zarr: str | Path,
    chrom: str | int,
    ingested: _IngestArrays,
    genetic_positions: np.ndarray | None,
    store_genotypes: bool,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
    attrs: dict | None,
) -> HapStore:
    store = HapStore(output_zarr, mode="a")
    store.create_chromosome(
        chrom=normalize_chrom(chrom, prefix=True),
        haplotypes=ingested.haplotypes,
        variant_pos=ingested.positions,
        sample_ids=ingested.sample_ids,
        ref=ingested.ref,
        alt=ingested.alt,
        vid=ingested.vid,
        genetic_positions=genetic_positions,
        store_genotypes=store_genotypes,
        chunk_variant_size=chunk_variant_size,
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=compression_level,
        attrs=attrs,
    )
    return store


def _estimate_initial_variant_capacity(
    shard_specs: list[_ShardSpec],
    variant_chunk: int,
    bp_per_variant_upper: int = 5_000,
) -> int:
    if int(variant_chunk) <= 0:
        raise ValueError(f"variant_chunk must be > 0, got {variant_chunk}")
    if int(bp_per_variant_upper) <= 0:
        raise ValueError(f"bp_per_variant_upper must be > 0, got {bp_per_variant_upper}")

    starts = [int(spec.start_bp) for spec in shard_specs if spec.start_bp is not None]
    ends = [int(spec.end_bp) for spec in shard_specs if spec.end_bp is not None]
    if not starts or not ends:
        return int(max(1, int(variant_chunk)))

    span_bp = int(max(1, max(ends) - min(starts)))
    heuristic = int((span_bp + int(bp_per_variant_upper) - 1) // int(bp_per_variant_upper))
    return int(max(1, int(variant_chunk), heuristic))


def _create_streaming_finalize_context(
    output_zarr: str | Path,
    chrom: str | int,
    subset_ids: np.ndarray,
    shard_specs: list[_ShardSpec],
    map_df,
    store_genotypes: bool,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
) -> _StreamingFinalizeContext:
    chrom_key = normalize_chrom(chrom, prefix=True)
    sample_ids_u = _to_unicode_array_safe(np.asarray(subset_ids, dtype=object))
    n_samples = int(sample_ids_u.shape[0])
    n_haplotypes = 2 * n_samples

    if any(spec.shard_index != i for i, spec in enumerate(shard_specs)):
        raise ValueError("staged partition manifest has non-canonical shard_id order")
    prev_start_bp: int | None = None
    for spec in shard_specs:
        if spec.start_bp is None:
            continue
        current_start = int(spec.start_bp)
        if prev_start_bp is not None and current_start <= prev_start_bp:
            raise ValueError(
                "staged partition manifest shard starts are not strictly increasing: "
                f"{current_start} after {prev_start_bp}"
            )
        prev_start_bp = current_start

    initial_capacity = _estimate_initial_variant_capacity(
        shard_specs=shard_specs,
        variant_chunk=int(chunk_variant_size),
    )
    chunk_layout = resolve_matrix_chunk_layout(
        n_haplotypes=n_haplotypes,
        n_samples=n_samples,
        n_variants=initial_capacity,
        chunk_variant_size=int(chunk_variant_size),
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
    )
    variant_chunk = int(chunk_layout.variant)
    hap_chunk = int(chunk_layout.haplotype)
    sample_chunk_geno = int(chunk_layout.sample)
    compressor = Zstd(level=int(compression_level))

    store = HapStore(output_zarr, mode="a")
    if store.has_chromosome(chrom_key):
        raise ValueError(
            f"chromosome {chrom_key!r} already exists in store. Delete it first or use a different store."
        )
    _validate_cross_chrom_samples(store, sample_ids_u, chrom_key)

    chrom_group = store.root.create_group(chrom_key)
    hap_arr = chrom_group.create_dataset(
        "haplotypes",
        shape=(n_haplotypes, initial_capacity),
        chunks=(hap_chunk, variant_chunk),
        dtype=np.uint8,
        compressor=compressor,
        overwrite=False,
    )

    geno_arr = None
    if store_genotypes:
        geno_arr = chrom_group.create_dataset(
            "genotypes",
            shape=(n_samples, initial_capacity),
            chunks=(sample_chunk_geno, variant_chunk),
            dtype=np.uint8,
            compressor=compressor,
            overwrite=False,
        )

    variants = chrom_group.create_group("variants")
    pos_arr = variants.create_dataset("pos", shape=(initial_capacity,), chunks=(variant_chunk,), dtype=np.int64, compressor=compressor, overwrite=False)
    ref_arr = variants.create_dataset("ref", shape=(initial_capacity,), chunks=(variant_chunk,), dtype=object, object_codec=VLenUTF8(), compressor=compressor, overwrite=False)
    alt_arr = variants.create_dataset("alt", shape=(initial_capacity,), chunks=(variant_chunk,), dtype=object, object_codec=VLenUTF8(), compressor=compressor, overwrite=False)
    vid_arr = variants.create_dataset("vid", shape=(initial_capacity,), chunks=(variant_chunk,), dtype=object, object_codec=VLenUTF8(), compressor=compressor, overwrite=False)

    genetic_arr = None
    if map_df is not None:
        genetic_arr = variants.create_dataset(
            "genetic_pos",
            shape=(initial_capacity,),
            chunks=(variant_chunk,),
            dtype=np.float64,
            compressor=compressor,
            overwrite=False,
        )

    sample_group = chrom_group.create_group("samples")
    sample_chunk = int(min(max(1, n_samples), 8192))
    sample_group.create_dataset(
        "ids",
        data=np.asarray(sample_ids_u, dtype=object),
        dtype=object,
        object_codec=VLenUTF8(),
        chunks=(sample_chunk,),
        compressor=compressor,
        overwrite=False,
    )

    return _StreamingFinalizeContext(
        store=store,
        chrom_key=chrom_key,
        chrom_group=chrom_group,
        hap_arr=hap_arr,
        geno_arr=geno_arr,
        pos_arr=pos_arr,
        ref_arr=ref_arr,
        alt_arr=alt_arr,
        vid_arr=vid_arr,
        genetic_arr=genetic_arr,
        n_samples=n_samples,
        n_haplotypes=n_haplotypes,
        n_shards=len(shard_specs),
        variant_chunk=variant_chunk,
        hap_chunk=hap_chunk,
        sample_chunk_geno=sample_chunk_geno,
        compression_level=int(compression_level),
        chunk_haplotype_size=int(hap_chunk),
        chunk_sample_size=(int(sample_chunk_geno) if store_genotypes else None),
        capacity=initial_capacity,
    )


def _ensure_streaming_finalize_capacity(ctx: _StreamingFinalizeContext, required_variants: int) -> None:
    if int(required_variants) <= int(ctx.capacity):
        return
    new_capacity = int(ctx.capacity)
    while new_capacity < int(required_variants):
        new_capacity = int(max(int(required_variants), new_capacity * 2))

    ctx.hap_arr.resize((ctx.n_haplotypes, new_capacity))
    if ctx.geno_arr is not None:
        ctx.geno_arr.resize((ctx.n_samples, new_capacity))
    ctx.pos_arr.resize((new_capacity,))
    ctx.ref_arr.resize((new_capacity,))
    ctx.alt_arr.resize((new_capacity,))
    ctx.vid_arr.resize((new_capacity,))
    if ctx.genetic_arr is not None:
        ctx.genetic_arr.resize((new_capacity,))
    ctx.capacity = int(new_capacity)


def _stream_finalize_write_shard(
    ctx: _StreamingFinalizeContext,
    shard_index: int,
    meta: _ShardArtifactMeta,
    chrom: str | int,
    map_df,
    started_at: float,
    *,
    missing_sentinel: int,
) -> None:
    with np.load(meta.artifact_path, allow_pickle=False) as shard_data:
        h = np.asarray(shard_data["haplotypes"], dtype=np.uint8)
        positions = np.asarray(shard_data["positions"], dtype=np.int64)
        ref = np.asarray(shard_data["ref"], dtype=object)
        alt = np.asarray(shard_data["alt"], dtype=object)
        vid = np.asarray(shard_data["vid"], dtype=object)

    _stream_finalize_write_arrays(
        ctx=ctx,
        shard_index=shard_index,
        haplotypes=h,
        positions=positions,
        ref=ref,
        alt=alt,
        vid=vid,
        chrom=chrom,
        map_df=map_df,
        started_at=started_at,
        missing_sentinel=missing_sentinel,
        missing_ref_count=int(meta.missing_ref_count),
        missing_alt_count=int(meta.missing_alt_count),
        missing_vid_count=int(meta.missing_vid_count),
        unphased_gt_count=int(meta.unphased_gt_count),
    )


def _stream_finalize_write_arrays(
    ctx: _StreamingFinalizeContext,
    shard_index: int,
    haplotypes: np.ndarray,
    positions: np.ndarray,
    ref: np.ndarray,
    alt: np.ndarray,
    vid: np.ndarray,
    chrom: str | int,
    map_df,
    started_at: float,
    *,
    missing_sentinel: int,
    missing_ref_count: int = 0,
    missing_alt_count: int = 0,
    missing_vid_count: int = 0,
    unphased_gt_count: int = 0,
) -> None:
    """Validate and append one in-memory variant block to a streaming store."""
    h = np.asarray(haplotypes, dtype=np.uint8)
    positions = np.asarray(positions, dtype=np.int64)
    ref = np.asarray(ref, dtype=object)
    alt = np.asarray(alt, dtype=object)
    vid = np.asarray(vid, dtype=object)

    width = int(positions.shape[0])
    if h.shape != (ctx.n_haplotypes, width):
        raise ValueError(
            f"shard {shard_index} haplotype shape mismatch: expected {(ctx.n_haplotypes, width)}, got {h.shape}"
        )
    if ref.shape[0] != width or alt.shape[0] != width or vid.shape[0] != width:
        raise ValueError(
            f"shard {shard_index} variant-field length mismatch: "
            f"pos={width}, ref={ref.shape[0]}, alt={alt.shape[0]}, vid={vid.shape[0]}"
        )
    if width > 1 and np.any(np.diff(positions) < 0):
        raise ValueError(f"shard {shard_index} positions are not non-decreasing")
    if width > 0 and ctx.prev_last_position is not None and int(positions[0]) < ctx.prev_last_position:
        raise ValueError(
            "global position order violated at shard boundary: "
            f"shard {shard_index} starts at {int(positions[0])} after {ctx.prev_last_position}"
        )

    next_offset = int(ctx.offset + width)
    _ensure_streaming_finalize_capacity(ctx, required_variants=next_offset)

    if width > 0:
        sl = slice(int(ctx.offset), next_offset)
        ctx.hap_arr[:, sl] = h
        if ctx.geno_arr is not None:
            ctx.geno_arr[:, sl] = _haplotypes_to_genotypes(h)
        ctx.pos_arr[sl] = positions
        ctx.ref_arr[sl] = ref
        ctx.alt_arr[sl] = alt
        ctx.vid_arr[sl] = vid
        if ctx.genetic_arr is not None:
            cm = interpolate_cm(positions, map_df, chrom=chrom)
            ctx.genetic_arr[sl] = np.asarray(cm, dtype=np.float64)
        ctx.prev_last_position = int(positions[-1])
        if not ctx.has_missing and bool(np.any(h == int(missing_sentinel))):
            ctx.has_missing = True

    ctx.offset = int(next_offset)
    ctx.missing_ref_count += int(missing_ref_count)
    ctx.missing_alt_count += int(missing_alt_count)
    ctx.missing_vid_count += int(missing_vid_count)
    ctx.unphased_gt_count += int(unphased_gt_count)
    logger.info(
        "Finalize shard %d/%d complete: variants=%d cumulative=%d elapsed=%.1fs",
        int(shard_index) + 1,
        int(ctx.n_shards),
        width,
        int(ctx.offset),
        time.perf_counter() - started_at,
    )


def _complete_streaming_finalize(
    ctx: _StreamingFinalizeContext,
    attrs: dict,
    workers: int,
    partition_plan: _PartitionPlan,
    *,
    missing_sentinel: int,
    parallel_mode: str = "staged_materialize_finalize",
) -> HapStore:
    final_variants = int(ctx.offset)
    ctx.hap_arr.resize((ctx.n_haplotypes, final_variants))
    if ctx.geno_arr is not None:
        ctx.geno_arr.resize((ctx.n_samples, final_variants))
    ctx.pos_arr.resize((final_variants,))
    ctx.ref_arr.resize((final_variants,))
    ctx.alt_arr.resize((final_variants,))
    ctx.vid_arr.resize((final_variants,))
    if ctx.genetic_arr is not None:
        ctx.genetic_arr.resize((final_variants,))

    payload = dict(attrs)
    payload["missing_ref_count"] = int(ctx.missing_ref_count)
    payload["missing_alt_count"] = int(ctx.missing_alt_count)
    payload["missing_vid_count"] = int(ctx.missing_vid_count)
    payload["unphased_gt_count"] = int(ctx.unphased_gt_count)
    payload["parallel_workers"] = int(workers)
    payload["parallel_shards"] = int(ctx.n_shards)
    payload["parallel_mode"] = str(parallel_mode)
    payload["partition_bp"] = int(partition_plan.partition_bp)
    payload["chunk_variant_size"] = int(ctx.variant_chunk)
    payload["chunk_haplotype_size"] = int(ctx.hap_chunk)
    payload["chunk_sample_size"] = int(ctx.sample_chunk_geno)

    source = payload.pop("source", None)
    build = payload.pop("build", None)
    full_matrix_chunks = ctx.hap_chunk == ctx.n_haplotypes and (
        ctx.geno_arr is None or ctx.sample_chunk_geno == ctx.n_samples
    )
    layout_version = 1 if full_matrix_chunks else 2
    ctx.chrom_group.attrs.update(
        build_chrom_attrs(
            n_samples=int(ctx.n_samples),
            n_haplotypes=int(ctx.n_haplotypes),
            n_variants=int(final_variants),
            missing_sentinel=int(missing_sentinel),
            has_missing=bool(ctx.has_missing),
            compression_level=int(ctx.compression_level),
            chunk_variant_size=int(ctx.variant_chunk),
            chunk_haplotype_size=ctx.chunk_haplotype_size,
            chunk_sample_size=(
                int(ctx.sample_chunk_geno) if ctx.geno_arr is not None else None
            ),
            layout_version=int(layout_version),
            source=source,
            build=build,
            extra=payload or None,
        )
    )
    ctx.finished = True
    return ctx.store


def _abort_streaming_finalize(ctx: _StreamingFinalizeContext | None) -> None:
    if ctx is None or ctx.finished:
        return
    if ctx.chrom_key in ctx.store.root:
        del ctx.store.root[ctx.chrom_key]
