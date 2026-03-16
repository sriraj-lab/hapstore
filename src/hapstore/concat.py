"""HapStore concatenation utilities."""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numcodecs import Zstd

from hapstore._internal.seeding import stable_seed
from hapstore._internal.chrom import normalize_chrom

from .merge import (
    _InputChrom,
    _MISSING_SENTINEL,
    _create_utf8_array,
    _create_utf8_array_empty,
    _find_duplicate_values,
    _merge_populations_or_none,
    _validate_and_collect_inputs,
    _validate_variant_axis_streaming,
    _write_variant_axis_from_reference,
)
from .store import HapStore, _haplotypes_to_genotypes


Axis = Literal["samples", "variants", 0, 1]
ValidateMode = Literal["spot", "full", "none"]


@dataclass(frozen=True)
class _SourceWritePlan:
    source_idx: int
    source_path: Path
    out_hap_offset: int
    out_variant_offset: int
    source_n_haps: int
    source_n_vars: int


@dataclass(frozen=True)
class _ResolvedLayout:
    compressor: Zstd
    compression_level: int
    variant_chunk_attr: int
    chunk_haplotype_size_attr: int | None
    chunk_sample_size_attr: int | None


def concat_hapstores(
    input_stores: Sequence[str | Path | HapStore],
    output_store: str | Path,
    chrom: str | int,
    axis: Axis = "samples",
    validate: ValidateMode = "spot",
    n_workers: int = 1,
    compression_level: int | None = 3,
    chunk_variant_size: int | None = None,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    store_genotypes: bool = False,
    strict_variant_ids: bool = True,
    forbid_duplicate_variants: bool = True,
    attrs: dict | None = None,
    overwrite: bool = False,
    tmp_suffix: str = ".tmp",
) -> HapStore:
    """Concatenate multiple HapStores along the sample or variant axis.

    This is a concat (stacking) operation, not a join/merge-by-key.
    """
    if not input_stores:
        raise ValueError("input_stores must not be empty")
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if tmp_suffix == "":
        raise ValueError("tmp_suffix must not be empty")

    axis_norm = _normalize_axis(axis)
    validate_norm = _normalize_validate(validate)
    chrom_key = normalize_chrom(chrom, prefix=True)

    output_path = Path(output_store)
    tmp_path = Path(str(output_path) + str(tmp_suffix))
    _prepare_output_paths(output_path=output_path, tmp_path=tmp_path, overwrite=overwrite)

    sources = _validate_and_collect_inputs(input_stores=input_stores, chrom_key=chrom_key)
    resolved = _resolve_layout(
        sources=sources,
        chrom_key=chrom_key,
        compression_level=compression_level,
        chunk_variant_size=chunk_variant_size,
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        store_genotypes=store_genotypes,
    )

    tmp_store = HapStore(tmp_path, mode="w")
    if axis_norm == "samples":
        _concat_samples_into_store(
            sources=sources,
            out=tmp_store,
            chrom_key=chrom_key,
            validate=validate_norm,
            n_workers=min(int(n_workers), len(sources)),
            strict_variant_ids=bool(strict_variant_ids),
            store_genotypes=bool(store_genotypes),
            resolved=resolved,
            attrs=attrs,
        )
    else:
        _concat_variants_into_store(
            sources=sources,
            out=tmp_store,
            chrom_key=chrom_key,
            validate=validate_norm,
            n_workers=min(int(n_workers), len(sources)),
            forbid_duplicate_variants=bool(forbid_duplicate_variants),
            store_genotypes=bool(store_genotypes),
            resolved=resolved,
            attrs=attrs,
        )

    del tmp_store
    gc.collect()

    tmp_path.replace(output_path)
    return HapStore(output_path, mode="r")


def _normalize_axis(axis: Axis) -> Literal["samples", "variants"]:
    if axis in (0, "samples"):
        return "samples"
    if axis in (1, "variants"):
        return "variants"
    raise ValueError("axis must be 'samples', 'variants', 0, or 1")


def _normalize_validate(validate: ValidateMode) -> ValidateMode:
    if validate not in ("spot", "full", "none"):
        raise ValueError("validate must be 'spot', 'full', or 'none'")
    return validate


def _prepare_output_paths(*, output_path: Path, tmp_path: Path, overwrite: bool) -> None:
    if not overwrite:
        if output_path.exists():
            raise FileExistsError(f"output_store already exists: {output_path}")
        if tmp_path.exists():
            raise FileExistsError(f"tmp output already exists: {tmp_path}")
        return

    if output_path.exists():
        _rm_tree_or_file(output_path)
    if tmp_path.exists():
        _rm_tree_or_file(tmp_path)


def _rm_tree_or_file(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _resolve_layout(
    *,
    sources: Sequence[_InputChrom],
    chrom_key: str,
    compression_level: int | None,
    chunk_variant_size: int | None,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    store_genotypes: bool,
) -> _ResolvedLayout:
    attrs0 = sources[0].store.attrs(chrom_key)
    default_level = int(attrs0.get("compression_level", 3))
    default_variant_chunk = int(attrs0.get("chunk_variant_size", 500))
    default_hap_chunk = attrs0.get("chunk_haplotype_size", None)
    default_sample_chunk = attrs0.get("chunk_sample_size", None)

    if compression_level is None:
        levels = [int(src.store.attrs(chrom_key).get("compression_level", 3)) for src in sources]
        if any(x != levels[0] for x in levels[1:]):
            raise ValueError(
                "inconsistent source compression_level; pass compression_level explicitly "
                f"(values={sorted(set(levels))})"
            )

    if chunk_variant_size is None:
        chunks = [int(src.store.attrs(chrom_key).get("chunk_variant_size", 500)) for src in sources]
        if any(x != chunks[0] for x in chunks[1:]):
            raise ValueError(
                "inconsistent source chunk_variant_size; pass chunk_variant_size explicitly "
                f"(values={sorted(set(chunks))})"
            )

    if chunk_haplotype_size is None:
        hap_chunks = [src.store.attrs(chrom_key).get("chunk_haplotype_size", None) for src in sources]
        if any(x != hap_chunks[0] for x in hap_chunks[1:]):
            raise ValueError(
                "inconsistent source chunk_haplotype_size; pass chunk_haplotype_size explicitly "
                f"(values={sorted(set(hap_chunks), key=lambda x: (-1 if x is None else int(x)))})"
            )

    if chunk_sample_size is None:
        sample_chunks = [src.store.attrs(chrom_key).get("chunk_sample_size", None) for src in sources]
        if any(x != sample_chunks[0] for x in sample_chunks[1:]):
            raise ValueError(
                "inconsistent source chunk_sample_size; pass chunk_sample_size explicitly "
                f"(values={sorted(set(sample_chunks), key=lambda x: (-1 if x is None else int(x)))})"
            )

    resolved_level = int(default_level if compression_level is None else compression_level)
    if resolved_level < 0:
        raise ValueError(f"compression_level must be >= 0, got {resolved_level}")

    resolved_variant_chunk = int(default_variant_chunk if chunk_variant_size is None else chunk_variant_size)
    if resolved_variant_chunk <= 0:
        raise ValueError(f"chunk_variant_size must be > 0, got {resolved_variant_chunk}")

    if chunk_haplotype_size is None:
        hap_chunk_attr = None if default_hap_chunk is None else int(default_hap_chunk)
    else:
        hap_chunk_attr = int(chunk_haplotype_size)
        if hap_chunk_attr <= 0:
            raise ValueError(f"chunk_haplotype_size must be > 0 when provided, got {hap_chunk_attr}")

    if chunk_sample_size is None:
        sample_chunk_attr = None if default_sample_chunk is None else int(default_sample_chunk)
    else:
        sample_chunk_attr = int(chunk_sample_size)
        if sample_chunk_attr <= 0:
            raise ValueError(f"chunk_sample_size must be > 0 when provided, got {sample_chunk_attr}")

    if store_genotypes and hap_chunk_attr is not None and (hap_chunk_attr % 2 != 0):
        raise ValueError(
            "chunk_haplotype_size must be even when store_genotypes=True "
            f"(got {hap_chunk_attr})"
        )

    return _ResolvedLayout(
        compressor=Zstd(level=resolved_level),
        compression_level=resolved_level,
        variant_chunk_attr=int(resolved_variant_chunk),
        chunk_haplotype_size_attr=hap_chunk_attr,
        chunk_sample_size_attr=sample_chunk_attr,
    )


def _concat_samples_into_store(
    *,
    sources: Sequence[_InputChrom],
    out: HapStore,
    chrom_key: str,
    validate: ValidateMode,
    n_workers: int,
    strict_variant_ids: bool,
    store_genotypes: bool,
    resolved: _ResolvedLayout,
    attrs: dict | None,
) -> None:
    logger = logging.getLogger(__name__)
    if out.has_chromosome(chrom_key):
        raise ValueError(
            f"chromosome {chrom_key!r} already exists in output store {out.path}; "
            "delete it first or use a different output path"
        )

    # All sources must have identical variant axis size.
    reference_n_variants = int(sources[0].n_variants)
    for idx, src in enumerate(sources[1:], start=1):
        if int(src.n_variants) != reference_n_variants:
            raise ValueError(
                "n_variants mismatch across input stores: "
                f"reference({sources[0].path})={reference_n_variants}, store[{idx}]({src.path})={src.n_variants}"
            )

    n_variants = reference_n_variants
    total_samples = int(sum(src.n_samples for src in sources))
    total_haps = int(sum(src.n_haplotypes for src in sources))
    if total_haps != (2 * total_samples):
        raise ValueError(
            "inconsistent haplotype/sample counts across inputs: "
            f"total_haplotypes={total_haps}, expected={2 * total_samples}"
        )

    merged_sample_ids = np.concatenate([src.sample_ids for src in sources], axis=0)
    duplicates = _find_duplicate_values(merged_sample_ids)
    if duplicates:
        raise ValueError(
            "duplicate sample IDs across input stores are not allowed; "
            f"first duplicate={duplicates[0]!r}"
        )

    merged_populations = _merge_populations_or_none(sources)
    has_missing = bool(any(src.has_missing for src in sources))

    variant_chunk = int(min(max(1, resolved.variant_chunk_attr), max(1, n_variants)))
    if strict_variant_ids:
        if validate == "full":
            _validate_variant_axis_streaming(sources=sources, variant_chunk=variant_chunk)
        elif validate == "spot":
            _spot_validate_variant_axis(sources=sources, variant_chunk=variant_chunk)

    hap_chunk = (
        int(total_haps)
        if resolved.chunk_haplotype_size_attr is None
        else int(min(max(1, resolved.chunk_haplotype_size_attr), total_haps))
    )
    sample_chunk_geno = (
        int(total_samples)
        if resolved.chunk_sample_size_attr is None
        else int(min(max(1, resolved.chunk_sample_size_attr), total_samples))
    )

    chrom_group = out.root.create_group(chrom_key)
    chrom_group.create_dataset(
        "haplotypes",
        shape=(total_haps, n_variants),
        dtype=np.uint8,
        chunks=(hap_chunk, variant_chunk),
        compressor=resolved.compressor,
        overwrite=False,
    )
    if store_genotypes:
        chrom_group.create_dataset(
            "genotypes",
            shape=(total_samples, n_variants),
            dtype=np.uint8,
            chunks=(sample_chunk_geno, variant_chunk),
            compressor=resolved.compressor,
            overwrite=False,
        )

    variant_group = chrom_group.create_group("variants")
    _write_variant_axis_from_reference(
        variant_group=variant_group,
        reference=sources[0],
        n_variants=n_variants,
        variant_chunk=variant_chunk,
        compressor=resolved.compressor,
    )

    sample_group = chrom_group.create_group("samples")
    sample_chunk = int(min(max(1, total_samples), 8192))
    _create_utf8_array(sample_group, "ids", merged_sample_ids, sample_chunk, resolved.compressor)
    if merged_populations is not None:
        _create_utf8_array(sample_group, "population", merged_populations, sample_chunk, resolved.compressor)

    plans = _build_sample_axis_plans(sources)
    _copy_dense_matrices(
        out_path=out.path,
        chrom_key=chrom_key,
        axis="samples",
        plans=plans,
        n_workers=min(int(n_workers), len(plans)),
        hap_chunk=hap_chunk,
        variant_chunk=variant_chunk,
        store_genotypes=store_genotypes,
        sample_chunk_geno=sample_chunk_geno,
        logger=logger,
    )

    if validate == "spot":
        _spot_validate_dense_copy(
            sources=sources,
            out=out,
            chrom_key=chrom_key,
            axis="samples",
            plans=plans,
            hap_chunk=hap_chunk,
            variant_chunk=variant_chunk,
        )

    attrs_payload = dict(attrs or {})
    source_attr = attrs_payload.pop("source", None)
    build_attr = attrs_payload.pop("build", None)
    layout_version = 1 if hap_chunk == total_haps and sample_chunk_geno == total_samples else 2
    chrom_group.attrs.update(
        {
            "n_samples": int(total_samples),
            "n_haplotypes": int(total_haps),
            "n_variants": int(n_variants),
            "phased": True,
            "ploidy": 2,
            "missing_sentinel": _MISSING_SENTINEL,
            "has_missing": bool(has_missing),
            "compression": "zstd",
            "compression_level": int(resolved.compression_level),
            "chunk_variant_size": int(variant_chunk),
            "chunk_haplotype_size": resolved.chunk_haplotype_size_attr,
            "chunk_sample_size": resolved.chunk_sample_size_attr,
            "layout_version": int(layout_version),
            "source": source_attr,
            "build": build_attr,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extra": attrs_payload or None,
        }
    )


def _concat_variants_into_store(
    *,
    sources: Sequence[_InputChrom],
    out: HapStore,
    chrom_key: str,
    validate: ValidateMode,
    n_workers: int,
    forbid_duplicate_variants: bool,
    store_genotypes: bool,
    resolved: _ResolvedLayout,
    attrs: dict | None,
) -> None:
    logger = logging.getLogger(__name__)
    if out.has_chromosome(chrom_key):
        raise ValueError(
            f"chromosome {chrom_key!r} already exists in output store {out.path}; "
            "delete it first or use a different output path"
        )

    reference_ids = sources[0].sample_ids
    for idx, src in enumerate(sources[1:], start=1):
        if src.sample_ids.shape[0] != reference_ids.shape[0] or not np.array_equal(src.sample_ids, reference_ids):
            raise ValueError(
                "sample IDs mismatch across input stores for variant concat: "
                f"reference={sources[0].path}, store[{idx}]={src.path}"
            )

    ref_has_pop = sources[0].populations is not None
    for idx, src in enumerate(sources[1:], start=1):
        if (src.populations is not None) != ref_has_pop:
            raise ValueError(
                "population annotation presence mismatch across input stores: "
                f"reference_has_population={ref_has_pop}, store[{idx}]={src.path}"
            )
        if ref_has_pop and not np.array_equal(src.populations, sources[0].populations):
            raise ValueError(
                "population annotation values mismatch across input stores: "
                f"reference={sources[0].path}, store[{idx}]={src.path}"
            )

    ref_has_genetic = "genetic_pos" in sources[0].variants_group
    for idx, src in enumerate(sources[1:], start=1):
        if ("genetic_pos" in src.variants_group) != ref_has_genetic:
            raise ValueError(
                "genetic position presence mismatch across input stores: "
                f"reference_has_genetic={ref_has_genetic}, store[{idx}]={src.path}"
            )

    if forbid_duplicate_variants and len(sources) >= 2:
        _enforce_pos_nonoverlap_boundary(sources)

    if validate == "full" and forbid_duplicate_variants:
        _enforce_no_duplicate_vids_full(sources)

    n_samples = int(sources[0].n_samples)
    n_haps = int(sources[0].n_haplotypes)
    total_variants = int(sum(src.n_variants for src in sources))
    has_missing = bool(any(src.has_missing for src in sources))

    variant_chunk = int(min(max(1, resolved.variant_chunk_attr), max(1, total_variants)))
    hap_chunk = (
        int(n_haps)
        if resolved.chunk_haplotype_size_attr is None
        else int(min(max(1, resolved.chunk_haplotype_size_attr), n_haps))
    )
    sample_chunk_geno = (
        int(n_samples)
        if resolved.chunk_sample_size_attr is None
        else int(min(max(1, resolved.chunk_sample_size_attr), n_samples))
    )

    chrom_group = out.root.create_group(chrom_key)
    chrom_group.create_dataset(
        "haplotypes",
        shape=(n_haps, total_variants),
        dtype=np.uint8,
        chunks=(hap_chunk, variant_chunk),
        compressor=resolved.compressor,
        overwrite=False,
    )
    if store_genotypes:
        chrom_group.create_dataset(
            "genotypes",
            shape=(n_samples, total_variants),
            dtype=np.uint8,
            chunks=(sample_chunk_geno, variant_chunk),
            compressor=resolved.compressor,
            overwrite=False,
        )

    variant_group = chrom_group.create_group("variants")
    pos_out = variant_group.create_dataset(
        "pos",
        shape=(total_variants,),
        dtype=np.int64,
        chunks=(variant_chunk,),
        compressor=resolved.compressor,
        overwrite=False,
    )
    ref_out = _create_utf8_array_empty(variant_group, "ref", total_variants, variant_chunk, resolved.compressor)
    alt_out = _create_utf8_array_empty(variant_group, "alt", total_variants, variant_chunk, resolved.compressor)
    vid_out = _create_utf8_array_empty(variant_group, "vid", total_variants, variant_chunk, resolved.compressor)
    genetic_out = None
    if ref_has_genetic:
        genetic_out = variant_group.create_dataset(
            "genetic_pos",
            shape=(total_variants,),
            dtype=np.float64,
            chunks=(variant_chunk,),
            compressor=resolved.compressor,
            overwrite=False,
        )

    v_offset = 0
    for src in sources:
        v_next = v_offset + src.n_variants
        pos_out[v_offset:v_next] = np.asarray(src.variants_group["pos"][:], dtype=np.int64)
        ref_out[v_offset:v_next] = np.asarray(src.variants_group["ref"][:], dtype=object)
        alt_out[v_offset:v_next] = np.asarray(src.variants_group["alt"][:], dtype=object)
        vid_out[v_offset:v_next] = np.asarray(src.variants_group["vid"][:], dtype=object)
        if genetic_out is not None:
            genetic_out[v_offset:v_next] = np.asarray(src.variants_group["genetic_pos"][:], dtype=np.float64)
        v_offset = v_next

    sample_group = chrom_group.create_group("samples")
    sample_chunk = int(min(max(1, n_samples), 8192))
    _create_utf8_array(sample_group, "ids", reference_ids, sample_chunk, resolved.compressor)
    if sources[0].populations is not None:
        _create_utf8_array(sample_group, "population", sources[0].populations, sample_chunk, resolved.compressor)

    plans = _build_variant_axis_plans(sources)
    _copy_dense_matrices(
        out_path=out.path,
        chrom_key=chrom_key,
        axis="variants",
        plans=plans,
        n_workers=min(int(n_workers), len(plans)),
        hap_chunk=hap_chunk,
        variant_chunk=variant_chunk,
        store_genotypes=store_genotypes,
        sample_chunk_geno=sample_chunk_geno,
        logger=logger,
    )

    if validate == "spot":
        _spot_validate_dense_copy(
            sources=sources,
            out=out,
            chrom_key=chrom_key,
            axis="variants",
            plans=plans,
            hap_chunk=hap_chunk,
            variant_chunk=variant_chunk,
        )

    attrs_payload = dict(attrs or {})
    source_attr = attrs_payload.pop("source", None)
    build_attr = attrs_payload.pop("build", None)
    layout_version = 1 if hap_chunk == n_haps and sample_chunk_geno == n_samples else 2
    chrom_group.attrs.update(
        {
            "n_samples": int(n_samples),
            "n_haplotypes": int(n_haps),
            "n_variants": int(total_variants),
            "phased": True,
            "ploidy": 2,
            "missing_sentinel": _MISSING_SENTINEL,
            "has_missing": bool(has_missing),
            "compression": "zstd",
            "compression_level": int(resolved.compression_level),
            "chunk_variant_size": int(variant_chunk),
            "chunk_haplotype_size": resolved.chunk_haplotype_size_attr,
            "chunk_sample_size": resolved.chunk_sample_size_attr,
            "layout_version": int(layout_version),
            "source": source_attr,
            "build": build_attr,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extra": attrs_payload or None,
        }
    )


def _build_sample_axis_plans(sources: Sequence[_InputChrom]) -> list[_SourceWritePlan]:
    plans: list[_SourceWritePlan] = []
    hap_offset = 0
    for i, src in enumerate(sources):
        plans.append(
            _SourceWritePlan(
                source_idx=i,
                source_path=src.path,
                out_hap_offset=hap_offset,
                out_variant_offset=0,
                source_n_haps=int(src.n_haplotypes),
                source_n_vars=int(src.n_variants),
            )
        )
        hap_offset += src.n_haplotypes
    return plans


def _build_variant_axis_plans(sources: Sequence[_InputChrom]) -> list[_SourceWritePlan]:
    plans: list[_SourceWritePlan] = []
    v_offset = 0
    for i, src in enumerate(sources):
        plans.append(
            _SourceWritePlan(
                source_idx=i,
                source_path=src.path,
                out_hap_offset=0,
                out_variant_offset=v_offset,
                source_n_haps=int(src.n_haplotypes),
                source_n_vars=int(src.n_variants),
            )
        )
        v_offset += src.n_variants
    return plans


def _copy_dense_matrices(
    *,
    out_path: Path,
    chrom_key: str,
    axis: Literal["samples", "variants"],
    plans: Sequence[_SourceWritePlan],
    n_workers: int,
    hap_chunk: int,
    variant_chunk: int,
    store_genotypes: bool,
    sample_chunk_geno: int,
    logger: logging.Logger,
) -> None:
    if n_workers <= 1:
        for plan in plans:
            _worker_copy_source(
                source_path=plan.source_path,
                out_path=out_path,
                chrom_key=chrom_key,
                axis=axis,
                out_hap_offset=plan.out_hap_offset,
                out_variant_offset=plan.out_variant_offset,
                hap_chunk=hap_chunk,
                variant_chunk=variant_chunk,
                store_genotypes=store_genotypes,
                sample_chunk_geno=sample_chunk_geno,
                log_queue=None,
            )
        return

    if axis == "samples":
        total_haps = int(sum(p.source_n_haps for p in plans))
        # Parallel-by-source is only safe/efficient when each source covers an integer
        # number of output hap-chunks (so that no output chunk file is written by
        # multiple sources).
        chunk_disjoint = bool(all((int(p.source_n_haps) % int(hap_chunk)) == 0 for p in plans))
        if not chunk_disjoint:
            _copy_dense_matrices_parallel_by_output_hap_chunks(
                out_path=out_path,
                chrom_key=chrom_key,
                plans=plans,
                n_workers=n_workers,
                hap_chunk=hap_chunk,
                variant_chunk=variant_chunk,
                total_haps=total_haps,
                store_genotypes=store_genotypes,
                sample_chunk_geno=sample_chunk_geno,
                logger=logger,
            )
            return

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    listener = _start_queue_listener(queue=queue)
    try:
        args = []
        for plan in plans:
            args.append(
                dict(
                    source_path=plan.source_path,
                    out_path=out_path,
                    chrom_key=chrom_key,
                    axis=axis,
                    out_hap_offset=plan.out_hap_offset,
                    out_variant_offset=plan.out_variant_offset,
                    hap_chunk=hap_chunk,
                    variant_chunk=variant_chunk,
                    store_genotypes=store_genotypes,
                    sample_chunk_geno=sample_chunk_geno,
                    log_queue=queue,
                )
            )
        logger.info("COPY_START n_sources=%d n_workers=%d axis=%s", len(plans), n_workers, axis)
        with ctx.Pool(processes=int(n_workers)) as pool:
            pool.map(_worker_copy_source_star, args)
        logger.info("COPY_DONE n_sources=%d axis=%s", len(plans), axis)
    finally:
        listener.stop()
        manager.shutdown()


@dataclass(frozen=True)
class _OutputChunkTask:
    out_h0: int
    out_h1: int
    overlaps: tuple[tuple[Path, int, int, int, int], ...]


def _copy_dense_matrices_parallel_by_output_hap_chunks(
    *,
    out_path: Path,
    chrom_key: str,
    plans: Sequence[_SourceWritePlan],
    n_workers: int,
    hap_chunk: int,
    variant_chunk: int,
    total_haps: int,
    store_genotypes: bool,
    sample_chunk_geno: int,
    logger: logging.Logger,
) -> None:
    del sample_chunk_geno
    if total_haps <= 0:
        raise ValueError("total_haps must be > 0")

    tasks: list[_OutputChunkTask] = []
    for out_h0 in range(0, int(total_haps), int(hap_chunk)):
        out_h1 = min(int(out_h0) + int(hap_chunk), int(total_haps))
        overlaps: list[tuple[Path, int, int, int, int]] = []
        for p in plans:
            src_global0 = int(p.out_hap_offset)
            src_global1 = int(p.out_hap_offset) + int(p.source_n_haps)
            if src_global1 <= int(out_h0) or src_global0 >= int(out_h1):
                continue
            ov0 = max(int(out_h0), src_global0)
            ov1 = min(int(out_h1), src_global1)
            out_local0 = ov0 - int(out_h0)
            out_local1 = ov1 - int(out_h0)
            src_local0 = ov0 - src_global0
            src_local1 = ov1 - src_global0
            overlaps.append((p.source_path, out_local0, out_local1, src_local0, src_local1))
        if not overlaps:
            raise AssertionError(f"no overlaps for output hap slice [{out_h0}, {out_h1})")
        tasks.append(_OutputChunkTask(out_h0=int(out_h0), out_h1=int(out_h1), overlaps=tuple(overlaps)))

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    listener = _start_queue_listener(queue=queue)
    try:
        args = []
        for task in tasks:
            args.append(
                dict(
                    out_path=out_path,
                    chrom_key=chrom_key,
                    out_h0=int(task.out_h0),
                    out_h1=int(task.out_h1),
                    overlaps=task.overlaps,
                    variant_chunk=int(variant_chunk),
                    log_queue=queue,
                )
            )
        worker_count = min(int(n_workers), len(args))
        logger.info(
            "COPY_START n_output_hap_chunks=%d n_workers=%d axis=samples mode=output_hap_chunks",
            len(args),
            worker_count,
        )
        with ctx.Pool(processes=int(worker_count)) as pool:
            pool.map(_worker_copy_output_hap_chunk_star, args)
        logger.info("COPY_DONE n_output_hap_chunks=%d axis=samples mode=output_hap_chunks", len(args))
    finally:
        listener.stop()
        manager.shutdown()


def _worker_copy_output_hap_chunk_star(kwargs: dict) -> None:
    _worker_copy_output_hap_chunk(**kwargs)


def _worker_copy_output_hap_chunk(
    *,
    out_path: Path,
    chrom_key: str,
    out_h0: int,
    out_h1: int,
    overlaps: tuple[tuple[Path, int, int, int, int], ...],
    variant_chunk: int,
    log_queue,
) -> None:
    import logging.handlers

    logger = logging.getLogger(__name__)
    if log_queue is not None:
        logger.handlers = []
        logger.addHandler(logging.handlers.QueueHandler(log_queue))
        logger.setLevel(logging.INFO)
        logger.propagate = False

    out = HapStore(out_path, mode="a")
    out_group = out.root[chrom_key]
    out_haps = out_group["haplotypes"]
    n_vars = int(out_haps.shape[1])
    block_haps = int(out_h1) - int(out_h0)

    sources: dict[Path, HapStore] = {}
    src_haps: dict[Path, Any] = {}
    for src_path, _, _, _, _ in overlaps:
        if src_path in sources:
            continue
        hs = HapStore(src_path, mode="r")
        sources[src_path] = hs
        src_haps[src_path] = hs.root[chrom_key]["haplotypes"]

    logger.info(
        "COPY_PROGRESS axis=samples mode=output_hap_chunks out_hap_offset=%d out_hap_end=%d n_sources=%d",
        int(out_h0),
        int(out_h1),
        len(sources),
    )

    for v0 in range(0, n_vars, int(variant_chunk)):
        v1 = min(int(v0) + int(variant_chunk), n_vars)
        block = np.empty((block_haps, int(v1) - int(v0)), dtype=np.uint8)
        for src_path, out_local0, out_local1, src_local0, src_local1 in overlaps:
            block[int(out_local0) : int(out_local1), :] = np.asarray(
                src_haps[src_path][int(src_local0) : int(src_local1), int(v0) : int(v1)],
                dtype=np.uint8,
            )
        out_haps[int(out_h0) : int(out_h1), int(v0) : int(v1)] = block

        if "genotypes" in out_group:
            if (int(out_h0) % 2) != 0 or (int(out_h1) % 2) != 0:
                raise ValueError(
                    "output hap slice must align to diploid sample boundaries when writing genotypes: "
                    f"out_h0={out_h0}, out_h1={out_h1}"
                )
            sample0 = int(out_h0) // 2
            sample1 = int(out_h1) // 2
            out_group["genotypes"][sample0:sample1, int(v0) : int(v1)] = _haplotypes_to_genotypes(block)


def _start_queue_listener(queue):
    import logging.handlers
    import sys

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s [%(levelname)s] pid=%(process)d %(name)s: %(message)s")
    )
    listener = logging.handlers.QueueListener(queue, handler, respect_handler_level=True)
    listener.start()
    return listener


def _worker_copy_source_star(kwargs: dict) -> None:
    _worker_copy_source(**kwargs)


def _worker_copy_source(
    *,
    source_path: Path,
    out_path: Path,
    chrom_key: str,
    axis: Literal["samples", "variants"],
    out_hap_offset: int,
    out_variant_offset: int,
    hap_chunk: int,
    variant_chunk: int,
    store_genotypes: bool,
    sample_chunk_geno: int,
    log_queue,
) -> None:
    import logging.handlers

    logger = logging.getLogger(__name__)
    if log_queue is not None:
        logger.handlers = []
        logger.addHandler(logging.handlers.QueueHandler(log_queue))
        logger.setLevel(logging.INFO)
        logger.propagate = False

    src = HapStore(source_path, mode="r")
    out = HapStore(out_path, mode="a")
    src_group = src.root[chrom_key]
    out_group = out.root[chrom_key]
    src_haps = src_group["haplotypes"]
    out_haps = out_group["haplotypes"]

    src_n_haps = int(src_haps.shape[0])
    src_n_vars = int(src_haps.shape[1])

    logger.info(
        "COPY_PROGRESS source=%s axis=%s out_hap_offset=%d out_variant_offset=%d",
        str(source_path),
        axis,
        int(out_hap_offset),
        int(out_variant_offset),
    )

    if axis == "samples":
        for h0 in range(0, src_n_haps, hap_chunk):
            h1 = min(h0 + hap_chunk, src_n_haps)
            out_h0 = out_hap_offset + h0
            out_h1 = out_hap_offset + h1
            for v0 in range(0, src_n_vars, variant_chunk):
                v1 = min(v0 + variant_chunk, src_n_vars)
                out_haps[out_h0:out_h1, v0:v1] = np.asarray(src_haps[h0:h1, v0:v1], dtype=np.uint8)

            if store_genotypes:
                geno_out = out_group["genotypes"]
                sample0 = out_h0 // 2
                sample1 = out_h1 // 2
                for v0 in range(0, src_n_vars, variant_chunk):
                    v1 = min(v0 + variant_chunk, src_n_vars)
                    h_block = np.asarray(src_haps[h0:h1, v0:v1], dtype=np.uint8)
                    geno_out[sample0:sample1, v0:v1] = _haplotypes_to_genotypes(h_block)

    else:
        out_var0 = int(out_variant_offset)
        out_n_haps = int(out_haps.shape[0])
        for h0 in range(0, out_n_haps, hap_chunk):
            h1 = min(h0 + hap_chunk, out_n_haps)
            for v0 in range(0, src_n_vars, variant_chunk):
                v1 = min(v0 + variant_chunk, src_n_vars)
                out_haps[h0:h1, out_var0 + v0 : out_var0 + v1] = np.asarray(
                    src_haps[h0:h1, v0:v1],
                    dtype=np.uint8,
                )

        if store_genotypes:
            geno_out = out_group["genotypes"]
            for s0 in range(0, int(geno_out.shape[0]), sample_chunk_geno):
                s1 = min(s0 + sample_chunk_geno, int(geno_out.shape[0]))
                h0 = 2 * s0
                h1 = 2 * s1
                for v0 in range(0, src_n_vars, variant_chunk):
                    v1 = min(v0 + variant_chunk, src_n_vars)
                    h_block = np.asarray(src_haps[h0:h1, v0:v1], dtype=np.uint8)
                    geno_out[s0:s1, out_var0 + v0 : out_var0 + v1] = _haplotypes_to_genotypes(h_block)

    logger.info("COPY_DONE source=%s axis=%s", str(source_path), axis)


def _spot_validate_variant_axis(*, sources: Sequence[_InputChrom], variant_chunk: int) -> None:
    reference = sources[0]
    n_variants = int(reference.n_variants)
    n_chunks = int((n_variants + variant_chunk - 1) // variant_chunk)
    if n_chunks <= 0:
        return

    seed = stable_seed(
        0,
        "concat_hapstores",
        "spot_variant_axis",
        reference.chrom_key,
        str(reference.path),
        *(str(s.path) for s in sources),
    )
    rng = np.random.default_rng(seed)
    picks = {0, max(0, n_chunks - 1)}
    if n_chunks > 2:
        extra = rng.choice(n_chunks, size=min(3, n_chunks), replace=False).tolist()
        picks.update(int(x) for x in extra)

    ref_variants = reference.variants_group
    has_genetic = "genetic_pos" in ref_variants
    for idx, src in enumerate(sources[1:], start=1):
        src_variants = src.variants_group
        if ("genetic_pos" in src_variants) != has_genetic:
            raise ValueError(
                "genetic position presence mismatch across input stores: "
                f"reference_has_genetic={has_genetic}, store[{idx}]_has_genetic={'genetic_pos' in src_variants}"
            )

    for chunk_idx in sorted(picks):
        start = int(chunk_idx * variant_chunk)
        end = int(min(start + variant_chunk, n_variants))
        ref_pos = np.asarray(ref_variants["pos"][start:end], dtype=np.int64)
        ref_ref = np.asarray(ref_variants["ref"][start:end], dtype=object)
        ref_alt = np.asarray(ref_variants["alt"][start:end], dtype=object)
        ref_vid = np.asarray(ref_variants["vid"][start:end], dtype=object)
        ref_gen = None if not has_genetic else np.asarray(ref_variants["genetic_pos"][start:end], dtype=np.float64)

        for idx, src in enumerate(sources[1:], start=1):
            src_variants = src.variants_group
            pos = np.asarray(src_variants["pos"][start:end], dtype=np.int64)
            if not np.array_equal(pos, ref_pos):
                raise ValueError(
                    "variant position mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            ref = np.asarray(src_variants["ref"][start:end], dtype=object)
            if not np.array_equal(ref, ref_ref):
                raise ValueError(
                    "REF allele mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            alt = np.asarray(src_variants["alt"][start:end], dtype=object)
            if not np.array_equal(alt, ref_alt):
                raise ValueError(
                    "ALT allele mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            vid = np.asarray(src_variants["vid"][start:end], dtype=object)
            if not np.array_equal(vid, ref_vid):
                raise ValueError(
                    "variant ID mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            if ref_gen is not None:
                gen = np.asarray(src_variants["genetic_pos"][start:end], dtype=np.float64)
                if not np.array_equal(gen, ref_gen):
                    raise ValueError(
                        "genetic position values mismatch across input stores: "
                        f"reference={reference.path}, store[{idx}]={src.path}"
                    )


def _spot_validate_dense_copy(
    *,
    sources: Sequence[_InputChrom],
    out: HapStore,
    chrom_key: str,
    axis: Literal["samples", "variants"],
    plans: Sequence[_SourceWritePlan],
    hap_chunk: int,
    variant_chunk: int,
) -> None:
    out_haps = out.root[chrom_key]["haplotypes"]

    seed = stable_seed(
        0,
        "concat_hapstores",
        chrom_key,
        axis,
        str(out.path),
        *(str(s.path) for s in sources),
    )
    rng = np.random.default_rng(seed)

    n_checks = 5
    for _ in range(n_checks):
        plan = plans[int(rng.integers(low=0, high=len(plans)))]
        src = HapStore(plan.source_path, mode="r")
        src_haps = src.root[chrom_key]["haplotypes"]

        if axis == "samples":
            src_h0 = int(rng.integers(0, max(1, int(src_haps.shape[0]))))
            src_h1 = int(min(int(src_haps.shape[0]), src_h0 + max(1, hap_chunk // 4)))
            v0 = int(rng.integers(0, max(1, int(src_haps.shape[1]))))
            v1 = int(min(int(src_haps.shape[1]), v0 + max(1, variant_chunk // 2)))
            out_h0 = plan.out_hap_offset + src_h0
            out_h1 = plan.out_hap_offset + src_h1
            got = np.asarray(out_haps[out_h0:out_h1, v0:v1], dtype=np.uint8)
            exp = np.asarray(src_haps[src_h0:src_h1, v0:v1], dtype=np.uint8)
        else:
            h0 = int(rng.integers(0, max(1, int(out_haps.shape[0]))))
            h1 = int(min(int(out_haps.shape[0]), h0 + max(1, hap_chunk // 4)))
            src_v0 = int(rng.integers(0, max(1, int(src_haps.shape[1]))))
            src_v1 = int(min(int(src_haps.shape[1]), src_v0 + max(1, variant_chunk // 2)))
            out_v0 = plan.out_variant_offset + src_v0
            out_v1 = plan.out_variant_offset + src_v1
            got = np.asarray(out_haps[h0:h1, out_v0:out_v1], dtype=np.uint8)
            exp = np.asarray(src_haps[h0:h1, src_v0:src_v1], dtype=np.uint8)

        if not np.array_equal(got, exp):
            raise AssertionError("spot validation failed: output haplotypes slab mismatch")


def _enforce_pos_nonoverlap_boundary(sources: Sequence[_InputChrom]) -> None:
    for i in range(len(sources) - 1):
        left = sources[i].variants_group["pos"]
        right = sources[i + 1].variants_group["pos"]
        left_last = int(np.asarray(left[-1], dtype=np.int64))
        right_first = int(np.asarray(right[0], dtype=np.int64))
        if left_last >= right_first:
            raise ValueError(
                "variant position ranges overlap or are not strictly increasing across inputs: "
                f"store[{i}] last_pos={left_last}, store[{i+1}] first_pos={right_first}"
            )


def _enforce_no_duplicate_vids_full(sources: Sequence[_InputChrom]) -> None:
    seen: set[str] = set()
    for idx, src in enumerate(sources):
        vids = np.asarray(src.variants_group["vid"][:], dtype=object).tolist()
        for v in vids:
            text = str(v)
            if text in seen:
                raise ValueError(
                    "duplicate variant IDs across input stores are not allowed for variant concat; "
                    f"first duplicate={text!r} at store[{idx}]={src.path}"
                )
            seen.add(text)
