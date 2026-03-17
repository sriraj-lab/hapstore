"""Converters into hapstore."""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr
from numcodecs import VLenUTF8, Zstd

from hapstore.chrom import normalize_chrom

from ._internal._ingest_index import _has_tabix_index
from ._internal._ingest_readers import (
    _coerce_ref_alt,
    _coerce_vid,
    _extract_vcf_arrays,
    _extract_vcf_arrays_cyvcf2,
    _extract_vcf_arrays_pysam,
    _resolve_subset,
)
from ._internal._ingest_sharding import (
    _build_shard_ranges,
    _infer_contig_bounds_from_records,
    _resolve_contig_span_bounds_for_parallel,
    _select_contig_for_parallel,
)
from ._internal.ingest_finalize import (
    _abort_streaming_finalize,
    _complete_streaming_finalize,
    _create_store_from_ingested,
    _create_streaming_finalize_context,
    _ensure_streaming_finalize_capacity,
    _estimate_initial_variant_capacity,
    _haplotypes_to_genotypes_local,
    _stream_finalize_write_shard,
    _to_unicode_array,
    _validate_cross_chrom_samples,
)
from ._internal.ingest_models import (
    Backend,
    MissingRefAltPolicy,
    MissingVidPolicy,
    UnphasedPolicy,
    _IngestArrays,
    _MISSING_SENTINEL,
    _PartitionPlan,
    _ShardArtifactMeta,
    _ShardResult,
    _ShardSpec,
    _StreamingFinalizeContext,
)
from .store import HapStore

logger = logging.getLogger(__name__)

_DEFAULT_PARTITION_BP = 1_000_000
_AUTO_PARTITION_ROUND_BP = 100_000


def _allocated_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        pass
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return None
    return int(cpu_count)


def _warn_reader_worker_budget(workers: int, backend: Backend) -> None:
    allocated = _allocated_cpu_count()
    if allocated is None or allocated <= 1:
        return
    max_readers = max(1, int(allocated) - 1)
    if int(workers) <= max_readers:
        return
    logger.warning(
        "Parallel %s ingest uses %d reader workers plus one writer. "
        "Reader workers should not exceed allocated CPU - 1 (%d).",
        backend,
        int(workers),
        max_readers,
    )


def _load_genetic_map_df(
    genetic_map_path: str | Path | None,
    *,
    chrom: str | int,
) -> Any | None:
    if genetic_map_path is None:
        return None

    from hapstore._internal.genetic_map import load_genetic_map

    map_path = Path(genetic_map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"genetic map not found: {map_path}")
    return load_genetic_map(map_path, chrom=chrom)


def _interpolate_genetic_positions(
    positions: np.ndarray,
    *,
    map_df: Any | None,
    chrom: str | int,
) -> np.ndarray | None:
    if map_df is None:
        return None

    from hapstore._internal.genetic_map import interpolate_cm

    return interpolate_cm(positions, map_df, chrom=chrom)


def _build_ingest_attrs_payload(
    *,
    attrs: dict | None,
    ingested: _IngestArrays,
    parallel_workers: int,
    parallel_shards: int,
    parallel_mode: str,
    partition_bp: int,
    total_seconds: float,
    partition_seconds: float | None = None,
) -> dict[str, Any]:
    payload = dict(attrs or {})
    payload["missing_ref_count"] = int(ingested.missing_ref_count)
    payload["missing_alt_count"] = int(ingested.missing_alt_count)
    payload["missing_vid_count"] = int(ingested.missing_vid_count)
    payload["unphased_gt_count"] = int(ingested.unphased_gt_count)
    payload["parallel_workers"] = int(parallel_workers)
    payload["parallel_shards"] = int(parallel_shards)
    payload["parallel_mode"] = str(parallel_mode)
    payload["partition_bp"] = int(partition_bp)
    payload["total_seconds"] = float(total_seconds)
    if partition_seconds is not None:
        payload["partition_seconds"] = float(partition_seconds)
    return payload


def _single_shard_store_from_extract(
    *,
    output_zarr: str | Path,
    chrom: str | int,
    genetic_map_path: str | Path | None,
    store_genotypes: bool,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
    attrs: dict | None,
    partition_plan: _PartitionPlan,
    total_start: float,
    extract_fn: Callable[[], _IngestArrays],
    parallel_mode: str = "single_shard",
    partition_seconds: float | None = None,
) -> HapStore:
    ingested = extract_fn()
    map_df = _load_genetic_map_df(genetic_map_path, chrom=chrom)
    genetic_positions = _interpolate_genetic_positions(ingested.positions, map_df=map_df, chrom=chrom)
    attrs_payload = _build_ingest_attrs_payload(
        attrs=attrs,
        ingested=ingested,
        parallel_workers=1,
        parallel_shards=1,
        parallel_mode=parallel_mode,
        partition_bp=int(partition_plan.partition_bp),
        partition_seconds=partition_seconds,
        total_seconds=float(time.perf_counter() - total_start),
    )
    return _create_store_from_ingested(
        output_zarr=output_zarr,
        chrom=chrom,
        ingested=ingested,
        genetic_positions=genetic_positions,
        store_genotypes=store_genotypes,
        chunk_variant_size=chunk_variant_size,
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=compression_level,
        attrs=attrs_payload,
    )


def _run_staged_backend(
    *,
    vcf_path: Path,
    output_zarr: str | Path,
    chrom: str | int,
    genetic_map_path: str | Path | None,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
    store_genotypes: bool,
    attrs: dict | None,
    workers: int,
    contig: str,
    subset_ids: list[str],
    partition_plan: _PartitionPlan,
    shard_specs: list[_ShardSpec],
    scratch_dir: str | Path | None,
    cleanup_intermediates: bool,
    submit_materialize: Callable[[cf.ProcessPoolExecutor, _ShardSpec, Path], cf.Future[_ShardArtifactMeta]],
    progress_label: str,
) -> HapStore:
    total_start = time.perf_counter()
    map_df = _load_genetic_map_df(genetic_map_path, chrom=chrom)
    scratch_run_dir = _create_staged_scratch_dir(
        base_dir=scratch_dir,
        chrom=chrom,
        vcf_path=vcf_path,
    )
    try:
        artifacts_dir = scratch_run_dir / "shards"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        partition_t0 = time.perf_counter()
        manifest_path = scratch_run_dir / "partition_manifest.json"
        _write_partition_manifest(
            manifest_path=manifest_path,
            contig=contig,
            partition_plan=partition_plan,
        )
        partition_seconds = float(time.perf_counter() - partition_t0)
        logger.info(
            "%s partitioned %d shards for %s (partition_bp=%d)",
            progress_label,
            len(shard_specs),
            contig,
            int(partition_plan.partition_bp),
        )

        materialize_t0 = time.perf_counter()
        finalize_t0 = time.perf_counter()
        ctx = _create_streaming_finalize_context(
            output_zarr=output_zarr,
            chrom=chrom,
            subset_ids=subset_ids,
            shard_specs=shard_specs,
            map_df=map_df,
            store_genotypes=store_genotypes,
            chunk_variant_size=chunk_variant_size,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
        )
        try:
            ready_by_idx: dict[int, _ShardArtifactMeta] = {}
            next_shard_to_finalize = 0
            completed = 0
            n_shards = len(shard_specs)
            with cf.ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    submit_materialize(pool, spec, artifacts_dir): spec.shard_index
                    for spec in shard_specs
                }
                for fut in cf.as_completed(futures):
                    shard_meta = fut.result()
                    completed += 1
                    ready_by_idx[shard_meta.shard_index] = shard_meta
                    logger.info(
                        "%s %d/%d: idx=%d variants=%d elapsed=%.1fs",
                        progress_label,
                        completed,
                        n_shards,
                        shard_meta.shard_index,
                        shard_meta.n_variants,
                        time.perf_counter() - materialize_t0,
                    )
                    while next_shard_to_finalize in ready_by_idx:
                        _stream_finalize_write_shard(
                            ctx=ctx,
                            shard_index=next_shard_to_finalize,
                            meta=ready_by_idx.pop(next_shard_to_finalize),
                            chrom=chrom,
                            map_df=map_df,
                            started_at=finalize_t0,
                            missing_sentinel=_MISSING_SENTINEL,
                        )
                        next_shard_to_finalize += 1

            materialize_seconds = float(time.perf_counter() - materialize_t0)
            if completed != len(shard_specs):
                raise RuntimeError(
                    f"staged ingest materialization incomplete: got {completed} of {len(shard_specs)} shards"
                )
            if next_shard_to_finalize != len(shard_specs):
                raise RuntimeError(
                    f"staged ingest finalize incomplete: wrote {next_shard_to_finalize} "
                    f"of {len(shard_specs)} shards"
                )

            attrs_payload = dict(attrs or {})
            attrs_payload["partition_seconds"] = float(partition_seconds)
            attrs_payload["materialize_seconds"] = float(materialize_seconds)
            store = _complete_streaming_finalize(
                ctx=ctx,
                attrs=attrs_payload,
                workers=workers,
                partition_plan=partition_plan,
                missing_sentinel=_MISSING_SENTINEL,
            )
            finalize_seconds = float(time.perf_counter() - finalize_t0)
            total_seconds = float(time.perf_counter() - total_start)
            logger.info(
                "%s completed: partition=%.2fs materialize=%.2fs finalize=%.2fs total=%.2fs",
                progress_label,
                partition_seconds,
                materialize_seconds,
                finalize_seconds,
                total_seconds,
            )
            return store
        except Exception:
            _abort_streaming_finalize(ctx)
            raise
    finally:
        if cleanup_intermediates:
            shutil.rmtree(scratch_run_dir, ignore_errors=True)
        else:
            logger.info("Retained staged ingest intermediates: %s", scratch_run_dir)


def ingest_vcf(
    vcf_path: str | Path,
    output_zarr: str | Path,
    chrom: str | int,
    genetic_map_path: str | Path | None = None,
    sample_ids: list[str] | None = None,
    chunk_variant_size: int = 500,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    compression_level: int = 3,
    store_genotypes: bool = False,
    backend: Backend = "cyvcf2",
    attrs: dict | None = None,
    on_missing_ref_alt: MissingRefAltPolicy = "error",
    on_missing_vid: MissingVidPolicy = "dot",
    on_unphased_gt: UnphasedPolicy = "missing",
    workers: int = 1,
    partition_bp: int | None = None,
    scratch_dir: str | Path | None = None,
    cleanup_intermediates: bool = True,
) -> HapStore:
    """Ingest a chromosome from VCF into a hapstore zarr hierarchy.

    For `workers>1`, this uses a staged path:
    deterministic bp partition manifest -> parallel shard materialization -> ordered finalize.
    """
    vcf = Path(vcf_path)
    if not vcf.exists():
        raise FileNotFoundError(f"VCF not found: {vcf}")

    if on_missing_ref_alt not in {"error", "dot"}:
        raise ValueError(f"on_missing_ref_alt must be 'error' or 'dot', got {on_missing_ref_alt!r}")
    if on_missing_vid not in {"dot", "error"}:
        raise ValueError(f"on_missing_vid must be 'dot' or 'error', got {on_missing_vid!r}")
    if on_unphased_gt not in {"missing", "error"}:
        raise ValueError(f"on_unphased_gt must be 'missing' or 'error', got {on_unphased_gt!r}")
    if int(workers) < 1:
        raise ValueError(f"workers must be >=1, got {workers}")
    if partition_bp is not None and int(partition_bp) <= 0:
        raise ValueError(f"partition_bp must be >0 when provided, got {partition_bp}")
    if chunk_haplotype_size is not None and int(chunk_haplotype_size) <= 0:
        raise ValueError(
            f"chunk_haplotype_size must be >0 when provided, got {chunk_haplotype_size}"
        )
    if chunk_sample_size is not None and int(chunk_sample_size) <= 0:
        raise ValueError(f"chunk_sample_size must be >0 when provided, got {chunk_sample_size}")

    if int(workers) > 1:
        _warn_reader_worker_budget(int(workers), backend)

    if backend == "pysam" and int(workers) > 1:
        return _ingest_vcf_pysam_staged(
            vcf_path=vcf,
            output_zarr=output_zarr,
            chrom=chrom,
            genetic_map_path=genetic_map_path,
            sample_ids=sample_ids,
            chunk_variant_size=chunk_variant_size,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
            store_genotypes=store_genotypes,
            attrs=attrs,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
            workers=int(workers),
            partition_bp=None if partition_bp is None else int(partition_bp),
            scratch_dir=scratch_dir,
            cleanup_intermediates=bool(cleanup_intermediates),
        )
    if backend == "cyvcf2" and int(workers) > 1:
        return _ingest_vcf_cyvcf2_staged(
            vcf_path=vcf,
            output_zarr=output_zarr,
            chrom=chrom,
            genetic_map_path=genetic_map_path,
            sample_ids=sample_ids,
            chunk_variant_size=chunk_variant_size,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
            store_genotypes=store_genotypes,
            attrs=attrs,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
            workers=int(workers),
            partition_bp=None if partition_bp is None else int(partition_bp),
            scratch_dir=scratch_dir,
            cleanup_intermediates=bool(cleanup_intermediates),
        )

    ingested = _extract_vcf_arrays(
        vcf_path=vcf,
        chrom=chrom,
        backend=backend,
        sample_ids=sample_ids,
        on_missing_ref_alt=on_missing_ref_alt,
        on_missing_vid=on_missing_vid,
        on_unphased_gt=on_unphased_gt,
    )

    map_df = _load_genetic_map_df(genetic_map_path, chrom=chrom)
    genetic_positions = _interpolate_genetic_positions(ingested.positions, map_df=map_df, chrom=chrom)
    attrs_payload = _build_ingest_attrs_payload(
        attrs=attrs,
        ingested=ingested,
        parallel_workers=1,
        parallel_shards=1,
        parallel_mode="single_backend",
        partition_bp=int(chunk_variant_size),
        total_seconds=0.0,
    )
    attrs_payload.pop("parallel_workers", None)
    attrs_payload.pop("parallel_shards", None)
    attrs_payload.pop("parallel_mode", None)
    attrs_payload.pop("partition_bp", None)
    attrs_payload.pop("total_seconds", None)

    return _create_store_from_ingested(
        output_zarr=output_zarr,
        chrom=chrom,
        ingested=ingested,
        genetic_positions=genetic_positions,
        store_genotypes=store_genotypes,
        chunk_variant_size=chunk_variant_size,
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=compression_level,
        attrs=attrs_payload,
    )


def ingest_arrays(
    output_zarr: str | Path,
    chrom: str | int,
    haplotypes: np.ndarray,
    variant_pos: np.ndarray,
    sample_ids: list[str] | np.ndarray,
    ref: list[str] | np.ndarray | None = None,
    alt: list[str] | np.ndarray | None = None,
    vid: list[str] | np.ndarray | None = None,
    genetic_positions: np.ndarray | None = None,
    populations: list[str] | np.ndarray | None = None,
    store_genotypes: bool = False,
    chunk_variant_size: int = 500,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    compression_level: int = 3,
    attrs: dict | None = None,
) -> HapStore:
    """Ingest in-memory arrays into a hapstore."""
    store = HapStore(output_zarr, mode="a")
    store.create_chromosome(
        chrom=normalize_chrom(chrom, prefix=True),
        haplotypes=haplotypes,
        variant_pos=variant_pos,
        sample_ids=sample_ids,
        ref=ref,
        alt=alt,
        vid=vid,
        genetic_positions=genetic_positions,
        populations=populations,
        store_genotypes=store_genotypes,
        chunk_variant_size=chunk_variant_size,
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=compression_level,
        attrs=attrs,
    )
    return store


def _ingest_vcf_pysam_staged(
    vcf_path: Path,
    output_zarr: str | Path,
    chrom: str | int,
    genetic_map_path: str | Path | None,
    sample_ids: list[str] | None,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
    store_genotypes: bool,
    attrs: dict | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
    workers: int,
    partition_bp: int | None,
    scratch_dir: str | Path | None,
    cleanup_intermediates: bool,
) -> HapStore:
    import pysam  # type: ignore

    if not _has_tabix_index(vcf_path):
        raise ValueError(
            "parallel pysam ingest requires an indexed VCF/BCF (.tbi/.csi). "
            f"No index found for {vcf_path}"
        )

    finalize_variant_chunk = int(chunk_variant_size)
    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        contig = _select_contig_for_parallel(vcf, chrom)
        all_samples = list(vcf.header.samples)
        subset_idx, subset_ids = _resolve_subset(all_samples, sample_ids)
        resolved_partition_bp = _resolve_staged_partition_bp(
            vcf=vcf,
            contig=contig,
            workers=workers,
            partition_bp=partition_bp,
        )
        partition_plan = _build_staged_partition_plan(
            vcf=vcf,
            contig=contig,
            partition_bp=resolved_partition_bp,
        )
        shard_specs = partition_plan.shards

    if len(shard_specs) <= 1:
        return _single_shard_store_from_extract(
            output_zarr=output_zarr,
            chrom=chrom,
            genetic_map_path=genetic_map_path,
            store_genotypes=store_genotypes,
            chunk_variant_size=finalize_variant_chunk,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
            attrs=attrs,
            partition_plan=partition_plan,
            total_start=time.perf_counter(),
            extract_fn=lambda: _extract_vcf_arrays_pysam(
                vcf_path=vcf_path,
                chrom=chrom,
                sample_ids=sample_ids,
                on_missing_ref_alt=on_missing_ref_alt,
                on_missing_vid=on_missing_vid,
                on_unphased_gt=on_unphased_gt,
            ),
        )
    subset_idx_list = subset_idx.tolist()
    return _run_staged_backend(
        vcf_path=vcf_path,
        output_zarr=output_zarr,
        chrom=chrom,
        genetic_map_path=genetic_map_path,
        chunk_variant_size=finalize_variant_chunk,
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=compression_level,
        store_genotypes=store_genotypes,
        attrs=attrs,
        workers=workers,
        contig=contig,
        subset_ids=subset_ids,
        partition_plan=partition_plan,
        shard_specs=shard_specs,
        scratch_dir=scratch_dir,
        cleanup_intermediates=cleanup_intermediates,
        submit_materialize=lambda pool, spec, artifacts_dir: pool.submit(
            _materialize_pysam_shard_worker,
            str(vcf_path),
            contig,
            spec.start_bp,
            spec.end_bp,
            subset_idx_list,
            on_missing_ref_alt,
            on_missing_vid,
            on_unphased_gt,
            spec.shard_index,
            str(artifacts_dir),
        ),
        progress_label="Staged ingest",
    )


def _ingest_vcf_cyvcf2_staged(
    vcf_path: Path,
    output_zarr: str | Path,
    chrom: str | int,
    genetic_map_path: str | Path | None,
    sample_ids: list[str] | None,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
    store_genotypes: bool,
    attrs: dict | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
    workers: int,
    partition_bp: int | None,
    scratch_dir: str | Path | None,
    cleanup_intermediates: bool,
) -> HapStore:
    import pysam  # type: ignore

    if not _has_tabix_index(vcf_path):
        raise ValueError(
            "parallel cyvcf2 ingest requires an indexed VCF/BCF (.tbi/.csi). "
            f"No index found for {vcf_path}"
        )

    finalize_variant_chunk = int(chunk_variant_size)
    partition_t0 = time.perf_counter()
    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        contig = _select_contig_for_parallel(vcf, chrom)
        all_samples = list(vcf.header.samples)
        subset_idx, subset_ids = _resolve_subset(all_samples, sample_ids)
        subset_names = [all_samples[int(i)] for i in subset_idx.tolist()]
        resolved_partition_bp = _resolve_staged_partition_bp(
            vcf=vcf,
            contig=contig,
            workers=workers,
            partition_bp=partition_bp,
        )
        partition_plan = _build_staged_partition_plan(
            vcf=vcf,
            contig=contig,
            partition_bp=resolved_partition_bp,
        )
        shard_specs = partition_plan.shards

    if len(shard_specs) <= 1:
        return _single_shard_store_from_extract(
            output_zarr=output_zarr,
            chrom=chrom,
            genetic_map_path=genetic_map_path,
            store_genotypes=store_genotypes,
            chunk_variant_size=finalize_variant_chunk,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
            attrs=attrs,
            partition_plan=partition_plan,
            total_start=time.perf_counter(),
            partition_seconds=float(time.perf_counter() - partition_t0),
            extract_fn=lambda: _extract_vcf_arrays_cyvcf2(
                vcf_path=vcf_path,
                chrom=chrom,
                sample_ids=sample_ids,
                on_missing_ref_alt=on_missing_ref_alt,
                on_missing_vid=on_missing_vid,
                on_unphased_gt=on_unphased_gt,
            ),
        )
    return _run_staged_backend(
        vcf_path=vcf_path,
        output_zarr=output_zarr,
        chrom=chrom,
        genetic_map_path=genetic_map_path,
        chunk_variant_size=finalize_variant_chunk,
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=compression_level,
        store_genotypes=store_genotypes,
        attrs=attrs,
        workers=workers,
        contig=contig,
        subset_ids=subset_ids,
        partition_plan=partition_plan,
        shard_specs=shard_specs,
        scratch_dir=scratch_dir,
        cleanup_intermediates=cleanup_intermediates,
        submit_materialize=lambda pool, spec, artifacts_dir: pool.submit(
            _materialize_cyvcf2_shard_worker,
            str(vcf_path),
            contig,
            spec.start_bp,
            spec.end_bp,
            subset_names,
            on_missing_ref_alt,
            on_missing_vid,
            on_unphased_gt,
            spec.shard_index,
            str(artifacts_dir),
        ),
        progress_label="Staged ingest (cyvcf2)",
    )


def _build_staged_partition_plan(
    vcf: "pysam.VariantFile",
    contig: str,
    partition_bp: int,
) -> _PartitionPlan:
    if int(partition_bp) <= 0:
        raise ValueError(f"partition_bp must be >0, got {partition_bp}")

    ranges = _build_shard_ranges(vcf, contig, shard_bp=int(partition_bp))
    return _PartitionPlan(
        partition_bp=int(partition_bp),
        shards=[
            _ShardSpec(
                shard_index=int(shard_index),
                start_bp=None if start is None else int(start),
                end_bp=None if end is None else int(end),
            )
            for shard_index, start, end in ranges
        ],
    )


def _resolve_staged_partition_bp(
    vcf: "pysam.VariantFile",
    contig: str,
    workers: int,
    partition_bp: int | None,
) -> int:
    if partition_bp is not None:
        if int(partition_bp) <= 0:
            raise ValueError(f"partition_bp must be >0 when provided, got {partition_bp}")
        return int(partition_bp)

    span = _resolve_contig_span_bounds_for_parallel(vcf, contig)
    if span is None:
        logger.warning(
            "partition_bp not provided; could not determine contig span for %s. "
            "Falling back to %d bp.",
            contig,
            int(_DEFAULT_PARTITION_BP),
        )
        return int(_DEFAULT_PARTITION_BP)

    start0, end0 = span
    total_bp = int(max(1, int(end0) - int(start0)))
    n_workers = int(max(1, int(workers)))
    per_worker = int((total_bp + n_workers - 1) // n_workers)
    round_to = int(_AUTO_PARTITION_ROUND_BP)
    rounded = int(((per_worker + round_to - 1) // round_to) * round_to)
    resolved = int(max(1, rounded))
    logger.info(
        "Auto-selected partition_bp=%d for %s (span_bp=%d workers=%d round_to=%d)",
        resolved,
        contig,
        total_bp,
        n_workers,
        round_to,
    )
    return resolved


def _create_staged_scratch_dir(
    base_dir: str | Path | None,
    chrom: str | int,
    vcf_path: Path,
) -> Path:
    chrom_key = normalize_chrom(chrom, prefix=True)
    unique = f"{chrom_key}_{vcf_path.stem}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    if base_dir is None:
        user = os.getenv("USER", "unknown")
        if Path("/scratch/tmp").exists():
            root = Path("/scratch/tmp") / user / "hapstore_ingest"
        else:
            root = Path(tempfile.gettempdir()) / user / "hapstore_ingest"
    else:
        root = Path(base_dir)

    run_dir = root / f"staged_{unique}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_partition_manifest(
    manifest_path: Path,
    contig: str,
    partition_plan: _PartitionPlan,
) -> None:
    payload = {
        "contig": contig,
        "partition_bp": int(partition_plan.partition_bp),
        "n_shards": int(len(partition_plan.shards)),
        "shards": [
            {
                "shard_id": int(shard.shard_index),
                "start_bp": None if shard.start_bp is None else int(shard.start_bp),
                "end_bp": None if shard.end_bp is None else int(shard.end_bp),
            }
            for shard in partition_plan.shards
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _materialize_pysam_shard_worker(
    vcf_path: str,
    contig: str,
    start: int | None,
    end: int | None,
    subset_idx: list[int],
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
    shard_index: int,
    artifacts_dir: str,
) -> _ShardArtifactMeta:
    shard = _read_pysam_shard_worker(
        vcf_path=vcf_path,
        contig=contig,
        start=start,
        end=end,
        subset_idx=subset_idx,
        on_missing_ref_alt=on_missing_ref_alt,
        on_missing_vid=on_missing_vid,
        on_unphased_gt=on_unphased_gt,
        shard_index=shard_index,
    )
    artifact_path = Path(artifacts_dir) / f"shard_{int(shard_index):06d}.npz"
    np.savez(
        str(artifact_path),
        haplotypes=np.asarray(shard.haplotypes, dtype=np.uint8),
        positions=np.asarray(shard.positions, dtype=np.int64),
        ref=np.asarray(shard.ref, dtype=str),
        alt=np.asarray(shard.alt, dtype=str),
        vid=np.asarray(shard.vid, dtype=str),
    )
    return _ShardArtifactMeta(
        shard_index=int(shard_index),
        start=None if start is None else int(start),
        end=None if end is None else int(end),
        artifact_path=str(artifact_path),
        n_variants=int(shard.positions.shape[0]),
        missing_ref_count=int(shard.missing_ref_count),
        missing_alt_count=int(shard.missing_alt_count),
        missing_vid_count=int(shard.missing_vid_count),
        unphased_gt_count=int(shard.unphased_gt_count),
    )


def _materialize_cyvcf2_shard_worker(
    vcf_path: str,
    contig: str,
    start: int | None,
    end: int | None,
    subset_samples: list[str],
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
    shard_index: int,
    artifacts_dir: str,
) -> _ShardArtifactMeta:
    shard = _read_cyvcf2_shard_worker(
        vcf_path=vcf_path,
        contig=contig,
        start=start,
        end=end,
        subset_samples=subset_samples,
        on_missing_ref_alt=on_missing_ref_alt,
        on_missing_vid=on_missing_vid,
        on_unphased_gt=on_unphased_gt,
        shard_index=shard_index,
    )
    artifact_path = Path(artifacts_dir) / f"shard_{int(shard_index):06d}.npz"
    np.savez(
        str(artifact_path),
        haplotypes=np.asarray(shard.haplotypes, dtype=np.uint8),
        positions=np.asarray(shard.positions, dtype=np.int64),
        ref=np.asarray(shard.ref, dtype=str),
        alt=np.asarray(shard.alt, dtype=str),
        vid=np.asarray(shard.vid, dtype=str),
    )
    return _ShardArtifactMeta(
        shard_index=int(shard_index),
        start=None if start is None else int(start),
        end=None if end is None else int(end),
        artifact_path=str(artifact_path),
        n_variants=int(shard.positions.shape[0]),
        missing_ref_count=int(shard.missing_ref_count),
        missing_alt_count=int(shard.missing_alt_count),
        missing_vid_count=int(shard.missing_vid_count),
        unphased_gt_count=int(shard.unphased_gt_count),
    )


def _read_cyvcf2_shard_worker(
    vcf_path: str,
    contig: str,
    start: int | None,
    end: int | None,
    subset_samples: list[str],
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
    shard_index: int,
) -> _ShardResult:
    from cyvcf2 import VCF  # type: ignore

    vcf = VCF(vcf_path)
    vcf.set_samples(subset_samples)

    if start is None or end is None:
        iterator = vcf(contig)
    else:
        start1 = int(start) + 1
        end1 = int(end)
        if end1 < start1:
            end1 = start1
        iterator = vcf(f"{contig}:{start1}-{end1}")

    n_samples = len(subset_samples)
    n_haplotypes = 2 * n_samples
    capacity = 1024
    haplotypes_buf = np.empty((n_haplotypes, capacity), dtype=np.uint8)
    n_variants = 0
    positions: list[int] = []
    ref: list[str] = []
    alt: list[str] = []
    vid: list[str] = []
    missing_ref_count = 0
    missing_alt_count = 0
    missing_vid_count = 0
    unphased_gt_count = 0

    for variant in iterator:
        if variant.ALT is None or len(variant.ALT) != 1:
            continue

        pos = int(variant.POS)
        chrom_name = str(variant.CHROM)
        ref_i, alt_i, mref, malt = _coerce_ref_alt(
            chrom=chrom_name,
            pos=pos,
            ref=variant.REF,
            alt=variant.ALT[0],
            on_missing_ref_alt=on_missing_ref_alt,
        )
        vid_i, mvid = _coerce_vid(chrom_name, pos, variant.ID, on_missing_vid)

        if n_variants >= capacity:
            new_capacity = int(capacity * 2)
            expanded = np.empty((n_haplotypes, new_capacity), dtype=np.uint8)
            expanded[:, :capacity] = haplotypes_buf
            haplotypes_buf = expanded
            capacity = new_capacity

        col = haplotypes_buf[:, n_variants]
        col.fill(_MISSING_SENTINEL)

        gt = np.asarray(variant.genotypes, dtype=np.int32)
        if gt.ndim != 2 or gt.shape[0] != n_samples or gt.shape[1] < 2:
            positions.append(pos)
            ref.append(ref_i)
            alt.append(alt_i)
            vid.append(vid_i)
            missing_ref_count += mref
            missing_alt_count += malt
            missing_vid_count += mvid
            n_variants += 1
            continue

        a0 = gt[:, 0]
        a1 = gt[:, 1]
        phased = np.ones(n_samples, dtype=bool) if gt.shape[1] < 3 else (gt[:, 2] != 0)
        not_phased = ~phased
        if bool(np.any(not_phased)):
            unphased_gt_count += int(np.sum(not_phased))
            if on_unphased_gt == "error":
                bad = int(np.flatnonzero(not_phased)[0])
                raise ValueError(f"unphased genotype at {chrom_name}:{pos} sample={subset_samples[bad]}")

        missing = (a0 < 0) | (a1 < 0) | not_phased
        col[0::2] = np.where(missing, _MISSING_SENTINEL, (a0 > 0).astype(np.uint8))
        col[1::2] = np.where(missing, _MISSING_SENTINEL, (a1 > 0).astype(np.uint8))
        positions.append(pos)
        ref.append(ref_i)
        alt.append(alt_i)
        vid.append(vid_i)
        missing_ref_count += mref
        missing_alt_count += malt
        missing_vid_count += mvid
        n_variants += 1

    if n_variants == 0:
        haplotypes = np.zeros((2 * n_samples, 0), dtype=np.uint8)
    else:
        haplotypes = haplotypes_buf[:, :n_variants]

    return _ShardResult(
        shard_index=int(shard_index),
        haplotypes=haplotypes,
        positions=np.asarray(positions, dtype=np.int64),
        ref=np.asarray(ref, dtype=object),
        alt=np.asarray(alt, dtype=object),
        vid=np.asarray(vid, dtype=object),
        missing_ref_count=int(missing_ref_count),
        missing_alt_count=int(missing_alt_count),
        missing_vid_count=int(missing_vid_count),
        unphased_gt_count=int(unphased_gt_count),
    )


def _finalize_staged_pysam_ingest(
    output_zarr: str | Path,
    chrom: str | int,
    subset_ids: np.ndarray,
    shard_specs: list[_ShardSpec],
    shard_results: dict[int, _ShardArtifactMeta],
    map_df,
    store_genotypes: bool,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
    attrs: dict,
    workers: int,
    partition_plan: _PartitionPlan,
) -> HapStore:
    ctx = _create_streaming_finalize_context(
        output_zarr=output_zarr,
        chrom=chrom,
        subset_ids=subset_ids,
        shard_specs=shard_specs,
        map_df=map_df,
        store_genotypes=store_genotypes,
        chunk_variant_size=chunk_variant_size,
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=compression_level,
    )
    finalize_t0 = time.perf_counter()
    try:
        for shard_index in range(len(shard_specs)):
            _stream_finalize_write_shard(
                ctx=ctx,
                shard_index=shard_index,
                meta=shard_results[shard_index],
                chrom=chrom,
                map_df=map_df,
                started_at=finalize_t0,
                missing_sentinel=_MISSING_SENTINEL,
            )
        return _complete_streaming_finalize(
            ctx=ctx,
            attrs=attrs,
            workers=workers,
            partition_plan=partition_plan,
            missing_sentinel=_MISSING_SENTINEL,
        )
    except Exception:
        _abort_streaming_finalize(ctx)
        raise


def _read_pysam_shard_worker(
    vcf_path: str,
    contig: str,
    start: int | None,
    end: int | None,
    subset_idx: list[int],
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
    shard_index: int,
) -> _ShardResult:
    import pysam  # type: ignore

    vcf = pysam.VariantFile(str(vcf_path), "r")
    all_samples = list(vcf.header.samples)

    if start is None or end is None:
        iterator = vcf.fetch(contig)
    else:
        iterator = vcf.fetch(contig, int(start), int(end))

    columns: list[np.ndarray] = []
    positions: list[int] = []
    ref: list[str] = []
    alt: list[str] = []
    vid: list[str] = []
    missing_ref_count = 0
    missing_alt_count = 0
    missing_vid_count = 0
    unphased_gt_count = 0

    for record in iterator:
        if record.alts is None or len(record.alts) != 1:
            continue

        pos = int(record.pos)
        chrom_name = str(record.contig)
        ref_i, alt_i, mref, malt = _coerce_ref_alt(
            chrom=chrom_name,
            pos=pos,
            ref=record.ref,
            alt=record.alts[0],
            on_missing_ref_alt=on_missing_ref_alt,
        )
        vid_i, mvid = _coerce_vid(chrom_name, pos, record.id, on_missing_vid)

        hap = np.empty(2 * len(subset_idx), dtype=np.uint8)
        for out_i, sample_i in enumerate(subset_idx):
            sample_name = all_samples[int(sample_i)]
            call = record.samples[sample_name]
            gt = call.get("GT")
            phased = bool(call.phased)

            if gt is None or len(gt) < 2:
                hap[2 * out_i] = _MISSING_SENTINEL
                hap[2 * out_i + 1] = _MISSING_SENTINEL
                continue

            if not phased:
                unphased_gt_count += 1
                if on_unphased_gt == "error":
                    raise ValueError(f"unphased genotype at {chrom_name}:{pos} sample={sample_name}")
                hap[2 * out_i] = _MISSING_SENTINEL
                hap[2 * out_i + 1] = _MISSING_SENTINEL
                continue

            a0, a1 = gt[0], gt[1]
            if a0 is None or a1 is None or int(a0) < 0 or int(a1) < 0:
                hap[2 * out_i] = _MISSING_SENTINEL
                hap[2 * out_i + 1] = _MISSING_SENTINEL
                continue

            hap[2 * out_i] = 1 if int(a0) > 0 else 0
            hap[2 * out_i + 1] = 1 if int(a1) > 0 else 0

        columns.append(hap)
        positions.append(pos)
        ref.append(ref_i)
        alt.append(alt_i)
        vid.append(vid_i)
        missing_ref_count += mref
        missing_alt_count += malt
        missing_vid_count += mvid

    n_samples = len(subset_idx)
    if not columns:
        haplotypes = np.zeros((2 * n_samples, 0), dtype=np.uint8)
    else:
        haplotypes = np.column_stack(columns).astype(np.uint8, copy=False)

    return _ShardResult(
        shard_index=int(shard_index),
        haplotypes=haplotypes,
        positions=np.asarray(positions, dtype=np.int64),
        ref=np.asarray(ref, dtype=object),
        alt=np.asarray(alt, dtype=object),
        vid=np.asarray(vid, dtype=object),
        missing_ref_count=int(missing_ref_count),
        missing_alt_count=int(missing_alt_count),
        missing_vid_count=int(missing_vid_count),
        unphased_gt_count=int(unphased_gt_count),
    )
