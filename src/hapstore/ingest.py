"""Converters into hapstore."""

from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import logging
import os
import shutil
import struct
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
from numcodecs import VLenUTF8, Zstd

from hapstore._internal.chrom import candidate_contigs, normalize_chrom

from .store import HapStore

logger = logging.getLogger(__name__)

Backend = Literal["auto", "cyvcf2", "pysam", "text"]
MissingRefAltPolicy = Literal["error", "dot"]
MissingVidPolicy = Literal["dot", "error"]
UnphasedPolicy = Literal["missing", "error"]
_MISSING_SENTINEL = 255
_DEFAULT_PARTITION_BP = 1_000_000
_AUTO_PARTITION_ROUND_BP = 100_000


@dataclass
class _IngestArrays:
    haplotypes: np.ndarray
    positions: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    vid: np.ndarray
    sample_ids: np.ndarray
    missing_ref_count: int
    missing_alt_count: int
    missing_vid_count: int
    unphased_gt_count: int


@dataclass
class _ShardResult:
    shard_index: int
    haplotypes: np.ndarray
    positions: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    vid: np.ndarray
    missing_ref_count: int
    missing_alt_count: int
    missing_vid_count: int
    unphased_gt_count: int


@dataclass
class _ShardArtifactMeta:
    shard_index: int
    start: int | None
    end: int | None
    artifact_path: str
    n_variants: int
    missing_ref_count: int
    missing_alt_count: int
    missing_vid_count: int
    unphased_gt_count: int


@dataclass
class _ShardSpec:
    shard_index: int
    start_bp: int | None
    end_bp: int | None


@dataclass
class _PartitionPlan:
    partition_bp: int
    shards: list[_ShardSpec]


@dataclass
class _StreamingFinalizeContext:
    store: HapStore
    chrom_key: str
    chrom_group: Any
    hap_arr: Any
    geno_arr: Any | None
    pos_arr: Any
    ref_arr: Any
    alt_arr: Any
    vid_arr: Any
    genetic_arr: Any | None
    n_samples: int
    n_haplotypes: int
    n_shards: int
    variant_chunk: int
    hap_chunk: int
    sample_chunk_geno: int
    compression_level: int
    chunk_haplotype_size: int | None
    chunk_sample_size: int | None
    capacity: int
    offset: int = 0
    has_missing: bool = False
    missing_ref_count: int = 0
    missing_alt_count: int = 0
    missing_vid_count: int = 0
    unphased_gt_count: int = 0
    prev_last_position: int | None = None
    finished: bool = False


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

    genetic_positions = None
    if genetic_map_path is not None:
        from hapstore._internal.genetic_map import interpolate_cm, load_genetic_map

        map_path = Path(genetic_map_path)
        if not map_path.exists():
            raise FileNotFoundError(f"genetic map not found: {map_path}")
        map_df = load_genetic_map(map_path, chrom=chrom)
        genetic_positions = interpolate_cm(ingested.positions, map_df, chrom=chrom)

    attrs_payload = dict(attrs or {})
    attrs_payload["missing_ref_count"] = int(ingested.missing_ref_count)
    attrs_payload["missing_alt_count"] = int(ingested.missing_alt_count)
    attrs_payload["missing_vid_count"] = int(ingested.missing_vid_count)
    attrs_payload["unphased_gt_count"] = int(ingested.unphased_gt_count)

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

    total_start = time.perf_counter()
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
        ingested = _extract_vcf_arrays_pysam(
            vcf_path=vcf_path,
            chrom=chrom,
            sample_ids=sample_ids,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
        )
        genetic_positions = None
        if genetic_map_path is not None:
            from hapstore._internal.genetic_map import interpolate_cm, load_genetic_map

            map_path = Path(genetic_map_path)
            if not map_path.exists():
                raise FileNotFoundError(f"genetic map not found: {map_path}")
            map_df = load_genetic_map(map_path, chrom=chrom)
            genetic_positions = interpolate_cm(ingested.positions, map_df, chrom=chrom)

        attrs_payload = dict(attrs or {})
        attrs_payload["missing_ref_count"] = int(ingested.missing_ref_count)
        attrs_payload["missing_alt_count"] = int(ingested.missing_alt_count)
        attrs_payload["missing_vid_count"] = int(ingested.missing_vid_count)
        attrs_payload["unphased_gt_count"] = int(ingested.unphased_gt_count)
        attrs_payload["parallel_workers"] = 1
        attrs_payload["parallel_shards"] = 1
        attrs_payload["parallel_mode"] = "single_shard"
        attrs_payload["partition_bp"] = int(partition_plan.partition_bp)
        attrs_payload["total_seconds"] = float(time.perf_counter() - total_start)

        return _create_store_from_ingested(
            output_zarr=output_zarr,
            chrom=chrom,
            ingested=ingested,
            genetic_positions=genetic_positions,
            store_genotypes=store_genotypes,
            chunk_variant_size=finalize_variant_chunk,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
            attrs=attrs_payload,
        )

    map_df = None
    if genetic_map_path is not None:
        from hapstore._internal.genetic_map import load_genetic_map

        map_path = Path(genetic_map_path)
        if not map_path.exists():
            raise FileNotFoundError(f"genetic map not found: {map_path}")
        map_df = load_genetic_map(map_path, chrom=chrom)

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
            "Staged ingest partitioned %d shards for %s (partition_bp=%d)",
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
            chunk_variant_size=finalize_variant_chunk,
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
                futures: dict[cf.Future[_ShardArtifactMeta], int] = {}
                subset_idx_list = subset_idx.tolist()
                for spec in shard_specs:
                    fut = pool.submit(
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
                    )
                    futures[fut] = spec.shard_index

                for fut in cf.as_completed(futures):
                    shard_meta = fut.result()
                    completed += 1
                    ready_by_idx[shard_meta.shard_index] = shard_meta
                    logger.info(
                        "Shard materialized %d/%d: idx=%d variants=%d elapsed=%.1fs",
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
            )
            finalize_seconds = float(time.perf_counter() - finalize_t0)
            total_seconds = float(time.perf_counter() - total_start)
            logger.info(
                "Staged ingest completed: partition=%.2fs materialize=%.2fs finalize=%.2fs total=%.2fs",
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

    total_start = time.perf_counter()
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
        partition_seconds = float(time.perf_counter() - partition_t0)
        ingested = _extract_vcf_arrays_cyvcf2(
            vcf_path=vcf_path,
            chrom=chrom,
            sample_ids=sample_ids,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
        )
        genetic_positions = None
        if genetic_map_path is not None:
            from hapstore._internal.genetic_map import interpolate_cm, load_genetic_map

            map_path = Path(genetic_map_path)
            if not map_path.exists():
                raise FileNotFoundError(f"genetic map not found: {map_path}")
            map_df = load_genetic_map(map_path, chrom=chrom)
            genetic_positions = interpolate_cm(ingested.positions, map_df, chrom=chrom)

        attrs_payload = dict(attrs or {})
        attrs_payload["missing_ref_count"] = int(ingested.missing_ref_count)
        attrs_payload["missing_alt_count"] = int(ingested.missing_alt_count)
        attrs_payload["missing_vid_count"] = int(ingested.missing_vid_count)
        attrs_payload["unphased_gt_count"] = int(ingested.unphased_gt_count)
        attrs_payload["parallel_workers"] = 1
        attrs_payload["parallel_shards"] = 1
        attrs_payload["parallel_mode"] = "single_shard"
        attrs_payload["partition_seconds"] = float(partition_seconds)
        attrs_payload["partition_bp"] = int(partition_plan.partition_bp)
        attrs_payload["total_seconds"] = float(time.perf_counter() - total_start)

        return _create_store_from_ingested(
            output_zarr=output_zarr,
            chrom=chrom,
            ingested=ingested,
            genetic_positions=genetic_positions,
            store_genotypes=store_genotypes,
            chunk_variant_size=finalize_variant_chunk,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
            attrs=attrs_payload,
        )

    map_df = None
    if genetic_map_path is not None:
        from hapstore._internal.genetic_map import load_genetic_map

        map_path = Path(genetic_map_path)
        if not map_path.exists():
            raise FileNotFoundError(f"genetic map not found: {map_path}")
        map_df = load_genetic_map(map_path, chrom=chrom)

    scratch_run_dir = _create_staged_scratch_dir(
        base_dir=scratch_dir,
        chrom=chrom,
        vcf_path=vcf_path,
    )
    try:
        artifacts_dir = scratch_run_dir / "shards"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = scratch_run_dir / "partition_manifest.json"
        _write_partition_manifest(
            manifest_path=manifest_path,
            contig=contig,
            partition_plan=partition_plan,
        )
        partition_seconds = float(time.perf_counter() - partition_t0)
        logger.info(
            "Staged ingest (cyvcf2) partitioned %d shards for %s (partition_bp=%d)",
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
            chunk_variant_size=finalize_variant_chunk,
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
                futures: dict[cf.Future[_ShardArtifactMeta], int] = {}
                for spec in shard_specs:
                    fut = pool.submit(
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
                    )
                    futures[fut] = spec.shard_index

                for fut in cf.as_completed(futures):
                    shard_meta = fut.result()
                    completed += 1
                    ready_by_idx[shard_meta.shard_index] = shard_meta
                    logger.info(
                        "Shard materialized (cyvcf2) %d/%d: idx=%d variants=%d elapsed=%.1fs",
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
            )
            finalize_seconds = float(time.perf_counter() - finalize_t0)
            total_seconds = float(time.perf_counter() - total_start)
            logger.info(
                "Staged ingest (cyvcf2) completed: partition=%.2fs materialize=%.2fs finalize=%.2fs total=%.2fs",
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
    sample_ids_u = _to_unicode_array(np.asarray(subset_ids, dtype=object))
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

    variant_chunk = int(max(1, int(chunk_variant_size)))
    hap_chunk = int(n_haplotypes if chunk_haplotype_size is None else chunk_haplotype_size)
    hap_chunk = int(min(max(1, hap_chunk), n_haplotypes))
    sample_chunk_geno = int(n_samples if chunk_sample_size is None else chunk_sample_size)
    sample_chunk_geno = int(min(max(1, sample_chunk_geno), n_samples))
    compressor = Zstd(level=int(compression_level))
    initial_capacity = _estimate_initial_variant_capacity(
        shard_specs=shard_specs,
        variant_chunk=variant_chunk,
    )

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
    pos_arr = variants.create_dataset(
        "pos",
        shape=(initial_capacity,),
        chunks=(variant_chunk,),
        dtype=np.int64,
        compressor=compressor,
        overwrite=False,
    )
    ref_arr = variants.create_dataset(
        "ref",
        shape=(initial_capacity,),
        chunks=(variant_chunk,),
        dtype=object,
        object_codec=VLenUTF8(),
        compressor=compressor,
        overwrite=False,
    )
    alt_arr = variants.create_dataset(
        "alt",
        shape=(initial_capacity,),
        chunks=(variant_chunk,),
        dtype=object,
        object_codec=VLenUTF8(),
        compressor=compressor,
        overwrite=False,
    )
    vid_arr = variants.create_dataset(
        "vid",
        shape=(initial_capacity,),
        chunks=(variant_chunk,),
        dtype=object,
        object_codec=VLenUTF8(),
        compressor=compressor,
        overwrite=False,
    )

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
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
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
) -> None:
    from hapstore._internal.genetic_map import interpolate_cm

    with np.load(meta.artifact_path, allow_pickle=False) as shard_data:
        h = np.asarray(shard_data["haplotypes"], dtype=np.uint8)
        positions = np.asarray(shard_data["positions"], dtype=np.int64)
        ref = np.asarray(shard_data["ref"], dtype=object)
        alt = np.asarray(shard_data["alt"], dtype=object)
        vid = np.asarray(shard_data["vid"], dtype=object)

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
    if width > 1:
        diffs = np.diff(positions)
        if np.any(diffs < 0):
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
            ctx.geno_arr[:, sl] = _haplotypes_to_genotypes_local(h)
        ctx.pos_arr[sl] = positions
        ctx.ref_arr[sl] = ref
        ctx.alt_arr[sl] = alt
        ctx.vid_arr[sl] = vid
        if ctx.genetic_arr is not None:
            cm = interpolate_cm(positions, map_df, chrom=chrom)
            ctx.genetic_arr[sl] = np.asarray(cm, dtype=np.float64)
        ctx.prev_last_position = int(positions[-1])
        if not ctx.has_missing and bool(np.any(h == _MISSING_SENTINEL)):
            ctx.has_missing = True

    ctx.offset = int(next_offset)
    ctx.missing_ref_count += int(meta.missing_ref_count)
    ctx.missing_alt_count += int(meta.missing_alt_count)
    ctx.missing_vid_count += int(meta.missing_vid_count)
    ctx.unphased_gt_count += int(meta.unphased_gt_count)
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
    payload["parallel_mode"] = "staged_materialize_finalize"
    payload["partition_bp"] = int(partition_plan.partition_bp)
    payload["chunk_variant_size"] = int(ctx.variant_chunk)
    payload["chunk_haplotype_size"] = int(ctx.hap_chunk)
    payload["chunk_sample_size"] = int(ctx.sample_chunk_geno)

    source = payload.pop("source", None)
    build = payload.pop("build", None)
    layout_version = 1 if ctx.hap_chunk == ctx.n_haplotypes and ctx.sample_chunk_geno == ctx.n_samples else 2
    ctx.chrom_group.attrs.update(
        {
            "n_samples": int(ctx.n_samples),
            "n_haplotypes": int(ctx.n_haplotypes),
            "n_variants": int(final_variants),
            "phased": True,
            "ploidy": 2,
            "missing_sentinel": _MISSING_SENTINEL,
            "has_missing": bool(ctx.has_missing),
            "compression": "zstd",
            "compression_level": int(ctx.compression_level),
            "chunk_variant_size": int(ctx.variant_chunk),
            "chunk_haplotype_size": None
            if ctx.chunk_haplotype_size is None
            else int(ctx.chunk_haplotype_size),
            "chunk_sample_size": None if ctx.chunk_sample_size is None else int(ctx.chunk_sample_size),
            "layout_version": int(layout_version),
            "source": source,
            "build": build,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extra": payload or None,
        }
    )
    ctx.finished = True
    return ctx.store


def _abort_streaming_finalize(ctx: _StreamingFinalizeContext | None) -> None:
    if ctx is None or ctx.finished:
        return
    if ctx.chrom_key in ctx.store.root:
        del ctx.store.root[ctx.chrom_key]


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
            )
        return _complete_streaming_finalize(
            ctx=ctx,
            attrs=attrs,
            workers=workers,
            partition_plan=partition_plan,
        )
    except Exception:
        _abort_streaming_finalize(ctx)
        raise


def _has_tabix_index(vcf_path: Path) -> bool:
    p = str(vcf_path)
    return Path(f"{p}.tbi").exists() or Path(f"{p}.csi").exists()


def _select_contig_for_parallel(vcf: "pysam.VariantFile", chrom: str | int) -> str:
    import pysam  # type: ignore

    fallback = None
    for contig in candidate_contigs(chrom):
        if contig in vcf.header.contigs and fallback is None:
            fallback = contig
        try:
            it = vcf.fetch(contig)
        except (ValueError, OSError):
            continue

        try:
            next(it)
            return contig
        except StopIteration:
            if fallback is None and contig in vcf.header.contigs:
                fallback = contig
            continue

    if fallback is not None:
        return fallback

    raise ValueError(
        f"chromosome {normalize_chrom(chrom, prefix=True)!r} not found in VCF header/records"
    )


def _build_shard_ranges(
    vcf: "pysam.VariantFile",
    contig: str,
    shard_bp: int,
) -> list[tuple[int, int | None, int | None]]:
    span = _resolve_contig_span_bounds_for_parallel(vcf, contig)
    if span is None:
        return [(0, None, None)]
    start0, end0 = span
    return _ranges_from_span(start0=start0, end0=end0, shard_bp=shard_bp)


def _resolve_contig_span_bounds_for_parallel(
    vcf: "pysam.VariantFile",
    contig: str,
) -> tuple[int, int] | None:
    contig_meta = vcf.header.contigs.get(contig)
    contig_length = None if contig_meta is None else contig_meta.length
    if contig_length is not None and int(contig_length) > 0:
        return (0, int(contig_length))

    vcf_filename = getattr(vcf, "filename", None)
    if vcf_filename is not None:
        vcf_path = Path(vcf_filename.decode("utf-8")) if isinstance(vcf_filename, bytes) else Path(vcf_filename)
        span_end = _resolve_contig_span_end_from_tabix_index(vcf_path, contig)
        if span_end is not None and int(span_end) > 0:
            return (0, int(span_end))

    return _infer_contig_bounds_from_records(vcf, contig)
def _resolve_contig_span_end_from_tabix_index(vcf_path: Path, contig: str) -> int | None:
    tbi = Path(f"{vcf_path}.tbi")
    if tbi.exists():
        end = _contig_span_end_from_tbi(tbi, contig)
        if end is not None:
            return int(end)
    csi = Path(f"{vcf_path}.csi")
    if csi.exists():
        end = _contig_span_end_from_csi(csi, contig)
        if end is not None:
            return int(end)
    return None


def _contig_span_end_from_tbi(tbi_path: Path, contig: str) -> int | None:
    contig_b = contig.encode("utf-8")
    try:
        with _open_tabix_index(tbi_path) as f:
            magic = f.read(4)
            if magic != b"TBI\x01":
                return None
            n_ref = _read_int32(f)
            _ = f.read(24)  # format + columns + meta + skip
            l_nm = _read_int32(f)
            names = f.read(int(l_nm))
            ref_names = names.rstrip(b"\0").split(b"\0") if names else []
            try:
                rid = ref_names.index(contig_b)
            except ValueError:
                return None

            min_shift = 14
            for ref_idx in range(int(n_ref)):
                n_bin = _read_int32(f)
                for _ in range(int(n_bin)):
                    f.read(4)  # bin
                    n_chunk = _read_int32(f)
                    f.seek(16 * int(n_chunk), os.SEEK_CUR)
                n_intv = _read_int32(f)
                if ref_idx == rid:
                    return int(n_intv) << int(min_shift)
                f.seek(8 * int(n_intv), os.SEEK_CUR)
    except Exception:  # pragma: no cover
        return None
    return None


def _contig_span_end_from_csi(csi_path: Path, contig: str) -> int | None:
    contig_b = contig.encode("utf-8")
    try:
        with _open_tabix_index(csi_path) as f:
            magic = f.read(4)
            if magic != b"CSI\x01":
                return None
            n_ref = _read_int32(f)
            min_shift = _read_int32(f)
            _depth = _read_int32(f)
            l_aux = _read_int32(f)
            aux = f.read(int(l_aux))
            rid = _resolve_ref_id_from_csi_aux(aux, contig_b)
            if rid is None:
                return None

            for ref_idx in range(int(n_ref)):
                n_bin = _read_int32(f)
                for _ in range(int(n_bin)):
                    f.read(4)  # bin
                    n_chunk = _read_int32(f)
                    f.seek(16 * int(n_chunk), os.SEEK_CUR)
                n_intv = _read_int32(f)
                if ref_idx == int(rid):
                    return int(n_intv) << int(min_shift)
                f.seek(8 * int(n_intv), os.SEEK_CUR)
    except Exception:  # pragma: no cover
        return None
    return None


def _resolve_ref_id_from_csi_aux(aux: bytes, contig: bytes) -> int | None:
    if len(aux) < 28:
        return None
    # tabix CSI aux stores: format+col_seq+col_beg+col_end+meta+skip (6x int32) then l_nm (int32) then names.
    l_nm = struct.unpack_from("<i", aux, 24)[0]
    if l_nm <= 0:
        return None
    end = 28 + int(l_nm)
    if end > len(aux):
        return None
    names = aux[28:end]
    ref_names = names.rstrip(b"\0").split(b"\0") if names else []
    try:
        return int(ref_names.index(contig))
    except ValueError:
        return None


def _open_tabix_index(index_path: Path):
    raw = index_path.open("rb")
    head = raw.read(2)
    raw.close()
    if head == b"\x1f\x8b":
        return gzip.open(index_path, "rb")
    return index_path.open("rb")


def _read_int32(f) -> int:
    raw = f.read(4)
    if len(raw) != 4:
        raise EOFError("unexpected EOF while reading int32")
    return int(struct.unpack("<i", raw)[0])


def _ranges_from_span(
    start0: int,
    end0: int,
    shard_bp: int,
) -> list[tuple[int, int | None, int | None]]:
    if end0 <= start0:
        return [(0, int(start0), int(start0 + 1))]

    ranges: list[tuple[int, int | None, int | None]] = []
    start = int(max(0, start0))
    shard_index = 0
    while start < end0:
        end = min(int(end0), start + int(shard_bp))
        ranges.append((shard_index, int(start), int(end)))
        start = end
        shard_index += 1
    return ranges


def _infer_contig_bounds_from_records(
    vcf: "pysam.VariantFile",
    contig: str,
) -> tuple[int, int] | None:
    try:
        iterator = vcf.fetch(contig)
    except Exception:
        return None

    min_start: int | None = None
    max_end: int | None = None
    for record in iterator:
        rec_start = int(record.start)
        rec_end = int(record.stop)
        if min_start is None or rec_start < min_start:
            min_start = rec_start
        if max_end is None or rec_end > max_end:
            max_end = rec_end

    if min_start is None or max_end is None:
        return None
    if max_end <= min_start:
        max_end = min_start + 1
    return int(min_start), int(max_end)


def _validate_cross_chrom_samples(
    store: HapStore,
    sample_ids: np.ndarray,
    target_chrom: str,
) -> None:
    for existing in store.chromosomes:
        if existing == target_chrom:
            continue
        existing_ids = np.asarray(store.root[existing]["samples"]["ids"][:], dtype=object)
        existing_ids = _to_unicode_array(existing_ids)
        if existing_ids.shape[0] != sample_ids.shape[0]:
            raise ValueError(
                "sample count mismatch with existing chromosome "
                f"{existing!r}: expected {existing_ids.shape[0]}, got {sample_ids.shape[0]}"
            )
        if not np.array_equal(existing_ids, sample_ids):
            raise ValueError(
                "sample ID ordering mismatch with existing chromosome "
                f"{existing!r}; all chromosomes in a store must share identical sample IDs/order"
            )
        break


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


def _extract_vcf_arrays(
    vcf_path: Path,
    chrom: str | int,
    backend: Backend,
    sample_ids: list[str] | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
) -> _IngestArrays:
    if backend == "auto":
        for candidate in ("cyvcf2", "pysam", "text"):
            try:
                return _extract_vcf_arrays(
                    vcf_path=vcf_path,
                    chrom=chrom,
                    backend=candidate,
                    sample_ids=sample_ids,
                    on_missing_ref_alt=on_missing_ref_alt,
                    on_missing_vid=on_missing_vid,
                    on_unphased_gt=on_unphased_gt,
                )
            except Exception:
                continue
        raise ValueError(f"failed to parse VCF {vcf_path} with backends auto(cyvcf2/pysam/text)")

    if backend == "cyvcf2":
        return _extract_vcf_arrays_cyvcf2(
            vcf_path=vcf_path,
            chrom=chrom,
            sample_ids=sample_ids,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
        )
    if backend == "pysam":
        return _extract_vcf_arrays_pysam(
            vcf_path=vcf_path,
            chrom=chrom,
            sample_ids=sample_ids,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
        )
    if backend == "text":
        return _extract_vcf_arrays_text(
            vcf_path=vcf_path,
            chrom=chrom,
            sample_ids=sample_ids,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
        )

    raise ValueError(f"unsupported backend: {backend}")


def _resolve_subset(
    all_samples: list[str],
    requested: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if requested is None:
        return np.arange(len(all_samples), dtype=np.int64), np.asarray(all_samples, dtype=object)

    sample_to_index = {sid: i for i, sid in enumerate(all_samples)}
    indices: list[int] = []
    for sid in requested:
        if sid not in sample_to_index:
            raise ValueError(f"requested sample_id {sid!r} not present in VCF header")
        indices.append(sample_to_index[sid])
    return np.asarray(indices, dtype=np.int64), np.asarray(requested, dtype=object)


def _finalize_arrays(
    sample_ids: np.ndarray,
    columns: list[np.ndarray],
    positions: list[int],
    ref: list[str],
    alt: list[str],
    vid: list[str],
    missing_ref_count: int,
    missing_alt_count: int,
    missing_vid_count: int,
    unphased_gt_count: int,
) -> _IngestArrays:
    n_samples = int(sample_ids.shape[0])
    if not columns:
        haps = np.zeros((2 * n_samples, 0), dtype=np.uint8)
    else:
        haps = np.column_stack(columns).astype(np.uint8, copy=False)

    return _IngestArrays(
        haplotypes=haps,
        positions=np.asarray(positions, dtype=np.int64),
        ref=np.asarray(ref, dtype=object),
        alt=np.asarray(alt, dtype=object),
        vid=np.asarray(vid, dtype=object),
        sample_ids=np.asarray(sample_ids, dtype=object),
        missing_ref_count=int(missing_ref_count),
        missing_alt_count=int(missing_alt_count),
        missing_vid_count=int(missing_vid_count),
        unphased_gt_count=int(unphased_gt_count),
    )


def _coerce_ref_alt(
    chrom: str,
    pos: int,
    ref: str | None,
    alt: str | None,
    on_missing_ref_alt: MissingRefAltPolicy,
) -> tuple[str, str, int, int]:
    r = "" if ref is None else str(ref)
    a = "" if alt is None else str(alt)
    missing_ref = int((r == "") or (r == "."))
    missing_alt = int((a == "") or (a == "."))

    if missing_ref or missing_alt:
        if on_missing_ref_alt == "error":
            raise ValueError(
                f"missing REF/ALT at {chrom}:{pos}; REF={ref!r} ALT={alt!r}. "
                "Use on_missing_ref_alt='dot' to allow placeholder values."
            )
        if missing_ref:
            r = "."
        if missing_alt:
            a = "."
    return r, a, missing_ref, missing_alt


def _coerce_vid(
    chrom: str,
    pos: int,
    vid: str | None,
    on_missing_vid: MissingVidPolicy,
) -> tuple[str, int]:
    out = "" if vid is None else str(vid)
    missing = int((out == "") or (out == "."))
    if missing:
        if on_missing_vid == "error":
            raise ValueError(
                f"missing variant ID at {chrom}:{pos}; set on_missing_vid='dot' to allow '.'"
            )
        out = "."
    return out, missing


def _extract_vcf_arrays_cyvcf2(
    vcf_path: Path,
    chrom: str | int,
    sample_ids: list[str] | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
) -> _IngestArrays:
    from cyvcf2 import VCF  # type: ignore

    vcf = VCF(str(vcf_path))
    subset_idx, subset_ids = _resolve_subset(list(vcf.samples), sample_ids)

    columns: list[np.ndarray] = []
    positions: list[int] = []
    ref: list[str] = []
    alt: list[str] = []
    vid: list[str] = []
    missing_ref_count = 0
    missing_alt_count = 0
    missing_vid_count = 0
    unphased_gt_count = 0

    for contig in candidate_contigs(chrom):
        found_for_contig = False
        try:
            iterator = vcf(contig)
        except Exception:
            continue

        for variant in iterator:
            found_for_contig = True
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

            gt = variant.genotypes
            hap = np.empty(2 * subset_idx.shape[0], dtype=np.uint8)
            if gt is None:
                hap.fill(_MISSING_SENTINEL)
            else:
                for out_i, sample_i in enumerate(subset_idx):
                    g = gt[int(sample_i)]
                    if g is None or len(g) < 2:
                        hap[2 * out_i] = _MISSING_SENTINEL
                        hap[2 * out_i + 1] = _MISSING_SENTINEL
                        continue

                    a0 = g[0]
                    a1 = g[1]
                    phased = bool(g[2]) if len(g) >= 3 else True

                    if not phased:
                        unphased_gt_count += 1
                        if on_unphased_gt == "error":
                            raise ValueError(
                                f"unphased genotype at {chrom_name}:{pos} sample_index={int(sample_i)}"
                            )
                        hap[2 * out_i] = _MISSING_SENTINEL
                        hap[2 * out_i + 1] = _MISSING_SENTINEL
                        continue

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

        if found_for_contig:
            break

    return _finalize_arrays(
        sample_ids=subset_ids,
        columns=columns,
        positions=positions,
        ref=ref,
        alt=alt,
        vid=vid,
        missing_ref_count=missing_ref_count,
        missing_alt_count=missing_alt_count,
        missing_vid_count=missing_vid_count,
        unphased_gt_count=unphased_gt_count,
    )


def _extract_vcf_arrays_pysam(
    vcf_path: Path,
    chrom: str | int,
    sample_ids: list[str] | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
) -> _IngestArrays:
    import pysam  # type: ignore

    vcf = pysam.VariantFile(str(vcf_path), "r")
    all_samples = list(vcf.header.samples)
    subset_idx, subset_ids = _resolve_subset(all_samples, sample_ids)

    columns: list[np.ndarray] = []
    positions: list[int] = []
    ref: list[str] = []
    alt: list[str] = []
    vid: list[str] = []
    missing_ref_count = 0
    missing_alt_count = 0
    missing_vid_count = 0
    unphased_gt_count = 0

    for contig in candidate_contigs(chrom):
        found_for_contig = False
        try:
            iterator = vcf.fetch(contig)
        except Exception:
            continue

        for record in iterator:
            found_for_contig = True
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

            hap = np.empty(2 * subset_idx.shape[0], dtype=np.uint8)
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
                        raise ValueError(
                            f"unphased genotype at {chrom_name}:{pos} sample={sample_name}"
                        )
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

        if found_for_contig:
            break

    return _finalize_arrays(
        sample_ids=subset_ids,
        columns=columns,
        positions=positions,
        ref=ref,
        alt=alt,
        vid=vid,
        missing_ref_count=missing_ref_count,
        missing_alt_count=missing_alt_count,
        missing_vid_count=missing_vid_count,
        unphased_gt_count=unphased_gt_count,
    )


def _extract_vcf_arrays_text(
    vcf_path: Path,
    chrom: str | int,
    sample_ids: list[str] | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
) -> _IngestArrays:
    import gzip

    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    wanted = set(candidate_contigs(chrom))

    header_samples: list[str] | None = None
    subset_idx: np.ndarray | None = None
    subset_ids: np.ndarray | None = None

    columns: list[np.ndarray] = []
    positions: list[int] = []
    ref: list[str] = []
    alt: list[str] = []
    vid: list[str] = []
    missing_ref_count = 0
    missing_alt_count = 0
    missing_vid_count = 0
    unphased_gt_count = 0

    with opener(vcf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                header_samples = fields[9:]
                subset_idx, subset_ids = _resolve_subset(header_samples, sample_ids)
                continue

            if line.startswith("#"):
                continue

            if header_samples is None or subset_idx is None or subset_ids is None:
                raise ValueError(f"#CHROM header not found in VCF: {vcf_path}")

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue

            contig = fields[0]
            if contig not in wanted:
                continue

            alt_field = fields[4]
            if "," in alt_field:
                continue

            pos = int(fields[1])
            ref_i, alt_i, mref, malt = _coerce_ref_alt(
                chrom=contig,
                pos=pos,
                ref=fields[3],
                alt=alt_field,
                on_missing_ref_alt=on_missing_ref_alt,
            )

            raw_vid = fields[2]
            vid_val = None if raw_vid in {"", "."} else raw_vid
            vid_i, mvid = _coerce_vid(contig, pos, vid_val, on_missing_vid)

            hap = np.empty(2 * subset_idx.shape[0], dtype=np.uint8)
            for out_i, sample_i in enumerate(subset_idx):
                gt_field = fields[9 + int(sample_i)]
                parsed = _parse_text_gt(gt_field)
                if parsed is None:
                    hap[2 * out_i] = _MISSING_SENTINEL
                    hap[2 * out_i + 1] = _MISSING_SENTINEL
                    continue

                a0, a1, phased = parsed
                if not phased:
                    unphased_gt_count += 1
                    if on_unphased_gt == "error":
                        sample_name = header_samples[int(sample_i)]
                        raise ValueError(
                            f"unphased genotype at {contig}:{pos} sample={sample_name}"
                        )
                    hap[2 * out_i] = _MISSING_SENTINEL
                    hap[2 * out_i + 1] = _MISSING_SENTINEL
                    continue

                if a0 is None or a1 is None:
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

    if subset_ids is None:
        raise ValueError(f"#CHROM header not found in VCF: {vcf_path}")

    return _finalize_arrays(
        sample_ids=subset_ids,
        columns=columns,
        positions=positions,
        ref=ref,
        alt=alt,
        vid=vid,
        missing_ref_count=missing_ref_count,
        missing_alt_count=missing_alt_count,
        missing_vid_count=missing_vid_count,
        unphased_gt_count=unphased_gt_count,
    )


def _parse_text_gt(gt_field: str) -> tuple[int | None, int | None, bool] | None:
    token = gt_field.split(":", 1)[0]
    if "|" in token:
        parts = token.split("|")
        phased = True
    elif "/" in token:
        parts = token.split("/")
        phased = False
    else:
        return None

    if len(parts) != 2:
        return None
    if parts[0] == "." or parts[1] == ".":
        return None, None, phased

    try:
        a0 = int(parts[0])
        a1 = int(parts[1])
    except ValueError:
        return None

    return a0, a1, phased


def _to_unicode_array(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=object)
    for i, value in enumerate(values):
        if isinstance(value, bytes):
            out[i] = value.decode("utf-8")
        else:
            out[i] = str(value)
    return out


def _haplotypes_to_genotypes_local(haplotypes: np.ndarray) -> np.ndarray:
    h = np.asarray(haplotypes, dtype=np.uint8)
    if h.ndim != 2 or h.shape[0] % 2 != 0:
        raise ValueError(f"haplotypes must have shape (2N, V), got {h.shape}")

    first = h[0::2, :]
    second = h[1::2, :]
    missing = (first == _MISSING_SENTINEL) | (second == _MISSING_SENTINEL)
    out = (first + second).astype(np.uint8, copy=False)
    out[missing] = _MISSING_SENTINEL
    return out
