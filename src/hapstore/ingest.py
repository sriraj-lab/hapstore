"""Convert phased genomic data into HapStore.

HapStore supports three ingestion paths:

1. In-memory array ingestion validates phased haplotypes and aligned metadata,
   then writes them directly through ``HapStore.create_chromosome()``.

2. Single-worker VCF ingestion reads the selected chromosome into memory,
   applies allele, variant-ID, phasing, and sample-subset policies, optionally
   interpolates genetic-map positions, and writes the result transactionally.

3. Staged parallel VCF ingestion partitions an indexed chromosome into
   deterministic base-pair shards. Worker processes materialize shards into
   run-specific scratch artifacts, which are buffered and streamed into the
   destination HapStore in genomic order. Finalization constructs aligned
   metadata, records ingestion diagnostics, and removes partial output if a
   newly created ingestion fails.

The ``auto`` backend prefers the PLINK2 bridge when it is installed and
compatible with the requested metadata policies; otherwise it falls back to
``pysam``. Explicit parallel ``pysam`` and ``cyvcf2`` ingestion use the shared
staged pipeline.
"""

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

from hapstore.chrom import candidate_contigs, normalize_chrom

from ._internal.command import CommandExecutionError, run_cmd
from ._internal.chunking import DEFAULT_VARIANT_CHUNK_SIZE
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
    _stream_finalize_write_shard,
    _validate_cross_chrom_samples,
)
from ._internal.ingest_models import (
    Backend,
    MultiallelicPolicy,
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
from ._internal.plink import resolve_plink2_binary
from .store import HapStore

logger = logging.getLogger(__name__)

_DEFAULT_PARTITION_BP = 1_000_000
_AUTO_PARTITION_ROUND_BP = 100_000


def _remove_output_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _create_store_transactional(
    *,
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
    """Create one chromosome and remove a newly-created root if it fails."""
    output_path = Path(output_zarr)
    existed_before = output_path.exists()
    try:
        return _create_store_from_ingested(
            output_zarr=output_path,
            chrom=chrom,
            ingested=ingested,
            genetic_positions=genetic_positions,
            store_genotypes=store_genotypes,
            chunk_variant_size=chunk_variant_size,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=compression_level,
            attrs=attrs,
        )
    except Exception:
        if not existed_before and output_path.exists():
            _remove_output_path(output_path)
        raise


def _preflight_vcf_records(
    *,
    vcf_path: Path,
    chrom: str | int,
    on_multiallelic: MultiallelicPolicy,
) -> dict[str, int | str]:
    """Resolve one VCF contig and enforce the biallelic record policy once."""
    import pysam

    candidates = list(candidate_contigs(chrom))
    counts = {contig: {"seen": 0, "biallelic": 0, "multiallelic": 0} for contig in candidates}
    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        for record in vcf:
            contig = str(record.contig)
            if contig not in counts:
                continue
            entry = counts[contig]
            entry["seen"] += 1
            if record.alts is not None and len(record.alts) > 1:
                entry["multiallelic"] += 1
            else:
                entry["biallelic"] += 1

    resolved = next((contig for contig in candidates if counts[contig]["seen"] > 0), None)
    if resolved is None:
        raise ValueError(
            f"no records found for requested chromosome {normalize_chrom(chrom, prefix=True)!r} "
            f"in {vcf_path}"
        )
    result = counts[resolved]
    if result["multiallelic"] and on_multiallelic == "error":
        raise ValueError(
            "multiallelic record encountered while ingesting "
            f"{resolved}; set on_multiallelic='skip' to skip such records "
            f"({result['multiallelic']} encountered)"
        )
    if result["biallelic"] == 0:
        raise ValueError(
            f"no biallelic variants remain for {resolved}; "
            f"seen={result['seen']}, multiallelic_skipped={result['multiallelic']}"
        )
    return {
        "ingest_resolved_contig": str(resolved),
        "records_seen": int(result["seen"]),
        "records_kept": int(result["biallelic"]),
        "multiallelic_skipped_count": int(result["multiallelic"]),
    }


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


def _bridge_backend_supported(
    *,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
) -> bool:
    return on_missing_ref_alt == "error" and on_missing_vid == "dot"


def _write_genetic_positions_to_store(
    *,
    output_zarr: str | Path,
    chrom: str | int,
    genetic_positions: np.ndarray,
) -> HapStore:
    store = HapStore(output_zarr, mode="a")
    chrom_key = normalize_chrom(chrom, prefix=True)
    chrom_group = store.root[chrom_key]
    variants_group = chrom_group["variants"]
    if "genetic_pos" in variants_group:
        del variants_group["genetic_pos"]

    pos_arr = variants_group["pos"]
    chunk_shape = pos_arr.chunks or (int(genetic_positions.shape[0]),)
    variants_group.create_dataset(
        "genetic_pos",
        data=np.asarray(genetic_positions, dtype=np.float64),
        dtype=np.float64,
        chunks=chunk_shape,
        compressor=pos_arr.compressor,
        overwrite=False,
    )
    return store


def _run_plink2_vcf_to_pgen(
    *,
    vcf_path: Path,
    chrom: str | int,
    out_prefix: Path,
    workers: int,
    skip_multiallelic: bool,
) -> tuple[Path, str]:
    plink2_bin = resolve_plink2_binary("plink2")
    failures: list[str] = []
    for contig in candidate_contigs(chrom):
        for suffix in (".pgen", ".pvar", ".psam", ".log"):
            out_prefix.with_suffix(suffix).unlink(missing_ok=True)
        cmd = [
            str(plink2_bin),
            "--vcf",
            str(vcf_path),
            "--allow-extra-chr",
            "--chr",
            str(contig),
            "--make-pgen",
            "--out",
            str(out_prefix),
            "--threads",
            str(int(max(1, workers))),
        ]
        if skip_multiallelic:
            cmd[6:6] = ["--max-alleles", "2"]
        try:
            run_cmd(cmd)
        except CommandExecutionError as exc:
            failures.append(exc.result.stderr or exc.result.stdout or str(exc))
            continue

        prefix = out_prefix
        if prefix.with_suffix(".pgen").exists() and prefix.with_suffix(".pvar").exists() and prefix.with_suffix(".psam").exists():
            return prefix, str(contig)
        failures.append(f"plink2 completed without writing a full PGEN triplet for contig {contig!r}")

    joined = "\n---\n".join(text.strip() for text in failures if text.strip())
    raise RuntimeError(
        f"plink2 bridge failed to create a chromosome-restricted PGEN for {vcf_path} "
        f"chrom={normalize_chrom(chrom, prefix=True)!r}.\n{joined}"
    )


def _ingest_vcf_via_plink2_bridge(
    *,
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
    on_multiallelic: MultiallelicPolicy,
    workers: int,
) -> HapStore:
    """Ingest a chromosome through the PLINK2-to-PGEN bridge.

    This path converts the VCF to a temporary PGEN triplet and then ingests
    that triplet into HapStore.
    """
    if not _bridge_backend_supported(
        on_missing_ref_alt=on_missing_ref_alt,
        on_missing_vid=on_missing_vid,
    ):
        raise ValueError(
            "plink2_bridge only supports on_missing_ref_alt='error' and on_missing_vid='dot'"
        )

    scratch_root = Path(
        tempfile.mkdtemp(
            prefix=f"{vcf_path.stem}.plink2_ingest_",
            dir=str(Path(output_zarr).resolve().parent),
        )
    )
    bridge_prefix = scratch_root / "bridge"
    try:
        pgen_prefix, _resolved_contig = _run_plink2_vcf_to_pgen(
            vcf_path=vcf_path,
            chrom=chrom,
            out_prefix=bridge_prefix,
            workers=int(workers),
            skip_multiallelic=on_multiallelic == "skip",
        )
        from . import pgen as pgen_mod

        attrs_payload = dict(attrs or {})
        attrs_payload.setdefault("source", "vcf")
        attrs_payload["ingest_backend"] = "plink2_bridge"
        store = pgen_mod.ingest_pgen(
            pgen_prefix=pgen_prefix,
            output_zarr=output_zarr,
            chrom=chrom,
            sample_ids=sample_ids,
            on_unphased_gt=on_unphased_gt,
            workers=int(workers),
            chunk_variant_size=int(chunk_variant_size),
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=int(compression_level),
            store_genotypes=bool(store_genotypes),
            attrs=attrs_payload,
        )

        map_df = _load_genetic_map_df(genetic_map_path, chrom=chrom)
        if map_df is not None:
            positions = store.variant_positions(chrom)
            genetic_positions = _interpolate_genetic_positions(positions, map_df=map_df, chrom=chrom)
            if genetic_positions is not None:
                store = _write_genetic_positions_to_store(
                    output_zarr=output_zarr,
                    chrom=chrom,
                    genetic_positions=genetic_positions,
                )
        return store
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


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
    return _create_store_transactional(
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
    """Run the shared staged-ingestion pipeline for one reader backend.

    The pipeline writes a partition manifest, materializes genomic shards in
    parallel into run-specific scratch storage, and streams completed shards
    into the destination HapStore. Worker processes may finish out of order, so
    completed artifacts are buffered by shard index and finalized only when the
    next genomic shard is available. This preserves variant order while
    allowing materialization and final writing to overlap.

    ``submit_materialize`` supplies backend-specific worker submission. The
    remaining partitioning, ordered finalization, metadata construction,
    failure handling, and scratch cleanup are shared across backends.

    On success, every expected shard must be materialized and finalized before
    the chromosome is committed. On failure, partial finalization is aborted
    and a newly created output is removed. Scratch artifacts are deleted when
    ``cleanup_intermediates=True`` and retained otherwise.

    Returns:
        The completed HapStore opened at ``output_zarr``.
"""
    total_start = time.perf_counter()
    output_path = Path(output_zarr)
    output_existed_before = output_path.exists()
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
            # Finalize in genomic shard order even when workers finish out of order.
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
            if not output_existed_before and output_path.exists():
                _remove_output_path(output_path)
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
    chunk_variant_size: int = DEFAULT_VARIANT_CHUNK_SIZE,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    compression_level: int = 3,
    store_genotypes: bool = False,
    backend: Backend = "auto",
    attrs: dict | None = None,
    on_missing_ref_alt: MissingRefAltPolicy = "error",
    on_missing_vid: MissingVidPolicy = "dot",
    on_unphased_gt: UnphasedPolicy = "error",
    on_multiallelic: MultiallelicPolicy = "error",
    workers: int = 1,
    partition_bp: int | None = None,
    scratch_dir: str | Path | None = None,
    cleanup_intermediates: bool = True,
) -> HapStore:
    """Ingest one phased VCF/BCF chromosome into a HapStore.

    With multiple reader workers, pysam and cyvcf2 use staged base-pair shards
    and ordered streaming finalization. Auto mode prefers the PLINK2 bridge
    when its metadata policies are compatible and otherwise uses pysam.

    Args:
        vcf_path: Path to a VCF or BCF file.
        output_zarr: HapStore Zarr hierarchy to create or append to.
        chrom: Chromosome name or number to ingest.
        genetic_map_path: Optional genetic map used to interpolate cM positions.
        sample_ids: Optional sample subset in the desired output order.
        chunk_variant_size: Requested number of variant columns per Zarr chunk.
        chunk_haplotype_size: Optional haplotype rows per Zarr chunk.
        chunk_sample_size: Optional genotype sample rows per Zarr chunk.
        compression_level: Non-negative Zstandard compression level.
        store_genotypes: Also materialize diploid genotype dosages.
        backend: ``"auto"``, ``"pysam"``, ``"cyvcf2"``, ``"text"``, or
            ``"plink2_bridge"``.
        attrs: Optional caller-supplied chromosome provenance metadata.
        on_missing_ref_alt: Reject missing REF/ALT or replace them with dots.
        on_missing_vid: Replace missing IDs with dots or reject them.
        on_unphased_gt: Reject unphased heterozygotes or mark both alleles
            missing. Unphased homozygotes remain unambiguous.
        on_multiallelic: Reject or skip multiallelic records.
        workers: Positive number of reader workers.
        partition_bp: Optional staged-ingestion shard width in base pairs.
        scratch_dir: Optional staged-artifact root. Otherwise, node-local
            ``/scratch/tmp`` is preferred before the system temporary directory.
        cleanup_intermediates: Remove staged artifacts after completion or
            failure; retained artifacts are logged when false.

    Returns:
        The HapStore containing the completed chromosome.

    Raises:
        FileNotFoundError: If the VCF file does not exist.
        ValueError: If inputs violate selection, phasing, metadata, or store
            invariants.
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
    if on_multiallelic not in {"error", "skip"}:
        raise ValueError(
            f"on_multiallelic must be 'error' or 'skip', got {on_multiallelic!r}"
        )
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

    requested_backend = backend
    attrs_payload = dict(attrs or {})
    attrs_payload.update(
        _preflight_vcf_records(
            vcf_path=vcf,
            chrom=chrom,
            on_multiallelic=on_multiallelic,
        )
    )
    attrs_payload["ingest_backend_requested"] = str(requested_backend)
    attrs_payload["on_missing_ref_alt"] = str(on_missing_ref_alt)
    attrs_payload["on_missing_vid"] = str(on_missing_vid)
    attrs_payload["on_unphased_gt"] = str(on_unphased_gt)
    attrs_payload["on_multiallelic"] = str(on_multiallelic)

    if int(workers) > 1:
        _warn_reader_worker_budget(int(workers), backend)

    if backend == "auto":
        if _bridge_backend_supported(
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
        ):
            try:
                resolve_plink2_binary("plink2")
            except FileNotFoundError as exc:
                attrs_payload["auto_fallback_reason"] = str(exc)
            else:
                attrs_payload["ingest_backend"] = "plink2_bridge"
                return _ingest_vcf_via_plink2_bridge(
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
                    attrs=attrs_payload,
                    on_missing_ref_alt=on_missing_ref_alt,
                    on_missing_vid=on_missing_vid,
                    on_unphased_gt=on_unphased_gt,
                    on_multiallelic=on_multiallelic,
                    workers=int(workers),
                )
        else:
            attrs_payload["auto_fallback_reason"] = "requested metadata policy is unsupported by plink2_bridge"
        backend = "pysam"
        attrs_payload["ingest_backend"] = "pysam"

    if backend == "plink2_bridge":
        attrs_payload["ingest_backend"] = "plink2_bridge"
        return _ingest_vcf_via_plink2_bridge(
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
            attrs=attrs_payload,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
            on_multiallelic=on_multiallelic,
            workers=int(workers),
        )

    attrs_payload["ingest_backend"] = str(backend)

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
            attrs=attrs_payload,
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
            attrs=attrs_payload,
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
        attrs=attrs_payload,
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

    return _create_store_transactional(
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
    chunk_variant_size: int = DEFAULT_VARIANT_CHUNK_SIZE,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    compression_level: int = 3,
    attrs: dict | None = None,
) -> HapStore:
    """Ingest in-memory arrays into a HapStore.

    Args:
        output_zarr: HapStore Zarr hierarchy to create or append to.
        chrom: Chromosome name or number to ingest.
        haplotypes: Phased ``uint8``-compatible array with shape ``(2N, V)``.
        variant_pos: One-based physical positions with shape ``(V,)``.
        sample_ids: Unique diploid sample IDs with shape ``(N,)``.
        ref: Optional reference alleles with shape ``(V,)``.
        alt: Optional alternate alleles with shape ``(V,)``.
        vid: Optional variant IDs with shape ``(V,)``.
        genetic_positions: Optional cM positions with shape ``(V,)``.
        populations: Optional population labels with shape ``(N,)``.
        store_genotypes: Also materialize diploid genotype dosages.
        chunk_variant_size: Requested variant columns per Zarr chunk.
        chunk_haplotype_size: Optional haplotype rows per Zarr chunk.
        chunk_sample_size: Optional genotype sample rows per Zarr chunk.
        compression_level: Non-negative Zstandard compression level.
        attrs: Optional caller-supplied chromosome provenance metadata.

    Returns:
        The HapStore containing the completed chromosome.

    Notes:
        Input row and column order is preserved. If ingestion fails after
        creating a new output path, that partial output is removed.
    """
    output_path = Path(output_zarr)
    existed_before = output_path.exists()
    try:
        store = HapStore(output_path, mode="a")
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
    except Exception:
        if not existed_before and output_path.exists():
            _remove_output_path(output_path)
        raise


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
    """Prepare and run staged ingestion with pysam workers.

    Requires a TBI- or CSI-indexed input. This adapter resolves the contig,
    sample subset, and partition plan before delegating parallel materialization
    and ordered finalization to ``_run_staged_backend``. A single-shard plan
    falls back to the non-staged path.
    """
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
    """Prepare and run staged ingestion with cyvcf2 workers.

    Pysam is used to inspect the indexed header and build the partition plan;
    cyvcf2 parses each resulting shard. Ordered finalization is shared with the
    pysam path.
    """
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
    """Resolve the base-pair partition size for staged ingestion.

    Args:
        vcf: A pysam.VariantFile object for the VCF/BCF.
        contig: The chromosome/contig name to ingest.
        workers: The number of parallel workers to use.
        partition_bp: Optional user-specified partition size in base pairs.

    Returns:
        The explicit partition size, or a size calculated from the contig span
        and worker count. A fixed default is used when the span is unavailable.

    Raises:
        ValueError: If partition_bp is provided and is not greater than 0.

    Notes:
        Automatic sizes are rounded up to ``_AUTO_PARTITION_ROUND_BP`` to avoid
        creating many small shards. ``_DEFAULT_PARTITION_BP`` is used when the
        contig span cannot be determined.
    """
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


def _serialize_shard_to_artifact(
    shard: _ShardResult,
    start: int | None,
    end: int | None,
    artifacts_dir: str,
) -> _ShardArtifactMeta:
    artifact_path = Path(artifacts_dir) / f"shard_{int(shard.shard_index):06d}.npz"
    np.savez(
        str(artifact_path),
        haplotypes=np.asarray(shard.haplotypes, dtype=np.uint8),
        positions=np.asarray(shard.positions, dtype=np.int64),
        ref=np.asarray(shard.ref, dtype=str),
        alt=np.asarray(shard.alt, dtype=str),
        vid=np.asarray(shard.vid, dtype=str),
    )
    return _ShardArtifactMeta(
        shard_index=int(shard.shard_index),
        start=None if start is None else int(start),
        end=None if end is None else int(end),
        artifact_path=str(artifact_path),
        n_variants=int(shard.positions.shape[0]),
        missing_ref_count=int(shard.missing_ref_count),
        missing_alt_count=int(shard.missing_alt_count),
        missing_vid_count=int(shard.missing_vid_count),
        unphased_gt_count=int(shard.unphased_gt_count),
    )


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
    return _serialize_shard_to_artifact(shard, start, end, artifacts_dir)


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
    return _serialize_shard_to_artifact(shard, start, end, artifacts_dir)


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
    """Read one cyvcf2 shard using the pysam worker's output contract.

    See :func:`_read_pysam_shard_worker` for the shared allele, missingness,
    phasing, ordering, and diagnostic-count semantics. This implementation
    selects samples by name and converts the shared zero-based, half-open shard
    bounds to cyvcf2's one-based region syntax.
    """
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
        if variant.ALT is not None and len(variant.ALT) > 1:
            continue

        pos = int(variant.POS)
        chrom_name = str(variant.CHROM)
        ref_i, alt_i, mref, malt = _coerce_ref_alt(
            chrom=chrom_name,
            pos=pos,
            ref=variant.REF,
            alt="." if variant.ALT is None else variant.ALT[0],
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
        called = (a0 >= 0) & (a1 >= 0)
        unphased_het = (~phased) & called & (a0 != a1)
        if bool(np.any(unphased_het)):
            unphased_gt_count += int(np.sum(unphased_het))
            if on_unphased_gt == "error":
                bad = int(np.flatnonzero(unphased_het)[0])
                raise ValueError(
                    "unphased genotype is heterozygous at "
                    f"{chrom_name}:{pos} sample={subset_samples[bad]}"
                )

        missing = (a0 < 0) | (a1 < 0) | unphased_het
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
    """Read one genomic shard into phased HapStore-oriented arrays.

    ``start`` and ``end`` are zero-based, half-open pysam fetch bounds; when
    either is ``None``, the full contig is read. Samples follow ``subset_idx``
    order, with each sample's two haplotypes stored in adjacent rows.

    Multiallelic records are skipped after top-level preflight. Biallelic
    alleles are encoded as ``0`` or ``1`` and missing alleles as
    ``_MISSING_SENTINEL``. Missing REF, ALT, and variant IDs follow their
    configured policies. Unphased heterozygotes either raise or become missing
    on both haplotypes; unphased homozygotes remain unambiguous. Returned
    variants preserve backend iteration order, and diagnostic counts summarize
    metadata substitutions and unphased heterozygotes.
    """
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
        if record.alts is not None and len(record.alts) > 1:
            continue

        pos = int(record.pos)
        chrom_name = str(record.contig)
        ref_i, alt_i, mref, malt = _coerce_ref_alt(
            chrom=chrom_name,
            pos=pos,
            ref=record.ref,
            alt="." if record.alts is None else record.alts[0],
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

            a0, a1 = gt[0], gt[1]
            if a0 is None or a1 is None or int(a0) < 0 or int(a1) < 0:
                hap[2 * out_i] = _MISSING_SENTINEL
                hap[2 * out_i + 1] = _MISSING_SENTINEL
                continue

            if not phased and int(a0) != int(a1):
                unphased_gt_count += 1
                if on_unphased_gt == "error":
                    raise ValueError(
                        "unphased genotype is heterozygous at "
                        f"{chrom_name}:{pos} sample={sample_name}"
                    )
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
