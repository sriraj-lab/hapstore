"""Direct conversion between HapStore and phased, biallelic PGEN data.

Serial export streams HapStore variant blocks into one PGEN/PVAR/PSAM triplet.
Parallel export divides the selected variant axis into independent PGEN shards
and uses PLINK2 to merge those shards in genomic order. Serial ingestion
streams PGEN blocks directly into a HapStore, while parallel ingestion writes
contiguous variant ranges to temporary HapStores and concatenates them along
the variant axis.

This module owns conversion policy, sample and variant selection, missingness
handling, and serial/parallel orchestration. Low-level pgenlib buffer handling
is isolated in ``hapstore._internal.pgenlib_io``.
"""

from __future__ import annotations

import concurrent.futures
import functools
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Literal

import numpy as np

from hapstore.chrom import normalize_chrom
from hapstore.ids import read_sample_ids

from ._internal.chunking import DEFAULT_VARIANT_CHUNK_SIZE
from ._internal.export_selection import (
    _iter_variant_slices,
    _iter_variant_slices_for_indices,
    _plan_sample_subset,
    _resolve_variant_indices_streaming,
    _subset_indices_in_requested_order,
)
from ._internal.pgenlib_io import PgenlibReader, PgenlibWriter, ensure_pgenlib_available
from ._internal.ingest_finalize import (
    _abort_streaming_finalize,
    _complete_streaming_finalize,
    _create_streaming_finalize_context,
    _stream_finalize_write_arrays,
)
from ._internal.ingest_models import _PartitionPlan, _ShardSpec
from ._internal.plink import resolve_plink2_binary
from ._internal.zarr_utils import _to_unicode_array
from .concat import concat_hapstores
from .store import HapStore, _MISSING_SENTINEL

PartialMissingPolicy = Literal["error", "missing"]
UnphasedPolicy = Literal["error", "missing"]


def _read_pgen_haplotype_block(
    *,
    reader: PgenlibReader,
    chrom_key: str,
    start: int,
    end: int,
    hap_reorder: np.ndarray | None,
    strict_biallelic: bool,
    on_unphased_gt: UnphasedPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Read one PGEN block into HapStore's haplotype-major representation.

    PGEN returns a variant-major matrix with each sample's two alleles in
    adjacent columns. This function converts it to the HapStore layout of
    ``(2 * samples, variants)``. Missing diploid pairs are encoded with
    ``_MISSING_SENTINEL``. Unphased heterozygotes either raise or become
    missing on both haplotypes; unphased homozygotes remain unambiguous.
    ``hap_reorder`` restores the caller's requested sample order after pgenlib
    reads a sorted sample subset.

    Returns:
        Haplotype data, positions, REF alleles, ALT alleles, variant IDs, and
        the number of unphased heterozygous genotypes encountered.
    """
    width = int(end - start)
    positions = np.empty(width, dtype=np.int64)
    refs = np.empty(width, dtype=object)
    alts = np.empty(width, dtype=object)
    vids = np.empty(width, dtype=object)
    for rel_idx, variant_idx in enumerate(range(int(start), int(end))):
        var_chrom = normalize_chrom(reader.variant_chrom(variant_idx), prefix=True)
        if var_chrom != chrom_key:
            raise ValueError(
                f"pvar chromosome mismatch at variant {variant_idx}: {var_chrom!r} vs requested {chrom_key!r}"
            )
        allele_ct = int(reader.allele_ct(variant_idx))
        if strict_biallelic and allele_ct > 2:
            raise ValueError(
                f"non-biallelic variant encountered at {chrom_key}:{reader.variant_pos(variant_idx)} "
                f"(allele_ct={allele_ct})"
            )
        pos = int(reader.variant_pos(variant_idx))
        ref, alt = _normalize_ref_alt_for_pvar(
            ref=reader.allele_code(variant_idx, 0),
            alt=reader.allele_code(variant_idx, 1) if allele_ct >= 2 else ".",
            strict_biallelic=bool(strict_biallelic),
            chrom_key=chrom_key,
            pos=pos,
        )
        raw_vid = reader.variant_id(variant_idx).strip()
        positions[rel_idx] = pos
        refs[rel_idx] = ref
        alts[rel_idx] = alt
        vids[rel_idx] = raw_vid if raw_vid not in {"", "."} else "."

    allele_block = np.empty((width, 2 * int(reader.sample_ct)), dtype=np.int32, order="C")
    phase_block = np.empty((width, int(reader.sample_ct)), dtype=np.uint8, order="C")
    reader.read_alleles_and_phasepresent_range(int(start), int(end), allele_block, phase_block)

    valid = allele_block >= 0
    if strict_biallelic and np.any(allele_block[valid] > 1):
        bad = np.argwhere((allele_block >= 0) & (allele_block > 1))[0]
        raise ValueError(
            "non-biallelic allele code encountered while ingesting "
            f"{chrom_key}:{int(positions[int(bad[0])])}"
        )

    a0 = allele_block[:, 0::2]
    a1 = allele_block[:, 1::2]
    missing = (a0 < 0) | (a1 < 0)
    unphased_het = (phase_block == 0) & (~missing) & (a0 != a1)
    unphased_gt_count = int(np.sum(unphased_het))
    if unphased_gt_count:
        if on_unphased_gt == "error":
            first = np.argwhere(unphased_het)[0]
            raise ValueError(
                "unphased heterozygous genotype encountered while ingesting PGEN "
                f"(variant={chrom_key}:{int(positions[int(first[0])])}, sample_index={int(first[1])})"
            )
        a0[unphased_het] = -9
        a1[unphased_het] = -9

    pair_missing = (a0 < 0) | (a1 < 0)
    a0[pair_missing] = -9
    a1[pair_missing] = -9
    haplotypes = np.where(allele_block.T < 0, _MISSING_SENTINEL, allele_block.T).astype(
        np.uint8,
        copy=False,
    )
    if hap_reorder is not None:
        haplotypes = haplotypes[hap_reorder, :]
    return (
        np.ascontiguousarray(haplotypes),
        positions,
        refs,
        alts,
        vids,
        unphased_gt_count,
    )


def _stream_pgen_range_to_store(
    *,
    pgen_path: Path,
    pvar_path: Path,
    output_zarr: str | Path,
    chrom_key: str,
    selected_sample_ids: list[str],
    reader_subset: np.ndarray | None,
    hap_reorder: np.ndarray | None,
    start_variant: int,
    end_variant: int,
    variant_block_size: int,
    on_unphased_gt: UnphasedPolicy,
    strict_biallelic: bool,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
    store_genotypes: bool,
    attrs: dict | None,
) -> HapStore:
    """Stream a bounded PGEN variant range into a new HapStore chromosome.

    The range is read in ordered blocks and appended through the shared
    streaming-finalization machinery. Successful completion trims provisional
    arrays and commits chromosome metadata. A failed write aborts the partial
    chromosome and removes a newly created output path.
    """
    if int(start_variant) < 0 or int(end_variant) <= int(start_variant):
        raise ValueError(f"invalid PGEN variant range [{start_variant}, {end_variant})")
    output_path = Path(output_zarr)
    output_existed_before = output_path.exists()
    starts = list(range(int(start_variant), int(end_variant), int(variant_block_size)))
    shard_specs = [_ShardSpec(shard_index=i, start_bp=None, end_bp=None) for i in range(len(starts))]
    partition_plan = _PartitionPlan(partition_bp=int(variant_block_size), shards=shard_specs)
    ctx = None
    try:
        ctx = _create_streaming_finalize_context(
            output_zarr=output_path,
            chrom=chrom_key,
            subset_ids=np.asarray(selected_sample_ids, dtype=object),
            shard_specs=shard_specs,
            map_df=None,
            store_genotypes=bool(store_genotypes),
            chunk_variant_size=int(chunk_variant_size),
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=int(compression_level),
        )
        with PgenlibReader(
            pgen_path=pgen_path,
            pvar_path=pvar_path,
            sample_subset=reader_subset,
        ) as reader:
            if int(reader.sample_ct) != len(selected_sample_ids):
                raise ValueError(
                    "PGEN sample subset mismatch: "
                    f"reader={reader.sample_ct}, selected={len(selected_sample_ids)}"
                )
            for shard_index, start in enumerate(starts):
                end = min(int(end_variant), int(start) + int(variant_block_size))
                haps, positions, ref, alt, vid, unphased_count = _read_pgen_haplotype_block(
                    reader=reader,
                    chrom_key=chrom_key,
                    start=int(start),
                    end=int(end),
                    hap_reorder=hap_reorder,
                    strict_biallelic=bool(strict_biallelic),
                    on_unphased_gt=on_unphased_gt,
                )
                _stream_finalize_write_arrays(
                    ctx=ctx,
                    shard_index=int(shard_index),
                    haplotypes=haps,
                    positions=positions,
                    ref=ref,
                    alt=alt,
                    vid=vid,
                    chrom=chrom_key,
                    map_df=None,
                    started_at=time.perf_counter(),
                    missing_sentinel=int(_MISSING_SENTINEL),
                    unphased_gt_count=int(unphased_count),
                )
        payload = dict(attrs or {})
        payload.setdefault("source", "pgen")
        payload["pgen_variant_block_size"] = int(variant_block_size)
        return _complete_streaming_finalize(
            ctx=ctx,
            attrs=payload,
            workers=1,
            partition_plan=partition_plan,
            missing_sentinel=int(_MISSING_SENTINEL),
            parallel_mode="pgen_direct_streaming",
        )
    except Exception:
        _abort_streaming_finalize(ctx)
        if not output_existed_before and output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path, ignore_errors=True)
            else:
                output_path.unlink(missing_ok=True)
        raise


def _cleanup_failed_pgen_export(func):
    """Ensure a failed public export cannot leave a plausible partial triplet."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        out_prefix = kwargs.get("out_prefix", args[1] if len(args) > 1 else None)
        if out_prefix is None:
            raise TypeError("export_pgen requires out_prefix")
        prefix = _normalize_pgen_prefix(out_prefix)
        try:
            result = func(*args, **kwargs)
            _validate_pgen_triplet(prefix)
            return result
        except Exception:
            _remove_pgen_triplet(prefix)
            raise

    return wrapped


@_cleanup_failed_pgen_export
def export_pgen(
    store_path: str | Path,
    out_prefix: str | Path,
    chrom: str | int,
    sample_ids: list[str] | None = None,
    variant_ids: list[str] | None = None,
    variant_indices: list[int] | None = None,
    min_maf: float | None = None,
    filter_monomorphic: bool = False,
    variant_block_size: int = 8192,
    strict_biallelic: bool = True,
    on_partial_missing: PartialMissingPolicy = "error",
    n_workers: int = 1,
    plink2_workers: int = 1,
    plink2_path: str | Path = "plink2",
    shard_tmp_dir: str | Path | None = None,
    keep_intermediate_shards: bool = False,
) -> Path:
    """Export one HapStore chromosome to a PGEN/PVAR/PSAM triplet.

    Current scope is diploid, phased, biallelic hardcalls only.

    Args:
        store_path: Path to the HapStore Zarr hierarchy.
        out_prefix: Output prefix for the ``.pgen``, ``.pvar``, and ``.psam``
            files.
        chrom: Chromosome to export. The key is normalized to ``chrN`` form.
        sample_ids: Optional sample subset in the desired output order. All
            samples are exported when omitted.
        variant_ids: Optional variant-ID or position-based subset. Mutually
            exclusive with ``variant_indices``.
        variant_indices: Optional zero-based variant-index subset. Indices are
            deduplicated and exported in store order. Mutually exclusive with
            ``variant_ids``.
        min_maf: Optional minimum minor allele frequency calculated within the
            selected output samples.
        filter_monomorphic: Remove variants with zero minor allele frequency
            within the selected output samples.
        variant_block_size: Number of variants per read/write block.
        strict_biallelic: Must remain ``True``; non-biallelic export is
            unsupported.
        on_partial_missing: Handling for a diploid genotype with one missing
            allele. ``"error"`` rejects it and ``"missing"`` marks both
            alleles missing.
        n_workers: Number of variant-shard export workers. Values greater than
            one require PLINK2 for the final merge.
        plink2_workers: Number of PLINK2 threads used to merge parallel shards.
        plink2_path: PLINK2 executable used to merge parallel shards.
        shard_tmp_dir: Optional directory for parallel-export shard triplets.
        keep_intermediate_shards: Retain parallel-export shard files instead
            of cleaning them after the export attempt.

    Returns:
        Path to the completed PGEN/PVAR/PSAM triplet prefix.

    Raises:
        ValueError: If selection, missingness, or biallelic validation fails.
        RuntimeError: If writing or merging does not produce a valid triplet.

    Notes:
        Variant subsets retain HapStore order. Parallel export writes
        independent variant shards and combines them with PLINK2. Public export
        validation removes the entire output triplet after any failure so a
        partial result cannot be mistaken for a complete export.
    """
    if not strict_biallelic:
        raise ValueError("HapStore PGEN export supports biallelic hardcalls only; strict_biallelic=False is unsupported")
    if int(n_workers) <= 0:
        raise ValueError(f"n_workers must be positive, got {n_workers}")
    if int(plink2_workers) <= 0:
        raise ValueError(f"plink2_workers must be positive, got {plink2_workers}")

    if int(n_workers) == 1:
        return _export_pgen_single(
            store_path=store_path,
            out_prefix=out_prefix,
            chrom=chrom,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
            variant_indices=variant_indices,
            min_maf=min_maf,
            filter_monomorphic=bool(filter_monomorphic),
            variant_block_size=int(variant_block_size),
            strict_biallelic=bool(strict_biallelic),
            on_partial_missing=on_partial_missing,
        )

    ensure_pgenlib_available()
    if int(variant_block_size) <= 0:
        raise ValueError(f"variant_block_size must be positive, got {variant_block_size}")
    if on_partial_missing not in {"error", "missing"}:
        raise ValueError(f"on_partial_missing must be 'error' or 'missing', got {on_partial_missing!r}")
    if variant_ids is not None and variant_indices is not None:
        raise ValueError("variant_ids and variant_indices are mutually exclusive")

    store = HapStore(store_path, mode="r")
    chrom_key = normalize_chrom(chrom, prefix=True)
    n_variants = int(store.n_variants(chrom_key))
    sample_plan = _plan_sample_subset(store.sample_ids(chrom_key), sample_ids)

    prefix = _normalize_pgen_prefix(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _remove_pgen_triplet(prefix)

    selected_indices = _resolve_export_variant_indices(
        store=store,
        chrom_key=chrom_key,
        variant_ids=variant_ids,
        variant_indices=variant_indices,
        min_maf=min_maf,
        filter_monomorphic=bool(filter_monomorphic),
        sample_mask=sample_plan.sample_mask,
        n_variants=n_variants,
        scan_block_size=int(variant_block_size),
    )
    if selected_indices is None:
        selected_indices = list(range(n_variants))
    if not selected_indices:
        raise ValueError("selected variant set is empty after applying filters")

    shard_idx_arrays = [arr for arr in np.array_split(np.asarray(selected_indices, dtype=np.int64), int(n_workers)) if arr.size > 0]
    if len(shard_idx_arrays) <= 1:
        return _export_pgen_single(
            store_path=store_path,
            out_prefix=prefix,
            chrom=chrom_key,
            sample_ids=sample_ids,
            variant_indices=[int(i) for i in selected_indices],
            min_maf=None,
            filter_monomorphic=False,
            variant_block_size=int(variant_block_size),
            strict_biallelic=bool(strict_biallelic),
            on_partial_missing=on_partial_missing,
        )

    plink2_bin = str(resolve_plink2_binary(plink2_path))
    shard_prefixes: list[Path] = []
    temp_parent = prefix.parent if shard_tmp_dir is None else Path(shard_tmp_dir)
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f"{prefix.name}.shards_", dir=str(temp_parent)))

    try:
        for shard_idx, idx_array in enumerate(shard_idx_arrays):
            shard_prefix = temp_root / f"{prefix.name}.shard{(shard_idx + 1):03d}of{len(shard_idx_arrays):03d}"
            shard_prefixes.append(shard_prefix)

        mp_context = mp.get_context("spawn")
        # Each worker writes one variant shard; PLINK2 combines them below.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(int(n_workers), len(shard_prefixes)),
            mp_context=mp_context,
        ) as pool:
            futures = []
            for shard_prefix, idx_array in zip(shard_prefixes, shard_idx_arrays, strict=True):
                futures.append(
                    pool.submit(
                        _export_pgen_shard_worker,
                        store_path=str(store_path),
                        out_prefix=str(shard_prefix),
                        chrom=str(chrom_key),
                        sample_ids=list(sample_ids) if sample_ids is not None else None,
                        variant_indices=[int(i) for i in idx_array.tolist()],
                        variant_block_size=int(variant_block_size),
                        strict_biallelic=bool(strict_biallelic),
                        on_partial_missing=on_partial_missing,
                    )
                )
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        _merge_pgen_shards_with_plink(
            shard_prefixes=shard_prefixes,
            out_prefix=prefix,
            plink2_bin=plink2_bin,
            threads=int(plink2_workers),
        )
        _validate_pgen_triplet_exists(prefix)
        return prefix
    finally:
        if not keep_intermediate_shards:
            for shard_prefix in shard_prefixes:
                _remove_pgen_triplet(shard_prefix)
            if temp_root.exists():
                shutil.rmtree(temp_root, ignore_errors=True)


def _export_pgen_shard_worker(
    *,
    store_path: str,
    out_prefix: str,
    chrom: str,
    sample_ids: list[str] | None,
    variant_indices: list[int] | None,
    variant_block_size: int,
    strict_biallelic: bool,
    on_partial_missing: PartialMissingPolicy,
) -> str:
    prefix = _export_pgen_single(
        store_path=store_path,
        out_prefix=out_prefix,
        chrom=chrom,
        sample_ids=sample_ids,
        variant_indices=variant_indices,
        variant_block_size=int(variant_block_size),
        strict_biallelic=bool(strict_biallelic),
        on_partial_missing=on_partial_missing,
    )
    return str(prefix)


def _export_pgen_single(
    *,
    store_path: str | Path,
    out_prefix: str | Path,
    chrom: str | int,
    sample_ids: list[str] | None = None,
    variant_ids: list[str] | None = None,
    variant_indices: list[int] | None = None,
    min_maf: float | None = None,
    filter_monomorphic: bool = False,
    variant_block_size: int = 8192,
    strict_biallelic: bool = True,
    on_partial_missing: PartialMissingPolicy = "error",
) -> Path:
    """Stream one HapStore chromosome or subset into a PGEN/PVAR/PSAM triplet.

    Sample order follows the caller's request and variant order follows the
    HapStore. Haplotype blocks are transposed into PGEN's variant-major,
    interleaved-allele representation. PVAR and PSAM metadata are written in
    the same order as the PGEN payload.
    """
    ensure_pgenlib_available()
    if int(variant_block_size) <= 0:
        raise ValueError(f"variant_block_size must be positive, got {variant_block_size}")
    if on_partial_missing not in {"error", "missing"}:
        raise ValueError(f"on_partial_missing must be 'error' or 'missing', got {on_partial_missing!r}")
    if variant_ids is not None and variant_indices is not None:
        raise ValueError("variant_ids and variant_indices are mutually exclusive")

    store = HapStore(store_path, mode="r")
    chrom_key = normalize_chrom(chrom, prefix=True)

    prefix = _normalize_pgen_prefix(out_prefix)
    pgen_path, pvar_path, psam_path = _pgen_paths(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _remove_pgen_triplet(prefix)

    all_sample_ids = store.sample_ids(chrom_key)
    sample_plan = _plan_sample_subset(all_sample_ids, sample_ids)

    chrom_group = store.root[chrom_key]
    variants = chrom_group["variants"]
    pos_ds = variants["pos"]
    ref_ds = variants["ref"]
    alt_ds = variants["alt"]
    vid_ds = variants["vid"]
    n_variants = int(pos_ds.shape[0])

    # Force block size to a multiple of zarr chunk size for efficient reads
    chunk_variant_size = int(
        store.attrs(chrom_key).get(
            "chunk_variant_size", DEFAULT_VARIANT_CHUNK_SIZE
        )
        or DEFAULT_VARIANT_CHUNK_SIZE
    )
    if chunk_variant_size > 0:
        original_block_size = int(variant_block_size)
        if original_block_size < chunk_variant_size:
            aligned = original_block_size
        else:
            aligned = max(chunk_variant_size, (original_block_size // chunk_variant_size) * chunk_variant_size)
        if aligned != original_block_size:
            import logging
            logging.getLogger(__name__).info(
                "export_pgen: variant_block_size %d rounded down to %d (multiple of chunk_variant_size=%d)",
                original_block_size, aligned, chunk_variant_size,
            )
            variant_block_size = aligned

    selected_indices = _resolve_export_variant_indices(
        store=store,
        chrom_key=chrom_key,
        variant_ids=variant_ids,
        variant_indices=variant_indices,
        min_maf=min_maf,
        filter_monomorphic=bool(filter_monomorphic),
        sample_mask=sample_plan.sample_mask,
        n_variants=n_variants,
        scan_block_size=int(variant_block_size),
    )
    if selected_indices is None:
        slices = _iter_variant_slices(n_variants, int(variant_block_size))
        n_out_variants = n_variants
    else:
        if not selected_indices:
            raise ValueError("selected variant set is empty after applying filters")
        slices = _iter_variant_slices_for_indices(selected_indices, int(variant_block_size))
        n_out_variants = len(selected_indices)

    with psam_path.open("w", encoding="utf-8") as psam_handle:
        psam_handle.write("#FID\tIID\tPAT\tMAT\tSEX\tPHENO\n")
        for sid in sample_plan.selected_sample_ids:
            psam_handle.write(f"0\t{sid}\t0\t0\t0\t-9\n")

    # Pre-load all variant metadata once to avoid per-block zarr reads
    all_pos = np.asarray(pos_ds[:], dtype=np.int64)
    all_ref = _to_unicode_array(np.asarray(ref_ds[:], dtype=object))
    all_alt = _to_unicode_array(np.asarray(alt_ds[:], dtype=object))
    all_vid = _to_unicode_array(np.asarray(vid_ds[:], dtype=object))

    written = 0
    with pvar_path.open("w", encoding="utf-8") as pvar_handle:
        pvar_handle.write("#CHROM\tPOS\tID\tREF\tALT\n")
        with PgenlibWriter(
            pgen_path=pgen_path,
            sample_ct=len(sample_plan.selected_sample_ids),
            variant_ct=n_out_variants,
        ) as writer:
            for variant_sel in slices:
                start = int(variant_sel.start or 0)
                stop = int(variant_sel.stop or 0)
                if start >= stop:
                    continue

                hap_block = store.haplotypes(
                    chrom_key,
                    variant_slice=(start, stop),
                    sample_mask=sample_plan.sample_mask,
                )
                if sample_plan.hap_perm is not None:
                    hap_block = hap_block[sample_plan.hap_perm, :]

                _validate_haplotypes_biallelic_block(
                    hap_block=hap_block,
                    chrom_key=chrom_key,
                    start=start,
                    strict_biallelic=bool(strict_biallelic),
                )

                pos_block = all_pos[start:stop]
                ref_block = all_ref[start:stop].tolist()
                alt_block = all_alt[start:stop].tolist()
                vid_block = all_vid[start:stop].tolist()

                allele_batch = np.ascontiguousarray(hap_block.T.astype(np.int32, copy=False))
                allele_batch[allele_batch == int(_MISSING_SENTINEL)] = -9
                _handle_partial_missing_pairs(
                    allele_batch=allele_batch,
                    policy=on_partial_missing,
                    chrom_key=chrom_key,
                    positions=pos_block,
                )

                for i in range(allele_batch.shape[0]):
                    ref_i, alt_i = _normalize_ref_alt_for_pvar(
                        ref=ref_block[i],
                        alt=alt_block[i],
                        strict_biallelic=bool(strict_biallelic),
                        chrom_key=chrom_key,
                        pos=int(pos_block[i]),
                    )
                    raw_vid = str(vid_block[i]).strip()
                    vid_i = raw_vid if raw_vid not in {"", "."} else "."
                    pvar_handle.write(f"{chrom_key}\t{int(pos_block[i])}\t{vid_i}\t{ref_i}\t{alt_i}\n")

                writer.append_alleles_batch(allele_batch)
                written += int(allele_batch.shape[0])

    if written != int(n_out_variants):
        raise RuntimeError(
            f"export_pgen wrote {written} variants but expected {n_out_variants}"
        )

    _validate_pgen_triplet_exists(prefix)
    return prefix


def ingest_pgen(
    pgen_prefix: str | Path,
    output_zarr: str | Path,
    chrom: str | int,
    sample_ids: list[str] | None = None,
    variant_block_size: int = 8192,
    on_unphased_gt: UnphasedPolicy = "error",
    strict_biallelic: bool = True,
    workers: int = 1,
    chunk_variant_size: int = DEFAULT_VARIANT_CHUNK_SIZE,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    compression_level: int = 3,
    store_genotypes: bool = False,
    attrs: dict | None = None,
    scratch_dir: str | Path | None = None,
    cleanup_intermediates: bool = True,
) -> HapStore:
    """Ingest one chromosome from a PGEN/PVAR/PSAM triplet into HapStore.

    Current scope is diploid, phased, biallelic hardcalls only.

    Args:
        pgen_prefix: Input prefix for the ``.pgen``, ``.pvar``, and ``.psam``
            files.
        output_zarr: Path to the HapStore Zarr hierarchy to create or append to.
        chrom: Chromosome to ingest. The key is normalized to ``chrN`` form.
        sample_ids: Optional sample subset in the desired HapStore order. All
            samples are ingested when omitted.
        variant_block_size: Number of variants per read/write block.
        on_unphased_gt: Handling for unphased heterozygous genotypes.
            ``"error"`` rejects them and ``"missing"`` marks both haplotypes
            missing. Unphased homozygotes are retained.
        strict_biallelic: Must remain ``True``; non-biallelic ingestion is
            unsupported.
        workers: Number of contiguous variant-range workers.
        chunk_variant_size: Chunk size for variants in the output store.
        chunk_haplotype_size: Chunk size for haplotypes in the output store.
        chunk_sample_size: Chunk size for samples in the output store.
        compression_level: Non-negative Zstandard compression level.
        store_genotypes: Also materialize diploid genotype dosages.
        attrs: Optional caller-supplied chromosome provenance metadata.
        scratch_dir: Optional directory for temporary parallel-ingestion
            HapStores.
        cleanup_intermediates: Remove temporary parallel-ingestion HapStores
            after completion or failure.

    Returns:
        The completed HapStore opened at ``output_zarr``.

    Raises:
        ValueError: If sample, chromosome, phasing, or biallelic validation
            fails.

    Notes:
        Serial ingestion streams blocks directly into the destination.
        Parallel ingestion creates one temporary HapStore per contiguous
        variant range and concatenates those stores along the variant axis.
    """
    if not strict_biallelic:
        raise ValueError("HapStore PGEN ingest supports biallelic hardcalls only; strict_biallelic=False is unsupported")
    ensure_pgenlib_available()
    if int(variant_block_size) <= 0:
        raise ValueError(f"variant_block_size must be positive, got {variant_block_size}")
    if on_unphased_gt not in {"error", "missing"}:
        raise ValueError(f"on_unphased_gt must be 'error' or 'missing', got {on_unphased_gt!r}")
    if int(workers) <= 0:
        raise ValueError(f"workers must be >=1, got {workers}")

    prefix = _normalize_pgen_prefix(pgen_prefix)
    pgen_path, pvar_path, psam_path = _pgen_paths(prefix)
    _validate_pgen_triplet_exists(prefix)

    chrom_key = normalize_chrom(chrom, prefix=True)
    all_sample_ids = read_sample_ids(psam_path)
    sample_plan = _plan_sample_subset(all_sample_ids, sample_ids)
    selected_sample_ids = sample_plan.selected_sample_ids
    reader_subset, hap_reorder = _resolve_pgen_ingest_sample_subset(
        all_sample_ids=all_sample_ids,
        selected_sample_ids=selected_sample_ids,
        sample_mask=sample_plan.sample_mask,
    )

    with PgenlibReader(
        pgen_path=pgen_path,
        pvar_path=pvar_path,
        sample_subset=reader_subset,
    ) as reader:
        n_variants = int(reader.variant_ct)
        if n_variants <= 0:
            raise ValueError(f"no variants found in {prefix}.pgen")
        if int(reader.sample_ct) <= 0:
            raise ValueError(f"no samples selected from {prefix}.psam")

    if int(workers) > 1:
        return _ingest_pgen_parallel(
            pgen_path=pgen_path,
            pvar_path=pvar_path,
            output_zarr=output_zarr,
            chrom_key=chrom_key,
            selected_sample_ids=selected_sample_ids,
            reader_subset=reader_subset,
            hap_reorder=hap_reorder,
            variant_block_size=int(variant_block_size),
            on_unphased_gt=on_unphased_gt,
            strict_biallelic=bool(strict_biallelic),
            workers=int(workers),
            chunk_variant_size=int(chunk_variant_size),
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=int(compression_level),
            store_genotypes=bool(store_genotypes),
            attrs=attrs,
            scratch_dir=scratch_dir,
            cleanup_intermediates=bool(cleanup_intermediates),
        )

    return _stream_pgen_range_to_store(
        pgen_path=pgen_path,
        pvar_path=pvar_path,
        output_zarr=output_zarr,
        chrom_key=chrom_key,
        selected_sample_ids=selected_sample_ids,
        reader_subset=reader_subset,
        hap_reorder=hap_reorder,
        start_variant=0,
        end_variant=n_variants,
        variant_block_size=int(variant_block_size),
        on_unphased_gt=on_unphased_gt,
        strict_biallelic=bool(strict_biallelic),
        chunk_variant_size=int(chunk_variant_size),
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=int(compression_level),
        store_genotypes=bool(store_genotypes),
        attrs=attrs,
    )


def _resolve_pgen_ingest_sample_subset(
    *,
    all_sample_ids: list[str],
    selected_sample_ids: list[str],
    sample_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build pgenlib's sorted subset and a HapStore-order permutation.

    pgenlib requires sample indices in increasing source order. HapStore allows
    callers to request any sample order, so ``hap_reorder`` maps the interleaved
    haplotypes returned for that sorted subset back to the requested order.
    """
    if sample_mask is None:
        return None, None

    req_indices = _subset_indices_in_requested_order(all_sample_ids, selected_sample_ids)
    sorted_indices = np.asarray(sorted(req_indices), dtype=np.uint32)
    idx_to_pos = {int(idx): i for i, idx in enumerate(sorted_indices.tolist())}
    pos_in_sorted = np.asarray([idx_to_pos[int(i)] for i in req_indices], dtype=np.int64)
    hap_reorder = np.empty(2 * pos_in_sorted.shape[0], dtype=np.int64)
    hap_reorder[0::2] = 2 * pos_in_sorted
    hap_reorder[1::2] = (2 * pos_in_sorted) + 1
    return sorted_indices, hap_reorder


def _scan_pgen_metadata(
    *,
    pgen_path: Path,
    pvar_path: Path,
    chrom_key: str,
    strict_biallelic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preflight PGEN variant metadata before parallel ingestion."""
    positions: list[int] = []
    refs: list[str] = []
    alts: list[str] = []
    vids: list[str] = []

    with PgenlibReader(pgen_path=pgen_path, pvar_path=pvar_path) as reader:
        n_variants = int(reader.variant_ct)
        if n_variants <= 0:
            raise ValueError(f"no variants found in {pgen_path}")

        for vidx in range(n_variants):
            var_chrom = normalize_chrom(reader.variant_chrom(vidx), prefix=True)
            if var_chrom != chrom_key:
                raise ValueError(
                    f"pvar chromosome mismatch at variant {vidx}: {var_chrom!r} vs requested {chrom_key!r}"
                )

            allele_ct = int(reader.allele_ct(vidx))
            ref_i = reader.allele_code(vidx, 0)
            alt_i = reader.allele_code(vidx, 1) if allele_ct >= 2 else "."
            if strict_biallelic and allele_ct > 2:
                raise ValueError(
                    f"non-biallelic variant encountered at {chrom_key}:{reader.variant_pos(vidx)} "
                    f"(allele_ct={allele_ct})"
                )

            pos_i = int(reader.variant_pos(vidx))
            ref_i, alt_i = _normalize_ref_alt_for_pvar(
                ref=ref_i,
                alt=alt_i,
                strict_biallelic=bool(strict_biallelic),
                chrom_key=chrom_key,
                pos=pos_i,
            )
            raw_vid = reader.variant_id(vidx).strip()

            positions.append(pos_i)
            refs.append(ref_i)
            alts.append(alt_i)
            vids.append(raw_vid if raw_vid not in {"", "."} else ".")

    return (
        np.asarray(positions, dtype=np.int64),
        np.asarray(refs, dtype=np.str_),
        np.asarray(alts, dtype=np.str_),
        np.asarray(vids, dtype=np.str_),
    )


def _partition_pgen_variant_shards(*, n_variants: int, workers: int) -> list[tuple[int, int, int]]:
    if int(n_variants) <= 0:
        raise ValueError(f"n_variants must be > 0, got {n_variants}")
    parts = [arr for arr in np.array_split(np.arange(int(n_variants), dtype=np.int64), int(workers)) if arr.size > 0]
    return [
        (int(shard_index), int(part[0]), int(part[-1]) + 1)
        for shard_index, part in enumerate(parts)
    ]


def _pgen_shard_store_path(scratch_dir: Path, shard_index: int) -> Path:
    return scratch_dir / f"pgen_ingest_shard_{int(shard_index):04d}.zarr"


def _ingest_pgen_variant_range_to_store(
    *,
    pgen_path: str,
    pvar_path: str,
    output_zarr: str,
    chrom_key: str,
    selected_sample_ids: list[str],
    start_variant: int,
    end_variant: int,
    reader_subset: np.ndarray | None,
    hap_reorder: np.ndarray | None,
    strict_biallelic: bool,
    on_unphased_gt: UnphasedPolicy,
    variant_block_size: int,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
    store_genotypes: bool,
) -> tuple[str, int]:
    """Worker adapter that materializes one variant range as a HapStore."""
    out = _stream_pgen_range_to_store(
        pgen_path=Path(pgen_path),
        pvar_path=Path(pvar_path),
        output_zarr=output_zarr,
        chrom_key=chrom_key,
        selected_sample_ids=selected_sample_ids,
        reader_subset=reader_subset,
        hap_reorder=hap_reorder,
        start_variant=int(start_variant),
        end_variant=int(end_variant),
        variant_block_size=int(variant_block_size),
        on_unphased_gt=on_unphased_gt,
        strict_biallelic=bool(strict_biallelic),
        chunk_variant_size=int(chunk_variant_size),
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=int(compression_level),
        store_genotypes=bool(store_genotypes),
        attrs={"source": "pgen"},
    )
    return str(Path(output_zarr)), int(
        out.attrs(chrom_key).get("extra", {}).get("unphased_gt_count", 0)
    )


def _ingest_pgen_parallel(
    *,
    pgen_path: Path,
    pvar_path: Path,
    output_zarr: str | Path,
    chrom_key: str,
    selected_sample_ids: list[str],
    reader_subset: np.ndarray | None,
    hap_reorder: np.ndarray | None,
    variant_block_size: int,
    on_unphased_gt: UnphasedPolicy,
    strict_biallelic: bool,
    workers: int,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    compression_level: int,
    store_genotypes: bool,
    attrs: dict | None,
    scratch_dir: str | Path | None,
    cleanup_intermediates: bool,
) -> HapStore:
    """Ingest contiguous PGEN ranges in parallel and concatenate the results.

    Metadata is preflighted once before workers launch. Each worker streams a
    disjoint, contiguous variant range into a temporary HapStore while using
    the same selected samples and sample order. The completed stores are then
    concatenated serially along the variant axis, and per-shard unphased-
    genotype counts are summed into the final chromosome metadata.
    """
    positions, _ref, _alt, _vid = _scan_pgen_metadata(
        pgen_path=pgen_path,
        pvar_path=pvar_path,
        chrom_key=chrom_key,
        strict_biallelic=bool(strict_biallelic),
    )
    n_variants = int(positions.shape[0])
    if n_variants <= 0:
        raise ValueError(f"no variants found in {pgen_path}")

    shard_specs = _partition_pgen_variant_shards(n_variants=n_variants, workers=int(workers))
    if len(shard_specs) <= 1:
        return ingest_pgen(
            pgen_prefix=_normalize_pgen_prefix(pgen_path),
            output_zarr=output_zarr,
            chrom=chrom_key,
            sample_ids=selected_sample_ids,
            variant_block_size=int(variant_block_size),
            on_unphased_gt=on_unphased_gt,
            strict_biallelic=bool(strict_biallelic),
            workers=1,
            chunk_variant_size=int(chunk_variant_size),
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=int(compression_level),
            store_genotypes=bool(store_genotypes),
            attrs=attrs,
            scratch_dir=scratch_dir,
            cleanup_intermediates=bool(cleanup_intermediates),
        )

    scratch_parent = Path(output_zarr).parent if scratch_dir is None else Path(scratch_dir)
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(
        tempfile.mkdtemp(prefix=f"{Path(output_zarr).stem}.pgen_ingest_", dir=str(scratch_parent))
    )
    shard_store_paths: list[Path] = []
    attrs_payload = dict(attrs or {})
    attrs_payload.setdefault("source", "pgen")
    try:
        mp_context = mp.get_context("spawn")
        futures: list[concurrent.futures.Future[tuple[str, int]]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(int(workers), len(shard_specs)),
            mp_context=mp_context,
        ) as pool:
            for shard_index, start, end in shard_specs:
                shard_store = _pgen_shard_store_path(scratch_root, int(shard_index))
                shard_store_paths.append(shard_store)
                futures.append(
                    pool.submit(
                        _ingest_pgen_variant_range_to_store,
                        pgen_path=str(pgen_path),
                        pvar_path=str(pvar_path),
                        output_zarr=str(shard_store),
                        chrom_key=chrom_key,
                        selected_sample_ids=selected_sample_ids,
                        start_variant=int(start),
                        end_variant=int(end),
                        reader_subset=reader_subset,
                        hap_reorder=hap_reorder,
                        strict_biallelic=bool(strict_biallelic),
                        on_unphased_gt=on_unphased_gt,
                        variant_block_size=int(variant_block_size),
                        chunk_variant_size=int(chunk_variant_size),
                        chunk_haplotype_size=chunk_haplotype_size,
                        chunk_sample_size=chunk_sample_size,
                        compression_level=int(compression_level),
                        store_genotypes=bool(store_genotypes),
                    )
                )

            shard_results = [fut.result() for fut in futures]

        attrs_payload["unphased_gt_count"] = int(sum(unphased for _, unphased in shard_results))
        attrs_payload["parallel_mode"] = "pgen_sharded_streaming"
        attrs_payload["pgen_variant_block_size"] = int(variant_block_size)
        store = concat_hapstores(
            input_stores=shard_store_paths,
            output_store=output_zarr,
            chrom=chrom_key,
            axis="variants",
            n_workers=1,
            chunk_variant_size=int(chunk_variant_size),
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            compression_level=int(compression_level),
            store_genotypes=bool(store_genotypes),
            attrs=attrs_payload,
            overwrite=True,
        )
        return store
    finally:
        if cleanup_intermediates:
            shutil.rmtree(scratch_root, ignore_errors=True)


def _normalize_pgen_prefix(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.suffix == ".pgen":
        return path.with_suffix("")
    if path.suffix in {".pvar", ".psam"}:
        return path.with_suffix("")
    return path


def _pgen_paths(prefix: Path) -> tuple[Path, Path, Path]:
    return Path(f"{prefix}.pgen"), Path(f"{prefix}.pvar"), Path(f"{prefix}.psam")


def _validate_pgen_triplet_exists(prefix: Path) -> None:
    pgen_path, pvar_path, psam_path = _pgen_paths(prefix)
    for path in (pgen_path, pvar_path, psam_path):
        if not path.exists():
            raise FileNotFoundError(f"missing expected PGEN artifact: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"PGEN artifact is empty: {path}")


def _validate_pgen_triplet(prefix: Path) -> None:
    """Validate PGEN/PVAR/PSAM presence, parseability, and count agreement."""
    _validate_pgen_triplet_exists(prefix)
    sample_ids = read_sample_ids(Path(f"{prefix}.psam"))
    with PgenlibReader(
        pgen_path=Path(f"{prefix}.pgen"),
        pvar_path=Path(f"{prefix}.pvar"),
    ) as reader:
        if int(reader.variant_ct) <= 0:
            raise ValueError(f"PGEN has no variants: {prefix}")
        if int(reader.sample_ct) != len(sample_ids):
            raise ValueError(
                "PGEN/PSAM sample count mismatch: "
                f"pgen={reader.sample_ct}, psam={len(sample_ids)}, prefix={prefix}"
            )


def _remove_pgen_triplet(prefix: Path) -> None:
    pgen_path, pvar_path, psam_path = _pgen_paths(prefix)
    for path in (pgen_path, pvar_path, psam_path):
        path.unlink(missing_ok=True)


def _resolve_export_variant_indices(
    *,
    store: HapStore,
    chrom_key: str,
    variant_ids: list[str] | None,
    variant_indices: list[int] | None,
    min_maf: float | None,
    filter_monomorphic: bool,
    sample_mask: np.ndarray | None,
    n_variants: int,
    scan_block_size: int,
) -> list[int] | None:
    """Resolve a store-ordered variant subset and apply sample-specific MAF.

    Explicit indices are deduplicated and sorted. Variant IDs and position-
    based identifiers are resolved against chromosome metadata. When a MAF
    threshold is active, frequencies are calculated after applying the output
    sample subset.
    """
    threshold = _resolve_export_min_maf(min_maf=min_maf, filter_monomorphic=filter_monomorphic)
    if variant_indices is not None:
        selected = sorted(set(int(i) for i in variant_indices))
        if not selected:
            raise ValueError("variant_indices resolved to an empty variant set")
        if selected[0] < 0 or selected[-1] >= int(n_variants):
            raise ValueError(
                "variant_indices out of bounds for chromosome "
                f"{chrom_key}: min={selected[0]}, max={selected[-1]}, n_variants={n_variants}"
            )
        return _apply_export_min_maf_filter(
            store=store,
            chrom_key=chrom_key,
            selected=selected,
            sample_mask=sample_mask,
            scan_block_size=int(scan_block_size),
            threshold=threshold,
        )

    if variant_ids is None:
        return _apply_export_min_maf_filter(
            store=store,
            chrom_key=chrom_key,
            selected=None,
            sample_mask=sample_mask,
            scan_block_size=int(scan_block_size),
            threshold=threshold,
        )

    chrom_group = store.root[chrom_key]
    variants = chrom_group["variants"]
    selected = _resolve_variant_indices_streaming(
        chrom_key=chrom_key,
        pos_ds=variants["pos"],
        vid_ds=variants["vid"],
        variant_ids=variant_ids,
        scan_block_size=int(scan_block_size),
    )
    if not selected:
        raise ValueError("variant_ids resolved to an empty variant set")
    return _apply_export_min_maf_filter(
        store=store,
        chrom_key=chrom_key,
        selected=selected,
        sample_mask=sample_mask,
        scan_block_size=int(scan_block_size),
        threshold=threshold,
    )


def _resolve_export_min_maf(*, min_maf: float | None, filter_monomorphic: bool) -> float | None:
    if min_maf is None:
        return 1e-12 if bool(filter_monomorphic) else None
    value = float(min_maf)
    if value < 0 or value > 0.5:
        raise ValueError(f"min_maf must be between 0 and 0.5, got {min_maf}")
    return value


def _apply_export_min_maf_filter(
    *,
    store: HapStore,
    chrom_key: str,
    selected: list[int] | None,
    sample_mask: np.ndarray | None,
    scan_block_size: int,
    threshold: float | None,
) -> list[int] | None:
    if threshold is None:
        return selected
    kept = store.non_monomorphic_variant_indices(
        chrom_key,
        sample_mask=sample_mask,
        variant_indices=selected,
        block_size=int(scan_block_size),
        min_maf=float(threshold),
    )
    out = [int(i) for i in np.asarray(kept, dtype=np.int64).tolist()]
    if not out:
        raise ValueError("selected variant set is empty after applying MAF filters")
    return out


def _merge_pgen_shards_with_plink(
    *,
    shard_prefixes: list[Path],
    out_prefix: Path,
    plink2_bin: str,
    threads: int,
) -> None:
    """Merge ordered variant-shard triplets while preserving sample order."""
    if not shard_prefixes:
        raise ValueError("shard_prefixes must be non-empty")
    if int(threads) <= 0:
        raise ValueError(f"threads must be positive, got {threads}")

    with tempfile.NamedTemporaryFile("w", suffix=".pmerge.list", delete=False, encoding="utf-8") as handle:
        merge_list = Path(handle.name)
        for shard_prefix in shard_prefixes:
            _validate_pgen_triplet_exists(shard_prefix)
            handle.write(str(shard_prefix) + "\n")

    try:
        _remove_pgen_triplet(out_prefix)
        cmd = [
            str(plink2_bin),
            "--threads",
            str(int(threads)),
            "--pmerge-list",
            str(merge_list),
            "pfile",
            "--indiv-sort",
            "none",
            "--make-pgen",
            "--out",
            str(out_prefix),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "PLINK2 --pmerge-list failed\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )
    finally:
        merge_list.unlink(missing_ok=True)


def _validate_haplotypes_biallelic_block(
    *,
    hap_block: np.ndarray,
    chrom_key: str,
    start: int,
    strict_biallelic: bool,
) -> None:
    if hap_block.ndim != 2:
        raise ValueError(f"hap_block must be 2D, got shape={hap_block.shape}")

    if not strict_biallelic:
        return

    ok = (hap_block == 0) | (hap_block == 1) | (hap_block == _MISSING_SENTINEL)
    if bool(np.all(ok)):
        return

    bad = np.argwhere(~ok)[0]
    hap_i = int(bad[0])
    rel_vidx = int(bad[1])
    got = int(hap_block[hap_i, rel_vidx])
    raise ValueError(
        "non-biallelic haplotype allele encountered while exporting PGEN "
        f"(chrom={chrom_key}, variant_index={start + rel_vidx}, haplotype_index={hap_i}, value={got})"
    )


def _handle_partial_missing_pairs(
    *,
    allele_batch: np.ndarray,
    policy: PartialMissingPolicy,
    chrom_key: str,
    positions: np.ndarray,
) -> None:
    """Reject one-missing-allele genotypes or make both alleles missing."""
    a0 = allele_batch[:, 0::2]
    a1 = allele_batch[:, 1::2]
    partial = (a0 == -9) ^ (a1 == -9)
    if not bool(np.any(partial)):
        return

    if policy == "error":
        first = np.argwhere(partial)[0]
        rel_vidx = int(first[0])
        sample_i = int(first[1])
        pos = int(positions[rel_vidx])
        raise ValueError(
            "partial missing diploid genotype encountered while exporting PGEN "
            f"(variant={chrom_key}:{pos}, sample_index={sample_i})"
        )

    pair_missing = (a0 == -9) | (a1 == -9)
    a0[pair_missing] = -9
    a1[pair_missing] = -9


def _normalize_ref_alt_for_pvar(
    *,
    ref: str,
    alt: str,
    strict_biallelic: bool,
    chrom_key: str,
    pos: int,
) -> tuple[str, str]:
    ref_clean = str(ref).strip()
    alt_clean = str(alt).strip()

    if strict_biallelic:
        if ref_clean in {"", "."} or alt_clean in {"", "."}:
            raise ValueError(
                f"missing REF/ALT for {chrom_key}:{pos}; REF={ref!r}, ALT={alt!r}"
            )
        if "," in alt_clean:
            raise ValueError(
                f"non-biallelic ALT for {chrom_key}:{pos}; ALT={alt_clean!r}"
            )

    if ref_clean in {"", "."}:
        ref_clean = "N"
    if alt_clean in {"", "."}:
        alt_clean = "<NON_REF>"

    return ref_clean, alt_clean
