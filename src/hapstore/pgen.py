"""Direct HapStore <-> PGEN I/O helpers (biallelic, phased)."""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Literal

import numpy as np

from hapstore.chrom import normalize_chrom
from hapstore.ids import read_sample_ids

from ._pgen_fast import PgenFastReader, PgenFastWriter, ensure_pgenlib_available
from ._internal.export_selection import (
    _iter_variant_slices,
    _iter_variant_slices_for_indices,
    _plan_sample_subset,
    _resolve_variant_indices_streaming,
    _subset_indices_in_requested_order,
)
from ._internal.plink import resolve_plink2_binary
from ._internal.zarr_utils import _to_unicode_array
from .ingest import ingest_arrays
from .store import HapStore

_MISSING_SENTINEL = 255

PartialMissingPolicy = Literal["error", "missing"]
UnphasedPolicy = Literal["error", "missing"]


def export_pgen(
    store_path: str | Path,
    out_prefix: str | Path,
    chrom: str | int,
    sample_ids: list[str] | None = None,
    variant_ids: list[str] | None = None,
    variant_indices: list[int] | None = None,
    variant_block_size: int = 8192,
    strict_biallelic: bool = True,
    on_partial_missing: PartialMissingPolicy = "error",
    n_workers: int = 1,
    plink2_workers: int = 1,
    plink2_path: str | Path = "plink2",
    keep_intermediate_shards: bool = False,
) -> Path:
    """Export one chromosome from HapStore directly to PGEN triplet.

    Current scope is diploid, phased, biallelic hardcalls only.
    """
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

    prefix = _normalize_pgen_prefix(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _remove_pgen_triplet(prefix)

    selected_indices = _resolve_export_variant_indices(
        store=store,
        chrom_key=chrom_key,
        variant_ids=variant_ids,
        variant_indices=variant_indices,
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
            variant_block_size=int(variant_block_size),
            strict_biallelic=bool(strict_biallelic),
            on_partial_missing=on_partial_missing,
        )

    plink2_bin = str(resolve_plink2_binary(plink2_path))
    shard_prefixes: list[Path] = []
    temp_root = Path(tempfile.mkdtemp(prefix=f"{prefix.name}.shards_", dir=str(prefix.parent)))

    try:
        for shard_idx, idx_array in enumerate(shard_idx_arrays):
            shard_prefix = temp_root / f"{prefix.name}.shard{(shard_idx + 1):03d}of{len(shard_idx_arrays):03d}"
            shard_prefixes.append(shard_prefix)

        mp_context = mp.get_context("spawn")
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
    variant_block_size: int = 8192,
    strict_biallelic: bool = True,
    on_partial_missing: PartialMissingPolicy = "error",
) -> Path:
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

    selected_indices = _resolve_export_variant_indices(
        store=store,
        chrom_key=chrom_key,
        variant_ids=variant_ids,
        variant_indices=variant_indices,
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

    written = 0
    with pvar_path.open("w", encoding="utf-8") as pvar_handle:
        pvar_handle.write("#CHROM\tPOS\tID\tREF\tALT\n")
        with PgenFastWriter(
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

                pos_block = np.asarray(pos_ds[start:stop], dtype=np.int64)
                ref_block = _to_unicode_array(np.asarray(ref_ds[start:stop], dtype=object)).tolist()
                alt_block = _to_unicode_array(np.asarray(alt_ds[start:stop], dtype=object)).tolist()
                vid_block = _to_unicode_array(np.asarray(vid_ds[start:stop], dtype=object)).tolist()

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
    chunk_variant_size: int = 500,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    compression_level: int = 3,
    store_genotypes: bool = False,
    attrs: dict | None = None,
) -> HapStore:
    """Ingest one chromosome from PGEN triplet into HapStore.

    Current scope is diploid, phased, biallelic hardcalls only.
    """
    ensure_pgenlib_available()
    if int(variant_block_size) <= 0:
        raise ValueError(f"variant_block_size must be positive, got {variant_block_size}")
    if on_unphased_gt not in {"error", "missing"}:
        raise ValueError(f"on_unphased_gt must be 'error' or 'missing', got {on_unphased_gt!r}")

    prefix = _normalize_pgen_prefix(pgen_prefix)
    pgen_path, pvar_path, psam_path = _pgen_paths(prefix)
    _validate_pgen_triplet_exists(prefix)

    chrom_key = normalize_chrom(chrom, prefix=True)
    all_sample_ids = read_sample_ids(psam_path)
    sample_plan = _plan_sample_subset(all_sample_ids, sample_ids)

    reader_subset: np.ndarray | None = None
    hap_reorder: np.ndarray | None = None
    selected_sample_ids = sample_plan.selected_sample_ids
    if sample_plan.sample_mask is not None:
        req_indices = _subset_indices_in_requested_order(all_sample_ids, selected_sample_ids)
        sorted_indices = np.asarray(sorted(req_indices), dtype=np.uint32)
        reader_subset = sorted_indices

        idx_to_pos = {int(idx): i for i, idx in enumerate(sorted_indices.tolist())}
        pos_in_sorted = np.asarray([idx_to_pos[int(i)] for i in req_indices], dtype=np.int64)
        hap_reorder = np.empty(2 * pos_in_sorted.shape[0], dtype=np.int64)
        hap_reorder[0::2] = 2 * pos_in_sorted
        hap_reorder[1::2] = (2 * pos_in_sorted) + 1

    positions: list[int] = []
    refs: list[str] = []
    alts: list[str] = []
    vids: list[str] = []

    blocks: list[np.ndarray] = []
    unphased_gt_count = 0

    with PgenFastReader(
        pgen_path=pgen_path,
        pvar_path=pvar_path,
        sample_subset=reader_subset,
    ) as reader:
        n_variants = int(reader.variant_ct)
        n_samples_reader = int(reader.sample_ct)

        if n_variants <= 0:
            raise ValueError(f"no variants found in {prefix}.pgen")
        if n_samples_reader <= 0:
            raise ValueError(f"no samples selected from {prefix}.psam")

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

            ref_i, alt_i = _normalize_ref_alt_for_pvar(
                ref=ref_i,
                alt=alt_i,
                strict_biallelic=bool(strict_biallelic),
                chrom_key=chrom_key,
                pos=int(reader.variant_pos(vidx)),
            )

            positions.append(int(reader.variant_pos(vidx)))
            refs.append(ref_i)
            alts.append(alt_i)
            raw_vid = reader.variant_id(vidx).strip()
            vids.append(raw_vid if raw_vid not in {"", "."} else ".")

        for start in range(0, n_variants, int(variant_block_size)):
            end = min(n_variants, start + int(variant_block_size))
            block_n = int(end - start)
            if block_n <= 0:
                continue

            allele_block = np.empty((block_n, 2 * n_samples_reader), dtype=np.int32, order="C")
            phase_block = np.empty((block_n, n_samples_reader), dtype=np.uint8, order="C")
            reader.read_alleles_and_phasepresent_range(start, end, allele_block, phase_block)

            valid = (allele_block >= 0)
            if strict_biallelic and np.any(allele_block[valid] > 1):
                bad = np.argwhere((allele_block >= 0) & (allele_block > 1))[0]
                rel_vidx = int(bad[0])
                pos = positions[start + rel_vidx]
                raise ValueError(
                    f"non-biallelic allele code encountered while ingesting {chrom_key}:{pos}"
                )

            a0 = allele_block[:, 0::2]
            a1 = allele_block[:, 1::2]
            missing = (a0 < 0) | (a1 < 0)
            het = (a0 != a1) & (~missing)
            unphased_het = (phase_block == 0) & het

            if bool(np.any(unphased_het)):
                unphased_gt_count += int(np.sum(unphased_het))
                if on_unphased_gt == "error":
                    first = np.argwhere(unphased_het)[0]
                    rel_vidx = int(first[0])
                    sample_i = int(first[1])
                    pos = positions[start + rel_vidx]
                    raise ValueError(
                        "unphased heterozygous genotype encountered while ingesting PGEN "
                        f"(variant={chrom_key}:{pos}, sample_index={sample_i})"
                    )
                a0[unphased_het] = -9
                a1[unphased_het] = -9

            pair_missing = (a0 < 0) | (a1 < 0)
            a0[pair_missing] = -9
            a1[pair_missing] = -9

            haps_block = allele_block.T
            haps_block = np.where(haps_block < 0, _MISSING_SENTINEL, haps_block).astype(np.uint8, copy=False)
            if hap_reorder is not None:
                haps_block = haps_block[hap_reorder, :]
            blocks.append(np.ascontiguousarray(haps_block))

    haplotypes = np.concatenate(blocks, axis=1)

    attrs_payload = dict(attrs or {})
    attrs_payload.setdefault("source", "pgen")
    attrs_payload["unphased_gt_count"] = int(unphased_gt_count)

    return ingest_arrays(
        output_zarr=output_zarr,
        chrom=chrom_key,
        haplotypes=haplotypes,
        variant_pos=np.asarray(positions, dtype=np.int64),
        sample_ids=selected_sample_ids,
        ref=refs,
        alt=alts,
        vid=vids,
        store_genotypes=bool(store_genotypes),
        chunk_variant_size=int(chunk_variant_size),
        chunk_haplotype_size=chunk_haplotype_size,
        chunk_sample_size=chunk_sample_size,
        compression_level=int(compression_level),
        attrs=attrs_payload,
    )


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
    n_variants: int,
    scan_block_size: int,
) -> list[int] | None:
    if variant_indices is not None:
        selected = sorted(set(int(i) for i in variant_indices))
        if not selected:
            raise ValueError("variant_indices resolved to an empty variant set")
        if selected[0] < 0 or selected[-1] >= int(n_variants):
            raise ValueError(
                "variant_indices out of bounds for chromosome "
                f"{chrom_key}: min={selected[0]}, max={selected[-1]}, n_variants={n_variants}"
            )
        return selected

    if variant_ids is None:
        return None

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
    return selected


def _merge_pgen_shards_with_plink(
    *,
    shard_prefixes: list[Path],
    out_prefix: Path,
    plink2_bin: str,
    threads: int,
) -> None:
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
