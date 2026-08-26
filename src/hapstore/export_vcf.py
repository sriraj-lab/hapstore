"""Export one HapStore chromosome to phased VCF."""

from __future__ import annotations

import functools
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pysam
import zarr

from hapstore.chrom import normalize_chrom

from ._internal.selection import (
    _iter_variant_slices,
    _iter_variant_slices_for_indices,
    _plan_sample_subset,
    _resolve_variant_indices_streaming,
)
from ._internal.plink import resolve_plink2_binary
from ._internal.zarr_utils import _to_unicode_array
from .store import HapStore, _MISSING_SENTINEL


def _cleanup_failed_vcf_export(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        output_vcf = kwargs.get("output_vcf", args[1] if len(args) > 1 else None)
        if output_vcf is None:
            raise TypeError("export_vcf requires output_vcf")
        output = Path(output_vcf)
        plain = output.with_suffix("") if output.suffix == ".gz" else output
        compressed = output if output.suffix == ".gz" else Path(f"{output}.gz")
        try:
            result = func(*args, **kwargs)
            if not result.exists() or result.stat().st_size <= 0:
                raise ValueError(f"VCF export did not produce a non-empty output: {result}")
            return result
        except Exception:
            for path in (plain, compressed, Path(f"{compressed}.tbi"), Path(f"{compressed}.csi")):
                path.unlink(missing_ok=True)
            raise

    return wrapped


@_cleanup_failed_vcf_export
def export_vcf(
    store_path: str | Path,
    output_vcf: str | Path,
    chrom: str | int,
    sample_ids: list[str] | None = None,
    variant_ids: list[str] | None = None,
    bgzip: bool = True,
    tabix_index: bool = True,
    strict_variant_info: bool = True,
    variant_block_size: int = 1024,
    plink2_location: str | Path | None = None,
    keep_intermediate_files: bool = False,
    n_shards: int | None = None,
    pgen_workers: int | None = None,
    plink2_workers: int = 1,
    shard_tmp_dir: str | Path | None = None,
) -> Path:
    """Export one chromosome from HapStore to phased VCF.

    Args:
        store_path: Path to the HapStore Zarr hierarchy.
        output_vcf: Output VCF path. ``.vcf.gz`` is produced when ``bgzip`` is
            true.
        chrom: Chromosome to export, normalized internally to ``chrN``.
        sample_ids: Optional sample subset in the requested output order.
        variant_ids: Optional variant-ID or position-style subset. Selected
            variants are written in HapStore order.
        bgzip: Compress the native-writer output with bgzip.
        tabix_index: Write a tabix index for a bgzipped result.
        strict_variant_info: Reject missing REF/ALT metadata. The native writer
            substitutes ``N`` and ``<NON_REF>`` when false. The PGEN bridge
            requires true because HapStore's current PGEN writer requires
            valid PVAR alleles.
        variant_block_size: Variants read per native-writer or PGEN block.
        plink2_location: Optional PLINK2 executable. When provided, export
            first materializes PGEN and then invokes PLINK2's VCF writer.
        keep_intermediate_files: Retain PGEN output and, for parallel export,
            its shard directory beside ``output_vcf``.
        n_shards: Deprecated compatibility alias for ``pgen_workers``.
        pgen_workers: Number of concurrent PGEN variant-shard writers in the
            PLINK2 bridge. Defaults to one.
        plink2_workers: PLINK2 threads used to merge PGEN shards and write VCF.
        shard_tmp_dir: Optional directory for temporary PGEN shard triplets.

    Returns:
        Path to the completed VCF or bgzipped VCF.

    Raises:
        ValueError: If selection or worker arguments are invalid, or if the
            PGEN bridge is requested with non-strict variant metadata.
        FileNotFoundError: If PLINK2 does not produce the expected VCF.

    Notes:
        Native export streams phased haplotype blocks through ``pysam``. The
        PGEN bridge delegates parallel materialization to ``export_pgen``;
        only that stage is parallelized, while final VCF creation is handled by
        PLINK2.
    """
    if int(variant_block_size) <= 0:
        raise ValueError(f"variant_block_size must be positive, got {variant_block_size}")
    if n_shards is not None and int(n_shards) <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")
    if pgen_workers is not None and int(pgen_workers) <= 0:
        raise ValueError(f"pgen_workers must be positive, got {pgen_workers}")
    if int(plink2_workers) <= 0:
        raise ValueError(f"plink2_workers must be positive, got {plink2_workers}")

    if plink2_location is not None:
        if not strict_variant_info:
            raise ValueError(
                "strict_variant_info=False is supported only by the native VCF writer; "
                "the PGEN/PLINK2 bridge requires valid REF/ALT metadata"
            )
        resolved_pgen_workers = _resolve_pgen_workers(
            n_shards=n_shards,
            pgen_workers=pgen_workers,
        )
        return _export_vcf_via_pgen(
            store_path=store_path,
            output_vcf=output_vcf,
            chrom=chrom,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
            bgzip=bgzip,
            tabix_index=tabix_index,
            variant_block_size=int(variant_block_size),
            plink2_location=plink2_location,
            keep_intermediate_files=bool(keep_intermediate_files),
            pgen_workers=resolved_pgen_workers,
            plink2_workers=int(plink2_workers),
            shard_tmp_dir=shard_tmp_dir,
        )

    store = HapStore(store_path, mode="r")
    chrom_key = normalize_chrom(chrom, prefix=True)

    all_sample_ids = store.sample_ids(chrom_key)
    sample_plan = _plan_sample_subset(all_sample_ids, sample_ids)

    chrom_group = store.root[chrom_key]
    if not isinstance(chrom_group, zarr.hierarchy.Group):
        raise KeyError(f"node for chromosome {chrom_key!r} is not a group")
    variants = chrom_group["variants"]
    pos_ds = variants["pos"]
    ref_ds = variants["ref"]
    alt_ds = variants["alt"]
    vid_ds = variants["vid"]
    n_variants = int(pos_ds.shape[0])

    if variant_ids is None:
        slices = _iter_variant_slices(n_variants, int(variant_block_size))
    else:
        selected_indices = _resolve_variant_indices_streaming(
            chrom_key=chrom_key,
            pos_ds=pos_ds,
            vid_ds=vid_ds,
            variant_ids=variant_ids,
            scan_block_size=int(variant_block_size),
        )
        slices = _iter_variant_slices_for_indices(selected_indices, int(variant_block_size))

    out_path = Path(output_vcf)
    if bgzip:
        plain_path = out_path.with_suffix("") if out_path.suffix == ".gz" else out_path
    else:
        plain_path = out_path

    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.contigs.add(chrom_key)
    header.formats.add("GT", number=1, type="String", description="Genotype")
    for sid in sample_plan.selected_sample_ids:
        header.add_sample(str(sid))

    n_samples = len(sample_plan.selected_sample_ids)
    expected_haplotypes = 2 * n_samples

    with pysam.VariantFile(str(plain_path), "w", header=header) as out:
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
            if hap_block.shape[0] != expected_haplotypes:
                raise ValueError(
                    "haplotype/sample mismatch while exporting: "
                    f"haplotypes.shape[0]={hap_block.shape[0]}, expected={expected_haplotypes}"
                )
            if sample_plan.hap_perm is not None:
                hap_block = hap_block[sample_plan.hap_perm, :]

            positions = np.asarray(pos_ds[start:stop], dtype=np.int64)
            ref = _to_unicode_list(np.asarray(ref_ds[start:stop], dtype=object))
            alt = _to_unicode_list(np.asarray(alt_ds[start:stop], dtype=object))
            vid = _to_unicode_list(np.asarray(vid_ds[start:stop], dtype=object))

            if positions.shape[0] != hap_block.shape[1]:
                raise ValueError(
                    "haplotype/variant mismatch while exporting: "
                    f"haplotypes.shape={hap_block.shape}, positions.shape={positions.shape}"
                )
            if len(ref) != positions.shape[0] or len(alt) != positions.shape[0] or len(vid) != positions.shape[0]:
                raise ValueError(
                    "variant metadata length mismatch while exporting: "
                    f"len(ref)={len(ref)}, len(alt)={len(alt)}, len(vid)={len(vid)}, n={positions.shape[0]}"
                )

            _write_vcf_records_to_open_file(
                out=out,
                chrom=chrom_key,
                sample_ids=sample_plan.selected_sample_ids,
                haplotypes=hap_block,
                positions=positions,
                ref=ref,
                alt=alt,
                vid=vid,
                strict_variant_info=strict_variant_info,
            )

    if not bgzip:
        return plain_path

    compressed = out_path if out_path.suffix == ".gz" else Path(f"{out_path}.gz")
    pysam.tabix_compress(str(plain_path), str(compressed), force=True)
    if tabix_index:
        pysam.tabix_index(str(compressed), preset="vcf", force=True)
    plain_path.unlink(missing_ok=True)
    return compressed


def _export_vcf_via_pgen(
    *,
    store_path: str | Path,
    output_vcf: str | Path,
    chrom: str | int,
    sample_ids: list[str] | None,
    variant_ids: list[str] | None,
    bgzip: bool,
    tabix_index: bool,
    variant_block_size: int,
    plink2_location: str | Path,
    keep_intermediate_files: bool,
    pgen_workers: int,
    plink2_workers: int,
    shard_tmp_dir: str | Path | None,
) -> Path:
    """Export VCF through the PGEN/PLINK2 bridge.

    Called only from :func:`export_vcf` when ``plink2_location`` is provided.
    """
    from hapstore._internal.command import run_cmd

    from .pgen import export_pgen

    plink2_bin = resolve_plink2_binary(plink2_location)
    chrom_key = normalize_chrom(chrom, prefix=True)

    out_path = Path(output_vcf)
    final_vcf = out_path if (not bgzip or out_path.suffix == ".gz") else Path(f"{out_path}.gz")
    export_prefix = _vcf_export_prefix(final_vcf)
    export_prefix.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = None
    if not keep_intermediate_files:
        tmp_root = Path(tempfile.mkdtemp(prefix="hapstore_export_vcf_", dir=str(export_prefix.parent)))
        work_dir = tmp_root
    else:
        work_dir = export_prefix.parent
    pgen_prefix = work_dir / f"{export_prefix.name}.merged_pgen"
    resolved_shard_tmp_dir = Path(shard_tmp_dir) if shard_tmp_dir is not None else None
    if keep_intermediate_files and resolved_shard_tmp_dir is None:
        resolved_shard_tmp_dir = work_dir

    try:
        pgen_prefix = export_pgen(
            store_path=store_path,
            out_prefix=pgen_prefix,
            chrom=chrom_key,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
            variant_block_size=int(variant_block_size),
            strict_biallelic=True,
            on_partial_missing="error",
            n_workers=int(pgen_workers),
            plink2_workers=int(plink2_workers),
            plink2_path=plink2_bin,
            shard_tmp_dir=resolved_shard_tmp_dir,
            keep_intermediate_shards=bool(keep_intermediate_files),
        )

        export_out = work_dir / export_prefix.name
        cmd = [
            str(plink2_bin),
            "--threads",
            str(int(plink2_workers)),
            "--pfile",
            str(pgen_prefix),
            "--export",
            "vcf",
            "bgz" if bgzip else "",
            "--out",
            str(export_out),
        ]
        cmd = [c for c in cmd if c != ""]
        run_cmd(cmd, check=True, capture_output=True)

        produced = Path(f"{export_out}.vcf.gz" if bgzip else f"{export_out}.vcf")
        if not produced.exists():
            raise FileNotFoundError(f"plink2 did not produce expected VCF: {produced}")

        final_vcf.parent.mkdir(parents=True, exist_ok=True)
        produced.replace(final_vcf)

        if bgzip and tabix_index:
            pysam.tabix_index(str(final_vcf), preset="vcf", force=True)

        return final_vcf
    finally:
        if tmp_root is not None and tmp_root.exists():
            shutil.rmtree(tmp_root)


def _resolve_pgen_workers(*, n_shards: int | None, pgen_workers: int | None) -> int:
    """Resolve the current bridge worker setting and its deprecated alias."""
    if n_shards is not None and pgen_workers is not None:
        raise ValueError("n_shards and pgen_workers are mutually exclusive")

    if pgen_workers is not None:
        workers = int(pgen_workers)
    elif n_shards is not None:
        workers = int(n_shards)
        warnings.warn(
            "n_shards is deprecated; use pgen_workers for parallel PGEN materialization",
            DeprecationWarning,
            stacklevel=3,
        )
    else:
        workers = 1

    if workers <= 0:
        raise ValueError(f"pgen_workers must be positive, got {workers}")
    return workers


def _vcf_export_prefix(output_vcf: Path) -> Path:
    # Normalize ``x.vcf[.gz]`` to the PLINK2 output prefix.
    base = output_vcf
    if base.suffix == ".gz":
        base = base.with_suffix("")
    if base.suffix == ".vcf":
        base = base.with_suffix("")
    return base


def _write_vcf_records_to_open_file(
    out: pysam.VariantFile,
    chrom: str,
    sample_ids: list[str],
    haplotypes: np.ndarray,
    positions: np.ndarray,
    ref: list[str],
    alt: list[str],
    vid: list[str],
    strict_variant_info: bool,
) -> None:
    """Write phased GT-only VCF records to an open :class:`pysam.VariantFile`."""
    n_samples = len(sample_ids)
    if haplotypes.shape[0] != 2 * n_samples:
        raise ValueError(
            "haplotype/sample mismatch while writing VCF: "
            f"haplotypes.shape[0]={haplotypes.shape[0]}, 2*len(sample_ids)={2 * n_samples}"
        )

    for j, pos in enumerate(positions):
        ref_j, alt_j = _validate_variant_alleles(
            chrom=chrom,
            pos=int(pos),
            ref=ref[j],
            alt=alt[j],
            strict_variant_info=strict_variant_info,
        )
        vid_j = vid[j]
        rid = None if (vid_j == "" or vid_j == ".") else vid_j

        record = out.new_record(
            contig=chrom,
            start=int(pos) - 1,
            stop=int(pos),
            id=rid,
            alleles=(ref_j, alt_j),
        )

        for i, sid in enumerate(sample_ids):
            a0 = int(haplotypes[2 * i, j])
            a1 = int(haplotypes[2 * i + 1, j])
            if a0 == _MISSING_SENTINEL or a1 == _MISSING_SENTINEL:
                record.samples[sid]["GT"] = (None, None)
                record.samples[sid].phased = True
            else:
                record.samples[sid]["GT"] = (a0, a1)
                record.samples[sid].phased = True

        out.write(record)


def _validate_variant_alleles(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    strict_variant_info: bool,
) -> tuple[str, str]:
    ref_clean = ref.strip() if ref is not None else ""
    alt_clean = alt.strip() if alt is not None else ""

    missing_ref = (ref_clean == "") or (ref_clean == ".")
    missing_alt = (alt_clean == "") or (alt_clean == ".")

    if strict_variant_info and (missing_ref or missing_alt):
        raise ValueError(
            f"missing REF/ALT for export at {chrom}:{pos}; REF={ref!r}, ALT={alt!r}. "
            "Set strict_variant_info=False to write fallbacks."
        )

    if missing_ref:
        ref_clean = "N"
    if missing_alt:
        alt_clean = "<NON_REF>"

    return ref_clean, alt_clean

def _to_unicode_list(values: np.ndarray) -> list[str]:
    return [str(x) for x in _to_unicode_array(np.asarray(values, dtype=object)).tolist()]
