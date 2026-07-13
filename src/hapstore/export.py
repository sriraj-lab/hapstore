"""Export hapstore content to external formats."""

from __future__ import annotations

import functools
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pysam
import zarr
from numcodecs import VLenUTF8, Zstd

from hapstore.chrom import normalize_chrom

from ._internal.chrom_attrs import build_chrom_attrs
from ._internal.chunking import (
    DEFAULT_VARIANT_CHUNK_SIZE,
    resolve_matrix_chunk_layout,
)
from ._internal.export_selection import (
    _iter_variant_slices,
    _iter_variant_slices_for_indices,
    _plan_sample_subset,
    _resolve_variant_indices_streaming,
    _subset_indices_in_requested_order,
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
    n_shards: int = 1,
) -> Path:
    """Export one chromosome from hapstore to phased VCF."""
    if int(variant_block_size) <= 0:
        raise ValueError(f"variant_block_size must be positive, got {variant_block_size}")
    if int(n_shards) <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")

    if plink2_location is not None:
        return _export_vcf_via_pgen(
            store_path=store_path,
            output_vcf=output_vcf,
            chrom=chrom,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
            bgzip=bgzip,
            tabix_index=tabix_index,
            strict_variant_info=strict_variant_info,
            variant_block_size=int(variant_block_size),
            plink2_location=plink2_location,
            keep_intermediate_files=bool(keep_intermediate_files),
            n_shards=int(n_shards),
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
    strict_variant_info: bool,
    variant_block_size: int,
    plink2_location: str | Path,
    keep_intermediate_files: bool,
    n_shards: int,
) -> Path:
    from hapstore._internal.command import run_cmd

    from .pgen import export_pgen

    plink2_bin = resolve_plink2_binary(plink2_location)
    chrom_key = normalize_chrom(chrom, prefix=True)

    out_path = Path(output_vcf)
    final_vcf = out_path if (not bgzip or out_path.suffix == ".gz") else Path(f"{out_path}.gz")
    export_prefix = _vcf_export_prefix(final_vcf)
    export_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Resolve variant indices once when variant_ids is specified and we need sharding.
    variant_indices: list[int] | None = None
    if variant_ids is not None and int(n_shards) > 1:
        store = HapStore(store_path, mode="r")
        chrom_group = store.root[chrom_key]
        variants = chrom_group["variants"]
        variant_indices = _resolve_variant_indices_streaming(
            chrom_key=chrom_key,
            pos_ds=variants["pos"],
            vid_ds=variants["vid"],
            variant_ids=variant_ids,
            scan_block_size=int(variant_block_size),
        )
        if not variant_indices:
            raise ValueError("variant_ids resolved to an empty variant set")

    shard_prefixes: list[Path] = []

    tmp_root = None
    if not keep_intermediate_files:
        tmp_root = Path(tempfile.mkdtemp(prefix="hapstore_export_vcf_", dir=str(export_prefix.parent)))
        work_dir = tmp_root
    else:
        work_dir = export_prefix.parent
    merged_prefix = work_dir / f"{export_prefix.name}.merged_pgen"

    try:
        if int(n_shards) == 1:
            merged_prefix = export_pgen(
                store_path=store_path,
                out_prefix=merged_prefix,
                chrom=chrom_key,
                sample_ids=sample_ids,
                variant_ids=variant_ids,
                variant_block_size=int(variant_block_size),
                strict_biallelic=True,
                on_partial_missing="error",
            )
        else:
            # Build contiguous or index-based shard definitions.
            if variant_indices is None:
                store = HapStore(store_path, mode="r")
                n_variants = int(store.n_variants(chrom_key))
                cuts = np.linspace(0, n_variants, int(n_shards) + 1, dtype=int).tolist()
                shard_indices_list = [
                    list(range(int(cuts[i]), int(cuts[i + 1]))) for i in range(int(n_shards))
                ]
            else:
                idx_sorted = sorted(set(int(i) for i in variant_indices))
                parts = np.array_split(np.asarray(idx_sorted, dtype=np.int64), int(n_shards))
                shard_indices_list = [p.astype(int).tolist() for p in parts if p.size > 0]

            for shard_idx, shard_indices in enumerate(shard_indices_list):
                shard_prefix = work_dir / f"{export_prefix.name}.shard{shard_idx}"
                shard_prefix = export_pgen(
                    store_path=store_path,
                    out_prefix=shard_prefix,
                    chrom=chrom_key,
                    sample_ids=sample_ids,
                    variant_indices=shard_indices,
                    variant_block_size=int(variant_block_size),
                    strict_biallelic=True,
                    on_partial_missing="error",
                )
                shard_prefixes.append(Path(shard_prefix))

            # Merge shards into one PGEN triplet.
            pmerge_list = work_dir / f"{export_prefix.name}.pmerge_list.txt"
            with pmerge_list.open("w", encoding="utf-8") as handle:
                for prefix in shard_prefixes:
                    handle.write(str(prefix) + "\n")
            run_cmd(
                [
                    str(plink2_bin),
                    "--pmerge-list",
                    str(pmerge_list),
                    "--make-pgen",
                    "--out",
                    str(merged_prefix),
                ],
                check=True,
                capture_output=True,
            )

        # Export VCF via plink2.
        export_out = work_dir / export_prefix.name
        cmd = [
            str(plink2_bin),
            "--pfile",
            str(merged_prefix),
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


def _vcf_export_prefix(output_vcf: Path) -> Path:
    # Drop .gz if present, then drop .vcf suffix.
    base = output_vcf
    if base.suffix == ".gz":
        base = base.with_suffix("")
    if base.suffix == ".vcf":
        base = base.with_suffix("")
    return base


def subset_samples_to_new_store(
    source_store: str | Path,
    chrom: str | int,
    sample_ids: list[str],
    output_store: str | Path,
    *,
    variant_block_size: int = 1000,
    variant_chunk: int = DEFAULT_VARIANT_CHUNK_SIZE,
    hap_chunk: int | None = None,
    compression_level: int = 3,
) -> Path:
    """Create a new HapStore containing one chromosome with a subset of samples.

    Streams haplotypes by variant blocks using HapStore.haplotypes(..., variant_slice=..., sample_mask=...).
    """
    if int(variant_block_size) <= 0:
        raise ValueError(f"variant_block_size must be positive, got {variant_block_size}")
    if int(variant_chunk) <= 0:
        raise ValueError(f"variant_chunk must be positive, got {variant_chunk}")
    if hap_chunk is not None and int(hap_chunk) <= 0:
        raise ValueError(f"hap_chunk must be positive, got {hap_chunk}")
    if int(compression_level) < 0:
        raise ValueError(f"compression_level must be >= 0, got {compression_level}")

    out_path = Path(output_store)
    if out_path.exists():
        raise FileExistsError(f"output_store already exists: {out_path}")

    src = HapStore(source_store, mode="r")
    chrom_key = normalize_chrom(chrom, prefix=True)
    chrom_group = src.root[chrom_key]
    if not isinstance(chrom_group, zarr.hierarchy.Group):
        raise KeyError(f"node for chromosome {chrom_key!r} is not a group")

    all_sample_ids = src.sample_ids(chrom_key)
    sample_plan = _plan_sample_subset(all_sample_ids, sample_ids)
    if sample_plan.sample_mask is None:
        raise ValueError("sample_ids must be provided for subset_samples_to_new_store")

    variants = chrom_group["variants"]
    pos_ds = variants["pos"]
    ref_ds = variants["ref"]
    alt_ds = variants["alt"]
    vid_ds = variants["vid"]
    genetic_ds = variants["genetic_pos"] if "genetic_pos" in variants else None
    has_genetic = genetic_ds is not None

    samples_group = chrom_group["samples"]
    pop_ds = samples_group["population"] if "population" in samples_group else None
    has_pop = pop_ds is not None

    n_variants = int(pos_ds.shape[0])
    n_samples = len(sample_plan.selected_sample_ids)
    n_haplotypes = 2 * n_samples

    compressor = Zstd(level=int(compression_level))
    chunk_layout = resolve_matrix_chunk_layout(
        n_haplotypes=n_haplotypes,
        n_samples=n_samples,
        n_variants=n_variants,
        chunk_variant_size=int(variant_chunk),
        chunk_haplotype_size=hap_chunk,
    )
    variant_chunk = int(chunk_layout.variant)
    hap_chunk = int(chunk_layout.haplotype)
    layout_version = 1 if hap_chunk == n_haplotypes else 2

    out = HapStore(out_path, mode="w")
    out_root = out.root
    if chrom_key in out_root:
        raise ValueError(f"chromosome {chrom_key!r} already exists in output store: {out_path}")

    out_chrom = out_root.create_group(chrom_key)
    out_hap = out_chrom.create_dataset(
        "haplotypes",
        shape=(n_haplotypes, n_variants),
        chunks=(hap_chunk, variant_chunk),
        dtype=np.uint8,
        compressor=compressor,
        overwrite=False,
    )

    out_variants = out_chrom.create_group("variants")
    out_pos = out_variants.create_dataset(
        "pos",
        shape=(n_variants,),
        chunks=(variant_chunk,),
        dtype=np.int64,
        compressor=compressor,
        overwrite=False,
    )
    out_ref = out_variants.create_dataset(
        "ref",
        shape=(n_variants,),
        chunks=(variant_chunk,),
        dtype=object,
        object_codec=VLenUTF8(),
        compressor=compressor,
        overwrite=False,
    )
    out_alt = out_variants.create_dataset(
        "alt",
        shape=(n_variants,),
        chunks=(variant_chunk,),
        dtype=object,
        object_codec=VLenUTF8(),
        compressor=compressor,
        overwrite=False,
    )
    out_vid = out_variants.create_dataset(
        "vid",
        shape=(n_variants,),
        chunks=(variant_chunk,),
        dtype=object,
        object_codec=VLenUTF8(),
        compressor=compressor,
        overwrite=False,
    )
    out_genetic = None
    if has_genetic:
        out_genetic = out_variants.create_dataset(
            "genetic_pos",
            shape=(n_variants,),
            chunks=(variant_chunk,),
            dtype=np.float64,
            compressor=compressor,
            overwrite=False,
        )

    out_samples = out_chrom.create_group("samples")
    sample_chunk = int(min(max(1, n_samples), 8192))
    out_samples.create_dataset(
        "ids",
        data=np.asarray(sample_plan.selected_sample_ids, dtype=object),
        dtype=object,
        object_codec=VLenUTF8(),
        chunks=(sample_chunk,),
        compressor=compressor,
        overwrite=False,
    )

    if has_pop:
        pop_all = _to_unicode_array(np.asarray(pop_ds[:], dtype=object))
        subset_idx_requested = _subset_indices_in_requested_order(all_sample_ids, sample_plan.selected_sample_ids)
        pop_subset = pop_all[np.asarray(subset_idx_requested, dtype=np.int64)]
        out_samples.create_dataset(
            "population",
            data=np.asarray(pop_subset, dtype=object),
            dtype=object,
            object_codec=VLenUTF8(),
            chunks=(sample_chunk,),
            compressor=compressor,
            overwrite=False,
        )

    has_missing = False
    for variant_sel in _iter_variant_slices(n_variants, int(variant_block_size)):
        start = int(variant_sel.start or 0)
        stop = int(variant_sel.stop or 0)
        if start >= stop:
            continue

        hap_block = src.haplotypes(
            chrom_key,
            variant_slice=(start, stop),
            sample_mask=sample_plan.sample_mask,
        )
        if sample_plan.hap_perm is not None:
            hap_block = hap_block[sample_plan.hap_perm, :]

        out_hap[:, start:stop] = hap_block
        out_pos[start:stop] = np.asarray(pos_ds[start:stop], dtype=np.int64)
        out_ref[start:stop] = _to_unicode_array(np.asarray(ref_ds[start:stop], dtype=object))
        out_alt[start:stop] = _to_unicode_array(np.asarray(alt_ds[start:stop], dtype=object))
        out_vid[start:stop] = _to_unicode_array(np.asarray(vid_ds[start:stop], dtype=object))
        if out_genetic is not None and genetic_ds is not None:
            out_genetic[start:stop] = np.asarray(genetic_ds[start:stop], dtype=np.float64)

        if not has_missing and bool(np.any(hap_block == _MISSING_SENTINEL)):
            has_missing = True

    src_attrs = src.attrs(chrom_key)
    extra_in = src_attrs.get("extra")
    extra_payload: dict[str, Any] = {}
    if isinstance(extra_in, dict):
        extra_payload.update(extra_in)
    extra_payload.update(
        {
            "subset_from": str(Path(source_store)),
            "subset_n_samples": int(n_samples),
        }
    )

    out_chrom.attrs.update(
        build_chrom_attrs(
            n_samples=int(n_samples),
            n_haplotypes=int(n_haplotypes),
            n_variants=int(n_variants),
            missing_sentinel=_MISSING_SENTINEL,
            has_missing=bool(has_missing),
            compression_level=int(compression_level),
            chunk_variant_size=int(variant_chunk),
            chunk_haplotype_size=int(hap_chunk),
            chunk_sample_size=None,
            layout_version=int(layout_version),
            source=src_attrs.get("source"),
            build=src_attrs.get("build"),
            extra=extra_payload or None,
        )
    )

    return out_path
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
