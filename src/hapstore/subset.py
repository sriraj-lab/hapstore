"""Copy selected samples from one HapStore chromosome into a new store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import VLenUTF8, Zstd

from hapstore.chrom import normalize_chrom

from ._internal.chrom_attrs import build_chrom_attrs
from ._internal.chunking import DEFAULT_VARIANT_CHUNK_SIZE, resolve_matrix_chunk_layout
from ._internal.selection import (
    _iter_variant_slices,
    _plan_sample_subset,
    _subset_indices_in_requested_order,
)
from ._internal.zarr_utils import _to_unicode_array
from .store import HapStore, _MISSING_SENTINEL


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
    """Create a one-chromosome HapStore containing selected samples.

    Args:
        source_store: Existing HapStore path.
        chrom: Chromosome to copy, normalized internally to ``chrN``.
        sample_ids: Unique source IDs in the requested output order.
        output_store: New destination store path. It must not already exist.
        variant_block_size: Variants copied per source read.
        variant_chunk: Variant-axis chunk size in the destination.
        hap_chunk: Optional haplotype-axis chunk size in the destination.
        compression_level: Zstandard compression level for destination arrays.

    Returns:
        The newly created one-chromosome HapStore path.

    Raises:
        FileExistsError: If ``output_store`` already exists.
        ValueError: If chunk parameters are invalid or requested sample IDs are
            duplicated or absent from the source store.

    Notes:
        Haplotype and sample metadata preserve ``sample_ids`` order. The full
        variant axis, genetic positions when available, and population labels
        when available are copied. The source's materialized genotype array is
        intentionally not retained. Data are streamed by variant blocks, so
        the source matrix is never loaded in full.
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
