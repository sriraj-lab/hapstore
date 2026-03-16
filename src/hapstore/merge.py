"""HapStore merge utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr
from numcodecs import VLenUTF8, Zstd

from hapstore._internal.chrom import normalize_chrom

from .store import HapStore, _haplotypes_to_genotypes, _to_unicode_array


_MISSING_SENTINEL = 255


@dataclass
class _InputChrom:
    path: Path
    store: HapStore
    chrom_key: str
    n_samples: int
    n_haplotypes: int
    n_variants: int
    sample_ids: np.ndarray
    populations: np.ndarray | None
    haplotypes_arr: zarr.Array
    variants_group: zarr.Group
    has_missing: bool



def merge_chromosomes_sample_axis(
    input_stores: Sequence[str | Path | HapStore],
    output_store: str | Path,
    chrom: str | int,
    output_mode: str = "w",
    chunk_variant_size: int | None = None,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
    compression_level: int | None = None,
    store_genotypes: bool = False,
    attrs: dict | None = None,
) -> HapStore:
    """Merge chromosome data from multiple stores along sample axis.

    All inputs must contain the same chromosome and identical variant-axis metadata.
    The output chromosome sample axis is the concatenation of input sample axes in
    the provided order.
    """
    if not input_stores:
        raise ValueError("input_stores must not be empty")
    if output_mode not in {"a", "w"}:
        raise ValueError(f"output_mode must be 'a' or 'w', got {output_mode!r}")

    warnings.warn(
        "merge_chromosomes_sample_axis is deprecated; use concat_hapstores(..., axis='samples') instead",
        DeprecationWarning,
        stacklevel=2,
    )

    if output_mode == "w":
        from .concat import concat_hapstores

        concat_hapstores(
            input_stores=input_stores,
            output_store=output_store,
            chrom=chrom,
            axis="samples",
            validate="full",
            n_workers=1,
            compression_level=compression_level,
            chunk_variant_size=chunk_variant_size,
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
            store_genotypes=store_genotypes,
            strict_variant_ids=True,
            attrs=attrs,
            overwrite=True,
            tmp_suffix=".tmp",
        )
        return HapStore(output_store, mode="a")

    chrom_key = normalize_chrom(chrom, prefix=True)
    sources = _validate_and_collect_inputs(input_stores=input_stores, chrom_key=chrom_key)

    default_chunk = int(sources[0].store.attrs(chrom_key).get("chunk_variant_size", 500))
    default_level = int(sources[0].store.attrs(chrom_key).get("compression_level", 3))
    resolved_chunk = int(default_chunk if chunk_variant_size is None else chunk_variant_size)
    resolved_level = int(default_level if compression_level is None else compression_level)

    if resolved_chunk <= 0:
        raise ValueError(f"chunk_variant_size must be > 0, got {resolved_chunk}")
    if chunk_haplotype_size is not None and int(chunk_haplotype_size) <= 0:
        raise ValueError(f"chunk_haplotype_size must be > 0 when provided, got {chunk_haplotype_size}")
    if chunk_sample_size is not None and int(chunk_sample_size) <= 0:
        raise ValueError(f"chunk_sample_size must be > 0 when provided, got {chunk_sample_size}")
    if resolved_level < 0:
        raise ValueError(f"compression_level must be >= 0, got {resolved_level}")

    n_variants = int(sources[0].n_variants)
    total_samples = int(sum(src.n_samples for src in sources))
    total_haps = int(sum(src.n_haplotypes for src in sources))
    if total_haps != (2 * total_samples):
        raise ValueError(
            "inconsistent haplotype/sample counts across inputs: "
            f"total_haplotypes={total_haps}, expected={2 * total_samples}"
        )
    has_missing = bool(any(src.has_missing for src in sources))

    merged_sample_ids = np.concatenate([src.sample_ids for src in sources], axis=0)
    duplicates = _find_duplicate_values(merged_sample_ids)
    if duplicates:
        raise ValueError(
            "duplicate sample IDs across input stores are not allowed; "
            f"first duplicate={duplicates[0]!r}"
        )

    merged_populations = _merge_populations_or_none(sources)

    out = HapStore(output_store, mode=output_mode)
    if out.has_chromosome(chrom_key):
        raise ValueError(
            f"chromosome {chrom_key!r} already exists in output store {out.path}; "
            "delete it first or use a different output path"
        )

    compressor = Zstd(level=resolved_level)
    variant_chunk = int(min(max(1, resolved_chunk), max(1, n_variants)))
    hap_chunk = int(total_haps if chunk_haplotype_size is None else chunk_haplotype_size)
    hap_chunk = int(min(max(1, hap_chunk), total_haps))
    sample_chunk_geno = int(total_samples if chunk_sample_size is None else chunk_sample_size)
    sample_chunk_geno = int(min(max(1, sample_chunk_geno), total_samples))
    _validate_variant_axis_streaming(sources=sources, variant_chunk=variant_chunk)

    chrom_group = out.root.create_group(chrom_key)
    hap_arr = chrom_group.create_dataset(
        "haplotypes",
        shape=(total_haps, n_variants),
        dtype=np.uint8,
        chunks=(hap_chunk, variant_chunk),
        compressor=compressor,
        overwrite=False,
    )

    geno_arr = None
    if store_genotypes:
        geno_arr = chrom_group.create_dataset(
            "genotypes",
            shape=(total_samples, n_variants),
            dtype=np.uint8,
            chunks=(sample_chunk_geno, variant_chunk),
            compressor=compressor,
            overwrite=False,
        )

    variant_group = chrom_group.create_group("variants")
    _write_variant_axis_from_reference(
        variant_group=variant_group,
        reference=sources[0],
        n_variants=n_variants,
        variant_chunk=variant_chunk,
        compressor=compressor,
    )

    sample_group = chrom_group.create_group("samples")
    sample_chunk = int(min(max(1, total_samples), 8192))
    _create_utf8_array(sample_group, "ids", merged_sample_ids, sample_chunk, compressor)
    if merged_populations is not None:
        _create_utf8_array(sample_group, "population", merged_populations, sample_chunk, compressor)

    _write_haplotypes_chunkwise(
        sources=sources,
        hap_arr=hap_arr,
        n_variants=n_variants,
        variant_chunk=variant_chunk,
        hap_chunk=hap_chunk,
    )
    if geno_arr is not None:
        _write_genotypes_chunkwise(
            sources=sources,
            geno_arr=geno_arr,
            n_variants=n_variants,
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
            "compression_level": int(resolved_level),
            "chunk_variant_size": int(variant_chunk),
            "chunk_haplotype_size": None
            if chunk_haplotype_size is None
            else int(chunk_haplotype_size),
            "chunk_sample_size": None if chunk_sample_size is None else int(chunk_sample_size),
            "layout_version": int(layout_version),
            "source": source_attr,
            "build": build_attr,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extra": attrs_payload or None,
        }
    )

    return out



def _validate_and_collect_inputs(
    input_stores: Sequence[str | Path | HapStore],
    chrom_key: str,
) -> list[_InputChrom]:
    collected: list[_InputChrom] = []

    for idx, item in enumerate(input_stores):
        store = item if isinstance(item, HapStore) else HapStore(item, mode="r")
        if not store.has_chromosome(chrom_key):
            raise KeyError(
                f"input store {store.path} does not contain chromosome {chrom_key!r}; "
                "use has_chromosome() before merging"
            )

        chrom_group = store.root[chrom_key]
        if "haplotypes" not in chrom_group:
            raise KeyError(f"input store {store.path} chromosome {chrom_key!r} missing 'haplotypes' dataset")
        if "variants" not in chrom_group:
            raise KeyError(f"input store {store.path} chromosome {chrom_key!r} missing 'variants' group")
        if "samples" not in chrom_group:
            raise KeyError(f"input store {store.path} chromosome {chrom_key!r} missing 'samples' group")

        variants_group = chrom_group["variants"]
        for field in ("pos", "ref", "alt", "vid"):
            if field not in variants_group:
                raise KeyError(
                    f"input store {store.path} chromosome {chrom_key!r} missing variants/{field!r} dataset"
                )

        haplotypes_arr = chrom_group["haplotypes"]
        sample_ids = np.asarray(store.sample_ids(chrom_key), dtype=object)
        sample_ids = _to_unicode_array(sample_ids)
        n_samples = int(store.n_samples(chrom_key))
        if sample_ids.shape[0] != n_samples:
            raise ValueError(
                f"input store {store.path} chromosome {chrom_key!r} sample ID count mismatch: "
                f"n_samples={n_samples}, len(sample_ids)={sample_ids.shape[0]}"
            )
        n_haplotypes = int(haplotypes_arr.shape[0])
        if n_haplotypes != (2 * n_samples):
            raise ValueError(
                f"input store {store.path} chromosome {chrom_key!r} has n_haplotypes={n_haplotypes} "
                f"but expected {2 * n_samples} from n_samples={n_samples}"
            )
        n_variants = int(haplotypes_arr.shape[1])

        samples_group = chrom_group["samples"]
        populations = None
        if "population" in samples_group:
            populations = _to_unicode_array(np.asarray(samples_group["population"][:], dtype=object))

        chrom_attrs = store.attrs(chrom_key)

        collected.append(
            _InputChrom(
                path=store.path,
                store=store,
                chrom_key=chrom_key,
                n_samples=n_samples,
                n_haplotypes=n_haplotypes,
                n_variants=n_variants,
                sample_ids=sample_ids,
                populations=populations,
                haplotypes_arr=haplotypes_arr,
                variants_group=variants_group,
                has_missing=bool(chrom_attrs.get("has_missing", False)),
            )
        )

    return collected


def _validate_variant_axis_streaming(
    *,
    sources: Sequence[_InputChrom],
    variant_chunk: int,
) -> None:
    reference = sources[0]
    for idx, src in enumerate(sources[1:], start=1):
        if src.n_variants != reference.n_variants:
            raise ValueError(
                "n_variants mismatch across input stores: "
                f"reference({reference.path})={reference.n_variants}, "
                f"store[{idx}]({src.path})={src.n_variants}"
            )

    ref_variants = reference.variants_group
    ref_has_genetic = "genetic_pos" in ref_variants
    ref_pos = ref_variants["pos"]
    ref_ref = ref_variants["ref"]
    ref_alt = ref_variants["alt"]
    ref_vid = ref_variants["vid"]
    ref_genetic = ref_variants["genetic_pos"] if ref_has_genetic else None

    for idx, src in enumerate(sources[1:], start=1):
        src_variants = src.variants_group
        src_has_genetic = "genetic_pos" in src_variants
        if ref_has_genetic != src_has_genetic:
            raise ValueError(
                "genetic position presence mismatch across input stores: "
                f"reference_has_genetic={ref_has_genetic}, store[{idx}]_has_genetic={src_has_genetic}"
            )

    for start in range(0, reference.n_variants, variant_chunk):
        end = min(start + variant_chunk, reference.n_variants)
        ref_pos_chunk = np.asarray(ref_pos[start:end], dtype=np.int64)
        ref_ref_chunk = _to_unicode_array(np.asarray(ref_ref[start:end], dtype=object))
        ref_alt_chunk = _to_unicode_array(np.asarray(ref_alt[start:end], dtype=object))
        ref_vid_chunk = _to_unicode_array(np.asarray(ref_vid[start:end], dtype=object))
        ref_genetic_chunk = (
            None if ref_genetic is None else np.asarray(ref_genetic[start:end], dtype=np.float64)
        )

        for idx, src in enumerate(sources[1:], start=1):
            src_variants = src.variants_group
            src_pos_chunk = np.asarray(src_variants["pos"][start:end], dtype=np.int64)
            if not np.array_equal(src_pos_chunk, ref_pos_chunk):
                raise ValueError(
                    "variant position mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            src_ref_chunk = _to_unicode_array(np.asarray(src_variants["ref"][start:end], dtype=object))
            if not np.array_equal(src_ref_chunk, ref_ref_chunk):
                raise ValueError(
                    "REF allele mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            src_alt_chunk = _to_unicode_array(np.asarray(src_variants["alt"][start:end], dtype=object))
            if not np.array_equal(src_alt_chunk, ref_alt_chunk):
                raise ValueError(
                    "ALT allele mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            src_vid_chunk = _to_unicode_array(np.asarray(src_variants["vid"][start:end], dtype=object))
            if not np.array_equal(src_vid_chunk, ref_vid_chunk):
                raise ValueError(
                    "variant ID mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )

            if ref_genetic_chunk is not None:
                src_genetic_chunk = np.asarray(src_variants["genetic_pos"][start:end], dtype=np.float64)
                if not np.array_equal(src_genetic_chunk, ref_genetic_chunk):
                    raise ValueError(
                        "genetic position values mismatch across input stores: "
                        f"reference={reference.path}, store[{idx}]={src.path}"
                    )


def _write_variant_axis_from_reference(
    *,
    variant_group: zarr.Group,
    reference: _InputChrom,
    n_variants: int,
    variant_chunk: int,
    compressor: Zstd,
) -> None:
    ref_variants = reference.variants_group
    pos_out = variant_group.create_dataset(
        "pos",
        shape=(n_variants,),
        dtype=np.int64,
        chunks=(variant_chunk,),
        compressor=compressor,
        overwrite=False,
    )
    ref_out = _create_utf8_array_empty(variant_group, "ref", n_variants, variant_chunk, compressor)
    alt_out = _create_utf8_array_empty(variant_group, "alt", n_variants, variant_chunk, compressor)
    vid_out = _create_utf8_array_empty(variant_group, "vid", n_variants, variant_chunk, compressor)
    has_genetic = "genetic_pos" in ref_variants
    genetic_out = None
    if has_genetic:
        genetic_out = variant_group.create_dataset(
            "genetic_pos",
            shape=(n_variants,),
            dtype=np.float64,
            chunks=(variant_chunk,),
            compressor=compressor,
            overwrite=False,
        )

    for start in range(0, n_variants, variant_chunk):
        end = min(start + variant_chunk, n_variants)
        pos_out[start:end] = np.asarray(ref_variants["pos"][start:end], dtype=np.int64)
        ref_out[start:end] = _to_unicode_array(np.asarray(ref_variants["ref"][start:end], dtype=object))
        alt_out[start:end] = _to_unicode_array(np.asarray(ref_variants["alt"][start:end], dtype=object))
        vid_out[start:end] = _to_unicode_array(np.asarray(ref_variants["vid"][start:end], dtype=object))
        if genetic_out is not None:
            genetic_out[start:end] = np.asarray(ref_variants["genetic_pos"][start:end], dtype=np.float64)


def _write_haplotypes_chunkwise(
    *,
    sources: Sequence[_InputChrom],
    hap_arr: zarr.Array,
    n_variants: int,
    variant_chunk: int,
    hap_chunk: int,
) -> None:
    source_ranges: list[tuple[_InputChrom, int, int]] = []
    hap_offset = 0
    for src in sources:
        next_offset = hap_offset + src.n_haplotypes
        source_ranges.append((src, hap_offset, next_offset))
        hap_offset = next_offset

    total_haps = int(hap_arr.shape[0])
    for out_h_start in range(0, total_haps, hap_chunk):
        out_h_end = min(out_h_start + hap_chunk, total_haps)
        out_rows = int(out_h_end - out_h_start)
        contributors: list[tuple[_InputChrom, int, int, int, int]] = []
        block_row = 0
        for src, src_global_start, src_global_end in source_ranges:
            overlap_start = max(out_h_start, src_global_start)
            overlap_end = min(out_h_end, src_global_end)
            if overlap_start >= overlap_end:
                continue
            src_local_start = overlap_start - src_global_start
            src_local_end = overlap_end - src_global_start
            rows = overlap_end - overlap_start
            contributors.append((src, src_local_start, src_local_end, block_row, block_row + rows))
            block_row += rows
            if overlap_end >= out_h_end:
                break
        if block_row != out_rows:
            raise RuntimeError(
                f"internal merge error: row-coverage mismatch for output rows {out_h_start}:{out_h_end} "
                f"(covered={block_row}, expected={out_rows})"
            )

        for start in range(0, n_variants, variant_chunk):
            end = min(start + variant_chunk, n_variants)
            merged_block = np.empty((out_rows, end - start), dtype=np.uint8)
            for src, src_local_start, src_local_end, block_start, block_end in contributors:
                merged_block[block_start:block_end, :] = np.asarray(
                    src.haplotypes_arr[src_local_start:src_local_end, start:end],
                    dtype=np.uint8,
                )
            hap_arr[out_h_start:out_h_end, start:end] = merged_block


def _write_genotypes_chunkwise(
    *,
    sources: Sequence[_InputChrom],
    geno_arr: zarr.Array,
    n_variants: int,
    variant_chunk: int,
) -> None:
    for start in range(0, n_variants, variant_chunk):
        end = min(start + variant_chunk, n_variants)
        sample_row = 0
        for src in sources:
            h = np.asarray(src.haplotypes_arr[:, start:end], dtype=np.uint8)
            geno_arr[sample_row : sample_row + src.n_samples, start:end] = _haplotypes_to_genotypes(h)
            sample_row += src.n_samples



def _merge_populations_or_none(sources: Sequence[_InputChrom]) -> np.ndarray | None:
    first_has = sources[0].populations is not None
    for idx, src in enumerate(sources[1:], start=1):
        if (src.populations is not None) != first_has:
            raise ValueError(
                "population annotation presence mismatch across input stores: "
                f"reference_has_population={first_has}, store[{idx}]_has_population={src.populations is not None}"
            )

    if not first_has:
        return None
    return np.concatenate([src.populations for src in sources if src.populations is not None], axis=0)



def _find_duplicate_values(values: np.ndarray) -> list[str]:
    seen: set[str] = set()
    dup: list[str] = []
    for value in values.tolist():
        text = str(value)
        if text in seen:
            dup.append(text)
            if len(dup) >= 10:
                break
        else:
            seen.add(text)
    return dup



def _create_utf8_array(
    group: zarr.Group,
    name: str,
    values: np.ndarray,
    chunk_size: int,
    compressor: Zstd,
) -> None:
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=object,
        object_codec=VLenUTF8(),
        chunks=(int(max(1, chunk_size)),),
        compressor=compressor,
        overwrite=False,
    )


def _create_utf8_array_empty(
    group: zarr.Group,
    name: str,
    length: int,
    chunk_size: int,
    compressor: Zstd,
) -> zarr.Array:
    return group.create_dataset(
        name,
        shape=(int(length),),
        dtype=object,
        object_codec=VLenUTF8(),
        chunks=(int(max(1, chunk_size)),),
        compressor=compressor,
        overwrite=False,
    )
