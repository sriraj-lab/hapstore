"""Concatenate HapStores along their sample or variant axis.

Concatenation stacks already aligned stores in input order; it does not join,
sort, or harmonize records by sample or variant key. The module validates the
axis that must remain fixed, builds the result in a temporary store, and only
promotes that store to the requested output path after successful completion.
"""

from __future__ import annotations

import gc
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import zarr
from numcodecs import VLenUTF8, Zstd

from hapstore.chrom import normalize_chrom
from hapstore.seeding import stable_seed

from ._internal.chrom_attrs import build_chrom_attrs
from ._internal.chunking import (
    DEFAULT_VARIANT_CHUNK_SIZE,
    resolve_matrix_chunk_layout,
)
from ._internal.concat_copy import (
    _build_sample_axis_plans,
    _build_variant_axis_plans,
    _copy_dense_matrices,
)
from ._internal.zarr_utils import _create_utf8_array, _to_unicode_array
from .store import HapStore, _MISSING_SENTINEL


@dataclass
class _InputChrom:
    """Validated chromosome state cached for one input store."""

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


def _validate_and_collect_inputs(
    input_stores: Sequence[str | Path | HapStore],
    chrom_key: str,
) -> list[_InputChrom]:
    """Open inputs and collect the structural state required by concat."""

    collected: list[_InputChrom] = []

    for item in input_stores:
        store = item if isinstance(item, HapStore) else HapStore(item, mode="r")
        if not store.has_chromosome(chrom_key):
            raise KeyError(
                f"input store {store.path} does not contain chromosome {chrom_key!r}; "
                "use has_chromosome() before concatenating"
            )

        chrom_group = store.root[chrom_key]
        for required_name in ("haplotypes", "variants", "samples"):
            if required_name not in chrom_group:
                raise KeyError(
                    f"input store {store.path} chromosome {chrom_key!r} missing {required_name!r}"
                )

        variants_group = chrom_group["variants"]
        for field in ("pos", "ref", "alt", "vid"):
            if field not in variants_group:
                raise KeyError(
                    f"input store {store.path} chromosome {chrom_key!r} missing variants/{field!r} dataset"
                )

        haplotypes_arr = chrom_group["haplotypes"]
        sample_ids = _to_unicode_array(np.asarray(store.sample_ids(chrom_key), dtype=object))
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

        populations = None
        samples_group = chrom_group["samples"]
        if "population" in samples_group:
            populations = _to_unicode_array(np.asarray(samples_group["population"][:], dtype=object))

        collected.append(
            _InputChrom(
                path=store.path,
                store=store,
                chrom_key=chrom_key,
                n_samples=n_samples,
                n_haplotypes=n_haplotypes,
                n_variants=int(haplotypes_arr.shape[1]),
                sample_ids=sample_ids,
                populations=populations,
                haplotypes_arr=haplotypes_arr,
                variants_group=variants_group,
                has_missing=bool(store.attrs(chrom_key).get("has_missing", False)),
            )
        )

    return collected


def _validate_variant_axis_streaming(
    *,
    sources: Sequence[_InputChrom],
    variant_chunk: int,
) -> None:
    """Compare complete ordered variant metadata without loading it at once."""

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
    ref_genetic = ref_variants["genetic_pos"] if ref_has_genetic else None

    for idx, src in enumerate(sources[1:], start=1):
        src_has_genetic = "genetic_pos" in src.variants_group
        if ref_has_genetic != src_has_genetic:
            raise ValueError(
                "genetic position presence mismatch across input stores: "
                f"reference_has_genetic={ref_has_genetic}, store[{idx}]_has_genetic={src_has_genetic}"
            )

    for start in range(0, reference.n_variants, variant_chunk):
        end = min(start + variant_chunk, reference.n_variants)
        ref_pos_chunk = np.asarray(ref_variants["pos"][start:end], dtype=np.int64)
        ref_ref_chunk = _to_unicode_array(np.asarray(ref_variants["ref"][start:end], dtype=object))
        ref_alt_chunk = _to_unicode_array(np.asarray(ref_variants["alt"][start:end], dtype=object))
        ref_vid_chunk = _to_unicode_array(np.asarray(ref_variants["vid"][start:end], dtype=object))
        ref_genetic_chunk = (
            None if ref_genetic is None else np.asarray(ref_genetic[start:end], dtype=np.float64)
        )

        for idx, src in enumerate(sources[1:], start=1):
            src_variants = src.variants_group
            if not np.array_equal(np.asarray(src_variants["pos"][start:end], dtype=np.int64), ref_pos_chunk):
                raise ValueError(
                    "variant position mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            if not np.array_equal(
                _to_unicode_array(np.asarray(src_variants["ref"][start:end], dtype=object)),
                ref_ref_chunk,
            ):
                raise ValueError(
                    "REF allele mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            if not np.array_equal(
                _to_unicode_array(np.asarray(src_variants["alt"][start:end], dtype=object)),
                ref_alt_chunk,
            ):
                raise ValueError(
                    "ALT allele mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            if not np.array_equal(
                _to_unicode_array(np.asarray(src_variants["vid"][start:end], dtype=object)),
                ref_vid_chunk,
            ):
                raise ValueError(
                    "variant ID mismatch across input stores: "
                    f"reference={reference.path}, store[{idx}]={src.path}"
                )
            if ref_genetic_chunk is not None and not np.array_equal(
                np.asarray(src_variants["genetic_pos"][start:end], dtype=np.float64),
                ref_genetic_chunk,
            ):
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
    """Copy the reference source's ordered variant metadata into the output."""

    ref_variants = reference.variants_group
    variant_group.create_dataset(
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
    genetic_out = None
    if "genetic_pos" in ref_variants:
        genetic_out = variant_group.create_dataset(
            "genetic_pos",
            shape=(n_variants,),
            dtype=np.float64,
            chunks=(variant_chunk,),
            compressor=compressor,
            overwrite=False,
        )

    pos_out = variant_group["pos"]
    for start in range(0, n_variants, variant_chunk):
        end = min(start + variant_chunk, n_variants)
        pos_out[start:end] = np.asarray(ref_variants["pos"][start:end], dtype=np.int64)
        ref_out[start:end] = _to_unicode_array(np.asarray(ref_variants["ref"][start:end], dtype=object))
        alt_out[start:end] = _to_unicode_array(np.asarray(ref_variants["alt"][start:end], dtype=object))
        vid_out[start:end] = _to_unicode_array(np.asarray(ref_variants["vid"][start:end], dtype=object))
        if genetic_out is not None:
            genetic_out[start:end] = np.asarray(ref_variants["genetic_pos"][start:end], dtype=np.float64)


def _merge_populations_or_none(sources: Sequence[_InputChrom]) -> np.ndarray | None:
    """Append population labels, requiring either all or no sources to have them."""

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


Axis = Literal["samples", "variants", 0, 1]
ValidateMode = Literal["spot", "full", "none"]

@dataclass(frozen=True)
class _ResolvedLayout:
    """Compression and chunk policy resolved for the output store."""

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
    validate: ValidateMode = "full",
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

    This is a stacking operation, not a join or merge by key. Input order is
    preserved. Sample-axis concatenation appends diploid samples and requires
    an identical ordered variant axis. Variant-axis concatenation appends
    variant columns without sorting and requires an identical ordered sample
    axis.

    Args:
        input_stores: Ordered input paths or open HapStore instances.
        output_store: Destination path for the HapStore Zarr hierarchy.
        chrom: Chromosome present in every input. The key is normalized to the
            ``chrN`` form used by HapStore.
        axis: ``"samples"``/``0`` appends haplotype rows; ``"variants"``/``1``
            appends variant columns.
        validate: ``"full"`` scans the complete required input invariant;
            ``"spot"`` uses deterministic input and output slab checks; and
            ``"none"`` skips optional content checks. Structural checks and
            axis-defining sample checks still run for every mode.
        n_workers: Maximum copy workers. The count is capped by the number of
            sources. Variant concat and sample concat with genotype
            materialization currently use one worker.
        compression_level: Output Zstandard level. Passing ``None`` inherits
            the level only when all source attributes agree.
        chunk_variant_size: Output variant chunk size. Passing ``None``
            inherits the size only when all source attributes agree.
        chunk_haplotype_size: Requested output haplotype chunk size. When
            omitted, the shared matrix chunk policy selects it.
        chunk_sample_size: Requested output genotype sample chunk size. When
            omitted, the shared matrix chunk policy selects it.
        store_genotypes: Materialize output genotypes by pairing the copied
            haplotypes. Source genotype arrays are not concatenated.
        strict_variant_ids: For sample concat, compare ordered ``pos``,
            ``ref``, ``alt``, ``vid``, and optional ``genetic_pos`` metadata
            when validation is enabled. When ``False``, metadata from the first
            source defines the output without comparing the other sources.
        forbid_duplicate_variants: For variant concat, require source ranges
            to be nondecreasing. Under full validation, also reject duplicate
            non-missing variant IDs and duplicate ``(pos, ref, alt)`` keys.
        attrs: Additional output chromosome attributes. ``source`` and
            ``build`` are routed through the standard chromosome metadata.
        overwrite: Replace an existing destination after the temporary output
            has completed successfully.
        tmp_suffix: Suffix used for the temporary sibling output path.

    Returns:
        A read-only HapStore opened at ``output_store``.

    Raises:
        ValueError: If parameters or required cross-source invariants are
            invalid.
        FileExistsError: If the output or temporary path already exists and
            ``overwrite`` is ``False``.
        KeyError: If required datasets are missing in the input stores.

    Notes:
        Sample concat without genotype materialization can parallelize either
        by source or by disjoint output haplotype chunk. Compatible complete
        Zarr chunks may be copied without decoding; other layouts use decoded
        array blocks. The output is constructed under ``tmp_suffix`` and
        promoted by rename. If overwrite promotion fails, the previous output
        is restored; failed temporary output is removed.
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
    try:
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
        effective_workers = min(int(n_workers), len(sources))
        if axis_norm == "variants" and effective_workers > 1:
            logging.getLogger(__name__).info(
                "Falling back to serial variant-axis concat because source boundaries can share output chunks"
            )
            effective_workers = 1
        if axis_norm == "samples" and store_genotypes and effective_workers > 1:
            logging.getLogger(__name__).info(
                "Falling back to serial sample-axis concat while materializing genotypes"
            )
            effective_workers = 1

        tmp_store = HapStore(tmp_path, mode="w")
        if axis_norm == "samples":
            _concat_samples_into_store(
                sources=sources,
                out=tmp_store,
                chrom_key=chrom_key,
                validate=validate_norm,
                n_workers=effective_workers,
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
                n_workers=effective_workers,
                forbid_duplicate_variants=bool(forbid_duplicate_variants),
                store_genotypes=bool(store_genotypes),
                resolved=resolved,
                attrs=attrs,
            )

        del tmp_store
        gc.collect()
        _promote_concat_output(output_path=output_path, tmp_path=tmp_path, overwrite=overwrite)
        return HapStore(output_path, mode="r")
    except Exception:
        if tmp_path.exists():
            _rm_tree_or_file(tmp_path)
        raise


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

    if tmp_path.exists():
        _rm_tree_or_file(tmp_path)


def _promote_concat_output(*, output_path: Path, tmp_path: Path, overwrite: bool) -> None:
    if not output_path.exists():
        tmp_path.replace(output_path)
        return
    if not overwrite:
        raise FileExistsError(f"output_store already exists: {output_path}")
    backup_path = output_path.with_name(f"{output_path.name}.backup-{uuid.uuid4().hex}")
    output_path.replace(backup_path)
    try:
        tmp_path.replace(output_path)
    except Exception:
        backup_path.replace(output_path)
        raise
    finally:
        if backup_path.exists():
            _rm_tree_or_file(backup_path)


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
    """Resolve output compression and chunk overrides against source metadata."""

    attrs0 = sources[0].store.attrs(chrom_key)
    default_level = int(attrs0.get("compression_level", 3))
    default_variant_chunk = int(
        attrs0.get("chunk_variant_size", DEFAULT_VARIANT_CHUNK_SIZE)
    )

    if compression_level is None:
        levels = [int(src.store.attrs(chrom_key).get("compression_level", 3)) for src in sources]
        if any(x != levels[0] for x in levels[1:]):
            raise ValueError(
                "inconsistent source compression_level; pass compression_level explicitly "
                f"(values={sorted(set(levels))})"
            )

    if chunk_variant_size is None:
        chunks = [
            int(
                src.store.attrs(chrom_key).get(
                    "chunk_variant_size", DEFAULT_VARIANT_CHUNK_SIZE
                )
            )
            for src in sources
        ]
        if any(x != chunks[0] for x in chunks[1:]):
            raise ValueError(
                "inconsistent source chunk_variant_size; pass chunk_variant_size explicitly "
                f"(values={sorted(set(chunks))})"
            )

    resolved_level = int(default_level if compression_level is None else compression_level)
    if resolved_level < 0:
        raise ValueError(f"compression_level must be >= 0, got {resolved_level}")

    resolved_variant_chunk = int(default_variant_chunk if chunk_variant_size is None else chunk_variant_size)
    if resolved_variant_chunk <= 0:
        raise ValueError(f"chunk_variant_size must be > 0, got {resolved_variant_chunk}")

    if chunk_haplotype_size is None:
        hap_chunk_attr = None
    else:
        hap_chunk_attr = int(chunk_haplotype_size)
        if hap_chunk_attr <= 0:
            raise ValueError(f"chunk_haplotype_size must be > 0 when provided, got {hap_chunk_attr}")

    if chunk_sample_size is None:
        sample_chunk_attr = None
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
    """Append samples while preserving one identical ordered variant axis.

    Source sample and haplotype rows are written in ``sources`` order. Variant
    metadata is copied from the first source after the requested validation.
    Duplicate sample IDs are always rejected.
    """

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

    chunk_layout = resolve_matrix_chunk_layout(
        n_haplotypes=total_haps,
        n_samples=total_samples,
        n_variants=n_variants,
        chunk_variant_size=resolved.variant_chunk_attr,
        chunk_haplotype_size=resolved.chunk_haplotype_size_attr,
        chunk_sample_size=resolved.chunk_sample_size_attr,
    )
    variant_chunk = int(chunk_layout.variant)
    if strict_variant_ids:
        if validate == "full":
            _validate_variant_axis_streaming(sources=sources, variant_chunk=variant_chunk)
        elif validate == "spot":
            _spot_validate_variant_axis(sources=sources, variant_chunk=variant_chunk)

    hap_chunk = int(chunk_layout.haplotype)
    sample_chunk_geno = int(chunk_layout.sample)

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
    full_matrix_chunks = hap_chunk == total_haps and (
        not store_genotypes or sample_chunk_geno == total_samples
    )
    layout_version = 1 if full_matrix_chunks else 2
    chrom_group.attrs.update(
        build_chrom_attrs(
            n_samples=int(total_samples),
            n_haplotypes=int(total_haps),
            n_variants=int(n_variants),
            missing_sentinel=_MISSING_SENTINEL,
            has_missing=bool(has_missing),
            compression_level=int(resolved.compression_level),
            chunk_variant_size=int(variant_chunk),
            chunk_haplotype_size=int(hap_chunk),
            chunk_sample_size=(
                int(sample_chunk_geno) if store_genotypes else None
            ),
            layout_version=int(layout_version),
            source=source_attr,
            build=build_attr,
            extra=attrs_payload or None,
        )
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
    """Append variant columns while preserving one identical sample axis.

    Source variants are written in ``sources`` order and are not sorted or
    harmonized. Sample IDs, sample order, and population labels are always
    required to match the first source.
    """

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
        _enforce_no_duplicate_variant_keys_full(sources, variant_chunk=resolved.variant_chunk_attr)

    n_samples = int(sources[0].n_samples)
    n_haps = int(sources[0].n_haplotypes)
    total_variants = int(sum(src.n_variants for src in sources))
    has_missing = bool(any(src.has_missing for src in sources))

    chunk_layout = resolve_matrix_chunk_layout(
        n_haplotypes=n_haps,
        n_samples=n_samples,
        n_variants=total_variants,
        chunk_variant_size=resolved.variant_chunk_attr,
        chunk_haplotype_size=resolved.chunk_haplotype_size_attr,
        chunk_sample_size=resolved.chunk_sample_size_attr,
    )
    variant_chunk = int(chunk_layout.variant)
    hap_chunk = int(chunk_layout.haplotype)
    sample_chunk_geno = int(chunk_layout.sample)

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
    full_matrix_chunks = hap_chunk == n_haps and (
        not store_genotypes or sample_chunk_geno == n_samples
    )
    layout_version = 1 if full_matrix_chunks else 2
    chrom_group.attrs.update(
        build_chrom_attrs(
            n_samples=int(n_samples),
            n_haplotypes=int(n_haps),
            n_variants=int(total_variants),
            missing_sentinel=_MISSING_SENTINEL,
            has_missing=bool(has_missing),
            compression_level=int(resolved.compression_level),
            chunk_variant_size=int(variant_chunk),
            chunk_haplotype_size=int(hap_chunk),
            chunk_sample_size=(
                int(sample_chunk_geno) if store_genotypes else None
            ),
            layout_version=int(layout_version),
            source=source_attr,
            build=build_attr,
            extra=attrs_payload or None,
        )
    )


def _spot_validate_variant_axis(*, sources: Sequence[_InputChrom], variant_chunk: int) -> None:
    """Deterministically compare boundary and sampled variant metadata chunks."""

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
    """Deterministically compare five copied output slabs with their sources."""

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
        if left_last > right_first:
            raise ValueError(
                "variant position ranges are not non-decreasing across inputs: "
                f"store[{i}] last_pos={left_last}, store[{i+1}] first_pos={right_first}"
            )


def _enforce_no_duplicate_vids_full(sources: Sequence[_InputChrom]) -> None:
    seen: set[str] = set()
    for idx, src in enumerate(sources):
        vids = np.asarray(src.variants_group["vid"][:], dtype=object).tolist()
        for v in vids:
            text = str(v).strip()
            if text in {"", "."}:
                continue
            if text in seen:
                raise ValueError(
                    "duplicate variant IDs across input stores are not allowed for variant concat; "
                    f"first duplicate={text!r} at store[{idx}]={src.path}"
                )
            seen.add(text)


def _enforce_no_duplicate_variant_keys_full(
    sources: Sequence[_InputChrom],
    *,
    variant_chunk: int,
) -> None:
    seen: set[tuple[int, str, str]] = set()
    block_size = max(1, int(variant_chunk))
    for src_index, src in enumerate(sources):
        variants = src.variants_group
        for start in range(0, int(src.n_variants), block_size):
            end = min(int(start) + block_size, int(src.n_variants))
            positions = np.asarray(variants["pos"][start:end], dtype=np.int64)
            refs = _to_unicode_array(np.asarray(variants["ref"][start:end], dtype=object))
            alts = _to_unicode_array(np.asarray(variants["alt"][start:end], dtype=object))
            for offset, (pos, ref, alt) in enumerate(zip(positions, refs, alts, strict=True)):
                key = (int(pos), str(ref), str(alt))
                if key in seen:
                    raise ValueError(
                        "duplicate variant identity across input stores for variant concat: "
                        f"key={key!r}, store[{src_index}]={src.path}, index={start + offset}"
                    )
                seen.add(key)
