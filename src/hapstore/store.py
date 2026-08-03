"""Zarr-backed haplotype/genotype store."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterator, Sequence

import numpy as np
import zarr
from numcodecs import VLenUTF8, Zstd

from hapstore.chrom import normalize_chrom

from ._internal.chrom_attrs import build_chrom_attrs
from ._internal.chunking import (
    DEFAULT_VARIANT_CHUNK_SIZE,
    resolve_matrix_chunk_layout,
)
from ._internal.zarr_utils import _create_utf8_array, _to_unicode_array
from .windows import compute_window_boundaries, compute_window_midpoints

_MISSING_SENTINEL = 255
_SCHEMA_VERSION = 2
_AUTO_READ_TILE_MEMORY_FRACTION = 0.5


@dataclass(frozen=True)
class HapStoreDosageRequest:
    """One named variant selection for a shared chromosome dosage read.

    Exactly one of ``variant_ids`` or ``variant_indices`` must be provided
    when the request is passed to :meth:`HapStore.read_dosage_batch`.
    """

    request_id: Hashable
    variant_ids: Sequence[str] | None = None
    variant_indices: Sequence[int] | np.ndarray | None = None


@dataclass(frozen=True)
class HapStoreDosageBatch:
    """Shared union dosage matrix and mappings for individual requests.

    ``matrix`` has shape ``(n_samples, n_union_variants)`` and follows the
    sorted ``variant_indices`` order. Each entry in ``request_columns`` maps
    the union matrix back to the original order, including duplicates, for one
    request.
    """

    matrix: np.ndarray
    variant_indices: np.ndarray
    request_columns: dict[Hashable, np.ndarray]
    request_ids: tuple[Hashable, ...]
    variant_ids: tuple[str, ...] | None = None
    metrics: dict[str, Any] | None = None

    def window_matrix(self, request_id: Hashable) -> np.ndarray:
        """Return the dosage matrix for one request in its original order."""
        cols = self.request_columns[request_id]
        return np.asarray(self.matrix[:, cols], dtype=self.matrix.dtype)

    def project(self, request_id: Hashable, weights: np.ndarray) -> np.ndarray:
        """Multiply one request's dosage matrix by a one-dimensional weight vector."""
        cols = self.request_columns[request_id]
        w = np.asarray(weights, dtype=self.matrix.dtype)
        if w.ndim != 1 or w.shape[0] != cols.shape[0]:
            raise ValueError(
                f"weights for request {request_id!r} must have shape ({cols.shape[0]},), "
                f"got {w.shape}"
            )
        return np.asarray(self.matrix[:, cols] @ w, dtype=self.matrix.dtype)


@dataclass(frozen=True)
class HapStoreBlock:
    """One variant-axis block with aligned values and variant metadata.

    ``values`` has shape ``(n_samples, n_variants)`` for genotype blocks and
    ``(2 * n_samples, n_variants)`` for haplotype blocks. ``start`` and
    ``stop`` bound the selected variant indices but do not imply that every
    intervening variant is present.
    """

    values: np.ndarray
    variant_indices: np.ndarray
    positions: np.ndarray
    ref: tuple[str, ...]
    alt: tuple[str, ...]
    vid: tuple[str, ...]
    start: int
    stop: int
    data: str
    sample_indices: np.ndarray
    genetic_pos: np.ndarray | None = None


@dataclass(frozen=True)
class _ReadGroup:
    """One contiguous read group for a shared chromosome read."""

    start: int
    stop: int
    offsets: np.ndarray
    out_positions: np.ndarray


class HapStore:
    """Open a chromosome-partitioned Zarr store of phased diploid haplotypes.

    A HapStore contains one group per chromosome, normalized to keys such as
    ``chr1``. Chromosome groups share one sample universe and ordering while
    retaining chromosome-specific variant metadata, haplotypes, and optional
    materialized genotypes.

    Args:
        path: Path to the HapStore Zarr hierarchy.
        mode: Zarr open mode. ``"r"`` opens an existing store read-only,
            ``"a"`` opens or creates a store for incremental writes, and
            ``"w"`` creates or replaces a store. Write methods reject
            read-only instances.

    Attributes:
        chromosomes: Sorted canonical chromosome group names present in the
            store.
        root: Underlying Zarr root group. Direct mutation is intended for
            advanced use and bypasses HapStore validation.
        path: Filesystem path used to open the store.
        last_read_metrics: Copy of the planning and timing metrics recorded by
            the most recent dosage or dosage-batch read on this instance.

    Raises:
        ValueError: If ``mode`` is not ``"r"``, ``"a"``, or ``"w"``.
        FileNotFoundError: If ``mode="r"`` and ``path`` does not exist.

    Notes:
        Writable stores are initialized with root-level ``format="hapstore"``
        and the current schema version when those attributes are absent.
        Chromosome creation enforces identical sample identifiers and ordering
        across chromosomes; read methods preserve requested sample and variant
        order unless their individual contracts state otherwise.

    Example:
        >>> store = HapStore("panel.hapstore.zarr", mode="r")
        >>> store.chromosomes
        ['chr1', 'chr2']
        >>> store.n_samples("1")
        1000
    """

    def __init__(self, path: str | Path, mode: str = "r") -> None:
        if mode not in {"r", "a", "w"}:
            raise ValueError(f"mode must be one of 'r', 'a', 'w'; got {mode!r}")

        self._path = Path(path)
        self._mode = mode

        if mode == "r" and not self._path.exists():
            raise FileNotFoundError(f"hapstore path does not exist: {self._path}")

        self._root = zarr.open(str(self._path), mode=mode)
        self._variant_index_cache: dict[str, dict[str, int]] = {}
        self._resolved_variant_alias_cache: dict[str, dict[str, int]] = {}
        self._missing_variant_alias_cache: dict[str, set[str]] = {}
        self._ambiguous_variant_alias_cache: dict[str, set[str]] = {}
        self._last_read_metrics: dict[str, Any] = {}
        if mode in {"a", "w"}:
            if "format" not in self._root.attrs:
                self._root.attrs["format"] = "hapstore"
            if "schema_version" not in self._root.attrs:
                self._root.attrs["schema_version"] = _SCHEMA_VERSION

    @property
    def chromosomes(self) -> list[str]:
        """Return chromosome group names in sorted order."""
        return sorted(list(self._root.group_keys()))

    @property
    def root(self) -> zarr.Group:
        """Return the underlying Zarr root group."""
        return self._root

    @property
    def path(self) -> Path:
        """Return the filesystem path to the store."""
        return self._path

    @property
    def last_read_metrics(self) -> dict[str, Any]:
        """Return metrics from the most recent non-empty dosage read."""
        return dict(self._last_read_metrics)

    def _chrom_key(self, chrom: str | int) -> str:
        return normalize_chrom(chrom, prefix=True)

    def _require_write_mode(self) -> None:
        if self._mode == "r":
            raise ValueError("store opened in read-only mode; use mode='a' or mode='w' for writes")

    def _require_chromosome(self, chrom: str | int) -> zarr.Group:
        chrom_key = self._chrom_key(chrom)
        if chrom_key not in self._root:
            raise KeyError(
                f"chromosome {chrom_key!r} not found in store {self._path}. "
                "Use has_chromosome() before reading."
            )
        node = self._root[chrom_key]
        if not isinstance(node, zarr.hierarchy.Group):
            raise KeyError(f"node for chromosome {chrom_key!r} is not a group")
        return node

    def _validate_sample_consistency(self, sample_ids: np.ndarray, chrom_key: str) -> None:
        for existing in self.chromosomes:
            if existing == chrom_key:
                continue
            existing_ids = np.asarray(self._root[existing]["samples"]["ids"][:], dtype=object)
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

    def create_chromosome(
        self,
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
    ) -> None:
        """Create and populate one chromosome group in the store.

        Haplotypes must use the phased diploid layout ``(2N, V)``: rows
        ``2*i`` and ``2*i + 1`` are the two haplotypes for sample ``i``, and
        columns follow the order of the variant metadata arrays. Values are
        restricted to reference allele ``0``, alternate allele ``1``, and the
        fixed missing sentinel ``255``.

        Args:
            chrom: Chromosome label. It is normalized to the canonical
                ``chrN`` group key, so labels such as ``1`` and ``chr1`` refer
                to the same chromosome.
            haplotypes: Phased allele matrix with shape ``(2N, V)``. The array
                is stored as ``uint8`` without changing row or column order.
            variant_pos: Positive integer base-pair coordinates with shape
                ``(V,)``. Coordinates must be non-decreasing; duplicate
                positions are allowed.
            sample_ids: Non-empty, unique sample identifiers with shape
                ``(N,)``. Every chromosome in one HapStore must use the same
                sample identifiers in the same order.
            ref: Reference alleles with shape ``(V,)``. Defaults to ``"."``
                for every variant when omitted.
            alt: Alternate alleles with shape ``(V,)``. Defaults to ``"."``
                for every variant when omitted.
            vid: Variant identifiers with shape ``(V,)``. Defaults to ``"."``
                when omitted. Non-placeholder identifiers must be unique
                within the chromosome.
            genetic_positions: Optional genetic-map coordinates in
                centimorgans with shape ``(V,)``. Values must be finite.
            populations: Optional population label for each sample, with
                shape ``(N,)``.
            store_genotypes: Also materialize a dosage matrix with shape
                ``(N, V)`` under ``genotypes``. Dosages are ``0``, ``1``, or
                ``2``; a genotype is ``255`` when either haplotype is missing.
            chunk_variant_size: Requested number of variant columns per Zarr
                chunk. The stored value is capped at ``V``.
            chunk_haplotype_size: Requested number of haplotype rows per
                ``haplotypes`` chunk. When omitted, HapStore chooses a
                byte-targeted, diploid-pair-aligned size.
            chunk_sample_size: Requested number of sample rows per
                ``genotypes`` chunk. It is relevant only when
                ``store_genotypes`` is true; when omitted, HapStore chooses a
                byte-targeted size.
            compression_level: Non-negative Zstandard compression level used
                for chromosome arrays.
            attrs: Optional provenance metadata. ``source`` and ``build`` are
                promoted to chromosome attributes; remaining entries are
                stored under the ``extra`` attribute. Format, shape, missing,
                compression, and chunk attributes are generated by HapStore.

        Returns:
            None.

        Raises:
            ValueError: If the store is read-only, the chromosome already
                exists, array shapes or values violate the HapStore contract,
                chunk/compression settings are invalid, or sample identity
                differs from an existing chromosome.

        Notes:
            The method writes ``haplotypes``, optional ``genotypes``,
            ``variants/{pos,ref,alt,vid}``, optional
            ``variants/genetic_pos``, ``samples/ids``, and optional
            ``samples/population``. If an exception occurs after chromosome
            group creation, the partial chromosome group is deleted before
            the exception is re-raised.
        """
        self._require_write_mode()

        chrom_key = self._chrom_key(chrom)
        if chrom_key in self._root:
            raise ValueError(
                f"chromosome {chrom_key!r} already exists in store. "
                "Delete it first or write to a different store."
            )

        haps = np.asarray(haplotypes)
        if haps.ndim != 2:
            raise ValueError(f"haplotypes must be 2D, got shape={haps.shape}")
        if haps.shape[0] % 2 != 0:
            raise ValueError(
                f"haplotypes row count must be even (2 * n_samples), got {haps.shape[0]}"
            )

        if not np.isin(haps, (0, 1, _MISSING_SENTINEL)).all():
            first_bad = np.argwhere(~np.isin(haps, (0, 1, _MISSING_SENTINEL)))[0]
            bad_value = haps[tuple(first_bad)]
            raise ValueError(
                "haplotypes must contain only 0, 1, or the missing sentinel "
                f"{_MISSING_SENTINEL}; found {bad_value!r} at "
                f"({int(first_bad[0])}, {int(first_bad[1])})"
            )

        sample_ids_raw = np.asarray(sample_ids, dtype=object)
        if sample_ids_raw.ndim != 1 or sample_ids_raw.shape[0] == 0:
            raise ValueError(
                "sample_ids must be a non-empty 1D array, got "
                f"shape={sample_ids_raw.shape}"
            )
        sample_ids_arr = _to_unicode_array(sample_ids_raw)
        if any(not str(sample_id).strip() for sample_id in sample_ids_arr.tolist()):
            raise ValueError("sample_ids must not contain empty values")
        if len(set(sample_ids_arr.tolist())) != int(sample_ids_arr.shape[0]):
            raise ValueError("sample_ids must be unique")
        n_samples = int(sample_ids_arr.shape[0])
        if haps.shape[0] != 2 * n_samples:
            raise ValueError(
                "haplotype/sample mismatch: "
                f"haplotypes.shape[0]={haps.shape[0]} but 2*len(sample_ids)={2 * n_samples}"
            )

        variant_pos_raw = np.asarray(variant_pos)
        if variant_pos_raw.ndim != 1:
            raise ValueError(f"variant_pos must be 1D, got shape={variant_pos_raw.shape}")
        try:
            variant_pos_float = np.asarray(variant_pos_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("variant_pos must contain numeric values") from exc
        if not np.all(np.isfinite(variant_pos_float)) or not np.all(variant_pos_float == np.floor(variant_pos_float)):
            raise ValueError("variant_pos must contain finite integer coordinates")
        if np.any(variant_pos_float <= 0):
            raise ValueError("variant_pos must contain positive coordinates")
        variant_pos_arr = np.asarray(variant_pos_float, dtype=np.int64)
        n_variants = int(variant_pos_arr.shape[0])
        if n_variants == 0:
            raise ValueError("hapstore chromosomes must contain at least one variant")
        if haps.shape[1] != n_variants:
            raise ValueError(
                "haplotype/variant mismatch: "
                f"haplotypes.shape[1]={haps.shape[1]} but len(variant_pos)={n_variants}"
            )
        if n_variants > 1 and np.any(np.diff(variant_pos_arr) < 0):
            raise ValueError("variant_pos must be non-decreasing")

        if chunk_variant_size <= 0:
            raise ValueError(f"chunk_variant_size must be positive, got {chunk_variant_size}")
        if chunk_haplotype_size is not None and int(chunk_haplotype_size) <= 0:
            raise ValueError(
                f"chunk_haplotype_size must be positive when provided, got {chunk_haplotype_size}"
            )
        if chunk_sample_size is not None and int(chunk_sample_size) <= 0:
            raise ValueError(
                f"chunk_sample_size must be positive when provided, got {chunk_sample_size}"
            )
        if compression_level < 0:
            raise ValueError(f"compression_level must be >= 0, got {compression_level}")

        ref_arr = _normalize_optional_str_array(ref, n_variants, "ref", default=".")
        alt_arr = _normalize_optional_str_array(alt, n_variants, "alt", default=".")
        vid_arr = _normalize_optional_str_array(vid, n_variants, "vid", default=".")
        real_vids = [str(text).strip() for text in vid_arr.tolist() if str(text).strip() not in {"", "."}]
        if len(set(real_vids)) != len(real_vids):
            raise ValueError("non-placeholder variant IDs must be unique within a chromosome")

        genetic_arr: np.ndarray | None = None
        if genetic_positions is not None:
            genetic_arr = np.asarray(genetic_positions, dtype=np.float64)
            if genetic_arr.ndim != 1 or genetic_arr.shape[0] != n_variants:
                raise ValueError(
                    "genetic_positions must be shape (V,), got "
                    f"shape={genetic_arr.shape}, expected ({n_variants},)"
                )
            if not np.all(np.isfinite(genetic_arr)):
                raise ValueError("genetic_positions must contain finite values")

        populations_arr: np.ndarray | None = None
        if populations is not None:
            populations_raw = np.asarray(populations, dtype=object)
            if populations_raw.ndim != 1:
                raise ValueError(
                    "populations must be 1D, got "
                    f"shape={populations_raw.shape}"
                )
            populations_arr = _to_unicode_array(populations_raw)
            if populations_arr.ndim != 1 or populations_arr.shape[0] != n_samples:
                raise ValueError(
                    "populations must be shape (N,), got "
                    f"shape={populations_arr.shape}, expected ({n_samples},)"
                )

        haps_u8 = haps.astype(np.uint8, copy=False)
        has_missing = bool(np.any(haps_u8 == _MISSING_SENTINEL))

        self._validate_sample_consistency(sample_ids_arr, chrom_key)

        compressor = Zstd(level=int(compression_level))
        n_haplotypes = int(haps_u8.shape[0])
        chunk_layout = resolve_matrix_chunk_layout(
            n_haplotypes=n_haplotypes,
            n_samples=n_samples,
            n_variants=n_variants,
            chunk_variant_size=int(chunk_variant_size),
            chunk_haplotype_size=chunk_haplotype_size,
            chunk_sample_size=chunk_sample_size,
        )
        variant_chunk = int(chunk_layout.variant)
        hap_chunk = int(chunk_layout.haplotype)
        sample_chunk_geno = int(chunk_layout.sample)

        hap_chunks = (hap_chunk, variant_chunk)
        geno_chunks = (sample_chunk_geno, variant_chunk)

        try:
            chrom_group = self._root.create_group(chrom_key)

            chrom_group.create_dataset(
                "haplotypes",
                data=haps_u8,
                dtype=np.uint8,
                chunks=hap_chunks,
                compressor=compressor,
                overwrite=False,
            )

            if store_genotypes:
                genotypes = _haplotypes_to_genotypes(haps_u8)
                chrom_group.create_dataset(
                    "genotypes",
                    data=genotypes,
                    dtype=np.uint8,
                    chunks=geno_chunks,
                    compressor=compressor,
                    overwrite=False,
                )

            variant_group = chrom_group.create_group("variants")
            variant_group.create_dataset(
                "pos",
                data=variant_pos_arr,
                dtype=np.int64,
                chunks=(variant_chunk,),
                compressor=compressor,
                overwrite=False,
            )
            _create_utf8_array(
                variant_group,
                "ref",
                ref_arr,
                chunk_size=variant_chunk,
                compressor=compressor,
            )
            _create_utf8_array(
                variant_group,
                "alt",
                alt_arr,
                chunk_size=variant_chunk,
                compressor=compressor,
            )
            _create_utf8_array(
                variant_group,
                "vid",
                vid_arr,
                chunk_size=variant_chunk,
                compressor=compressor,
            )
            if genetic_arr is not None:
                variant_group.create_dataset(
                    "genetic_pos",
                    data=genetic_arr,
                    dtype=np.float64,
                    chunks=(variant_chunk,),
                    compressor=compressor,
                    overwrite=False,
                )

            sample_group = chrom_group.create_group("samples")
            sample_chunk = int(min(max(1, n_samples), 8192))
            _create_utf8_array(
                sample_group,
                "ids",
                sample_ids_arr,
                chunk_size=sample_chunk,
                compressor=compressor,
            )
            if populations_arr is not None:
                _create_utf8_array(
                    sample_group,
                    "population",
                    populations_arr,
                    chunk_size=sample_chunk,
                    compressor=compressor,
                )

            attrs_payload = dict(attrs or {})
            source = attrs_payload.pop("source", None)
            build = attrs_payload.pop("build", None)
            full_matrix_chunks = hap_chunk == n_haplotypes and (
                not store_genotypes or sample_chunk_geno == n_samples
            )
            layout_version = 1 if full_matrix_chunks else 2
            chrom_group.attrs.update(
                build_chrom_attrs(
                    n_samples=n_samples,
                    n_haplotypes=int(haps_u8.shape[0]),
                    n_variants=n_variants,
                    missing_sentinel=_MISSING_SENTINEL,
                    has_missing=has_missing,
                    compression_level=int(compression_level),
                    chunk_variant_size=int(variant_chunk),
                    chunk_haplotype_size=int(hap_chunk),
                    chunk_sample_size=(
                        int(sample_chunk_geno) if store_genotypes else None
                    ),
                    layout_version=int(layout_version),
                    source=source,
                    build=build,
                    extra=attrs_payload or None,
                )
            )
        except Exception:
            if chrom_key in self._root:
                del self._root[chrom_key]
            raise
        self._variant_index_cache.pop(chrom_key, None)
        self._resolved_variant_alias_cache.pop(chrom_key, None)
        self._missing_variant_alias_cache.pop(chrom_key, None)
        self._ambiguous_variant_alias_cache.pop(chrom_key, None)

    def delete_chromosome(self, chrom: str | int) -> None:
        """Permanently delete one chromosome group from a writable store."""

        self._require_write_mode()
        chrom_key = self._chrom_key(chrom)
        if chrom_key not in self._root:
            raise KeyError(f"chromosome {chrom_key!r} not found; cannot delete")
        del self._root[chrom_key]
        self._variant_index_cache.pop(chrom_key, None)
        self._resolved_variant_alias_cache.pop(chrom_key, None)
        self._missing_variant_alias_cache.pop(chrom_key, None)
        self._ambiguous_variant_alias_cache.pop(chrom_key, None)

    def haplotypes(
        self,
        chrom: str | int,
        variant_slice: slice | tuple[int, int] | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Read phased haplotypes for one chromosome.

        Returns a ``uint8`` array of shape ``(2 * N, V)``, where ``N`` is the
        number of selected diploid samples and ``V`` is the number of selected
        variants. A sample mask selects both haplotypes for each sample.
        """

        chrom_group = self._require_chromosome(chrom)
        arr = chrom_group["haplotypes"]

        variant_sel = _normalize_variant_slice(variant_slice, arr.shape[1])
        if sample_mask is None:
            return np.asarray(arr[:, variant_sel], dtype=np.uint8)

        mask = _validate_sample_mask(sample_mask, self.n_samples(chrom))
        n_selected = int(np.count_nonzero(mask))
        n_total = int(mask.shape[0])

        # When selecting >50% of samples, contiguous read + numpy mask is faster
        # than zarr oindex fancy indexing which does per-chunk random access.
        if n_total > 0 and n_selected > n_total // 2:
            full = np.asarray(arr[:, variant_sel], dtype=np.uint8)
            hap_mask = np.empty(full.shape[0], dtype=bool)
            hap_mask[0::2] = mask
            hap_mask[1::2] = mask
            return full[hap_mask]

        sample_indices = np.flatnonzero(mask)
        hap_indices = np.empty(sample_indices.size * 2, dtype=np.int64)
        hap_indices[0::2] = 2 * sample_indices
        hap_indices[1::2] = (2 * sample_indices) + 1
        return np.asarray(arr.oindex[hap_indices, variant_sel], dtype=np.uint8)

    def genotypes(
        self,
        chrom: str | int,
        variant_slice: slice | tuple[int, int] | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Read diploid alternate-allele dosages for one chromosome.

        Returns a ``uint8`` array of shape ``(N, V)``. Dosages are computed
        from paired haplotypes when a materialized genotype dataset is absent.
        """

        chrom_group = self._require_chromosome(chrom)
        variant_sel = _normalize_variant_slice(variant_slice, self.n_variants(chrom))

        if "genotypes" in chrom_group:
            if sample_mask is None:
                return np.asarray(chrom_group["genotypes"][:, variant_sel], dtype=np.uint8)
            mask = _validate_sample_mask(sample_mask, self.n_samples(chrom))
            n_selected = int(np.count_nonzero(mask))
            n_total = int(mask.shape[0])
            if n_total > 0 and n_selected > n_total // 2:
                full = np.asarray(chrom_group["genotypes"][:, variant_sel], dtype=np.uint8)
                return full[mask]
            sample_indices = np.flatnonzero(mask)
            return np.asarray(chrom_group["genotypes"].oindex[sample_indices, variant_sel], dtype=np.uint8)

        haps = self.haplotypes(chrom, variant_slice=variant_sel, sample_mask=sample_mask)
        return _haplotypes_to_genotypes(haps)

    def iter_blocks(
        self,
        chrom: str | int,
        *,
        axis: str = "variants",
        block_size: int | None = None,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        sample_indices: Sequence[int] | np.ndarray | None = None,
        sample_mask: np.ndarray | None = None,
        data: str = "genotypes",
        dtype: str | np.dtype = "uint8",
    ) -> Iterator[HapStoreBlock]:
        """Yield variant blocks in caller order with aligned metadata.

        Sparse explicit indices are read as chunk-local spans. Yielded blocks
        still follow ``block_size`` and preserve duplicate indices. For sparse
        blocks, ``start`` and ``stop`` are the minimum index and one past the
        maximum index; only ``variant_indices`` are present in ``values``.

        Genotype blocks have shape ``(N, V)`` and haplotype blocks have shape
        ``(2 * N, V)``. Genotypes are computed from paired haplotypes when a
        materialized genotype dataset is unavailable.
        """
        axis_name = str(axis).lower()
        if axis_name != "variants":
            raise ValueError(f"axis must be 'variants', got {axis!r}")
        data_name = str(data).lower()
        if data_name not in {"genotypes", "haplotypes"}:
            raise ValueError(f"data must be one of 'genotypes' or 'haplotypes', got {data!r}")
        if sample_indices is not None and sample_mask is not None:
            raise ValueError("pass at most one of sample_indices or sample_mask")

        chrom_group = self._require_chromosome(chrom)
        sample_source = sample_mask if sample_mask is not None else sample_indices
        sample_idx = self._resolve_sample_indices(chrom=chrom, sample_indices=sample_source)
        arr_name = "genotypes" if data_name == "genotypes" and "genotypes" in chrom_group else "haplotypes"
        variant_chunk = int(chrom_group[arr_name].chunks[1])
        n_variants = self.n_variants(chrom)
        block_len = _normalize_block_size(block_size, default=variant_chunk)

        if variant_indices is None:
            blocks = (
                np.arange(start, min(start + block_len, n_variants), dtype=np.int64)
                for start in range(0, n_variants, block_len)
            )
        else:
            requested = np.asarray(variant_indices, dtype=np.int64)
            if requested.ndim != 1:
                raise ValueError(f"variant_indices must be 1D, got shape={requested.shape}")
            if requested.size and (int(np.min(requested)) < 0 or int(np.max(requested)) >= n_variants):
                raise IndexError(f"variant index out of bounds for n_variants={n_variants}")
            blocks = (
                np.asarray(requested[start : start + block_len], dtype=np.int64)
                for start in range(0, requested.size, block_len)
            )

        for block_indices in blocks:
            if block_indices.size == 0:
                continue
            block_start = int(np.min(block_indices))
            block_stop = int(np.max(block_indices)) + 1
            values = self._read_block_values(
                chrom_group=chrom_group,
                chrom=chrom,
                data=data_name,
                variant_indices=block_indices,
                sample_indices=sample_idx,
                dtype=np.dtype(dtype),
            )
            metadata = self._read_variant_metadata_block(
                chrom_group=chrom_group,
                variant_indices=block_indices,
            )
            yield HapStoreBlock(
                values=values,
                variant_indices=np.asarray(block_indices, dtype=np.int64).copy(),
                positions=metadata["positions"],
                ref=metadata["ref"],
                alt=metadata["alt"],
                vid=metadata["vid"],
                start=block_start,
                stop=block_stop,
                data=data_name,
                sample_indices=np.asarray(sample_idx, dtype=np.int64).copy(),
                genetic_pos=metadata["genetic_pos"],
            )

    def variant_maf(
        self,
        chrom: str | int,
        *,
        sample_indices: Sequence[int] | np.ndarray | None = None,
        sample_mask: np.ndarray | None = None,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        block_size: int | None = None,
        min_maf: float | None = None,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Compute per-variant alternate-allele frequency and MAF.

        MAF is the minimum of the alternate-allele frequency and its
        complement. Missing genotypes are excluded from frequency denominators;
        variants with no called genotypes receive NaN frequency and MAF.
        ``called_genotype_count`` counts nonmissing diploid genotypes and
        ``called_allele_count`` is twice that count.

        The returned dictionary always contains variant indices, frequencies,
        MAFs, alternate-allele counts, and called counts. When
        ``include_metadata=True``, it also contains positions, alleles, IDs,
        and genetic positions. ``min_maf`` filters the returned rows while
        preserving their requested order.
        """
        threshold = _normalize_min_maf(min_maf)
        chunks: dict[str, list[np.ndarray]] = {
            "variant_indices": [],
            "allele_frequency": [],
            "maf": [],
            "alt_allele_count": [],
            "called_genotype_count": [],
            "called_allele_count": [],
        }
        if include_metadata:
            chunks.update(
                {
                    "positions": [],
                    "ref": [],
                    "alt": [],
                    "vid": [],
                    "genetic_pos": [],
                }
            )
        saw_genetic = False

        for block in self.iter_blocks(
            chrom,
            block_size=block_size,
            variant_indices=variant_indices,
            sample_indices=sample_indices,
            sample_mask=sample_mask,
            data="genotypes",
            dtype="uint8",
        ):
            values = np.asarray(block.values, dtype=np.uint8)
            called = values != _MISSING_SENTINEL
            called_genotypes = np.count_nonzero(called, axis=0).astype(np.int64, copy=False)
            alt_counts = np.where(called, values, 0).sum(axis=0, dtype=np.int64)
            called_alleles = (2 * called_genotypes).astype(np.int64, copy=False)
            af = np.full(values.shape[1], np.nan, dtype=np.float64)
            np.divide(
                alt_counts,
                called_alleles,
                out=af,
                where=called_alleles > 0,
                casting="unsafe",
            )
            maf = np.minimum(af, 1.0 - af)

            chunks["variant_indices"].append(np.asarray(block.variant_indices, dtype=np.int64))
            chunks["allele_frequency"].append(af)
            chunks["maf"].append(maf)
            chunks["alt_allele_count"].append(np.asarray(alt_counts, dtype=np.int64))
            chunks["called_genotype_count"].append(np.asarray(called_genotypes, dtype=np.int64))
            chunks["called_allele_count"].append(np.asarray(called_alleles, dtype=np.int64))
            if include_metadata:
                chunks["positions"].append(np.asarray(block.positions, dtype=np.int64))
                chunks["ref"].append(np.asarray(block.ref, dtype=object))
                chunks["alt"].append(np.asarray(block.alt, dtype=object))
                chunks["vid"].append(np.asarray(block.vid, dtype=object))
                if block.genetic_pos is not None:
                    saw_genetic = True
                    chunks["genetic_pos"].append(np.asarray(block.genetic_pos, dtype=np.float64))
                else:
                    chunks["genetic_pos"].append(np.full(block.variant_indices.size, np.nan, dtype=np.float64))

        table = _concatenate_maf_chunks(chunks, include_metadata=include_metadata, saw_genetic=saw_genetic)
        if threshold is not None:
            keep = np.isfinite(table["maf"]) & (table["maf"] >= threshold)
            table = _filter_table_rows(table, keep)
        return table

    def non_monomorphic_variant_indices(
        self,
        chrom: str | int,
        *,
        sample_indices: Sequence[int] | np.ndarray | None = None,
        sample_mask: np.ndarray | None = None,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        block_size: int | None = None,
        min_maf: float = 1e-12,
    ) -> np.ndarray:
        """Return variant indices with MAF at least ``min_maf``."""
        table = self.variant_maf(
            chrom,
            sample_indices=sample_indices,
            sample_mask=sample_mask,
            variant_indices=variant_indices,
            block_size=block_size,
            min_maf=min_maf,
            include_metadata=False,
        )
        return np.asarray(table["variant_indices"], dtype=np.int64)

    def read_dosages(
        self,
        chrom: str | int,
        *,
        variant_ids: Sequence[str] | None = None,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        sample_indices: Sequence[int] | np.ndarray | None = None,
        dtype: str | np.dtype = "float32",
        missing: str = "nan",
        workers: int | str = "auto",
        backend: str = "auto",
        assume_no_missing: bool = False,
    ) -> np.ndarray:
        """Read diploid alternate-allele dosages for selected samples and variants.

        Exactly one of ``variant_ids`` or ``variant_indices`` must be provided.
        Requested sample and variant order is preserved, including duplicate indices.

        Args:
            chrom: Chromosome label.
            variant_ids: Variant identifiers to read.
            variant_indices: Zero-based variant indices to read.
            sample_indices: Zero-based sample indices or a boolean sample mask.
                When omitted, all samples are read.
            dtype: Output array data type. A floating type is required when
                ``missing="nan"``.
            missing: Represent missing genotypes as ``numpy.nan`` or retain the
                HapStore sentinel value with ``"sentinel"``.
            workers: Positive thread count or ``"auto"``. Setting this to ``1``
                forces serial execution.
            backend: Reading backend: ``"auto"``, ``"threaded"``, or ``"serial"``.
            assume_no_missing: Skip missing-value checks for improved performance.
                The caller must ensure that the selected data contain no missing
                alleles.

        Returns:
            An array of shape ``(n_samples, n_variants)``. Dosages are read from
            materialized genotypes when available and otherwise computed from paired
            haplotypes.

        Raises:
            ValueError: If variant selection is ambiguous, an unsupported backend or
                missing-value mode is requested, or the output dtype is incompatible
                with ``missing="nan"``.
        """
        chrom_group = self._require_chromosome(chrom)
        variant_idx = self._resolve_variant_indices(
            chrom=chrom,
            variant_ids=variant_ids,
            variant_indices=variant_indices,
        )
        sample_idx = self._resolve_sample_indices(chrom=chrom, sample_indices=sample_indices)
        if variant_idx.size == 0:
            return np.empty((sample_idx.size, 0), dtype=np.dtype(dtype))
        target_dtype = np.dtype(dtype)
        missing_mode = _normalize_missing_mode(missing)
        if missing_mode == "nan" and not np.issubdtype(target_dtype, np.floating):
            raise ValueError("missing='nan' requires a floating dtype")

        backend_name = str(backend).lower()
        if backend_name not in {"auto", "threaded", "serial"}:
            raise ValueError(f"backend must be one of auto/threaded/serial, got {backend!r}")
        if _is_explicit_single_worker(workers):
            explicit_single_worker = True
        else:
            explicit_single_worker = False
        if backend_name == "serial" or explicit_single_worker:
            return self._read_dosages_serial(
                chrom=chrom,
                variant_indices=variant_idx,
                sample_indices=sample_idx,
                dtype=target_dtype,
                missing=missing_mode,
                assume_no_missing=assume_no_missing,
            )

        arr_name = "genotypes" if "genotypes" in chrom_group else "haplotypes"
        arr = chrom_group[arr_name]
        attrs = self.attrs(chrom)
        has_missing = bool(attrs.get("has_missing", True)) and not bool(assume_no_missing)
        variant_chunk = int(arr.chunks[1])
        if arr_name == "genotypes":
            row_chunk = int(arr.chunks[0])
        else:
            row_chunk = max(1, int(arr.chunks[0]) // 2)
        variant_groups = _make_read_groups(variant_idx, chunk_size=variant_chunk, limit=int(self.n_variants(chrom)))
        sample_groups = _make_read_groups(sample_idx, chunk_size=row_chunk, limit=int(self.n_samples(chrom)))
        native_variant_groups = len(variant_groups)
        native_sample_groups = len(sample_groups)
        variant_group_chunks, variant_group_source = _env_positive_int(
            ("HAPSTORE_READ_VARIANT_GROUP_CHUNKS",),
            default=4,
        )
        sample_group_chunks, sample_group_source = _env_positive_int(
            ("HAPSTORE_READ_SAMPLE_GROUP_CHUNKS",),
            default=2,
        )
        min_density, min_density_source = _env_positive_float(
            ("HAPSTORE_READ_COALESCE_MIN_DENSITY",),
            default=0.25,
        )
        assert variant_group_chunks is not None
        assert sample_group_chunks is not None
        assert min_density is not None
        variant_groups = _coalesce_read_groups(
            variant_groups,
            max_span=max(1, int(variant_chunk) * int(variant_group_chunks)),
            max_gap=max(0, int(variant_chunk)),
            min_density=float(min_density),
        )
        sample_groups = _coalesce_read_groups(
            sample_groups,
            max_span=max(1, int(row_chunk) * int(sample_group_chunks)),
            max_gap=0,
            min_density=float(min_density),
        )
        total_tasks = len(variant_groups) * len(sample_groups)
        row_factor = 1 if arr_name == "genotypes" else 2
        max_input_rows = max(int(group.stop) - int(group.start) for group in sample_groups) * row_factor
        max_input_cols = max(int(group.stop) - int(group.start) for group in variant_groups)
        max_output_rows = max(int(group.out_positions.size) for group in sample_groups)
        max_output_cols = max(int(group.out_positions.size) for group in variant_groups)
        input_itemsize = np.dtype(arr.dtype).itemsize
        # Bound worker selection using the largest tile's input, conversion,
        # and output buffers.
        estimated_tile_bytes = int(
            max_input_rows * max_input_cols * input_itemsize
            + max_output_rows * max_output_cols * row_factor * input_itemsize
            + max_output_rows * max_output_cols * target_dtype.itemsize
        )
        output_bytes = int(sample_idx.size * variant_idx.size * target_dtype.itemsize)
        worker_count, worker_policy = _resolve_read_worker_count(
            workers,
            n_tasks=total_tasks,
            n_sample_groups=len(sample_groups),
            n_variant_groups=len(variant_groups),
            n_samples=sample_idx.size,
            n_variants=variant_idx.size,
            row_chunk=row_chunk,
            variant_chunk=variant_chunk,
            dataset=arr_name,
            estimated_tile_bytes=estimated_tile_bytes,
            output_bytes=output_bytes,
        )
        out = np.empty((sample_idx.size, variant_idx.size), dtype=target_dtype)
        local = threading.local()
        path = str(self._path)
        chrom_key = self._chrom_key(chrom)
        read_sec = 0.0
        convert_sec = 0.0
        assign_sec = 0.0
        total_start = time.perf_counter()

        def read_tile(sample_group: _ReadGroup, variant_group: _ReadGroup) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
            if not hasattr(local, "root"):
                local.root = zarr.open(path, mode="r")
            local_arr = local.root[chrom_key][arr_name]
            read_start = time.perf_counter()
            if arr_name == "genotypes":
                block = np.asarray(
                    local_arr[
                        int(sample_group.start) : int(sample_group.stop),
                        int(variant_group.start) : int(variant_group.stop),
                    ],
                    dtype=np.uint8,
                )
            else:
                block = np.asarray(
                    local_arr[
                        int(2 * sample_group.start) : int(2 * sample_group.stop),
                        int(variant_group.start) : int(variant_group.stop),
                    ],
                    dtype=np.uint8,
                )
            tile_read_sec = time.perf_counter() - read_start
            convert_start = time.perf_counter()
            if arr_name == "genotypes":
                values_u8 = block[np.ix_(sample_group.offsets, variant_group.offsets)]
                values = _format_genotype_values(
                    values_u8,
                    dtype=target_dtype,
                    missing=missing_mode,
                    has_missing=has_missing,
                )
            else:
                first = block[0::2, :]
                second = block[1::2, :]
                a = first[np.ix_(sample_group.offsets, variant_group.offsets)]
                b = second[np.ix_(sample_group.offsets, variant_group.offsets)]
                values = _format_haplotype_pair_values(
                    a,
                    b,
                    dtype=target_dtype,
                    missing=missing_mode,
                    has_missing=has_missing,
                )
            tile_convert_sec = time.perf_counter() - convert_start
            return sample_group.out_positions, variant_group.out_positions, values, tile_read_sec, tile_convert_sec

        def assign_tile(tile: tuple[np.ndarray, np.ndarray, np.ndarray, float, float]) -> None:
            nonlocal read_sec, convert_sec, assign_sec
            out_rows, out_cols, values, tile_read_sec, tile_convert_sec = tile
            read_sec += float(tile_read_sec)
            convert_sec += float(tile_convert_sec)
            assign_start = time.perf_counter()
            out[np.ix_(out_rows, out_cols)] = values
            assign_sec += time.perf_counter() - assign_start

        max_in_flight_tiles = 1
        if total_tasks <= 1:
            assign_tile(read_tile(sample_groups[0], variant_groups[0]))
        else:
            max_in_flight_tiles = min(int(worker_count), int(total_tasks))
            task_iter = iter(
                (sample_group, variant_group)
                for sample_group in sample_groups
                for variant_group in variant_groups
            )
            # Read tiles concurrently and assign each completed tile into its
            # disjoint output coordinates.
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="hapstore-read") as pool:
                pending = set()
                for _ in range(max_in_flight_tiles):
                    sample_group, variant_group = next(task_iter)
                    pending.add(pool.submit(read_tile, sample_group, variant_group))
                while pending:
                    completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        assign_tile(future.result())
                        try:
                            sample_group, variant_group = next(task_iter)
                        except StopIteration:
                            continue
                        pending.add(pool.submit(read_tile, sample_group, variant_group))
        self._last_read_metrics = {
            "backend": "threaded",
            "dataset": arr_name,
            "requested_workers": str(workers),
            "workers": int(worker_count),
            **worker_policy,
            "n_samples": int(sample_idx.size),
            "n_variants": int(variant_idx.size),
            "n_unique_variants": int(np.unique(variant_idx).size),
            "row_chunk": int(row_chunk),
            "variant_chunk": int(variant_chunk),
            "native_sample_groups": int(native_sample_groups),
            "native_variant_groups": int(native_variant_groups),
            "sample_groups": int(len(sample_groups)),
            "variant_groups": int(len(variant_groups)),
            "sample_group_chunks": int(sample_group_chunks),
            "sample_group_chunks_source": sample_group_source or "default",
            "variant_group_chunks": int(variant_group_chunks),
            "variant_group_chunks_source": variant_group_source or "default",
            "coalesce_min_density": float(min_density),
            "coalesce_min_density_source": min_density_source or "default",
            "n_tiles": int(total_tasks),
            "max_in_flight_tiles": int(max_in_flight_tiles),
            "estimated_tile_mb": float(estimated_tile_bytes / (1024.0**2)),
            "output_mb": float(output_bytes / (1024.0**2)),
            "has_missing": bool(has_missing),
            "assume_no_missing": bool(assume_no_missing),
            "read_sec_sum": float(read_sec),
            "convert_sec_sum": float(convert_sec),
            "assign_sec": float(assign_sec),
            "total_sec": float(time.perf_counter() - total_start),
        }
        return out

    def _read_block_values(
        self,
        *,
        chrom_group: zarr.Group,
        chrom: str | int,
        data: str,
        variant_indices: np.ndarray,
        sample_indices: np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        n_variants = int(variant_indices.size)

        if data == "haplotypes":
            if sample_indices.size == 0:
                return np.empty((0, n_variants), dtype=dtype)
            arr = chrom_group["haplotypes"]
            hap_indices = np.empty(sample_indices.size * 2, dtype=np.int64)
            hap_indices[0::2] = 2 * sample_indices
            hap_indices[1::2] = (2 * sample_indices) + 1
            block = _read_chunk_grouped_array(
                arr,
                row_indices=hap_indices,
                variant_indices=variant_indices,
            )
            return np.asarray(block, dtype=dtype)

        if data != "genotypes":
            raise ValueError(f"data must be one of 'genotypes' or 'haplotypes', got {data!r}")
        if sample_indices.size == 0:
            return np.empty((0, n_variants), dtype=dtype)
        if "genotypes" in chrom_group:
            arr = chrom_group["genotypes"]
            block = _read_chunk_grouped_array(
                arr,
                row_indices=sample_indices,
                variant_indices=variant_indices,
            )
            return np.asarray(block, dtype=dtype)

        haps = self._read_block_values(
            chrom_group=chrom_group,
            chrom=chrom,
            data="haplotypes",
            variant_indices=variant_indices,
            sample_indices=sample_indices,
            dtype=np.dtype("uint8"),
        )
        return np.asarray(_haplotypes_to_genotypes(haps), dtype=dtype)

    def _read_variant_metadata_block(
        self,
        *,
        chrom_group: zarr.Group,
        variant_indices: np.ndarray,
    ) -> dict[str, Any]:
        variants = chrom_group["variants"]
        n_variants = int(variant_indices.size)
        groups = _plan_variant_read_groups(
            variant_indices,
            chunk_size=int(variants["pos"].chunks[0]),
            limit=int(variants["pos"].shape[0]),
        )
        positions = np.empty(n_variants, dtype=np.int64)
        ref = np.empty(n_variants, dtype=object)
        alt = np.empty(n_variants, dtype=object)
        vid = np.empty(n_variants, dtype=object)
        genetic = np.empty(n_variants, dtype=np.float64) if "genetic_pos" in variants else None
        for group in groups:
            span = slice(int(group.start), int(group.stop))
            out_pos = group.out_positions
            offsets = group.offsets
            positions[out_pos] = np.asarray(variants["pos"][span], dtype=np.int64)[offsets]
            ref[out_pos] = _to_unicode_array(np.asarray(variants["ref"][span], dtype=object))[offsets]
            alt[out_pos] = _to_unicode_array(np.asarray(variants["alt"][span], dtype=object))[offsets]
            vid[out_pos] = _to_unicode_array(np.asarray(variants["vid"][span], dtype=object))[offsets]
            if genetic is not None:
                genetic[out_pos] = np.asarray(variants["genetic_pos"][span], dtype=np.float64)[offsets]
        return {
            "positions": np.asarray(positions, dtype=np.int64),
            "ref": tuple(str(x) for x in ref),
            "alt": tuple(str(x) for x in alt),
            "vid": tuple(str(x) for x in vid),
            "genetic_pos": genetic,
        }

    def read_dosage_batch(
        self,
        chrom: str | int,
        requests: Sequence[HapStoreDosageRequest | dict[str, Any] | tuple[Hashable, Sequence[str] | Sequence[int] | np.ndarray]],
        *,
        sample_indices: Sequence[int] | np.ndarray | None = None,
        dtype: str | np.dtype = "float32",
        missing: str = "nan",
        workers: int | str = "auto",
        backend: str = "auto",
        assume_no_missing: bool = False,
    ) -> HapStoreDosageBatch:
        """Read multiple dosage requests in one call.

        Each request supplies a hashable request ID and exactly one sequence of
        variant IDs or zero-based variant indices. Requests may be represented
        as ``HapStoreDosageRequest`` objects, mappings, or
        ``(request_id, variants)`` tuples.

        Variants resolving to the same stored index are read once into a sorted
        union matrix. ``request_columns`` maps that matrix back to each
        request's original variant order, including duplicates;
        ``window_matrix()`` applies this mapping. Request IDs must be unique.

        Args:
            chrom: Chromosome label.
            requests: A sequence of dosage requests.
            sample_indices: Zero-based sample indices or a boolean sample mask.
                When omitted, all samples are read.
            dtype: Output array data type. A floating type is required when
                ``missing="nan"``.
            missing: Represent missing genotypes as ``numpy.nan`` or retain the
                HapStore sentinel value with ``"sentinel"``.
            workers: Positive thread count or ``"auto"``. Setting this to ``1``
                forces serial execution.
            backend: Reading backend: ``"auto"``, ``"threaded"``, or ``"serial"``.
            assume_no_missing: Skip missing-value checks for improved performance.
                The caller must ensure that the selected data contain no missing
                alleles.
        Returns:
            A batch containing the union dosage matrix, sorted variant indices,
            request-to-column mappings, resolved variant IDs, and read metrics.

        Raises:
            ValueError: If request IDs are duplicated, variant selection is
                ambiguous, an unsupported backend or missing-value mode is
                requested, or the output dtype is incompatible with
                ``missing="nan"``.
        """

        request_specs = [self._parse_dosage_request(request, i) for i, request in enumerate(requests)]
        if not request_specs:
            sample_idx = self._resolve_sample_indices(chrom=chrom, sample_indices=sample_indices)
            return HapStoreDosageBatch(
                matrix=np.empty((sample_idx.size, 0), dtype=np.dtype(dtype)),
                variant_indices=np.empty(0, dtype=np.int64),
                request_columns={},
                request_ids=(),
                metrics={"backend": "empty"},
            )
        seen_request_ids: set[Hashable] = set()
        for request_id, _variant_ids, _variant_indices in request_specs:
            try:
                if request_id in seen_request_ids:
                    raise ValueError(f"request IDs must be unique; duplicate={request_id!r}")
                seen_request_ids.add(request_id)
            except TypeError as exc:
                raise TypeError(f"request ID must be hashable, got {request_id!r}") from exc
        requested_ids = [
            str(variant_id)
            for _request_id, variant_ids, _variant_indices in request_specs
            if variant_ids is not None
            for variant_id in variant_ids
        ]
        if requested_ids:
            self._resolve_variant_ids_chunked(chrom=chrom, variant_ids=requested_ids)
        parsed: list[tuple[Hashable, np.ndarray, tuple[str, ...] | None]] = []
        for request_id, variant_ids, variant_indices in request_specs:
            indices = self._resolve_variant_indices(
                chrom=chrom,
                variant_ids=variant_ids,
                variant_indices=variant_indices,
            )
            ids_tuple = tuple(str(value) for value in variant_ids) if variant_ids is not None else None
            parsed.append((request_id, indices, ids_tuple))
        seen: set[int] = set()
        union_indices: list[int] = []
        for _request_id, indices, _ids in parsed:
            for idx in indices:
                idx_int = int(idx)
                if idx_int in seen:
                    continue
                seen.add(idx_int)
                union_indices.append(idx_int)
        union_indices = sorted(union_indices)
        union_indices_arr = np.asarray(union_indices, dtype=np.int64)
        union_pos = {idx: i for i, idx in enumerate(union_indices)}
        request_columns = {
            request_id: np.asarray([union_pos[int(idx)] for idx in indices], dtype=np.int64)
            for request_id, indices, _ids in parsed
        }
        matrix = self.read_dosages(
            chrom,
            variant_indices=union_indices_arr,
            sample_indices=sample_indices,
            dtype=dtype,
            missing=missing,
            workers=workers,
            backend=backend,
            assume_no_missing=assume_no_missing,
        )
        metadata = self._read_variant_metadata_block(
            chrom_group=self._require_chromosome(chrom),
            variant_indices=union_indices_arr,
        )
        chrom_alias = self._chrom_key(chrom).removeprefix("chr")
        union_variant_ids = tuple(
            str(vid) if str(vid) else f"{chrom_alias}:{int(pos)}"
            for vid, pos in zip(metadata["vid"], metadata["positions"])
        )
        return HapStoreDosageBatch(
            matrix=matrix,
            variant_indices=union_indices_arr,
            request_columns=request_columns,
            request_ids=tuple(request_id for request_id, _indices, _ids in parsed),
            variant_ids=union_variant_ids,
            metrics=self.last_read_metrics,
        )

    def variant_positions(self, chrom: str | int) -> np.ndarray:
        """Return one-based physical positions in variant-row order."""
        chrom_group = self._require_chromosome(chrom)
        return np.asarray(chrom_group["variants"]["pos"][:], dtype=np.int64)

    def genetic_positions(self, chrom: str | int) -> np.ndarray | None:
        """Return genetic positions in cM, or ``None`` when unavailable."""
        chrom_group = self._require_chromosome(chrom)
        variants = chrom_group["variants"]
        if "genetic_pos" not in variants:
            return None
        return np.asarray(variants["genetic_pos"][:], dtype=np.float64)

    def sample_ids(self, chrom: str | int) -> list[str]:
        """Return diploid sample IDs in stored row order."""
        chrom_group = self._require_chromosome(chrom)
        return _to_unicode_array(np.asarray(chrom_group["samples"]["ids"][:], dtype=object)).tolist()

    def variant_info(self, chrom: str | int) -> dict[str, Any]:
        """Return physical positions, IDs, alleles, and optional genetic positions."""
        chrom_group = self._require_chromosome(chrom)
        variants = chrom_group["variants"]
        genetic = None
        if "genetic_pos" in variants:
            genetic = np.asarray(variants["genetic_pos"][:], dtype=np.float64)

        return {
            "pos": np.asarray(variants["pos"][:], dtype=np.int64),
            "vid": _to_unicode_array(np.asarray(variants["vid"][:], dtype=object)).tolist(),
            "ref": _to_unicode_array(np.asarray(variants["ref"][:], dtype=object)).tolist(),
            "alt": _to_unicode_array(np.asarray(variants["alt"][:], dtype=object)).tolist(),
            "genetic_pos": genetic,
        }

    def variant_index_map(self, chrom: str | int) -> dict[str, int]:
        """Map unambiguous variant ID and chromosome-position aliases to rows."""
        return dict(self._variant_index_map(chrom))

    def _variant_index_map(self, chrom: str | int) -> dict[str, int]:
        chrom_key = self._chrom_key(chrom)
        if chrom_key not in self._variant_index_cache:
            chrom_group = self._require_chromosome(chrom)
            variants = chrom_group["variants"]
            chrom_alias = str(chrom_key).removeprefix("chr")
            out: dict[str, int] = {}
            ambiguous: set[str] = set()

            def add_alias(alias: str, index: int) -> None:
                if alias in ambiguous:
                    return
                previous = out.get(alias)
                if previous is None or previous == int(index):
                    out[alias] = int(index)
                    return
                del out[alias]
                ambiguous.add(alias)

            vid_arr = _to_unicode_array(np.asarray(variants["vid"][:], dtype=object))
            pos_arr = np.asarray(variants["pos"][:], dtype=np.int64)
            for i, vid in enumerate(vid_arr):
                text = str(vid).strip()
                if text not in {"", "."}:
                    add_alias(text, int(i))
            for i, pos in enumerate(pos_arr):
                add_alias(f"chr{chrom_alias}:{int(pos)}", int(i))
                add_alias(f"{chrom_alias}:{int(pos)}", int(i))
            self._variant_index_cache[chrom_key] = out
            self._ambiguous_variant_alias_cache[chrom_key] = ambiguous
        return self._variant_index_cache[chrom_key]

    def _scan_variant_aliases(
        self,
        *,
        chrom: str | int,
        aliases: set[str],
    ) -> tuple[dict[str, int], set[str]]:
        """Scan variant metadata in storage chunks for only requested aliases."""
        if not aliases:
            return {}, set()
        chrom_key = self._chrom_key(chrom)
        chrom_alias = chrom_key.removeprefix("chr")
        variants = self._require_chromosome(chrom)["variants"]
        vid_array = variants["vid"]
        pos_array = variants["pos"]
        position_prefixes = (f"{chrom_alias}:", f"chr{chrom_alias}:")
        need_positions = any(alias.startswith(position_prefixes) for alias in aliases)
        chunk_size = int(vid_array.chunks[0])
        if need_positions:
            chunk_size = min(chunk_size, int(pos_array.chunks[0]))
        found: dict[str, int] = {}
        ambiguous: set[str] = set()

        def add_alias(alias: str, index: int) -> None:
            if alias not in aliases or alias in ambiguous:
                return
            previous = found.get(alias)
            if previous is None or previous == int(index):
                found[alias] = int(index)
                return
            found.pop(alias, None)
            ambiguous.add(alias)

        for start in range(0, int(vid_array.shape[0]), chunk_size):
            stop = min(int(vid_array.shape[0]), start + chunk_size)
            vids = _to_unicode_array(np.asarray(vid_array[start:stop], dtype=object))
            positions = (
                np.asarray(pos_array[start:stop], dtype=np.int64)
                if need_positions
                else None
            )
            for offset, vid in enumerate(vids):
                index = int(start + offset)
                text = str(vid).strip()
                if text not in {"", "."}:
                    add_alias(text, index)
                if positions is not None:
                    pos = int(positions[offset])
                    add_alias(f"chr{chrom_alias}:{pos}", index)
                    add_alias(f"{chrom_alias}:{pos}", index)
        return found, ambiguous

    def _resolve_variant_ids_chunked(
        self,
        *,
        chrom: str | int,
        variant_ids: Sequence[str],
    ) -> np.ndarray:
        """Resolve IDs with chunk-bounded scans and request-sized caches."""
        requested = [str(variant_id) for variant_id in variant_ids]
        if not requested:
            return np.empty(0, dtype=np.int64)
        chrom_key = self._chrom_key(chrom)
        if chrom_key in self._variant_index_cache:
            lookup = self._variant_index_cache[chrom_key]
            ambiguous = self._ambiguous_variant_alias_cache.get(chrom_key, set())
        else:
            lookup = self._resolved_variant_alias_cache.setdefault(chrom_key, {})
            ambiguous = self._ambiguous_variant_alias_cache.setdefault(chrom_key, set())
            known_missing = self._missing_variant_alias_cache.setdefault(chrom_key, set())
            unknown = set(requested) - set(lookup) - ambiguous - known_missing
            if unknown:
                found, newly_ambiguous = self._scan_variant_aliases(
                    chrom=chrom,
                    aliases=unknown,
                )
                lookup.update(found)
                ambiguous.update(newly_ambiguous)
                known_missing.update(unknown - set(found) - newly_ambiguous)

        requested_ambiguous = [alias for alias in requested if alias in ambiguous]
        if requested_ambiguous:
            raise ValueError(
                f"ambiguous variant alias on HapStore chromosome {chrom_key}: "
                f"first={requested_ambiguous[0]!r}; use variant_indices instead"
            )
        missing = [variant_id for variant_id in requested if variant_id not in lookup]
        if missing:
            raise ValueError(
                f"{len(missing)} variant IDs missing from HapStore chromosome {chrom_key}; "
                f"first={missing[:5]}"
            )
        return np.asarray([lookup[variant_id] for variant_id in requested], dtype=np.int64)

    def _resolve_variant_indices(
        self,
        *,
        chrom: str | int,
        variant_ids: Sequence[str] | None,
        variant_indices: Sequence[int] | np.ndarray | None,
    ) -> np.ndarray:
        """Resolve variant IDs or indices to a 1D array of variant indices."""

        if (variant_ids is None) == (variant_indices is None):
            raise ValueError("pass exactly one of variant_ids or variant_indices")
        if variant_ids is not None:
            arr = self._resolve_variant_ids_chunked(chrom=chrom, variant_ids=variant_ids)
        else:
            arr = np.asarray(variant_indices, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"variant indices must be 1D, got shape={arr.shape}")
        n_variants = self.n_variants(chrom)
        if arr.size and (int(np.min(arr)) < 0 or int(np.max(arr)) >= n_variants):
            raise IndexError(f"variant index out of bounds for n_variants={n_variants}")
        return arr

    def _resolve_sample_indices(
        self,
        *,
        chrom: str | int,
        sample_indices: Sequence[int] | np.ndarray | None,
    ) -> np.ndarray:
        n_samples = self.n_samples(chrom)
        if sample_indices is None:
            return np.arange(n_samples, dtype=np.int64)
        arr = np.asarray(sample_indices)
        if arr.dtype == np.bool_:
            if arr.ndim != 1 or arr.shape[0] != n_samples:
                raise ValueError(f"sample mask must be shape ({n_samples},), got {arr.shape}")
            out = np.flatnonzero(arr).astype(np.int64, copy=False)
        else:
            out = np.asarray(arr, dtype=np.int64)
        if out.ndim != 1:
            raise ValueError(f"sample indices must be 1D, got shape={out.shape}")
        if out.size and (int(np.min(out)) < 0 or int(np.max(out)) >= n_samples):
            raise IndexError(f"sample index out of bounds for n_samples={n_samples}")
        return out

    def _parse_dosage_request(
        self,
        request: HapStoreDosageRequest | dict[str, Any] | tuple[Hashable, Sequence[str] | Sequence[int] | np.ndarray],
        fallback_index: int,
    ) -> tuple[Hashable, Sequence[str] | None, Sequence[int] | np.ndarray | None]:
        if isinstance(request, HapStoreDosageRequest):
            request_id = request.request_id
            variant_ids = request.variant_ids
            variant_indices = request.variant_indices
        elif isinstance(request, dict):
            request_id = request.get("request_id", request.get("id", fallback_index))
            variant_ids = request.get("variant_ids")
            variant_indices = request.get("variant_indices")
        elif isinstance(request, tuple) and len(request) == 2:
            request_id, variants = request
            arr = np.asarray(variants)
            if arr.dtype.kind in {"U", "S", "O"}:
                variant_ids = [str(v) for v in variants]
                variant_indices = None
            else:
                variant_ids = None
                variant_indices = variants
        else:
            raise TypeError(f"unsupported dosage request: {request!r}")
        return request_id, variant_ids, variant_indices

    def _read_dosages_serial(
        self,
        *,
        chrom: str | int,
        variant_indices: np.ndarray,
        sample_indices: np.ndarray,
        dtype: np.dtype,
        missing: str,
        assume_no_missing: bool,
    ) -> np.ndarray:
        """Read and format dosages serially while preserving requested order.

        ``assume_no_missing`` bypasses missing-value checks; callers using it
        are responsible for ensuring that the selected data are complete.
        """

        chrom_group = self._require_chromosome(chrom)
        dataset = "genotypes" if "genotypes" in chrom_group else "haplotypes"
        arr = chrom_group[dataset]
        variant_groups = _plan_variant_read_groups(
            variant_indices,
            chunk_size=int(arr.chunks[1]),
            limit=int(arr.shape[1]),
        )
        span_start = int(np.min(variant_indices))
        span_stop = int(np.max(variant_indices)) + 1
        read_start = time.perf_counter()
        values = self._read_block_values(
            chrom_group=chrom_group,
            chrom=chrom,
            data="genotypes",
            variant_indices=variant_indices,
            sample_indices=sample_indices,
            dtype=np.dtype("uint8"),
        )
        read_sec = time.perf_counter() - read_start
        format_start = time.perf_counter()
        has_missing = bool(self.attrs(chrom).get("has_missing", True)) and not bool(assume_no_missing)
        out = _format_genotype_values(
            values,
            dtype=dtype,
            missing=missing,
            has_missing=has_missing,
        )
        format_sec = time.perf_counter() - format_start
        read_span_variants = int(sum(int(group.stop) - int(group.start) for group in variant_groups))
        self._last_read_metrics = {
            "backend": "serial",
            "dataset": dataset,
            "workers": 1,
            "n_samples": int(sample_indices.size),
            "n_variants": int(variant_indices.size),
            "n_unique_variants": int(np.unique(variant_indices).size),
            "span_variants": int(span_stop - span_start),
            "read_span_variants": read_span_variants,
            "max_read_span_variants": int(
                max(int(group.stop) - int(group.start) for group in variant_groups)
            ),
            "variant_groups": int(len(variant_groups)),
            "has_missing": has_missing,
            "assume_no_missing": bool(assume_no_missing),
            "read_sec_sum": float(read_sec),
            "assign_sec": float(format_sec),
            "total_sec": float(read_sec + format_sec),
        }
        return out

    def n_samples(self, chrom: str | int) -> int:
        """Return the number of diploid samples."""
        chrom_group = self._require_chromosome(chrom)
        return int(chrom_group.attrs["n_samples"])

    def n_variants(self, chrom: str | int) -> int:
        """Return the number of variant rows."""
        chrom_group = self._require_chromosome(chrom)
        return int(chrom_group.attrs["n_variants"])

    def n_haplotypes(self, chrom: str | int) -> int:
        """Return the number of phased haplotypes."""
        chrom_group = self._require_chromosome(chrom)
        return int(chrom_group.attrs["n_haplotypes"])

    def attrs(self, chrom: str | int) -> dict:
        """Return a copy of the chromosome-group attributes."""
        chrom_group = self._require_chromosome(chrom)
        return dict(chrom_group.attrs.asdict())

    def has_chromosome(self, chrom: str | int) -> bool:
        """Return whether the normalized chromosome group exists."""
        return self._chrom_key(chrom) in self._root

    def compute_windows(
        self,
        chrom: str | int,
        window_size_cm: float,
        store: bool = False,
    ) -> np.ndarray:
        """Compute non-overlapping genetic windows for one chromosome.

        The chromosome must contain genetic positions in cM. The returned
        ``(W, 2)`` integer array contains half-open ``[start, stop)`` variant
        boundaries. When ``store=True``, boundaries and physical/genetic
        midpoints are written under a namespace determined by
        ``window_size_cm``. An existing namespace for that size is replaced,
        and the HapStore must be writable.
        """

        chrom_group = self._require_chromosome(chrom)
        genetic_pos = self.genetic_positions(chrom)
        if genetic_pos is None:
            raise ValueError(
                f"chromosome {self._chrom_key(chrom)!r} has no genetic_pos data; "
                "cannot compute cM windows"
            )

        boundaries = compute_window_boundaries(genetic_pos, window_size_cm=window_size_cm)

        if store:
            self._require_write_mode()
            windows_group = chrom_group.require_group("windows")
            namespace = _window_namespace(window_size_cm)
            if namespace in windows_group:
                del windows_group[namespace]
            namespace_group = windows_group.create_group(namespace)

            mid_bp, mid_cm = compute_window_midpoints(
                variant_positions_bp=self.variant_positions(chrom),
                genetic_positions_cm=genetic_pos,
                boundaries=boundaries,
            )

            window_chunk = int(min(max(1, boundaries.shape[0]), 4096))
            namespace_group.create_dataset(
                "boundaries",
                data=boundaries,
                dtype=np.int64,
                chunks=(window_chunk, 2),
                overwrite=False,
            )
            namespace_group.create_dataset(
                "midpoints_bp",
                data=mid_bp,
                dtype=np.int64,
                chunks=(window_chunk,),
                overwrite=False,
            )
            namespace_group.create_dataset(
                "midpoints_cm",
                data=mid_cm,
                dtype=np.float64,
                chunks=(window_chunk,),
                overwrite=False,
            )
            namespace_group.attrs.update(
                {
                    "window_size_cm": float(window_size_cm),
                    "n_windows": int(boundaries.shape[0]),
                }
            )

        return boundaries

    def iter_windows(
        self,
        chrom: str | int,
        boundaries: np.ndarray | None = None,
        window_size_cm: float | None = None,
    ) -> Iterator[tuple[int, int, int, np.ndarray]]:
        """Yield haplotypes for half-open variant windows.

        Each item is ``(window_index, start, stop, haplotype_block)``, where
        ``haplotype_block`` has shape ``(2 * N, stop - start)``. Explicit
        ``boundaries`` take precedence. Otherwise, stored boundaries are used;
        ``window_size_cm`` selects their namespace. If multiple namespaces
        exist, a window size must be supplied.
        """

        chrom_group = self._require_chromosome(chrom)

        if boundaries is None:
            if "windows" not in chrom_group:
                raise ValueError(
                    "stored windows not found for chromosome; call compute_windows(store=True) "
                    "or pass boundaries explicitly"
                )

            windows_group = chrom_group["windows"]
            if window_size_cm is not None:
                namespace = _window_namespace(window_size_cm)
                if namespace not in windows_group:
                    raise ValueError(
                        f"stored windows namespace {namespace!r} not found for chromosome "
                        f"{self._chrom_key(chrom)!r}"
                    )
                boundaries_arr = np.asarray(windows_group[namespace]["boundaries"][:], dtype=np.int64)
            else:
                namespaces = sorted(list(windows_group.group_keys()))
                if not namespaces:
                    raise ValueError(
                        "stored windows not found for chromosome; call compute_windows(store=True) "
                        "or pass boundaries explicitly"
                    )
                if len(namespaces) > 1:
                    raise ValueError(
                        "multiple stored window namespaces found; pass window_size_cm or boundaries"
                    )
                boundaries_arr = np.asarray(
                    windows_group[namespaces[0]]["boundaries"][:],
                    dtype=np.int64,
                )
        else:
            boundaries_arr = np.asarray(boundaries, dtype=np.int64)

        if boundaries_arr.ndim != 2 or boundaries_arr.shape[1] != 2:
            raise ValueError(f"boundaries must have shape (W, 2), got shape={boundaries_arr.shape}")

        hap_array = chrom_group["haplotypes"]
        n_variants = int(hap_array.shape[1])

        for i, (start, end) in enumerate(boundaries_arr):
            s = int(start)
            e = int(end)
            if s < 0 or e < 0 or s >= e or e > n_variants:
                raise ValueError(
                    f"invalid boundary at row {i}: start={s}, end={e}, n_variants={n_variants}"
                )
            yield i, s, e, np.asarray(hap_array[:, s:e], dtype=np.uint8)


def _window_namespace(window_size_cm: float) -> str:
    text = format(float(window_size_cm), ".12g")
    text = text.replace("-", "m").replace(".", "p")
    return f"cm_{text}"


def _normalize_variant_slice(
    variant_slice: slice | tuple[int, int] | None,
    n_variants: int,
) -> slice:
    if variant_slice is None:
        return slice(None)
    if isinstance(variant_slice, tuple):
        if len(variant_slice) != 2:
            raise ValueError(
                f"variant_slice tuple must have length 2, got {len(variant_slice)}: {variant_slice}"
            )
        start, end = variant_slice
        return slice(int(start), int(end))
    if isinstance(variant_slice, slice):
        if variant_slice.step not in (None, 1):
            raise ValueError(f"variant_slice step must be None or 1, got {variant_slice.step}")
        return variant_slice
    raise ValueError(f"variant_slice must be slice, tuple(start,end), or None; got {type(variant_slice)}")


def _normalize_block_size(block_size: int | None, *, default: int) -> int:
    if block_size is None:
        value = int(default)
    else:
        value = int(block_size)
    if value <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return value


def _normalize_min_maf(min_maf: float | None) -> float | None:
    if min_maf is None:
        return None
    value = float(min_maf)
    if value < 0 or value > 0.5:
        raise ValueError(f"min_maf must be between 0 and 0.5, got {min_maf}")
    return value


def _concatenate_maf_chunks(
    chunks: dict[str, list[np.ndarray]],
    *,
    include_metadata: bool,
    saw_genetic: bool,
) -> dict[str, Any]:
    table: dict[str, Any] = {}
    numeric_defaults: dict[str, np.dtype] = {
        "variant_indices": np.dtype(np.int64),
        "allele_frequency": np.dtype(np.float64),
        "maf": np.dtype(np.float64),
        "alt_allele_count": np.dtype(np.int64),
        "called_genotype_count": np.dtype(np.int64),
        "called_allele_count": np.dtype(np.int64),
        "positions": np.dtype(np.int64),
        "genetic_pos": np.dtype(np.float64),
    }
    for key in (
        "variant_indices",
        "allele_frequency",
        "maf",
        "alt_allele_count",
        "called_genotype_count",
        "called_allele_count",
    ):
        values = chunks.get(key, [])
        table[key] = np.concatenate(values) if values else np.empty(0, dtype=numeric_defaults[key])
    if include_metadata:
        for key in ("positions", "ref", "alt", "vid"):
            values = chunks.get(key, [])
            if values:
                table[key] = np.concatenate(values)
            else:
                dtype = numeric_defaults.get(key, np.dtype(object))
                table[key] = np.empty(0, dtype=dtype)
        genetic_values = chunks.get("genetic_pos", [])
        table["genetic_pos"] = (
            np.concatenate(genetic_values) if saw_genetic and genetic_values else None
        )
    return table


def _filter_table_rows(table: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in table.items():
        if value is None:
            out[key] = None
            continue
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == keep.shape[0]:
            out[key] = arr[keep]
        else:
            out[key] = value
    return out


def _validate_sample_mask(sample_mask: np.ndarray, n_samples: int) -> np.ndarray:
    mask = np.asarray(sample_mask)
    if mask.ndim != 1 or mask.shape[0] != n_samples:
        raise ValueError(
            f"sample_mask must be shape ({n_samples},), got shape={mask.shape}"
        )
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    return mask


def _parse_positive_worker_count(workers: int | str) -> int | None:
    if isinstance(workers, str):
        text = workers.strip().lower()
        if text == "auto":
            return None
        if not text:
            raise ValueError("workers must be a positive integer or 'auto'")
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"workers must be a positive integer or 'auto', got {workers!r}") from exc
    else:
        value = int(workers)
    if value <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    return value


def _is_explicit_single_worker(workers: int | str) -> bool:
    value = _parse_positive_worker_count(workers)
    return value == 1


def _env_positive_int(names: Sequence[str], default: int | None = None) -> tuple[int | None, str | None]:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {raw!r}")
        return value, name
    return default, None


def _env_positive_float(names: Sequence[str], default: float | None = None) -> tuple[float | None, str | None]:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = float(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive number, got {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {raw!r}")
        return value, name
    return default, None


def _resolve_read_worker_count(
    workers: int | str,
    *,
    n_tasks: int,
    n_sample_groups: int,
    n_variant_groups: int,
    n_samples: int,
    n_variants: int,
    row_chunk: int,
    variant_chunk: int,
    dataset: str,
    estimated_tile_bytes: int | None = None,
    output_bytes: int = 0,
) -> tuple[int, dict[str, Any]]:
    """Choose a worker count bounded by tasks, CPUs, and available memory.

    Returns the selected count and a metrics dictionary recording the inputs,
    limits, and source of the decision.
    """
    available_cpus, cpu_source = _available_read_cpu_count()
    max_task_workers = max(1, min(int(n_tasks), int(available_cpus)))
    if estimated_tile_bytes is None:
        row_factor = 1 if str(dataset) == "genotypes" else 2
        estimated_tile_bytes = int(
            min(max(1, int(row_chunk)), max(1, int(n_samples)))
            * min(max(1, int(variant_chunk)), max(1, int(n_variants)))
            * row_factor
        )
    estimated_tile_bytes = max(1, int(estimated_tile_bytes))
    output_bytes = max(0, int(output_bytes))
    estimated_tile_mb = float(estimated_tile_bytes / (1024.0**2))
    available_memory_bytes, memory_source = _available_read_memory_bytes(available_cpus)

    base_policy = {
        "auto_estimated_tile_mb": estimated_tile_mb,
        "auto_output_mb": float(output_bytes / (1024.0**2)),
        "auto_available_cpus": int(available_cpus),
        "auto_cpu_source": str(cpu_source),
        "auto_available_memory_mb": (
            float(available_memory_bytes / (1024.0**2))
            if available_memory_bytes is not None
            else float("nan")
        ),
        "auto_memory_source": str(memory_source),
    }
    explicit_workers = _parse_positive_worker_count(workers)
    if explicit_workers is not None:
        resolved = max(1, min(int(explicit_workers), max_task_workers))
        return resolved, {
            **base_policy,
            "auto_worker_source": "explicit",
            "auto_worker_cap": float("nan"),
            "auto_memory_worker_cap": float("nan"),
        }

    env_workers, env_name = _env_positive_int(("HAPSTORE_READ_WORKERS", "HAPSTORE_AUTO_READ_WORKERS"))
    if env_workers is not None:
        resolved = max(1, min(int(env_workers), max_task_workers))
        return resolved, {
            **base_policy,
            "auto_worker_source": str(env_name),
            "auto_worker_cap": float("nan"),
            "auto_memory_worker_cap": float("nan"),
        }

    auto_cap, cap_source = _env_positive_int(
        ("HAPSTORE_READ_AUTO_MAX_WORKERS", "HAPSTORE_READ_AUTO_MAX"),
        default=16,
    )
    assert auto_cap is not None

    tasks = max(1, int(n_tasks))
    if tasks <= 1:
        target = 1
        source = "single_tile"
    elif tasks <= 4:
        target = tasks
        source = "few_tiles"
    elif int(n_variant_groups) <= 1 and int(n_sample_groups) < 8:
        target = 4
        source = "few_row_tiles"
    elif tasks < 16:
        target = 8
        source = "moderate_tiles"
    elif tasks < 48:
        target = 12
        source = "many_tiles"
    else:
        target = 16
        source = "large_tile_grid"

    if available_memory_bytes is None:
        memory_budget_bytes = None
        memory_worker_cap = max_task_workers
    else:
        memory_budget_bytes = max(
            0,
            int(float(available_memory_bytes) * _AUTO_READ_TILE_MEMORY_FRACTION) - output_bytes,
        )
        memory_worker_cap = max(1, int(memory_budget_bytes // estimated_tile_bytes))

    resolved = max(
        1,
        min(int(target), int(auto_cap), max_task_workers, int(memory_worker_cap)),
    )
    cap_label = cap_source if cap_source is not None else "default"
    return resolved, {
        **base_policy,
        "auto_worker_source": source,
        "auto_worker_cap": int(auto_cap),
        "auto_worker_cap_source": cap_label,
        "auto_memory_worker_cap": int(memory_worker_cap),
        "auto_memory_budget_mb": (
            float(memory_budget_bytes / (1024.0**2))
            if memory_budget_bytes is not None
            else float("nan")
        ),
        "auto_max_task_workers": int(max_task_workers),
    }


def _available_read_cpu_count() -> tuple[int, str]:
    """Return the CPU budget visible to this process and its limiting source."""
    candidates: list[tuple[str, int]] = []
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = len(affinity(0))
            if count > 0:
                candidates.append(("sched_getaffinity", int(count)))
        except (AttributeError, NotImplementedError, OSError):
            pass

    slurm_raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_raw:
        try:
            slurm_count = int(slurm_raw)
        except ValueError:
            slurm_count = 0
        if slurm_count > 0:
            candidates.append(("SLURM_CPUS_PER_TASK", slurm_count))

    system_count = os.cpu_count() or 1
    candidates.append(("os.cpu_count", max(1, int(system_count))))
    count = min(value for _source, value in candidates)
    limiting_sources = [source for source, value in candidates if value == count]
    return int(count), "+".join(limiting_sources)


def _available_read_memory_bytes(cpu_count: int) -> tuple[int | None, str]:
    """Return the most restrictive observable memory allowance in bytes.

    The selection strategy is inspired by ``dask.utils.get_memory_limit`` but
    is limited to SLURM allocations, cgroup limits, and ``/proc/meminfo``.
    """
    mib = 1024**2
    candidates: list[tuple[str, int]] = []

    for name, multiplier in (
        ("SLURM_MEM_PER_NODE", 1),
        ("SLURM_MEM_PER_CPU", max(1, int(cpu_count))),
    ):
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            value = int(raw) * int(multiplier) * mib
        except ValueError:
            continue
        if value > 0:
            candidates.append((name, value))

    cgroup_pairs = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, current_path in cgroup_pairs:
        try:
            raw_limit = limit_path.read_text(encoding="utf-8").strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
            current = int(current_path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, OSError, ValueError):
            continue
        available = limit - current
        if 0 < available < (1 << 60):
            candidates.append(("cgroup", available))
            break

    meminfo = Path("/proc/meminfo")
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(("proc_memavailable", int(line.split()[1]) * 1024))
                break
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass

    if not candidates:
        return None, "unavailable"
    source, value = min(candidates, key=lambda item: item[1])
    return int(value), str(source)


def _make_read_groups(indices: np.ndarray, *, chunk_size: int, limit: int) -> list[_ReadGroup]:
    """Group requested indices by storage chunk while retaining output positions."""
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    pairs_by_chunk: dict[int, list[tuple[int, int]]] = {}
    for out_pos, idx in enumerate(np.asarray(indices, dtype=np.int64)):
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= int(limit):
            raise IndexError(f"index {idx_int} out of bounds for limit={limit}")
        pairs_by_chunk.setdefault(idx_int // int(chunk_size), []).append((idx_int, int(out_pos)))
    groups: list[_ReadGroup] = []
    for chunk_id in sorted(pairs_by_chunk):
        pairs = sorted(pairs_by_chunk[chunk_id], key=lambda item: (item[0], item[1]))
        start = int(pairs[0][0])
        stop = int(pairs[-1][0]) + 1
        groups.append(
            _ReadGroup(
                start=start,
                stop=stop,
                offsets=np.asarray([idx - start for idx, _out in pairs], dtype=np.int64),
                out_positions=np.asarray([out for _idx, out in pairs], dtype=np.int64),
            )
        )
    return groups


def _plan_variant_read_groups(
    indices: np.ndarray,
    *,
    chunk_size: int,
    limit: int,
) -> list[_ReadGroup]:
    """Use one span for dense requests and bounded chunk spans for sparse ones."""
    indices_arr = np.asarray(indices, dtype=np.int64)
    groups = _make_read_groups(indices_arr, chunk_size=chunk_size, limit=limit)
    if len(groups) <= 1:
        return groups
    span_start = int(np.min(indices_arr))
    span_stop = int(np.max(indices_arr)) + 1
    if int(np.unique(indices_arr).size) != int(span_stop - span_start):
        return groups

    out_positions = np.arange(indices_arr.size, dtype=np.int64)
    order = np.lexsort((out_positions, indices_arr))
    return [
        _ReadGroup(
            start=span_start,
            stop=span_stop,
            offsets=np.asarray(indices_arr[order] - span_start, dtype=np.int64),
            out_positions=np.asarray(out_positions[order], dtype=np.int64),
        )
    ]


def _read_chunk_grouped_array(
    arr: zarr.Array,
    *,
    row_indices: np.ndarray,
    variant_indices: np.ndarray,
) -> np.ndarray:
    """Read planned variant spans and restore requested row/column ordering."""
    rows = np.asarray(row_indices, dtype=np.int64)
    variants = np.asarray(variant_indices, dtype=np.int64)
    out = np.empty((rows.size, variants.size), dtype=np.dtype(arr.dtype))
    if rows.size == 0 or variants.size == 0:
        return out

    groups = _plan_variant_read_groups(
        variants,
        chunk_size=int(arr.chunks[1]),
        limit=int(arr.shape[1]),
    )
    for group in groups:
        span = slice(int(group.start), int(group.stop))
        if rows.size > int(arr.shape[0]) // 2:
            dense = np.asarray(arr[:, span], dtype=arr.dtype)
            block = dense[rows]
        else:
            block = np.asarray(arr.oindex[rows, span], dtype=arr.dtype)
        out[:, group.out_positions] = block[:, group.offsets]
    return out


def _coalesce_read_groups(
    groups: list[_ReadGroup],
    *,
    max_span: int,
    max_gap: int,
    min_density: float,
) -> list[_ReadGroup]:
    """Merge nearby read groups without violating span, gap, or density limits."""
    if len(groups) <= 1:
        return groups
    if int(max_span) <= 0:
        raise ValueError(f"max_span must be positive, got {max_span}")
    if int(max_gap) < 0:
        raise ValueError(f"max_gap must be non-negative, got {max_gap}")
    if float(min_density) <= 0:
        raise ValueError(f"min_density must be positive, got {min_density}")

    out: list[_ReadGroup] = []
    current_start = int(groups[0].start)
    current_stop = int(groups[0].stop)
    current_abs = [np.asarray(groups[0].start + groups[0].offsets, dtype=np.int64)]
    current_out = [np.asarray(groups[0].out_positions, dtype=np.int64)]

    def flush() -> None:
        abs_idx = np.concatenate(current_abs).astype(np.int64, copy=False)
        out_pos = np.concatenate(current_out).astype(np.int64, copy=False)
        order = np.lexsort((out_pos, abs_idx))
        abs_sorted = abs_idx[order]
        out_sorted = out_pos[order]
        out.append(
            _ReadGroup(
                start=int(abs_sorted[0]),
                stop=int(abs_sorted[-1]) + 1,
                offsets=np.asarray(abs_sorted - int(abs_sorted[0]), dtype=np.int64),
                out_positions=np.asarray(out_sorted, dtype=np.int64),
            )
        )

    for group in groups[1:]:
        group_start = int(group.start)
        group_stop = int(group.stop)
        candidate_start = min(current_start, group_start)
        candidate_stop = max(current_stop, group_stop)
        candidate_span = int(candidate_stop - candidate_start)
        candidate_count = int(sum(arr.size for arr in current_abs) + group.offsets.size)
        gap = int(group_start - current_stop)
        density = float(candidate_count / max(1, candidate_span))
        if candidate_span <= int(max_span) and gap <= int(max_gap) and density >= float(min_density):
            current_start = candidate_start
            current_stop = candidate_stop
            current_abs.append(np.asarray(group.start + group.offsets, dtype=np.int64))
            current_out.append(np.asarray(group.out_positions, dtype=np.int64))
            continue
        flush()
        current_start = group_start
        current_stop = group_stop
        current_abs = [np.asarray(group.start + group.offsets, dtype=np.int64)]
        current_out = [np.asarray(group.out_positions, dtype=np.int64)]

    flush()
    return out


def _normalize_missing_mode(missing: str) -> str:
    mode = str(missing).lower()
    if mode not in {"nan", "sentinel"}:
        raise ValueError(f"missing must be one of 'nan' or 'sentinel', got {missing!r}")
    return mode


def _format_genotype_values(
    values: np.ndarray,
    *,
    dtype: np.dtype,
    missing: str,
    has_missing: bool,
) -> np.ndarray:
    out = np.asarray(values, dtype=dtype)
    if has_missing:
        missing_mask = np.asarray(values) == _MISSING_SENTINEL
        if np.any(missing_mask):
            if missing == "nan":
                out = out.astype(np.float32 if dtype.itemsize <= 4 else np.float64, copy=False)
                out[missing_mask] = np.nan
            else:
                out[missing_mask] = np.asarray(_MISSING_SENTINEL, dtype=out.dtype)
    return out


def _format_haplotype_pair_values(
    first: np.ndarray,
    second: np.ndarray,
    *,
    dtype: np.dtype,
    missing: str,
    has_missing: bool,
) -> np.ndarray:
    if has_missing:
        missing_mask = (first == _MISSING_SENTINEL) | (second == _MISSING_SENTINEL)
    else:
        missing_mask = None
    summed = np.asarray(first, dtype=np.uint16) + np.asarray(second, dtype=np.uint16)
    out = summed.astype(dtype, copy=False)
    if missing_mask is not None and np.any(missing_mask):
        if missing == "nan":
            out = out.astype(np.float32 if dtype.itemsize <= 4 else np.float64, copy=False)
            out[missing_mask] = np.nan
        else:
            out[missing_mask] = np.asarray(_MISSING_SENTINEL, dtype=out.dtype)
    return out


def _normalize_optional_str_array(
    values: list[str] | np.ndarray | None,
    expected_len: int,
    name: str,
    default: str,
) -> np.ndarray:
    if values is None:
        return np.full(expected_len, default, dtype=object)
    arr = _to_unicode_array(np.asarray(values, dtype=object))
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(
            f"{name} must be shape ({expected_len},), got shape={arr.shape}"
        )
    return arr


def _haplotypes_to_genotypes(haplotypes: np.ndarray) -> np.ndarray:
    h = np.asarray(haplotypes, dtype=np.uint8)
    if h.ndim != 2 or h.shape[0] % 2 != 0:
        raise ValueError(f"haplotypes must have shape (2N, V), got {h.shape}")

    first = h[0::2, :]
    second = h[1::2, :]
    missing = (first == _MISSING_SENTINEL) | (second == _MISSING_SENTINEL)
    out = (first + second).astype(np.uint8, copy=False)
    out[missing] = _MISSING_SENTINEL
    return out
