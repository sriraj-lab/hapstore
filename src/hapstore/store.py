"""Zarr-backed haplotype/genotype store."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterator, Sequence

import numpy as np
import zarr
from numcodecs import VLenUTF8, Zstd

from hapstore.chrom import normalize_chrom

from ._internal.chrom_attrs import build_chrom_attrs
from ._internal.zarr_utils import _create_utf8_array, _to_unicode_array
from .windows import compute_window_boundaries, compute_window_midpoints

_MISSING_SENTINEL = 255
_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class HapStoreDosageRequest:
    """One named dosage request for a shared chromosome read."""

    request_id: Hashable
    variant_ids: Sequence[str] | None = None
    variant_indices: Sequence[int] | np.ndarray | None = None


@dataclass(frozen=True)
class HapStoreDosageBatch:
    """Shared union dosage matrix plus per-request column maps."""

    matrix: np.ndarray
    variant_indices: np.ndarray
    request_columns: dict[Hashable, np.ndarray]
    request_ids: tuple[Hashable, ...]
    variant_ids: tuple[str, ...] | None = None
    metrics: dict[str, Any] | None = None

    def window_matrix(self, request_id: Hashable) -> np.ndarray:
        cols = self.request_columns[request_id]
        return np.asarray(self.matrix[:, cols], dtype=self.matrix.dtype)

    def project(self, request_id: Hashable, weights: np.ndarray) -> np.ndarray:
        cols = self.request_columns[request_id]
        w = np.asarray(weights, dtype=self.matrix.dtype)
        if w.ndim != 1 or w.shape[0] != cols.shape[0]:
            raise ValueError(
                f"weights for request {request_id!r} must have shape ({cols.shape[0]},), "
                f"got {w.shape}"
            )
        return np.asarray(self.matrix[:, cols] @ w, dtype=self.matrix.dtype)


@dataclass(frozen=True)
class _ReadGroup:
    start: int
    stop: int
    offsets: np.ndarray
    out_positions: np.ndarray


class HapStore:
    """High-level API over a chromosome-partitioned zarr haplotype store."""

    def __init__(self, path: str | Path, mode: str = "r") -> None:
        if mode not in {"r", "a", "w"}:
            raise ValueError(f"mode must be one of 'r', 'a', 'w'; got {mode!r}")

        self._path = Path(path)
        self._mode = mode

        if mode == "r" and not self._path.exists():
            raise FileNotFoundError(f"hapstore path does not exist: {self._path}")

        self._root = zarr.open(str(self._path), mode=mode)
        self._variant_index_cache: dict[str, dict[str, int]] = {}
        self._last_read_metrics: dict[str, Any] = {}
        if mode in {"a", "w"}:
            if "format" not in self._root.attrs:
                self._root.attrs["format"] = "hapstore"
            if "schema_version" not in self._root.attrs:
                self._root.attrs["schema_version"] = _SCHEMA_VERSION

    @property
    def chromosomes(self) -> list[str]:
        return sorted(list(self._root.group_keys()))

    @property
    def root(self) -> zarr.Group:
        return self._root

    @property
    def path(self) -> Path:
        return self._path

    @property
    def last_read_metrics(self) -> dict[str, Any]:
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
        chunk_variant_size: int = 500,
        chunk_haplotype_size: int | None = None,
        chunk_sample_size: int | None = None,
        compression_level: int = 3,
        attrs: dict | None = None,
    ) -> None:
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

        sample_ids_arr = _to_unicode_array(np.asarray(sample_ids, dtype=object))
        n_samples = int(sample_ids_arr.shape[0])
        if haps.shape[0] != 2 * n_samples:
            raise ValueError(
                "haplotype/sample mismatch: "
                f"haplotypes.shape[0]={haps.shape[0]} but 2*len(sample_ids)={2 * n_samples}"
            )

        variant_pos_arr = np.asarray(variant_pos, dtype=np.int64)
        if variant_pos_arr.ndim != 1:
            raise ValueError(f"variant_pos must be 1D, got shape={variant_pos_arr.shape}")
        n_variants = int(variant_pos_arr.shape[0])
        if haps.shape[1] != n_variants:
            raise ValueError(
                "haplotype/variant mismatch: "
                f"haplotypes.shape[1]={haps.shape[1]} but len(variant_pos)={n_variants}"
            )

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

        genetic_arr: np.ndarray | None = None
        if genetic_positions is not None:
            genetic_arr = np.asarray(genetic_positions, dtype=np.float64)
            if genetic_arr.ndim != 1 or genetic_arr.shape[0] != n_variants:
                raise ValueError(
                    "genetic_positions must be shape (V,), got "
                    f"shape={genetic_arr.shape}, expected ({n_variants},)"
                )

        populations_arr: np.ndarray | None = None
        if populations is not None:
            populations_arr = _to_unicode_array(np.asarray(populations, dtype=object))
            if populations_arr.ndim != 1 or populations_arr.shape[0] != n_samples:
                raise ValueError(
                    "populations must be shape (N,), got "
                    f"shape={populations_arr.shape}, expected ({n_samples},)"
                )

        haps_u8 = haps.astype(np.uint8, copy=False)
        has_missing = bool(np.any(haps_u8 == _MISSING_SENTINEL))

        self._validate_sample_consistency(sample_ids_arr, chrom_key)

        compressor = Zstd(level=int(compression_level))
        variant_chunk = int(min(max(1, int(chunk_variant_size)), max(1, n_variants)))
        n_haplotypes = int(haps_u8.shape[0])
        hap_chunk = int(n_haplotypes if chunk_haplotype_size is None else chunk_haplotype_size)
        hap_chunk = int(min(max(1, hap_chunk), n_haplotypes))
        sample_chunk_geno = int(n_samples if chunk_sample_size is None else chunk_sample_size)
        sample_chunk_geno = int(min(max(1, sample_chunk_geno), n_samples))

        hap_chunks = (hap_chunk, variant_chunk)
        geno_chunks = (sample_chunk_geno, variant_chunk)

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
        layout_version = 1 if hap_chunk == n_haplotypes and sample_chunk_geno == n_samples else 2
        chrom_group.attrs.update(
            build_chrom_attrs(
                n_samples=n_samples,
                n_haplotypes=int(haps_u8.shape[0]),
                n_variants=n_variants,
                missing_sentinel=_MISSING_SENTINEL,
                has_missing=has_missing,
                compression_level=int(compression_level),
                chunk_variant_size=int(chunk_variant_size),
                chunk_haplotype_size=chunk_haplotype_size,
                chunk_sample_size=chunk_sample_size,
                layout_version=int(layout_version),
                source=source,
                build=build,
                extra=attrs_payload or None,
            )
        )

    def delete_chromosome(self, chrom: str | int) -> None:
        self._require_write_mode()
        chrom_key = self._chrom_key(chrom)
        if chrom_key not in self._root:
            raise KeyError(f"chromosome {chrom_key!r} not found; cannot delete")
        del self._root[chrom_key]

    def haplotypes(
        self,
        chrom: str | int,
        variant_slice: slice | tuple[int, int] | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
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
        """Read dosages with chunk-aware threaded 2D tiling.

        Output columns follow the requested variant order. ``workers=1`` or
        ``backend="serial"`` preserves the historical contiguous-span read path.
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

        if total_tasks <= 1:
            tasks = [
                read_tile(sample_groups[0], variant_groups[0])
            ]
        else:
            tasks = []
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="hapstore-read") as pool:
                futures = [
                    pool.submit(read_tile, sample_group, variant_group)
                    for sample_group in sample_groups
                    for variant_group in variant_groups
                ]
                for future in as_completed(futures):
                    tasks.append(future.result())
        for out_rows, out_cols, values, tile_read_sec, tile_convert_sec in tasks:
            read_sec += float(tile_read_sec)
            convert_sec += float(tile_convert_sec)
            assign_start = time.perf_counter()
            out[np.ix_(out_rows, out_cols)] = values
            assign_sec += time.perf_counter() - assign_start
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
            "has_missing": bool(has_missing),
            "assume_no_missing": bool(assume_no_missing),
            "read_sec_sum": float(read_sec),
            "convert_sec_sum": float(convert_sec),
            "assign_sec": float(assign_sec),
            "total_sec": float(time.perf_counter() - total_start),
        }
        return out

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
        parsed = [self._normalize_dosage_request(chrom, request, i) for i, request in enumerate(requests)]
        if not parsed:
            sample_idx = self._resolve_sample_indices(chrom=chrom, sample_indices=sample_indices)
            return HapStoreDosageBatch(
                matrix=np.empty((sample_idx.size, 0), dtype=np.dtype(dtype)),
                variant_indices=np.empty(0, dtype=np.int64),
                request_columns={},
                request_ids=(),
                metrics={"backend": "empty"},
            )
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
        union_pos = {idx: i for i, idx in enumerate(union_indices)}
        request_columns = {
            request_id: np.asarray([union_pos[int(idx)] for idx in indices], dtype=np.int64)
            for request_id, indices, _ids in parsed
        }
        matrix = self.read_dosages(
            chrom,
            variant_indices=np.asarray(union_indices, dtype=np.int64),
            sample_indices=sample_indices,
            dtype=dtype,
            missing=missing,
            workers=workers,
            backend=backend,
            assume_no_missing=assume_no_missing,
        )
        info = self.variant_info(chrom)
        chrom_alias = self._chrom_key(chrom).removeprefix("chr")
        union_variant_ids = tuple(
            str(info["vid"][idx]) if str(info["vid"][idx]) else f"{chrom_alias}:{int(info['pos'][idx])}"
            for idx in union_indices
        )
        return HapStoreDosageBatch(
            matrix=matrix,
            variant_indices=np.asarray(union_indices, dtype=np.int64),
            request_columns=request_columns,
            request_ids=tuple(request_id for request_id, _indices, _ids in parsed),
            variant_ids=union_variant_ids,
            metrics=self.last_read_metrics,
        )

    def variant_positions(self, chrom: str | int) -> np.ndarray:
        chrom_group = self._require_chromosome(chrom)
        return np.asarray(chrom_group["variants"]["pos"][:], dtype=np.int64)

    def genetic_positions(self, chrom: str | int) -> np.ndarray | None:
        chrom_group = self._require_chromosome(chrom)
        variants = chrom_group["variants"]
        if "genetic_pos" not in variants:
            return None
        return np.asarray(variants["genetic_pos"][:], dtype=np.float64)

    def sample_ids(self, chrom: str | int) -> list[str]:
        chrom_group = self._require_chromosome(chrom)
        return _to_unicode_array(np.asarray(chrom_group["samples"]["ids"][:], dtype=object)).tolist()

    def variant_info(self, chrom: str | int) -> dict[str, Any]:
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
        """Return variant ID/position aliases to variant-row indices."""
        return dict(self._variant_index_map(chrom))

    def _variant_index_map(self, chrom: str | int) -> dict[str, int]:
        chrom_key = self._chrom_key(chrom)
        if chrom_key not in self._variant_index_cache:
            chrom_group = self._require_chromosome(chrom)
            variants = chrom_group["variants"]
            chrom_alias = str(chrom_key).removeprefix("chr")
            out: dict[str, int] = {}
            vid_arr = _to_unicode_array(np.asarray(variants["vid"][:], dtype=object))
            pos_arr = np.asarray(variants["pos"][:], dtype=np.int64)
            for i, vid in enumerate(vid_arr):
                if vid:
                    out.setdefault(str(vid), int(i))
            for i, pos in enumerate(pos_arr):
                out.setdefault(f"chr{chrom_alias}:{int(pos)}", int(i))
                out.setdefault(f"{chrom_alias}:{int(pos)}", int(i))
            self._variant_index_cache[chrom_key] = out
        return self._variant_index_cache[chrom_key]

    def _resolve_variant_indices(
        self,
        *,
        chrom: str | int,
        variant_ids: Sequence[str] | None,
        variant_indices: Sequence[int] | np.ndarray | None,
    ) -> np.ndarray:
        if (variant_ids is None) == (variant_indices is None):
            raise ValueError("pass exactly one of variant_ids or variant_indices")
        if variant_ids is not None:
            lookup = self._variant_index_map(chrom)
            missing = [str(variant_id) for variant_id in variant_ids if str(variant_id) not in lookup]
            if missing:
                raise ValueError(
                    f"{len(missing)} variant IDs missing from HapStore chromosome {self._chrom_key(chrom)}; "
                    f"first={missing[:5]}"
                )
            arr = np.asarray([lookup[str(variant_id)] for variant_id in variant_ids], dtype=np.int64)
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

    def _normalize_dosage_request(
        self,
        chrom: str | int,
        request: HapStoreDosageRequest | dict[str, Any] | tuple[Hashable, Sequence[str] | Sequence[int] | np.ndarray],
        fallback_index: int,
    ) -> tuple[Hashable, np.ndarray, tuple[str, ...] | None]:
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
        indices = self._resolve_variant_indices(
            chrom=chrom,
            variant_ids=variant_ids,
            variant_indices=variant_indices,
        )
        ids_tuple = tuple(str(v) for v in variant_ids) if variant_ids is not None else None
        return request_id, indices, ids_tuple

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
        span_start = int(np.min(variant_indices))
        span_stop = int(np.max(variant_indices)) + 1
        read_start = time.perf_counter()
        geno = self.genotypes(chrom, variant_slice=(span_start, span_stop))
        read_sec = time.perf_counter() - read_start
        slice_start = time.perf_counter()
        cols = np.asarray(variant_indices - span_start, dtype=np.int64)
        values = geno[np.ix_(sample_indices, cols)]
        out = _format_genotype_values(
            values,
            dtype=dtype,
            missing=missing,
            has_missing=bool(self.attrs(chrom).get("has_missing", True)) and not bool(assume_no_missing),
        )
        self._last_read_metrics = {
            "backend": "serial",
            "dataset": "genotypes" if "genotypes" in self._require_chromosome(chrom) else "haplotypes",
            "workers": 1,
            "n_samples": int(sample_indices.size),
            "n_variants": int(variant_indices.size),
            "span_variants": int(span_stop - span_start),
            "has_missing": bool(self.attrs(chrom).get("has_missing", True)) and not bool(assume_no_missing),
            "assume_no_missing": bool(assume_no_missing),
            "read_sec_sum": float(read_sec),
            "assign_sec": float(time.perf_counter() - slice_start),
            "total_sec": float(read_sec + (time.perf_counter() - slice_start)),
        }
        return out

    def n_samples(self, chrom: str | int) -> int:
        chrom_group = self._require_chromosome(chrom)
        return int(chrom_group.attrs["n_samples"])

    def n_variants(self, chrom: str | int) -> int:
        chrom_group = self._require_chromosome(chrom)
        return int(chrom_group.attrs["n_variants"])

    def n_haplotypes(self, chrom: str | int) -> int:
        chrom_group = self._require_chromosome(chrom)
        return int(chrom_group.attrs["n_haplotypes"])

    def attrs(self, chrom: str | int) -> dict:
        chrom_group = self._require_chromosome(chrom)
        return dict(chrom_group.attrs.asdict())

    def has_chromosome(self, chrom: str | int) -> bool:
        return self._chrom_key(chrom) in self._root

    def compute_windows(
        self,
        chrom: str | int,
        window_size_cm: float,
        store: bool = False,
    ) -> np.ndarray:
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
) -> tuple[int, dict[str, Any]]:
    max_task_workers = max(1, min(int(n_tasks), os.cpu_count() or 1))
    explicit_workers = _parse_positive_worker_count(workers)
    if explicit_workers is not None:
        resolved = max(1, min(int(explicit_workers), max_task_workers))
        return resolved, {
            "auto_worker_source": "explicit",
            "auto_worker_cap": float("nan"),
            "auto_estimated_tile_mb": float("nan"),
        }

    env_workers, env_name = _env_positive_int(("HAPSTORE_READ_WORKERS", "HAPSTORE_AUTO_READ_WORKERS"))
    if env_workers is not None:
        resolved = max(1, min(int(env_workers), max_task_workers))
        return resolved, {
            "auto_worker_source": str(env_name),
            "auto_worker_cap": float("nan"),
            "auto_estimated_tile_mb": float("nan"),
        }

    auto_cap, cap_source = _env_positive_int(
        ("HAPSTORE_READ_AUTO_MAX_WORKERS", "HAPSTORE_READ_AUTO_MAX"),
        default=8,
    )
    assert auto_cap is not None

    input_rows = min(max(1, int(row_chunk)), max(1, int(n_samples)))
    input_cols = min(max(1, int(variant_chunk)), max(1, int(n_variants)))
    row_factor = 1 if str(dataset) == "genotypes" else 2
    estimated_tile_mb = float(input_rows * input_cols * row_factor / (1024.0**2))

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

    resolved = max(1, min(int(target), int(auto_cap), max_task_workers))
    cap_label = cap_source if cap_source is not None else "default"
    return resolved, {
        "auto_worker_source": source,
        "auto_worker_cap": int(auto_cap),
        "auto_worker_cap_source": cap_label,
        "auto_estimated_tile_mb": estimated_tile_mb,
        "auto_max_task_workers": int(max_task_workers),
    }


def _make_read_groups(indices: np.ndarray, *, chunk_size: int, limit: int) -> list[_ReadGroup]:
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


def _coalesce_read_groups(
    groups: list[_ReadGroup],
    *,
    max_span: int,
    max_gap: int,
    min_density: float,
) -> list[_ReadGroup]:
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
