"""Zarr-backed haplotype/genotype store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr
from numcodecs import VLenUTF8, Zstd

from hapstore.chrom import normalize_chrom

from ._internal.chrom_attrs import build_chrom_attrs
from ._internal.zarr_utils import _create_utf8_array, _to_unicode_array
from .windows import compute_window_boundaries, compute_window_midpoints

_MISSING_SENTINEL = 255
_SCHEMA_VERSION = 2


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
            sample_indices = np.flatnonzero(mask)
            return np.asarray(chrom_group["genotypes"].oindex[sample_indices, variant_sel], dtype=np.uint8)

        haps = self.haplotypes(chrom, variant_slice=variant_sel, sample_mask=sample_mask)
        return _haplotypes_to_genotypes(haps)

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
