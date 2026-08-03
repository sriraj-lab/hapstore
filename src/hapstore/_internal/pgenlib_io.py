"""Private adapter layer for pgenlib read/write operations.

This module intentionally provides a small compatibility surface used by
`hapstore.pgen`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


try:  # pragma: no cover - optional dependency import branch
    import pgenlib as _pgenlib  # type: ignore
except Exception:  # pragma: no cover
    _pgenlib = None


def ensure_pgenlib_available() -> None:
    """Raise a clear error when pgenlib is unavailable."""
    if _pgenlib is None:
        raise RuntimeError(
            "pgenlib is required for direct HapStore<->PGEN operations. "
            "Install pgenlib (>=0.93.0) in the active environment."
        )


class PgenlibWriter:
    """Write phased, biallelic hardcalls to a PGEN payload.

    This adapter writes only the ``.pgen`` file; ``hapstore.pgen`` owns the
    aligned PVAR and PSAM files. The sample and variant counts are fixed when
    the writer opens, and every appended batch is marked as phased.

    Args:
        pgen_path: Output path for the PGEN payload.
        sample_ct: Number of diploid samples represented by every batch.
        variant_ct: Total number of variants that will be appended.
    """

    def __init__(self, pgen_path: str | Path, sample_ct: int, variant_ct: int) -> None:
        ensure_pgenlib_available()
        if int(sample_ct) <= 0:
            raise ValueError(f"sample_ct must be > 0, got {sample_ct}")
        if int(variant_ct) <= 0:
            raise ValueError(f"variant_ct must be > 0, got {variant_ct}")

        path_bytes = str(Path(pgen_path)).encode("utf-8")
        self._writer = _pgenlib.PgenWriter(
            path_bytes,
            int(sample_ct),
            variant_ct=int(variant_ct),
            nonref_flags=False,
            allele_ct_limit=2,
            hardcall_phase_present=True,
        )

    def append_alleles_batch(self, allele_batch: np.ndarray) -> None:
        """Append variant-major, interleaved diploid allele hardcalls.

        ``allele_batch`` must have shape ``(variants, 2 * samples)`` with each
        sample's two alleles in adjacent columns. Alleles are encoded as
        ``int32`` hardcalls, with ``-9`` representing missing alleles. Input is
        converted to C-contiguous ``int32`` storage when necessary.
        """
        arr = np.asarray(allele_batch)
        if arr.ndim != 2:
            raise ValueError(f"allele_batch must be 2D, got shape={arr.shape}")
        if arr.shape[1] % 2 != 0:
            raise ValueError(
                f"allele_batch second dimension must be even (2N), got {arr.shape[1]}"
            )
        if arr.dtype != np.int32:
            arr = arr.astype(np.int32, copy=False)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        self._writer.append_alleles_batch(arr, all_phased=True)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "PgenlibWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except Exception:
            # Preserve the original exception (e.g., validation error inside export loop).
            pass


class PgenlibReader:
    """Read PGEN hardcalls together with aligned PVAR metadata.

    The optional sample subset is passed to pgenlib and controls the sample
    axis returned by hardcall reads. ``hapstore.pgen`` supplies a sorted subset
    and separately restores any caller-requested sample order.

    Args:
        pgen_path: Input PGEN payload.
        pvar_path: Aligned PVAR variant metadata.
        sample_subset: Optional zero-based source-sample indices to read.
    """

    def __init__(
        self,
        pgen_path: str | Path,
        pvar_path: str | Path,
        sample_subset: np.ndarray | None = None,
    ) -> None:
        ensure_pgenlib_available()

        self._pvar = _pgenlib.PvarReader(str(Path(pvar_path)).encode("utf-8"))
        kwargs: dict[str, object] = {"pvar": self._pvar}
        if sample_subset is not None:
            subset = np.asarray(sample_subset, dtype=np.uint32)
            if subset.ndim != 1:
                raise ValueError(f"sample_subset must be 1D, got shape={subset.shape}")
            kwargs["sample_subset"] = subset
            self._sample_ct = int(subset.shape[0])
        else:
            self._sample_ct = None

        self._reader = _pgenlib.PgenReader(str(Path(pgen_path)).encode("utf-8"), **kwargs)

    @property
    def variant_ct(self) -> int:
        return int(self._reader.get_variant_ct())

    @property
    def raw_sample_ct(self) -> int:
        return int(self._reader.get_raw_sample_ct())

    @property
    def sample_ct(self) -> int:
        if self._sample_ct is not None:
            return int(self._sample_ct)
        return int(self._reader.get_raw_sample_ct())

    def hardcall_phase_present(self) -> bool:
        return bool(self._reader.hardcall_phase_present())

    def read_alleles_and_phasepresent_range(
        self,
        start: int,
        end: int,
        allele_buf: np.ndarray,
        phasepresent_buf: np.ndarray,
    ) -> None:
        """Fill caller-owned buffers for a half-open variant range.

        ``allele_buf`` must be writable, C-contiguous ``int32`` with shape
        ``(end - start, 2 * sample_ct)``. ``phasepresent_buf`` must be writable,
        C-contiguous ``uint8`` with shape ``(end - start, sample_ct)``. Requiring
        exact dtypes prevents NumPy from creating temporary output arrays that
        would not update the caller's buffers.
        """
        start_i = int(start)
        end_i = int(end)
        if start_i < 0 or end_i <= start_i or end_i > self.variant_ct:
            raise ValueError(
                "invalid PGEN variant range: "
                f"[{start_i}, {end_i}) for variant_ct={self.variant_ct}"
            )

        alleles = np.asarray(allele_buf)
        phase = np.asarray(phasepresent_buf)
        if alleles.dtype != np.dtype(np.int32):
            raise ValueError(f"allele_buf must have dtype int32, got {alleles.dtype}")
        if phase.dtype != np.dtype(np.uint8):
            raise ValueError(
                f"phasepresent_buf must have dtype uint8, got {phase.dtype}"
            )

        width = int(end_i - start_i)
        expected_allele_shape = (width, 2 * self.sample_ct)
        expected_phase_shape = (width, self.sample_ct)
        if alleles.shape != expected_allele_shape:
            raise ValueError(
                "allele_buf shape mismatch: "
                f"expected {expected_allele_shape}, got {alleles.shape}"
            )
        if phase.shape != expected_phase_shape:
            raise ValueError(
                "phasepresent_buf shape mismatch: "
                f"expected {expected_phase_shape}, got {phase.shape}"
            )
        if not alleles.flags["C_CONTIGUOUS"]:
            raise ValueError("allele_buf must be C-contiguous")
        if not phase.flags["C_CONTIGUOUS"]:
            raise ValueError("phasepresent_buf must be C-contiguous")
        if not alleles.flags["WRITEABLE"]:
            raise ValueError("allele_buf must be writable")
        if not phase.flags["WRITEABLE"]:
            raise ValueError("phasepresent_buf must be writable")

        # hap_maj=False requests the same variant-major layout used by export.
        self._reader.read_alleles_and_phasepresent_range(
            start_i,
            end_i,
            alleles,
            phase,
            hap_maj=False,
        )

    def variant_chrom(self, variant_idx: int) -> str:
        val = self._pvar.get_variant_chrom(int(variant_idx))
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)

    def variant_pos(self, variant_idx: int) -> int:
        return int(self._pvar.get_variant_pos(int(variant_idx)))

    def variant_id(self, variant_idx: int) -> str:
        val = self._pvar.get_variant_id(int(variant_idx))
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)

    def allele_ct(self, variant_idx: int) -> int:
        return int(self._pvar.get_allele_ct(int(variant_idx)))

    def allele_code(self, variant_idx: int, allele_idx: int) -> str:
        val = self._pvar.get_allele_code(int(variant_idx), int(allele_idx))
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)

    def close(self) -> None:
        self._reader.close()
        self._pvar.close()

    def __enter__(self) -> "PgenlibReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
