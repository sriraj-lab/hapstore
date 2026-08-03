from __future__ import annotations

import numpy as np
import pytest

from hapstore._internal.pgenlib_io import PgenlibReader


class _FakePgenReader:
    def __init__(self, *, variant_ct: int = 3) -> None:
        self.variant_ct = int(variant_ct)
        self.calls: list[tuple] = []

    def get_variant_ct(self) -> int:
        return self.variant_ct

    def read_alleles_and_phasepresent_range(self, *args, **kwargs) -> None:
        self.calls.append((*args, kwargs))


def _reader(*, sample_ct: int = 2, variant_ct: int = 3) -> tuple[PgenlibReader, _FakePgenReader]:
    backend = _FakePgenReader(variant_ct=variant_ct)
    reader = PgenlibReader.__new__(PgenlibReader)
    reader._reader = backend
    reader._sample_ct = int(sample_ct)
    return reader, backend


@pytest.mark.parametrize(
    ("allele_dtype", "phase_dtype", "message"),
    [
        (np.int16, np.uint8, "allele_buf must have dtype int32"),
        (np.int32, np.int8, "phasepresent_buf must have dtype uint8"),
    ],
)
def test_read_range_rejects_output_buffer_dtype(allele_dtype, phase_dtype, message):
    reader, backend = _reader()
    alleles = np.empty((2, 4), dtype=allele_dtype)
    phase = np.empty((2, 2), dtype=phase_dtype)

    with pytest.raises(ValueError, match=message):
        reader.read_alleles_and_phasepresent_range(0, 2, alleles, phase)

    assert backend.calls == []


@pytest.mark.parametrize(
    ("allele_shape", "phase_shape", "message"),
    [
        ((2, 2), (2, 2), "allele_buf shape mismatch"),
        ((2, 4), (2, 1), "phasepresent_buf shape mismatch"),
    ],
)
def test_read_range_rejects_output_buffer_shape(allele_shape, phase_shape, message):
    reader, backend = _reader()
    alleles = np.empty(allele_shape, dtype=np.int32)
    phase = np.empty(phase_shape, dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        reader.read_alleles_and_phasepresent_range(0, 2, alleles, phase)

    assert backend.calls == []


def test_read_range_passes_caller_owned_buffers_without_copying():
    reader, backend = _reader()
    alleles = np.empty((2, 4), dtype=np.int32, order="C")
    phase = np.empty((2, 2), dtype=np.uint8, order="C")

    reader.read_alleles_and_phasepresent_range(1, 3, alleles, phase)

    assert len(backend.calls) == 1
    start, end, got_alleles, got_phase, kwargs = backend.calls[0]
    assert (start, end) == (1, 3)
    assert got_alleles is alleles
    assert got_phase is phase
    assert kwargs == {"hap_maj": False}
