from __future__ import annotations

import numpy as np
import pytest

from hapstore import HapStore, ingest_pgen


try:  # pragma: no cover
    import pgenlib

    HAS_PGENLIB = True
except Exception:  # pragma: no cover
    HAS_PGENLIB = False


pytestmark = pytest.mark.skipif(not HAS_PGENLIB, reason="pgenlib is required for PGEN tests")


def _write_triplet(prefix, *, allele_rows: np.ndarray, phase_rows: np.ndarray | None = None, alt_field: str = "G"):
    prefix = prefix
    pgen_path = str(prefix.with_suffix(".pgen"))
    pvar_path = prefix.with_suffix(".pvar")
    psam_path = prefix.with_suffix(".psam")

    n_variants = int(allele_rows.shape[0])
    if n_variants <= 0:
        raise ValueError("need at least one variant")
    if allele_rows.shape[1] % 2 != 0:
        raise ValueError("allele_rows must be shape (V, 2N)")
    n_samples = int(allele_rows.shape[1] // 2)

    with pgenlib.PgenWriter(
        pgen_path.encode("utf-8"),
        n_samples,
        variant_ct=n_variants,
        nonref_flags=False,
        hardcall_phase_present=True,
        allele_ct_limit=2,
    ) as writer:
        if phase_rows is None:
            writer.append_alleles_batch(np.ascontiguousarray(allele_rows, dtype=np.int32), all_phased=True)
        else:
            writer.append_partially_phased_batch(
                np.ascontiguousarray(allele_rows, dtype=np.int32),
                np.ascontiguousarray(phase_rows, dtype=np.uint8),
            )

    with pvar_path.open("w", encoding="utf-8") as handle:
        handle.write("#CHROM\tPOS\tID\tREF\tALT\n")
        for i in range(n_variants):
            handle.write(f"chr1\t{100 + i}\tv{i+1}\tA\t{alt_field}\n")

    with psam_path.open("w", encoding="utf-8") as handle:
        handle.write("#FID\tIID\tPAT\tMAT\tSEX\tPHENO\n")
        for i in range(n_samples):
            handle.write(f"0\ts{i+1}\t0\t0\t0\t-9\n")

    return prefix


def test_ingest_pgen_unphased_policy_error(tmp_path):
    prefix = _write_triplet(
        tmp_path / "unphased",
        allele_rows=np.asarray([[0, 1]], dtype=np.int32),
        phase_rows=np.asarray([[0]], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="unphased heterozygous genotype"):
        ingest_pgen(
            pgen_prefix=prefix,
            output_zarr=tmp_path / "out_error.zarr",
            chrom="1",
            on_unphased_gt="error",
        )


def test_ingest_pgen_unphased_policy_error_parallel(tmp_path):
    prefix = _write_triplet(
        tmp_path / "unphased_parallel",
        allele_rows=np.asarray([[0, 1], [1, 1]], dtype=np.int32),
        phase_rows=np.asarray([[0], [1]], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="unphased heterozygous genotype"):
        ingest_pgen(
            pgen_prefix=prefix,
            output_zarr=tmp_path / "out_error_parallel.zarr",
            chrom="1",
            on_unphased_gt="error",
            workers=2,
            variant_block_size=1,
        )


def test_ingest_pgen_unphased_policy_missing(tmp_path):
    prefix = _write_triplet(
        tmp_path / "unphased_missing",
        allele_rows=np.asarray([[0, 1]], dtype=np.int32),
        phase_rows=np.asarray([[0]], dtype=np.uint8),
    )

    out = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "out_missing.zarr",
        chrom="1",
        on_unphased_gt="missing",
    )

    h = out.haplotypes("1")
    assert h.shape == (2, 1)
    assert int(h[0, 0]) == 255
    assert int(h[1, 0]) == 255

    extra = out.attrs("1").get("extra")
    assert isinstance(extra, dict)
    assert int(extra.get("unphased_gt_count", -1)) == 1


def test_ingest_pgen_unphased_policy_missing_parallel(tmp_path):
    prefix = _write_triplet(
        tmp_path / "unphased_missing_parallel",
        allele_rows=np.asarray([[0, 1], [1, 1]], dtype=np.int32),
        phase_rows=np.asarray([[0], [1]], dtype=np.uint8),
    )

    out = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "out_missing_parallel.zarr",
        chrom="1",
        on_unphased_gt="missing",
        workers=2,
        variant_block_size=1,
    )

    h = out.haplotypes("1")
    assert h.shape == (2, 2)
    assert int(h[0, 0]) == 255
    assert int(h[1, 0]) == 255
    assert int(h[0, 1]) == 1
    assert int(h[1, 1]) == 1

    extra = out.attrs("1").get("extra")
    assert isinstance(extra, dict)
    assert int(extra.get("unphased_gt_count", -1)) == 1


def test_ingest_pgen_rejects_non_biallelic_pvar(tmp_path):
    prefix = _write_triplet(
        tmp_path / "multi_alt",
        allele_rows=np.asarray([[0, 1]], dtype=np.int32),
        phase_rows=np.asarray([[1]], dtype=np.uint8),
        alt_field="G,C",
    )

    with pytest.raises(ValueError, match="non-biallelic"):
        ingest_pgen(
            pgen_prefix=prefix,
            output_zarr=tmp_path / "out_multi.zarr",
            chrom="1",
            strict_biallelic=True,
        )


def test_ingest_pgen_rejects_non_biallelic_parallel(tmp_path):
    prefix = _write_triplet(
        tmp_path / "multi_alt_parallel",
        allele_rows=np.asarray([[0, 1], [0, 1]], dtype=np.int32),
        phase_rows=np.asarray([[1], [1]], dtype=np.uint8),
        alt_field="G,C",
    )

    with pytest.raises(ValueError, match="non-biallelic"):
        ingest_pgen(
            pgen_prefix=prefix,
            output_zarr=tmp_path / "out_multi_parallel.zarr",
            chrom="1",
            strict_biallelic=True,
            workers=2,
            variant_block_size=1,
        )


def test_ingest_pgen_sample_subset_order(tmp_path):
    prefix = _write_triplet(
        tmp_path / "subset",
        allele_rows=np.asarray(
            [
                [0, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 1],
            ],
            dtype=np.int32,
        ),
        phase_rows=np.asarray(
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.uint8,
        ),
    )

    out = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "out_subset.zarr",
        chrom="1",
        sample_ids=["s3", "s1"],
    )

    assert out.sample_ids("1") == ["s3", "s1"]
    h = out.haplotypes("1")
    expected = np.asarray(
        [
            [1, 0],  # s3 hap0
            [0, 1],  # s3 hap1
            [0, 1],  # s1 hap0
            [1, 0],  # s1 hap1
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(h, expected)

    hs = HapStore(tmp_path / "out_subset.zarr", mode="r")
    assert hs.n_samples("1") == 2


def test_ingest_pgen_sample_subset_order_parallel(tmp_path):
    prefix = _write_triplet(
        tmp_path / "subset_parallel",
        allele_rows=np.asarray(
            [
                [0, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 0],
            ],
            dtype=np.int32,
        ),
        phase_rows=np.asarray(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.uint8,
        ),
    )

    out = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "out_subset_parallel.zarr",
        chrom="1",
        sample_ids=["s3", "s1"],
        workers=2,
        variant_block_size=1,
    )

    assert out.sample_ids("1") == ["s3", "s1"]
    h = out.haplotypes("1")
    expected = np.asarray(
        [
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(h, expected)


def test_ingest_pgen_parallel_matches_serial_roundtrip(tmp_path):
    allele_rows = np.asarray(
        [
            [0, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 1, 0],
        ],
        dtype=np.int32,
    )
    phase_rows = np.ones((allele_rows.shape[0], allele_rows.shape[1] // 2), dtype=np.uint8)
    prefix = _write_triplet(
        tmp_path / "serial_vs_parallel",
        allele_rows=allele_rows,
        phase_rows=phase_rows,
    )

    serial = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "serial.zarr",
        chrom="1",
        variant_block_size=2,
        workers=1,
    )
    parallel = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "parallel.zarr",
        chrom="1",
        variant_block_size=2,
        workers=2,
    )

    np.testing.assert_array_equal(serial.haplotypes("1"), parallel.haplotypes("1"))
    assert serial.variant_positions("1").tolist() == parallel.variant_positions("1").tolist()
    assert serial.sample_ids("1") == parallel.sample_ids("1")
    assert serial.attrs("1").get("extra") == parallel.attrs("1").get("extra")
