from __future__ import annotations

import numpy as np
import pytest

from hapstore import HapStore, concat_hapstores, ingest_arrays


def _split_fixture_arrays(synthetic_arrays: dict) -> tuple[dict, dict]:
    h = synthetic_arrays["haplotypes"]
    left = {
        "haplotypes": np.asarray(h[0:2, :], dtype=np.uint8),
        "variant_pos": synthetic_arrays["variant_pos"],
        "sample_ids": np.asarray(["s1"], dtype=object),
        "ref": synthetic_arrays["ref"],
        "alt": synthetic_arrays["alt"],
        "vid": synthetic_arrays["vid"],
        "genetic_positions": synthetic_arrays["genetic_positions"],
        "populations": np.asarray(["P1"], dtype=object),
    }
    right = {
        "haplotypes": np.asarray(h[2:4, :], dtype=np.uint8),
        "variant_pos": synthetic_arrays["variant_pos"],
        "sample_ids": np.asarray(["s2"], dtype=object),
        "ref": synthetic_arrays["ref"],
        "alt": synthetic_arrays["alt"],
        "vid": synthetic_arrays["vid"],
        "genetic_positions": synthetic_arrays["genetic_positions"],
        "populations": np.asarray(["P2"], dtype=object),
    }
    return left, right


def test_concat_samples_roundtrip(tmp_path, synthetic_arrays):
    left, right = _split_fixture_arrays(synthetic_arrays)
    s1 = ingest_arrays(output_zarr=tmp_path / "left.zarr", chrom="chr21", **left)
    s2 = ingest_arrays(output_zarr=tmp_path / "right.zarr", chrom=21, **right)

    out_path = tmp_path / "concat_samples.zarr"
    concat_hapstores(
        input_stores=[s1.path, s2.path],
        output_store=out_path,
        chrom="chr21",
        axis="samples",
        validate="full",
        n_workers=1,
        chunk_variant_size=2,
        chunk_haplotype_size=2,
        chunk_sample_size=1,
        compression_level=3,
        store_genotypes=True,
        attrs={"source": "concat_test"},
        overwrite=False,
    )

    merged = HapStore(out_path, mode="r")
    assert merged.has_chromosome("chr21")
    assert merged.sample_ids("chr21") == ["s1", "s2"]
    assert merged.n_samples("chr21") == 2
    assert merged.n_haplotypes("chr21") == 4
    assert merged.n_variants("chr21") == 4

    expected_h = np.concatenate([left["haplotypes"], right["haplotypes"]], axis=0)
    np.testing.assert_array_equal(merged.haplotypes("chr21"), expected_h)
    np.testing.assert_array_equal(merged.variant_positions("chr21"), synthetic_arrays["variant_pos"])

    attrs = merged.attrs("chr21")
    assert attrs["source"] == "concat_test"
    assert attrs["chunk_variant_size"] == 2
    assert attrs["chunk_haplotype_size"] == 2
    assert attrs["chunk_sample_size"] == 1


def test_concat_samples_parallel_equals_serial(tmp_path, synthetic_arrays):
    left, right = _split_fixture_arrays(synthetic_arrays)
    s1 = ingest_arrays(output_zarr=tmp_path / "left.zarr", chrom="chr21", **left)
    s2 = ingest_arrays(output_zarr=tmp_path / "right.zarr", chrom="chr21", **right)

    out_serial = tmp_path / "serial.zarr"
    out_parallel = tmp_path / "parallel.zarr"

    concat_hapstores(
        input_stores=[s1.path, s2.path],
        output_store=out_serial,
        chrom="chr21",
        axis="samples",
        validate="spot",
        n_workers=1,
        chunk_variant_size=2,
        chunk_haplotype_size=2,
        compression_level=3,
        overwrite=False,
    )
    concat_hapstores(
        input_stores=[s1.path, s2.path],
        output_store=out_parallel,
        chrom="chr21",
        axis="samples",
        validate="spot",
        n_workers=2,
        chunk_variant_size=2,
        chunk_haplotype_size=2,
        compression_level=3,
        overwrite=False,
    )

    a = HapStore(out_serial, mode="r")
    b = HapStore(out_parallel, mode="r")
    np.testing.assert_array_equal(a.haplotypes("chr21"), b.haplotypes("chr21"))
    np.testing.assert_array_equal(a.variant_positions("chr21"), b.variant_positions("chr21"))
    assert a.sample_ids("chr21") == b.sample_ids("chr21")


def test_concat_variants_roundtrip(tmp_path):
    sample_ids = np.asarray(["s1", "s2"], dtype=object)
    pops = np.asarray(["P1", "P1"], dtype=object)

    h_left = np.asarray(
        [
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
        ],
        dtype=np.uint8,
    )
    h_right = np.asarray(
        [
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    left = ingest_arrays(
        output_zarr=tmp_path / "left.zarr",
        chrom="chr21",
        haplotypes=h_left,
        variant_pos=np.asarray([100, 200], dtype=np.int64),
        sample_ids=sample_ids,
        ref=np.asarray(["A", "C"], dtype=object),
        alt=np.asarray(["G", "T"], dtype=object),
        vid=np.asarray(["v1", "v2"], dtype=object),
        genetic_positions=np.asarray([0.1, 0.2], dtype=np.float64),
        populations=pops,
    )
    right = ingest_arrays(
        output_zarr=tmp_path / "right.zarr",
        chrom="chr21",
        haplotypes=h_right,
        variant_pos=np.asarray([300, 400, 500], dtype=np.int64),
        sample_ids=sample_ids,
        ref=np.asarray(["G", "T", "A"], dtype=object),
        alt=np.asarray(["A", "C", "G"], dtype=object),
        vid=np.asarray(["v3", "v4", "v5"], dtype=object),
        genetic_positions=np.asarray([0.3, 0.4, 0.5], dtype=np.float64),
        populations=pops,
    )

    out_path = tmp_path / "concat_vars.zarr"
    concat_hapstores(
        input_stores=[left.path, right.path],
        output_store=out_path,
        chrom="chr21",
        axis="variants",
        validate="full",
        n_workers=1,
        chunk_variant_size=2,
        chunk_haplotype_size=4,
        compression_level=3,
        overwrite=False,
    )

    out = HapStore(out_path, mode="r")
    expected = np.concatenate([h_left, h_right], axis=1)
    np.testing.assert_array_equal(out.haplotypes("chr21"), expected)
    np.testing.assert_array_equal(out.variant_positions("chr21"), np.asarray([100, 200, 300, 400, 500], dtype=np.int64))


def test_concat_variants_duplicate_vid_raises_full_validate(tmp_path):
    sample_ids = np.asarray(["s1", "s2"], dtype=object)
    pops = np.asarray(["P1", "P1"], dtype=object)
    h = np.asarray([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.uint8)

    left = ingest_arrays(
        output_zarr=tmp_path / "left.zarr",
        chrom="chr21",
        haplotypes=h,
        variant_pos=np.asarray([100, 200], dtype=np.int64),
        sample_ids=sample_ids,
        ref=np.asarray(["A", "C"], dtype=object),
        alt=np.asarray(["G", "T"], dtype=object),
        vid=np.asarray(["dup", "v2"], dtype=object),
        genetic_positions=np.asarray([0.1, 0.2], dtype=np.float64),
        populations=pops,
    )
    right = ingest_arrays(
        output_zarr=tmp_path / "right.zarr",
        chrom="chr21",
        haplotypes=h,
        variant_pos=np.asarray([300, 400], dtype=np.int64),
        sample_ids=sample_ids,
        ref=np.asarray(["G", "T"], dtype=object),
        alt=np.asarray(["A", "C"], dtype=object),
        vid=np.asarray(["v3", "dup"], dtype=object),
        genetic_positions=np.asarray([0.3, 0.4], dtype=np.float64),
        populations=pops,
    )

    with pytest.raises(ValueError, match="duplicate variant IDs"):
        concat_hapstores(
            input_stores=[left.path, right.path],
            output_store=tmp_path / "out.zarr",
            chrom="chr21",
            axis="variants",
            validate="full",
            forbid_duplicate_variants=True,
        )


def test_overwrite_and_tmp_suffix_behavior(tmp_path, synthetic_arrays):
    left, right = _split_fixture_arrays(synthetic_arrays)
    s1 = ingest_arrays(output_zarr=tmp_path / "left.zarr", chrom="chr21", **left)
    s2 = ingest_arrays(output_zarr=tmp_path / "right.zarr", chrom="chr21", **right)

    out_path = tmp_path / "out.zarr"
    concat_hapstores(
        input_stores=[s1.path, s2.path],
        output_store=out_path,
        chrom="chr21",
        axis="samples",
        validate="spot",
        overwrite=False,
    )

    with pytest.raises(FileExistsError):
        concat_hapstores(
            input_stores=[s1.path, s2.path],
            output_store=out_path,
            chrom="chr21",
            axis="samples",
            validate="spot",
            overwrite=False,
        )

    concat_hapstores(
        input_stores=[s1.path, s2.path],
        output_store=out_path,
        chrom="chr21",
        axis="samples",
        validate="spot",
        overwrite=True,
    )

