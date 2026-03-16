from __future__ import annotations

import numpy as np

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



def test_concat_hapstores_samples_roundtrip(tmp_path, synthetic_arrays):
    left, right = _split_fixture_arrays(synthetic_arrays)

    s1 = ingest_arrays(output_zarr=tmp_path / "left.zarr", chrom="chr21", **left)
    s2 = ingest_arrays(output_zarr=tmp_path / "right.zarr", chrom=21, **right)

    out_path = tmp_path / "merged.zarr"
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
        attrs={"source": "merge_test"},
        overwrite=True,
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

    info = merged.variant_info("chr21")
    assert info["ref"] == synthetic_arrays["ref"].tolist()
    assert info["alt"] == synthetic_arrays["alt"].tolist()
    assert info["vid"] == synthetic_arrays["vid"].tolist()
    np.testing.assert_array_equal(info["genetic_pos"], synthetic_arrays["genetic_positions"])

    attrs = merged.attrs("chr21")
    assert attrs["has_missing"] is True
    assert attrs["source"] == "merge_test"
    assert attrs["chunk_variant_size"] == 2
    assert attrs["chunk_haplotype_size"] == 2
    assert attrs["chunk_sample_size"] == 1
    assert attrs["layout_version"] == 2

    chrom_group = merged.root["chr21"]
    assert tuple(chrom_group["haplotypes"].chunks) == (2, 2)
    assert tuple(chrom_group["genotypes"].chunks) == (1, 2)



def test_concat_hapstores_does_not_require_variant_info_calls(
    tmp_path,
    synthetic_arrays,
    monkeypatch,
):
    left, right = _split_fixture_arrays(synthetic_arrays)
    s1 = ingest_arrays(output_zarr=tmp_path / "left.zarr", chrom="chr21", **left)
    s2 = ingest_arrays(output_zarr=tmp_path / "right.zarr", chrom="chr21", **right)

    def _bomb_variant_info(*args, **kwargs):
        raise AssertionError("merge should stream from zarr variant arrays, not call HapStore.variant_info")

    monkeypatch.setattr(HapStore, "variant_info", _bomb_variant_info)
    out_path = tmp_path / "merged_no_variant_info.zarr"
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
        overwrite=True,
    )

    merged = HapStore(out_path, mode="r")
    assert merged.n_samples("chr21") == 2
    assert merged.n_haplotypes("chr21") == 4
