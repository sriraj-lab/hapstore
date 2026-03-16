from __future__ import annotations

import numpy as np
import pytest

from hapstore import HapStore


def test_create_and_read_roundtrip(tmp_path, synthetic_arrays):
    store_path = tmp_path / "panel.zarr"
    store = HapStore(store_path, mode="w")

    store.create_chromosome(chrom="1", **synthetic_arrays)

    assert store.has_chromosome("chr1")
    assert store.has_chromosome("1")
    assert store.chromosomes == ["chr1"]
    assert store.n_samples("chr1") == 2
    assert store.n_haplotypes("chr1") == 4
    assert store.n_variants("chr1") == 4

    h = store.haplotypes("1")
    np.testing.assert_array_equal(h, synthetic_arrays["haplotypes"])

    info = store.variant_info("chr1")
    np.testing.assert_array_equal(info["pos"], synthetic_arrays["variant_pos"])
    assert info["vid"] == synthetic_arrays["vid"].tolist()
    assert info["ref"] == synthetic_arrays["ref"].tolist()
    assert info["alt"] == synthetic_arrays["alt"].tolist()
    np.testing.assert_array_equal(info["genetic_pos"], synthetic_arrays["genetic_positions"])

    attrs = store.attrs("chr1")
    assert attrs["missing_sentinel"] == 255
    assert attrs["has_missing"] is True


def test_duplicate_chromosome_raises(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "dup.zarr", mode="w")
    store.create_chromosome(chrom="1", **synthetic_arrays)

    with pytest.raises(ValueError, match="already exists"):
        store.create_chromosome(chrom="chr1", **synthetic_arrays)


def test_delete_chromosome(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "delete.zarr", mode="w")
    store.create_chromosome(chrom="1", **synthetic_arrays)

    store.delete_chromosome("1")
    assert store.has_chromosome("chr1") is False


def test_haplotype_variant_slice_and_sample_mask(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "slice.zarr", mode="w")
    store.create_chromosome(chrom="1", **synthetic_arrays)

    sub = store.haplotypes("chr1", variant_slice=(1, 3))
    np.testing.assert_array_equal(sub, synthetic_arrays["haplotypes"][:, 1:3])

    sample_mask = np.asarray([True, False])
    sub_masked = store.haplotypes("chr1", sample_mask=sample_mask)
    np.testing.assert_array_equal(sub_masked, synthetic_arrays["haplotypes"][0:2, :])


def test_create_chromosome_with_2d_chunks_enables_sample_subset_reads(tmp_path, synthetic_arrays):
    store_path = tmp_path / "chunks_v2.zarr"
    store = HapStore(store_path, mode="w")
    store.create_chromosome(
        chrom="1",
        **synthetic_arrays,
        chunk_variant_size=2,
        chunk_haplotype_size=2,
        chunk_sample_size=1,
        store_genotypes=True,
    )

    root = store.root
    assert int(root.attrs["schema_version"]) >= 2

    chrom = root["chr1"]
    assert tuple(chrom["haplotypes"].chunks) == (2, 2)
    assert tuple(chrom["genotypes"].chunks) == (1, 2)

    mask = np.asarray([False, True])
    h = store.haplotypes("1", sample_mask=mask)
    np.testing.assert_array_equal(h, synthetic_arrays["haplotypes"][2:4, :])

    g = store.genotypes("1", sample_mask=mask)
    np.testing.assert_array_equal(g, store.genotypes("1")[1:2, :])


def test_genotypes_on_the_fly_and_stored(tmp_path, synthetic_arrays):
    expected = np.asarray(
        [
            [1, 2, 0, 255],
            [0, 1, 2, 255],
        ],
        dtype=np.uint8,
    )

    store_a = HapStore(tmp_path / "geno_a.zarr", mode="w")
    store_a.create_chromosome(chrom="1", **synthetic_arrays, store_genotypes=False)
    got_a = store_a.genotypes("1")
    np.testing.assert_array_equal(got_a, expected)

    store_b = HapStore(tmp_path / "geno_b.zarr", mode="w")
    store_b.create_chromosome(chrom="1", **synthetic_arrays, store_genotypes=True)
    got_b = store_b.genotypes("1")
    np.testing.assert_array_equal(got_b, expected)


def test_mode_enforcement_and_missing_chrom(tmp_path, synthetic_arrays):
    store_path = tmp_path / "mode.zarr"
    writer = HapStore(store_path, mode="w")
    writer.create_chromosome(chrom="1", **synthetic_arrays)

    reader = HapStore(store_path, mode="r")

    with pytest.raises(ValueError, match="read-only"):
        reader.create_chromosome(chrom="2", **synthetic_arrays)

    with pytest.raises(ValueError, match="read-only"):
        reader.delete_chromosome("1")

    with pytest.raises(KeyError, match="not found"):
        reader.haplotypes("chr2")


def test_cross_chrom_sample_consistency_check(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "sample_consistency.zarr", mode="w")
    store.create_chromosome(chrom="1", **synthetic_arrays)

    bad = dict(synthetic_arrays)
    bad["sample_ids"] = np.asarray(["x1", "x2"], dtype=object)
    with pytest.raises(ValueError, match="sample ID ordering mismatch"):
        store.create_chromosome(chrom="2", **bad)
