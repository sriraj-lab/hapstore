from __future__ import annotations

import numpy as np
import pytest

from hapstore import HapStore, HapStoreDosageRequest
from hapstore import store as store_module


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


def test_read_dosages_contiguous_matches_genotypes(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "dosage_contiguous.zarr", mode="w")
    store.create_chromosome(
        chrom="1",
        **synthetic_arrays,
        chunk_variant_size=2,
        chunk_haplotype_size=2,
    )

    got = store.read_dosages("1", variant_indices=np.asarray([0, 1, 2]), workers=2)
    expected = store.genotypes("1", variant_slice=(0, 3)).astype(np.float32)
    np.testing.assert_array_equal(got, expected)


def test_read_dosages_unsorted_variant_ids_preserve_request_order(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "dosage_ids.zarr", mode="w")
    store.create_chromosome(
        chrom="1",
        **synthetic_arrays,
        chunk_variant_size=1,
        chunk_haplotype_size=2,
    )

    got = store.read_dosages("chr1", variant_ids=["v3", "v1", "v2"], workers=4)
    expected = store.genotypes("1")[:, [2, 0, 1]].astype(np.float32)
    np.testing.assert_array_equal(got, expected)


def test_read_dosages_worker_counts_are_equal(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "dosage_workers.zarr", mode="w")
    store.create_chromosome(
        chrom="1",
        **synthetic_arrays,
        chunk_variant_size=1,
        chunk_haplotype_size=2,
        store_genotypes=True,
    )

    request = np.asarray([3, 0, 2, 1], dtype=np.int64)
    one = store.read_dosages("1", variant_indices=request, sample_indices=[1, 0], workers=1)
    many = store.read_dosages("1", variant_indices=request, sample_indices=[1, 0], workers=4)
    np.testing.assert_allclose(one, many, equal_nan=True)


def test_read_dosages_auto_workers_match_serial(tmp_path, synthetic_arrays, monkeypatch):
    store = HapStore(tmp_path / "dosage_auto_workers.zarr", mode="w")
    store.create_chromosome(
        chrom="1",
        **synthetic_arrays,
        chunk_variant_size=1,
        chunk_haplotype_size=2,
    )

    monkeypatch.setenv("HAPSTORE_READ_WORKERS", "3")
    request = np.asarray([3, 0, 2, 1], dtype=np.int64)
    auto = store.read_dosages("1", variant_indices=request, workers="auto")
    metrics = store.last_read_metrics
    serial = store.read_dosages("1", variant_indices=request, workers=1)
    np.testing.assert_allclose(auto, serial, equal_nan=True)
    assert metrics["auto_worker_source"] == "HAPSTORE_READ_WORKERS"
    assert 1 <= int(metrics["workers"]) <= 3


def test_read_dosages_auto_default_cap_is_sixteen(monkeypatch):
    monkeypatch.delenv("HAPSTORE_READ_WORKERS", raising=False)
    monkeypatch.delenv("HAPSTORE_AUTO_READ_WORKERS", raising=False)
    monkeypatch.delenv("HAPSTORE_READ_AUTO_MAX_WORKERS", raising=False)
    monkeypatch.delenv("HAPSTORE_READ_AUTO_MAX", raising=False)
    monkeypatch.setattr(store_module.os, "cpu_count", lambda: 32)

    workers, policy = store_module._resolve_read_worker_count(
        "auto",
        n_tasks=100,
        n_sample_groups=10,
        n_variant_groups=10,
        n_samples=150_000,
        n_variants=5_000,
        row_chunk=7_500,
        variant_chunk=500,
        dataset="haplotypes",
    )

    assert workers == 16
    assert policy["auto_worker_cap"] == 16
    assert policy["auto_worker_cap_source"] == "default"
    assert policy["auto_worker_source"] == "large_tile_grid"


def test_read_dosages_missing_modes(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "dosage_missing.zarr", mode="w")
    store.create_chromosome(chrom="1", **synthetic_arrays, chunk_variant_size=1, chunk_haplotype_size=2)

    as_nan = store.read_dosages("1", variant_indices=[3], missing="nan", workers=2)
    assert np.isnan(as_nan[0, 0])
    assert np.isnan(as_nan[1, 0])

    as_sentinel = store.read_dosages("1", variant_indices=[3], missing="sentinel", workers=2)
    np.testing.assert_array_equal(as_sentinel, np.asarray([[255.0], [255.0]], dtype=np.float32))


def test_read_dosages_stored_and_haplotype_only_are_equal(tmp_path, synthetic_arrays):
    hap_only = HapStore(tmp_path / "dosage_hap_only.zarr", mode="w")
    hap_only.create_chromosome(chrom="1", **synthetic_arrays, chunk_variant_size=1, chunk_haplotype_size=2)

    stored = HapStore(tmp_path / "dosage_stored.zarr", mode="w")
    stored.create_chromosome(
        chrom="1",
        **synthetic_arrays,
        chunk_variant_size=1,
        chunk_haplotype_size=2,
        chunk_sample_size=1,
        store_genotypes=True,
    )

    request = ["v4", "v1", "v3"]
    got_hap = hap_only.read_dosages("1", variant_ids=request, workers=4)
    got_stored = stored.read_dosages("1", variant_ids=request, workers=4)
    np.testing.assert_allclose(got_hap, got_stored, equal_nan=True)


def test_read_dosage_batch_window_matrix_and_project(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "dosage_batch.zarr", mode="w")
    store.create_chromosome(
        chrom="1",
        **synthetic_arrays,
        chunk_variant_size=1,
        chunk_haplotype_size=2,
    )

    batch = store.read_dosage_batch(
        "1",
        [
            HapStoreDosageRequest("a", variant_ids=["v3", "v1"]),
            {"request_id": "b", "variant_indices": [1, 2]},
        ],
        workers=4,
    )
    expected_a = store.read_dosages("1", variant_ids=["v3", "v1"], workers=1)
    np.testing.assert_allclose(batch.window_matrix("a"), expected_a, equal_nan=True)

    weights = np.asarray([0.5, -1.0], dtype=np.float32)
    np.testing.assert_allclose(
        batch.project("a", weights),
        expected_a @ weights,
        equal_nan=True,
    )


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
