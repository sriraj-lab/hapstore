from __future__ import annotations

import numpy as np
import pytest

from hapstore import HapStore, compute_window_boundaries, compute_window_midpoints


def test_compute_window_boundaries_uniform_spacing():
    cm = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)
    boundaries = compute_window_boundaries(cm, window_size_cm=0.25)
    expected = np.asarray([[0, 3], [3, 5], [5, 8], [8, 10]], dtype=np.int64)
    np.testing.assert_array_equal(boundaries, expected)


def test_compute_window_boundaries_drops_empty_windows():
    cm = np.asarray([0.0, 1.0], dtype=float)
    boundaries = compute_window_boundaries(cm, window_size_cm=0.4)
    expected = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(boundaries, expected)


def test_compute_window_midpoints():
    bp = np.asarray([100, 200, 300, 400], dtype=np.int64)
    cm = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
    boundaries = np.asarray([[0, 2], [2, 4]], dtype=np.int64)

    mid_bp, mid_cm = compute_window_midpoints(bp, cm, boundaries)
    np.testing.assert_array_equal(mid_bp, np.asarray([150, 350], dtype=np.int64))
    np.testing.assert_allclose(mid_cm, np.asarray([0.15, 0.35], dtype=float))


def test_hapstore_compute_windows_store_and_iter(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "windows.zarr", mode="w")
    store.create_chromosome(chrom="1", **synthetic_arrays)

    boundaries = store.compute_windows("1", window_size_cm=0.25, store=True)
    np.testing.assert_array_equal(boundaries, np.asarray([[0, 2], [2, 3], [3, 4]], dtype=np.int64))

    chrom_group = store.root["chr1"]
    assert "windows" in chrom_group
    assert "cm_0p25" in chrom_group["windows"]

    windows = list(store.iter_windows("1", window_size_cm=0.25))
    assert len(windows) == 3
    assert windows[0][0] == 0
    assert windows[0][1] == 0
    assert windows[0][2] == 2
    np.testing.assert_array_equal(windows[0][3], synthetic_arrays["haplotypes"][:, 0:2])


def test_iter_windows_requires_boundaries_or_stored(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "iter_fail.zarr", mode="w")
    store.create_chromosome(chrom="1", **synthetic_arrays)

    with pytest.raises(ValueError, match="stored windows not found"):
        list(store.iter_windows("1"))


def test_iter_windows_multiple_namespaces_requires_selection(tmp_path, synthetic_arrays):
    store = HapStore(tmp_path / "iter_multi.zarr", mode="w")
    store.create_chromosome(chrom="1", **synthetic_arrays)
    store.compute_windows("1", window_size_cm=0.25, store=True)
    store.compute_windows("1", window_size_cm=0.5, store=True)

    with pytest.raises(ValueError, match="multiple stored window namespaces"):
        list(store.iter_windows("1"))


def test_compute_window_boundaries_validation():
    with pytest.raises(ValueError, match="positive"):
        compute_window_boundaries(np.asarray([0.0, 0.1]), window_size_cm=0.0)

    with pytest.raises(ValueError, match="monotonically"):
        compute_window_boundaries(np.asarray([0.0, 0.2, 0.1]), window_size_cm=0.1)
