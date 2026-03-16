"""Window boundary helpers for hapstore."""

from __future__ import annotations

import numpy as np


def compute_window_boundaries(
    genetic_positions_cm: np.ndarray,
    window_size_cm: float,
) -> np.ndarray:
    """Compute non-overlapping [start, end) variant index windows in cM space."""
    cm = np.asarray(genetic_positions_cm, dtype=np.float64)
    if cm.ndim != 1:
        raise ValueError(f"genetic_positions_cm must be 1D, got shape={cm.shape}")
    if window_size_cm <= 0:
        raise ValueError(f"window_size_cm must be positive, got {window_size_cm}")
    if cm.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    diffs = np.diff(cm)
    if np.any(diffs < 0):
        bad = int(np.where(diffs < 0)[0][0])
        raise ValueError(
            "genetic_positions_cm must be monotonically non-decreasing; "
            f"found decrease at index {bad}: {cm[bad]} -> {cm[bad + 1]}"
        )

    cm_start = float(cm[0])
    cm_end = float(cm[-1])

    n_steps = int(np.floor((cm_end - cm_start) / float(window_size_cm))) + 1
    windows: list[tuple[int, int]] = []
    for k in range(n_steps):
        left = cm_start + (k * float(window_size_cm))
        right = left + float(window_size_cm)
        start_idx = int(np.searchsorted(cm, left, side="left"))
        end_idx = int(np.searchsorted(cm, right, side="left"))
        if end_idx > start_idx:
            windows.append((start_idx, end_idx))

    if not windows:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(windows, dtype=np.int64)


def compute_window_midpoints(
    variant_positions_bp: np.ndarray,
    genetic_positions_cm: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute midpoint coordinates for each [start, end) index window."""
    bp = np.asarray(variant_positions_bp, dtype=np.int64)
    cm = np.asarray(genetic_positions_cm, dtype=np.float64)
    bounds = np.asarray(boundaries, dtype=np.int64)

    if bp.ndim != 1:
        raise ValueError(f"variant_positions_bp must be 1D, got shape={bp.shape}")
    if cm.ndim != 1:
        raise ValueError(f"genetic_positions_cm must be 1D, got shape={cm.shape}")
    if bp.shape[0] != cm.shape[0]:
        raise ValueError(
            "variant_positions_bp and genetic_positions_cm length mismatch: "
            f"{bp.shape[0]} != {cm.shape[0]}"
        )
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"boundaries must have shape (W, 2), got shape={bounds.shape}")

    mid_bp = np.empty(bounds.shape[0], dtype=np.int64)
    mid_cm = np.empty(bounds.shape[0], dtype=np.float64)

    n_variants = bp.shape[0]
    for i, (start, end) in enumerate(bounds):
        if start < 0 or end < 0 or start >= end or end > n_variants:
            raise ValueError(
                f"invalid boundary at row {i}: start={start}, end={end}, n_variants={n_variants}"
            )
        bp_slice = bp[start:end]
        cm_slice = cm[start:end]
        mid_bp[i] = int(np.round(np.median(bp_slice)))
        mid_cm[i] = float(np.median(cm_slice))

    return mid_bp, mid_cm
