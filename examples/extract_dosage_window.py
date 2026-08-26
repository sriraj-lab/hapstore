"""Extract one genetic window from HapStore as a NumPy dosage matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hapstore import HapStore


def parse_args() -> argparse.Namespace:
    """Parse a store, chromosome, output path, and window selector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store_path", type=Path)
    parser.add_argument("chrom")
    parser.add_argument("output_npy", type=Path)
    parser.add_argument("--window-size-cm", type=float, default=0.5)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--workers", default="auto")
    return parser.parse_args()


def main() -> None:
    """Read one cM window and write its dosage matrix as a NumPy file."""
    args = parse_args()
    store = HapStore(args.store_path, mode="r")
    boundaries = store.compute_windows(args.chrom, window_size_cm=args.window_size_cm)
    if not 0 <= args.window_index < boundaries.shape[0]:
        raise IndexError(
            f"window_index must be in [0, {boundaries.shape[0]}), got {args.window_index}"
        )

    start, stop = (int(value) for value in boundaries[args.window_index])
    dosages = store.read_dosages(
        args.chrom,
        variant_indices=np.arange(start, stop, dtype=np.int64),
        dtype="float32",
        workers=args.workers,
    )
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, dosages)
    print(
        f"wrote {args.output_npy}: shape={dosages.shape}, "
        f"variants=[{start}, {stop})"
    )


if __name__ == "__main__":
    main()
