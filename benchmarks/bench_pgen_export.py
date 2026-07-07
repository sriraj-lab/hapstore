#!/usr/bin/env python
"""Benchmark PGEN-export-adjacent HapStore read components.

This component benchmark is intentionally smaller than
``bench_read_dosages.py``. It focuses on read patterns that matter for direct
PGEN export: variant metadata reads, high-fraction sample-mask reads, and
variant block-size choices.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hapstore import HapStore  # noqa: E402
from hapstore._internal.zarr_utils import _to_unicode_array  # noqa: E402


def _parse_int_csv(value: str) -> list[int]:
    out = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("all integers must be positive")
    return out


def _chrom_key(chrom: str | int) -> str:
    text = str(chrom)
    return text if text.startswith("chr") else f"chr{text}"


def _median_seconds(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _time_variant_metadata_reads(
    *,
    variants,
    n_variants: int,
    reps: int,
    block_sizes: Sequence[int],
) -> None:
    pos_ds = variants["pos"]
    ref_ds = variants["ref"]
    alt_ds = variants["alt"]
    vid_ds = variants["vid"]

    print("\n=== Variant metadata reads ===")
    for block_size in block_sizes:
        times: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for start in range(0, n_variants, int(block_size)):
                end = min(n_variants, start + int(block_size))
                _ = np.asarray(pos_ds[start:end], dtype=np.int64)
                _ = _to_unicode_array(np.asarray(ref_ds[start:end], dtype=object))
                _ = _to_unicode_array(np.asarray(alt_ds[start:end], dtype=object))
                _ = _to_unicode_array(np.asarray(vid_ds[start:end], dtype=object))
            times.append(time.perf_counter() - t0)
        n_blocks = (n_variants + int(block_size) - 1) // int(block_size)
        print(
            f"  per-block bs={int(block_size):5d} ({n_blocks:4d} blocks): "
            f"{_median_seconds(times):.3f}s"
        )

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = np.asarray(pos_ds[:], dtype=np.int64)
        _ = _to_unicode_array(np.asarray(ref_ds[:], dtype=object))
        _ = _to_unicode_array(np.asarray(alt_ds[:], dtype=object))
        _ = _to_unicode_array(np.asarray(vid_ds[:], dtype=object))
        times.append(time.perf_counter() - t0)
    print(f"  pre-load once:                         {_median_seconds(times):.3f}s")


def _time_high_fraction_sample_mask(
    *,
    chrom_group,
    n_samples: int,
    n_variants: int,
    reps: int,
    sample_fraction: float,
    window_variants: int,
    seed: int,
) -> None:
    if sample_fraction <= 0.0 or sample_fraction > 1.0:
        raise ValueError("--sample-fraction must be in (0, 1]")

    rng = np.random.default_rng(int(seed))
    selected = max(1, min(n_samples, int(round(n_samples * float(sample_fraction)))))
    mask = np.zeros(n_samples, dtype=bool)
    mask[:selected] = True
    rng.shuffle(mask)

    sample_indices = np.flatnonzero(mask)
    hap_indices = np.empty(sample_indices.size * 2, dtype=np.int64)
    hap_indices[0::2] = 2 * sample_indices
    hap_indices[1::2] = (2 * sample_indices) + 1

    hap_mask = np.empty(2 * n_samples, dtype=bool)
    hap_mask[0::2] = mask
    hap_mask[1::2] = mask

    arr = chrom_group["haplotypes"]
    n_window = max(1, min(int(window_variants), int(n_variants)))
    variant_slice = slice(0, n_window)

    print(f"\n=== Haplotype read with {sample_fraction:.0%} sample mask ===")
    times_oindex: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = np.asarray(arr.oindex[hap_indices, variant_slice], dtype=np.uint8)
        times_oindex.append(time.perf_counter() - t0)
    print(
        f"  oindex {n_window:,} vars:              "
        f"{_median_seconds(times_oindex):.3f}s  shape={result.shape}"
    )

    times_full_mask: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        full = np.asarray(arr[:, variant_slice], dtype=np.uint8)
        result = full[hap_mask]
        times_full_mask.append(time.perf_counter() - t0)
    full_mask_sec = _median_seconds(times_full_mask)
    print(
        f"  full-read + numpy mask {n_window:,} vars: "
        f"{full_mask_sec:.3f}s  shape={result.shape}"
    )
    if full_mask_sec > 0:
        print(f"  ratio oindex/full_mask:              {_median_seconds(times_oindex) / full_mask_sec:.1f}x")


def _time_chunk_alignment(
    *,
    store: HapStore,
    chrom: str | int,
    n_variants: int,
    reps: int,
    block_sizes: Sequence[int],
    max_variants: int,
) -> None:
    n_scanned = max(1, min(int(max_variants), int(n_variants)))

    print("\n=== Chunk alignment: haplotype read timing ===")
    for block_size in block_sizes:
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for start in range(0, n_scanned, int(block_size)):
                end = min(n_scanned, start + int(block_size))
                _ = store.haplotypes(chrom, variant_slice=(start, end))
            times.append(time.perf_counter() - t0)
        n_blocks = (n_scanned + int(block_size) - 1) // int(block_size)
        print(
            f"  block={int(block_size):5d} ({n_blocks:4d} blocks, first {n_scanned:,} vars): "
            f"{_median_seconds(times):.3f}s"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="Input HapStore zarr path")
    parser.add_argument("--chrom", default="21", help="Chromosome to benchmark")
    parser.add_argument("--reps", type=int, default=3, help="Timing repetitions per case")
    parser.add_argument(
        "--metadata-block-sizes",
        type=_parse_int_csv,
        default=[1024, 5000, 8192],
        help="Comma-separated metadata block sizes",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.67,
        help="Fraction of samples selected for high-fraction mask timing",
    )
    parser.add_argument(
        "--sample-window-variants",
        type=int,
        default=5000,
        help="Number of leading variants for sample-mask timing",
    )
    parser.add_argument(
        "--chunk-block-sizes",
        type=_parse_int_csv,
        default=[500, 1000, 1024, 5000, 8192, 10000],
        help="Comma-separated haplotype read block sizes",
    )
    parser.add_argument(
        "--chunk-max-variants",
        type=int,
        default=20000,
        help="Number of leading variants for block-size timing",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sample-mask RNG seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if int(args.reps) <= 0:
        parser.error("--reps must be positive")
    if int(args.sample_window_variants) <= 0:
        parser.error("--sample-window-variants must be positive")
    if int(args.chunk_max_variants) <= 0:
        parser.error("--chunk-max-variants must be positive")

    store = HapStore(args.store, mode="r")
    chrom = args.chrom
    chrom_key = _chrom_key(chrom)
    n_samples = store.n_samples(chrom)
    n_variants = store.n_variants(chrom)
    attrs = store.attrs(chrom)
    print(f"Store: {args.store}")
    print(f"Chromosome: {chrom_key}")
    print(f"Samples: {n_samples:,}")
    print(f"Variants: {n_variants:,}")
    print(
        "Chunks: "
        f"variant={attrs.get('chunk_variant_size')}, "
        f"haplotype={attrs.get('chunk_haplotype_size')}, "
        f"sample={attrs.get('chunk_sample_size')}"
    )

    chrom_group = store.root[chrom_key]
    _time_variant_metadata_reads(
        variants=chrom_group["variants"],
        n_variants=n_variants,
        reps=int(args.reps),
        block_sizes=args.metadata_block_sizes,
    )
    _time_high_fraction_sample_mask(
        chrom_group=chrom_group,
        n_samples=n_samples,
        n_variants=n_variants,
        reps=int(args.reps),
        sample_fraction=float(args.sample_fraction),
        window_variants=int(args.sample_window_variants),
        seed=int(args.seed),
    )
    _time_chunk_alignment(
        store=store,
        chrom=chrom,
        n_variants=n_variants,
        reps=int(args.reps),
        block_sizes=args.chunk_block_sizes,
        max_variants=int(args.chunk_max_variants),
    )

    print("\nDone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
