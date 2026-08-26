"""Ingest one phased VCF or BCF chromosome into a HapStore."""

from __future__ import annotations

import argparse
from pathlib import Path

from hapstore import ingest_vcf


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one chromosome ingest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vcf_path", type=Path)
    parser.add_argument("output_zarr", type=Path)
    parser.add_argument("chrom")
    parser.add_argument("--genetic-map", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--store-genotypes", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run VCF ingest and report the resulting chromosome dimensions."""
    args = parse_args()
    store = ingest_vcf(
        vcf_path=args.vcf_path,
        output_zarr=args.output_zarr,
        chrom=args.chrom,
        genetic_map_path=args.genetic_map,
        backend=args.backend,
        workers=args.workers,
        scratch_dir=args.scratch_dir,
        store_genotypes=args.store_genotypes,
    )
    print(
        f"wrote {args.output_zarr}: "
        f"{store.n_samples(args.chrom)} samples, "
        f"{store.n_variants(args.chrom)} variants"
    )


if __name__ == "__main__":
    main()
