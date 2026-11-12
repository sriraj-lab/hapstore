"""Shared chromosome-level attribute payload builders."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def retained_unphased_genotypes(extra: dict[str, Any] | None) -> bool:
    """Whether ingest kept unphased heterozygotes with arbitrary allele order."""
    if not isinstance(extra, dict) or extra.get("on_unphased_gt") != "arbitrary":
        return False
    return int(extra.get("unphased_gt_count") or 0) > 0


def build_chrom_attrs(
    *,
    n_samples: int,
    n_haplotypes: int,
    n_variants: int,
    missing_sentinel: int,
    has_missing: bool,
    compression_level: int,
    chunk_variant_size: int,
    chunk_haplotype_size: int | None,
    chunk_sample_size: int | None,
    layout_version: int,
    source: Any = None,
    build: Any = None,
    extra: dict[str, Any] | None = None,
    phased: bool = True,
) -> dict[str, Any]:
    """Build the standardized attribute payload for one HapStore chromosome.

    ``phased`` is forced to ``False`` when ``extra`` records that unphased
    heterozygotes were kept with arbitrary allele order
    (``on_unphased_gt="arbitrary"`` and a positive ``unphased_gt_count``).
    Evidence can only clear the flag, never set it.
    """
    return {
        "n_samples": int(n_samples),
        "n_haplotypes": int(n_haplotypes),
        "n_variants": int(n_variants),
        "phased": bool(phased) and not retained_unphased_genotypes(extra),
        "ploidy": 2,
        "missing_sentinel": int(missing_sentinel),
        "has_missing": bool(has_missing),
        "compression": "zstd",
        "compression_level": int(compression_level),
        "chunk_variant_size": int(chunk_variant_size),
        "chunk_haplotype_size": None
        if chunk_haplotype_size is None
        else int(chunk_haplotype_size),
        "chunk_sample_size": None if chunk_sample_size is None else int(chunk_sample_size),
        "layout_version": int(layout_version),
        "source": source,
        "build": build,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra or None,
    }
