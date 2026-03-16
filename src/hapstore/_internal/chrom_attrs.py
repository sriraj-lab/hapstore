"""Shared chromosome-level attribute payload builders."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


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
) -> dict[str, Any]:
    return {
        "n_samples": int(n_samples),
        "n_haplotypes": int(n_haplotypes),
        "n_variants": int(n_variants),
        "phased": True,
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
