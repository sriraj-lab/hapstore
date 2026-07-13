"""Shared HapStore matrix chunk policy."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_VARIANT_CHUNK_SIZE = 500
DEFAULT_MATRIX_CHUNK_TARGET_BYTES = 7_500_000


@dataclass(frozen=True)
class MatrixChunkLayout:
    variant: int
    haplotype: int
    sample: int


def _resolve_row_chunk(
    *,
    n_rows: int,
    variant_chunk: int,
    requested: int | None,
    row_alignment: int,
) -> int:
    if int(n_rows) <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if requested is not None:
        if int(requested) <= 0:
            raise ValueError(f"row chunk must be positive, got {requested}")
        return int(min(int(requested), int(n_rows)))
    target_rows = max(
        int(row_alignment),
        int(DEFAULT_MATRIX_CHUNK_TARGET_BYTES // int(variant_chunk)),
    )
    target_rows -= target_rows % int(row_alignment)
    return int(min(max(int(row_alignment), target_rows), int(n_rows)))


def resolve_matrix_chunk_layout(
    *,
    n_haplotypes: int,
    n_samples: int,
    n_variants: int,
    chunk_variant_size: int = DEFAULT_VARIANT_CHUNK_SIZE,
    chunk_haplotype_size: int | None = None,
    chunk_sample_size: int | None = None,
) -> MatrixChunkLayout:
    """Resolve explicit or byte-targeted uint8 matrix chunks."""
    if int(n_variants) <= 0:
        raise ValueError(f"n_variants must be positive, got {n_variants}")
    if int(chunk_variant_size) <= 0:
        raise ValueError(
            f"chunk_variant_size must be positive, got {chunk_variant_size}"
        )
    variant = int(min(int(chunk_variant_size), int(n_variants)))
    haplotype = _resolve_row_chunk(
        n_rows=int(n_haplotypes),
        variant_chunk=variant,
        requested=chunk_haplotype_size,
        row_alignment=2,
    )
    sample = _resolve_row_chunk(
        n_rows=int(n_samples),
        variant_chunk=variant,
        requested=chunk_sample_size,
        row_alignment=1,
    )
    return MatrixChunkLayout(variant=variant, haplotype=haplotype, sample=sample)
