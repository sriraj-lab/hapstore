from __future__ import annotations

import pytest

from hapstore._internal.chunking import (
    DEFAULT_MATRIX_CHUNK_TARGET_BYTES,
    DEFAULT_VARIANT_CHUNK_SIZE,
    resolve_matrix_chunk_layout,
)


def test_default_chunk_layout_matches_manuscript_shape():
    layout = resolve_matrix_chunk_layout(
        n_haplotypes=300_000,
        n_samples=150_000,
        n_variants=102_498,
    )

    assert DEFAULT_VARIANT_CHUNK_SIZE == 500
    assert DEFAULT_MATRIX_CHUNK_TARGET_BYTES == 7_500_000
    assert layout.variant == 500
    assert layout.haplotype == 15_000
    assert layout.sample == 15_000


def test_chunk_layout_scales_rows_with_variant_chunk_and_honors_overrides():
    automatic = resolve_matrix_chunk_layout(
        n_haplotypes=300_000,
        n_samples=150_000,
        n_variants=102_498,
        chunk_variant_size=1_000,
    )
    explicit = resolve_matrix_chunk_layout(
        n_haplotypes=300_000,
        n_samples=150_000,
        n_variants=102_498,
        chunk_variant_size=1_000,
        chunk_haplotype_size=20_000,
        chunk_sample_size=10_000,
    )

    assert (
        automatic.variant,
        automatic.haplotype,
        automatic.sample,
    ) == (1_000, 7_500, 7_500)
    assert (
        explicit.variant,
        explicit.haplotype,
        explicit.sample,
    ) == (1_000, 20_000, 10_000)


def test_chunk_layout_clamps_to_small_arrays_and_rejects_invalid_values():
    layout = resolve_matrix_chunk_layout(
        n_haplotypes=8,
        n_samples=4,
        n_variants=3,
    )
    assert (layout.variant, layout.haplotype, layout.sample) == (3, 8, 4)

    with pytest.raises(ValueError, match="chunk_variant_size"):
        resolve_matrix_chunk_layout(
            n_haplotypes=8,
            n_samples=4,
            n_variants=3,
            chunk_variant_size=0,
        )
