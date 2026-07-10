from __future__ import annotations

import numpy as np
import pytest
import shutil
from pathlib import Path

from hapstore import HapStore, export_pgen, ingest_arrays, ingest_pgen


try:  # pragma: no cover - availability gate
    import pgenlib  # noqa: F401

    HAS_PGENLIB = True
except Exception:  # pragma: no cover
    HAS_PGENLIB = False


pytestmark = pytest.mark.skipif(not HAS_PGENLIB, reason="pgenlib is required for PGEN tests")


def _build_store(path):
    ingest_arrays(
        output_zarr=path,
        chrom="1",
        haplotypes=np.asarray(
            [
                [0, 1, 0, 1],  # s1 hap0
                [1, 0, 1, 0],  # s1 hap1
                [1, 1, 0, 0],  # s2 hap0
                [0, 0, 1, 1],  # s2 hap1
                [0, 0, 1, 1],  # s3 hap0
                [1, 1, 0, 0],  # s3 hap1
            ],
            dtype=np.uint8,
        ),
        variant_pos=np.asarray([100, 200, 300, 400], dtype=np.int64),
        sample_ids=["s1", "s2", "s3"],
        ref=["A", "C", "G", "T"],
        alt=["G", "T", "A", "C"],
        vid=["v1", "v2", "v3", "v4"],
        chunk_variant_size=2,
        chunk_haplotype_size=2,
    )


def test_export_pgen_roundtrip(tmp_path):
    store_path = tmp_path / "source.zarr"
    _build_store(store_path)

    prefix = export_pgen(
        store_path=store_path,
        out_prefix=tmp_path / "out" / "chr1",
        chrom="1",
        variant_block_size=2,
    )

    ingested = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "roundtrip.zarr",
        chrom="1",
        variant_block_size=2,
    )

    src = HapStore(store_path, mode="r")
    dst = HapStore(tmp_path / "roundtrip.zarr", mode="r")

    np.testing.assert_array_equal(src.haplotypes("1"), dst.haplotypes("1"))
    assert src.sample_ids("1") == dst.sample_ids("1")
    assert src.variant_positions("1").tolist() == dst.variant_positions("1").tolist()

    extra = ingested.attrs("1").get("extra")
    assert isinstance(extra, dict)
    assert int(extra.get("unphased_gt_count", -1)) == 0


def test_export_pgen_sample_subset_and_order(tmp_path):
    store_path = tmp_path / "source.zarr"
    _build_store(store_path)

    prefix = export_pgen(
        store_path=store_path,
        out_prefix=tmp_path / "subset" / "chr1",
        chrom="1",
        sample_ids=["s3", "s1"],
        variant_block_size=2,
    )

    out = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "subset_ingest.zarr",
        chrom="1",
        variant_block_size=2,
    )

    assert out.sample_ids("1") == ["s3", "s1"]

    got = out.haplotypes("1")
    expected = np.asarray(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(got, expected)


def test_export_pgen_streams_blocks(monkeypatch, tmp_path):
    store_path = tmp_path / "source.zarr"
    _build_store(store_path)

    calls = []
    orig = HapStore.haplotypes

    def wrapped(self, chrom, variant_slice=None, sample_mask=None):
        calls.append(variant_slice)
        assert variant_slice is not None
        if isinstance(variant_slice, tuple):
            start, end = variant_slice
            assert int(end) - int(start) <= 1
        return orig(self, chrom, variant_slice=variant_slice, sample_mask=sample_mask)

    monkeypatch.setattr(HapStore, "haplotypes", wrapped)

    export_pgen(
        store_path=store_path,
        out_prefix=tmp_path / "stream" / "chr1",
        chrom="1",
        variant_block_size=1,
    )

    assert len(calls) >= 4


def test_export_pgen_rejects_multiallelic_alt_metadata(tmp_path):
    store_path = tmp_path / "multi_alt.zarr"
    ingest_arrays(
        output_zarr=store_path,
        chrom="1",
        haplotypes=np.asarray([[0], [1]], dtype=np.uint8),
        variant_pos=np.asarray([100], dtype=np.int64),
        sample_ids=["s1"],
        ref=["A"],
        alt=["G,C"],
        vid=["v1"],
    )

    with pytest.raises(ValueError, match="non-biallelic ALT"):
        export_pgen(
            store_path=store_path,
            out_prefix=tmp_path / "bad" / "chr1",
            chrom="1",
            strict_biallelic=True,
        )
    prefix = tmp_path / "bad" / "chr1"
    assert not Path(f"{prefix}.pgen").exists()
    assert not Path(f"{prefix}.pvar").exists()
    assert not Path(f"{prefix}.psam").exists()


def test_export_pgen_partial_missing_policy(tmp_path):
    store_path = tmp_path / "partial_missing.zarr"
    ingest_arrays(
        output_zarr=store_path,
        chrom="1",
        haplotypes=np.asarray(
            [
                [0, 1],
                [255, 0],
            ],
            dtype=np.uint8,
        ),
        variant_pos=np.asarray([100, 200], dtype=np.int64),
        sample_ids=["s1"],
        ref=["A", "C"],
        alt=["G", "T"],
        vid=["v1", "v2"],
    )

    with pytest.raises(ValueError, match="partial missing"):
        export_pgen(
            store_path=store_path,
            out_prefix=tmp_path / "partial_error" / "chr1",
            chrom="1",
            on_partial_missing="error",
        )

    prefix = export_pgen(
        store_path=store_path,
        out_prefix=tmp_path / "partial_ok" / "chr1",
        chrom="1",
        on_partial_missing="missing",
    )
    out = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "partial_roundtrip.zarr",
        chrom="1",
    )
    h = out.haplotypes("1")
    assert h[0, 0] == 255
    assert h[1, 0] == 255


def test_export_pgen_filter_monomorphic_prefilter(tmp_path):
    store_path = tmp_path / "prefilter_source.zarr"
    ingest_arrays(
        output_zarr=store_path,
        chrom="1",
        haplotypes=np.asarray(
            [
                [0, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        ),
        variant_pos=np.asarray([100, 200, 300], dtype=np.int64),
        sample_ids=["s1", "s2"],
        ref=["A", "C", "G"],
        alt=["G", "T", "A"],
        vid=["v1", "v2", "v3"],
        chunk_variant_size=1,
        chunk_haplotype_size=2,
    )

    prefix = export_pgen(
        store_path=store_path,
        out_prefix=tmp_path / "prefilter" / "chr1",
        chrom="1",
        filter_monomorphic=True,
        variant_block_size=1,
    )
    out = ingest_pgen(
        pgen_prefix=prefix,
        output_zarr=tmp_path / "prefilter_roundtrip.zarr",
        chrom="1",
    )

    assert out.variant_positions("1").tolist() == [200]
    np.testing.assert_array_equal(
        out.haplotypes("1"),
        HapStore(store_path, mode="r").haplotypes("1", variant_slice=(1, 2)),
    )


def test_export_pgen_parallel_sharded_roundtrip(tmp_path):
    if shutil.which("plink2") is None:
        pytest.skip("plink2 binary is required for sharded export_pgen merge test")

    store_path = tmp_path / "source.zarr"
    _build_store(store_path)

    serial_prefix = export_pgen(
        store_path=store_path,
        out_prefix=tmp_path / "serial" / "chr1",
        chrom="1",
        variant_block_size=2,
    )
    sharded_prefix = export_pgen(
        store_path=store_path,
        out_prefix=tmp_path / "sharded" / "chr1",
        chrom="1",
        variant_block_size=2,
        n_workers=2,
        plink2_workers=1,
        plink2_path="system",
    )

    serial_store = ingest_pgen(
        pgen_prefix=serial_prefix,
        output_zarr=tmp_path / "serial_roundtrip.zarr",
        chrom="1",
        variant_block_size=2,
    )
    sharded_store = ingest_pgen(
        pgen_prefix=sharded_prefix,
        output_zarr=tmp_path / "sharded_roundtrip.zarr",
        chrom="1",
        variant_block_size=2,
    )

    np.testing.assert_array_equal(serial_store.haplotypes("1"), sharded_store.haplotypes("1"))
    assert serial_store.variant_positions("1").tolist() == sharded_store.variant_positions("1").tolist()
    assert serial_store.sample_ids("1") == sharded_store.sample_ids("1")


def test_export_pgen_parallel_shards_use_configured_tmp_dir(tmp_path):
    if shutil.which("plink2") is None:
        pytest.skip("plink2 binary is required for sharded export_pgen merge test")

    store_path = tmp_path / "source.zarr"
    _build_store(store_path)
    scratch = tmp_path / "scratch"

    export_pgen(
        store_path=store_path,
        out_prefix=tmp_path / "sharded_tmp" / "chr1",
        chrom="1",
        variant_block_size=2,
        n_workers=2,
        plink2_workers=1,
        plink2_path="system",
        shard_tmp_dir=scratch,
        keep_intermediate_shards=True,
    )

    assert scratch.exists()
    assert any(path.name.startswith("chr1.shards_") for path in scratch.iterdir())
