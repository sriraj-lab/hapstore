from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import shutil

from hapstore import HapStore, export_vcf, ingest_arrays
from hapstore.export import subset_samples_to_new_store
from ._vcf_helpers import extract_haplotypes, get_sample_ids


def _build_store_for_export(path):
    haplotypes = np.asarray(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
        ],
        dtype=np.uint8,
    )
    ingest_arrays(
        output_zarr=path,
        chrom="1",
        haplotypes=haplotypes,
        variant_pos=np.asarray([100, 200], dtype=np.int64),
        sample_ids=["s1", "s2"],
        ref=["A", "C"],
        alt=["G", "T"],
        vid=["rs1", "rs2"],
    )


def test_export_vcf_roundtrip(tmp_path):
    store_path = tmp_path / "store.zarr"
    _build_store_for_export(store_path)

    out_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "out.vcf",
        chrom="1",
        bgzip=False,
    )

    samples = get_sample_ids(out_vcf, backend="text")
    haps, vids, pos = extract_haplotypes(out_vcf, chrom="1", backend="text")

    assert samples == ["s1", "s2"]
    assert vids == ["rs1", "rs2"]
    assert pos.tolist() == [100, 200]

    expected = np.asarray(
        [
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(haps, expected)


def test_export_vcf_sample_subset(tmp_path):
    store_path = tmp_path / "subset_store.zarr"
    _build_store_for_export(store_path)

    out_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "subset.vcf",
        chrom="1",
        sample_ids=["s2"],
        bgzip=False,
    )

    samples = get_sample_ids(out_vcf, backend="text")
    assert samples == ["s2"]

    haps, vids, pos = extract_haplotypes(out_vcf, chrom="1", backend="text")
    assert vids == ["rs1", "rs2"]
    assert pos.tolist() == [100, 200]
    np.testing.assert_array_equal(haps, np.asarray([[1, 1], [0, 0]], dtype=np.int8))


def test_export_vcf_variant_subset_by_position_style_ids(tmp_path):
    store_path = tmp_path / "store.zarr"
    ingest_arrays(
        output_zarr=store_path,
        chrom="1",
        haplotypes=np.asarray(
            [
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        ),
        variant_pos=np.asarray([100, 200, 300], dtype=np.int64),
        sample_ids=["s1", "s2"],
        ref=["A", "C", "G"],
        alt=["G", "T", "A"],
        vid=["rs1", "rs2", "rs3"],
    )

    out_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "subset_pos.vcf",
        chrom="1",
        variant_ids=["chr1 (line 200)", "chr1:100"],
        bgzip=False,
        variant_block_size=1,
    )

    samples = get_sample_ids(out_vcf, backend="text")
    haps, vids, pos = extract_haplotypes(out_vcf, chrom="1", backend="text")

    assert samples == ["s1", "s2"]
    assert pos.tolist() == [100, 200]
    assert vids == ["rs1", "rs2"]


def test_export_vcf_variant_subset_by_vid(tmp_path):
    store_path = tmp_path / "store_vid.zarr"
    _build_store_for_export(store_path)

    out_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "subset_vid.vcf",
        chrom="1",
        variant_ids=["rs2"],
        bgzip=False,
        variant_block_size=1,
    )

    haps, vids, pos = extract_haplotypes(out_vcf, chrom="1", backend="text")
    assert pos.tolist() == [200]
    assert vids == ["rs2"]
    np.testing.assert_array_equal(haps, np.asarray([[1], [0], [1], [0]], dtype=np.int8))


def test_export_vcf_preserves_sample_order(tmp_path):
    store_path = tmp_path / "store_order.zarr"
    _build_store_for_export(store_path)

    out_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "order.vcf",
        chrom="1",
        sample_ids=["s2", "s1"],
        bgzip=False,
        variant_block_size=1,
    )

    samples = get_sample_ids(out_vcf, backend="text")
    assert samples == ["s2", "s1"]

    haps, vids, pos = extract_haplotypes(out_vcf, chrom="1", backend="text")
    assert vids == ["rs1", "rs2"]
    assert pos.tolist() == [100, 200]
    np.testing.assert_array_equal(
        haps,
        np.asarray(
            [
                [1, 1],  # s2 hap0
                [0, 0],  # s2 hap1
                [0, 1],  # s1 hap0
                [1, 0],  # s1 hap1
            ],
            dtype=np.int8,
        ),
    )


def test_export_vcf_streams_variant_blocks(monkeypatch, tmp_path):
    from hapstore.store import HapStore as HapStoreClass

    store_path = tmp_path / "store_stream.zarr"
    ingest_arrays(
        output_zarr=store_path,
        chrom="1",
        haplotypes=np.asarray(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.uint8,
        ),
        variant_pos=np.asarray([100, 200, 300, 400], dtype=np.int64),
        sample_ids=["s1"],
        ref=["A", "C", "G", "T"],
        alt=["G", "T", "A", "C"],
        vid=["v1", "v2", "v3", "v4"],
    )

    calls = []
    orig = HapStoreClass.haplotypes

    def wrapped(self, chrom, variant_slice=None, sample_mask=None):
        calls.append((variant_slice, sample_mask))
        assert variant_slice is not None
        if isinstance(variant_slice, tuple):
            start, end = variant_slice
            assert int(end) - int(start) <= 1
        return orig(self, chrom, variant_slice=variant_slice, sample_mask=sample_mask)

    monkeypatch.setattr(HapStoreClass, "haplotypes", wrapped)

    export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "stream.vcf",
        chrom="1",
        bgzip=False,
        variant_block_size=1,
    )

    assert len(calls) >= 4


def test_export_vcf_via_pgen_preserves_content(tmp_path):
    try:
        import pgenlib  # noqa: F401
    except Exception:
        pytest.skip("pgenlib is required for PGEN-backed VCF export test")
    if shutil.which("plink2") is None:
        pytest.skip("plink2 binary is required for PGEN-backed VCF export test")

    store_path = tmp_path / "store_via_pgen.zarr"
    _build_store_for_export(store_path)

    native_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "native.vcf",
        chrom="1",
        bgzip=False,
    )
    pgen_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "via_pgen.vcf",
        chrom="1",
        bgzip=False,
        plink2_location="system",
        n_shards=2,
        keep_intermediate_files=True,
    )

    native_samples = get_sample_ids(native_vcf, backend="text")
    native_haps, native_vids, native_pos = extract_haplotypes(native_vcf, chrom="1", backend="text")

    pgen_samples = get_sample_ids(pgen_vcf, backend="text")
    pgen_haps, pgen_vids, pgen_pos = extract_haplotypes(pgen_vcf, chrom="1", backend="text")

    assert pgen_samples == native_samples
    assert pgen_vids == native_vids
    assert pgen_pos.tolist() == native_pos.tolist()
    np.testing.assert_array_equal(pgen_haps, native_haps)

    merged_prefix = tmp_path / "via_pgen.merged_pgen"
    for ext in (".pgen", ".pvar", ".psam"):
        assert Path(f"{merged_prefix}{ext}").exists()


def test_export_vcf_via_pgen_removes_intermediates_by_default(tmp_path):
    try:
        import pgenlib  # noqa: F401
    except Exception:
        pytest.skip("pgenlib is required for PGEN-backed VCF export test")
    if shutil.which("plink2") is None:
        pytest.skip("plink2 binary is required for PGEN-backed VCF export test")

    store_path = tmp_path / "cleanup_store.zarr"
    _build_store_for_export(store_path)
    output_vcf = tmp_path / "cleanup.vcf"

    result = export_vcf(
        store_path=store_path,
        output_vcf=output_vcf,
        chrom="1",
        bgzip=False,
        plink2_location="system",
        n_shards=2,
        keep_intermediate_files=False,
    )

    assert result == output_vcf
    assert output_vcf.exists()
    assert not list(tmp_path.glob("cleanup.merged_pgen.*"))
    assert not list(tmp_path.glob("cleanup.shard*.pgen.*"))
    assert not list(tmp_path.glob("hapstore_export_vcf_*"))


def test_export_vcf_via_pgen_cleanup_is_covered_without_plink(monkeypatch, tmp_path):
    import hapstore.export as export_mod
    import hapstore.pgen as pgen_mod
    from hapstore._internal import command as command_mod

    monkeypatch.setattr(export_mod, "resolve_plink2_binary", lambda _location: Path("plink2"))

    def fake_export_pgen(*, out_prefix, **_kwargs):
        prefix = Path(out_prefix)
        for ext in (".pgen", ".pvar", ".psam"):
            Path(f"{prefix}{ext}").write_text("stub\n", encoding="utf-8")
        return prefix

    def fake_run_cmd(cmd, **_kwargs):
        if "--export" in cmd:
            out_prefix = Path(cmd[cmd.index("--out") + 1])
            Path(f"{out_prefix}.vcf").write_text("##fileformat=VCFv4.2\n", encoding="utf-8")

    monkeypatch.setattr(pgen_mod, "export_pgen", fake_export_pgen)
    monkeypatch.setattr(command_mod, "run_cmd", fake_run_cmd)

    output_vcf = tmp_path / "mock_cleanup.vcf"
    result = export_vcf(
        store_path=tmp_path / "unused.zarr",
        output_vcf=output_vcf,
        chrom="1",
        bgzip=False,
        plink2_location="system",
        n_shards=1,
        keep_intermediate_files=False,
    )

    assert result == output_vcf
    assert output_vcf.exists()
    assert not list(tmp_path.glob("hapstore_export_vcf_*"))


def test_subset_samples_to_new_store_streams_and_preserves_order(monkeypatch, tmp_path):
    from hapstore.store import HapStore as HapStoreClass

    source_store = tmp_path / "source.zarr"
    haplotypes = np.asarray(
        [
            [0, 1, 0, 1],  # s1 hap0
            [1, 0, 1, 0],  # s1 hap1
            [1, 1, 0, 0],  # s2 hap0
            [0, 0, 1, 1],  # s2 hap1
            [0, 0, 1, 1],  # s3 hap0
            [1, 1, 0, 0],  # s3 hap1
        ],
        dtype=np.uint8,
    )
    variant_pos = np.asarray([100, 200, 300, 400], dtype=np.int64)
    ref = ["A", "C", "G", "T"]
    alt = ["G", "T", "A", "C"]
    vid = ["v1", "v2", "v3", "v4"]
    ingest_arrays(
        output_zarr=source_store,
        chrom="1",
        haplotypes=haplotypes,
        variant_pos=variant_pos,
        sample_ids=["s1", "s2", "s3"],
        ref=ref,
        alt=alt,
        vid=vid,
        chunk_variant_size=2,
        chunk_haplotype_size=2,
    )

    calls = []
    orig = HapStoreClass.haplotypes

    def wrapped(self, chrom, variant_slice=None, sample_mask=None):
        calls.append((variant_slice, sample_mask))
        assert variant_slice is not None
        assert sample_mask is not None
        if isinstance(variant_slice, tuple):
            start, end = variant_slice
            assert int(end) - int(start) <= 2
        return orig(self, chrom, variant_slice=variant_slice, sample_mask=sample_mask)

    monkeypatch.setattr(HapStoreClass, "haplotypes", wrapped)

    out_store = tmp_path / "subset.zarr"
    subset_samples_to_new_store(
        source_store=source_store,
        chrom="1",
        sample_ids=["s3", "s1"],
        variant_block_size=2,
        output_store=out_store,
        variant_chunk=2,
        hap_chunk=2,
        compression_level=1,
    )

    assert len(calls) >= 2

    monkeypatch.setattr(HapStoreClass, "haplotypes", orig)

    out = HapStore(out_store, mode="r")
    assert out.sample_ids("1") == ["s3", "s1"]
    np.testing.assert_array_equal(out.variant_positions("1"), variant_pos)
    info = out.variant_info("1")
    assert info["ref"] == ref
    assert info["alt"] == alt
    assert info["vid"] == vid
    attrs = out.attrs("1")
    assert attrs["chunk_haplotype_size"] == 2
    assert attrs["layout_version"] == 2
    expected = np.asarray(
        [
            haplotypes[4, :],
            haplotypes[5, :],
            haplotypes[0, :],
            haplotypes[1, :],
        ],
        dtype=np.uint8,
    ).astype(np.int8)
    np.testing.assert_array_equal(out.haplotypes("1").astype(np.int8), expected)


def test_export_vcf_missing_sentinel_becomes_dot(tmp_path):
    store_path = tmp_path / "missing_store.zarr"
    ingest_arrays(
        output_zarr=store_path,
        chrom="1",
        haplotypes=np.asarray(
            [
                [0, 255],
                [1, 1],
            ],
            dtype=np.uint8,
        ),
        variant_pos=np.asarray([100, 200], dtype=np.int64),
        sample_ids=["s1"],
        ref=["A", "C"],
        alt=["G", "T"],
        vid=["rs1", "rs2"],
    )

    out_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "missing.vcf",
        chrom="1",
        bgzip=False,
    )

    text = out_vcf.read_text(encoding="utf-8")
    assert ".|1" not in text
    assert ".|." in text


def test_export_vcf_strict_variant_info_guardrail(tmp_path):
    store_path = tmp_path / "strict_store.zarr"
    store = HapStore(store_path, mode="w")
    store.create_chromosome(
        chrom="1",
        haplotypes=np.asarray([[0], [1]], dtype=np.uint8),
        variant_pos=np.asarray([100], dtype=np.int64),
        sample_ids=["s1"],
        ref=None,
        alt=None,
        vid=["."],
    )

    with pytest.raises(ValueError, match="missing REF/ALT"):
        export_vcf(
            store_path=store_path,
            output_vcf=tmp_path / "strict_fail.vcf",
            chrom="1",
            bgzip=False,
            strict_variant_info=True,
        )

    out_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "strict_off.vcf",
        chrom="1",
        bgzip=False,
        strict_variant_info=False,
    )
    text = out_vcf.read_text(encoding="utf-8")
    assert "\tN\t<NON_REF>\t" in text
