from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hapstore.ingest as ingest_mod
from hapstore import HapStore, ingest_arrays, ingest_vcf

from .conftest import write_genetic_map, write_text_vcf


def _write_indexed_pysam_vcf(path: Path) -> Path:
    import pysam

    temp_vcf = path.with_suffix("")
    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.contigs.add("chr1", length=1_000_000)
    header.formats.add("GT", number=1, type="String", description="Genotype")
    header.add_sample("s1")
    header.add_sample("s2")

    with pysam.VariantFile(str(temp_vcf), "w", header=header) as out:
        records = [
            (100, "rs1", "A", "G", (0, 1), (1, 1)),
            (200_000, "rs2", "C", "T", (0, 0), (0, 1)),
            (600_000, "rs3", "G", "A", (1, 0), (0, 0)),
            (900_000, "rs4", "T", "C", (1, 1), (1, 0)),
        ]
        for pos, vid, ref, alt, gt1, gt2 in records:
            rec = out.new_record(
                contig="chr1",
                start=pos - 1,
                stop=pos,
                id=vid,
                alleles=(ref, alt),
            )
            rec.samples["s1"]["GT"] = gt1
            rec.samples["s1"].phased = True
            rec.samples["s2"]["GT"] = gt2
            rec.samples["s2"].phased = True
            out.write(rec)

    pysam.tabix_compress(str(temp_vcf), str(path), force=True)
    pysam.tabix_index(str(path), preset="vcf", force=True)
    temp_vcf.unlink(missing_ok=True)
    return path


def _write_indexed_pysam_vcf_no_contig_len(path: Path) -> Path:
    import pysam

    temp_vcf = path.with_suffix("")
    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    # Deliberately omit contig length to exercise inferred-span sharding.
    header.contigs.add("chr1")
    header.formats.add("GT", number=1, type="String", description="Genotype")
    header.add_sample("s1")
    header.add_sample("s2")

    with pysam.VariantFile(str(temp_vcf), "w", header=header) as out:
        records = [
            (100, "rs1", "A", "G", (0, 1), (1, 1)),
            (200_000, "rs2", "C", "T", (0, 0), (0, 1)),
            (600_000, "rs3", "G", "A", (1, 0), (0, 0)),
            (900_000, "rs4", "T", "C", (1, 1), (1, 0)),
        ]
        for pos, vid, ref, alt, gt1, gt2 in records:
            rec = out.new_record(
                contig="chr1",
                start=pos - 1,
                stop=pos,
                id=vid,
                alleles=(ref, alt),
            )
            rec.samples["s1"]["GT"] = gt1
            rec.samples["s1"].phased = True
            rec.samples["s2"]["GT"] = gt2
            rec.samples["s2"].phased = True
            out.write(rec)

    pysam.tabix_compress(str(temp_vcf), str(path), force=True)
    pysam.tabix_index(str(path), preset="vcf", force=True)
    temp_vcf.unlink(missing_ok=True)
    return path


def test_ingest_vcf_text_backend_roundtrip(tmp_path):
    vcf = write_text_vcf(
        tmp_path / "tiny.vcf",
        rows=[
            ("chr1", 100, "rs1", "A", "G", ["0|1", "1|1"]),
            ("chr1", 120, "rs2", "C", "T", ["0|0", "0|1"]),
        ],
    )

    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "out.zarr",
        chrom="1",
        backend="text",
    )

    assert store.chromosomes == ["chr1"]
    h = store.haplotypes("chr1")
    assert h.shape == (4, 2)
    assert store.attrs("1")["has_missing"] is False


def test_ingest_vcf_with_genetic_map(tmp_path):
    vcf = write_text_vcf(
        tmp_path / "tiny.vcf",
        rows=[
            ("chr1", 100, "rs1", "A", "G", ["0|1", "1|1"]),
            ("chr1", 200, "rs2", "C", "T", ["0|0", "0|1"]),
        ],
    )
    gmap = write_genetic_map(
        tmp_path / "map.tsv",
        chrom="1",
        positions=[100, 200],
        cms=[0.10, 0.30],
    )

    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "with_map.zarr",
        chrom="1",
        backend="text",
        genetic_map_path=gmap,
    )

    gpos = store.genetic_positions("chr1")
    assert gpos is not None
    np.testing.assert_allclose(gpos, np.asarray([0.10, 0.30], dtype=float))


def test_ingest_vcf_missing_genotypes_kept_as_255(tmp_path):
    vcf = write_text_vcf(
        tmp_path / "missing.vcf",
        rows=[
            ("chr1", 100, "rs1", "A", "G", ["0|1", "1|1"]),
            ("chr1", 120, "rs2", "C", "T", ["0|.", "0|1"]),
        ],
    )

    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "missing.zarr",
        chrom="1",
        backend="text",
    )

    h = store.haplotypes("1")
    assert h.shape == (4, 2)
    assert np.any(h == 255)
    assert store.attrs("chr1")["has_missing"] is True


def test_ingest_vcf_unphased_policy(tmp_path):
    vcf = write_text_vcf(
        tmp_path / "unphased.vcf",
        rows=[
            ("chr1", 100, "rs1", "A", "G", ["0/1", "1|1"]),
        ],
    )

    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "allow_unphased.zarr",
        chrom="1",
        backend="text",
        on_unphased_gt="missing",
    )
    h = store.haplotypes("1")
    assert h[0, 0] == 255
    assert h[1, 0] == 255
    assert store.attrs("1")["extra"]["unphased_gt_count"] == 1

    with pytest.raises(ValueError, match="unphased genotype"):
        ingest_vcf(
            vcf_path=vcf,
            output_zarr=tmp_path / "error_unphased.zarr",
            chrom="1",
            backend="text",
            on_unphased_gt="error",
        )


def test_ingest_vcf_missing_variant_info_policies(tmp_path):
    vcf = write_text_vcf(
        tmp_path / "missing_info.vcf",
        rows=[
            ("chr1", 100, ".", ".", "G", ["0|1", "1|1"]),
            ("chr1", 120, ".", "A", ".", ["0|0", "0|1"]),
        ],
    )

    with pytest.raises(ValueError, match="missing REF/ALT"):
        ingest_vcf(
            vcf_path=vcf,
            output_zarr=tmp_path / "strict_fail.zarr",
            chrom="1",
            backend="text",
            on_missing_ref_alt="error",
        )

    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "missing_ok.zarr",
        chrom="1",
        backend="text",
        on_missing_ref_alt="dot",
        on_missing_vid="dot",
    )
    extra = store.attrs("1")["extra"]
    assert extra["missing_ref_count"] == 1
    assert extra["missing_alt_count"] == 1
    assert extra["missing_vid_count"] == 2


def test_ingest_arrays_roundtrip_and_missing_attr(tmp_path, synthetic_arrays):
    store = ingest_arrays(
        output_zarr=tmp_path / "arrays.zarr",
        chrom="1",
        **synthetic_arrays,
    )
    np.testing.assert_array_equal(store.haplotypes("chr1"), synthetic_arrays["haplotypes"])
    assert store.attrs("chr1")["has_missing"] is True


def test_ingest_multiple_chromosomes_same_store(tmp_path, synthetic_arrays):
    out = tmp_path / "multi.zarr"

    ingest_arrays(output_zarr=out, chrom="1", **synthetic_arrays)

    arr2 = dict(synthetic_arrays)
    arr2["variant_pos"] = np.asarray([500, 600, 700, 800], dtype=np.int64)
    ingest_arrays(output_zarr=out, chrom="2", **arr2)

    store = HapStore(out, mode="r")
    assert sorted(store.chromosomes) == ["chr1", "chr2"]


def test_ingest_vcf_pysam_parallel_multi_reader_single_writer(tmp_path):
    vcf = _write_indexed_pysam_vcf(tmp_path / "parallel.vcf.gz")

    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "parallel.zarr",
        chrom="1",
        backend="pysam",
        workers=2,
        partition_bp=300_000,
        store_genotypes=True,
    )

    assert store.has_chromosome("chr1")
    assert store.n_samples("1") == 2
    assert store.n_variants("1") == 4
    np.testing.assert_array_equal(
        store.variant_positions("1"),
        np.asarray([100, 200_000, 600_000, 900_000], dtype=np.int64),
    )

    extra = store.attrs("1")["extra"]
    assert extra["parallel_workers"] == 2
    assert extra["parallel_shards"] >= 2


def test_ingest_vcf_pysam_parallel_without_contig_length(tmp_path):
    vcf = _write_indexed_pysam_vcf_no_contig_len(tmp_path / "parallel_no_len.vcf.gz")

    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "parallel_no_len.zarr",
        chrom="1",
        backend="pysam",
        workers=2,
        partition_bp=300_000,
    )

    assert store.n_variants("1") == 4
    np.testing.assert_array_equal(
        store.variant_positions("1"),
        np.asarray([100, 200_000, 600_000, 900_000], dtype=np.int64),
    )
    extra = store.attrs("1")["extra"]
    assert extra["parallel_workers"] == 2
    assert extra["parallel_shards"] >= 2


def test_ingest_vcf_parallel_warns_when_readers_exceed_cpu_budget(tmp_path, monkeypatch, caplog):
    vcf = _write_indexed_pysam_vcf(tmp_path / "parallel_warn.vcf.gz")
    monkeypatch.setattr(ingest_mod, "_allocated_cpu_count", lambda: 2)

    with caplog.at_level("WARNING"):
        store = ingest_vcf(
            vcf_path=vcf,
            output_zarr=tmp_path / "parallel_warn.zarr",
            chrom="1",
            backend="pysam",
            workers=2,
            partition_bp=300_000,
        )

    assert store.n_variants("1") == 4
    assert "allocated CPU - 1" in caplog.text


def test_ingest_vcf_parallel_auto_partition_uses_worker_heuristic(tmp_path):
    vcf = _write_indexed_pysam_vcf(tmp_path / "parallel_auto.vcf.gz")

    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "parallel_auto.zarr",
        chrom="1",
        backend="pysam",
        workers=3,
        partition_bp=None,
    )

    extra = store.attrs("1")["extra"]
    assert int(extra["parallel_workers"]) == 3
    assert int(extra["partition_bp"]) == 400_000
    assert int(extra["parallel_shards"]) == 3


def test_ingest_vcf_parallel_auto_partition_rounds_with_missing_contig_length(tmp_path):
    vcf = _write_indexed_pysam_vcf_no_contig_len(tmp_path / "parallel_auto_no_len.vcf.gz")

    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "parallel_auto_no_len.zarr",
        chrom="1",
        backend="pysam",
        workers=3,
        partition_bp=None,
    )

    extra = store.attrs("1")["extra"]
    auto_partition = int(extra["partition_bp"])
    assert auto_partition > 0
    assert auto_partition % 100_000 == 0
    assert int(extra["parallel_shards"]) >= 2


def test_estimate_initial_variant_capacity_from_shard_span():
    specs = [
        ingest_mod._ShardSpec(shard_index=0, start_bp=0, end_bp=2_500_000),
        ingest_mod._ShardSpec(shard_index=1, start_bp=2_500_000, end_bp=5_000_000),
    ]
    cap = ingest_mod._estimate_initial_variant_capacity(specs, variant_chunk=500)
    # 5 Mb span at 1 variant / 5 kb gives ~1000 initial variants.
    assert cap == 1000


def test_estimate_initial_variant_capacity_fallback_without_bounds():
    specs = [ingest_mod._ShardSpec(shard_index=0, start_bp=None, end_bp=None)]
    cap = ingest_mod._estimate_initial_variant_capacity(specs, variant_chunk=500)
    assert cap == 500
