from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hapstore import ingest_vcf

from .conftest import write_genetic_map


def _write_indexed_pysam_vcf(
    path: Path,
    records: list[
        tuple[
            int,
            str,
            str,
            str,
            tuple[int | None, int | None, bool],
            tuple[int | None, int | None, bool],
        ]
    ],
    contig_length: int = 1_000_000,
) -> Path:
    import pysam

    temp_vcf = path.with_suffix("")
    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.contigs.add("chr1", length=int(contig_length))
    header.formats.add("GT", number=1, type="String", description="Genotype")
    header.add_sample("s1")
    header.add_sample("s2")

    with pysam.VariantFile(str(temp_vcf), "w", header=header) as out:
        for pos, vid, ref, alt, s1_gt, s2_gt in records:
            rec = out.new_record(
                contig="chr1",
                start=int(pos) - 1,
                stop=int(pos),
                id=str(vid),
                alleles=(str(ref), str(alt)),
            )
            rec.samples["s1"]["GT"] = (s1_gt[0], s1_gt[1])
            rec.samples["s1"].phased = bool(s1_gt[2])
            rec.samples["s2"]["GT"] = (s2_gt[0], s2_gt[1])
            rec.samples["s2"].phased = bool(s2_gt[2])
            out.write(rec)

    pysam.tabix_compress(str(temp_vcf), str(path), force=True)
    pysam.tabix_index(str(path), preset="vcf", force=True)
    temp_vcf.unlink(missing_ok=True)
    return path


def test_staged_partition_manifest_is_deterministic(tmp_path):
    vcf = _write_indexed_pysam_vcf(
        tmp_path / "det.vcf.gz",
        records=[
            (100, "rs1", "A", "G", (0, 1, True), (1, 1, True)),
            (200_000, "rs2", "C", "T", (0, 0, True), (0, 1, True)),
            (600_000, "rs3", "G", "A", (1, 0, True), (0, 0, True)),
            (900_000, "rs4", "T", "C", (1, 1, True), (1, 0, True)),
        ],
        contig_length=1_000_000,
    )

    scratch1 = tmp_path / "scratch_run1"
    scratch2 = tmp_path / "scratch_run2"

    ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "det_run1.zarr",
        chrom="1",
        backend="pysam",
        workers=2,
        partition_bp=300_000,
        scratch_dir=scratch1,
        cleanup_intermediates=False,
    )
    ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "det_run2.zarr",
        chrom="1",
        backend="pysam",
        workers=2,
        partition_bp=300_000,
        scratch_dir=scratch2,
        cleanup_intermediates=False,
    )

    manifest1 = next(scratch1.rglob("partition_manifest.json"))
    manifest2 = next(scratch2.rglob("partition_manifest.json"))
    payload1 = json.loads(manifest1.read_text(encoding="utf-8"))
    payload2 = json.loads(manifest2.read_text(encoding="utf-8"))

    assert payload1["contig"] == "chr1"
    assert int(payload1["partition_bp"]) == 300_000
    assert payload1["shards"] == payload2["shards"]
    assert payload1["n_shards"] == payload2["n_shards"]


def test_staged_matches_single_worker_with_genetic_map_and_genotypes(tmp_path):
    vcf = _write_indexed_pysam_vcf(
        tmp_path / "eq.vcf.gz",
        records=[
            (100, "rs1", "A", "G", (0, 1, True), (1, 1, True)),
            (200_000, "rs2", "C", "T", (0, 0, True), (0, 1, True)),
            (600_000, "rs3", "G", "A", (1, 0, True), (0, 0, True)),
            (900_000, "rs4", "T", "C", (1, 1, True), (1, 0, True)),
        ],
        contig_length=1_000_000,
    )
    gmap = write_genetic_map(
        tmp_path / "eq.map",
        chrom="1",
        positions=[100, 200_000, 600_000, 900_000],
        cms=[0.10, 0.30, 0.80, 1.20],
    )

    baseline = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "eq_baseline.zarr",
        chrom="1",
        backend="pysam",
        workers=1,
        store_genotypes=True,
        genetic_map_path=gmap,
    )
    staged = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "eq_staged.zarr",
        chrom="1",
        backend="pysam",
        workers=2,
        partition_bp=300_000,
        store_genotypes=True,
        genetic_map_path=gmap,
    )

    np.testing.assert_array_equal(staged.variant_positions("1"), baseline.variant_positions("1"))
    np.testing.assert_array_equal(staged.haplotypes("1"), baseline.haplotypes("1"))
    np.testing.assert_array_equal(staged.genotypes("1"), baseline.genotypes("1"))
    np.testing.assert_allclose(staged.genetic_positions("1"), baseline.genetic_positions("1"))

    staged_info = staged.variant_info("1")
    baseline_info = baseline.variant_info("1")
    assert staged_info["vid"] == baseline_info["vid"]
    assert staged_info["ref"] == baseline_info["ref"]
    assert staged_info["alt"] == baseline_info["alt"]


def test_staged_cyvcf2_matches_single_worker(tmp_path):
    vcf = _write_indexed_pysam_vcf(
        tmp_path / "cy.vcf.gz",
        records=[
            (100, "rs1", "A", "G", (0, 1, True), (1, 1, True)),
            (200_000, "rs2", "C", "T", (0, 0, True), (0, 1, True)),
            (600_000, "rs3", "G", "A", (1, 0, True), (0, 0, True)),
            (900_000, "rs4", "T", "C", (1, 1, True), (1, 0, True)),
        ],
        contig_length=1_000_000,
    )

    baseline = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "cy_baseline.zarr",
        chrom="1",
        backend="cyvcf2",
        workers=1,
        store_genotypes=True,
    )
    staged = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "cy_staged.zarr",
        chrom="1",
        backend="cyvcf2",
        workers=2,
        partition_bp=300_000,
        store_genotypes=True,
    )

    np.testing.assert_array_equal(staged.variant_positions("1"), baseline.variant_positions("1"))
    np.testing.assert_array_equal(staged.haplotypes("1"), baseline.haplotypes("1"))
    np.testing.assert_array_equal(staged.genotypes("1"), baseline.genotypes("1"))

    staged_info = staged.variant_info("1")
    baseline_info = baseline.variant_info("1")
    assert staged_info["vid"] == baseline_info["vid"]
    assert staged_info["ref"] == baseline_info["ref"]
    assert staged_info["alt"] == baseline_info["alt"]


def test_staged_handles_empty_shards_and_boundary_positions(tmp_path):
    vcf = _write_indexed_pysam_vcf(
        tmp_path / "boundary.vcf.gz",
        records=[
            (100, "rs1", "A", "G", (0, 1, True), (1, 1, True)),
            (300_000, "rs2", "C", "T", (0, 0, True), (0, 1, True)),
            (300_001, "rs3", "G", "A", (1, 0, True), (0, 0, True)),
            (900_000, "rs4", "T", "C", (1, 1, True), (1, 0, True)),
        ],
        contig_length=1_600_000,
    )
    staged = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "boundary.zarr",
        chrom="1",
        backend="pysam",
        workers=3,
        partition_bp=300_000,
    )

    np.testing.assert_array_equal(
        staged.variant_positions("1"),
        np.asarray([100, 300_000, 300_001, 900_000], dtype=np.int64),
    )
    extra = staged.attrs("1")["extra"]
    assert int(extra["parallel_shards"]) == 6
    assert int(staged.n_variants("1")) == 4


def test_staged_aggregates_missing_and_unphased_counts_like_single_worker(tmp_path):
    vcf = _write_indexed_pysam_vcf(
        tmp_path / "counts.vcf.gz",
        records=[
            (100, "rs1", "A", "G", (0, 1, True), (1, 1, True)),
            (200_000, "rs2", "C", "T", (0, None, True), (0, 1, True)),
            (600_000, "rs3", "G", "A", (1, 0, False), (0, 0, True)),
        ],
        contig_length=1_000_000,
    )

    baseline = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "counts_baseline.zarr",
        chrom="1",
        backend="pysam",
        workers=1,
        on_unphased_gt="missing",
    )
    staged = ingest_vcf(
        vcf_path=vcf,
        output_zarr=tmp_path / "counts_staged.zarr",
        chrom="1",
        backend="pysam",
        workers=2,
        partition_bp=300_000,
        on_unphased_gt="missing",
    )

    baseline_extra = baseline.attrs("1")["extra"]
    staged_extra = staged.attrs("1")["extra"]
    assert int(staged_extra["missing_ref_count"]) == int(baseline_extra["missing_ref_count"])
    assert int(staged_extra["missing_alt_count"]) == int(baseline_extra["missing_alt_count"])
    assert int(staged_extra["missing_vid_count"]) == int(baseline_extra["missing_vid_count"])
    assert int(staged_extra["unphased_gt_count"]) == int(baseline_extra["unphased_gt_count"])
    np.testing.assert_array_equal(staged.haplotypes("1"), baseline.haplotypes("1"))


def test_staged_cleanup_intermediates_on_failure(tmp_path):
    vcf = _write_indexed_pysam_vcf(
        tmp_path / "fail.vcf.gz",
        records=[
            (100, "rs1", "A", "G", (0, 1, False), (1, 1, True)),
        ],
        contig_length=500_000,
    )
    scratch_base = tmp_path / "scratch_fail"
    output_store = tmp_path / "fail_out.zarr"

    with pytest.raises(ValueError, match="unphased genotype"):
        ingest_vcf(
            vcf_path=vcf,
            output_zarr=output_store,
            chrom="1",
            backend="pysam",
            workers=2,
            partition_bp=250_000,
            on_unphased_gt="error",
            scratch_dir=scratch_base,
            cleanup_intermediates=True,
        )

    assert not output_store.exists()
    assert not list(scratch_base.rglob("partition_manifest.json"))
    assert not list(scratch_base.glob("staged_*"))
