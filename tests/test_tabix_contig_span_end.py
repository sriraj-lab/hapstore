from __future__ import annotations

from pathlib import Path

import pytest


def _write_indexed_pysam_vcf_missing_contig_length(
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
) -> Path:
    import pysam

    temp_vcf = path.with_suffix("")
    header = pysam.VariantHeader()
    header.add_meta("fileformat", "VCFv4.2")
    header.contigs.add("chr1")
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


def test_build_shard_ranges_uses_tabix_when_contig_length_missing(tmp_path, monkeypatch):
    from hapstore import ingest as ingest_mod

    vcf_path = _write_indexed_pysam_vcf_missing_contig_length(
        tmp_path / "nolength.vcf.gz",
        records=[
            (100, "rs1", "A", "G", (0, 1, True), (1, 1, True)),
            (200_000, "rs2", "C", "T", (0, 0, True), (0, 1, True)),
            (3_000_000, "rs3", "G", "A", (1, 0, True), (0, 0, True)),
        ],
    )
    assert Path(f"{vcf_path}.tbi").exists()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("should not scan contig bounds when tabix index is available")

    from hapstore._internal import ingest_sharding as sharding_mod

    monkeypatch.setattr(sharding_mod, "_infer_contig_bounds_from_records", _boom)

    import pysam

    with pysam.VariantFile(str(vcf_path), "r") as vcf:
        ranges = ingest_mod._build_shard_ranges(vcf, "chr1", shard_bp=1_000_000)

    assert ranges
    last_end = ranges[-1][2]
    assert last_end is not None
    assert int(last_end) >= 3_000_000
    assert int(last_end) <= 3_000_000 + (1 << 14)


def test_csi_span_parser_uses_real_index_header_and_bins(tmp_path):
    from hapstore._internal.ingest_index import _resolve_contig_span_end_from_tabix_index

    vcf_path = _write_indexed_pysam_vcf_missing_contig_length(
        tmp_path / "csi.vcf.gz",
        records=[
            (100, "rs1", "A", "G", (0, 1, True), (1, 1, True)),
            (3_000_000, "rs2", "G", "A", (1, 0, True), (0, 0, True)),
        ],
    )

    import pysam

    Path(f"{vcf_path}.tbi").unlink()
    pysam.tabix_index(str(vcf_path), preset="vcf", force=True, csi=True)

    csi_path = Path(f"{vcf_path}.csi")
    assert csi_path.exists()
    span_end = _resolve_contig_span_end_from_tabix_index(vcf_path, "chr1")
    assert span_end is not None
    assert int(span_end) >= 3_000_000
    assert int(span_end) <= 3_000_000 + (1 << 14)
