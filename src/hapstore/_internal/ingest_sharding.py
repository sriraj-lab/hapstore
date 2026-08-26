"""Resolve contigs and genomic shard ranges for parallel VCF ingestion.

This module resolves chromosome aliases against indexed VCF records, determines
the usable contig span from header metadata, TBI/CSI indexes, or record
scanning, and splits that span into deterministic zero-based, half-open shard
intervals.

When no reliable span can be determined, a single shard with unspecified bounds
is returned so the backend can read the complete contig.
"""

from __future__ import annotations

from pathlib import Path

from hapstore.chrom import candidate_contigs, normalize_chrom

from .ingest_index import _resolve_contig_span_end_from_tabix_index


def _select_contig_for_parallel(vcf: "pysam.VariantFile", chrom: str | int) -> str:
    import pysam  # type: ignore

    fallback = None
    for contig in candidate_contigs(chrom):
        if contig in vcf.header.contigs and fallback is None:
            fallback = contig
        try:
            it = vcf.fetch(contig)
        except (ValueError, OSError):
            continue

        try:
            next(it)
            return contig
        except StopIteration:
            if fallback is None and contig in vcf.header.contigs:
                fallback = contig
            continue

    if fallback is not None:
        return fallback

    raise ValueError(
        f"chromosome {normalize_chrom(chrom, prefix=True)!r} not found in VCF header/records"
    )


def _build_shard_ranges(
    vcf: "pysam.VariantFile",
    contig: str,
    shard_bp: int,
) -> list[tuple[int, int | None, int | None]]:
    span = _resolve_contig_span_bounds_for_parallel(vcf, contig)
    if span is None:
        return [(0, None, None)]
    start0, end0 = span
    return _ranges_from_span(start0=start0, end0=end0, shard_bp=shard_bp)


def _resolve_contig_span_bounds_for_parallel(
    vcf: "pysam.VariantFile",
    contig: str,
) -> tuple[int, int] | None:
    contig_meta = vcf.header.contigs.get(contig)
    contig_length = None if contig_meta is None else contig_meta.length
    if contig_length is not None and int(contig_length) > 0:
        return (0, int(contig_length))

    vcf_filename = getattr(vcf, "filename", None)
    if vcf_filename is not None:
        vcf_path = Path(vcf_filename.decode("utf-8")) if isinstance(vcf_filename, bytes) else Path(vcf_filename)
        span_end = _resolve_contig_span_end_from_tabix_index(vcf_path, contig)
        if span_end is not None and int(span_end) > 0:
            return (0, int(span_end))

    return _infer_contig_bounds_from_records(vcf, contig)


def _ranges_from_span(
    start0: int,
    end0: int,
    shard_bp: int,
) -> list[tuple[int, int | None, int | None]]:
    if end0 <= start0:
        return [(0, int(start0), int(start0 + 1))]

    ranges: list[tuple[int, int | None, int | None]] = []
    start = int(max(0, start0))
    shard_index = 0
    while start < end0:
        end = min(int(end0), start + int(shard_bp))
        ranges.append((shard_index, int(start), int(end)))
        start = end
        shard_index += 1
    return ranges


def _infer_contig_bounds_from_records(
    vcf: "pysam.VariantFile",
    contig: str,
) -> tuple[int, int] | None:
    try:
        iterator = vcf.fetch(contig)
    except Exception:
        return None

    min_start: int | None = None
    max_end: int | None = None
    for record in iterator:
        rec_start = int(record.start)
        rec_end = int(record.stop)
        if min_start is None or rec_start < min_start:
            min_start = rec_start
        if max_end is None or rec_end > max_end:
            max_end = rec_end

    if min_start is None or max_end is None:
        return None
    if max_end <= min_start:
        max_end = min_start + 1
    return int(min_start), int(max_end)
