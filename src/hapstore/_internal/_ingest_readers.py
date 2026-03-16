"""VCF reader backends for hapstore ingest."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from hapstore.chrom import candidate_contigs, normalize_chrom

from .ingest_models import (
    Backend,
    MissingRefAltPolicy,
    MissingVidPolicy,
    UnphasedPolicy,
    _IngestArrays,
    _MISSING_SENTINEL,
)


def _coerce_ref_alt(
    chrom: str,
    pos: int,
    ref: str | None,
    alt: str | None,
    on_missing_ref_alt: MissingRefAltPolicy,
) -> tuple[str, str, int, int]:
    r = "" if ref is None else str(ref)
    a = "" if alt is None else str(alt)
    missing_ref = int((r == "") or (r == "."))
    missing_alt = int((a == "") or (a == "."))

    if missing_ref or missing_alt:
        if on_missing_ref_alt == "error":
            raise ValueError(
                f"missing REF/ALT at {chrom}:{pos}; REF={ref!r} ALT={alt!r}. "
                "Use on_missing_ref_alt='dot' to allow placeholder values."
            )
        if missing_ref:
            r = "."
        if missing_alt:
            a = "."
    return r, a, missing_ref, missing_alt


def _coerce_vid(
    chrom: str,
    pos: int,
    vid: str | None,
    on_missing_vid: MissingVidPolicy,
) -> tuple[str, int]:
    out = "" if vid is None else str(vid)
    missing = int((out == "") or (out == "."))
    if missing:
        if on_missing_vid == "error":
            raise ValueError(
                f"missing variant ID at {chrom}:{pos}; set on_missing_vid='dot' to allow '.'"
            )
        out = "."
    return out, missing


def _resolve_subset(
    all_samples: list[str],
    requested: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if requested is None:
        return np.arange(len(all_samples), dtype=np.int64), np.asarray(all_samples, dtype=object)

    sample_to_index = {sid: i for i, sid in enumerate(all_samples)}
    indices: list[int] = []
    for sid in requested:
        if sid not in sample_to_index:
            raise ValueError(f"requested sample_id {sid!r} not present in VCF header")
        indices.append(sample_to_index[sid])
    return np.asarray(indices, dtype=np.int64), np.asarray(requested, dtype=object)


def _finalize_arrays(
    sample_ids: np.ndarray,
    columns: list[np.ndarray],
    positions: list[int],
    ref: list[str],
    alt: list[str],
    vid: list[str],
    missing_ref_count: int,
    missing_alt_count: int,
    missing_vid_count: int,
    unphased_gt_count: int,
) -> _IngestArrays:
    n_samples = int(sample_ids.shape[0])
    if not columns:
        haps = np.zeros((2 * n_samples, 0), dtype=np.uint8)
    else:
        haps = np.column_stack(columns).astype(np.uint8, copy=False)

    return _IngestArrays(
        haplotypes=haps,
        positions=np.asarray(positions, dtype=np.int64),
        ref=np.asarray(ref, dtype=object),
        alt=np.asarray(alt, dtype=object),
        vid=np.asarray(vid, dtype=object),
        sample_ids=np.asarray(sample_ids, dtype=object),
        missing_ref_count=int(missing_ref_count),
        missing_alt_count=int(missing_alt_count),
        missing_vid_count=int(missing_vid_count),
        unphased_gt_count=int(unphased_gt_count),
    )


def _extract_vcf_arrays(
    vcf_path: Path,
    chrom: str | int,
    backend: Backend,
    sample_ids: list[str] | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
) -> _IngestArrays:
    if backend == "auto":
        for candidate in ("cyvcf2", "pysam", "text"):
            try:
                return _extract_vcf_arrays(
                    vcf_path=vcf_path,
                    chrom=chrom,
                    backend=candidate,
                    sample_ids=sample_ids,
                    on_missing_ref_alt=on_missing_ref_alt,
                    on_missing_vid=on_missing_vid,
                    on_unphased_gt=on_unphased_gt,
                )
            except Exception:
                continue
        raise ValueError(f"failed to parse VCF {vcf_path} with backends auto(cyvcf2/pysam/text)")

    if backend == "cyvcf2":
        return _extract_vcf_arrays_cyvcf2(
            vcf_path=vcf_path,
            chrom=chrom,
            sample_ids=sample_ids,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
        )
    if backend == "pysam":
        return _extract_vcf_arrays_pysam(
            vcf_path=vcf_path,
            chrom=chrom,
            sample_ids=sample_ids,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
        )
    if backend == "text":
        return _extract_vcf_arrays_text(
            vcf_path=vcf_path,
            chrom=chrom,
            sample_ids=sample_ids,
            on_missing_ref_alt=on_missing_ref_alt,
            on_missing_vid=on_missing_vid,
            on_unphased_gt=on_unphased_gt,
        )

    raise ValueError(f"unsupported backend: {backend}")


def _extract_vcf_arrays_cyvcf2(
    vcf_path: Path,
    chrom: str | int,
    sample_ids: list[str] | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
) -> _IngestArrays:
    from cyvcf2 import VCF  # type: ignore

    vcf = VCF(str(vcf_path))
    subset_idx, subset_ids = _resolve_subset(list(vcf.samples), sample_ids)

    columns: list[np.ndarray] = []
    positions: list[int] = []
    ref: list[str] = []
    alt: list[str] = []
    vid: list[str] = []
    missing_ref_count = 0
    missing_alt_count = 0
    missing_vid_count = 0
    unphased_gt_count = 0

    for contig in candidate_contigs(chrom):
        found_for_contig = False
        try:
            iterator = vcf(contig)
        except Exception:
            continue

        for variant in iterator:
            found_for_contig = True
            if variant.ALT is None or len(variant.ALT) != 1:
                continue

            pos = int(variant.POS)
            chrom_name = str(variant.CHROM)
            ref_i, alt_i, mref, malt = _coerce_ref_alt(
                chrom=chrom_name,
                pos=pos,
                ref=variant.REF,
                alt=variant.ALT[0],
                on_missing_ref_alt=on_missing_ref_alt,
            )
            vid_i, mvid = _coerce_vid(chrom_name, pos, variant.ID, on_missing_vid)

            gt = variant.genotypes
            hap = np.empty(2 * subset_idx.shape[0], dtype=np.uint8)
            if gt is None:
                hap.fill(_MISSING_SENTINEL)
            else:
                for out_i, sample_i in enumerate(subset_idx):
                    g = gt[int(sample_i)]
                    if g is None or len(g) < 2:
                        hap[2 * out_i] = _MISSING_SENTINEL
                        hap[2 * out_i + 1] = _MISSING_SENTINEL
                        continue

                    a0 = g[0]
                    a1 = g[1]
                    phased = bool(g[2]) if len(g) >= 3 else True

                    if not phased:
                        unphased_gt_count += 1
                        if on_unphased_gt == "error":
                            raise ValueError(
                                f"unphased genotype at {chrom_name}:{pos} sample_index={int(sample_i)}"
                            )
                        hap[2 * out_i] = _MISSING_SENTINEL
                        hap[2 * out_i + 1] = _MISSING_SENTINEL
                        continue

                    if a0 is None or a1 is None or int(a0) < 0 or int(a1) < 0:
                        hap[2 * out_i] = _MISSING_SENTINEL
                        hap[2 * out_i + 1] = _MISSING_SENTINEL
                        continue

                    hap[2 * out_i] = 1 if int(a0) > 0 else 0
                    hap[2 * out_i + 1] = 1 if int(a1) > 0 else 0

            columns.append(hap)
            positions.append(pos)
            ref.append(ref_i)
            alt.append(alt_i)
            vid.append(vid_i)
            missing_ref_count += mref
            missing_alt_count += malt
            missing_vid_count += mvid

        if found_for_contig:
            break

    return _finalize_arrays(
        sample_ids=subset_ids,
        columns=columns,
        positions=positions,
        ref=ref,
        alt=alt,
        vid=vid,
        missing_ref_count=missing_ref_count,
        missing_alt_count=missing_alt_count,
        missing_vid_count=missing_vid_count,
        unphased_gt_count=unphased_gt_count,
    )


def _extract_vcf_arrays_pysam(
    vcf_path: Path,
    chrom: str | int,
    sample_ids: list[str] | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
) -> _IngestArrays:
    import pysam  # type: ignore

    vcf = pysam.VariantFile(str(vcf_path), "r")
    all_samples = list(vcf.header.samples)
    subset_idx, subset_ids = _resolve_subset(all_samples, sample_ids)

    columns: list[np.ndarray] = []
    positions: list[int] = []
    ref: list[str] = []
    alt: list[str] = []
    vid: list[str] = []
    missing_ref_count = 0
    missing_alt_count = 0
    missing_vid_count = 0
    unphased_gt_count = 0

    for contig in candidate_contigs(chrom):
        found_for_contig = False
        try:
            iterator = vcf.fetch(contig)
        except Exception:
            continue

        for record in iterator:
            found_for_contig = True
            if record.alts is None or len(record.alts) != 1:
                continue

            pos = int(record.pos)
            chrom_name = str(record.contig)
            ref_i, alt_i, mref, malt = _coerce_ref_alt(
                chrom=chrom_name,
                pos=pos,
                ref=record.ref,
                alt=record.alts[0],
                on_missing_ref_alt=on_missing_ref_alt,
            )
            vid_i, mvid = _coerce_vid(chrom_name, pos, record.id, on_missing_vid)

            hap = np.empty(2 * subset_idx.shape[0], dtype=np.uint8)
            for out_i, sample_i in enumerate(subset_idx):
                sample_name = all_samples[int(sample_i)]
                call = record.samples[sample_name]
                gt = call.get("GT")
                phased = bool(call.phased)

                if gt is None or len(gt) < 2:
                    hap[2 * out_i] = _MISSING_SENTINEL
                    hap[2 * out_i + 1] = _MISSING_SENTINEL
                    continue

                if not phased:
                    unphased_gt_count += 1
                    if on_unphased_gt == "error":
                        raise ValueError(
                            f"unphased genotype at {chrom_name}:{pos} sample={sample_name}"
                        )
                    hap[2 * out_i] = _MISSING_SENTINEL
                    hap[2 * out_i + 1] = _MISSING_SENTINEL
                    continue

                a0, a1 = gt[0], gt[1]
                if a0 is None or a1 is None or int(a0) < 0 or int(a1) < 0:
                    hap[2 * out_i] = _MISSING_SENTINEL
                    hap[2 * out_i + 1] = _MISSING_SENTINEL
                    continue

                hap[2 * out_i] = 1 if int(a0) > 0 else 0
                hap[2 * out_i + 1] = 1 if int(a1) > 0 else 0

            columns.append(hap)
            positions.append(pos)
            ref.append(ref_i)
            alt.append(alt_i)
            vid.append(vid_i)
            missing_ref_count += mref
            missing_alt_count += malt
            missing_vid_count += mvid

        if found_for_contig:
            break

    return _finalize_arrays(
        sample_ids=subset_ids,
        columns=columns,
        positions=positions,
        ref=ref,
        alt=alt,
        vid=vid,
        missing_ref_count=missing_ref_count,
        missing_alt_count=missing_alt_count,
        missing_vid_count=missing_vid_count,
        unphased_gt_count=unphased_gt_count,
    )


def _extract_vcf_arrays_text(
    vcf_path: Path,
    chrom: str | int,
    sample_ids: list[str] | None,
    on_missing_ref_alt: MissingRefAltPolicy,
    on_missing_vid: MissingVidPolicy,
    on_unphased_gt: UnphasedPolicy,
) -> _IngestArrays:
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    wanted = set(candidate_contigs(chrom))

    header_samples: list[str] | None = None
    subset_idx: np.ndarray | None = None
    subset_ids: np.ndarray | None = None

    columns: list[np.ndarray] = []
    positions: list[int] = []
    ref: list[str] = []
    alt: list[str] = []
    vid: list[str] = []
    missing_ref_count = 0
    missing_alt_count = 0
    missing_vid_count = 0
    unphased_gt_count = 0

    with opener(vcf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                header_samples = fields[9:]
                subset_idx, subset_ids = _resolve_subset(header_samples, sample_ids)
                continue

            if line.startswith("#"):
                continue

            if header_samples is None or subset_idx is None or subset_ids is None:
                raise ValueError(f"#CHROM header not found in VCF: {vcf_path}")

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue

            contig = fields[0]
            if contig not in wanted:
                continue

            alt_field = fields[4]
            if "," in alt_field:
                continue

            pos = int(fields[1])
            ref_i, alt_i, mref, malt = _coerce_ref_alt(
                chrom=contig,
                pos=pos,
                ref=fields[3],
                alt=alt_field,
                on_missing_ref_alt=on_missing_ref_alt,
            )

            raw_vid = fields[2]
            vid_val = None if raw_vid in {"", "."} else raw_vid
            vid_i, mvid = _coerce_vid(contig, pos, vid_val, on_missing_vid)

            hap = np.empty(2 * subset_idx.shape[0], dtype=np.uint8)
            for out_i, sample_i in enumerate(subset_idx):
                gt_field = fields[9 + int(sample_i)]
                parsed = _parse_text_gt(gt_field)
                if parsed is None:
                    hap[2 * out_i] = _MISSING_SENTINEL
                    hap[2 * out_i + 1] = _MISSING_SENTINEL
                    continue

                a0, a1, phased = parsed
                if not phased:
                    unphased_gt_count += 1
                    if on_unphased_gt == "error":
                        sample_name = header_samples[int(sample_i)]
                        raise ValueError(
                            f"unphased genotype at {contig}:{pos} sample={sample_name}"
                        )
                    hap[2 * out_i] = _MISSING_SENTINEL
                    hap[2 * out_i + 1] = _MISSING_SENTINEL
                    continue

                if a0 is None or a1 is None:
                    hap[2 * out_i] = _MISSING_SENTINEL
                    hap[2 * out_i + 1] = _MISSING_SENTINEL
                    continue

                hap[2 * out_i] = 1 if int(a0) > 0 else 0
                hap[2 * out_i + 1] = 1 if int(a1) > 0 else 0

            columns.append(hap)
            positions.append(pos)
            ref.append(ref_i)
            alt.append(alt_i)
            vid.append(vid_i)
            missing_ref_count += mref
            missing_alt_count += malt
            missing_vid_count += mvid

    if subset_ids is None:
        raise ValueError(f"#CHROM header not found in VCF: {vcf_path}")

    return _finalize_arrays(
        sample_ids=subset_ids,
        columns=columns,
        positions=positions,
        ref=ref,
        alt=alt,
        vid=vid,
        missing_ref_count=missing_ref_count,
        missing_alt_count=missing_alt_count,
        missing_vid_count=missing_vid_count,
        unphased_gt_count=unphased_gt_count,
    )


def _parse_text_gt(gt_field: str) -> tuple[int | None, int | None, bool] | None:
    token = gt_field.split(":", 1)[0]
    if "|" in token:
        parts = token.split("|")
        phased = True
    elif "/" in token:
        parts = token.split("/")
        phased = False
    else:
        return None

    if len(parts) != 2:
        return None
    if parts[0] == "." or parts[1] == ".":
        return None, None, phased

    try:
        a0 = int(parts[0])
        a1 = int(parts[1])
    except ValueError:
        return None

    return a0, a1, phased
