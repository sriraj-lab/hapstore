"""Minimal VCF parsing helpers for tests (text backend only).

Avoids dependency on rajlab_utils.popgen.vcf.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def get_sample_ids(vcf_path: str | Path, backend: str = "text") -> list[str]:
    """Return sample IDs from a VCF file using text parsing."""
    path = Path(vcf_path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                return fields[9:]
    raise ValueError(f"#CHROM header not found in VCF: {path}")


def extract_haplotypes(
    vcf_path: str | Path,
    chrom: str | int,
    start_bp: int | None = None,
    end_bp: int | None = None,
    backend: str = "text",
    *,
    require_phased: bool = False,
    missing_policy: str = "skip_variant",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Extract haplotype matrix from a VCF using text parsing.

    Returns (haplotypes, variant_ids, positions) where haplotypes has shape
    (2 * n_samples, n_variants).
    """
    path = Path(vcf_path)
    chrom_str = str(chrom)
    chrom_candidates = {chrom_str, f"chr{chrom_str}", chrom_str.removeprefix("chr")}

    opener = gzip.open if str(path).endswith(".gz") else open
    hap_rows: list[list[int]] = []
    var_ids: list[str] = []
    positions: list[int] = []

    with opener(path, "rt", encoding="utf-8") as handle:
        n_samples = 0
        for line in handle:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                n_samples = len(fields) - 9
                continue
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            rec_chrom = fields[0]
            if rec_chrom not in chrom_candidates:
                continue

            pos = int(fields[1])
            if start_bp is not None and pos < start_bp:
                continue
            if end_bp is not None and pos > end_bp:
                continue

            vid = fields[2] if fields[2] != "." else f"{rec_chrom}:{pos}"
            gt_fields = fields[9:]

            hap_alleles: list[int] = []
            has_missing = False
            for gt_str in gt_fields:
                gt = gt_str.split(":")[0]
                sep = "|" if "|" in gt else "/"
                alleles = gt.split(sep)
                for a in alleles:
                    if a == ".":
                        has_missing = True
                        hap_alleles.append(0)
                    else:
                        hap_alleles.append(int(a))

            if has_missing and missing_policy == "skip_variant":
                continue

            hap_rows.append(hap_alleles)
            var_ids.append(vid)
            positions.append(pos)

    if not hap_rows:
        return np.empty((2 * n_samples, 0), dtype=np.uint8), [], np.array([], dtype=np.int64)

    hap_matrix = np.array(hap_rows, dtype=np.uint8).T
    return hap_matrix, var_ids, np.array(positions, dtype=np.int64)
