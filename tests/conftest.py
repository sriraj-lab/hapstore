from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def write_text_vcf(path: Path, rows: list[tuple]) -> Path:
    """Write a minimal diploid phased VCF.

    Row tuple fields:
        (chrom, pos, vid, ref, alt, [gt_per_sample...])
    """
    if not rows:
        raise ValueError("rows must not be empty")

    n_samples = len(rows[0][5])
    samples = [f"s{i+1}" for i in range(n_samples)]

    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={rows[0][0]}>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples),
    ]

    for chrom, pos, vid, ref, alt, gt_fields in rows:
        if len(gt_fields) != n_samples:
            raise ValueError("all rows must have same sample count")
        lines.append(
            "\t".join(
                [
                    str(chrom),
                    str(pos),
                    str(vid),
                    str(ref),
                    str(alt),
                    ".",
                    "PASS",
                    ".",
                    "GT",
                    *[str(x) for x in gt_fields],
                ]
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_genetic_map(path: Path, chrom: str, positions: list[int], cms: list[float]) -> Path:
    if len(positions) != len(cms):
        raise ValueError("positions/cms length mismatch")
    with path.open("w", encoding="utf-8") as handle:
        for bp, cm in zip(positions, cms):
            handle.write(f"{chrom}\t.\t{cm:.6f}\t{bp}\n")
    return path


@pytest.fixture
def synthetic_arrays() -> dict:
    haplotypes = np.asarray(
        [
            [0, 1, 0, 255],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 255],
        ],
        dtype=np.uint8,
    )
    variant_pos = np.asarray([100, 200, 300, 400], dtype=np.int64)
    sample_ids = np.asarray(["s1", "s2"], dtype=object)
    ref = np.asarray(["A", "C", "G", "T"], dtype=object)
    alt = np.asarray(["G", "T", "A", "C"], dtype=object)
    vid = np.asarray(["v1", "v2", "v3", "v4"], dtype=object)
    genetic = np.asarray([0.10, 0.20, 0.35, 0.60], dtype=np.float64)
    return {
        "haplotypes": haplotypes,
        "variant_pos": variant_pos,
        "sample_ids": sample_ids,
        "ref": ref,
        "alt": alt,
        "vid": vid,
        "genetic_positions": genetic,
    }
