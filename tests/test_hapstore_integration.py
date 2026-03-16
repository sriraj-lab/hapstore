from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from hapstore import export_vcf, ingest_vcf
from ._vcf_helpers import extract_haplotypes

from .conftest import write_genetic_map, write_text_vcf


def test_full_pipeline_ingest_windows_export_roundtrip(tmp_path):
    vcf = write_text_vcf(
        tmp_path / "input.vcf",
        rows=[
            ("chr1", 100, "rs1", "A", "G", ["0|1", "1|1"]),
            ("chr1", 150, "rs2", "C", "T", ["0|0", "0|1"]),
            ("chr1", 220, "rs3", "G", "A", ["1|0", "0|0"]),
        ],
    )
    gmap = write_genetic_map(
        tmp_path / "map.tsv",
        chrom="1",
        positions=[100, 150, 220],
        cms=[0.10, 0.21, 0.45],
    )

    store_path = tmp_path / "pipeline.zarr"
    store = ingest_vcf(
        vcf_path=vcf,
        output_zarr=store_path,
        chrom="1",
        backend="text",
        genetic_map_path=gmap,
        store_genotypes=True,
    )

    boundaries = store.compute_windows("1", window_size_cm=0.2, store=True)
    assert boundaries.shape[1] == 2

    windows = list(store.iter_windows("1", window_size_cm=0.2))
    assert len(windows) == boundaries.shape[0]
    assert windows[0][3].shape[0] == 4

    out_vcf = export_vcf(
        store_path=store_path,
        output_vcf=tmp_path / "roundtrip.vcf",
        chrom="1",
        bgzip=False,
    )
    h, vids, pos = extract_haplotypes(out_vcf, chrom="1", backend="text")

    assert vids == ["rs1", "rs2", "rs3"]
    assert pos.tolist() == [100, 150, 220]
    np.testing.assert_array_equal(h, store.haplotypes("1").astype(np.int8))



def test_store_readable_via_raw_zarr_api(tmp_path):
    vcf = write_text_vcf(
        tmp_path / "input.vcf",
        rows=[
            ("chr1", 100, "rs1", "A", "G", ["0|1", "1|1"]),
        ],
    )

    store_path = tmp_path / "raw.zarr"
    ingest_vcf(
        vcf_path=vcf,
        output_zarr=store_path,
        chrom="1",
        backend="text",
    )

    root = zarr.open(str(store_path), mode="r")
    h = np.asarray(root["chr1"]["haplotypes"][:], dtype=np.uint8)
    pos = np.asarray(root["chr1"]["variants"]["pos"][:], dtype=np.int64)

    assert h.shape == (4, 1)
    assert pos.tolist() == [100]


@pytest.mark.skipif(
    os.getenv("RAJLAB_UTILS_RUN_EXTERNAL_FIXTURES") != "1",
    reason="External fixture test disabled; set RAJLAB_UTILS_RUN_EXTERNAL_FIXTURES=1 to run.",
)
def test_external_chr22_sample_vcf_smoke(tmp_path):
    """Optional smoke test against shared AIM2 sample VCF fixture."""
    external_vcf = Path(
        "/Users/hershgupta/work/raj-lab/aim2/ibs-hash-spectral-graph/sample_data/chr22.homogenous.vcf.gz"
    )
    if not external_vcf.exists():
        pytest.skip("External sample VCF not found")

    store_path = tmp_path / "external_sample.zarr"
    store = ingest_vcf(
        vcf_path=external_vcf,
        output_zarr=store_path,
        chrom="22",
        backend="cyvcf2",
    )

    assert store.has_chromosome("chr22")
    assert store.n_variants("22") > 0
    assert store.n_samples("22") > 0
