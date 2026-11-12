"""Unphased genotypes kept with arbitrary allele order (on_unphased_gt="arbitrary")."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from hapstore import HapStore, concat_hapstores, export_vcf, ingest_arrays, ingest_vcf
from hapstore.subset import subset_samples_to_new_store

from .conftest import write_text_vcf
from .test_ingest_staged import _write_indexed_pysam_vcf

try:  # pragma: no cover
    import pgenlib  # noqa: F401

    HAS_PGENLIB = True
except Exception:  # pragma: no cover
    HAS_PGENLIB = False

# Three samples, four variants: unphased hets at v1 s1, v3 s1 and v3 s3 ("1/0");
# "1/1" and "0/0" are unphased homozygotes, which are unambiguous.
ROWS = [
    ("chr1", 100, "v1", "A", "G", ["0/1", "1|1", "0|0"]),
    ("chr1", 200, "v2", "C", "T", ["0|0", "0|1", "1/1"]),
    ("chr1", 300, "v3", "G", "A", ["1/0", "0/0", "1/0"]),
    ("chr1", 400, "v4", "T", "C", ["1|0", "0|1", "1|1"]),
]
EXPECTED_DOSAGE = np.asarray([[1, 0, 1, 1], [2, 1, 0, 1], [0, 2, 1, 2]], dtype=np.float64)


def _dosages(store: HapStore) -> np.ndarray:
    return store.read_dosages("1", variant_indices=range(store.n_variants("1")), dtype="float64")


@pytest.mark.parametrize("backend", ["text", "pysam", "cyvcf2"])
def test_vcf_arbitrary_keeps_unphased_hets_with_exact_dosages(tmp_path, backend):
    if backend == "cyvcf2":
        pytest.importorskip("cyvcf2")
    if backend == "pysam":
        pytest.importorskip("pysam")
    vcf = write_text_vcf(tmp_path / "in.vcf", ROWS)
    if backend != "text":
        import pysam

        pysam.tabix_compress(str(vcf), str(vcf) + ".gz", force=True)
        pysam.tabix_index(str(vcf) + ".gz", preset="vcf", force=True)
        vcf = tmp_path / "in.vcf.gz"
    store = ingest_vcf(vcf_path=vcf, output_zarr=tmp_path / "out.zarr", chrom="1",
                       backend=backend, on_unphased_gt="arbitrary")
    np.testing.assert_array_equal(_dosages(store), EXPECTED_DOSAGE)
    assert not bool(np.any(store.haplotypes("1") == 255))
    attrs = store.attrs("1")
    assert attrs["has_missing"] is False
    assert attrs["phased"] is False and store.is_phased("1") is False
    assert attrs["extra"]["on_unphased_gt"] == "arbitrary"
    assert int(attrs["extra"]["unphased_gt_count"]) == 3  # v1 s1, v3 s1, v3 s3


def test_vcf_arbitrary_on_fully_phased_input_stays_phased(tmp_path):
    rows = [(c, p, v, r, a, [g.replace("/", "|") for g in gts]) for c, p, v, r, a, gts in ROWS]
    vcf = write_text_vcf(tmp_path / "phased.vcf", rows)
    store = ingest_vcf(vcf_path=vcf, output_zarr=tmp_path / "out.zarr", chrom="1",
                       backend="text", on_unphased_gt="arbitrary")
    assert store.is_phased("1") is True
    assert int(store.attrs("1")["extra"]["unphased_gt_count"]) == 0


def test_vcf_missing_policy_unchanged(tmp_path):
    vcf = write_text_vcf(tmp_path / "in.vcf", ROWS)
    store = ingest_vcf(vcf_path=vcf, output_zarr=tmp_path / "out.zarr", chrom="1",
                       backend="text", on_unphased_gt="missing")
    haps = store.haplotypes("1")
    assert int(haps[0, 0]) == 255 and int(haps[1, 0]) == 255
    assert store.is_phased("1") is True and store.attrs("1")["has_missing"] is True


def test_invalid_policy_rejected(tmp_path):
    vcf = write_text_vcf(tmp_path / "in.vcf", ROWS)
    with pytest.raises(ValueError, match="arbitrary"):
        ingest_vcf(vcf_path=vcf, output_zarr=tmp_path / "out.zarr", chrom="1",
                   backend="text", on_unphased_gt="random")


def test_staged_pysam_arbitrary_matches_single_worker(tmp_path):
    pytest.importorskip("pysam")
    vcf = _write_indexed_pysam_vcf(
        tmp_path / "staged.vcf.gz",
        records=[
            (100, "rs1", "A", "G", (0, 1, False), (1, 1, True)),
            (200_000, "rs2", "C", "T", (1, 0, False), (0, 1, True)),
            (600_000, "rs3", "G", "A", (1, 1, False), (0, 0, True)),
        ],
    )
    single = ingest_vcf(vcf_path=vcf, output_zarr=tmp_path / "single.zarr", chrom="1",
                        backend="pysam", workers=1, on_unphased_gt="arbitrary")
    staged = ingest_vcf(vcf_path=vcf, output_zarr=tmp_path / "staged.zarr", chrom="1",
                        backend="pysam", workers=2, partition_bp=300_000, on_unphased_gt="arbitrary")
    np.testing.assert_array_equal(_dosages(staged), _dosages(single))
    np.testing.assert_array_equal(_dosages(single), [[1, 1, 2], [2, 1, 0]])
    assert single.is_phased("1") is False and staged.is_phased("1") is False


def _unphased_store(path, sample_ids=("s1", "s2")):
    haps = np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0]], dtype=np.uint8)
    return ingest_arrays(output_zarr=path, chrom="1", haplotypes=haps[: 2 * len(sample_ids)],
                         variant_pos=np.asarray([100, 200, 300]), sample_ids=list(sample_ids),
                         ref=["A", "C", "G"], alt=["G", "T", "A"], vid=["v1", "v2", "v3"],
                         phased=False)


def test_ingest_arrays_flag_and_subset_preserve_unphased(tmp_path):
    store = _unphased_store(tmp_path / "u.zarr")
    assert store.is_phased("1") is False
    out = subset_samples_to_new_store(tmp_path / "u.zarr", "1", ["s2"], tmp_path / "sub.zarr")
    assert HapStore(out).is_phased("1") is False


def test_concat_with_any_unphased_source_is_unphased(tmp_path):
    _unphased_store(tmp_path / "u.zarr", sample_ids=("s1", "s2"))
    phased = ingest_arrays(output_zarr=tmp_path / "p.zarr", chrom="1",
                           haplotypes=np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.uint8),
                           variant_pos=np.asarray([100, 200, 300]), sample_ids=["s3"],
                           ref=["A", "C", "G"], alt=["G", "T", "A"], vid=["v1", "v2", "v3"])
    assert phased.is_phased("1") is True
    merged = concat_hapstores([tmp_path / "p.zarr", tmp_path / "u.zarr"], tmp_path / "m.zarr",
                              chrom="1", axis="samples")
    assert merged.is_phased("1") is False


def test_native_vcf_export_writes_unphased_gt(tmp_path):
    pytest.importorskip("pysam")
    _unphased_store(tmp_path / "u.zarr")
    out = export_vcf(tmp_path / "u.zarr", tmp_path / "u.vcf.gz", chrom="1")
    with gzip.open(out, "rt") as handle:
        calls = [f for line in handle if not line.startswith("#") for f in line.split()[9:]]
    assert calls and all("|" not in c for c in calls)
    assert calls[0] == "0/1"  # stored 0,1 -> written in ascending order


@pytest.mark.skipif(not HAS_PGENLIB, reason="pgenlib is required for PGEN tests")
def test_pgen_round_trip_unphased(tmp_path):
    from hapstore import export_pgen, ingest_pgen

    _unphased_store(tmp_path / "u.zarr")
    prefix = export_pgen(store_path=tmp_path / "u.zarr", out_prefix=tmp_path / "u_pgen", chrom="1")
    # The exported PGEN is genuinely unphased: the strict policy rejects it.
    with pytest.raises(ValueError, match="unphased heterozygous genotype"):
        ingest_pgen(pgen_prefix=prefix, output_zarr=tmp_path / "strict.zarr", chrom="1",
                    on_unphased_gt="error")
    for workers in (1, 2):
        back = ingest_pgen(pgen_prefix=prefix, output_zarr=tmp_path / f"back{workers}.zarr",
                           chrom="1", on_unphased_gt="arbitrary", workers=workers,
                           variant_block_size=1)
        np.testing.assert_array_equal(_dosages(back), _dosages(HapStore(tmp_path / "u.zarr")))
        assert back.is_phased("1") is False
