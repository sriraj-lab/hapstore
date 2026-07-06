# hapstore

`hapstore` is a chromosome-partitioned, zarr-backed phased haplotype store with direct VCF and PGEN ingest/export.

It is the canonical genotype storage layer used by `forgephast`, but it is designed to stand on its own as a public package.

## Install

Core package:

```bash
pip install -e .
```

With PGEN support:

```bash
pip install -e ".[pgen]"
```

For development:

```bash
pip install -e ".[dev,pgen]"
```

## Public API

Core store:

- `hapstore.HapStore`
- `hapstore.HapStoreDosageRequest`
- `hapstore.HapStoreDosageBatch`

Ingest:

- `hapstore.ingest_vcf(...)`
- `hapstore.ingest_arrays(...)`
- `hapstore.ingest_pgen(...)`

Concatenation:

- `hapstore.concat_hapstores(..., axis="samples")`
- `hapstore.concat_hapstores(..., axis="variants")`

Export:

- `hapstore.export_vcf(...)`
- `hapstore.export_pgen(...)`
- `hapstore.subset_samples_to_new_store(...)`

Helpers:

- `hapstore.normalize_chrom(...)`
- `hapstore.read_sample_ids(...)`
- `hapstore.align_by_ids(...)`
- `hapstore.compute_window_boundaries(...)`
- `hapstore.compute_window_midpoints(...)`

## Data model

Each store is a zarr hierarchy keyed by chromosome, for example `chr1`, `chr2`, `chrX`.

Per chromosome, the main groups are:

- `haplotypes`
- `genotypes` when requested
- `variants/pos`, `variants/ref`, `variants/alt`, `variants/vid`
- `variants/genetic_pos` when provided
- `samples/ids`
- `samples/population` when provided

The runtime contract is diploid, phased haplotypes with missing values represented by an internal sentinel.

## Common workflows

### Ingest a chromosome from VCF

```python
from hapstore import ingest_vcf

store = ingest_vcf(
    vcf_path="panel.chr21.vcf.gz",
    output_zarr="panel.chr21.hapstore.zarr",
    chrom="21",
    backend="cyvcf2",
    workers=4,
)
```

### Concatenate stores across samples

```python
from hapstore import concat_hapstores

merged = concat_hapstores(
    input_stores=["pop_a.chr21.zarr", "pop_b.chr21.zarr"],
    output_store="merged.chr21.zarr",
    chrom="21",
    axis="samples",
)
```

### Export phased VCF

```python
from hapstore import export_vcf

export_vcf(
    store_path="merged.chr21.zarr",
    output_vcf="merged.chr21.vcf.gz",
    chrom="21",
)
```

### Export PGEN

```python
from hapstore import export_pgen

export_pgen(
    store_path="merged.chr21.zarr",
    out_prefix="merged.chr21",
    chrom="21",
)
```

## Behavioral notes

- `concat_hapstores` is a concat operation, not a key-based merge.
- Sample-axis concat requires compatible variant axes across inputs.
- Variant-axis concat requires identical sample IDs and order across inputs.
- Parallel staged VCF ingest requires indexed input (`.tbi` or `.csi`).
- `export_vcf(..., plink2_location=...)` uses the `HapStore -> PGEN -> VCF` bridge.

## Reader performance notes

Repeated downstream window extraction should avoid many small random variant reads. For large local-window workloads, use the core dosage reader instead of reimplementing chunk planning in downstream scripts:

```python
from hapstore import HapStore, HapStoreDosageRequest

store = HapStore("panel.chr1.hapstore.zarr", mode="r")

dosages = store.read_dosages(
    "1",
    variant_ids=["rs1", "rs2", "rs3"],
    dtype="float32",
    workers="auto",
)

batch = store.read_dosage_batch(
    "1",
    [
        HapStoreDosageRequest("w1", variant_ids=["rs1", "rs2"]),
        HapStoreDosageRequest("w2", variant_ids=["rs2", "rs3"]),
    ],
    workers="auto",
)
w1_matrix = batch.window_matrix("w1")
w1_score = batch.project("w1", weights=[0.2, -0.1])
```

`read_dosages` groups requested variants and sample rows by native zarr chunks, coalesces adjacent dense chunks into larger production-oriented tiles, opens one read handle per worker, reads 2D row-by-variant tiles concurrently, and returns columns in request order. If a stored `genotypes` array exists, it reads that array directly; otherwise it sums paired haplotypes tile-by-tile. Stores with `has_missing=false` skip missing-sentinel scans by default. `workers="auto"` uses a chunk-grid-aware worker policy capped at 16 workers by default; set `HAPSTORE_READ_AUTO_MAX_WORKERS` to tune the cap or `HAPSTORE_READ_WORKERS` to force a worker count for benchmarking. Contiguous-batch tile coalescing defaults to `4` variant chunks by `2` sample chunks; tune with `HAPSTORE_READ_VARIANT_GROUP_CHUNKS`, `HAPSTORE_READ_SAMPLE_GROUP_CHUNKS`, and `HAPSTORE_READ_COALESCE_MIN_DENSITY`. Use `workers=1` or `backend="serial"` to opt out of threaded tile reads.

For chunk-aligned scans, `iter_blocks(chrom, data="genotypes")` yields `HapStoreBlock` objects with `values`, `variant_indices`, `positions`, `ref`, `alt`, `vid`, `start`, and `stop`. Default blocks follow the on-disk variant chunk size. Explicit `variant_indices`, `sample_indices`, or `sample_mask` requests preserve caller order within each yielded block.

`variant_maf(...)` computes alternate-allele frequency, MAF, alternate-allele counts, and called genotype/allele counts with missing genotypes excluded from denominators. It supports the same sample and variant subset arguments as `iter_blocks`; `non_monomorphic_variant_indices(...)` returns the ordered subset with MAF above a threshold.

`export_pgen(...)` can prefilter variants before writing with `filter_monomorphic=True` or `min_maf=<threshold>`, using HapStore MAF scans for the selected sample set. Multi-worker PGEN export accepts `shard_tmp_dir` to place temporary shard triplets on scratch before PLINK2 merge.

## Relationship to forgephast

`forgephast` depends on the public `hapstore` API for donor-panel caching, synthetic output storage, QC input loading, and phenotype genotype access. Internal refactors in this repo are intended to keep that public surface stable.
