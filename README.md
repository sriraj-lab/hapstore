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

## Relationship to forgephast

`forgephast` depends on the public `hapstore` API for donor-panel caching, synthetic output storage, QC input loading, and phenotype genotype access. Internal refactors in this repo are intended to keep that public surface stable.
