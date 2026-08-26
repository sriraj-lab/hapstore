# HapStore

HapStore is a chromosome-partitioned, Zarr-backed store for phased diploid
haplotypes. It provides direct VCF and PGEN ingest/export, bounded block reads,
and numerical genotype or dosage matrices without requiring an intermediate VCF
or PLINK conversion for every downstream analysis.

Each chromosome stores a phased hard-call matrix with shape `(2N, V)`, where
`N` is the number of diploid samples and `V` is the number of variants. Variant
metadata, optional genetic positions, sample IDs, and optional population labels
are stored alongside the matrix.

## Start Here

- [Quickstart](quickstart.md) creates a store and reads a dosage matrix.
- [Ingest a phased VCF](guides/ingest-vcf.md) covers serial and staged VCF
  ingest.
- [Extract a numerical cM window](guides/numerical-window.md) shows direct
  window-to-NumPy extraction for statistical or machine-learning workflows.
- [API Reference](api/store.md) documents the supported Python surface.

## Public API Boundary

Import public objects from `hapstore`:

```python
from hapstore import HapStore, ingest_vcf, export_pgen
```

Modules beneath `hapstore._internal` are implementation details. They may change
without preserving an import contract.

## Relationship to FORGEPHAST

FORGEPHAST uses HapStore as its canonical genotype storage layer for donor
panels, synthetic haplotypes, QC, and phenotype access. HapStore is maintained
as a standalone package and can also be used directly in unrelated workflows.
