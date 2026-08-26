# Ingest a Phased VCF

HapStore ingests one chromosome from a phased, biallelic VCF or BCF. By default,
unphased heterozygotes and multiallelic records fail explicitly so the output
does not silently lose phase or allele information.

## Requirements

- The input must contain the selected chromosome and phased genotype calls.
- Multi-worker staged ingest requires a compressed, indexed VCF/BCF (`.tbi` or
  `.csi`).
- An optional genetic map supplies per-variant cM positions for genetic-window
  operations.

## Command-line Example

The checked-in example exposes the common ingest controls:

```bash
python examples/ingest_vcf.py \
  panel.chr21.vcf.gz \
  panel.chr21.hapstore.zarr \
  21 \
  --workers 4 \
  --genetic-map chr21.map \
  --store-genotypes
```

`workers=1` performs a single-reader ingest. Larger values use staged ingestion
with deterministic, ordered finalization for supported reader backends. Set
`--scratch-dir` to place staged artifacts on local SSD or node-local scratch.

## Python Equivalent

```python
from hapstore import ingest_vcf

store = ingest_vcf(
    vcf_path="panel.chr21.vcf.gz",
    output_zarr="panel.chr21.hapstore.zarr",
    chrom="21",
    genetic_map_path="chr21.map",
    workers=4,
    store_genotypes=True,
)
```

The returned `HapStore` points to the completed output hierarchy. See
[core store API](../api/store.md) for direct reads after ingest.
