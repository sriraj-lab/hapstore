# Quickstart

Install HapStore from a source checkout. The PGEN extra is required only for
PGEN ingest/export.

```bash
pip install -e .
pip install -e ".[pgen]"
```

## Create a Small Store

`ingest_arrays` is useful when a workflow already has phased haplotypes in a
NumPy array. Haplotype rows are paired by sample: rows `0` and `1` belong to the
first sample, rows `2` and `3` to the second, and so on.

```python
import numpy as np

from hapstore import HapStore, ingest_arrays

haplotypes = np.array(
    [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]],
    dtype=np.uint8,
)

ingest_arrays(
    output_zarr="example.hapstore.zarr",
    chrom="1",
    haplotypes=haplotypes,
    variant_pos=np.array([101, 205, 310]),
    ref=np.array(["A", "C", "G"]),
    alt=np.array(["G", "T", "A"]),
    vid=np.array(["rs1", "rs2", "rs3"]),
    sample_ids=np.array(["sample_1", "sample_2"]),
    store_genotypes=True,
)
```

## Read a Dosage Matrix

`read_dosages` returns a `(n_samples, n_variants)` alternate-allele dosage
matrix. Variant IDs and indices are mutually exclusive; request order is
preserved.

```python
store = HapStore("example.hapstore.zarr", mode="r")
dosages = store.read_dosages(
    "1",
    variant_ids=["rs3", "rs1"],
    dtype="float32",
)

assert dosages.shape == (2, 2)
```

For repeated overlapping windows, use `read_dosage_batch` to share a single
union read and then recover each window in its requested order.
`read_haplotypes` and `read_haplotype_batch` provide the same threaded and
union reads for phased haplotypes, for example to fit local haplotype PCA bases.
