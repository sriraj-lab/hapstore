# Extract a Numerical cM Window

HapStore stores optional genetic positions in centimorgans. Once present, a
genetic window can be converted directly to an alternate-allele dosage matrix
for numerical analysis. This avoids exporting a temporary VCF or PGEN solely to
obtain a NumPy array.

## Command-line Example

```bash
python examples/extract_dosage_window.py \
  panel.chr21.hapstore.zarr \
  21 \
  output/window_000.npy \
  --window-size-cm 0.5 \
  --window-index 0
```

The script writes a float32 matrix of shape `(n_samples, n_variants_in_window)`.
It also reports the selected half-open variant interval `[start, stop)`.

## Python Equivalent

```python
import numpy as np

from hapstore import HapStore

store = HapStore("panel.chr21.hapstore.zarr", mode="r")
boundaries = store.compute_windows("21", window_size_cm=0.5)
start, stop = boundaries[0]

dosages = store.read_dosages(
    "21",
    variant_indices=np.arange(start, stop),
    dtype="float32",
)
```

`compute_windows` requires genetic positions. During VCF ingest, provide a
genetic map through `genetic_map_path`; array ingest accepts
`genetic_positions` directly.
