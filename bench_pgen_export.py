#!/usr/bin/env python3
"""Benchmark PGEN export components on real HapStore data."""

import time
import numpy as np
import sys
sys.path.insert(0, "/gs/gsfs0/users/raj-lab/hersh/aim2/hapstore/src")

from hapstore import HapStore
from hapstore._internal.zarr_utils import _to_unicode_array

STORE = "/gs/gsfs0/users/raj-lab/final_ref_panel/synthetic/4way_homogenous/autosomes_4pop150k/merged/chr21/populations/EUR.chr21.merged.hapstore.zarr"
CHROM = 21
N_REPS = 3

store = HapStore(STORE, mode="r")
n_samples = store.n_samples(CHROM)
n_variants = store.n_variants(CHROM)
print(f"Store: {n_samples} samples, {n_variants} variants")
print(f"Attrs: {store.attrs(CHROM).get('chunk_variant_size')}, {store.attrs(CHROM).get('chunk_haplotype_size')}")

chrom_group = store.root["chr21"]
variants = chrom_group["variants"]
pos_ds = variants["pos"]
ref_ds = variants["ref"]
alt_ds = variants["alt"]
vid_ds = variants["vid"]

# === Test 1: Per-block metadata reads (OLD) vs pre-load (NEW) ===
print("\n=== Variant metadata reads ===")
for bs in [1024, 5000, 8192]:
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        for start in range(0, n_variants, bs):
            end = min(n_variants, start + bs)
            p = np.asarray(pos_ds[start:end], dtype=np.int64)
            r = _to_unicode_array(np.asarray(ref_ds[start:end], dtype=object))
            a = _to_unicode_array(np.asarray(alt_ds[start:end], dtype=object))
            v = _to_unicode_array(np.asarray(vid_ds[start:end], dtype=object))
        times.append(time.perf_counter() - t0)
    n_blocks = (n_variants + bs - 1) // bs
    print(f"  OLD per-block bs={bs:5d} ({n_blocks:3d} blocks): {np.median(times):.3f}s")

times = []
for _ in range(N_REPS):
    t0 = time.perf_counter()
    all_pos = np.asarray(pos_ds[:], dtype=np.int64)
    all_ref = _to_unicode_array(np.asarray(ref_ds[:], dtype=object))
    all_alt = _to_unicode_array(np.asarray(alt_ds[:], dtype=object))
    all_vid = _to_unicode_array(np.asarray(vid_ds[:], dtype=object))
    times.append(time.perf_counter() - t0)
print(f"  NEW pre-load once:              {np.median(times):.3f}s")

# === Test 2: Haplotype read with sample_mask (OLD oindex vs NEW contiguous) ===
print("\n=== Haplotype read with 67% sample mask (100K of 150K) ===")
rng = np.random.default_rng(42)
mask = np.zeros(n_samples, dtype=bool)
mask[:int(n_samples * 0.67)] = True
rng.shuffle(mask)

sample_indices = np.flatnonzero(mask)
hap_indices = np.empty(sample_indices.size * 2, dtype=np.int64)
hap_indices[0::2] = 2 * sample_indices
hap_indices[1::2] = (2 * sample_indices) + 1

hap_mask = np.empty(2 * n_samples, dtype=bool)
hap_mask[0::2] = mask
hap_mask[1::2] = mask

arr = chrom_group["haplotypes"]
vslice = slice(0, 5000)

# OLD: oindex
times_old = []
for _ in range(N_REPS):
    t0 = time.perf_counter()
    result = np.asarray(arr.oindex[hap_indices, vslice], dtype=np.uint8)
    times_old.append(time.perf_counter() - t0)
print(f"  OLD oindex 5K vars: {np.median(times_old):.3f}s  shape={result.shape}")

# NEW: contiguous + mask
times_new = []
for _ in range(N_REPS):
    t0 = time.perf_counter()
    full = np.asarray(arr[:, vslice], dtype=np.uint8)
    result = full[hap_mask]
    times_new.append(time.perf_counter() - t0)
print(f"  NEW full+mask 5K vars: {np.median(times_new):.3f}s  shape={result.shape}")
print(f"  Speedup: {np.median(times_old) / np.median(times_new):.1f}x")

# === Test 3: Chunk alignment impact ===
print("\n=== Chunk alignment: haplotype read timing ===")
for bs in [500, 1000, 1024, 5000, 8192, 10000]:
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        for start in range(0, min(20000, n_variants), bs):
            end = min(n_variants, start + bs)
            h = store.haplotypes(CHROM, variant_slice=(start, end))
        times.append(time.perf_counter() - t0)
    n_blocks = min(20000, n_variants) // bs
    print(f"  block={bs:5d} ({n_blocks:3d} blocks, first 20K vars): {np.median(times):.3f}s")

print("\nDone")
