# Core Store and Concatenation

::: hapstore.HapStore
    options:
      members: []
      show_root_heading: true

## Create and Manage Chromosomes

::: hapstore.HapStore.create_chromosome

::: hapstore.HapStore.delete_chromosome

## Inspect Store Contents

::: hapstore.HapStore.chromosomes

::: hapstore.HapStore.has_chromosome

::: hapstore.HapStore.n_samples

::: hapstore.HapStore.n_haplotypes

::: hapstore.HapStore.n_variants

::: hapstore.HapStore.sample_ids

::: hapstore.HapStore.variant_positions

::: hapstore.HapStore.genetic_positions

::: hapstore.HapStore.variant_info

::: hapstore.HapStore.variant_index_map

::: hapstore.HapStore.attrs

## Read Haplotype and Dosage Matrices

::: hapstore.HapStore.haplotypes

::: hapstore.HapStore.genotypes

::: hapstore.HapStore.iter_blocks

::: hapstore.HapStore.read_dosages

::: hapstore.HapStore.read_dosage_batch

## Summarize Variant Frequencies

::: hapstore.HapStore.variant_maf

::: hapstore.HapStore.non_monomorphic_variant_indices

## Define Genetic Windows

::: hapstore.HapStore.compute_windows

::: hapstore.HapStore.iter_windows

## Concatenate Stores

::: hapstore.concat_hapstores

## Supporting Data Objects

::: hapstore.HapStoreDosageRequest

::: hapstore.HapStoreDosageBatch
    options:
      members: true

::: hapstore.HapStoreBlock
