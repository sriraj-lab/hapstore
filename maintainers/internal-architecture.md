# HapStore Internal Architecture

This note is for maintainers. Public package users should import from `hapstore`
and should not rely on modules under `hapstore._internal`.

## Public API Boundary

The public API is exposed from `src/hapstore/__init__.py` and includes:

- `HapStore`
- `HapStoreDosageRequest`
- `HapStoreDosageBatch`
- `HapStoreBlock`
- `ingest_vcf`
- `ingest_arrays`
- `ingest_pgen`
- `concat_hapstores`
- `export_vcf`
- `export_pgen`
- sample/chromosome/window helper functions

Implementation modules such as `pgen.py`, `store.py`, `ingest.py`, `concat.py`,
`export_vcf.py`, and `subset.py` may be imported by tests and internal callers,
but the stable user-facing import path should remain `from hapstore import ...`.

`export_vcf.py` owns native VCF writing and the optional
`HapStore -> PGEN -> PLINK2 -> VCF` bridge. It delegates PGEN materialization,
including parallel sharding and merging, to `pgen.export_pgen(...)` rather than
maintaining a second sharding path. A single PLINK2 invocation then creates the
VCF, with a separate thread count.

`subset.py` owns the distinct sample-axis copy operation. The older
`export.py` path is a compatibility shim that re-exports both functions; new
code should import from the top-level package or the specific implementation
module.

## PGEN Module Split

`src/hapstore/pgen.py` owns HapStore-to-PGEN and PGEN-to-HapStore behavior:

- public `export_pgen(...)` and `ingest_pgen(...)` functions,
- sample and variant subset resolution,
- MAF/non-monomorphic prefilter integration,
- PGEN metadata validation,
- serial and sharded export/ingest orchestration,
- PLINK2 merge calls for sharded PGEN export,
- conversion between HapStore missing sentinels and PGEN missing hardcalls.

`src/hapstore/_internal/pgenlib_io.py` is only the optional `pgenlib` adapter:

- import and availability check for `pgenlib`,
- thin reader/writer wrappers,
- pgenlib path encoding,
- pgenlib buffer shape, dtype, and contiguity checks,
- PVAR metadata accessors.

This split keeps the C-extension-specific details out of `pgen.py`. If pgenlib
changes its Python API, most repairs should stay inside `_internal/pgenlib_io.py`.

## Optional Dependency Contract

Core HapStore functionality should import without `pgenlib`. Direct PGEN ingest
or export should fail with a clear runtime error if the optional dependency is
missing. Keep optional imports inside PGEN-specific modules and tests.

## Internal Module Guidelines

- Keep `_internal` modules narrow and dependency-oriented.
- Prefer stable public functions in top-level modules over importing internals
  from downstream projects.
- If an internal helper starts carrying user-facing semantics, promote the
  behavior through a documented public API rather than encouraging private
  imports.
- Keep benchmark or HPC-specific provenance out of README examples. Put
  environment-specific results in the Obsidian project notes.

## Current Cleanup Direction

Before wiring Forgephast to the new HapStore PGEN prefilter API, keep this repo
easy to review:

- make the PGEN/pgenlib boundary explicit,
- add comments only where data layout, missing-value semantics, phase handling,
  chunk planning, or concurrency would otherwise be easy to misread,
- keep public API docs focused on behavior and examples.
