#!/usr/bin/env python
"""Benchmark HapStore dosage read patterns with correctness checks.

This script is intentionally standalone: run it from a source checkout with
``PYTHONPATH=src`` or from an installed HapStore environment.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hapstore import HapStore, HapStoreDosageRequest  # noqa: E402
from hapstore.chrom import normalize_chrom  # noqa: E402

READ_ENV_KEYS = (
    "HAPSTORE_READ_VARIANT_GROUP_CHUNKS",
    "HAPSTORE_READ_SAMPLE_GROUP_CHUNKS",
    "HAPSTORE_READ_COALESCE_MIN_DENSITY",
)

DEFAULT_CASES = (
    "full_contiguous",
    "high_fraction",
    "low_fraction",
    "sparse_random",
    "window_5k",
    "window_20k",
    "batch_union",
    "maf_scan",
)


@dataclass(frozen=True)
class ReadCase:
    name: str
    variant_indices: np.ndarray | None = None
    sample_indices: np.ndarray | None = None
    batch_requests: tuple[HapStoreDosageRequest, ...] = ()
    maf_block_size: int | None = None


@dataclass(frozen=True)
class RunConfig:
    workers: str
    backend: str
    variant_group_chunks: str | None
    sample_group_chunks: str | None
    coalesce_min_density: str | None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _parse_csv_args(values: Sequence[str] | None, *, default: Sequence[str]) -> list[str]:
    if not values:
        return list(default)
    out: list[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out or list(default)


def _parse_optional_csv_args(values: Sequence[str] | None) -> list[str | None]:
    if not values:
        return [None]
    out: list[str | None] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token.lower() in {"", "default", "none"}:
                out.append(None)
            else:
                out.append(token)
    return out or [None]


def _positive_int(text: str, *, name: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer, got {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive, got {value}")
    return value


def _run_git(args: Sequence[str]) -> str | None:
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _max_rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _store_metadata(store: HapStore, chrom: str | int) -> dict[str, Any]:
    chrom_key = normalize_chrom(chrom, prefix=True)
    chrom_group = store.root[chrom_key]
    dataset = "genotypes" if "genotypes" in chrom_group else "haplotypes"
    arr = chrom_group[dataset]
    attrs = store.attrs(chrom)
    return {
        "chrom_key": chrom_key,
        "n_samples": int(store.n_samples(chrom)),
        "n_variants": int(store.n_variants(chrom)),
        "dataset": dataset,
        "array_shape": tuple(int(x) for x in arr.shape),
        "array_chunks": tuple(int(x) for x in arr.chunks),
        "chunk_variant_size_attr": attrs.get("chunk_variant_size"),
        "chunk_haplotype_size_attr": attrs.get("chunk_haplotype_size"),
        "chunk_sample_size_attr": attrs.get("chunk_sample_size"),
        "has_missing": bool(attrs.get("has_missing", True)),
    }


def _contiguous_indices(n_variants: int, width: int) -> np.ndarray:
    width = min(max(1, int(width)), max(1, int(n_variants)))
    return np.arange(width, dtype=np.int64)


def _sample_indices(n_samples: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(n_samples, max(1, int(round(float(fraction) * float(n_samples)))))
    return np.asarray(sorted(rng.choice(n_samples, size=k, replace=False).tolist()), dtype=np.int64)


def _build_cases(
    *,
    store: HapStore,
    chrom: str | int,
    requested_cases: Sequence[str],
    seed: int,
    high_fraction: float,
    low_fraction: float,
    sparse_variants: int,
    batch_requests: int,
    batch_window_size: int,
    maf_block_size: int | None,
) -> list[ReadCase]:
    n_samples = int(store.n_samples(chrom))
    n_variants = int(store.n_variants(chrom))
    rng = np.random.default_rng(int(seed))
    cases: list[ReadCase] = []
    for case_name in requested_cases:
        if case_name == "all":
            cases.extend(
                _build_cases(
                    store=store,
                    chrom=chrom,
                    requested_cases=DEFAULT_CASES,
                    seed=seed,
                    high_fraction=high_fraction,
                    low_fraction=low_fraction,
                    sparse_variants=sparse_variants,
                    batch_requests=batch_requests,
                    batch_window_size=batch_window_size,
                    maf_block_size=maf_block_size,
                )
            )
            continue
        if case_name == "full_contiguous":
            cases.append(ReadCase(case_name, variant_indices=_contiguous_indices(n_variants, min(5000, n_variants))))
        elif case_name == "high_fraction":
            cases.append(
                ReadCase(
                    case_name,
                    variant_indices=_contiguous_indices(n_variants, min(5000, n_variants)),
                    sample_indices=_sample_indices(n_samples, high_fraction, rng),
                )
            )
        elif case_name == "low_fraction":
            cases.append(
                ReadCase(
                    case_name,
                    variant_indices=_contiguous_indices(n_variants, min(5000, n_variants)),
                    sample_indices=_sample_indices(n_samples, low_fraction, rng),
                )
            )
        elif case_name == "sparse_random":
            k = min(max(1, int(sparse_variants)), max(1, n_variants))
            cases.append(
                ReadCase(
                    case_name,
                    variant_indices=np.asarray(rng.choice(n_variants, size=k, replace=False), dtype=np.int64),
                )
            )
        elif case_name == "window_5k":
            cases.append(ReadCase(case_name, variant_indices=_contiguous_indices(n_variants, 5000)))
        elif case_name == "window_20k":
            cases.append(ReadCase(case_name, variant_indices=_contiguous_indices(n_variants, 20000)))
        elif case_name == "batch_union":
            width = min(max(1, int(batch_window_size)), max(1, n_variants))
            max_start = max(1, n_variants - width + 1)
            requests: list[HapStoreDosageRequest] = []
            for request_i in range(max(1, int(batch_requests))):
                start = int(rng.integers(0, max_start))
                requests.append(
                    HapStoreDosageRequest(
                        request_id=f"window_{request_i:03d}",
                        variant_indices=np.arange(start, start + width, dtype=np.int64),
                    )
                )
            cases.append(ReadCase(case_name, batch_requests=tuple(requests)))
        elif case_name == "maf_scan":
            cases.append(ReadCase(case_name, maf_block_size=maf_block_size))
        else:
            raise ValueError(f"unknown benchmark case {case_name!r}; valid cases: {', '.join(DEFAULT_CASES)}")
    seen: set[str] = set()
    unique: list[ReadCase] = []
    for case in cases:
        if case.name in seen:
            continue
        seen.add(case.name)
        unique.append(case)
    return unique


@contextmanager
def _read_env(config: RunConfig) -> Iterable[None]:
    old = {key: os.environ.get(key) for key in READ_ENV_KEYS}
    try:
        updates = {
            "HAPSTORE_READ_VARIANT_GROUP_CHUNKS": config.variant_group_chunks,
            "HAPSTORE_READ_SAMPLE_GROUP_CHUNKS": config.sample_group_chunks,
            "HAPSTORE_READ_COALESCE_MIN_DENSITY": config.coalesce_min_density,
        }
        for key, value in updates.items():
            if value is not None:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _compute_maf_scan(
    store: HapStore,
    chrom: str | int,
    *,
    block_size: int | None = None,
    sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    n_variants = int(store.n_variants(chrom))
    if n_variants <= 0:
        return np.empty(0, dtype=np.float64)
    attrs = store.attrs(chrom)
    width = int(block_size or attrs.get("chunk_variant_size", 500) or 500)
    width = min(max(1, width), n_variants)
    sample_mask = None
    if sample_indices is not None:
        sample_mask = np.zeros(int(store.n_samples(chrom)), dtype=bool)
        sample_mask[np.asarray(sample_indices, dtype=np.int64)] = True

    out = np.empty(n_variants, dtype=np.float64)
    for start in range(0, n_variants, width):
        stop = min(n_variants, start + width)
        g = np.asarray(store.genotypes(chrom, variant_slice=(start, stop), sample_mask=sample_mask), dtype=np.uint8)
        observed = g != 255
        observed_n = observed.sum(axis=0).astype(np.float64)
        denom = 2.0 * observed_n
        allele_sum = np.where(observed, g, 0).sum(axis=0).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            freq = allele_sum / denom
        maf = np.minimum(freq, 1.0 - freq)
        maf[denom <= 0] = np.nan
        out[start:stop] = maf
    return out


def _run_case(
    store: HapStore,
    chrom: str | int,
    case: ReadCase,
    config: RunConfig,
    *,
    assume_no_missing: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if case.name == "maf_scan":
        result = _compute_maf_scan(store, chrom, block_size=case.maf_block_size, sample_indices=case.sample_indices)
        metrics = {
            "backend": "maf_scan",
            "dataset": "genotypes" if "genotypes" in store.root[normalize_chrom(chrom, prefix=True)] else "haplotypes",
            "n_samples": int(store.n_samples(chrom)),
            "n_variants": int(result.shape[0]),
            "maf_block_size": int(case.maf_block_size or store.attrs(chrom).get("chunk_variant_size", 500) or 500),
        }
        return result, metrics

    if case.batch_requests:
        batch = store.read_dosage_batch(
            chrom,
            case.batch_requests,
            sample_indices=case.sample_indices,
            workers=config.workers,
            backend=config.backend,
            assume_no_missing=assume_no_missing,
        )
        return batch.matrix, dict(batch.metrics or {})

    if case.variant_indices is None:
        raise ValueError(f"case {case.name!r} has neither variant indices nor batch requests")
    result = store.read_dosages(
        chrom,
        variant_indices=case.variant_indices,
        sample_indices=case.sample_indices,
        workers=config.workers,
        backend=config.backend,
        assume_no_missing=assume_no_missing,
    )
    return result, store.last_read_metrics


def _subset_case_for_correctness(
    case: ReadCase,
    *,
    variant_limit: int,
) -> ReadCase:
    if case.name == "maf_scan":
        return case
    if case.batch_requests:
        limited_requests: list[HapStoreDosageRequest] = []
        for request in case.batch_requests:
            indices = np.asarray(request.variant_indices, dtype=np.int64)
            limited_requests.append(
                HapStoreDosageRequest(
                    request_id=request.request_id,
                    variant_indices=indices[: max(1, min(int(variant_limit), int(indices.shape[0])))],
                )
            )
        return ReadCase(
            name=case.name,
            sample_indices=case.sample_indices,
            batch_requests=tuple(limited_requests),
        )
    if case.variant_indices is None:
        return case
    return ReadCase(
        name=case.name,
        variant_indices=np.asarray(case.variant_indices, dtype=np.int64)[: max(1, min(int(variant_limit), int(case.variant_indices.shape[0])))],
        sample_indices=case.sample_indices,
    )


def _correctness_check(
    store: HapStore,
    chrom: str | int,
    case: ReadCase,
    config: RunConfig,
    *,
    assume_no_missing: bool,
    variant_limit: int,
) -> dict[str, Any]:
    limited = _subset_case_for_correctness(case, variant_limit=variant_limit)
    baseline_config = RunConfig(
        workers="1",
        backend="serial",
        variant_group_chunks=config.variant_group_chunks,
        sample_group_chunks=config.sample_group_chunks,
        coalesce_min_density=config.coalesce_min_density,
    )
    with _read_env(config):
        observed, _ = _run_case(store, chrom, limited, config, assume_no_missing=assume_no_missing)
    with _read_env(baseline_config):
        expected, _ = _run_case(store, chrom, limited, baseline_config, assume_no_missing=assume_no_missing)
    if observed.shape != expected.shape:
        return {
            "correctness_pass": False,
            "correctness_error": f"shape mismatch observed={observed.shape} expected={expected.shape}",
            "correctness_max_abs_diff": float("nan"),
        }
    diff = np.asarray(observed, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    if np.all(np.isnan(diff)):
        max_abs = 0.0
    else:
        max_abs = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    ok = bool(np.allclose(observed, expected, equal_nan=True))
    return {
        "correctness_pass": ok,
        "correctness_error": "" if ok else "values differ from serial baseline",
        "correctness_max_abs_diff": max_abs,
        "correctness_shape": "x".join(str(int(x)) for x in observed.shape),
    }


def _result_checksum(result: np.ndarray) -> float:
    arr = np.asarray(result)
    if arr.size == 0:
        return 0.0
    return float(np.nansum(arr.astype(np.float64, copy=False)))


def _flatten_row(
    *,
    store_path: Path,
    chrom: str | int,
    case: ReadCase,
    config: RunConfig,
    phase: str,
    repeat: int,
    elapsed: float,
    result: np.ndarray,
    metrics: dict[str, Any],
    correctness: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "store": str(store_path),
        "chrom": str(chrom),
        "case": case.name,
        "phase": phase,
        "repeat": int(repeat),
        "wall_seconds": f"{float(elapsed):.9f}",
        "result_shape": "x".join(str(int(x)) for x in result.shape),
        "result_checksum": f"{_result_checksum(result):.9f}",
        "requested_workers": str(config.workers),
        "backend": str(config.backend),
        "variant_group_chunks_arg": "" if config.variant_group_chunks is None else str(config.variant_group_chunks),
        "sample_group_chunks_arg": "" if config.sample_group_chunks is None else str(config.sample_group_chunks),
        "coalesce_min_density_arg": "" if config.coalesce_min_density is None else str(config.coalesce_min_density),
        "env_variant_group_chunks": os.environ.get("HAPSTORE_READ_VARIANT_GROUP_CHUNKS", ""),
        "env_sample_group_chunks": os.environ.get("HAPSTORE_READ_SAMPLE_GROUP_CHUNKS", ""),
        "env_coalesce_min_density": os.environ.get("HAPSTORE_READ_COALESCE_MIN_DENSITY", ""),
        "max_rss_mb": f"{_max_rss_mb():.3f}",
        "hostname": metadata["hostname"],
        "git_commit": metadata.get("git_commit") or "",
        "metrics_json": json.dumps(metrics, sort_keys=True, default=_json_default),
    }
    for key in (
        "dataset",
        "workers",
        "auto_worker_source",
        "sample_groups",
        "variant_groups",
        "native_sample_groups",
        "native_variant_groups",
        "n_tiles",
        "read_sec_sum",
        "convert_sec_sum",
        "assign_sec",
        "total_sec",
    ):
        if key in metrics:
            row[f"metrics_{key}"] = metrics[key]
    row.update(correctness)
    return row


def _benchmark_columns() -> list[str]:
    columns = [
        "store",
        "chrom",
        "case",
        "phase",
        "repeat",
        "wall_seconds",
        "result_shape",
        "result_checksum",
        "requested_workers",
        "backend",
        "variant_group_chunks_arg",
        "sample_group_chunks_arg",
        "coalesce_min_density_arg",
        "env_variant_group_chunks",
        "env_sample_group_chunks",
        "env_coalesce_min_density",
        "max_rss_mb",
        "hostname",
        "git_commit",
        "metrics_json",
    ]
    columns.extend(
        f"metrics_{key}"
        for key in (
            "dataset",
            "workers",
            "auto_worker_source",
            "sample_groups",
            "variant_groups",
            "native_sample_groups",
            "native_variant_groups",
            "n_tiles",
            "read_sec_sum",
            "convert_sec_sum",
            "assign_sec",
            "total_sec",
        )
    )
    columns.extend(
        [
            "correctness_pass",
            "correctness_error",
            "correctness_max_abs_diff",
            "correctness_shape",
        ]
    )
    return columns


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no benchmark rows to write")
    columns = _benchmark_columns()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_row(path: Path, row: dict[str, Any], *, write_header: bool) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_benchmark_columns(), delimiter="\t", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _metadata(args: argparse.Namespace, store_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "args": vars(args),
        "created_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_status_short": _run_git(["status", "--short"]),
        "git_diff_stat": _run_git(["diff", "--stat"]),
        "store_metadata": store_meta,
        "read_env_at_start": {key: os.environ.get(key) for key in READ_ENV_KEYS},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path, help="HapStore zarr path")
    parser.add_argument("--chrom", required=True, help="Chromosome to benchmark")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for TSV/JSON benchmark artifacts")
    parser.add_argument("--case", action="append", help="Benchmark case(s), comma-separated or repeated; use all for defaults")
    parser.add_argument("--workers", action="append", default=None, help="Worker setting(s), comma-separated or repeated")
    parser.add_argument("--backend", action="append", default=None, help="Backend setting(s), comma-separated or repeated")
    parser.add_argument("--variant-group-chunks", action="append", default=None, help="HAPSTORE_READ_VARIANT_GROUP_CHUNKS values; use default for ambient")
    parser.add_argument("--sample-group-chunks", action="append", default=None, help="HAPSTORE_READ_SAMPLE_GROUP_CHUNKS values; use default for ambient")
    parser.add_argument("--coalesce-min-density", action="append", default=None, help="HAPSTORE_READ_COALESCE_MIN_DENSITY values; use default for ambient")
    parser.add_argument("--reps", type=lambda x: _positive_int(x, name="reps"), default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--high-sample-fraction", type=float, default=0.75)
    parser.add_argument("--low-sample-fraction", type=float, default=0.10)
    parser.add_argument("--sparse-variants", type=int, default=1000)
    parser.add_argument("--batch-requests", type=int, default=8)
    parser.add_argument("--batch-window-size", type=int, default=512)
    parser.add_argument("--maf-block-size", type=int, default=None)
    parser.add_argument("--assume-no-missing", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--correctness-variant-limit", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error("--warmups must be >= 0")
    if args.correctness_variant_limit <= 0:
        parser.error("--correctness-variant-limit must be positive")
    if not args.store.exists():
        parser.error(f"--store does not exist: {args.store}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    store = HapStore(args.store, mode="r")
    store_meta = _store_metadata(store, args.chrom)
    metadata = _metadata(args, store_meta)
    requested_cases = _parse_csv_args(args.case, default=DEFAULT_CASES)
    cases = _build_cases(
        store=store,
        chrom=args.chrom,
        requested_cases=requested_cases,
        seed=int(args.seed),
        high_fraction=float(args.high_sample_fraction),
        low_fraction=float(args.low_sample_fraction),
        sparse_variants=int(args.sparse_variants),
        batch_requests=int(args.batch_requests),
        batch_window_size=int(args.batch_window_size),
        maf_block_size=args.maf_block_size,
    )
    configs = [
        RunConfig(
            workers=str(workers),
            backend=str(backend),
            variant_group_chunks=None if vchunks is None else str(vchunks),
            sample_group_chunks=None if schunks is None else str(schunks),
            coalesce_min_density=None if density is None else str(density),
        )
        for workers, backend, vchunks, schunks, density in itertools.product(
            _parse_csv_args(args.workers, default=("auto",)),
            _parse_csv_args(args.backend, default=("auto",)),
            _parse_optional_csv_args(args.variant_group_chunks),
            _parse_optional_csv_args(args.sample_group_chunks),
            _parse_optional_csv_args(args.coalesce_min_density),
        )
    ]

    rows_path = out_dir / "benchmark_rows.tsv"
    meta_path = out_dir / "benchmark_metadata.json"
    rows_path.unlink(missing_ok=True)
    metadata["cases"] = [case.name for case in cases]
    metadata["configs"] = [config.__dict__ for config in configs]
    metadata["planned_rows"] = int(len(cases) * len(configs) * (int(args.warmups) + int(args.reps)))
    metadata["completed_rows"] = 0
    metadata["status"] = "running"
    _write_metadata(meta_path, metadata)

    completed_rows = 0
    for case in cases:
        for config in configs:
            correctness = {
                "correctness_pass": "",
                "correctness_error": "",
                "correctness_max_abs_diff": "",
                "correctness_shape": "",
            }
            if not args.skip_correctness:
                correctness = _correctness_check(
                    store,
                    args.chrom,
                    case,
                    config,
                    assume_no_missing=bool(args.assume_no_missing),
                    variant_limit=int(args.correctness_variant_limit),
                )
                if not correctness.get("correctness_pass"):
                    raise RuntimeError(
                        f"correctness failed for case={case.name} config={config}: "
                        f"{correctness.get('correctness_error')}"
                    )
            total_runs = [("warmup", i) for i in range(int(args.warmups))]
            total_runs.extend(("timed", i) for i in range(int(args.reps)))
            for phase, repeat in total_runs:
                with _read_env(config):
                    start = time.perf_counter()
                    result, metrics = _run_case(
                        store,
                        args.chrom,
                        case,
                        config,
                        assume_no_missing=bool(args.assume_no_missing),
                    )
                    elapsed = time.perf_counter() - start
                    row = _flatten_row(
                        store_path=args.store,
                        chrom=args.chrom,
                        case=case,
                        config=config,
                        phase=phase,
                        repeat=repeat,
                        elapsed=elapsed,
                        result=result,
                        metrics=metrics,
                        correctness=correctness,
                        metadata=metadata,
                    )
                    _append_row(rows_path, row, write_header=(completed_rows == 0))
                    completed_rows += 1
                    metadata["completed_rows"] = completed_rows
                    metadata["last_row"] = {
                        "case": case.name,
                        "phase": phase,
                        "repeat": int(repeat),
                        "workers": str(config.workers),
                        "backend": str(config.backend),
                        "variant_group_chunks": "" if config.variant_group_chunks is None else str(config.variant_group_chunks),
                        "sample_group_chunks": "" if config.sample_group_chunks is None else str(config.sample_group_chunks),
                        "coalesce_min_density": "" if config.coalesce_min_density is None else str(config.coalesce_min_density),
                        "wall_seconds": float(elapsed),
                    }
                    _write_metadata(meta_path, metadata)
                    print(
                        "completed "
                        f"row={completed_rows}/{metadata['planned_rows']} "
                        f"case={case.name} workers={config.workers} "
                        f"vchunks={config.variant_group_chunks or 'default'} "
                        f"phase={phase} repeat={repeat} "
                        f"wall_seconds={elapsed:.3f}",
                        flush=True,
                    )

    if completed_rows == 0:
        raise RuntimeError("no benchmark rows completed")
    metadata["status"] = "complete"
    metadata["completed_unix"] = time.time()
    _write_metadata(meta_path, metadata)
    print(f"Wrote {completed_rows} benchmark rows to {rows_path}")
    print(f"Wrote benchmark metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
