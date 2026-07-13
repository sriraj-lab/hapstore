#!/usr/bin/env python
"""Benchmark HapStore write/read behavior across explicit chunk layouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hapstore import HapStore  # noqa: E402
from hapstore.chrom import normalize_chrom  # noqa: E402

from benchmarks.bench_read_dosages import main as read_benchmark_main  # noqa: E402


@dataclass(frozen=True)
class Layout:
    name: str
    variant_chunk: int
    haplotype_chunk: int | None
    sample_chunk: int | None


SUMMARY_COLUMNS = (
    "layout",
    "variant_chunk_requested",
    "haplotype_chunk_requested",
    "sample_chunk_requested",
    "haplotype_chunks_actual",
    "genotype_chunks_actual",
    "write_seconds",
    "store_bytes",
    "n_samples",
    "n_variants",
    "correctness_pass",
    "max_rss_mb",
)


def _positive_int(value: str, *, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be positive, got {parsed}")
    return parsed


def _optional_chunk(value: str, *, name: str) -> int | None:
    if value.strip().lower() in {"full", "none", "auto"}:
        return None
    return _positive_int(value, name=name)


def _parse_layout(value: str) -> Layout:
    parts = value.split(":")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "layout must be NAME:VARIANT_CHUNK:HAPLOTYPE_CHUNK[:SAMPLE_CHUNK]"
        )
    name = parts[0].strip()
    if not name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in name
    ):
        raise argparse.ArgumentTypeError(f"invalid layout name {name!r}")
    variant_chunk = _positive_int(parts[1], name="variant chunk")
    haplotype_chunk = _optional_chunk(parts[2], name="haplotype chunk")
    sample_chunk = (
        _optional_chunk(parts[3], name="sample chunk") if len(parts) == 4 else None
    )
    return Layout(name, variant_chunk, haplotype_chunk, sample_chunk)


def _run_git(args: Sequence[str]) -> str | None:
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
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _directory_size(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _append_summary(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_read_rows(
    *,
    source: Path,
    destination: Path,
    layout: Layout,
    actual_haplotype_chunks: tuple[int, int],
) -> None:
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"read benchmark produced no rows: {source}")
    prefix = {
        "layout": layout.name,
        "layout_variant_chunk": str(layout.variant_chunk),
        "layout_haplotype_chunk": "full"
        if layout.haplotype_chunk is None
        else str(layout.haplotype_chunk),
        "layout_actual_chunks": "x".join(
            str(value) for value in actual_haplotype_chunks
        ),
    }
    fieldnames = [*prefix, *rows[0].keys()]
    write_header = not destination.exists()
    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({**prefix, **row})


def _load_source_slice(
    *,
    store: HapStore,
    chrom: str,
    sample_count: int | None,
    variant_start: int,
    variant_count: int,
) -> dict[str, Any]:
    n_samples_total = store.n_samples(chrom)
    n_variants_total = store.n_variants(chrom)
    n_samples = (
        n_samples_total
        if sample_count is None
        else min(int(sample_count), n_samples_total)
    )
    variant_stop = min(n_variants_total, int(variant_start) + int(variant_count))
    if (
        variant_start < 0
        or variant_start >= n_variants_total
        or variant_stop <= variant_start
    ):
        raise ValueError(
            f"invalid variant range {variant_start}:{variant_stop} "
            f"for {n_variants_total} variants"
        )
    sample_mask = None
    if n_samples != n_samples_total:
        sample_mask = np.zeros(n_samples_total, dtype=bool)
        sample_mask[:n_samples] = True
    haplotypes = store.haplotypes(
        chrom,
        variant_slice=(variant_start, variant_stop),
        sample_mask=sample_mask,
    )
    chrom_group = store.root[normalize_chrom(chrom, prefix=True)]
    variant_indices = np.arange(variant_start, variant_stop, dtype=np.int64)
    info = store._read_variant_metadata_block(
        chrom_group=chrom_group,
        variant_indices=variant_indices,
    )
    populations = None
    if "population" in chrom_group["samples"]:
        populations = np.asarray(
            chrom_group["samples"]["population"][:n_samples], dtype=object
        )
    return {
        "haplotypes": haplotypes,
        "variant_pos": np.asarray(info["positions"], dtype=np.int64),
        "sample_ids": np.asarray(store.sample_ids(chrom)[:n_samples], dtype=object),
        "ref": np.asarray(info["ref"], dtype=object),
        "alt": np.asarray(info["alt"], dtype=object),
        "vid": np.asarray(info["vid"], dtype=object),
        "genetic_positions": None
        if info["genetic_pos"] is None
        else info["genetic_pos"],
        "populations": populations,
        "has_missing": bool(store.attrs(chrom).get("has_missing", True)),
    }


def _verify_candidate(
    *,
    candidate: HapStore,
    chrom: str,
    source: dict[str, Any],
    seed: int,
) -> None:
    n_samples = int(len(source["sample_ids"]))
    rng = np.random.default_rng(seed)
    selected = np.asarray(
        sorted(rng.choice(n_samples, size=min(64, n_samples), replace=False).tolist()),
        dtype=np.int64,
    )
    sample_mask = np.zeros(n_samples, dtype=bool)
    sample_mask[selected] = True
    hap_mask = np.empty(2 * n_samples, dtype=bool)
    hap_mask[0::2] = sample_mask
    hap_mask[1::2] = sample_mask
    observed = candidate.haplotypes(chrom, sample_mask=sample_mask)
    np.testing.assert_array_equal(observed, source["haplotypes"][hap_mask])
    info = candidate.variant_info(chrom)
    np.testing.assert_array_equal(info["pos"], source["variant_pos"])
    if info["vid"] != source["vid"].tolist():
        raise AssertionError("candidate variant IDs differ from source")
    if info["ref"] != source["ref"].tolist() or info["alt"] != source["alt"].tolist():
        raise AssertionError("candidate alleles differ from source")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store", required=True, type=Path)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--layout",
        action="append",
        type=_parse_layout,
        required=True,
        help=(
            "NAME:VARIANT_CHUNK:HAPLOTYPE_CHUNK[:SAMPLE_CHUNK]; "
            "use full for a full row axis"
        ),
    )
    parser.add_argument(
        "--sample-count", type=lambda value: _positive_int(value, name="sample count")
    )
    parser.add_argument("--variant-start", type=int, default=0)
    parser.add_argument(
        "--variant-count",
        type=lambda value: _positive_int(value, name="variant count"),
        default=5000,
    )
    parser.add_argument("--store-genotypes", action="store_true")
    parser.add_argument("--read-case", action="append")
    parser.add_argument("--read-workers", default="auto")
    parser.add_argument("--read-backend", default="auto")
    parser.add_argument(
        "--read-reps",
        type=lambda value: _positive_int(value, name="read reps"),
        default=2,
    )
    parser.add_argument("--read-warmups", type=int, default=1)
    parser.add_argument(
        "--contiguous-variants",
        type=lambda value: _positive_int(value, name="contiguous variants"),
        default=5000,
    )
    parser.add_argument(
        "--sparse-variants",
        type=lambda value: _positive_int(value, name="sparse variants"),
        default=512,
    )
    parser.add_argument(
        "--batch-requests",
        type=lambda value: _positive_int(value, name="batch requests"),
        default=4,
    )
    parser.add_argument(
        "--batch-window-size",
        type=lambda value: _positive_int(value, name="batch window size"),
        default=128,
    )
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--keep-stores", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.variant_start < 0:
        parser.error("--variant-start must be >= 0")
    if args.read_warmups < 0:
        parser.error("--read-warmups must be >= 0")
    if not args.source_store.exists():
        parser.error(f"--source-store does not exist: {args.source_store}")
    names = [layout.name for layout in args.layout]
    if len(set(names)) != len(names):
        parser.error("layout names must be unique")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "layout_summary.tsv"
    combined_reads_path = args.out_dir / "read_benchmark_rows.tsv"
    metadata_path = args.out_dir / "benchmark_metadata.json"
    summary_path.unlink(missing_ok=True)
    combined_reads_path.unlink(missing_ok=True)
    metadata: dict[str, Any] = {
        "status": "loading_source",
        "argv": sys.argv if argv is None else [str(Path(__file__)), *argv],
        "source_store": args.source_store,
        "chrom": args.chrom,
        "sample_count": args.sample_count,
        "variant_start": args.variant_start,
        "variant_count": args.variant_count,
        "layouts": [layout.__dict__ for layout in args.layout],
        "git_commit": _run_git(["rev-parse", "HEAD"]),
        "git_status_short": _run_git(["status", "--short"]),
        "completed_layouts": [],
    }
    _write_json(metadata_path, metadata)

    load_start = time.perf_counter()
    source_store = HapStore(args.source_store, mode="r")
    source = _load_source_slice(
        store=source_store,
        chrom=args.chrom,
        sample_count=args.sample_count,
        variant_start=args.variant_start,
        variant_count=args.variant_count,
    )
    metadata["source_load_seconds"] = time.perf_counter() - load_start
    metadata["source_shape"] = tuple(int(value) for value in source["haplotypes"].shape)
    metadata["status"] = "running"
    _write_json(metadata_path, metadata)

    requested_cases = args.read_case or [
        "full_contiguous,high_fraction,low_fraction,sparse_random,batch_union"
    ]
    for layout_index, layout in enumerate(args.layout):
        candidate_path = args.work_dir / f"{layout.name}.zarr"
        if candidate_path.exists():
            shutil.rmtree(candidate_path)
        print(
            f"layout_start name={layout.name} variant_chunk={layout.variant_chunk} "
            f"haplotype_chunk={layout.haplotype_chunk or 'full'}",
            flush=True,
        )
        write_start = time.perf_counter()
        candidate = HapStore(candidate_path, mode="w")
        candidate.create_chromosome(
            chrom=args.chrom,
            haplotypes=source["haplotypes"],
            variant_pos=source["variant_pos"],
            sample_ids=source["sample_ids"],
            ref=source["ref"],
            alt=source["alt"],
            vid=source["vid"],
            genetic_positions=source["genetic_positions"],
            populations=source["populations"],
            store_genotypes=bool(args.store_genotypes),
            chunk_variant_size=layout.variant_chunk,
            chunk_haplotype_size=layout.haplotype_chunk,
            chunk_sample_size=layout.sample_chunk,
        )
        write_seconds = time.perf_counter() - write_start
        _verify_candidate(
            candidate=candidate,
            chrom=args.chrom,
            source=source,
            seed=int(args.seed) + layout_index,
        )
        chrom_group = candidate.root[normalize_chrom(args.chrom, prefix=True)]
        hap_chunks = tuple(int(value) for value in chrom_group["haplotypes"].chunks)
        genotype_chunks = None
        if "genotypes" in chrom_group:
            genotype_chunks = tuple(
                int(value) for value in chrom_group["genotypes"].chunks
            )
        summary_row = {
            "layout": layout.name,
            "variant_chunk_requested": layout.variant_chunk,
            "haplotype_chunk_requested": "full"
            if layout.haplotype_chunk is None
            else layout.haplotype_chunk,
            "sample_chunk_requested": "full"
            if layout.sample_chunk is None
            else layout.sample_chunk,
            "haplotype_chunks_actual": "x".join(str(value) for value in hap_chunks),
            "genotype_chunks_actual": ""
            if genotype_chunks is None
            else "x".join(str(value) for value in genotype_chunks),
            "write_seconds": f"{write_seconds:.9f}",
            "store_bytes": _directory_size(candidate_path),
            "n_samples": candidate.n_samples(args.chrom),
            "n_variants": candidate.n_variants(args.chrom),
            "correctness_pass": True,
            "max_rss_mb": f"{_max_rss_mb():.3f}",
        }
        _append_summary(summary_path, summary_row)

        read_out = args.out_dir / "reads" / layout.name
        read_args = [
            "--store",
            str(candidate_path),
            "--chrom",
            str(args.chrom),
            "--out-dir",
            str(read_out),
            "--workers",
            str(args.read_workers),
            "--backend",
            str(args.read_backend),
            "--variant-group-chunks",
            "4",
            "--sample-group-chunks",
            "2",
            "--coalesce-min-density",
            "0.25",
            "--reps",
            str(args.read_reps),
            "--warmups",
            str(args.read_warmups),
            "--contiguous-variants",
            str(min(args.contiguous_variants, len(source["variant_pos"]))),
            "--sparse-variants",
            str(min(args.sparse_variants, len(source["variant_pos"]))),
            "--batch-requests",
            str(args.batch_requests),
            "--batch-window-size",
            str(min(args.batch_window_size, len(source["variant_pos"]))),
            "--correctness-variant-limit",
            str(min(64, len(source["variant_pos"]))),
            "--seed",
            str(args.seed),
        ]
        for case in requested_cases:
            read_args.extend(["--case", case])
        if not source["has_missing"]:
            read_args.append("--assume-no-missing")
        if read_benchmark_main(read_args) != 0:
            raise RuntimeError(f"read benchmark failed for layout {layout.name}")
        _append_read_rows(
            source=read_out / "benchmark_rows.tsv",
            destination=combined_reads_path,
            layout=layout,
            actual_haplotype_chunks=hap_chunks,
        )
        metadata["completed_layouts"].append(layout.name)
        metadata["last_layout"] = summary_row
        _write_json(metadata_path, metadata)
        if not args.keep_stores:
            shutil.rmtree(candidate_path)
        print(
            f"layout_complete name={layout.name} write_seconds={write_seconds:.3f} "
            f"chunks={hap_chunks} store_bytes={summary_row['store_bytes']}",
            flush=True,
        )

    metadata["status"] = "complete"
    metadata["completed_unix"] = time.time()
    _write_json(metadata_path, metadata)
    print(f"Wrote layout summary to {summary_path}", flush=True)
    print(f"Wrote combined read rows to {combined_reads_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
