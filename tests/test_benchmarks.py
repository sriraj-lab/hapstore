from __future__ import annotations

import csv
import json

import numpy as np
from benchmarks.bench_chunk_layout import _parse_layout
from benchmarks.bench_chunk_layout import main as chunk_layout_main
from benchmarks.bench_pgen_export import main as pgen_export_main
from benchmarks.bench_read_dosages import main
from hapstore import HapStore


def _make_benchmark_store(path):
    rng = np.random.default_rng(123)
    n_samples = 12
    n_variants = 48
    haplotypes = rng.integers(0, 2, size=(2 * n_samples, n_variants), dtype=np.uint8)
    haplotypes[0, 3] = 255
    haplotypes[5, 17] = 255
    store = HapStore(path, mode="w")
    store.create_chromosome(
        chrom="1",
        haplotypes=haplotypes,
        variant_pos=np.arange(100, 100 + n_variants, dtype=np.int64),
        sample_ids=np.asarray([f"s{i + 1}" for i in range(n_samples)], dtype=object),
        ref=np.asarray(["A"] * n_variants, dtype=object),
        alt=np.asarray(["G"] * n_variants, dtype=object),
        vid=np.asarray([f"v{i + 1}" for i in range(n_variants)], dtype=object),
        chunk_variant_size=6,
        chunk_haplotype_size=8,
        chunk_sample_size=4,
        store_genotypes=True,
    )
    return path


def test_bench_read_dosages_cli_writes_rows_and_metadata(tmp_path):
    store_path = _make_benchmark_store(tmp_path / "bench.zarr")
    out_dir = tmp_path / "bench_out"

    exit_code = main(
        [
            "--store",
            str(store_path),
            "--chrom",
            "1",
            "--out-dir",
            str(out_dir),
            "--case",
            "full_contiguous,sparse_random,batch_union,maf_scan",
            "--workers",
            "1,2",
            "--backend",
            "auto",
            "--variant-group-chunks",
            "1",
            "--sample-group-chunks",
            "1",
            "--coalesce-min-density",
            "0.25",
            "--reps",
            "1",
            "--warmups",
            "0",
            "--sparse-variants",
            "10",
            "--batch-requests",
            "3",
            "--batch-window-size",
            "5",
            "--correctness-variant-limit",
            "8",
        ]
    )
    assert exit_code == 0

    rows_path = out_dir / "benchmark_rows.tsv"
    meta_path = out_dir / "benchmark_metadata.json"
    assert rows_path.exists()
    assert meta_path.exists()

    with rows_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 8
    assert {row["case"] for row in rows} == {
        "full_contiguous",
        "sparse_random",
        "batch_union",
        "maf_scan",
    }
    assert all(row["phase"] == "timed" for row in rows)
    assert all(row["correctness_pass"] == "True" for row in rows)
    assert all(row["env_variant_group_chunks"] == "1" for row in rows)
    assert all(row["metrics_json"] for row in rows)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["store_metadata"]["n_samples"] == 12
    assert metadata["store_metadata"]["n_variants"] == 48
    assert metadata["store_metadata"]["dataset"] == "genotypes"


def test_bench_chunk_layout_cli_writes_and_compares_layouts(tmp_path):
    store_path = _make_benchmark_store(tmp_path / "source.zarr")
    out_dir = tmp_path / "chunk_results"
    work_dir = tmp_path / "chunk_work"

    exit_code = chunk_layout_main(
        [
            "--source-store",
            str(store_path),
            "--chrom",
            "1",
            "--work-dir",
            str(work_dir),
            "--out-dir",
            str(out_dir),
            "--layout",
            "full:6:full",
            "--layout",
            "bounded:12:8",
            "--sample-count",
            "8",
            "--variant-start",
            "4",
            "--variant-count",
            "24",
            "--read-case",
            "full_contiguous,low_fraction,batch_union",
            "--read-workers",
            "1",
            "--read-reps",
            "1",
            "--read-warmups",
            "0",
            "--contiguous-variants",
            "12",
            "--sparse-variants",
            "5",
            "--batch-requests",
            "2",
            "--batch-window-size",
            "4",
        ]
    )
    assert exit_code == 0

    with (out_dir / "layout_summary.tsv").open("r", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["layout"] for row in summary] == ["full", "bounded"]
    assert [row["haplotype_chunks_actual"] for row in summary] == ["16x6", "8x12"]
    assert all(row["correctness_pass"] == "True" for row in summary)

    with (out_dir / "read_benchmark_rows.tsv").open("r", encoding="utf-8") as handle:
        reads = list(csv.DictReader(handle, delimiter="\t"))
    assert len(reads) == 6
    assert {row["layout"] for row in reads} == {"full", "bounded"}
    assert all(row["correctness_pass"] == "True" for row in reads)
    assert not any(work_dir.iterdir())

    metadata = json.loads(
        (out_dir / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "complete"
    assert metadata["source_shape"] == [16, 24]
    assert metadata["completed_layouts"] == ["full", "bounded"]


def test_bench_chunk_layout_distinguishes_full_and_auto_chunks():
    assert _parse_layout("legacy:500:full").haplotype_chunk == "full"
    assert _parse_layout("default:500:auto").haplotype_chunk == "auto"


def test_bench_pgen_export_cli_runs_component_timings(tmp_path, capsys):
    store_path = _make_benchmark_store(tmp_path / "bench.zarr")

    exit_code = pgen_export_main(
        [
            "--store",
            str(store_path),
            "--chrom",
            "1",
            "--reps",
            "1",
            "--metadata-block-sizes",
            "4,8",
            "--sample-window-variants",
            "8",
            "--chunk-block-sizes",
            "4,8",
            "--chunk-max-variants",
            "16",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Variant metadata reads" in out
    assert "Haplotype read with" in out
    assert "Chunk alignment" in out
