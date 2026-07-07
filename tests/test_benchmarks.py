from __future__ import annotations

import csv
import json

import numpy as np

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
        sample_ids=np.asarray([f"s{i+1}" for i in range(n_samples)], dtype=object),
        ref=np.asarray(["A"] * n_variants, dtype=object),
        alt=np.asarray(["G"] * n_variants, dtype=object),
        vid=np.asarray([f"v{i+1}" for i in range(n_variants)], dtype=object),
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
