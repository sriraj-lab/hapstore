"""Dense matrix copy backend for HapStore concatenation."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from hapstore.store import HapStore, _haplotypes_to_genotypes


@dataclass(frozen=True)
class _SourceWritePlan:
    source_path: Path
    source_n_haps: int
    source_n_vars: int
    out_hap_offset: int
    out_variant_offset: int


def _build_sample_axis_plans(sources: Sequence[Any]) -> list[_SourceWritePlan]:
    plans: list[_SourceWritePlan] = []
    hap_offset = 0
    for src in sources:
        plans.append(
            _SourceWritePlan(
                source_path=src.path,
                source_n_haps=int(src.n_haplotypes),
                source_n_vars=int(src.n_variants),
                out_hap_offset=int(hap_offset),
                out_variant_offset=0,
            )
        )
        hap_offset += int(src.n_haplotypes)
    return plans


def _build_variant_axis_plans(sources: Sequence[Any]) -> list[_SourceWritePlan]:
    plans: list[_SourceWritePlan] = []
    var_offset = 0
    for src in sources:
        plans.append(
            _SourceWritePlan(
                source_path=src.path,
                source_n_haps=int(src.n_haplotypes),
                source_n_vars=int(src.n_variants),
                out_hap_offset=0,
                out_variant_offset=int(var_offset),
            )
        )
        var_offset += int(src.n_variants)
    return plans


def _copy_dense_matrices(
    *,
    out_path: Path,
    chrom_key: str,
    axis: Literal["samples", "variants"],
    plans: Sequence[_SourceWritePlan],
    n_workers: int,
    hap_chunk: int,
    variant_chunk: int,
    store_genotypes: bool,
    sample_chunk_geno: int,
    logger: logging.Logger,
) -> None:
    if n_workers <= 1:
        for plan in plans:
            _worker_copy_source(
                source_path=plan.source_path,
                out_path=out_path,
                chrom_key=chrom_key,
                axis=axis,
                out_hap_offset=plan.out_hap_offset,
                out_variant_offset=plan.out_variant_offset,
                hap_chunk=hap_chunk,
                variant_chunk=variant_chunk,
                store_genotypes=store_genotypes,
                sample_chunk_geno=sample_chunk_geno,
                log_queue=None,
            )
        return

    if axis == "samples":
        total_haps = int(sum(p.source_n_haps for p in plans))
        chunk_disjoint = bool(all((int(p.source_n_haps) % int(hap_chunk)) == 0 for p in plans))
        if not chunk_disjoint:
            _copy_dense_matrices_parallel_by_output_hap_chunks(
                out_path=out_path,
                chrom_key=chrom_key,
                plans=plans,
                n_workers=n_workers,
                hap_chunk=hap_chunk,
                variant_chunk=variant_chunk,
                total_haps=total_haps,
                store_genotypes=store_genotypes,
                sample_chunk_geno=sample_chunk_geno,
                logger=logger,
            )
            return

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    listener = _start_queue_listener(queue=queue)
    try:
        args = []
        for plan in plans:
            args.append(
                dict(
                    source_path=plan.source_path,
                    out_path=out_path,
                    chrom_key=chrom_key,
                    axis=axis,
                    out_hap_offset=plan.out_hap_offset,
                    out_variant_offset=plan.out_variant_offset,
                    hap_chunk=hap_chunk,
                    variant_chunk=variant_chunk,
                    store_genotypes=store_genotypes,
                    sample_chunk_geno=sample_chunk_geno,
                    log_queue=queue,
                )
            )
        logger.info("COPY_START n_sources=%d n_workers=%d axis=%s", len(plans), n_workers, axis)
        with ctx.Pool(processes=int(n_workers)) as pool:
            pool.map(_worker_copy_source_star, args)
        logger.info("COPY_DONE n_sources=%d axis=%s", len(plans), axis)
    finally:
        listener.stop()
        manager.shutdown()


@dataclass(frozen=True)
class _OutputChunkTask:
    out_h0: int
    out_h1: int
    overlaps: tuple[tuple[Path, int, int, int, int], ...]


def _codec_config(codec: Any) -> Any:
    if codec is None:
        return None
    if hasattr(codec, "get_config"):
        return codec.get_config()
    return repr(codec)


def _filters_config(filters: Any) -> Any:
    if filters is None:
        return None
    return [_codec_config(filt) for filt in filters]


def _arrays_support_direct_chunk_copy(src_arr: Any, out_arr: Any) -> bool:
    return (
        tuple(src_arr.chunks) == tuple(out_arr.chunks)
        and np.dtype(src_arr.dtype) == np.dtype(out_arr.dtype)
        and getattr(src_arr, "order", "C") == getattr(out_arr, "order", "C")
        and _codec_config(getattr(src_arr, "compressor", None))
        == _codec_config(getattr(out_arr, "compressor", None))
        and _filters_config(getattr(src_arr, "filters", None))
        == _filters_config(getattr(out_arr, "filters", None))
        and getattr(src_arr, "fill_value", None) == getattr(out_arr, "fill_value", None)
        and getattr(src_arr, "chunk_store", None) is not None
        and getattr(out_arr, "chunk_store", None) is not None
    )


def _copy_raw_haplotype_chunks(
    *,
    src_arr: Any,
    out_arr: Any,
    src_chunk_index: int,
    out_chunk_index: int,
) -> None:
    n_var_chunks = int((int(src_arr.shape[1]) + int(src_arr.chunks[1]) - 1) // int(src_arr.chunks[1]))
    for var_chunk_index in range(n_var_chunks):
        src_key = src_arr._chunk_key((int(src_chunk_index), int(var_chunk_index)))
        out_key = out_arr._chunk_key((int(out_chunk_index), int(var_chunk_index)))
        if src_key in src_arr.chunk_store:
            out_arr.chunk_store[out_key] = src_arr.chunk_store[src_key]


def _try_direct_copy_source_chunks(
    *,
    src_haps: Any,
    out_haps: Any,
    out_hap_offset: int,
    src_n_haps: int,
    store_genotypes: bool,
    logger: logging.Logger,
) -> bool:
    if store_genotypes or not _arrays_support_direct_chunk_copy(src_haps, out_haps):
        return False

    hap_chunk = int(src_haps.chunks[0])
    if hap_chunk <= 0:
        return False
    if int(src_n_haps) % hap_chunk != 0:
        return False
    if int(out_hap_offset) % hap_chunk != 0:
        return False

    src_chunk0 = 0
    out_chunk0 = int(out_hap_offset) // hap_chunk
    n_chunks = int(src_n_haps) // hap_chunk
    logger.info(
        "COPY_FASTPATH axis=samples mode=source_chunks out_hap_offset=%d n_hap_chunks=%d",
        int(out_hap_offset),
        int(n_chunks),
    )
    for delta in range(n_chunks):
        _copy_raw_haplotype_chunks(
            src_arr=src_haps,
            out_arr=out_haps,
            src_chunk_index=int(src_chunk0 + delta),
            out_chunk_index=int(out_chunk0 + delta),
        )
    return True


def _try_direct_copy_output_hap_chunk(
    *,
    src_haps: Any,
    out_haps: Any,
    out_h0: int,
    out_h1: int,
    src_local0: int,
    src_local1: int,
    store_genotypes: bool,
    logger: logging.Logger,
) -> bool:
    if store_genotypes or not _arrays_support_direct_chunk_copy(src_haps, out_haps):
        return False

    hap_chunk = int(src_haps.chunks[0])
    out_chunk = int(out_haps.chunks[0])
    block_haps = int(out_h1) - int(out_h0)
    src_block_haps = int(src_local1) - int(src_local0)
    if block_haps != src_block_haps:
        return False
    if block_haps != out_chunk or src_block_haps != hap_chunk:
        return False
    if int(out_h0) % out_chunk != 0 or int(src_local0) % hap_chunk != 0:
        return False

    out_chunk_index = int(out_h0) // out_chunk
    src_chunk_index = int(src_local0) // hap_chunk
    logger.info(
        "COPY_FASTPATH axis=samples mode=output_hap_chunk out_hap_offset=%d out_hap_end=%d",
        int(out_h0),
        int(out_h1),
    )
    _copy_raw_haplotype_chunks(
        src_arr=src_haps,
        out_arr=out_haps,
        src_chunk_index=src_chunk_index,
        out_chunk_index=out_chunk_index,
    )
    return True


def _copy_dense_matrices_parallel_by_output_hap_chunks(
    *,
    out_path: Path,
    chrom_key: str,
    plans: Sequence[_SourceWritePlan],
    n_workers: int,
    hap_chunk: int,
    variant_chunk: int,
    total_haps: int,
    store_genotypes: bool,
    sample_chunk_geno: int,
    logger: logging.Logger,
) -> None:
    del sample_chunk_geno
    if total_haps <= 0:
        raise ValueError("total_haps must be > 0")

    tasks: list[_OutputChunkTask] = []
    for out_h0 in range(0, int(total_haps), int(hap_chunk)):
        out_h1 = min(int(out_h0) + int(hap_chunk), int(total_haps))
        overlaps: list[tuple[Path, int, int, int, int]] = []
        for p in plans:
            src_global0 = int(p.out_hap_offset)
            src_global1 = int(p.out_hap_offset) + int(p.source_n_haps)
            if src_global1 <= int(out_h0) or src_global0 >= int(out_h1):
                continue
            ov0 = max(int(out_h0), src_global0)
            ov1 = min(int(out_h1), src_global1)
            out_local0 = ov0 - int(out_h0)
            out_local1 = ov1 - int(out_h0)
            src_local0 = ov0 - src_global0
            src_local1 = ov1 - src_global0
            overlaps.append((p.source_path, out_local0, out_local1, src_local0, src_local1))
        if not overlaps:
            raise AssertionError(f"no overlaps for output hap slice [{out_h0}, {out_h1})")
        tasks.append(_OutputChunkTask(out_h0=int(out_h0), out_h1=int(out_h1), overlaps=tuple(overlaps)))

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    listener = _start_queue_listener(queue=queue)
    try:
        args = []
        for task in tasks:
            args.append(
                dict(
                    out_path=out_path,
                    chrom_key=chrom_key,
                    out_h0=int(task.out_h0),
                    out_h1=int(task.out_h1),
                    overlaps=task.overlaps,
                    variant_chunk=int(variant_chunk),
                    log_queue=queue,
                )
            )
        worker_count = min(int(n_workers), len(args))
        logger.info(
            "COPY_START n_output_hap_chunks=%d n_workers=%d axis=samples mode=output_hap_chunks",
            len(args),
            worker_count,
        )
        with ctx.Pool(processes=int(worker_count)) as pool:
            pool.map(_worker_copy_output_hap_chunk_star, args)
        logger.info("COPY_DONE n_output_hap_chunks=%d axis=samples mode=output_hap_chunks", len(args))
    finally:
        listener.stop()
        manager.shutdown()


def _worker_copy_output_hap_chunk_star(kwargs: dict) -> None:
    _worker_copy_output_hap_chunk(**kwargs)


def _worker_copy_output_hap_chunk(
    *,
    out_path: Path,
    chrom_key: str,
    out_h0: int,
    out_h1: int,
    overlaps: tuple[tuple[Path, int, int, int, int], ...],
    variant_chunk: int,
    log_queue,
) -> None:
    import logging.handlers

    logger = logging.getLogger(__name__)
    if log_queue is not None:
        logger.handlers = []
        logger.addHandler(logging.handlers.QueueHandler(log_queue))
        logger.setLevel(logging.INFO)
        logger.propagate = False

    out = HapStore(out_path, mode="a")
    out_group = out.root[chrom_key]
    out_haps = out_group["haplotypes"]
    n_vars = int(out_haps.shape[1])
    block_haps = int(out_h1) - int(out_h0)

    sources: dict[Path, HapStore] = {}
    src_haps: dict[Path, Any] = {}
    for src_path, _, _, _, _ in overlaps:
        if src_path in sources:
            continue
        hs = HapStore(src_path, mode="r")
        sources[src_path] = hs
        src_haps[src_path] = hs.root[chrom_key]["haplotypes"]

    logger.info(
        "COPY_PROGRESS axis=samples mode=output_hap_chunks out_hap_offset=%d out_hap_end=%d n_sources=%d",
        int(out_h0),
        int(out_h1),
        len(sources),
    )

    if len(overlaps) == 1:
        src_path, _out_local0, _out_local1, src_local0, src_local1 = overlaps[0]
        if _try_direct_copy_output_hap_chunk(
            src_haps=src_haps[src_path],
            out_haps=out_haps,
            out_h0=int(out_h0),
            out_h1=int(out_h1),
            src_local0=int(src_local0),
            src_local1=int(src_local1),
            store_genotypes=("genotypes" in out_group),
            logger=logger,
        ):
            return

    for v0 in range(0, n_vars, int(variant_chunk)):
        v1 = min(int(v0) + int(variant_chunk), n_vars)
        block = np.empty((block_haps, int(v1) - int(v0)), dtype=np.uint8)
        for src_path, out_local0, out_local1, src_local0, src_local1 in overlaps:
            block[int(out_local0) : int(out_local1), :] = np.asarray(
                src_haps[src_path][int(src_local0) : int(src_local1), int(v0) : int(v1)],
                dtype=np.uint8,
            )
        out_haps[int(out_h0) : int(out_h1), int(v0) : int(v1)] = block

        if "genotypes" in out_group:
            if (int(out_h0) % 2) != 0 or (int(out_h1) % 2) != 0:
                raise ValueError(
                    "output hap slice must align to diploid sample boundaries when writing genotypes: "
                    f"out_h0={out_h0}, out_h1={out_h1}"
                )
            sample0 = int(out_h0) // 2
            sample1 = int(out_h1) // 2
            out_group["genotypes"][sample0:sample1, int(v0) : int(v1)] = _haplotypes_to_genotypes(block)


def _start_queue_listener(queue):
    import logging.handlers
    import sys

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s [%(levelname)s] pid=%(process)d %(name)s: %(message)s")
    )
    listener = logging.handlers.QueueListener(queue, handler, respect_handler_level=True)
    listener.start()
    return listener


def _worker_copy_source_star(kwargs: dict) -> None:
    _worker_copy_source(**kwargs)


def _worker_copy_source(
    *,
    source_path: Path,
    out_path: Path,
    chrom_key: str,
    axis: Literal["samples", "variants"],
    out_hap_offset: int,
    out_variant_offset: int,
    hap_chunk: int,
    variant_chunk: int,
    store_genotypes: bool,
    sample_chunk_geno: int,
    log_queue,
) -> None:
    import logging.handlers

    logger = logging.getLogger(__name__)
    if log_queue is not None:
        logger.handlers = []
        logger.addHandler(logging.handlers.QueueHandler(log_queue))
        logger.setLevel(logging.INFO)
        logger.propagate = False

    src = HapStore(source_path, mode="r")
    out = HapStore(out_path, mode="a")
    src_group = src.root[chrom_key]
    out_group = out.root[chrom_key]
    src_haps = src_group["haplotypes"]
    out_haps = out_group["haplotypes"]

    src_n_haps = int(src_haps.shape[0])
    src_n_vars = int(src_haps.shape[1])

    logger.info(
        "COPY_PROGRESS source=%s axis=%s out_hap_offset=%d out_variant_offset=%d",
        str(source_path),
        axis,
        int(out_hap_offset),
        int(out_variant_offset),
    )

    if axis == "samples":
        if _try_direct_copy_source_chunks(
            src_haps=src_haps,
            out_haps=out_haps,
            out_hap_offset=int(out_hap_offset),
            src_n_haps=int(src_n_haps),
            store_genotypes=bool(store_genotypes),
            logger=logger,
        ):
            logger.info("COPY_DONE source=%s axis=%s", str(source_path), axis)
            return

        for h0 in range(0, src_n_haps, hap_chunk):
            h1 = min(h0 + hap_chunk, src_n_haps)
            out_h0 = out_hap_offset + h0
            out_h1 = out_hap_offset + h1
            for v0 in range(0, src_n_vars, variant_chunk):
                v1 = min(v0 + variant_chunk, src_n_vars)
                out_haps[out_h0:out_h1, v0:v1] = np.asarray(src_haps[h0:h1, v0:v1], dtype=np.uint8)

            if store_genotypes:
                geno_out = out_group["genotypes"]
                sample0 = out_h0 // 2
                sample1 = out_h1 // 2
                for v0 in range(0, src_n_vars, variant_chunk):
                    v1 = min(v0 + variant_chunk, src_n_vars)
                    h_block = np.asarray(src_haps[h0:h1, v0:v1], dtype=np.uint8)
                    geno_out[sample0:sample1, v0:v1] = _haplotypes_to_genotypes(h_block)

    else:
        out_var0 = int(out_variant_offset)
        out_n_haps = int(out_haps.shape[0])
        for h0 in range(0, out_n_haps, hap_chunk):
            h1 = min(h0 + hap_chunk, out_n_haps)
            for v0 in range(0, src_n_vars, variant_chunk):
                v1 = min(v0 + variant_chunk, src_n_vars)
                out_haps[h0:h1, out_var0 + v0 : out_var0 + v1] = np.asarray(
                    src_haps[h0:h1, v0:v1],
                    dtype=np.uint8,
                )

        if store_genotypes:
            geno_out = out_group["genotypes"]
            for s0 in range(0, int(geno_out.shape[0]), sample_chunk_geno):
                s1 = min(s0 + sample_chunk_geno, int(geno_out.shape[0]))
                h0 = 2 * s0
                h1 = 2 * s1
                for v0 in range(0, src_n_vars, variant_chunk):
                    v1 = min(v0 + variant_chunk, src_n_vars)
                    h_block = np.asarray(src_haps[h0:h1, v0:v1], dtype=np.uint8)
                    geno_out[s0:s1, out_var0 + v0 : out_var0 + v1] = _haplotypes_to_genotypes(h_block)

    logger.info("COPY_DONE source=%s axis=%s", str(source_path), axis)
