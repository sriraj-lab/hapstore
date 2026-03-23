"""Shared ingest dataclasses and type aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from hapstore.store import _MISSING_SENTINEL

Backend = Literal["auto", "cyvcf2", "pysam", "text", "plink2_bridge"]
MissingRefAltPolicy = Literal["error", "dot"]
MissingVidPolicy = Literal["dot", "error"]
UnphasedPolicy = Literal["missing", "error"]


@dataclass
class _IngestArrays:
    haplotypes: np.ndarray
    positions: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    vid: np.ndarray
    sample_ids: np.ndarray
    missing_ref_count: int
    missing_alt_count: int
    missing_vid_count: int
    unphased_gt_count: int


@dataclass
class _ShardResult:
    shard_index: int
    haplotypes: np.ndarray
    positions: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    vid: np.ndarray
    missing_ref_count: int
    missing_alt_count: int
    missing_vid_count: int
    unphased_gt_count: int


@dataclass
class _ShardArtifactMeta:
    shard_index: int
    start: int | None
    end: int | None
    artifact_path: str
    n_variants: int
    missing_ref_count: int
    missing_alt_count: int
    missing_vid_count: int
    unphased_gt_count: int


@dataclass
class _ShardSpec:
    shard_index: int
    start_bp: int | None
    end_bp: int | None


@dataclass
class _PartitionPlan:
    partition_bp: int
    shards: list[_ShardSpec]


@dataclass
class _StreamingFinalizeContext:
    store: Any
    chrom_key: str
    chrom_group: Any
    hap_arr: Any
    geno_arr: Any | None
    pos_arr: Any
    ref_arr: Any
    alt_arr: Any
    vid_arr: Any
    genetic_arr: Any | None
    n_samples: int
    n_haplotypes: int
    n_shards: int
    variant_chunk: int
    hap_chunk: int
    sample_chunk_geno: int
    compression_level: int
    chunk_haplotype_size: int | None
    chunk_sample_size: int | None
    capacity: int
    offset: int = 0
    has_missing: bool = False
    missing_ref_count: int = 0
    missing_alt_count: int = 0
    missing_vid_count: int = 0
    unphased_gt_count: int = 0
    prev_last_position: int | None = None
    finished: bool = False
