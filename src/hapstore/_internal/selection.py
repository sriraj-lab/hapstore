"""Shared sample/variant selection helpers for VCF and PGEN exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import zarr

from .zarr_utils import _to_unicode_array


@dataclass(frozen=True)
class _SampleSubsetPlan:
    selected_sample_ids: list[str]
    sample_mask: np.ndarray | None
    hap_perm: np.ndarray | None


@dataclass(frozen=True)
class _ParsedVariantId:
    kind: str  # "pos" or "vid"
    value: int | str


def _plan_sample_subset(all_sample_ids: list[str], subset: list[str] | None) -> _SampleSubsetPlan:
    if subset is None:
        return _SampleSubsetPlan(
            selected_sample_ids=list(all_sample_ids),
            sample_mask=None,
            hap_perm=None,
        )

    if len(set(subset)) != len(subset):
        raise ValueError("sample_ids must be unique")

    index_by_id = {sid: i for i, sid in enumerate(all_sample_ids)}
    missing = [sid for sid in subset if sid not in index_by_id]
    if missing:
        raise ValueError(f"sample_ids not found in store: first missing={missing[0]!r}")

    requested_indices = np.asarray([index_by_id[sid] for sid in subset], dtype=np.int64)
    sorted_indices = np.sort(requested_indices)
    sample_mask = np.zeros(len(all_sample_ids), dtype=bool)
    sample_mask[sorted_indices] = True

    selected_sample_ids = [all_sample_ids[int(i)] for i in sorted_indices.tolist()]
    if np.array_equal(requested_indices, sorted_indices):
        hap_perm = None
    else:
        position_by_sorted = {int(idx): pos for pos, idx in enumerate(sorted_indices.tolist())}
        sample_perm = np.asarray([position_by_sorted[int(idx)] for idx in requested_indices.tolist()], dtype=np.int64)
        hap_perm = np.empty(2 * len(sample_perm), dtype=np.int64)
        hap_perm[0::2] = 2 * sample_perm
        hap_perm[1::2] = 2 * sample_perm + 1
        selected_sample_ids = list(subset)

    return _SampleSubsetPlan(
        selected_sample_ids=selected_sample_ids,
        sample_mask=sample_mask,
        hap_perm=hap_perm,
    )


def _iter_variant_slices(n_variants: int, variant_block_size: int) -> Iterable[slice]:
    for start in range(0, int(n_variants), int(variant_block_size)):
        stop = min(int(start) + int(variant_block_size), int(n_variants))
        yield slice(int(start), int(stop))


def _iter_variant_slices_for_indices(indices: list[int], variant_block_size: int) -> Iterable[slice]:
    if not indices:
        return

    ordered = [int(i) for i in indices]
    block_size = int(variant_block_size)
    block_start = ordered[0]
    prev = ordered[0]
    count = 1
    for idx in ordered[1:]:
        if idx == prev + 1 and count < block_size:
            prev = idx
            count += 1
            continue
        yield slice(int(block_start), int(prev + 1))
        block_start = idx
        prev = idx
        count = 1
    yield slice(int(block_start), int(prev + 1))


def _subset_indices_in_requested_order(all_sample_ids: list[str], subset: list[str]) -> list[int]:
    index_by_id = {sid: i for i, sid in enumerate(all_sample_ids)}
    return [int(index_by_id[sid]) for sid in subset]


def _resolve_variant_indices_streaming(
    *,
    chrom_key: str,
    pos_ds: zarr.Array,
    vid_ds: zarr.Array,
    variant_ids: list[str],
    scan_block_size: int,
) -> list[int]:
    """Resolve unique position or ID aliases through chunk-bounded scans.

    Returned indices are unique and sorted in HapStore variant order, not the
    order in which aliases were requested.
    """
    if len(set(variant_ids)) != len(variant_ids):
        raise ValueError("variant_ids must be unique")
    if not variant_ids:
        raise ValueError("variant_ids must be non-empty when provided")

    requested = [_parse_variant_id(chrom_key, raw) for raw in variant_ids]
    requested_positions = {int(entry.value) for entry in requested if entry.kind == "pos"}
    requested_vids = {str(entry.value) for entry in requested if entry.kind == "vid"}

    found_pos: dict[int, int] = {}
    found_vid: dict[str, int] = {}

    n_variants = int(pos_ds.shape[0])
    for variant_sel in _iter_variant_slices(n_variants, int(scan_block_size)):
        start = int(variant_sel.start or 0)
        stop = int(variant_sel.stop or 0)
        if start >= stop:
            continue

        if requested_positions:
            pos_chunk = np.asarray(pos_ds[start:stop], dtype=np.int64)
            for offset, pos in enumerate(pos_chunk.tolist()):
                pos_int = int(pos)
                if pos_int in requested_positions:
                    if pos_int in found_pos:
                        raise ValueError(
                            f"ambiguous position in store for requested pos={pos_int}: "
                            f"indices {found_pos[pos_int]} and {start + offset}"
                        )
                    found_pos[pos_int] = int(start + offset)

        if requested_vids:
            vid_chunk = _to_unicode_array(np.asarray(vid_ds[start:stop], dtype=object))
            for offset, vid in enumerate(vid_chunk.tolist()):
                vid_str = str(vid)
                if vid_str in requested_vids:
                    if vid_str in found_vid:
                        raise ValueError(
                            f"ambiguous vid in store for requested vid={vid_str!r}: "
                            f"indices {found_vid[vid_str]} and {start + offset}"
                        )
                    found_vid[vid_str] = int(start + offset)

    missing_positions = sorted([p for p in requested_positions if p not in found_pos])
    missing_vids = sorted([v for v in requested_vids if v not in found_vid])
    if missing_positions or missing_vids:
        parts = []
        if missing_positions:
            parts.append(f"positions={missing_positions}")
        if missing_vids:
            parts.append(f"vids={missing_vids}")
        raise ValueError("variant_ids not found in store: " + ", ".join(parts))

    indices = list(found_pos.values()) + list(found_vid.values())
    return sorted(set(int(i) for i in indices))


_POS_LINE_RE = re.compile(r"(?i)\bline\s+(\d+)\b")


def _parse_variant_id(chrom_key: str, raw: str) -> _ParsedVariantId:
    text = str(raw).strip()
    if text == "":
        raise ValueError("variant_ids contains an empty string")

    match = _POS_LINE_RE.search(text)
    if match is not None:
        pos = int(match.group(1))
        chrom_part = text.split("(")[0].strip()
        if chrom_part:
            chrom_in_id = chrom_part if chrom_part.startswith("chr") else f"chr{chrom_part}"
            if chrom_in_id != chrom_key:
                raise ValueError(f"variant_id chrom mismatch: {raw!r} vs chrom={chrom_key!r}")
        return _ParsedVariantId(kind="pos", value=pos)

    if ":" in text:
        left, right = text.rsplit(":", 1)
        if right.isdigit():
            chrom_in_id = left.strip()
            chrom_in_id = chrom_in_id if chrom_in_id.startswith("chr") else f"chr{chrom_in_id}"
            if chrom_in_id != chrom_key:
                raise ValueError(f"variant_id chrom mismatch: {raw!r} vs chrom={chrom_key!r}")
            return _ParsedVariantId(kind="pos", value=int(right))

    if text.isdigit():
        return _ParsedVariantId(kind="pos", value=int(text))

    return _ParsedVariantId(kind="vid", value=text)
