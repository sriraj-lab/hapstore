"""Tabix/CSI index parsing for hapstore ingest."""

from __future__ import annotations

import gzip
import os
import struct
from pathlib import Path


def _has_tabix_index(vcf_path: Path) -> bool:
    p = str(vcf_path)
    return Path(f"{p}.tbi").exists() or Path(f"{p}.csi").exists()


def _resolve_contig_span_end_from_tabix_index(vcf_path: Path, contig: str) -> int | None:
    """Best-effort estimate of a contig's exclusive end from a tabix index."""
    tbi = Path(f"{vcf_path}.tbi")
    if tbi.exists():
        end = _contig_span_end_from_tbi(tbi, contig)
        if end is not None:
            return int(end)
    csi = Path(f"{vcf_path}.csi")
    if csi.exists():
        end = _contig_span_end_from_csi(csi, contig)
        if end is not None:
            return int(end)
    return None


def _contig_span_end_from_tbi(tbi_path: Path, contig: str) -> int | None:
    contig_b = contig.encode("utf-8")
    try:
        with _open_tabix_index(tbi_path) as f:
            magic = f.read(4)
            if magic != b"TBI\x01":
                return None
            n_ref = _read_int32(f)
            _ = f.read(24)  # format + columns + meta + skip
            l_nm = _read_int32(f)
            names = f.read(int(l_nm))
            ref_names = names.rstrip(b"\0").split(b"\0") if names else []
            try:
                rid = ref_names.index(contig_b)
            except ValueError:
                return None

            min_shift = 14
            for ref_idx in range(int(n_ref)):
                n_bin = _read_int32(f)
                for _ in range(int(n_bin)):
                    f.read(4)  # bin
                    n_chunk = _read_int32(f)
                    f.seek(16 * int(n_chunk), os.SEEK_CUR)
                n_intv = _read_int32(f)
                if ref_idx == rid:
                    return int(n_intv) << int(min_shift)
                f.seek(8 * int(n_intv), os.SEEK_CUR)
    except Exception:  # pragma: no cover
        return None
    return None


def _contig_span_end_from_csi(csi_path: Path, contig: str) -> int | None:
    contig_b = contig.encode("utf-8")
    try:
        with _open_tabix_index(csi_path) as f:
            magic = f.read(4)
            if magic != b"CSI\x01":
                return None
            min_shift = _read_int32(f)
            depth = _read_int32(f)
            l_aux = _read_int32(f)
            aux = f.read(int(l_aux))
            rid = _resolve_ref_id_from_csi_aux(aux, contig_b)
            if rid is None:
                return None
            n_ref = _read_int32(f)

            for ref_idx in range(int(n_ref)):
                n_bin = _read_int32(f)
                span_end = None
                for _ in range(int(n_bin)):
                    bin_id = _read_uint32(f)
                    f.read(8)  # loffset
                    n_chunk = _read_int32(f)
                    f.seek(16 * int(n_chunk), os.SEEK_CUR)
                    if ref_idx == int(rid):
                        bin_end = _csi_bin_span_end(bin_id, min_shift=min_shift, depth=depth)
                        if bin_end is not None:
                            span_end = max(int(span_end or 0), int(bin_end))
                if ref_idx == int(rid):
                    return span_end
    except Exception:  # pragma: no cover
        return None
    return None


def _resolve_ref_id_from_csi_aux(aux: bytes, contig: bytes) -> int | None:
    if len(aux) < 28:
        return None
    # tabix CSI aux stores: format+col_seq+col_beg+col_end+meta+skip (6x int32) then l_nm (int32) then names.
    l_nm = struct.unpack_from("<i", aux, 24)[0]
    if l_nm <= 0:
        return None
    end = 28 + int(l_nm)
    if end > len(aux):
        return None
    names = aux[28:end]
    ref_names = names.rstrip(b"\0").split(b"\0") if names else []
    try:
        return int(ref_names.index(contig))
    except ValueError:
        return None


def _csi_bin_span_end(bin_id: int, *, min_shift: int, depth: int) -> int | None:
    """Return the exclusive genomic end represented by a regular CSI bin."""
    bin_value = int(bin_id)
    for level in range(int(depth) + 1):
        first = ((1 << (3 * level)) - 1) // 7
        next_first = ((1 << (3 * (level + 1))) - 1) // 7
        if first <= bin_value < next_first:
            shift = int(min_shift) + 3 * (int(depth) - level)
            return int(bin_value - first + 1) << shift
    return None


def _open_tabix_index(index_path: Path):
    raw = index_path.open("rb")
    head = raw.read(2)
    raw.close()
    if head == b"\x1f\x8b":
        return gzip.open(index_path, "rb")
    return index_path.open("rb")


def _read_int32(f) -> int:
    raw = f.read(4)
    if len(raw) != 4:
        raise EOFError("unexpected EOF while reading int32")
    return int(struct.unpack("<i", raw)[0])


def _read_uint32(f) -> int:
    raw = f.read(4)
    if len(raw) != 4:
        raise EOFError("unexpected EOF while reading uint32")
    return int(struct.unpack("<I", raw)[0])
