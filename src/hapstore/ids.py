"""Stable ID parsing and alignment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def read_sample_ids(path: str | Path) -> list[str]:
    """Read sample IDs from one- or two-column text tables."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"sample ID file not found: {p}")

    sample_ids: list[str] = []
    iid_col: int | None = None
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue

            parts = raw.split()
            first = parts[0]

            # Recognize common headers and pin the IID column for the rest of the file.
            if first.upper() in {"IID", "#IID"}:
                iid_col = 0
                continue
            if len(parts) >= 2 and first.upper() in {"FID", "#FID"} and parts[1].upper() == "IID":
                iid_col = 1
                continue
            if raw.startswith("#"):
                continue

            if len(parts) == 1:
                sample_ids.append(first)
                continue

            if iid_col is None:
                iid_col = 1
            if iid_col >= len(parts):
                raise ValueError(f"no IID column {iid_col} found in row for {p}: {raw}")
            sample_ids.append(parts[iid_col])

    if not sample_ids:
        raise ValueError(f"no sample IDs parsed from {p}")
    return sample_ids


def ensure_id_column(df: pd.DataFrame, preferred: Iterable[str] = ("IID", "#IID", "ID")) -> str:
    """Return the first present ID column from a preferred candidate list."""
    for col in preferred:
        if col in df.columns:
            return col
    raise ValueError(f"no ID column found; tried {list(preferred)} in columns {list(df.columns)}")


def align_by_ids(df: pd.DataFrame, id_column: str, ids: Sequence[str], context: str) -> pd.DataFrame:
    """Align dataframe rows to an explicit ID ordering and fail on missing IDs."""
    if id_column not in df.columns:
        raise ValueError(f"{context}: missing ID column {id_column}")

    indexed = df.set_index(id_column)
    missing = [sample_id for sample_id in ids if sample_id not in indexed.index]
    if missing:
        raise ValueError(
            f"{context}: missing {len(missing)} required IDs; first few: {missing[:5]}"
        )

    return indexed.loc[list(ids)].reset_index()
