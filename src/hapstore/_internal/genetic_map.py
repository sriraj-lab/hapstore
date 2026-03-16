"""Genetic map loading and interpolation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .chrom import normalize_chrom


def _read_first_data_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw and not raw.startswith("#"):
                return raw
    raise ValueError(f"genetic map file has no data lines: {path}")


def _split_tokens(line: str) -> list[str]:
    return [tok for tok in re.split(r"[\s,]+", line.strip()) if tok]


def _is_numeric_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _looks_like_header(tokens: list[str]) -> bool:
    if not tokens:
        return False

    numeric_count = sum(1 for token in tokens if _is_numeric_token(token))

    if len(tokens) >= 3 and numeric_count >= 2:
        return False
    if len(tokens) == 2 and numeric_count == 2:
        return False

    if re.match(r"^(chr)?[0-9XYM]+$", tokens[0], flags=re.IGNORECASE) and numeric_count >= 1:
        return False

    return True


def _choose_columns(df: pd.DataFrame, has_header: bool) -> tuple[str, str, str]:
    if has_header:
        lower_map = {str(col).lower(): col for col in df.columns}

        chrom_col = None
        pos_col = None
        cm_col = None

        for low, col in lower_map.items():
            if chrom_col is None and "chr" in low:
                chrom_col = col
            if pos_col is None and ("pos" in low or "bp" in low):
                pos_col = col
            if cm_col is None and ("cm" in low or "genetic" in low or "map" in low):
                cm_col = col

        if chrom_col is None and len(df.columns) >= 3:
            chrom_col = df.columns[0]
        if pos_col is None and len(df.columns) >= 3:
            pos_col = df.columns[1]
        if cm_col is None and len(df.columns) >= 3:
            cm_col = df.columns[2]

        if chrom_col is None or pos_col is None or cm_col is None:
            raise ValueError("could not infer required columns from genetic map header")

        return str(chrom_col), str(pos_col), str(cm_col)

    if df.shape[1] >= 4:
        # PLINK format: chrom, id, cm, bp
        return "0", "3", "2"
    if df.shape[1] >= 3:
        return "0", "1", "2"
    if df.shape[1] == 2:
        # No chromosome column provided.
        return "_chrom", "0", "1"
    raise ValueError("genetic map must have at least two columns")


def load_genetic_map(path: str | Path, chrom: str | int | None = None) -> pd.DataFrame:
    """Load a genetic map into standardized columns.

    Returns dataframe with columns: `chrom`, `position_bp`, `position_cm`.
    """
    map_path = Path(path)
    first_line = _read_first_data_line(map_path)
    tokens = _split_tokens(first_line)
    has_header = _looks_like_header(tokens)

    if has_header:
        df = pd.read_csv(map_path, sep=r"\s+|,", engine="python", comment="#")
    else:
        df = pd.read_csv(
            map_path,
            sep=r"\s+|,",
            engine="python",
            comment="#",
            header=None,
        )

    chrom_col, pos_col, cm_col = _choose_columns(df, has_header=has_header)

    if chrom_col == "_chrom":
        normalized = pd.DataFrame(
            {
                "chrom": "",
                "position_bp": pd.to_numeric(df.iloc[:, int(pos_col)], errors="coerce"),
                "position_cm": pd.to_numeric(df.iloc[:, int(cm_col)], errors="coerce"),
            }
        )
    elif has_header:
        normalized = pd.DataFrame(
            {
                "chrom": df[chrom_col].astype(str),
                "position_bp": pd.to_numeric(df[pos_col], errors="coerce"),
                "position_cm": pd.to_numeric(df[cm_col], errors="coerce"),
            }
        )
    else:
        normalized = pd.DataFrame(
            {
                "chrom": df.iloc[:, int(chrom_col)].astype(str),
                "position_bp": pd.to_numeric(df.iloc[:, int(pos_col)], errors="coerce"),
                "position_cm": pd.to_numeric(df.iloc[:, int(cm_col)], errors="coerce"),
            }
        )

    normalized = normalized.dropna(subset=["position_bp", "position_cm"]).copy()
    normalized["position_bp"] = normalized["position_bp"].astype(np.int64)
    normalized["position_cm"] = normalized["position_cm"].astype(float)

    if chrom is not None:
        chrom_plain = normalize_chrom(chrom, prefix=False)
        normalized = normalized[
            normalized["chrom"].astype(str).str.replace("chr", "", regex=False) == chrom_plain
        ].copy()

    if normalized.empty:
        if chrom is None:
            raise ValueError(f"no valid rows parsed from genetic map: {map_path}")
        raise ValueError(f"no valid rows parsed for chromosome {chrom} from genetic map: {map_path}")

    normalized = normalized.sort_values(["chrom", "position_bp"]).reset_index(drop=True)
    return normalized


def interpolate_cm(
    positions_bp: np.ndarray,
    map_df: pd.DataFrame,
    chrom: str | int | None = None,
) -> np.ndarray:
    """Interpolate genetic positions (cM) from physical positions (bp)."""
    if chrom is not None:
        chrom_plain = normalize_chrom(chrom, prefix=False)
        subset = map_df[
            map_df["chrom"].astype(str).str.replace("chr", "", regex=False) == chrom_plain
        ]
    else:
        subset = map_df

    if subset.empty:
        raise ValueError("no genetic map rows available for interpolation")

    bp = subset["position_bp"].to_numpy(dtype=float)
    cm = subset["position_cm"].to_numpy(dtype=float)
    return np.interp(np.asarray(positions_bp, dtype=float), bp, cm)
