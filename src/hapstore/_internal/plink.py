"""Shared PLINK tool resolution helpers."""

from __future__ import annotations

from pathlib import Path
import shutil


def resolve_plink2_binary(plink2_path: str | Path) -> Path:
    """Resolve a PLINK2 executable path from an explicit value or PATH lookup."""
    text = str(plink2_path)
    if text == "system":
        found = shutil.which("plink2")
        if found is None:
            raise FileNotFoundError("plink2_path='system' but plink2 is not on PATH")
        return Path(found)

    if Path(text).name == text and "/" not in text:
        found = shutil.which(text)
        if found is None:
            raise FileNotFoundError(f"plink2 executable not found on PATH: {text}")
        return Path(found)

    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(f"plink2 binary not found: {path}")
    return path
