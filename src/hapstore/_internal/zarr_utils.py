"""Shared zarr helper functions used across hapstore modules."""

from __future__ import annotations

import numpy as np
import zarr
from numcodecs import VLenUTF8, Zstd


def _to_unicode_array(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=object)
    for i, value in enumerate(values):
        if isinstance(value, bytes):
            out[i] = value.decode("utf-8")
        else:
            out[i] = str(value)
    return out


def _create_utf8_array(
    group: zarr.Group,
    name: str,
    values: np.ndarray,
    chunk_size: int,
    compressor: Zstd,
) -> None:
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=object,
        object_codec=VLenUTF8(),
        chunks=(int(max(1, chunk_size)),),
        compressor=compressor,
        overwrite=False,
    )
