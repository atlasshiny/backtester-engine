import numpy as np
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None


def select_array_module(prefer_gpu: object, length: int, gpu_min_size: int):
    """Return xp module (cupy or numpy) based on preference and availability."""
    if prefer_gpu is False:
        return np
    if cp is not None and (prefer_gpu is True or (prefer_gpu == "auto" and length >= gpu_min_size)):
        return cp
    return np


def to_numpy(arr):
    """Convert array (NumPy or CuPy) to NumPy array."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def ensure_array(arr, xp):
    """Ensure `arr` is an array in the array module `xp` (np or cp).

    - If `arr` is already a NumPy or CuPy ndarray, preserve or convert it as needed.
    - For Python lists or scalars, construct an `xp` array.
    """
    if arr is None:
        return None
    # If requested xp is cupy and cupy is available
    if xp is not np and cp is not None and xp is cp:
        # If it's already a CuPy array, return as-is
        if isinstance(arr, cp.ndarray):
            return arr
        # If it's a NumPy array, inspect dtype. CuPy does not support
        # object/string dtypes reliably; for those, keep NumPy arrays.
        if isinstance(arr, np.ndarray):
            # dtype.kind: 'O' object, 'U' unicode, 'S' bytes, numeric kinds include 'i','u','f'
            if arr.dtype.kind in ('O', 'U', 'S') or not np.issubdtype(arr.dtype, np.number):
                return arr
            try:
                return cp.asarray(arr)
            except Exception:
                return arr
        # For lists/scalars: convert to NumPy first, then decide
        arr_np = np.asarray(arr)
        if arr_np.dtype.kind in ('O', 'U', 'S') or not np.issubdtype(arr_np.dtype, np.number):
            return arr_np
        try:
            return cp.asarray(arr_np)
        except Exception:
            return arr_np
    # xp is numpy or cupy not available
    if isinstance(arr, np.ndarray):
        return arr
    if cp is not None and isinstance(arr, cp.ndarray):
        # convert cupy -> numpy only when numpy requested
        return cp.asnumpy(arr)
    return np.asarray(arr)


def df_to_arrays(df, xp=None):
    """Convert a pandas DataFrame to a dict of arrays.

    If `xp` is provided (np or cp), data will be converted to that array module.
    Otherwise returns NumPy arrays (shallow views where possible).
    """
    # Local import to avoid pandas requirement at module import time for some tooling
    import pandas as _pd
    if not isinstance(df, _pd.DataFrame):
        raise TypeError("df_to_arrays requires a pandas DataFrame")
    xp = xp or np
    out = {}
    for col in df.columns:
        arr = df[col].to_numpy(copy=False)
        out[col] = ensure_array(arr, xp)
    return out
