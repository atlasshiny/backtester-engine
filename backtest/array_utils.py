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
        if isinstance(arr, cp.ndarray):
            return arr
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        # fallback for lists/scalars
        return cp.asarray(arr)
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
