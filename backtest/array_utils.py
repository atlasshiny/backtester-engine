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


def df_to_arrays(df):
    """Convert a pandas DataFrame to a dict of NumPy arrays (shallow views where possible)."""
    # Local import to avoid pandas requirement at module import time for some tooling
    import pandas as _pd
    if not isinstance(df, _pd.DataFrame):
        raise TypeError("df_to_arrays requires a pandas DataFrame")
    return {col: df[col].to_numpy(copy=False) for col in df.columns}
