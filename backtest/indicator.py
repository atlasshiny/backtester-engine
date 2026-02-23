"""backtest.indicator

Technical indicator computation utilities.

These helpers are designed to precompute indicators on the full dataset *before*
running the backtest so strategies can read indicator values directly from the
event.

Supported input formats
-----------------------
- Multi-asset long format: requires a Symbol column. Each indicator is computed
    per symbol via groupby/transform.
- Single-asset: computed on the full series.

Important
---------
For time-series correctness, the data should be sorted in chronological order
within each symbol (typically by Date).
"""

import pandas as pd
import numpy as np
from typing import List, Literal
from numba import njit

try:  # Optional GPU acceleration
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:  # noqa: BLE001 - import-time probe only
    cp = None
    _CUPY_AVAILABLE = False

# add a list of methods that calculate certain indicators and have parameters in the add_indicator method to attach them to the main dataset

@njit
def _rolling_mean_numpy(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using NumPy."""
    # Ensure we always return a float array to keep numba return types consistent
    result = np.empty(len(arr), dtype=np.float64)
    if window <= 0:
        for i in range(len(arr)):
            result[i] = arr[i]
        return result

    result[:] = np.nan
    # preserve same warmup behavior: first window-1 values are NaN
    result[:window-1] = np.nan
    for i in range(window-1, len(arr)):
        result[i] = np.mean(arr[i-window+1:i+1])
    return result

@njit
def _rolling_std_numpy(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation using NumPy."""
    # Ensure we always return a float array to keep numba return types consistent
    result = np.empty(len(arr), dtype=np.float64)
    if window <= 0:
        for i in range(len(arr)):
            result[i] = arr[i]
        return result

    result[:] = np.nan
    result[:window-1] = np.nan
    for i in range(window-1, len(arr)):
        result[i] = np.std(arr[i-window+1:i+1])
    return result


def _select_array_module(prefer_gpu: Literal["auto", True, False], length: int, gpu_min_size: int) -> object:
    """Return cupy or numpy array module based on preference/availability/size."""
    if prefer_gpu is False:
        return np
    if _CUPY_AVAILABLE and (prefer_gpu is True or (prefer_gpu == "auto" and length >= gpu_min_size)):
        return cp
    return np


def _rolling_mean_xp(arr, window: int, xp):
    """Vectorized rolling mean for NumPy/CuPy with NaN warmup."""
    if window <= 0:
        return xp.asarray(arr, dtype=xp.float64)
    kernel = xp.ones(window, dtype=xp.float64) / float(window)
    valid = xp.convolve(arr, kernel, mode="valid")
    result = xp.empty(arr.shape, dtype=xp.float64)
    result[:] = xp.nan
    result[window-1:] = valid
    return result


def _rolling_std_xp(arr, window: int, xp):
    """Vectorized rolling std for NumPy/CuPy with NaN warmup."""
    if window <= 0:
        return xp.asarray(arr, dtype=xp.float64)
    kernel = xp.ones(window, dtype=xp.float64) / float(window)
    mean = xp.convolve(arr, kernel, mode="valid")
    mean_sq = xp.convolve(arr * arr, kernel, mode="valid")
    var = xp.maximum(mean_sq - mean * mean, 0.0)
    std_valid = xp.sqrt(var)
    result = xp.empty(arr.shape, dtype=xp.float64)
    result[:] = xp.nan
    result[window-1:] = std_valid
    return result


def _rsi_xp(arr, window: int, xp, rolling_mean_fn):
    """RSI for NumPy/CuPy using provided rolling mean implementation."""
    if len(arr) == 0:
        return xp.asarray(arr, dtype=xp.float64)
    delta = xp.diff(arr, prepend=arr[:1])
    gain = xp.where(delta > 0, delta, 0.0)
    loss = xp.where(delta < 0, -delta, 0.0)
    avg_gain = rolling_mean_fn(gain, window)
    avg_loss = rolling_mean_fn(loss, window)
    rs = avg_gain / xp.where(avg_loss == 0, xp.nan, avg_loss)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result

class TechnicalIndicators:
    def __init__(self, data, prefer_gpu: Literal["auto", True, False] = "auto", gpu_min_size: int = 10000):
        """Create an indicator helper bound to a dataset.

        Parameters
        ----------
        data:
            Any object convertible to a pandas DataFrame. The class stores and
            mutates a DataFrame copy in self.data.
        prefer_gpu:
            "auto" (default) will use CuPy when available and the dataset is larger than gpu_min_size.
            True forces CuPy (if installed), False forces CPU/NumPy.
        gpu_min_size:
            Minimum number of rows required before attempting GPU acceleration in auto mode.
        """
        # Accept either a pandas DataFrame or a dict of arrays (from df_to_arrays)
        self.data = None
        self.arrays = None
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, dict):
            # assume dict[col] -> ndarray (NumPy)
            self.arrays = data
        else:
            # try to coerce to DataFrame
            try:
                self.data = pd.DataFrame(data)
            except Exception:
                raise TypeError("TechnicalIndicators expects a pandas DataFrame or dict of arrays")
        self.prefer_gpu = prefer_gpu
        self.gpu_min_size = gpu_min_size
        # Track which indicator methods have been executed on this instance
        self._called_methods: set[str] = set()

    @staticmethod
    def _to_numpy(arr):
        if cp is not None and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def _xp(self, length: int):
        return _select_array_module(self.prefer_gpu, length, self.gpu_min_size)
    
    def simple_moving_average(self, fast_window: int = 7, slow_window: int = 25, column: str | None = "Close"):
        """
        Calculate and add fast and slow Simple Moving Averages (SMA) to the dataset.
        
        Parameters:
            fast_window (int): Window size for the fast SMA.
            slow_window (int): Window size for the slow SMA.
            column (str | None): Column name to calculate SMA on. Defaults to last column if None.

        Output
        ------
        Adds columns:
        - SMA_fast
        - SMA_slow

        Notes
        -----
        The first (window-1) rows per symbol will be NaN due to rolling warmup.
        """
        col = column if column else (list(self.arrays.keys())[-1] if self.arrays is not None else self.data.columns[-1])
        # prefer length from arrays or df
        length = len(self.arrays[next(iter(self.arrays))]) if self.arrays is not None else len(self.data)
        xp = self._xp(length)
        use_gpu = xp is not np
        rolling_mean = (lambda arr, window: _rolling_mean_xp(arr, window, xp)) if use_gpu else _rolling_mean_numpy

        # record that this precompute ran
        self._called_methods.add('simple_moving_average')
        if self.arrays is not None:
            arrays = self.arrays
            symbols = arrays.get('Symbol', None)
            prices = arrays[col].astype(float)
            fast_ma = np.empty(len(prices), dtype=float)
            slow_ma = np.empty(len(prices), dtype=float)
            if symbols is not None:
                for sym in np.unique(symbols):
                    mask = symbols == sym
                    indices = np.nonzero(mask)[0]
                    sym_prices = prices[mask]
                    sym_prices_xp = xp.asarray(sym_prices, dtype=xp.float64) if use_gpu else sym_prices
                    fast_vals = rolling_mean(sym_prices_xp, fast_window)
                    slow_vals = rolling_mean(sym_prices_xp, slow_window)
                    fast_ma[indices] = self._to_numpy(fast_vals)
                    slow_ma[indices] = self._to_numpy(slow_vals)
            else:
                prices_xp = xp.asarray(prices, dtype=xp.float64) if use_gpu else prices
                fast_ma = self._to_numpy(rolling_mean(prices_xp, fast_window))
                slow_ma = self._to_numpy(rolling_mean(prices_xp, slow_window))
            arrays['SMA_fast'] = fast_ma
            arrays['SMA_slow'] = slow_ma
            # keep arrays updated; materialization to DataFrame happens in final_df()
        else:
            df = self.data
            col = column if column else df.columns[-1]
            xp = self._xp(len(df))
            use_gpu = xp is not np
            rolling_mean = (lambda arr, window: _rolling_mean_xp(arr, window, xp)) if use_gpu else _rolling_mean_numpy

            if 'Symbol' in df.columns:
                symbols = df['Symbol'].values
                prices = df[col].values.astype(float)
                fast_ma = np.empty(len(df), dtype=float)
                slow_ma = np.empty(len(df), dtype=float)
                for sym in np.unique(symbols):
                    mask = symbols == sym
                    indices = np.where(mask)[0]
                    sym_prices = prices[mask]
                    sym_prices_xp = xp.asarray(sym_prices, dtype=xp.float64) if use_gpu else sym_prices
                    fast_vals = rolling_mean(sym_prices_xp, fast_window)
                    slow_vals = rolling_mean(sym_prices_xp, slow_window)
                    fast_ma[indices] = self._to_numpy(fast_vals)
                    slow_ma[indices] = self._to_numpy(slow_vals)
                df['SMA_fast'] = fast_ma
                df['SMA_slow'] = slow_ma
            else:
                prices = xp.asarray(df[col].values, dtype=xp.float64) if use_gpu else df[col].values
                df['SMA_fast'] = self._to_numpy(rolling_mean(prices, fast_window))
                df['SMA_slow'] = self._to_numpy(rolling_mean(prices, slow_window))
            self.data = df

    def get_called_methods(self) -> set:
        """Return the set of indicator method names that have been executed on this instance."""
        return set(self._called_methods)

    def exponential_moving_average(self, window: int = 14, column: str | None = "Close"):
        """
        Calculate and add Exponential Moving Average (EMA) to the dataset.
        
        Parameters:
            window (int): Window size for the EMA.
            column (str | None): Column name to calculate EMA on. Defaults to last column if None.

        Output
        ------
        Adds column:
        - EMA
        """
        df = self.data
        col = column if column else df.columns[-1]
        alpha = 2.0 / (window + 1.0)
        xp = self._xp(len(df))
        use_gpu = xp is not np

        @njit
        def _ema_numpy(arr: np.ndarray, alpha_local: float) -> np.ndarray:
            if len(arr) == 0:
                return np.empty(0, dtype=float)
            result = np.empty(len(arr), dtype=float)
            result[0] = arr[0]
            for i in range(1, len(arr)):
                result[i] = alpha_local * arr[i] + (1 - alpha_local) * result[i-1]
            return result

        def _ema_xp(arr_xp):
            if len(arr_xp) == 0:
                return xp.asarray(arr_xp, dtype=xp.float64)
            result = xp.empty(len(arr_xp), dtype=xp.float64)
            result[0] = arr_xp[0]
            for i in range(1, len(arr_xp)):
                result[i] = alpha * arr_xp[i] + (1 - alpha) * result[i-1]
            return result

        # record execution
        self._called_methods.add('exponential_moving_average')
        ema_out = np.empty(len(df), dtype=float)
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].values
            prices = df[col].values.astype(float)
            for sym in np.unique(symbols):
                mask = symbols == sym
                indices = np.where(mask)[0]
                sym_prices = prices[mask]
                if use_gpu:
                    ema_vals = _ema_xp(xp.asarray(sym_prices, dtype=xp.float64))
                else:
                    ema_vals = _ema_numpy(sym_prices, alpha)
                ema_out[indices] = self._to_numpy(ema_vals)
        else:
            prices = df[col].values.astype(float)
            if use_gpu:
                ema_vals = _ema_xp(xp.asarray(prices, dtype=xp.float64))
            else:
                ema_vals = _ema_numpy(prices, alpha)
            ema_out = self._to_numpy(ema_vals)
        df['EMA'] = ema_out
        self.data = df

    def rsi(self, window: int = 14, column: str | None = "Close"):
        """
        Calculate and add Relative Strength Index (RSI) to the dataset.
        
        Parameters:
            window (int): Window size for the RSI calculation.
            column (str | None): Column name to calculate RSI on. Defaults to last column if None.

        Output
        ------
        Adds column:
        - RSI
        """
        df = self.data
        col = column if column else df.columns[-1]
        xp = self._xp(len(df))
        use_gpu = xp is not np
        rolling_mean = (lambda arr, win: _rolling_mean_xp(arr, win, xp)) if use_gpu else _rolling_mean_numpy

        # record execution
        self._called_methods.add('rsi')
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].values
            prices = df[col].values.astype(float)
            rsi_out = np.empty(len(df), dtype=float)
            for sym in np.unique(symbols):
                mask = symbols == sym
                indices = np.where(mask)[0]
                sym_prices = prices[mask]
                sym_prices_xp = xp.asarray(sym_prices, dtype=xp.float64) if use_gpu else sym_prices
                rsi_vals = _rsi_xp(sym_prices_xp, window, xp, rolling_mean)
                rsi_out[indices] = self._to_numpy(rsi_vals)
            df['RSI'] = rsi_out
        else:
            prices = xp.asarray(df[col].values, dtype=xp.float64) if use_gpu else df[col].values
            df['RSI'] = self._to_numpy(_rsi_xp(prices, window, xp, rolling_mean))
        self.data = df

    def bollinger_bands(self, window: int = 20, num_std: float = 2.0, column: str | None = "Close"):
        """
        Calculate and add Bollinger Bands (upper and lower) to the dataset.
        
        Parameters:
            window (int): Window size for the moving average and standard deviation.
            num_std (float): Number of standard deviations for the bands.
            column (str | None): Column name to calculate bands on. Defaults to last column if None.

        Output
        ------
        Adds columns:
        - BB_upper
        - BB_lower
        """
        df = self.data
        col = column if column else df.columns[-1]
        xp = self._xp(len(df))
        use_gpu = xp is not np
        rolling_mean = (lambda arr, win: _rolling_mean_xp(arr, win, xp)) if use_gpu else _rolling_mean_numpy
        rolling_std = (lambda arr, win: _rolling_std_xp(arr, win, xp)) if use_gpu else _rolling_std_numpy

        # record execution
        self._called_methods.add('bollinger_bands')
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].values
            prices = df[col].values.astype(float)
            upper = np.empty(len(df), dtype=float)
            lower = np.empty(len(df), dtype=float)
            for sym in np.unique(symbols):
                mask = symbols == sym
                indices = np.where(mask)[0]
                sym_prices = prices[mask]
                sym_prices_xp = xp.asarray(sym_prices, dtype=xp.float64) if use_gpu else sym_prices
                sma = rolling_mean(sym_prices_xp, window)
                std = rolling_std(sym_prices_xp, window)
                sma_np = self._to_numpy(sma)
                std_np = self._to_numpy(std)
                upper[indices] = sma_np + num_std * std_np
                lower[indices] = sma_np - num_std * std_np
            df['BB_upper'] = upper
            df['BB_lower'] = lower
        else:
            prices = xp.asarray(df[col].values, dtype=xp.float64) if use_gpu else df[col].values
            sma = rolling_mean(prices, window)
            std = rolling_std(prices, window)
            sma_np = self._to_numpy(sma)
            std_np = self._to_numpy(std)
            df['BB_upper'] = sma_np + num_std * std_np
            df['BB_lower'] = sma_np - num_std * std_np
        self.data = df

    def final_df(self):
        """Return the current DataFrame with all computed indicators.

        If the object was constructed from arrays, materialize a DataFrame from
        the arrays before returning (this keeps backward compatibility)."""
        if self.data is not None:
            return self.data
        if self.arrays is not None:
            return pd.DataFrame(self.arrays)
        return pd.DataFrame()