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
from typing import List
from numba import njit

# add a list of methods that calculate certain indicators and have parameters in the add_indicator method to attach them to the main dataset

@njit
def _rolling_mean_numpy(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using NumPy."""
    if window <= 0:
        return arr
    result = np.empty(len(arr), dtype=float)
    result[:window-1] = np.nan
    for i in range(window-1, len(arr)):
        result[i] = np.mean(arr[i-window+1:i+1])
    return result

@njit
def _rolling_std_numpy(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation using NumPy."""
    if window <= 0:
        return arr
    result = np.empty(len(arr), dtype=float)
    result[:window-1] = np.nan
    for i in range(window-1, len(arr)):
        result[i] = np.std(arr[i-window+1:i+1])
    return result

class TechnicalIndicators:
    def __init__(self, data):
        """Create an indicator helper bound to a dataset.

        Parameters
        ----------
        data:
            Any object convertible to a pandas DataFrame. The class stores and
            mutates a DataFrame copy in self.data.
        """
        self.data = data
        pass
    
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
        df = self.data
        col = column if column else df.columns[-1]
        if 'Symbol' in df.columns:
            # Process per symbol using NumPy for speed
            symbols = df['Symbol'].values
            prices = df[col].values
            fast_ma = np.empty(len(df), dtype=float)
            slow_ma = np.empty(len(df), dtype=float)
            for sym in np.unique(symbols):
                mask = symbols == sym
                indices = np.where(mask)[0]
                fast_ma[indices] = _rolling_mean_numpy(prices[mask], fast_window)
                slow_ma[indices] = _rolling_mean_numpy(prices[mask], slow_window)
            df['SMA_fast'] = fast_ma
            df['SMA_slow'] = slow_ma
        else:
            df['SMA_fast'] = _rolling_mean_numpy(df[col].values, fast_window)
            df['SMA_slow'] = _rolling_mean_numpy(df[col].values, slow_window)
        self.data = df

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
        @njit
        def _ema_numpy(arr: np.ndarray, alpha: float) -> np.ndarray:
            """Fast EMA using NumPy."""
            result = np.empty(len(arr), dtype=float)
            result[0] = arr[0]
            for i in range(1, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
            return result
        
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].values
            prices = df[col].values
            ema = np.empty(len(df), dtype=float)
            for sym in np.unique(symbols):
                mask = symbols == sym
                indices = np.where(mask)[0]
                ema[indices] = _ema_numpy(prices[mask], alpha)
            df['EMA'] = ema
        else:
            df['EMA'] = _ema_numpy(df[col].values)
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
        
        @njit
        def _rsi_numpy(arr: np.ndarray, window: int) -> np.ndarray:
            """Fast RSI using NumPy."""
            delta = np.diff(arr, prepend=arr[0])
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            avg_gain = _rolling_mean_numpy(gain, window)
            avg_loss = _rolling_mean_numpy(loss, window)
            rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
            return 100 - (100 / (1 + rs))
        
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].values
            prices = df[col].values
            rsi = np.empty(len(df), dtype=float)
            for sym in np.unique(symbols):
                mask = symbols == sym
                indices = np.where(mask)[0]
                rsi[indices] = _rsi_numpy(prices[mask], window)
            df['RSI'] = rsi
        else:
            df['RSI'] = _rsi_numpy(df[col].values, window)
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
        
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].values
            prices = df[col].values
            upper = np.empty(len(df), dtype=float)
            lower = np.empty(len(df), dtype=float)
            for sym in np.unique(symbols):
                mask = symbols == sym
                indices = np.where(mask)[0]
                sym_prices = prices[mask]
                sma = _rolling_mean_numpy(sym_prices, window)
                std = _rolling_std_numpy(sym_prices, window)
                upper[indices] = sma + num_std * std
                lower[indices] = sma - num_std * std
            df['BB_upper'] = upper
            df['BB_lower'] = lower
        else:
            sma = _rolling_mean_numpy(df[col].values, window)
            std = _rolling_std_numpy(df[col].values, window)
            df['BB_upper'] = sma + num_std * std
            df['BB_lower'] = sma - num_std * std
        self.data = df

    def final_df(self):
        """Return the current DataFrame with all computed indicators."""
        return self.data