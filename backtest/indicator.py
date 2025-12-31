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
from typing import List

# add a list of methods that calculate certain indicators and have parameters in the add_indicator method to attach them to the main dataset

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
        df = pd.DataFrame(self.data)
        col = column if column else df.columns[-1]
        if 'Symbol' in df.columns:
            df['SMA_fast'] = df.groupby('Symbol')[col].transform(lambda x: x.rolling(window=fast_window).mean())
            df['SMA_slow'] = df.groupby('Symbol')[col].transform(lambda x: x.rolling(window=slow_window).mean())
        else:
            df['SMA_fast'] = df[col].rolling(window=fast_window).mean()
            df['SMA_slow'] = df[col].rolling(window=slow_window).mean()
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
        df = pd.DataFrame(self.data)
        col = column if column else df.columns[-1]
        if 'Symbol' in df.columns:
            df['EMA'] = df.groupby('Symbol')[col].transform(lambda x: x.ewm(span=window, adjust=False).mean())
        else:
            df['EMA'] = df[col].ewm(span=window, adjust=False).mean()
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
        df = pd.DataFrame(self.data)
        col = column if column else df.columns[-1]
        if 'Symbol' in df.columns:
            def rsi_calc(x):
                delta = x.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            df['RSI'] = df.groupby('Symbol')[col].transform(rsi_calc)
        else:
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
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
        df = pd.DataFrame(self.data)
        col = column if column else df.columns[-1]
        if 'Symbol' in df.columns:
            def bb_calc(x):
                sma = x.rolling(window=window).mean()
                std = x.rolling(window=window).std()
                upper = sma + num_std * std
                lower = sma - num_std * std
                return pd.DataFrame({'BB_upper': upper, 'BB_lower': lower})
            bb = df.groupby('Symbol')[col].apply(lambda x: bb_calc(x)).reset_index(level=0, drop=True)
            df['BB_upper'] = bb['BB_upper']
            df['BB_lower'] = bb['BB_lower']
        else:
            sma = df[col].rolling(window=window).mean()
            std = df[col].rolling(window=window).std()
            df['BB_upper'] = sma + num_std * std
            df['BB_lower'] = sma - num_std * std
        self.data = df

    def final_df(self):
        """Return the current DataFrame with all computed indicators."""
        return self.data