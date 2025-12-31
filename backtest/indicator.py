import pandas as pd
from typing import List

# add a list of methods that calculate certain indicators and have parameters in the add_indicator method to attach them to the main dataset

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data
        pass
    
    def simple_moving_average(self, fast_window: int = 7, slow_window: int = 25, column: str | None = "Close"):
        """
        Calculate and add fast and slow Simple Moving Averages (SMA) to the dataset.
        Supports both single-asset and long-format (multi-asset) DataFrames.
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
        Supports both single-asset and long-format (multi-asset) DataFrames.
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
        Supports both single-asset and long-format (multi-asset) DataFrames.
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
        Supports both single-asset and long-format (multi-asset) DataFrames.
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
        return self.data