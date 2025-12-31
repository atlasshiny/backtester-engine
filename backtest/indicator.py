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
        
        Parameters:
            fast_window (int): Window size for the fast SMA.
            slow_window (int): Window size for the slow SMA.
            column (str | None): Column name to calculate SMA on. Defaults to last column if None.
        """
        df = pd.DataFrame(self.data)
        col = column if column else df.columns[-1]
        df['SMA_fast'] = df[col].rolling(window=fast_window).mean()
        df['SMA_slow'] = df[col].rolling(window=slow_window).mean()
        self.data = df

    def exponential_moving_average(self, window: int = 14, column: str | None = "Close"):
        """
        Calculate and add Exponential Moving Average (EMA) to the dataset.
        
        Parameters:
            window (int): Window size for the EMA.
            column (str | None): Column name to calculate EMA on. Defaults to last column if None.
        """
        df = pd.DataFrame(self.data)
        col = column if column else df.columns[-1]
        df['EMA'] = df[col].ewm(span=window, adjust=False).mean()
        self.data = df

    def rsi(self, window: int = 14, column: str | None = "Close"):
        """
        Calculate and add Relative Strength Index (RSI) to the dataset.
        
        Parameters:
            window (int): Window size for the RSI calculation.
            column (str | None): Column name to calculate RSI on. Defaults to last column if None.
        """
        df = pd.DataFrame(self.data)
        col = column if column else df.columns[-1]
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
        """
        df = pd.DataFrame(self.data)
        col = column if column else df.columns[-1]
        sma = df[col].rolling(window=window).mean()
        std = df[col].rolling(window=window).std()
        df['BB_upper'] = sma + num_std * std
        df['BB_lower'] = sma - num_std * std
        self.data = df

    def final_df(self):
        return self.data