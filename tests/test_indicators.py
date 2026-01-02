import unittest
import pandas as pd
from backtest import TechnicalIndicators

class TestIndicators(unittest.TestCase):
    def test_sma_ema(self):
        df = pd.DataFrame({'Symbol': ['A']*10, 'Close': range(10)})
        ti = TechnicalIndicators(df)
        ti.simple_moving_average(fast_window=3, slow_window=5)
        ti.exponential_moving_average(window=3)
        out = ti.final_df()
        self.assertIn('SMA_fast', out.columns)
        self.assertIn('SMA_slow', out.columns)
        self.assertIn('EMA', out.columns)
        self.assertTrue(pd.isna(out['SMA_fast'].iloc[1]))
        self.assertTrue(out['EMA'].notna().all())
    def test_rsi_bollinger(self):
        df = pd.DataFrame({'Symbol': ['A']*20, 'Close': range(20)})
        ti = TechnicalIndicators(df)
        ti.rsi(window=5)
        ti.bollinger_bands(window=5, num_std=2)
        out = ti.final_df()
        self.assertIn('RSI', out.columns)
        self.assertIn('BB_upper', out.columns)
        self.assertIn('BB_lower', out.columns)
        self.assertTrue(pd.isna(out['RSI'].iloc[3]))
        self.assertTrue(out['BB_upper'].notna().sum() > 0)

if __name__ == "__main__":
    unittest.main()
