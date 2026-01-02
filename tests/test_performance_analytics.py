import unittest
import numpy as np
import pandas as pd
from backtest.performance_analytics import PerformanceAnalytics
from backtest.portfolio import Portfolio

class TestPerformanceAnalytics(unittest.TestCase):
    def setUp(self):
        # Create a synthetic equity curve with known stats
        self.portfolio = Portfolio(1000)
        # Simulate a simple equity curve: +100, -50, +200, -100
        self.portfolio.value_history = [1000, 1100, 1050, 1250, 1150]
        # Simulate a trade log
        self.trade_log = [
            {'side': 'BUY', 'qty': 1, 'symbol': 'A', 'price': 100, 'commission': 1, 'slippage': 0.5, 'timestamp': 1, 'order_type': 'MARKET', 'limit_price': None, 'comment': ''},
            {'side': 'SELL', 'qty': 1, 'symbol': 'A', 'price': 150, 'commission': 1, 'slippage': 0.5, 'timestamp': 2, 'order_type': 'MARKET', 'limit_price': None, 'comment': ''},
            {'side': 'BUY', 'qty': 1, 'symbol': 'A', 'price': 200, 'commission': 1, 'slippage': 0.5, 'timestamp': 3, 'order_type': 'MARKET', 'limit_price': None, 'comment': ''},
            {'side': 'SELL', 'qty': 1, 'symbol': 'A', 'price': 250, 'commission': 1, 'slippage': 0.5, 'timestamp': 4, 'order_type': 'MARKET', 'limit_price': None, 'comment': ''},
        ]
        self.data_set = pd.DataFrame({'Date': [1,2,3,4,5], 'Symbol': ['A']*5, 'Close': [100,150,200,250,300]})

    def test_basic_statistics(self):
        analytics = PerformanceAnalytics()
        stats = analytics.calculate_statistics(self.portfolio, self.data_set, trade_log=self.trade_log)
        # Check total PnL
        self.assertAlmostEqual(stats['TotalPnL'], 150 + 50, delta=1e-6)  # (150-100) + (250-200)
        # Check win rate
        self.assertAlmostEqual(stats['WinRate'], 1.0)  # Both trades are profitable
        # Check max drawdown
        self.assertAlmostEqual(stats['MaxDrawdown'], 100/1250, delta=1e-2)  # (1250-1150)/1250
        # Check profit factor
        self.assertGreater(stats['ProfitFactor'], 1.0)
        # Check expectancy
        self.assertTrue(isinstance(stats['Expectancy'], float))
        # Check Sharpe ratio
        self.assertTrue(isinstance(stats['SharpeRatio'], float))
        # Check Sortino ratio
        self.assertTrue(isinstance(stats['SortinoRatio'], float))

    def test_trade_log_aggregation(self):
        analytics = PerformanceAnalytics()
        stats = analytics.calculate_statistics(self.portfolio, self.data_set, trade_log=self.trade_log)
        # Check total commission
        self.assertEqual(stats['TotalCommission'], 4)
        # Check total trades
        self.assertEqual(stats['TotalTrades'], 4)
        # Check average win/loss
        self.assertTrue(stats['AvgWin'] > 0)
        self.assertTrue(stats['AvgLoss'] <= 0 or np.isnan(stats['AvgLoss']))

if __name__ == '__main__':
    unittest.main()
