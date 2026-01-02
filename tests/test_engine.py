import unittest
import pandas as pd
from backtest import BacktestEngine, Portfolio, Broker, Order, Strategy

class DummyStrategy(Strategy):
    def __init__(self):
        super().__init__(history_window=None)
        self.toggle = {}
    def check_condition(self, event, history=None):
        symbol = getattr(event, 'Symbol', 'SINGLE')
        if self.toggle.get(symbol, 0) % 2 == 0:
            self.toggle[symbol] = self.toggle.get(symbol, 0) + 1
            return Order(symbol=symbol, side="BUY", qty=1)
        else:
            self.toggle[symbol] = self.toggle.get(symbol, 0) + 1
            return Order(symbol=symbol, side="SELL", qty=1)

class TestEngine(unittest.TestCase):
    def make_single_asset_df(self):
        return pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=4),
            'Open': [10, 11, 12, 13],
            'High': [11, 12, 13, 14],
            'Low': [9, 10, 11, 12],
            'Close': [10.5, 11.5, 12.5, 13.5],
            'Volume': [100, 110, 120, 130],
        })
    def make_multi_asset_df(self):
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=4).repeat(2),
            'Symbol': ['A', 'B'] * 4,
            'Open': [10, 20, 11, 21, 12, 22, 13, 23],
            'High': [11, 21, 12, 22, 13, 23, 14, 24],
            'Low': [9, 19, 10, 20, 11, 21, 12, 22],
            'Close': [10.5, 20.5, 11.5, 21.5, 12.5, 22.5, 13.5, 23.5],
            'Volume': [100, 200, 110, 210, 120, 220, 130, 230],
        })
        return df.sort_values(['Date', 'Symbol']).reset_index(drop=True)
    def test_single_asset_row_by_row(self):
        df = self.make_single_asset_df()
        strat = DummyStrategy()
        port = Portfolio(100)
        broker = Broker(port)
        engine = BacktestEngine(strat, port, broker, df, warm_up=0, group_by_date=False)
        engine.run()
        self.assertEqual(sum(p.qty for p in port.positions.values()), 1)
        self.assertEqual(len(broker.trade_log), 3)
    def test_multi_asset_row_by_row(self):
        df = self.make_multi_asset_df()
        strat = DummyStrategy()
        port = Portfolio(200)
        broker = Broker(port)
        engine = BacktestEngine(strat, port, broker, df, warm_up=0, group_by_date=False)
        engine.run()
        self.assertEqual(set(p.qty for p in port.positions.values()), {1} if port.positions else set())
        self.assertEqual(len(broker.trade_log), 6)
    def test_multi_asset_group_by_date(self):
        df = self.make_multi_asset_df()
        strat = DummyStrategy()
        port = Portfolio(200)
        broker = Broker(port)
        engine = BacktestEngine(strat, port, broker, df, warm_up=0, group_by_date=True)
        engine.run()
        self.assertEqual(set(p.qty for p in port.positions.values()), {1} if port.positions else set())
        self.assertEqual(len(broker.trade_log), 6)
    def test_symbol_injection_for_single_asset(self):
        df = self.make_single_asset_df()
        df = df.drop(columns=['Date'])
        strat = DummyStrategy()
        port = Portfolio(100)
        broker = Broker(port)
        engine = BacktestEngine(strat, port, broker, df, warm_up=0, group_by_date=False)
        engine.run()
        self.assertTrue(all(t['symbol'] == 'SINGLE' for t in broker.trade_log if t['qty'] > 0))

if __name__ == "__main__":
    unittest.main()
