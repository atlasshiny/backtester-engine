import unittest
import pandas as pd
from backtest import (
    BacktestEngine, Portfolio, Broker, Order, Position, TechnicalIndicators, Strategy
)
from strategies.simple_moving_average import SimpleMovingAverage
from strategies.buy_n_hold import BuyNHold

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

class TestPosition(unittest.TestCase):
    def test_add_and_remove(self):
        pos = Position('A', 10, 5)
        pos.add(5, 15)
        self.assertEqual(pos.qty, 15)
        self.assertEqual(pos.avg_price, (10*5+5*15)/15)
        pos.remove(5)
        self.assertEqual(pos.qty, 10)
        with self.assertRaises(AssertionError):
            pos.remove(20)
    def test_market_value(self):
        pos = Position('A', 3, 7)
        self.assertEqual(pos.market_value(10), 30)

class TestOrder(unittest.TestCase):
    def test_order_fields(self):
        o = Order(symbol='A', side='BUY', qty=2, order_type='LIMIT', limit_price=9.5, timestamp=123)
        self.assertEqual(o.symbol, 'A')
        self.assertEqual(o.side, 'BUY')
        self.assertEqual(o.qty, 2)
        self.assertEqual(o.order_type, 'LIMIT')
        self.assertEqual(o.limit_price, 9.5)
        self.assertEqual(o.timestamp, 123)

class TestPortfolio(unittest.TestCase):
    def test_add_remove_and_value(self):
        p = Portfolio(100)
        p.add_position('A', 10, 5)
        p.add_position('A', 10, 15)
        self.assertAlmostEqual(p.positions['A'].avg_price, 10)
        p.remove_position('A', 5)
        self.assertEqual(p.positions['A'].qty, 15)
        p.remove_position('A', 15)
        self.assertNotIn('A', p.positions)
        p.add_position('B', 2, 20)
        p.update_value_history({'B': 25})
        self.assertAlmostEqual(p.value_history[-1], 100 + 2*25)
        p.update_value_history(30)
        self.assertAlmostEqual(p.value_history[-1], 100 + 2*30)
    def test_snapshot(self):
        p = Portfolio(50)
        p.add_position('A', 2, 10)
        self.assertEqual(p.portfolio_value_snapshot(12), 50 + 2*12)

class TestBroker(unittest.TestCase):
    def test_market_and_limit_orders(self):
        p = Portfolio(100)
        b = Broker(p, slippage=0, commission=0)
        class Event: Open=10; Low=9; High=11; Close=10.5; Date='2020-01-01'; Symbol='A'
        # Market order
        order = Order(symbol='A', side='BUY', qty=5)
        b.execute(Event, order)
        self.assertEqual(p.positions['A'].qty, 5)
        self.assertEqual(b.trade_log[-1]['side'], 'BUY')
        # Limit order fill
        order2 = Order(symbol='A', side='BUY', qty=2, order_type='LIMIT', limit_price=9.5)
        b.execute(Event, order2)
        self.assertEqual(b.trade_log[-1]['qty'], 2)
        # Limit order unfilled
        order3 = Order(symbol='A', side='BUY', qty=2, order_type='LIMIT', limit_price=8.5)
        b.execute(Event, order3)
        self.assertEqual(b.trade_log[-1]['qty'], 0)
    def test_sell_and_insufficient(self):
        p = Portfolio(100)
        b = Broker(p, slippage=0, commission=0)
        class Event: Open=10; Low=9; High=11; Close=10.5; Date='2020-01-01'; Symbol='A'
        order = Order(symbol='A', side='BUY', qty=2)
        b.execute(Event, order)
        order2 = Order(symbol='A', side='SELL', qty=2)
        b.execute(Event, order2)
        self.assertNotIn('A', p.positions)
        # Try to sell more than owned
        order3 = Order(symbol='A', side='SELL', qty=5)
        b.execute(Event, order3)
        self.assertEqual(b.trade_log[-1]['qty'], 0)

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
        # Next-bar execution: last BUY is not offset, expect qty=1
        self.assertEqual(sum(p.qty for p in port.positions.values()), 1)
        self.assertEqual(len(broker.trade_log), 3)
    def test_multi_asset_row_by_row(self):
        df = self.make_multi_asset_df()
        strat = DummyStrategy()
        port = Portfolio(200)
        broker = Broker(port)
        engine = BacktestEngine(strat, port, broker, df, warm_up=0, group_by_date=False)
        engine.run()
        # Next-bar execution: last BUY for each symbol is not offset, expect qty=1 for each
        self.assertEqual(set(p.qty for p in port.positions.values()), {1} if port.positions else set())
        self.assertEqual(len(broker.trade_log), 6)
    def test_multi_asset_group_by_date(self):
        df = self.make_multi_asset_df()
        strat = DummyStrategy()
        port = Portfolio(200)
        broker = Broker(port)
        engine = BacktestEngine(strat, port, broker, df, warm_up=0, group_by_date=True)
        engine.run()
        # Next-bar execution: last BUY for each symbol is not offset, expect qty=1 for each
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

class TestStrategies(unittest.TestCase):
    def test_simple_moving_average(self):
        df = pd.DataFrame({
            'Symbol': ['A']*10,
            'SMA_fast': [1,2,3,4,5,6,7,8,9,10],
            'SMA_slow': [2,2,3,3,4,5,6,7,8,9],
        })
        strat = SimpleMovingAverage()
        # Should buy when fast > slow and last_signal != 'BUY',
        # sell when fast < slow and last_signal != 'SELL',
        # hold otherwise.
        events = df.itertuples()
        orders = [strat.check_condition(e) for e in events]
        # Compute expected signals step by step:
        # 0: fast=1, slow=2 -> SELL (last_signal=None)
        # 1: fast=2, slow=2 -> HOLD (no cross)
        # 2: fast=3, slow=3 -> HOLD (no cross)
        # 3: fast=4, slow=3 -> BUY (cross above, last_signal='SELL')
        # 4: fast=5, slow=4 -> HOLD (already 'BUY')
        # 5: fast=6, slow=5 -> HOLD (already 'BUY')
        # 6: fast=7, slow=6 -> HOLD (already 'BUY')
        # 7: fast=8, slow=7 -> HOLD (already 'BUY')
        # 8: fast=9, slow=8 -> HOLD (already 'BUY')
        # 9: fast=10, slow=9 -> HOLD (already 'BUY')
        expected = ['SELL', 'HOLD', 'HOLD', 'BUY', 'HOLD', 'HOLD', 'HOLD', 'HOLD', 'HOLD', 'HOLD']
        actual = [o.side for o in orders]
        self.assertEqual(actual, expected)
    def test_buy_n_hold(self):
        df = pd.DataFrame({'Symbol': ['A','B','A','B']})
        strat = BuyNHold()
        events = df.itertuples()
        orders = [strat.check_condition(e) for e in events]
        self.assertEqual(orders[0].side, 'BUY')
        self.assertEqual(orders[1].side, 'BUY')
        self.assertEqual(orders[2].side, 'HOLD')
        self.assertEqual(orders[3].side, 'HOLD')

if __name__ == "__main__":
    unittest.main()
