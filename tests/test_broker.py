import unittest
from backtest import Portfolio, Broker, Order

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

if __name__ == "__main__":
    unittest.main()
