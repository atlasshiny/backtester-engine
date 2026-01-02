import unittest
from backtest import Order

class TestOrder(unittest.TestCase):
    def test_order_fields(self):
        o = Order(symbol='A', side='BUY', qty=2, order_type='LIMIT', limit_price=9.5, timestamp=123)
        self.assertEqual(o.symbol, 'A')
        self.assertEqual(o.side, 'BUY')
        self.assertEqual(o.qty, 2)
        self.assertEqual(o.order_type, 'LIMIT')
        self.assertEqual(o.limit_price, 9.5)
        self.assertEqual(o.timestamp, 123)

if __name__ == "__main__":
    unittest.main()
