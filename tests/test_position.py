import unittest
from backtest import Position

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

if __name__ == "__main__":
    unittest.main()
