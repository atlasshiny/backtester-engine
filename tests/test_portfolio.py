import unittest
from backtest import Portfolio

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

if __name__ == "__main__":
    unittest.main()
