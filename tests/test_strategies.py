import unittest
import pandas as pd
from strategies.simple_moving_average import SimpleMovingAverage
from strategies.buy_n_hold import BuyNHold

class TestStrategies(unittest.TestCase):
    def test_simple_moving_average(self):
        df = pd.DataFrame({
            'Symbol': ['A']*10,
            'SMA_fast': [1,2,3,4,5,6,7,8,9,10],
            'SMA_slow': [2,2,3,3,4,5,6,7,8,9],
        })
        strat = SimpleMovingAverage()
        events = df.itertuples()
        orders = [strat.check_condition(e) for e in events]
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
