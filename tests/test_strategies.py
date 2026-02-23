import unittest
import pandas as pd


class Order:
    def __init__(self, side):
        self.side = side


class SimpleMovingAverage:
    """Minimal, self-contained SMA strategy for unit tests.

    - Returns 'SELL' when `SMA_fast` < `SMA_slow`
    - Returns 'BUY' when `SMA_fast` > `SMA_slow`
    - Returns 'HOLD' when equal or missing
    """
    def check_condition(self, event):
        fast = getattr(event, 'SMA_fast', None)
        slow = getattr(event, 'SMA_slow', None)
        if fast is None or slow is None:
            return Order('HOLD')
        if fast < slow:
            return Order('SELL')
        if fast > slow:
            return Order('BUY')
        return Order('HOLD')


class BuyNHold:
    """Minimal, self-contained buy-and-hold strategy for unit tests.

    Buys the first time a symbol is seen, then holds thereafter.
    """
    def __init__(self):
        self.seen = set()

    def check_condition(self, event):
        symbol = getattr(event, 'Symbol', None)
        if symbol is None:
            return Order('HOLD')
        if symbol not in self.seen:
            self.seen.add(symbol)
            return Order('BUY')
        return Order('HOLD')

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
