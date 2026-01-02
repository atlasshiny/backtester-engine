import unittest
import numpy as np
from backtest.event_view import EventView, HistoryView

class TestEventView(unittest.TestCase):
    def setUp(self):
        self.arrays = {
            'Open': np.array([1.0, 2.0, 3.0]),
            'Close': np.array([1.5, 2.5, 3.5]),
            'Symbol': np.array(['A', 'B', 'C'])
        }
        self.columns = ['Open', 'Close', 'Symbol']

    def test_attribute_access(self):
        event = EventView(self.arrays, 1, self.columns)
        self.assertEqual(event.Open, 2.0)
        self.assertEqual(event.Close, 2.5)
        self.assertEqual(event.Symbol, 'B')

    def test_index_access(self):
        event = EventView(self.arrays, 2, self.columns)
        self.assertEqual(event[0], 3.0)
        self.assertEqual(event['Close'], 3.5)

    def test_len(self):
        event = EventView(self.arrays, 0, self.columns)
        self.assertEqual(len(event), 3)

    def test_missing_attribute(self):
        event = EventView(self.arrays, 0, self.columns)
        with self.assertRaises(AttributeError):
            _ = event.Nonexistent

class TestHistoryView(unittest.TestCase):
    def setUp(self):
        self.arrays = {
            'Open': np.array([1.0, 2.0, 3.0, 4.0]),
            'Close': np.array([1.5, 2.5, 3.5, 4.5]),
            'Symbol': np.array(['A', 'A', 'B', 'B'])
        }
        self.columns = ['Open', 'Close', 'Symbol']

    def test_column_access(self):
        history = HistoryView(self.arrays, 1, 3, self.columns)
        np.testing.assert_array_equal(history['Open'], np.array([2.0, 3.0]))
        np.testing.assert_array_equal(history['Close'], np.array([2.5, 3.5]))

    def test_iloc_single(self):
        history = HistoryView(self.arrays, 0, 4, self.columns)
        event = history.iloc[2]
        self.assertEqual(event.Open, 3.0)
        self.assertEqual(event.Symbol, 'B')

    def test_iloc_slice(self):
        history = HistoryView(self.arrays, 0, 4, self.columns)
        sub = history.iloc[1:3]
        np.testing.assert_array_equal(sub['Open'], np.array([2.0, 3.0]))

    def test_len_and_shape(self):
        history = HistoryView(self.arrays, 1, 4, self.columns)
        self.assertEqual(len(history), 3)
        self.assertEqual(history.shape, (3, 3))

    def test_to_pandas(self):
        history = HistoryView(self.arrays, 0, 2, self.columns)
        df = history.to_pandas()
        self.assertListEqual(list(df['Open']), [1.0, 2.0])
        self.assertListEqual(list(df['Symbol']), ['A', 'A'])

    def test_iloc_out_of_bounds(self):
        history = HistoryView(self.arrays, 0, 2, self.columns)
        with self.assertRaises(IndexError):
            _ = history.iloc[2]

    def test_iloc_wrong_type(self):
        history = HistoryView(self.arrays, 0, 2, self.columns)
        with self.assertRaises(TypeError):
            _ = history.iloc['bad']

    def test_iloc_step(self):
        history = HistoryView(self.arrays, 0, 4, self.columns)
        with self.assertRaises(ValueError):
            _ = history.iloc[0:4:2]

if __name__ == '__main__':
    unittest.main()
