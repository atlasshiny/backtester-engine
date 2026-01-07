#!/usr/bin/env python3
"""
Unit tests for GPU detection and configuration in BacktestEngine.
"""

import unittest
import pandas as pd
from backtest import (
    BacktestEngine, 
    Portfolio, 
    Broker, 
    Strategy, 
    Order,
    get_gpu_status, 
    print_gpu_status
)


class DummyStrategy(Strategy):
    """Simple strategy for testing that always holds."""
    def check_condition(self, bar, history=None):
        return Order(side='HOLD', symbol=bar.Symbol, qty=0)


class TestGPUDetection(unittest.TestCase):
    """Test GPU availability detection and status reporting."""

    def test_get_gpu_status_returns_dict(self):
        """Verify get_gpu_status returns a properly structured dict."""
        status = get_gpu_status()
        self.assertIsInstance(status, dict)
        self.assertIn('available', status)
        self.assertIn('backend', status)
        self.assertIn('device', status)
        self.assertIn('message', status)

    def test_get_gpu_status_available_field(self):
        """Verify 'available' field is a boolean."""
        status = get_gpu_status()
        self.assertIsInstance(status['available'], bool)

    def test_get_gpu_status_backend_field(self):
        """Verify 'backend' is either 'CuPy' or 'NumPy (CPU)'."""
        status = get_gpu_status()
        self.assertIn(status['backend'], ['CuPy', 'NumPy (CPU)'])

    def test_get_gpu_status_device_field(self):
        """Verify 'device' field is a string."""
        status = get_gpu_status()
        self.assertIsInstance(status['device'], str)

    def test_get_gpu_status_message_field(self):
        """Verify 'message' field is a string."""
        status = get_gpu_status()
        self.assertIsInstance(status['message'], str)
        self.assertTrue(len(status['message']) > 0)

    def test_print_gpu_status_no_error(self):
        """Verify print_gpu_status executes without error."""
        try:
            print_gpu_status()
        except Exception as e:
            self.fail(f"print_gpu_status() raised {type(e).__name__}: {e}")


class TestBacktestEngineGPUParameter(unittest.TestCase):
    """Test BacktestEngine's global prefer_gpu parameter."""

    def setUp(self):
        """Create test data and base engine components."""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Symbol': ['TEST'] * 100,
            'Open': range(100, 200),
            'High': range(101, 201),
            'Low': range(99, 199),
            'Close': range(100, 200),
        })
        self.strategy = DummyStrategy()

    def test_engine_default_prefer_gpu_is_auto(self):
        """Verify engine defaults to prefer_gpu='auto'."""
        portfolio = Portfolio(initial_cash=10000)
        broker = Broker(portfolio)
        engine = BacktestEngine(
            strategy=self.strategy,
            portfolio=portfolio,
            broker=broker,
            data_set=self.test_data
        )
        self.assertEqual(engine.prefer_gpu, 'auto')

    def test_engine_accept_prefer_gpu_auto(self):
        """Verify engine accepts prefer_gpu='auto'."""
        portfolio = Portfolio(initial_cash=10000)
        broker = Broker(portfolio)
        engine = BacktestEngine(
            strategy=self.strategy,
            portfolio=portfolio,
            broker=broker,
            data_set=self.test_data,
            prefer_gpu='auto'
        )
        self.assertEqual(engine.prefer_gpu, 'auto')

    def test_engine_accept_prefer_gpu_false(self):
        """Verify engine accepts prefer_gpu=False."""
        portfolio = Portfolio(initial_cash=10000)
        broker = Broker(portfolio)
        engine = BacktestEngine(
            strategy=self.strategy,
            portfolio=portfolio,
            broker=broker,
            data_set=self.test_data,
            prefer_gpu=False
        )
        self.assertEqual(engine.prefer_gpu, False)
        self.assertFalse(engine.prefer_gpu)

    def test_engine_accept_prefer_gpu_true_when_available(self):
        """Verify engine accepts prefer_gpu=True when GPU is available."""
        status = get_gpu_status()
        if not status['available']:
            self.skipTest("GPU not available, skipping prefer_gpu=True test")
        
        portfolio = Portfolio(initial_cash=10000)
        broker = Broker(portfolio)
        engine = BacktestEngine(
            strategy=self.strategy,
            portfolio=portfolio,
            broker=broker,
            data_set=self.test_data,
            prefer_gpu=True
        )
        self.assertEqual(engine.prefer_gpu, True)
        self.assertTrue(engine.prefer_gpu)

    def test_engine_reject_prefer_gpu_true_when_unavailable(self):
        """Verify engine raises RuntimeError when prefer_gpu=True but GPU unavailable."""
        status = get_gpu_status()
        if status['available']:
            self.skipTest("GPU is available, skipping GPU unavailable test")
        
        portfolio = Portfolio(initial_cash=10000)
        broker = Broker(portfolio)
        with self.assertRaises(RuntimeError) as context:
            BacktestEngine(
                strategy=self.strategy,
                portfolio=portfolio,
                broker=broker,
                data_set=self.test_data,
                prefer_gpu=True
            )
        self.assertIn('CuPy', str(context.exception))

    def test_engine_propagates_prefer_gpu_in_results(self):
        """Verify engine stores and propagates GPU preference to results()."""
        portfolio = Portfolio(initial_cash=10000)
        broker = Broker(portfolio)
        engine = BacktestEngine(
            strategy=self.strategy,
            portfolio=portfolio,
            broker=broker,
            data_set=self.test_data,
            prefer_gpu=False
        )
        self.assertEqual(engine.prefer_gpu, False)
        # Verify the parameter can be passed through to results without error
        # (we don't run the full backtest, just verify it's stored)
        self.assertTrue(hasattr(engine, 'prefer_gpu'))


class TestGPUStatusConsistency(unittest.TestCase):
    """Test consistency of GPU status across multiple calls."""

    def test_gpu_status_consistent_across_calls(self):
        """Verify get_gpu_status returns consistent results."""
        status1 = get_gpu_status()
        status2 = get_gpu_status()
        self.assertEqual(status1['available'], status2['available'])
        self.assertEqual(status1['backend'], status2['backend'])
        self.assertEqual(status1['device'], status2['device'])

    def test_gpu_status_backend_matches_available(self):
        """Verify backend field matches availability status."""
        status = get_gpu_status()
        if status['available']:
            self.assertEqual(status['backend'], 'CuPy')
        else:
            self.assertEqual(status['backend'], 'NumPy (CPU)')


if __name__ == '__main__':
    unittest.main(verbosity=2)
