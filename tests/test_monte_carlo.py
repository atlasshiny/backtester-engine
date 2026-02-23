import matplotlib
matplotlib.use('Agg')

import pandas as pd

from backtest.monte_carlo import MonteCarloSim


class SimplePortfolio:
    def __init__(self, initial_cash=100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.value_history = []


class MinimalEngine:
    def __init__(self, data_frame, initial_cash=100000.0):
        self.data_set = data_frame.copy()
        self.portfolio = SimplePortfolio(initial_cash=initial_cash)

    def run(self):
        last = float(self.data_set['Close'].iloc[-1])
        self.portfolio.value_history.append(last * 10)


def load_synthetic():
    # Small synthetic DataFrame so tests are self-contained
    return pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=4),
        'Open': [10, 11, 12, 13],
        'High': [11, 12, 13, 14],
        'Low': [9, 10, 11, 12],
        'Close': [10.5, 11.5, 12.5, 13.5],
        'Volume': [100, 110, 120, 130],
    })


def test_monte_carlo_smoke():
    df = load_synthetic()
    eng = MinimalEngine(df)
    mc = MonteCarloSim(eng, sim_amount=10)
    mc.run_simulation(change_pct=0.01)

    assert len(mc.results) == 10
    assert all(r is not None for r in mc.results)
    assert all(isinstance(r, (int, float)) for r in mc.results)
