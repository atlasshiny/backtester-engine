import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import pandas as pd

from simulations.monte_carlo import MonteCarloSim


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
    p = Path(__file__).resolve().parents[1] / 'data' / 'synthetic.csv'
    return pd.read_csv(p)


def test_monte_carlo_smoke():
    df = load_synthetic()
    eng = MinimalEngine(df)
    mc = MonteCarloSim(eng, sim_amount=10)
    mc.run_simulation(change_pct=0.01)

    assert len(mc.results) == 10
    assert all(r is not None for r in mc.results)
    assert all(isinstance(r, (int, float)) for r in mc.results)
