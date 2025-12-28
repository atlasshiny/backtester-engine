import sys
import os

# Ensure root and subfolders are in sys.path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'backtest'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'strategies'))

import pandas as pd
from backtest.engine import BacktestEngine
from backtest.portfolio import Portfolio
from backtest.strategy import Strategy
from backtest.order import Order
from strategies.buy_n_hold import BuyNHold  # Example custom strategy


def main():
    # Load data
    data_path = os.path.join('data', 'synthetic.csv')
    data_set = pd.read_csv(data_path)

    # Choose strategy (replace with your own as needed)
    strategy = BuyNHold()

    # Create portfolio
    account = Portfolio(initial_cash=10000, slippage=0.001, commission=0.001)

    # Create and run engine
    engine = BacktestEngine(strategy=strategy, portfolio=account, data_set=data_set)
    engine.run()
    engine.results(plot=True, save=False)


if __name__ == "__main__":
    main()
