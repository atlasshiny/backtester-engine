import sys
import os
import cProfile
import pstats

# Ensure root and subfolders are in sys.path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'backtest'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'strategies'))

import pandas as pd
from backtest import BacktestEngine
from backtest import Portfolio
from backtest import Broker
from backtest import TechnicalIndicators
from strategies.simple_moving_average import SimpleMovingAverage # replace with choosen strategy

# follow the steps outlines in the backtest module description. 
# after loading data, if technical indicators are needed add them to dataset using the TechnicalIndicators class

def main():
    # Load data
    data_path = os.path.join('data', 'synthetic.csv')
    data_set = pd.read_csv(data_path)    

    # Generate strategy indicators if needed and attach to data_set
    # If Symbol column exists, sort by Date then Symbol (long format)
    data_processing = TechnicalIndicators(data=data_set)
    if 'Symbol' in data_set.columns:
        data_processing.data = data_processing.data.sort_values(['Date', 'Symbol']).reset_index(drop=True)
    data_processing.simple_moving_average()
    data_set = data_processing.final_df()

    # Choose strategy (replace with your own as needed)
    strategy = SimpleMovingAverage()

    # Create portfolio
    account = Portfolio(initial_cash=10000)

    # Create broker
    broker = Broker(portfolio=account, slippage=0.001, commission=0.001, log_hold=False)

    # Create and run engine
    engine = BacktestEngine(strategy=strategy, portfolio=account, broker=broker, data_set=data_set, warm_up=30, group_by_date=True)

    engine.run()
    engine.results(plot=True, save=False)


if __name__ == "__main__":
    main()

    # for debugging
    # cProfile.run('main()', 'profile_output.txt')
    # with open('profile_report.txt', 'w') as f:
    #     stats = pstats.Stats('profile_output.txt', stream=f)
    #     stats.sort_stats('cumtime').print_stats()

    # if os.path.exists('profile_output.txt'):
    #     os.remove('profile_output.txt')
