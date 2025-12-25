from backtest import BacktestEngine
from backtest import Strategy
from backtest import Portfolio
import pandas as pd
import cProfile

# strategy for testing
class TestStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def check_condition(self, event: tuple): 
        # assuming that event follows the structure in synthetic.csv
        if event[7] > 500:
            return ["SELL",1]
        elif event[7] < 250:
            return ["BUY",4]
        else:
            return "HOLD"

strat = TestStrategy()
account = Portfolio(120000.62)

# turn the data set csv into a dataframe
data_set = pd.read_csv("./data/synthetic.csv")

engine = BacktestEngine(strategy=strat, portfolio=account, data_set=data_set)
engine.run()
engine.results(plot=True, save=False)
# cProfile.run('engine.run()', sort='cumtime')
