from backtest import BacktestEngine
from backtest import Strategy
import pandas as pd

# strategy for testing
class TestStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def check_condition(self, event: tuple) -> bool: 
        # assuming that event follows the structure in synthetic.csv
        if event[6] > 300:
            return True
        else:
            return False

    def execute(self, event):
        return "Would-be execution"

strat = TestStrategy()

# turn the data set csv into a dataframe
data_set = pd.read_csv("./data/synthetic.csv")

engine = BacktestEngine(strategy=strat, data_set=data_set)
engine.run()