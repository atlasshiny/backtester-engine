import pandas as pd
from strategy import Strategy

class BacktestEngine():
    def __init__(self, strategy: Strategy, data_set: pd.DataFrame):
        self.strategy = strategy
        self.data_set = data_set
        pass

    def run(self):
        for event in self.data_set.itertuples():
            pass
