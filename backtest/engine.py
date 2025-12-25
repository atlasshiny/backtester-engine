import pandas as pd
from .strategy import Strategy
from .portfolio import Portfolio

class BacktestEngine():
    def __init__(self, strategy: Strategy, portfolio: Portfolio, data_set: pd.DataFrame):
        self.strategy = strategy
        self.data_set = data_set
        self.portfolio = portfolio
        self.successful_trades = pd.DataFrame()
        self.unsuccessful_trades = pd.DataFrame()
        pass

    def run(self):
        for event in self.data_set.itertuples():
            Portfolio.execute(event=event, action=self.strategy.check_condition(event=event))
                
    def results(self, save: bool):
        # have all other code run before saving
        if save == True:
            print("Enter saved results file name:")
            file_name = input("")
            # figure out how the data will be represented before trying to save a .txt or .csv file of it
        pass