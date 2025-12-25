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
            print(event)
            print("position count")
            print(self.portfolio.position)
            print("cash")
            print(self.portfolio.cash)
            match self.strategy.check_condition(event=event):
                case "BUY":
                    if self.portfolio.cash > event[3]: # check if there is enough money to buy (this is checking the "high" of the day to make the trade)
                        self.portfolio.cash = self.portfolio.cash - event[3]
                        self.portfolio.position += 1 # keep a simple count of positions, no information yet
                        print("Action executed : Buying")
                    else:
                        print("No Action taken : Not enough money to buy in")
                    pass
                case "HOLD":
                    print("Action executed : Holding")
                    pass
                case "SELL":
                    if self.portfolio.position >= 1: # check if there is any positions to sell
                        self.portfolio.position -= 1 # remove 1 from the count of positions, add some random amount of money in order for the system to work for testing purposes
                        self.portfolio.cash += 100
                        print("Action executed : Selling")
                    else:
                        print("No Action taken : No positions exist to sell")
                    pass
