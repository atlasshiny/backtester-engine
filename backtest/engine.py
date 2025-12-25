import pandas as pd
import matplotlib.pyplot as plt
from .strategy import Strategy
from .portfolio import Portfolio

class BacktestEngine():
    def __init__(self, strategy: Strategy, portfolio: Portfolio, data_set: pd.DataFrame):
        self.strategy = strategy
        self.data_set = data_set
        self.portfolio = portfolio
        pass

    def run(self):
        for event in self.data_set.itertuples():
            self.portfolio.execute(event=event, action=self.strategy.check_condition(event=event), symbol_or_name=event[2]) #fiugre out how to pull the symbol/name in a better manner
                
    def results(self, plot: bool, save: bool):
        final_portfolio_value = self.portfolio.portfolio_value_snapshot(self.data_set.iloc[-1]['Close'])
        print(f"Final Portfolio Value : {final_portfolio_value}"),
        print(f"PnL : {final_portfolio_value - self.portfolio.initial_cash}")

        if plot == True:
            plt.figure(figsize=(10, 5))
            plt.plot(self.portfolio.value_history, label='Equity Curve')
            plt.xlabel('Time (events)')
            plt.ylabel('Portfolio Value')
            plt.title('Backtest Equity Curve')
            plt.legend()
            plt.show()

        # have all other code run before saving
        if save == True:
            trade_log = pd.DataFrame(self.portfolio.trade_log)
            trade_log.to_csv("./trade_log")
            # consider adding final portfolio value and pnl to seperate file
        else:
            pass
