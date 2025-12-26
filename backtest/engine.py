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
            self.portfolio.execute(event=event, action=self.strategy.check_condition(event=event), symbol_or_name=event.Symbol) #fiugre out how to pull the symbol/name in a better manner
                
    def results(self, plot: bool, save: bool):
        final_portfolio_value = self.portfolio.portfolio_value_snapshot(self.data_set.iloc[-1]['Close'])
        print(f"Final Portfolio Value : {final_portfolio_value}")
        print(f"PnL : {final_portfolio_value - self.portfolio.initial_cash}")
        print(f"Remaining Positions : {self.portfolio.positions['SYNTH']['amount']}")
        if plot == True:
            # equity curve
            plt.figure(figsize=(10, 5))
            plt.plot(self.portfolio.value_history, label='Equity Curve')
            plt.xlabel('Time (events)')
            plt.ylabel('Portfolio Value')
            plt.title('Backtest Equity Curve')
            plt.legend()

            # price curve
            plt.figure(figsize=(10, 5))
            plt.plot(self.data_set['Close'].values, label='Price (Close)')
            plt.xlabel('Time (events)')
            plt.ylabel('Price')
            plt.title('Price Curve (Close)')
            plt.legend()


            # show all graphs in seperate windows
            plt.show()

        if plot == True:
            # Combined equity and price curve in one window
            fig, ax1 = plt.subplots(figsize=(12, 6))
            color1 = 'tab:blue'
            ax1.set_xlabel('Time (events)')
            ax1.set_ylabel('Portfolio Value', color=color1)
            ax1.plot(self.portfolio.value_history, color=color1, label='Equity Curve')
            ax1.tick_params(axis='y', labelcolor=color1)

            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('Price (Close)', color=color2)
            ax2.plot(self.data_set['Close'].values, color=color2, label='Price (Close)')
            ax2.tick_params(axis='y', labelcolor=color2)

            fig.suptitle('Equity Curve and Price Curve')
            fig.tight_layout()
            plt.show()

        if save == True:
            pass
