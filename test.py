from backtest import BacktestEngine
from backtest import Strategy
from backtest import Portfolio
from backtest import Order
import pandas as pd
import cProfile

# strategy for testing
class TestStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def check_condition(self, event: tuple):
        # assuming data looks similar to synthetic.csv
        if event.Volume > 700:
            return Order(symbol=event.Symbol, side="SELL", qty=1)
        elif event.Volume < 600:
            return Order(symbol=event.Symbol, side="BUY", qty=1)
        else:
            return Order(symbol=event.Symbol, side="HOLD", qty=0)

strat = TestStrategy()
account = Portfolio(5000.62, slippage=0.001, commission=0.001)

# turn the data set csv into a dataframe
data_set = pd.read_csv("./data/synthetic.csv")

engine = BacktestEngine(strategy=strat, portfolio=account, data_set=data_set)
engine.run()
engine.results(plot=True, save=False)
# cProfile.run('engine.run()', sort='cumtime')
