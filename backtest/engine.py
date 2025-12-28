import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .strategy import Strategy
from .portfolio import Portfolio
from .performance_analytics import PerformanceAnalytics

class BacktestEngine():
    def __init__(self, strategy: Strategy, portfolio: Portfolio, data_set: pd.DataFrame):
        self.strategy = strategy
        self.data_set = data_set
        self.portfolio = portfolio
        pass

    def run(self):
        pending_order = None
        window_size = getattr(self.strategy, 'history_window', None)
        for idx, event in enumerate(self.data_set.itertuples()):
            # execute the previous bar's decision using the CURRENT bar's prices
            if pending_order is not None:
                self.portfolio.execute(
                    event=event,
                    order=pending_order
                )

            # Prepare historical window for strategy
            if window_size is not None and window_size > 0:
                history = self.data_set.iloc[max(0, idx - window_size + 1):idx + 1]
                pending_order = self.strategy.check_condition(event, history)
            else:
                pending_order = self.strategy.check_condition(event)

                
    def results(self, plot: bool = True, save: bool = False, risk_free_rate: float = 0.0):
        analytics = PerformanceAnalytics()
        analytics.analyze_and_plot(self.portfolio, self.data_set, plot=plot, save=save, risk_free_rate=risk_free_rate)