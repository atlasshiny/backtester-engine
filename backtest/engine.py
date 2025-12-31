import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .strategy import Strategy
from .portfolio import Portfolio
from .performance_analytics import PerformanceAnalytics
from .broker import Broker

class BacktestEngine():
    def __init__(self, strategy: Strategy, portfolio: Portfolio, broker: Broker, data_set: pd.DataFrame, warm_up: int = 0):
        """
        Initialize the BacktestEngine.
        Args:
            strategy (Strategy): The trading strategy instance.
            portfolio (Portfolio): The portfolio instance.
            broker (Broker): The broker instance.
            data_set (pd.DataFrame): The market data.
            warm_up (int): Number of bars to use for warm-up (no trading).
        """
        self.strategy = strategy
        self.data_set = data_set
        self.portfolio = portfolio
        self.broker = broker
        self.warm_up = warm_up

    def run(self):
        """
        Run the backtest loop: feed events to the strategy, execute orders, and update portfolio/broker state.
        """
        try:
            self.strategy.on_start()
        except NotImplementedError:
            pass

        window_size = getattr(self.strategy, 'history_window', None)

        # Multi-asset safe state
        pending_order_by_symbol: dict[str, object] = {}
        warmup_count_by_symbol: dict[str, int] = {}
        bar_index_by_symbol: dict[str, int] = {}

        # If strategy requests history, pre-split by symbol to avoid mixing assets in history slices
        data_by_symbol = None
        if window_size and window_size > 0 and 'Symbol' in self.data_set.columns:
            data_by_symbol = {sym: grp.reset_index(drop=True) for sym, grp in self.data_set.groupby('Symbol', sort=False)}

        for idx, event in enumerate(self.data_set.itertuples()):
            symbol = getattr(event, 'Symbol', None)
            if symbol is None:
                # Fallback for unexpected schemas
                symbol = getattr(event, 'Index', None)

            if window_size and window_size > 0:
                # Mode B: Only slice if strategy.history_window is set
                if data_by_symbol is not None and symbol in data_by_symbol:
                    sym_i = bar_index_by_symbol.get(symbol, 0)
                    history = data_by_symbol[symbol].iloc[max(0, sym_i - window_size + 1): sym_i + 1]
                    signal = self.strategy.check_condition(event, history)
                else:
                    history = self.data_set.iloc[max(0, idx - window_size + 1):idx + 1]
                    signal = self.strategy.check_condition(event, history)
            else:
                # Mode A: Fast path - skip slicing entirely
                signal = self.strategy.check_condition(event)

            # Per-symbol warmup for long-format data
            if self.warm_up and symbol is not None:
                warm_i = warmup_count_by_symbol.get(symbol, 0)
                if warm_i < self.warm_up:
                    warmup_count_by_symbol[symbol] = warm_i + 1
                    bar_index_by_symbol[symbol] = bar_index_by_symbol.get(symbol, 0) + 1
                    continue
            elif idx < self.warm_up:
                if symbol is not None:
                    bar_index_by_symbol[symbol] = bar_index_by_symbol.get(symbol, 0) + 1
                continue

            # execute the previous bar's decision for THIS symbol using the CURRENT bar's prices
            if symbol is not None and symbol in pending_order_by_symbol and pending_order_by_symbol[symbol] is not None:
                self.broker.execute(
                    event=event,
                    order=pending_order_by_symbol[symbol]
                )

            pending_order_by_symbol[symbol] = signal
            if symbol is not None:
                bar_index_by_symbol[symbol] = bar_index_by_symbol.get(symbol, 0) + 1

        try:
            self.strategy.on_finish()
        except NotImplementedError:
            pass
                
    def results(self, plot: bool = True, save: bool = False, risk_free_rate: float = 0.0):
        """
        Run performance analytics and plotting after the backtest.
        Args:
            plot (bool): Whether to plot results.
            save (bool): Whether to save results to file.
            risk_free_rate (float): Risk-free rate for Sharpe/Sortino.
        """
        analytics = PerformanceAnalytics()
        # Pass broker.trade_log to analytics for trade statistics
        analytics.analyze_and_plot(self.portfolio, self.data_set, plot=plot, save=save, risk_free_rate=risk_free_rate, trade_log=self.broker.trade_log)