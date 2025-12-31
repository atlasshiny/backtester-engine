"""backtest.engine

Event-driven backtesting engine.

This engine iterates over a market dataset and:
- Calls a Strategy to generate an Order per bar (or per symbol-bar in long format).
- Executes orders via a Broker (commission/slippage and fill rules live there).
- Updates portfolio value history through the broker/portfolio.

Data formats supported
----------------------
1) Multi-asset long format (recommended)
   One row per (Date, Symbol) with OHLCV columns. The dataset should be sorted by
   Date then Symbol to ensure deterministic processing.

2) Single-asset
   A dataset without a Symbol column is treated as a single synthetic asset by
   injecting Symbol='SINGLE'.

Timing model
------------
The engine uses a "next-bar" execution model:
- At time t: strategy observes the bar and emits an order.
- At time t+1: that order is executed using the next bar's prices.

For multi-asset long format you can choose between:
- Row-by-row processing: simpler/faster but introduces intra-timestamp ordering bias.
- group_by_date processing: bundles all symbols at a given Date so they are treated
  as simultaneous.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .strategy import Strategy
from .portfolio import Portfolio
from .performance_analytics import PerformanceAnalytics
from .broker import Broker

class BacktestEngine():
    """Coordinates the backtest loop.

    Parameters
    ----------
    strategy:
        Strategy implementation that produces an Order given the current event/bar.
    portfolio:
        Portfolio instance that tracks cash and positions.
    broker:
        Broker instance responsible for fills, costs, and trade logging.
    data_set:
        Market data in either single-asset (no Symbol column) or long format.
    warm_up:
        Number of bars to skip trading for (per symbol). Useful for indicator warmup.
    group_by_date:
        When True and a Date column exists, process all symbols at a given Date
        together (reduces ordering/time bias in multi-asset data).
    """

    def __init__(self, strategy: Strategy, portfolio: Portfolio, broker: Broker, data_set: pd.DataFrame, warm_up: int = 0, group_by_date: bool = False):
        """
        Initialize the BacktestEngine.

        Notes
        -----
        If the provided dataset does not contain a Symbol column, this constructor
        injects Symbol='SINGLE' so that the rest of the engine can treat the input
        consistently as long format.
        """
        self.strategy = strategy
        # Ensure single-asset datasets behave like multi-asset long format by injecting a stable Symbol.
        # This avoids treating each row index as a separate "symbol".
        if 'Symbol' not in data_set.columns:
            data_set = data_set.copy()
            data_set['Symbol'] = 'SINGLE'
        self.data_set = data_set
        self.portfolio = portfolio
        self.broker = broker
        self.warm_up = warm_up
        self.group_by_date = group_by_date

    def run(self):
        """
        Run the backtest loop.

        High-level flow
        ---------------
        1) Optionally call strategy.on_start().
        2) For each timestamp (or row):
           - Execute the previous bar's pending order for the symbol using the
             current bar's prices (next-bar execution).
           - Ask the strategy for the next order.
           - Store that order as the pending order for the symbol.
           - Update portfolio value via Broker/Portfolio.
        3) Optionally call strategy.on_finish().

        Performance notes
        -----------------
        If the strategy sets history_window > 0, the engine will construct a
        historical slice DataFrame per event. Otherwise, it takes a fast path and
        passes only the event to the strategy.
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

        # Optionally bundle processing by timestamp to avoid intra-date ordering bias in long-format data.
        # This makes all symbols at the same Date behave as "simultaneous".
        if self.group_by_date and 'Date' in self.data_set.columns:
            grouped = self.data_set.groupby('Date', sort=False)
            for _, date_df in grouped:
                # 1) Execute previous bar's decision for each symbol using this bar's prices
                for event in date_df.itertuples(index=False):
                    symbol = getattr(event, 'Symbol', 'SINGLE')
                    if symbol in pending_order_by_symbol and pending_order_by_symbol[symbol] is not None:
                        self.broker.execute(event=event, order=pending_order_by_symbol[symbol])

                # 2) Generate new signals for each symbol for this timestamp
                for event in date_df.itertuples(index=False):
                    symbol = getattr(event, 'Symbol', 'SINGLE')

                    # Per-symbol warmup (counted in bars, not rows)
                    warm_i = warmup_count_by_symbol.get(symbol, 0)
                    if self.warm_up and warm_i < self.warm_up:
                        warmup_count_by_symbol[symbol] = warm_i + 1
                        pending_order_by_symbol[symbol] = None
                        bar_index_by_symbol[symbol] = bar_index_by_symbol.get(symbol, 0) + 1
                        continue

                    if window_size and window_size > 0:
                        if data_by_symbol is not None and symbol in data_by_symbol:
                            sym_i = bar_index_by_symbol.get(symbol, 0)
                            history = data_by_symbol[symbol].iloc[max(0, sym_i - window_size + 1): sym_i + 1]
                            signal = self.strategy.check_condition(event, history)
                        else:
                            # Fallback: slice from full dataset (single-asset mode only)
                            # Note: this is slower, but only active when a strategy requests history.
                            # Use the same sequential bar index for the synthetic SINGLE symbol.
                            sym_i = bar_index_by_symbol.get(symbol, 0)
                            history = self.data_set.iloc[max(0, sym_i - window_size + 1): sym_i + 1]
                            signal = self.strategy.check_condition(event, history)
                    else:
                        signal = self.strategy.check_condition(event)

                    pending_order_by_symbol[symbol] = signal
                    bar_index_by_symbol[symbol] = bar_index_by_symbol.get(symbol, 0) + 1

        else:
            # Row-by-row processing (fast/simple, but multi-asset has intra-date ordering bias)
            for idx, event in enumerate(self.data_set.itertuples()):
                symbol = getattr(event, 'Symbol', 'SINGLE')

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
                if self.warm_up:
                    warm_i = warmup_count_by_symbol.get(symbol, 0)
                    if warm_i < self.warm_up:
                        warmup_count_by_symbol[symbol] = warm_i + 1
                        bar_index_by_symbol[symbol] = bar_index_by_symbol.get(symbol, 0) + 1
                        pending_order_by_symbol[symbol] = None
                        continue

                # execute the previous bar's decision for THIS symbol using the CURRENT bar's prices
                if symbol in pending_order_by_symbol and pending_order_by_symbol[symbol] is not None:
                    self.broker.execute(event=event, order=pending_order_by_symbol[symbol])

                pending_order_by_symbol[symbol] = signal
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

        Notes
        -----
        The analytics module consumes Portfolio.value_history (equity curve) and
        Broker.trade_log. In multi-asset row-by-row mode, each equity "step" is an
        event (Date, Symbol) rather than a single Date. For true per-Date returns,
        prefer running with group_by_date=True.
        """
        analytics = PerformanceAnalytics()
        # Pass broker.trade_log to analytics for trade statistics
        analytics.analyze_and_plot(self.portfolio, self.data_set, plot=plot, save=save, risk_free_rate=risk_free_rate, trade_log=self.broker.trade_log)