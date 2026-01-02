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
from collections import defaultdict
from .strategy import Strategy
from .portfolio import Portfolio
from .performance_analytics import PerformanceAnalytics
from .broker import Broker
from .event_view import EventView, HistoryView

class BacktestEngine:
    """
    Coordinates the backtest loop.

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
        warmup_count_by_symbol: dict[str, int] = defaultdict(int)
        bar_index_by_symbol: dict[str, int] = defaultdict(int)

        # If strategy requests history, pre-split by symbol to avoid mixing assets in history slices
        # Convert to NumPy arrays per symbol for fast slicing
        data_by_symbol = None
        arrays_by_symbol = None
        if window_size and window_size > 0 and 'Symbol' in self.data_set.columns:
            data_by_symbol = {sym: grp.reset_index(drop=True) for sym, grp in self.data_set.groupby('Symbol', sort=False)}
            # Pre-extract arrays per symbol for fast history slicing
            arrays_by_symbol = {
                sym: {col: df[col].values for col in df.columns}
                for sym, df in data_by_symbol.items()
            }

        # Optionally bundle processing by timestamp to avoid intra-date ordering bias in long-format data.
        # This makes all symbols at the same Date behave as "simultaneous".
        if self.group_by_date and 'Date' in self.data_set.columns:
            grouped = self.data_set.groupby('Date', sort=False)
            # Precompute arrays for each date to avoid repeated DataFrame slicing
            arrays_by_date = {
                date: {col: df[col].to_numpy(copy=False) for col in df.columns}
                for date, df in grouped
            }
            # Also store columns and index for each date
            columns_by_date = {date: df.columns for date, df in grouped}
            index_by_date = {date: df.index.to_numpy(copy=False) for date, df in grouped}

            for date, date_df in grouped:
                columns = columns_by_date[date]
                arrays = arrays_by_date[date]
                arrays['Index'] = index_by_date[date]
                n_rows = len(date_df)

                symbol_arr = arrays.get('Symbol', None)

                # 1) Execute previous bar's decision for each symbol using this bar's prices
                for idx in range(n_rows):
                    event = EventView(arrays, idx, columns)
                    symbol = symbol_arr[idx] if symbol_arr is not None else 'SINGLE'
                    if symbol in pending_order_by_symbol and pending_order_by_symbol[symbol] is not None:
                        self.broker.execute(event=event, order=pending_order_by_symbol[symbol])

                # 2) Generate new signals for each symbol for this timestamp
                for idx in range(n_rows):
                    event = EventView(arrays, idx, columns)
                    symbol = symbol_arr[idx] if symbol_arr is not None else 'SINGLE'

                    # Per-symbol warmup (counted in bars, not rows)
                    if self.warm_up and warmup_count_by_symbol[symbol] < self.warm_up:
                        warmup_count_by_symbol[symbol] += 1
                        pending_order_by_symbol[symbol] = None
                        bar_index_by_symbol[symbol] += 1
                        continue

                    if window_size and window_size > 0:
                        if arrays_by_symbol is not None and symbol in arrays_by_symbol:
                            # Use pre-extracted NumPy arrays for fast slicing
                            sym_i = bar_index_by_symbol[symbol]
                            start_idx = max(0, sym_i - window_size + 1)
                            end_idx = sym_i + 1
                            history = HistoryView(arrays_by_symbol[symbol], start_idx, end_idx, data_by_symbol[symbol].columns)
                            signal = self.strategy.check_condition(event, history)
                        else:
                            # Fallback: slice from full dataset (single-asset mode only)
                            # Note: this is slower, but only active when a strategy requests history.
                            # Use the same sequential bar index for the synthetic SINGLE symbol.
                            sym_i = bar_index_by_symbol[symbol]
                            start_idx = max(0, sym_i - window_size + 1)
                            end_idx = sym_i + 1
                            history = HistoryView(arrays, start_idx, end_idx, columns)
                            signal = self.strategy.check_condition(event, history)
                    else:
                        signal = self.strategy.check_condition(event)

                    pending_order_by_symbol[symbol] = signal
                    bar_index_by_symbol[symbol] += 1

        else:
            # Row-by-row processing (fast/simple, but multi-asset has intra-date ordering bias)
            # Pre-extract numpy arrays for faster attribute access
            columns = self.data_set.columns
            arrays = {col: self.data_set[col].to_numpy(copy=False) for col in columns}
            # Provide a stable Index-like field for timestamp fallbacks/logging.
            arrays['Index'] = self.data_set.index.to_numpy(copy=False)
            n_rows = len(self.data_set)
            symbol_arr = arrays.get('Symbol', None)

            for idx in range(n_rows):
                event = EventView(arrays, idx, columns)
                symbol = symbol_arr[idx] if symbol_arr is not None else 'SINGLE'

                if window_size and window_size > 0:
                    # Mode B: Only slice if strategy.history_window is set
                    if arrays_by_symbol is not None and symbol in arrays_by_symbol:
                        # Use pre-extracted NumPy arrays for fast slicing
                        sym_i = bar_index_by_symbol[symbol]
                        start_idx = max(0, sym_i - window_size + 1)
                        end_idx = sym_i + 1
                        history = HistoryView(arrays_by_symbol[symbol], start_idx, end_idx, data_by_symbol[symbol].columns)
                        signal = self.strategy.check_condition(event, history)
                    else:
                        # Fallback: slice from full dataset (single-asset mode)
                        start_idx = max(0, idx - window_size + 1)
                        end_idx = idx + 1
                        history = HistoryView(arrays, start_idx, end_idx, columns)
                        signal = self.strategy.check_condition(event, history)
                else:
                    # Mode A: Fast path - skip slicing entirely
                    signal = self.strategy.check_condition(event)

                # Per-symbol warmup for long-format data
                if self.warm_up and warmup_count_by_symbol[symbol] < self.warm_up:
                    warmup_count_by_symbol[symbol] += 1
                    bar_index_by_symbol[symbol] += 1
                    pending_order_by_symbol[symbol] = None
                    continue

                # execute the previous bar's decision for THIS symbol using the CURRENT bar's prices
                if symbol in pending_order_by_symbol and pending_order_by_symbol[symbol] is not None:
                    self.broker.execute(event=event, order=pending_order_by_symbol[symbol])

                pending_order_by_symbol[symbol] = signal
                bar_index_by_symbol[symbol] += 1

        try:
            self.strategy.on_finish()
        except NotImplementedError:
            pass
                
    def results(self, plot: bool = True, save: bool = False, risk_free_rate: float = 0.0, annualization_factor: float = 252.0):
        """
        Run performance analytics and plotting after the backtest.
        Args:
            plot (bool): Whether to plot results.
            save (bool): Whether to save results to file.
            risk_free_rate (float): Risk-free rate for Sharpe/Sortino.
            annualization_factor (float): Factor for annualizing Sharpe/Sortino (e.g., 252 for daily, 12 for monthly).

        Notes
        -----
        The analytics module consumes Portfolio.value_history (equity curve) and
        Broker.trade_log. In multi-asset row-by-row mode, each equity "step" is an
        event (Date, Symbol) rather than a single Date. For true per-Date returns,
        prefer running with group_by_date=True.
        """
        analytics = PerformanceAnalytics()
        # Pass broker.trade_log to analytics for trade statistics
        analytics.analyze_and_plot(self.portfolio, self.data_set, plot=plot, save=save, risk_free_rate=risk_free_rate, trade_log=self.broker.trade_log, annualization_factor=annualization_factor)