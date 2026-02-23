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
from typing import Any, Callable, Literal
from .strategy import Strategy
from .portfolio import Portfolio
from .performance_analytics import PerformanceAnalytics
from .broker import Broker
from .event_view import HistoryView
from .indicator import TechnicalIndicators

try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:  # noqa: BLE001
    cp = None
    _CUPY_AVAILABLE = False


def get_gpu_status() -> dict:
    """Check GPU availability and return status information.
    
    Returns
    -------
    dict with keys:
        - available (bool): Whether CuPy is installed and GPU is accessible
        - backend (str): 'CuPy' if available, else 'NumPy (CPU)'
        - device (str): GPU device name if available, else 'CPU'
        - message (str): Human-readable status message
    """
    if not _CUPY_AVAILABLE:
        return {
            'available': False,
            'backend': 'NumPy (CPU)',
            'device': 'CPU',
            'message': 'CuPy not installed. Using CPU/NumPy. Install via: pip install cupy-cuda12x'
        }
    try:
        device_info = cp.cuda.Device().attributes
        device_name = device_info.get('ComputeCapability', 'Unknown')
        device_name = f"GPU (Compute Capability {device_name})"
        return {
            'available': True,
            'backend': 'CuPy',
            'device': device_name,
            'message': f'GPU acceleration enabled. Backend: CuPy, Device: {device_name}'
        }
    except Exception as e:
        return {
            'available': False,
            'backend': 'NumPy (CPU)',
            'device': 'CPU (CuPy unavailable)',
            'message': f'CuPy available but GPU not accessible. Falling back to CPU. Error: {e}'
        }


def print_gpu_status():
    """Print GPU status to stdout."""
    status = get_gpu_status()
    print(f"\n{'='*60}")
    print(f"GPU Status: {status['message']}")
    print(f"Backend: {status['backend']} | Device: {status['device']}")
    print(f"{'='*60}\n")


class BarView:
    """Lightweight view over a single bar using pre-extracted NumPy arrays."""
    __slots__ = ("Open", "High", "Low", "Close", "Date", "Symbol", "Index", "SMA_fast", "SMA_slow")

    def __init__(self, open_, high, low, close, date, symbol, index, sma_fast=None, sma_slow=None):
        self.Open = open_
        self.High = high
        self.Low = low
        self.Close = close
        self.Date = date
        self.Symbol = symbol
        self.Index = index
        self.SMA_fast = sma_fast
        self.SMA_slow = sma_slow

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
    prefer_gpu:
        "auto" (default): uses GPU if available and dataset is large enough.
        True: force GPU (raise error if unavailable).
        False: force CPU/NumPy.
        This parameter is passed to TechnicalIndicators and PerformanceAnalytics.
    """

    def __init__(self, strategy: Strategy, portfolio: Portfolio, broker: Broker, data_set: pd.DataFrame, warm_up: int = 0, group_by_date: bool = False, prefer_gpu: Literal["auto", True, False] = "auto",  precalculate: bool = False):
        """
        Initialize the BacktestEngine.

        Notes
        -----
        If the provided dataset does not contain a Symbol column, this constructor
        injects Symbol='SINGLE' so that the rest of the engine can treat the input
        consistently as long format.
        
        The prefer_gpu parameter is stored and passed to downstream components
        (TechnicalIndicators, PerformanceAnalytics) to enable consistent GPU usage.
        """
        self.strategy = strategy
        # Default minimum size for GPU heuristics (used by downstream modules)
        self.gpu_min_size = 10000
        # Ensure single-asset datasets behave like multi-asset long format by injecting a stable Symbol.
        # This avoids treating each row index as a separate "symbol". Accept either a pandas DataFrame or a dict of arrays (from df_to_arrays).
        self.arrays = None
        if isinstance(data_set, dict):
            # Choose array module based on explicit preference and dataset length
            from .array_utils import select_array_module, ensure_array
            xp = select_array_module(prefer_gpu, len(next(iter(data_set.values()))), self.gpu_min_size)
            self.arrays = {k: ensure_array(v, xp) for k, v in data_set.items()}
            # inject Symbol if missing (preserve xp array type)
            if 'Symbol' not in self.arrays:
                n = len(next(iter(self.arrays.values())))
                if xp is not np:
                    self.arrays['Symbol'] = xp.asarray(['SINGLE'] * n)
                else:
                    self.arrays['Symbol'] = np.array(['SINGLE'] * n)
            self.data_set = None
        else:
            if 'Symbol' not in data_set.columns:
                data_set = data_set.copy()
                data_set['Symbol'] = 'SINGLE'
            self.data_set = data_set
        self.portfolio = portfolio
        self.broker = broker
        self.warm_up = warm_up
        self.group_by_date = group_by_date
        self.prefer_gpu = prefer_gpu
        
        # Validate GPU preference if forced
        if prefer_gpu is True and not _CUPY_AVAILABLE:
            raise RuntimeError(
                "prefer_gpu=True but CuPy is not available. "
                "Install via: pip install cupy-cuda12x\n"
                "Or set prefer_gpu='auto' or False to use CPU."
            )

        # Optionally precompute indicators at construction time to avoid
        # computing them inside the tight run loop.
        if precalculate:
            ti_data = self.arrays if self.arrays is not None else self.data_set
            ti = TechnicalIndicators(ti_data, prefer_gpu=self.prefer_gpu, gpu_min_size=self.gpu_min_size)
            # Precompute a common set of indicators used by strategies.
            # This can be extended or made configurable if needed.
            try:
                ti.simple_moving_average()
            except Exception:
                pass
            try:
                ti.exponential_moving_average()
            except Exception:
                pass
            try:
                ti.rsi()
            except Exception:
                pass
            try:
                ti.bollinger_bands()
            except Exception:
                pass
            # Update engine's data sources with computed indicators
            if ti.arrays is not None:
                self.arrays = ti.arrays
                self.data_set = None
            else:
                self.data_set = ti.data
                self.arrays = None

            # If the strategy declares required_indicators, assert they exist
            required = getattr(self.strategy, 'required_indicators', None)
            if required:
                missing = []
                source = self.arrays if self.arrays is not None else self.data_set
                for col in required:
                    if source is None:
                        missing.append(col)
                    elif isinstance(source, dict):
                        if col not in source:
                            missing.append(col)
                    else:
                        # pandas DataFrame
                        if col not in source.columns:
                            missing.append(col)
                if missing:
                    raise RuntimeError(f"Precalculation requested but required indicators missing: {missing}")

    def set_gpu_policy(self, prefer_gpu: Literal["auto", True, False], gpu_min_size: int | None = None):
        """Set the engine-wide GPU preference.

        prefer_gpu: 'auto' | True | False
            - 'auto': use GPU when available and beneficial
            - True: force GPU (raises RuntimeError if unavailable)
            - False: force CPU/NumPy
        gpu_min_size: optional minimum array length for auto heuristics
        """
        if prefer_gpu is True and not _CUPY_AVAILABLE:
            raise RuntimeError("prefer_gpu=True but CuPy is not available on this system.")
        self.prefer_gpu = prefer_gpu
        if gpu_min_size is not None:
            self.gpu_min_size = int(gpu_min_size)

    def get_gpu_policy(self) -> dict:
        """Return current engine GPU policy as a dict."""
        return {
            'prefer_gpu': self.prefer_gpu,
            'gpu_min_size': getattr(self, 'gpu_min_size', None)
        }

    def reset(self):
        """Reset engine state for a clean backtest run.
        
        Clears warmup counters and internal tracking, but preserves data/portfolio/broker/strategy.
        Useful when running multiple backtests sequentially.
        """
        # Note: Data, portfolio, broker, strategy are NOT reset here; only internal tracking.
        # Call portfolio.reset() and strategy state management separately if needed.
        pass  # Engine state is re-initialized in run() via local variables

    def run(self, monte_carlo: bool = False, monte_carlo_sim_amount: int = 5000, monte_carlo_change_pct: float = 0.01, monte_carlo_seed: int | None = None, monte_carlo_plot: bool = False, monte_carlo_portfolio_factory: Callable[[], Any] | None = None, monte_carlo_progress: bool = False):
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

        Monte Carlo mode
        ----------------
        When `monte_carlo=True`, this method delegates to `MonteCarloSim` and
        returns a stats dict instead of running a single deterministic pass.
        """
        if monte_carlo:
            from .monte_carlo import MonteCarloSim

            mc = MonteCarloSim(self, sim_amount=monte_carlo_sim_amount)
            stats = mc.run_simulation(
                change_pct=monte_carlo_change_pct,
                seed=monte_carlo_seed,
                plot=monte_carlo_plot,
                portfolio_factory=monte_carlo_portfolio_factory,
                progress=monte_carlo_progress,
            )
            self.monte_carlo_results = mc.results
            self.monte_carlo_stats = stats
            return stats

        try:
            self.strategy.on_start()
        except NotImplementedError:
            pass

        window_size = getattr(self.strategy, 'history_window', None)

        # Multi-asset safe state
        pending_order_by_symbol: dict[str, object] = {}
        warmup_count_by_symbol: dict[str, int] = defaultdict(int)
        bar_index_by_symbol: dict[str, int] = defaultdict(int)

        # If strategy requests history, pre-split by symbol to avoid mixing assets in history slices.
        # Convert to NumPy arrays per symbol for fast slicing. Support dict-of-arrays.
        data_by_symbol = None
        arrays_by_symbol = None
        if window_size and window_size > 0:
            if self.arrays is not None and 'Symbol' in self.arrays:
                symbols = self.arrays['Symbol']
                arrays_by_symbol = {}
                for sym in np.unique(symbols):
                    # use integer indices for slicing to avoid boolean mask allocations
                    indices = np.nonzero(symbols == sym)[0]
                    arrays_by_symbol[sym] = {col: arr[indices] for col, arr in self.arrays.items()}
            elif self.data_set is not None and 'Symbol' in self.data_set.columns:
                data_by_symbol = {sym: grp.reset_index(drop=True) for sym, grp in self.data_set.groupby('Symbol', sort=False)}
                # Pre-extract arrays per symbol for fast history slicing
                arrays_by_symbol = {
                    sym: {col: df[col].values for col in df.columns}
                    for sym, df in data_by_symbol.items()
                }

        # Optionally bundle processing by timestamp to avoid intra-date ordering bias in long-format data.
        # This makes all symbols at the same Date behave as "simultaneous".
        if self.group_by_date:
            # Ensure everything is a dict of raw arrays once (fast path)
            if self.arrays is None and self.data_set is not None:
                # Convert DataFrame to per-column NumPy views (no copies)
                cols = list(self.data_set.columns)
                df_arr = self.data_set.to_numpy(copy=False)
                self.arrays = {col: df_arr[:, i] for i, col in enumerate(cols)}
                self.arrays['Index'] = self.data_set.index.to_numpy(copy=False)

            # If we still don't have Date, fall back to existing DataFrame grouping
            if self.arrays is not None and 'Date' in self.arrays:
                dates = self.arrays['Date']
                # Find group boundaries (first index of each unique date)
                _, first_indices = np.unique(dates, return_index=True)
                group_boundaries = np.sort(first_indices)
                group_boundaries = np.append(group_boundaries, len(dates))

                # Loop over date batches using integer slices/views
                for gi in range(len(group_boundaries) - 1):
                    start = int(group_boundaries[gi])
                    end = int(group_boundaries[gi + 1])
                    # date_batch is a view into the full arrays (O(1) slicing)
                    date_batch = {col: arr[start:end] for col, arr in self.arrays.items()}

                    columns = list(date_batch.keys())
                    date_batch['Index'] = np.arange(start, end)
                    n_rows = end - start

                    # Pre-extract frequently accessed arrays for ultra-fast inner loop
                    symbol_arr = date_batch.get('Symbol', None)
                    sma_fast_arr = date_batch.get('SMA_fast', None)
                    sma_slow_arr = date_batch.get('SMA_slow', None)
                    open_arr = date_batch.get('Open', None)
                    high_arr = date_batch.get('High', None)
                    low_arr = date_batch.get('Low', None)
                    close_arr = date_batch.get('Close', None)
                    date_arr = date_batch.get('Date', None)
                    index_arr = date_batch.get('Index', None)

                    # Combined loop: execute previous order and generate new signal in one pass
                    for local_idx in range(n_rows):
                        symbol = symbol_arr[local_idx] if symbol_arr is not None else 'SINGLE'
                        bar = BarView(
                            open_=open_arr[local_idx] if open_arr is not None else None,
                            high=high_arr[local_idx] if high_arr is not None else None,
                            low=low_arr[local_idx] if low_arr is not None else None,
                            close=close_arr[local_idx] if close_arr is not None else None,
                            date=date_arr[local_idx] if date_arr is not None else None,
                            symbol=symbol,
                            index=index_arr[local_idx] if index_arr is not None else None,
                            sma_fast=sma_fast_arr[local_idx] if sma_fast_arr is not None else None,
                            sma_slow=sma_slow_arr[local_idx] if sma_slow_arr is not None else None,
                        )

                        # 1) Execute previous bar's decision for each symbol using this bar's prices
                        if symbol in pending_order_by_symbol and pending_order_by_symbol[symbol] is not None:
                            self.broker.execute(event=bar, order=pending_order_by_symbol[symbol])

                        # 2) Generate new signals for each symbol for this timestamp
                        # Per-symbol warmup (counted in bars, not rows)
                        if self.warm_up and warmup_count_by_symbol[symbol] < self.warm_up:
                            warmup_count_by_symbol[symbol] += 1
                            pending_order_by_symbol[symbol] = None
                            bar_index_by_symbol[symbol] += 1
                            continue

                        if window_size and window_size > 0:
                            if arrays_by_symbol is not None and symbol in arrays_by_symbol:
                                # Use pre-extracted arrays for fast slicing
                                sym_i = bar_index_by_symbol[symbol]
                                start_idx = max(0, sym_i - window_size + 1)
                                end_idx = sym_i + 1
                                history = HistoryView(arrays_by_symbol[symbol], start_idx, end_idx, list(arrays_by_symbol[symbol].keys()))
                                signal = self.strategy.check_condition(bar, history)
                            else:
                                sym_i = bar_index_by_symbol[symbol]
                                start_idx = max(0, sym_i - window_size + 1)
                                end_idx = sym_i + 1
                                history = HistoryView(date_batch, start_idx, end_idx, columns)
                                signal = self.strategy.check_condition(bar, history)
                        else:
                            # Fast path: strategy reads indicators from bar attributes
                            signal = self.strategy.check_condition(bar)

                        pending_order_by_symbol[symbol] = signal
                        bar_index_by_symbol[symbol] += 1
                # end for each unique date
            elif self.data_set is not None and 'Date' in self.data_set.columns:
                # Fallback: group via DataFrame as before (rare path)
                grouped = self.data_set.groupby('Date', sort=False)
                for _, date_df in grouped:
                    # Convert the whole DataFrame to a single 2D ndarray once
                    # and create column views to avoid per-column allocations.
                    columns = list(date_df.columns)
                    df_arr = date_df.to_numpy(copy=False)
                    arrays = {col: df_arr[:, i] for i, col in enumerate(columns)}
                    # Provide a stable Index-like field for timestamp fallbacks/logging.
                    arrays['Index'] = date_df.index.to_numpy(copy=False)
                    n_rows = len(date_df)

                    # Pre-extract frequently accessed arrays for ultra-fast inner loop
                    symbol_arr = arrays.get('Symbol', None)
                    sma_fast_arr = arrays.get('SMA_fast', None)
                    sma_slow_arr = arrays.get('SMA_slow', None)
                    open_arr = arrays.get('Open', None)
                    high_arr = arrays.get('High', None)
                    low_arr = arrays.get('Low', None)
                    close_arr = arrays.get('Close', None)
                    date_arr = arrays.get('Date', None)
                    index_arr = arrays.get('Index', None)

                    # Combined loop: execute previous order and generate new signal in one pass
                    for idx in range(n_rows):
                        symbol = symbol_arr[idx] if symbol_arr is not None else 'SINGLE'
                        bar = BarView(
                            open_=open_arr[idx] if open_arr is not None else None,
                            high=high_arr[idx] if high_arr is not None else None,
                            low=low_arr[idx] if low_arr is not None else None,
                            close=close_arr[idx] if close_arr is not None else None,
                            date=date_arr[idx] if date_arr is not None else None,
                            symbol=symbol,
                            index=index_arr[idx] if index_arr is not None else None,
                            sma_fast=sma_fast_arr[idx] if sma_fast_arr is not None else None,
                            sma_slow=sma_slow_arr[idx] if sma_slow_arr is not None else None,
                        )

                        # 1) Execute previous bar's decision for each symbol using this bar's prices
                        if symbol in pending_order_by_symbol and pending_order_by_symbol[symbol] is not None:
                            self.broker.execute(event=bar, order=pending_order_by_symbol[symbol])

                        # 2) Generate new signals for each symbol for this timestamp
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
                                signal = self.strategy.check_condition(bar, history)
                            else:
                                sym_i = bar_index_by_symbol[symbol]
                                start_idx = max(0, sym_i - window_size + 1)
                                end_idx = sym_i + 1
                                history = HistoryView(arrays, start_idx, end_idx, columns)
                                signal = self.strategy.check_condition(bar, history)
                        else:
                            # Fast path: strategy reads indicators from bar attributes
                            signal = self.strategy.check_condition(bar)

                        pending_order_by_symbol[symbol] = signal
                        bar_index_by_symbol[symbol] += 1

        else:
            # Row-by-row processing (fast/simple, but multi-asset has intra-date ordering bias)
            # Support either DataFrame or pre-extracted arrays.
            if self.arrays is not None:
                arrays = self.arrays.copy()
                # Provide a stable Index-like field for timestamp fallbacks/logging.
                if 'Index' not in arrays:
                    arrays['Index'] = np.arange(len(next(iter(arrays.values()))))
                columns = list(arrays.keys())
                n_rows = len(next(iter(arrays.values())))
            else:
                # Convert full DataFrame to a single 2D ndarray and slice into column views.
                columns = list(self.data_set.columns)
                df_arr = self.data_set.to_numpy(copy=False)
                arrays = {col: df_arr[:, i] for i, col in enumerate(columns)}
                # Provide a stable Index-like field for timestamp fallbacks/logging.
                arrays['Index'] = self.data_set.index.to_numpy(copy=False)
                n_rows = len(self.data_set)

            symbol_arr = arrays.get('Symbol', None)
            sma_fast_arr = arrays.get('SMA_fast', None)
            sma_slow_arr = arrays.get('SMA_slow', None)
            open_arr = arrays.get('Open', None)
            high_arr = arrays.get('High', None)
            low_arr = arrays.get('Low', None)
            close_arr = arrays.get('Close', None)
            date_arr = arrays.get('Date', None)
            index_arr = arrays.get('Index', None)

            for idx in range(n_rows):
                symbol = symbol_arr[idx] if symbol_arr is not None else 'SINGLE'
                bar = BarView(
                    open_=open_arr[idx] if open_arr is not None else None,
                    high=high_arr[idx] if high_arr is not None else None,
                    low=low_arr[idx] if low_arr is not None else None,
                    close=close_arr[idx] if close_arr is not None else None,
                    date=date_arr[idx] if date_arr is not None else None,
                    symbol=symbol,
                    index=index_arr[idx] if index_arr is not None else None,
                    sma_fast=sma_fast_arr[idx] if sma_fast_arr is not None else None,
                    sma_slow=sma_slow_arr[idx] if sma_slow_arr is not None else None,
                )

                # Per-symbol warmup for long-format data (check BEFORE calling strategy)
                if self.warm_up and warmup_count_by_symbol[symbol] < self.warm_up:
                    warmup_count_by_symbol[symbol] += 1
                    bar_index_by_symbol[symbol] += 1
                    pending_order_by_symbol[symbol] = None
                    continue

                # execute the previous bar's decision for THIS symbol using the CURRENT bar's prices
                if symbol in pending_order_by_symbol and pending_order_by_symbol[symbol] is not None:
                    self.broker.execute(event=bar, order=pending_order_by_symbol[symbol])

                if window_size and window_size > 0:
                    # Mode B: Only slice if strategy.history_window is set
                    if arrays_by_symbol is not None and symbol in arrays_by_symbol:
                        # Use pre-extracted NumPy arrays for fast slicing
                        sym_i = bar_index_by_symbol[symbol]
                        start_idx = max(0, sym_i - window_size + 1)
                        end_idx = sym_i + 1
                        history = HistoryView(arrays_by_symbol[symbol], start_idx, end_idx, list(arrays_by_symbol[symbol].keys()) if isinstance(arrays_by_symbol[symbol], dict) else columns)
                        signal = self.strategy.check_condition(bar, history)
                    else:
                        # Fallback: slice from full dataset (single-asset mode)
                        start_idx = max(0, idx - window_size + 1)
                        end_idx = idx + 1
                        history = HistoryView(arrays, start_idx, end_idx, columns)
                        signal = self.strategy.check_condition(bar, history)
                else:
                    # Mode A: Fast path - skip slicing entirely
                    signal = self.strategy.check_condition(bar)

                # Per-symbol warmup for long-format data
                if self.warm_up and warmup_count_by_symbol[symbol] < self.warm_up:
                    warmup_count_by_symbol[symbol] += 1
                    bar_index_by_symbol[symbol] += 1
                    pending_order_by_symbol[symbol] = None
                    continue

                # execute the previous bar's decision for THIS symbol using the CURRENT bar's prices
                if symbol in pending_order_by_symbol and pending_order_by_symbol[symbol] is not None:
                    self.broker.execute(event=bar, order=pending_order_by_symbol[symbol])

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
        
        GPU acceleration is controlled by the engine's prefer_gpu parameter, which
        is passed to the analytics engine.
        """
        analytics = PerformanceAnalytics()
        # Pass broker.trade_log to analytics for trade statistics, and propagate GPU preference
        data_for_analytics = self.data_set if self.data_set is not None else self.arrays
        analytics.analyze_and_plot(
            self.portfolio, data_for_analytics,
            plot=plot, save=save,
            risk_free_rate=risk_free_rate,
            trade_log=self.broker.trade_log,
            annualization_factor=annualization_factor,
            prefer_gpu=self.prefer_gpu
        )