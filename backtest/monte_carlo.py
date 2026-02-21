import copy
import logging
from typing import Any, Callable, Optional
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Type hints for better IDE support
try:
    from .indicator import TechnicalIndicators
    _TI_AVAILABLE = True
except ImportError:
    _TI_AVAILABLE = False


class MonteCarloSim:
    """Monte Carlo simulation wrapper for a backtest engine.

        Runs `sim_amount` synthetic backtests by applying multiplicative noise to
        the `Close` series of engine market data (`engine.data_set` or
        `engine.arrays`). The wrapper requires that the
    provided `engine` object expose at least the following minimal interface:
            - `data_set` (pandas.DataFrame) and/or `arrays` (dict) with `Close`
      - `portfolio`: object representing account state (may expose
                     `value_history` and/or `total_value`)
      - `run()`: method that executes a backtest over `data_set` and updates
                 the `portfolio` accordingly

    Key features
    - Optional reproducible RNG via `seed`.
    - Use `portfolio_factory` (callable) to create a fresh portfolio per run
      to avoid expensive deep copies of large portfolio objects.
    - Optional plotting (`plot=True`) and optional progress logging.

    Parameters
    ----------
    engine: object
        Engine-like object implementing the minimal interface described above.
    sim_amount: int
        Number of Monte Carlo repetitions to run.
    """

    def __init__(self, engine: Any, sim_amount: int = 5000):
        self.engine = engine
        self.sim_amount = int(sim_amount)
        self.results: list = []

    def run_simulation(
        self,
        change_pct: float = 0.01,
        *,
        seed: Optional[int] = None,
        plot: bool = False,
        portfolio_factory: Optional[Callable[[], Any]] = None,
        progress: bool = False,
    ) -> dict:
        """Execute Monte Carlo runs and return summary statistics.

        Parameters
        ----------
        change_pct: float
            Standard deviation for multiplicative noise applied to `Close`.
        seed: int | None
            Optional RNG seed for reproducibility.
        plot: bool
            If True, generate a histogram plot and return stats; default False.
        portfolio_factory: callable | None
            If provided, called before each run to create a fresh portfolio
            instance. If omitted a deepcopy of the original portfolio is used.
        progress: bool
            If True, log progress messages.

        Returns
        -------
        dict
            Summary statistics computed from successful runs: mean, p5, p95,
            and count of successful results.
        """

        # Validate engine has required attributes
        if not hasattr(self.engine, 'run'):
            raise TypeError('engine must provide `run()`')

        original_df = getattr(self.engine, 'data_set', None)
        has_arrays_attr = hasattr(self.engine, 'arrays')
        original_arrays = getattr(self.engine, 'arrays', None) if has_arrays_attr else None

        base_arrays = None
        close_base = None
        if has_arrays_attr:
            if original_arrays is not None:
                if 'Close' not in original_arrays:
                    raise KeyError("engine.arrays must contain 'Close' for Monte Carlo")
                base_arrays = dict(original_arrays)
            elif original_df is not None:
                if 'Close' not in original_df.columns:
                    raise KeyError("engine.data_set must contain 'Close' for Monte Carlo")
                # Build column-wise NumPy views once; avoid full DataFrame copies per run.
                base_arrays = {col: original_df[col].to_numpy(copy=False) for col in original_df.columns}
            else:
                raise TypeError('engine must provide market data in `arrays` or `data_set`')
            close_base = np.asarray(base_arrays['Close'], dtype=float)
        else:
            if original_df is None or 'Close' not in original_df.columns:
                raise TypeError("engine must provide `data_set` DataFrame with a 'Close' column")
            close_base = np.asarray(original_df['Close'].to_numpy(copy=False), dtype=float)

        n_rows = len(close_base)

        # Prepare portfolio snapshot or factory
        use_factory = portfolio_factory is not None
        port_snapshot = None if use_factory else copy.deepcopy(getattr(self.engine, 'portfolio', None))

        rng = np.random.default_rng(seed)
        self.results = []

        original_port = getattr(self.engine, 'portfolio', None)
        original_broker_port = None
        broker = getattr(self.engine, 'broker', None)
        original_trade_log = None
        original_last_prices = None
        original_strategy_state = None
        strategy = getattr(self.engine, 'strategy', None)
        if broker is not None:
            original_broker_port = getattr(broker, 'portfolio', None)
            if hasattr(broker, 'trade_log'):
                original_trade_log = copy.deepcopy(broker.trade_log)
            if hasattr(broker, 'last_prices'):
                original_last_prices = copy.deepcopy(broker.last_prices)
        if strategy is not None:
            try:
                original_strategy_state = copy.deepcopy(getattr(strategy, '__dict__', {}))
            except Exception:
                original_strategy_state = None
        try:
            for i in range(self.sim_amount):
                # prepare synthetic close path without copying full tabular data
                noise = rng.normal(loc=1.0, scale=change_pct, size=n_rows)
                sim_close = close_base * noise

                # swap market data and recompute indicators with perturbed Close
                if has_arrays_attr:
                    sim_arrays = dict(base_arrays)
                    sim_arrays['Close'] = sim_close
                    # Recompute technical indicators on perturbed data to ensure MC variance
                    self._recompute_indicators(sim_arrays)
                    self.engine.arrays = sim_arrays
                    self.engine.data_set = None
                else:
                    sim_df = original_df.copy(deep=False)
                    sim_df['Close'] = sim_close
                    # Recompute technical indicators on perturbed data
                    self._recompute_indicators_df(sim_df)
                    self.engine.data_set = sim_df
                if use_factory:
                    try:
                        self.engine.portfolio = portfolio_factory()
                    except Exception as e:
                        logger.warning('portfolio_factory failed: %s', e)
                        self.engine.portfolio = copy.deepcopy(port_snapshot)
                else:
                    # deepcopy snapshot to isolate runs
                    self.engine.portfolio = copy.deepcopy(port_snapshot)

                if broker is not None and hasattr(broker, 'portfolio'):
                    broker.portfolio = self.engine.portfolio
                if broker is not None and hasattr(broker, 'trade_log'):
                    broker.trade_log = []
                if broker is not None and hasattr(broker, 'last_prices'):
                    broker.last_prices = {}
                if strategy is not None and original_strategy_state is not None:
                    strategy.__dict__.clear()
                    strategy.__dict__.update(copy.deepcopy(original_strategy_state))

                # run
                self.engine.run()

                # extract final value with fallbacks
                final = getattr(self.engine.portfolio, 'total_value', None)
                if final is None:
                    vh = getattr(self.engine.portfolio, 'value_history', None)
                    final = vh[-1] if vh else None

                self.results.append(final)

                if progress and ((i + 1) % 100 == 0 or i == self.sim_amount - 1):
                    logger.info('Completed %d/%d simulations', i + 1, self.sim_amount)

        finally:
            # restore original engine state
            try:
                self.engine.data_set = original_df
                if has_arrays_attr:
                    self.engine.arrays = original_arrays
                self.engine.portfolio = original_port
                if broker is not None and hasattr(broker, 'portfolio'):
                    broker.portfolio = original_broker_port
                if broker is not None and hasattr(broker, 'trade_log') and original_trade_log is not None:
                    broker.trade_log = original_trade_log
                if broker is not None and hasattr(broker, 'last_prices') and original_last_prices is not None:
                    broker.last_prices = original_last_prices
                if strategy is not None and original_strategy_state is not None:
                    strategy.__dict__.clear()
                    strategy.__dict__.update(original_strategy_state)
            except Exception:
                logger.exception('Failed to fully restore engine state')

        # Clean results: keep numeric ones only
        numeric_results = [r for r in self.results if isinstance(r, (int, float))]
        if not numeric_results:
            logger.warning('No numeric results produced by simulations')

        stats = {}
        if numeric_results:
            arr = np.array(numeric_results)
            stats = {
                'mean': float(np.mean(arr)),
                'p5': float(np.percentile(arr, 5)),
                'p95': float(np.percentile(arr, 95)),
                'count': len(numeric_results),
            }

            if plot:
                plt.hist(arr, bins=50, color='skyblue', edgecolor='black')
                plt.title(f'Distribution of Final Portfolio Value ({len(arr)} runs)')
                plt.axvline(np.mean(arr), color='red', linestyle='dashed', label='Mean')
                plt.legend()
                try:
                    plt.show()
                except Exception:
                    logger.debug('Non-interactive backend; skipping plt.show()')

        return stats

    def _recompute_indicators(self, arrays_dict: dict) -> None:
        """Recompute technical indicators on perturbed Close to ensure MC variance.
        
        Modifies arrays_dict in-place by updating indicator columns (SMA_fast, SMA_slow, etc).
        """
        if not _TI_AVAILABLE:
            return
        try:
            ti = TechnicalIndicators(arrays_dict)
            # Check which indicators exist and recompute them
            if 'SMA_fast' in arrays_dict or 'SMA_slow' in arrays_dict:
                ti.simple_moving_average()
            # Update the arrays_dict with recomputed indicators
            if ti.arrays is not None:
                for key in ['SMA_fast', 'SMA_slow', 'RSI', 'BB_upper', 'BB_lower', 'BB_middle']:
                    if key in ti.arrays and key not in ['Close', 'Open', 'High', 'Low', 'Volume', 'Date', 'Symbol', 'Index']:
                        arrays_dict[key] = ti.arrays[key]
        except Exception as e:
            logger.debug('Indicator recomputation skipped for arrays: %s', e)

    def _recompute_indicators_df(self, df) -> None:
        """Recompute technical indicators on perturbed Close (DataFrame version).
        
        Modifies df in-place by updating indicator columns.
        """
        if not _TI_AVAILABLE:
            return
        try:
            # Convert to arrays temporarily
            arrays = {col: df[col].to_numpy(copy=False) for col in df.columns}
            ti = TechnicalIndicators(arrays)
            if 'SMA_fast' in df.columns or 'SMA_slow' in df.columns:
                ti.simple_moving_average()
            # Write recomputed indicators back to DataFrame
            if ti.data is not None:
                for col in ['SMA_fast', 'SMA_slow', 'RSI', 'BB_upper', 'BB_lower', 'BB_middle']:
                    if col in ti.data.columns and col in df.columns:
                        df[col] = ti.data[col].values
        except Exception as e:
            logger.debug('Indicator recomputation skipped for DataFrame: %s', e)