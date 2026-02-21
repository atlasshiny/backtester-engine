import copy
import logging
from typing import Any, Callable, Optional
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class MonteCarloSim:
    """Monte Carlo simulation wrapper for a backtest engine.

    Runs `sim_amount` synthetic backtests by applying multiplicative noise to
    the `Close` column of `engine.data_set`. The wrapper requires that the
    provided `engine` object expose at least the following minimal interface:
      - `data_set`: pandas.DataFrame with a `Close` column
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
        if not hasattr(self.engine, 'data_set') or not hasattr(self.engine, 'run'):
            raise TypeError('engine must provide `data_set` (DataFrame) and `run()`')

        df_orig = self.engine.data_set.copy()
        n_rows = len(df_orig)

        # Prepare portfolio snapshot or factory
        use_factory = portfolio_factory is not None
        port_snapshot = None if use_factory else copy.deepcopy(getattr(self.engine, 'portfolio', None))

        rng = np.random.default_rng(seed)
        self.results = []

        original_port = getattr(self.engine, 'portfolio', None)
        try:
            for i in range(self.sim_amount):
                # prepare synthetic data
                sim = df_orig.copy()
                noise = rng.normal(loc=1.0, scale=change_pct, size=n_rows)
                sim['Close'] = sim['Close'] * noise

                # swap data and portfolio
                self.engine.data_set = sim
                if use_factory:
                    try:
                        self.engine.portfolio = portfolio_factory()
                    except Exception as e:
                        logger.warning('portfolio_factory failed: %s', e)
                        self.engine.portfolio = copy.deepcopy(port_snapshot)
                else:
                    # deepcopy snapshot to isolate runs
                    self.engine.portfolio = copy.deepcopy(port_snapshot)

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
                self.engine.data_set = df_orig
                self.engine.portfolio = original_port
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