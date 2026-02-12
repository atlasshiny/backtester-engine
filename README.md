# backtester_engine

A lightweight, extensible event-driven backtesting engine and simple
simulation utilities for research and strategy development.

Features
--------
- Event-driven backtest loop with next-bar execution semantics.
- Modular components: `Strategy`, `Broker`, `Portfolio`, `BacktestEngine`.
- Supports single-asset and multi-asset "long format" input data (Date, Symbol, OHLCV).
- Performance analytics and GPU-aware computation helper (`get_gpu_status`).
- Monte Carlo wrapper for quick sensitivity / robustness checks (`simulations/monte_carlo.py`).

Quick start
-----------
1. Prepare market data as a pandas DataFrame. For single-asset data omit `Symbol` and the engine will inject `SINGLE` automatically. Required columns: `Close` (and typically `Open`, `High`, `Low`, `Date`).

2. Create components and run a backtest:

```python
from backtest import BacktestEngine, Portfolio, Broker
from strategies.simple_moving_average import SimpleMovingAverage

data = ...  # pandas DataFrame with Date, Open, High, Low, Close, (Symbol)
strategy = SimpleMovingAverage()
portfolio = Portfolio(initial_cash=100_000)
broker = Broker(portfolio)

engine = BacktestEngine(strategy=strategy, portfolio=portfolio, broker=broker, data_set=data)
engine.run()
engine.results()  # run analytics / plotting
```

Monte Carlo simulations
-----------------------
Use `simulations/monte_carlo.py` to run many small perturbations of the `Close` prices and collect distributional outcomes. Example (ad-hoc):

```python
from simulations.monte_carlo import MonteCarloSim

mc = MonteCarloSim(engine, sim_amount=100)
stats = mc.run_simulation(change_pct=0.01, seed=42, plot=False)
print(stats)
```

Performance Analytics
---------------------
At the end of a backtest `PerformanceAnalytics` computes a set of summary
statistics and optional charts. The following metrics are computed and
printed/saved; the module also returns a `stats` dictionary for programmatic
use with these keys:

- `FinalValue`: final portfolio equity (last point of `portfolio.value_history`).
- `TotalPnL`: profit and loss relative to `portfolio.initial_cash`.
- `MaxDrawdown`: maximum peak-to-trough drawdown (as a fraction, e.g. 0.25 = 25%).
- `SharpeRatio`: annualized Sharpe ratio using the per-step returns series.
- `SortinoRatio`: annualized Sortino ratio (downside deviation based).
- `TotalTrades`: number of completed trades (matched entry/exit pairs).
- `TotalCommission`: summed commissions from the broker trade log.
- `WinRate`: fraction of winning trades (0..1).
- `ProfitFactor`: gross profit / gross loss (net PnL basis).
- `AvgWin` / `AvgLoss`: average net PnL for winning / losing trades.
- `Expectancy`: average net PnL per trade.
- `MaxConsecWins` / `MaxConsecLosses`: longest streaks of wins/losses.

Additional outputs and charts (when `plot=True`) include:
- Equity curve and underwater (drawdown) plot
- Returns distribution and rolling Sharpe
- Monthly returns heatmap (when `Date` is present)
- Trade PnL distribution and per-symbol PnL bar chart
- Price vs Equity overlay and other diagnostic charts (saved to `./backtest_results.png`)

Saving metrics: when `save=True` a `backtest_metrics.csv` file is written containing
`Final Value`, `PnL`, `Max Drawdown`, `Sharpe`, `Sortino`, `Total Trades` and `Total Commission`.

Use the returned `stats` dict for programmatic checks (e.g., CI or automated reporting).

Testing
-------
- Unit tests live under `tests/`. Run the test suite with `pytest`:

```bash
python -m pip install -r requirements.txt  # ensure test deps
python -m pytest -q
```

Development notes
Optional dependencies
---------------------
Some features rely on optional, heavier dependencies that are not strictly required for the core engine to import and run simple tests.

- CuPy (GPU acceleration): CuPy wheels are CUDA-version specific. Install the wheel matching your CUDA toolkit, for example for CUDA 12.x:

```bash
python -m pip install cupy
```

Replace `12x` with the correct CUDA variant for your environment. Installing the generic `cupy` package may not provide a working GPU build on many systems.

- Seaborn (visualization in performance analytics): `performance_analytics` imports `seaborn` for nicer plots. If you don't need plotting or prefer a lightweight install, you can skip installing `seaborn` and still use the engine; plotting features will require it.

Recommended layout for requirements
----------------------------------
- `requirements.txt` — minimal runtime deps: `pandas`, `numpy`, `matplotlib`.
- `requirements-dev.txt` — development and test deps: `pytest`, `seaborn`, `numba`, and optional `cupy-cuda*` if you plan to run GPU tests locally.

This keeps core installs small while documenting optional capabilities for contributors and CI.

License
-------
See the [LICENSE](LICENSE) file in the repository.
