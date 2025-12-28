# Backtester Engine

## Overview

This is a modular, extensible Python backtesting engine for systematic trading strategies. It supports realistic trade execution, multi-symbol data, slippage, commission, and advanced analytics. Designed for research, prototyping, and educational use.

---

## Features

- **Modular OOP Design:** Strategy, Portfolio, Engine, Order, and Analytics components
- **Indicator Support:** Rolling window/historical data for indicator-based strategies (e.g., moving averages)
- **Realistic Execution:** Slippage and commission modeling
- **Multi-Symbol Support:** Backtest multiple assets simultaneously
- **Trade Logging:** Detailed trade logs and position tracking
- **Performance Analytics:** Equity curve, drawdown, Sharpe/Sortino ratios, and more
- **Synthetic Data Generation:** Tools for creating test datasets
- **Extensible:** Easy to add new strategies, order types, or analytics

---

## Installation

1. Clone this repository:
	```bash
	git clone https://github.com/yourusername/backtester_engine.git
	cd backtester_engine
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
	*(Typical dependencies: pandas, numpy, matplotlib)*

---

## Quick Start

1. **Prepare Data:** Place your OHLCV CSV in the `data/` folder (see `data/synthetic.csv` for format).
2. **Choose or Implement a Strategy:**
	- Use built-in strategies (e.g., `SimpleMovingAverage`, `BuyNHold`)
	- Or create your own by subclassing `Strategy` and implementing `check_condition()`
3. **Run a Backtest:**
	- Edit `main.py` to select your strategy and data
	- Run:
	  ```bash
	  python main.py
	  ```

---

## Example Usage

```python
from backtest.engine import BacktestEngine
from backtest.portfolio import Portfolio
from strategies.simple_moving_average import SimpleMovingAverage
import pandas as pd

data = pd.read_csv('data/synthetic.csv')
strategy = SimpleMovingAverage(fast_window=5, slow_window=20)
portfolio = Portfolio(initial_cash=10000, slippage=0.001, commission=0.001)
engine = BacktestEngine(strategy, portfolio, data)
engine.run()
engine.results(plot=True, save=False)
```

---

## Project Structure

```
backtest/
	 engine.py                # BacktestEngine: main loop
	 portfolio.py             # Portfolio: cash, positions, trade log
	 order.py                 # Order dataclass
	 performance_analytics.py # Analytics and plotting
	 strategy.py              # Base Strategy class
strategies/
	 simple_moving_average.py # Example indicator strategy
	 buy_n_hold.py            # Example passive strategy
data/
	 synthetic.csv            # Example dataset
	 synthetic_data_generator.py # Data generation tools
main.py                     # Entry point
test.py                     # Example/test script
```

---

## Extending

- **Add a Strategy:**
  1. Subclass `Strategy` in `strategies/`
  2. Implement `check_condition(self, event, history=None)`
  3. Set `self.history_window` in `__init__` if you need rolling data

- **Add Analytics:**
  - Extend `performance_analytics.py` with new metrics or plots

---

## License

MIT License. See LICENSE file for details.
