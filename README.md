# Backtester Engine

Lightweight, modular Python backtesting framework for researching and prototyping systematic trading strategies.

Key capabilities: realistic execution (slippage & commission), multi-symbol backtests, indicator support, trade logging, and performance analytics.

---

## Quick Highlights

- Modular components: `Engine`, `Strategy`, `Portfolio`, `Order`, and analytics
- Support for indicator-driven strategies and rolling-history windows
- Realistic execution modeling: slippage and commission
- Multi-symbol backtesting and detailed trade logs
- Built-in utilities for synthetic data generation and plotting

---

## Getting Started

1. Clone this repository:
	```bash
	git clone https://github.com/atlasshiny/backtester_engine.git
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
backtest/                          # Core engine, portfolio, order, analytics
	 engine.py                     # BacktestEngine: main loop
	 portfolio.py                  # Portfolio: cash, positions, trade log
	 order.py                      # Order dataclass
	 performance_analytics.py      # Analytics and plotting
	 strategy.py                   # Base Strategy class
data/                              # Example and real datasets
	 synthetic.csv                 # Example dataset
	 synthetic_data_generator.py   # Data generation tools
simulations/                  
	 monte_carlo.py                # run multiple simulations with added noise
main.py                            # Entry point
test.py                            # Example/test script
```

---

## Extending the Engine

- Add new strategies by subclassing `Strategy` in `strategies/` and implementing the decision logic (e.g., `check_condition()`).
- Add analytics by extending `performance_analytics.py` with new metrics or visualizations.
- Add custom order/execution models by modifying `broker.py` and `order.py`.

---

## Tests

Run the provided tests (requires pytest):

```bash
pytest -q
```

---

## License

MIT License — see the `LICENSE` file.

```
