"""backtest package

Core components for a lightweight event-driven backtesting engine.

Typical usage
-------------
1) Load market data into a pandas DataFrame.
	- Recommended: long format with columns including Date, Symbol, Open, High, Low,
	  Close, Volume (and any indicators you want).
2) (Optional) Precompute indicators with TechnicalIndicators and attach them to
	the dataset.
3) Create:
	- Strategy
	- Portfolio
	- Broker
	- BacktestEngine
4) Run:
	engine.run()
	engine.results(...)

GPU Acceleration
----------------
To enable GPU acceleration:
1. Install CuPy: pip install cupy-cuda12x (adjust cuda version for your system)
2. Use print_gpu_status() to verify GPU availability
3. Pass prefer_gpu='auto' (default) or True to BacktestEngine

Timing model
------------
Orders generated at time t are executed on the next bar for that symbol.
"""

from .engine import BacktestEngine, get_gpu_status, print_gpu_status
from .strategy import Strategy
from .portfolio import Portfolio
from .broker import Broker
from .performance_analytics import PerformanceAnalytics
from .order import Order
from .position import Position
from .indicator import TechnicalIndicators