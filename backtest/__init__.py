"""
The backtest module provides core classes and logic for running trading strategy backtests,
including the engine, broker, portfolio, order, analytics, and strategy interfaces.

Initalization Order
- Strategy
- Portfolio
- Broker
- Engine

After initalization, use engine.run() and engine.results() to run the backtest and get statistics

"""

from .engine import BacktestEngine
from .strategy import Strategy
from .portfolio import Portfolio
from .broker import Broker
from .performance_analytics import PerformanceAnalytics
from .order import Order
from .position import Position