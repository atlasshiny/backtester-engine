import pandas as pd
from .dataclasses.order import Order
from typing import Literal, Union, Tuple

class Strategy():
    """
    Base class for all trading strategies used in the backtesting engine.

    Usage:
        - Subclass this and implement the check_condition method.
        - Optionally set history_window in __init__ to specify the required length of historical data.
        - The engine will call check_condition(event, history) for each event/bar.

    Methods:
        - __init__(history_window: int | None = None):
            Set the required rolling window size for historical data (if needed).
        - on_start():
            Optional. Override to run logic before the backtest starts.
        - on_finish():
            Optional. Override to run logic after the backtest ends.
        - check_condition(event: tuple, history=None) -> Order:
            Must be overridden. Should return an Order object based on the event and optional history DataFrame.
    """
    def __init__(self, history_window: int | None = None):
        self.history_window = history_window
        pass

    def on_start(self):
        """Override this for each individual subclass if needed (Optional Method)"""
        raise NotImplementedError("on_start must be implemented by subclass.")

    def on_finish(self):
        """Override this for each individual subclass if needed (Optional Method)"""
        raise NotImplementedError("on_finish must be implemented by subclass.")

    # old expected return value
    # Union[Tuple[Literal["BUY"], int], Tuple[Literal["SELL"], int], Tuple[Literal["HOLD"], int]]
    def check_condition(self, event: tuple) -> Order:
        """Override this for each individual subclass. Should return an Order object."""
        raise NotImplementedError("check_condition must be implemented by subclass.")