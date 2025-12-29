import pandas as pd
from .order import Order
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
        """
        Initialize a Strategy instance.
        Args:
            history_window (int | None): Number of bars of historical data required for the strategy (optional).
        """
        self.history_window = history_window
        pass

    def on_start(self):
        """
        Optional method to run logic before the backtest starts.
        Override this in subclasses if needed.
        """
        raise NotImplementedError("on_start must be implemented by subclass.")

    def on_finish(self):
        """
        Optional method to run logic after the backtest ends.
        Override this in subclasses if needed.
        """
        raise NotImplementedError("on_finish must be implemented by subclass.")

    # old expected return value
    # Union[Tuple[Literal["BUY"], int], Tuple[Literal["SELL"], int], Tuple[Literal["HOLD"], int]]
    def check_condition(self, event: tuple) -> Order:
        """
        Generate an order based on the current market event.
        Must be overridden in subclasses.
        Args:
            event (tuple): The current market event/bar.
        Returns:
            Order: The generated order for this bar.
        """
        raise NotImplementedError("check_condition must be implemented by subclass.")