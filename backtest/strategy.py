"""backtest.strategy

Strategy interface.

Strategies are stateless or stateful objects that receive market events and
produce Order objects. The engine decides when orders are executed (next-bar
model) and the broker decides how orders are filled.

Event format
------------
The engine passes an event-like object with attribute access (e.g. event.Close).
In long-format multi-asset data, events usually contain Date, Symbol, and OHLCV
(plus any precomputed indicator columns).

History slicing
--------------
If a strategy sets history_window > 0, the engine will pass a lightweight
history window object as the second argument to check_condition(). The history
object supports fast NumPy-style column access and a minimal .iloc interface.
Otherwise, the engine omits history for performance.
"""

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
        Create a Strategy.

        Parameters
        ----------
        history_window:
            Number of bars of history required to make a decision.
            - None or 0: the engine will call check_condition(event) only.
            - > 0: the engine will call check_condition(event, history_df).
        """
        self.history_window = history_window
        pass

    def on_start(self):
        """
        Optional hook called before processing begins.

        Notes
        -----
        You may leave this unimplemented; the engine treats NotImplementedError
        as "no-op".
        """
        raise NotImplementedError("on_start must be implemented by subclass.")

    def on_finish(self):
        """
        Optional hook called after processing completes.

        Notes
        -----
        You may leave this unimplemented; the engine treats NotImplementedError
        as "no-op".
        """
        raise NotImplementedError("on_finish must be implemented by subclass.")

    def check_condition(self, event, history=None) -> Order:
        """
        Produce an Order given the current event and optional history.

        Parameters
        ----------
        event:
            The current bar/event (event-like object with attribute access).
        history:
            Optional historical window, only provided when history_window > 0.

        Returns
        -------
        Order
            BUY/SELL/HOLD decision for the engine to schedule.
        """
        raise NotImplementedError("check_condition must be implemented by subclass.")