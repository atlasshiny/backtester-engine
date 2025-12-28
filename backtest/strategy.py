import pandas as pd
from .order import Order
from typing import Literal, Union, Tuple

class Strategy():
    def __init__(self):
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