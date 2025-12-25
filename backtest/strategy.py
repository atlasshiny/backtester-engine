import pandas as pd
from .portfolio import Portfolio
from typing import Literal

class Strategy():
    def __init__(self):
        pass

    def on_start(self):
        """Override this for each individual subclass if needed (Optional Method)"""
        raise NotImplementedError("on_start must be implemented by subclass.")

    def on_finish(self):
        """Override this for each individual subclass if needed (Optional Method)"""
        raise NotImplementedError("on_finish must be implemented by subclass.")

    def check_condition(self, event: tuple) -> Literal["BUY", "SELL", "HOLD"]:
        """Override this for each individual subclass"""
        raise NotImplementedError("check_condition must be implemented by subclass.")