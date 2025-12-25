import pandas as pd
from .portfolio import Portfolio
from typing import Literal
class Strategy():
    def __init__(self):
        pass

    def check_condition(self, event: tuple) -> Literal["BUY", "SELL", "HOLD"]:
        """Override this for each individual subclass"""
        raise NotImplementedError("check_condition must be implemented by subclass.")