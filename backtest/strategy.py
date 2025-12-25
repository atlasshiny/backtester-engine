import pandas as pd

class Strategy():
    def __init__(self):
        pass

    def check_condition(self, event: tuple) -> bool:
        """Override this for each individual subclass"""
        raise NotImplementedError("check_condition must be implemented by subclass.")

    def execute(self, event: tuple):
        """Override this for each individual subclass"""
        raise NotImplementedError("execute must be implemented by subclass.")