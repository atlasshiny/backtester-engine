class Portfolio():
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.position = 0 # change to a dict or list eventually
        self.trade_log = []
        pass

    def execute(self, event: tuple):
        """Override this for each individual subclass"""
        raise NotImplementedError("execute must be implemented by subclass.")
