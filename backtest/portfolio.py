from typing import Literal

class Portfolio():
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.position = {} # change to a dict or list eventually
        self.trade_log = []
        pass

    def execute(self, event: tuple, action: Literal["BUY", "SELL", "HOLD"]):
        match action:
            case "BUY":
                if self.cash > event[3]: # check if there is enough money to buy (this is checking the "high" of the day to make the trade)
                    self.cash = self.cash - event[3]
                    # add the payload to be added to self.position
            case "HOLD":
                pass
            case "SELL":
                if self.position >= 1:
                    self.cash += 105 # arbitrary value for now
                    # add the logic to remove positions from self.position
