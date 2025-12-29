
from typing import Literal

class Broker:
    def __init__(self, portfolio, slippage: float = 0.001, commission: float = 0.001, log_hold: bool = False):
        self.portfolio = portfolio
        self.slippage = slippage
        self.commission = commission
        self.log_hold = log_hold
        self.trade_log = []

    def log_trade(self, side: Literal["BUY", "SELL", "HOLD"], qty: int, symbol: str, price: float, commission: float, slippage: float, comment=""):
        trade = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "commission": commission,
            "slippage": slippage,
            "comment": comment
        }
        self.trade_log.append(trade)

    def execute(self, event: tuple, order):
        price = event.Open
        side = order.side
        symbol = order.symbol
        qty = order.qty
        match side:
            case "BUY":
                exec_price = price * (1 + self.slippage)
                trade_value = exec_price * qty
                commission = trade_value * self.commission
                total_cost = trade_value + commission
                if self.portfolio.cash >= total_cost:
                    self.portfolio.cash -= total_cost
                    self.portfolio.add_position(symbol=symbol, qty=qty, price=exec_price)
                    self.log_trade(side=side, qty=qty, symbol=symbol, price=exec_price, commission=commission, slippage=self.slippage)
                else:
                    self.log_trade(side=side, qty=0, symbol=symbol, price=None, commission=0.0, slippage=self.slippage, comment="Insufficient Funds")
            case "HOLD":
                if self.log_hold:
                    self.log_trade(side=side, qty=qty, symbol=symbol, price=price, commission=0.0, slippage=0.0, comment="HOLD")
            case "SELL":
                exec_price = price * (1 - self.slippage)
                trade_value = exec_price * qty
                commission = trade_value * self.commission
                if symbol in self.portfolio.positions and self.portfolio.positions[symbol].qty >= qty:
                    self.portfolio.cash += trade_value - commission
                    self.portfolio.remove_position(symbol=symbol, qty=qty)
                    self.log_trade(side=side, qty=qty, symbol=symbol, price=exec_price, commission=commission, slippage=self.slippage)
                else:
                    self.log_trade(side=side, qty=0, symbol=symbol, price=None, commission=0.0, slippage=self.slippage, comment="Insufficient Position")
        self.portfolio.update_value_history(event.Close)

