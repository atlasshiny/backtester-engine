
from typing import Literal, Union, Tuple
from backtest.position import Position

class Portfolio():
    def __init__(self, initial_cash: float, slippage: float = 0.001, commission: float = 0.001, log_hold: bool = False):
        # initialization variables
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage = slippage
        self.commission = commission
        self.log_hold = log_hold

        # state variables
        self.positions = {}
        self.trade_log = []
        self.value_history = []

        pass

    def add_position(self, symbol_or_name: str, amount: int, price: float):
        if symbol_or_name in self.positions:
            pos = self.positions[symbol_or_name]
            pos.add(amount, price)
        else:
            self.positions[symbol_or_name] = Position(symbol=symbol_or_name, qty=amount, avg_price=price)

    def remove_position(self, symbol: str, amount: int, price: float):
        if symbol in self.positions:
            pos = self.positions[symbol]
            if amount > pos.qty:
                print(f"Cannot sell {amount} units of {symbol}; only {pos.qty} available.")
                return
            elif amount == pos.qty:
                del self.positions[symbol]
            else:
                pos.remove(amount)

    def update_value_history(self, current_price):
        value = self.cash + sum(pos.qty * current_price for pos in self.positions.values())
        self.value_history.append(value)

    def log_trade(self, side: Literal["BUY", "SELL", "HOLD"], amount: int, symbol_or_name: str, price: float, commission: float, slippage: float, comment=""):
        trade = {
            "symbol": symbol_or_name,
            "side": side,
            "qty": amount,
            "price": price,
            "commission": commission,
            "slippage": slippage,
            "comment": comment
        }
        self.trade_log.append(trade)

    def execute(self, event: tuple, order):
        price = event.Open
        # Slippage and commission are both percent-based (e.g., 0.001 for 0.1%)
        side = order.side
        symbol = order.symbol
        qty = order.qty
        match side:
            case "BUY":
                exec_price = price * (1 + self.slippage)
                trade_value = exec_price * qty
                commission = trade_value * self.commission
                total_cost = trade_value + commission
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.add_position(symbol_or_name=symbol, amount=qty, price=exec_price)
                    self.log_trade(side=side, amount=qty, symbol_or_name=symbol, price=exec_price, commission=commission, slippage=self.slippage)
                else:
                    self.log_trade(side=side, amount=0, symbol_or_name=symbol, price=None, commission=0.0, slippage=self.slippage, comment="Insufficient Funds")
            case "HOLD":
                if self.log_hold == True:
                    self.log_trade(side=side, amount=qty, symbol_or_name=symbol, price=price, commission=0.0, slippage=0.0, comment="HOLD")
                pass
            case "SELL":
                exec_price = price * (1 - self.slippage)
                trade_value = exec_price * qty
                commission = trade_value * self.commission
                # Check if there is enough to sell
                if symbol in self.positions and self.positions[symbol].qty >= qty:
                    self.cash += trade_value - commission
                    self.remove_position(symbol=symbol, amount=qty, price=exec_price)
                    self.log_trade(side=side, amount=qty, symbol_or_name=symbol, price=exec_price, commission=commission, slippage=self.slippage)
                else:
                    self.log_trade(side=side, amount=0, symbol_or_name=symbol, price=None, commission=0.0, slippage=self.slippage, comment="Insufficient Position")
        self.update_value_history(event.Close)

    def portfolio_value_snapshot(self, event: float) -> float:
        if not self.positions:
            return self.cash
        else:
            return self.cash + sum(pos.qty * event for pos in self.positions.values())