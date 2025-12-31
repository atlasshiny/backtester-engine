
from typing import Literal

class Broker:
    def __init__(self, portfolio, slippage: float = 0.001, commission: float = 0.001, log_hold: bool = False):
        """
        Initialize a Broker instance.
        Args:
            portfolio (Portfolio): The portfolio to manage.
            slippage (float): Slippage rate as a decimal (e.g., 0.001 for 0.1%).
            commission (float): Commission rate as a decimal.
            log_hold (bool): Whether to log HOLD trades.
        """
        self.portfolio = portfolio
        self.slippage = slippage
        self.commission = commission
        self.log_hold = log_hold
        self.trade_log = []
        self.last_prices = {}

    def log_trade(self, side: Literal["BUY", "SELL", "HOLD"], qty: int, symbol: str, price: float, commission: float, slippage: float, comment="", timestamp=None, order_type: str | None = None, limit_price: float | None = None):
        """
        Log a trade in the broker's trade log.
        Args:
            side (str): Trade side ('BUY', 'SELL', 'HOLD').
            qty (int): Quantity traded.
            symbol (str): Asset symbol.
            price (float): Execution price.
            commission (float): Commission paid.
            slippage (float): Slippage applied.
            comment (str): Optional comment.
            timestamp: Bar timestamp/date associated with this log entry.
            order_type (str | None): Order type (e.g., 'MARKET', 'LIMIT').
            limit_price (float | None): Limit price if applicable.
        """
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "commission": commission,
            "slippage": slippage,
            "order_type": order_type,
            "limit_price": limit_price,
            "comment": comment
        }
        self.trade_log.append(trade)

    def execute(self, event: tuple, order):
        """
        Execute an order, update the portfolio, and log the trade.
        Supports MARKET and LIMIT orders.
        Args:
            event (tuple): The current market event/bar.
            order: The order object with side, symbol, qty, order_type, limit_price.
        """
        price = event.Open
        price_low = event.Low
        price_high = event.High
        side = order.side
        symbol = order.symbol
        qty = order.qty
        order_type = getattr(order, 'order_type', 'MARKET')
        limit_price = getattr(order, 'limit_price', None)
        timestamp = getattr(order, 'timestamp', None)
        if timestamp is None:
            timestamp = getattr(event, 'Date', None)
        if timestamp is None:
            timestamp = getattr(event, 'Index', None)

        # Update last known close for valuation (works for multi-asset)
        if symbol is not None:
            self.last_prices[symbol] = event.Close

        # LIMIT order logic (use event.Low for buy, event.High for sell)
        if order_type == "LIMIT":
            if side == "BUY":
                # Buy limit: fill if event.Low <= limit_price
                if limit_price is None or price_low > limit_price:
                    self.log_trade(side=side, qty=0, symbol=symbol, price=None, commission=0.0, slippage=0.0, comment="Unfilled Limit Order (BUY)", timestamp=timestamp, order_type=order_type, limit_price=limit_price)
                    self.portfolio.update_value_history(self.last_prices)
                    return
            if side == "SELL":
                # Sell limit: fill if event.High >= limit_price
                if limit_price is None or price_high < limit_price:
                    self.log_trade(side=side, qty=0, symbol=symbol, price=None, commission=0.0, slippage=0.0, comment="Unfilled Limit Order (SELL)", timestamp=timestamp, order_type=order_type, limit_price=limit_price)
                    self.portfolio.update_value_history(self.last_prices)
                    return

        match side:
            case "BUY":
                exec_price = price * (1 + self.slippage)
                trade_value = exec_price * qty
                commission = trade_value * self.commission
                total_cost = trade_value + commission
                if self.portfolio.cash >= total_cost:
                    self.portfolio.cash -= total_cost
                    self.portfolio.add_position(symbol=symbol, qty=qty, price=exec_price)
                    self.log_trade(side=side, qty=qty, symbol=symbol, price=exec_price, commission=commission, slippage=self.slippage, timestamp=timestamp, order_type=order_type, limit_price=limit_price)
                else:
                    self.log_trade(side=side, qty=0, symbol=symbol, price=None, commission=0.0, slippage=self.slippage, comment="Insufficient Funds", timestamp=timestamp, order_type=order_type, limit_price=limit_price)
            case "HOLD":
                if self.log_hold:
                    self.log_trade(side=side, qty=qty, symbol=symbol, price=price, commission=0.0, slippage=0.0, comment="HOLD", timestamp=timestamp, order_type=order_type, limit_price=limit_price)
            case "SELL":
                exec_price = price * (1 - self.slippage)
                trade_value = exec_price * qty
                commission = trade_value * self.commission
                if symbol in self.portfolio.positions and self.portfolio.positions[symbol].qty >= qty:
                    self.portfolio.cash += trade_value - commission
                    self.portfolio.remove_position(symbol=symbol, qty=qty)
                    self.log_trade(side=side, qty=qty, symbol=symbol, price=exec_price, commission=commission, slippage=self.slippage, timestamp=timestamp, order_type=order_type, limit_price=limit_price)
                else:
                    self.log_trade(side=side, qty=0, symbol=symbol, price=None, commission=0.0, slippage=self.slippage, comment="Insufficient Position", timestamp=timestamp, order_type=order_type, limit_price=limit_price)
        self.portfolio.update_value_history(self.last_prices)

