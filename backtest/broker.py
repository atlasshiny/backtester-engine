"""backtest.broker

Broker/execution simulator.

Responsibilities
----------------
- Apply a simplified fill model for MARKET and LIMIT orders on OHLC bars.
- Apply slippage and commission.
- Update the Portfolio's cash/positions.
- Maintain a trade log and a last-known price map for valuation.

Execution model
---------------
The engine schedules orders as "pending" and executes them on the next bar for the
same symbol (next-bar execution). The broker itself does not schedule; it only
executes given an event and an order.
"""

from typing import Literal

class Broker:
    def __init__(self, portfolio, slippage: float = 0.001, commission: float = 0.001, log_hold: bool = False):
        """
        Create a Broker.

        Parameters
        ----------
        portfolio:
            Portfolio instance to mutate.
        slippage:
            Proportional slippage applied to execution price.
            BUY uses (1 + slippage); SELL uses (1 - slippage).
        commission:
            Proportional commission charged on trade notional.
        log_hold:
            If True, HOLD signals are written into the trade log.

        Notes
        -----
        The broker keeps a last-known Close price per symbol (updated on every bar)
        and passes that to the portfolio for multi-asset valuation.
        """
        self.portfolio = portfolio
        self.slippage = slippage
        self.commission = commission
        self.log_hold = log_hold
        self.trade_log = []
        self.last_prices = {}

    def log_trade(self, side: Literal["BUY", "SELL", "HOLD"], qty: int, symbol: str, price: float, commission: float, slippage: float, comment="", timestamp=None, order_type: str | None = None, limit_price: float | None = None):
        """
        Append a record to trade_log.

        The trade log is a list of dictionaries with these keys:
        - timestamp: datetime-like or raw value from the event/order
        - symbol: str
        - side: 'BUY' | 'SELL' | 'HOLD'
        - qty: int (0 for rejected/unfilled)
        - price: float | None
        - commission: float
        - slippage: float
        - order_type: 'MARKET' | 'LIMIT' | None
        - limit_price: float | None
        - comment: str

        This structure is intentionally simple so PerformanceAnalytics can convert
        it directly into a DataFrame.
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
        Execute an Order on a given event/bar.

        Parameters
        ----------
        event:
            A row from DataFrame.itertuples(), expected to have Open/High/Low/Close,
            and optionally Date/Symbol.
        order:
            Order-like object with side, symbol, qty, and optional order_type,
            limit_price, timestamp.

        Fill model
        ----------
        - MARKET: executes on event.Open with slippage applied.
        - LIMIT:
            - BUY fills if event.Low <= limit_price
            - SELL fills if event.High >= limit_price
            If unfilled, logs a qty=0 record and does not change positions.

        Timestamping
        -----------
        The broker attaches a timestamp to each log entry. Resolution order:
        order.timestamp -> event.Date -> event.Index.
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

