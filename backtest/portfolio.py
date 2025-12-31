from backtest.position import Position

class Portfolio:
    def __init__(self, initial_cash: float):
        """
        Initialize a Portfolio instance.
        Args:
            initial_cash (float): The starting cash for the portfolio.
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.value_history = []

    def add_position(self, symbol: str, qty: int, price: float):
        """
        Add to or create a position in the portfolio.
        If the symbol already exists, increases the quantity and updates the average price.
        Args:
            symbol (str): The asset symbol.
            qty (int): Quantity to add.
            price (float): Price at which the quantity is added.
        """
        if symbol in self.positions:
            self.positions[symbol].add(qty, price)
        else:
            self.positions[symbol] = Position(symbol=symbol, qty=qty, avg_price=price)

    def remove_position(self, symbol: str, qty: int):
        """
        Remove quantity from a position, or delete the position if fully sold.
        Args:
            symbol (str): The asset symbol.
            qty (int): Quantity to remove.
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            if qty > pos.qty:
                print(f"Cannot sell {qty} units of {symbol}; only {pos.qty} available.")
                return
            elif qty == pos.qty:
                del self.positions[symbol]
            else:
                pos.remove(qty)

    def update_value_history(self, current_price: float | dict):
        """
        Update the portfolio's value history with the current total value.
        Args:
            current_price (float | dict):
                - float: use the same price for all positions (single-asset mode).
                - dict: mapping {symbol: price} for multi-asset valuation.
        """
        if isinstance(current_price, dict):
            value = self.cash
            for sym, pos in self.positions.items():
                px = current_price.get(sym)
                if px is None:
                    # If we don't have a price yet, fall back to avg_price
                    px = pos.avg_price
                value += pos.qty * px
        else:
            value = self.cash + sum(pos.qty * current_price for pos in self.positions.values())

        self.value_history.append(value)

    def portfolio_value_snapshot(self, price: float) -> float:
        """
        Get a snapshot of the portfolio's total value at a given price.
        Args:
            price (float): The price to use for all positions.
        Returns:
            float: The total portfolio value (cash + market value of all positions).
        """
        if not self.positions:
            return self.cash
        else:
            return self.cash + sum(pos.qty * price for pos in self.positions.values())