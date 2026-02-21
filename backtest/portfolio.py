"""backtest.portfolio

Portfolio/account state.

The portfolio tracks:
- Cash balance
- Open positions (per symbol)
- Equity curve (value_history)

Valuation
---------
Value history is updated using either:
- A single float price (legacy / single-asset style), or
- A {symbol: price} mapping for multi-asset valuation.
"""

from backtest.position import Position
from .array_utils import select_array_module, ensure_array

class Portfolio:
    def __init__(self, initial_cash: float):
        """
        Create a new Portfolio.

        Parameters
        ----------
        initial_cash:
            Starting cash balance.

        Attributes
        ----------
        positions:
            Dict mapping symbol -> Position.
        value_history:
            List of portfolio equity values over time (as updated by the broker).
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        # Keep a Python list buffer for fast appends during the run.
        # Convert to an `xp` array on-demand when analytics need it.
        self.value_history = []

    def add_position(self, symbol: str, qty: int, price: float):
        """
        Add to (or create) a position.

        If the symbol already exists, increases quantity and updates average cost
        using a simple weighted-average approach.

        Parameters
        ----------
        symbol:
            Asset identifier.
        qty:
            Quantity to add.
        price:
            Executed unit price.
        """
        if symbol in self.positions:
            self.positions[symbol].add(qty, price)
        else:
            self.positions[symbol] = Position(symbol=symbol, qty=qty, avg_price=price)

    def remove_position(self, symbol: str, qty: int):
        """
        Remove quantity from a position.

        If qty equals the full position size, the position is deleted.

        Notes
        -----
        This is a long-only position model; it does not support negative quantities.
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
        Append a new portfolio equity value to value_history.

        Parameters
        ----------
        current_price:
            - float: use the same price for all positions (single-asset style)
            - dict: mapping {symbol: price} for multi-asset valuation

        Notes
        -----
        If a symbol is missing from the provided price mapping, this method falls
        back to the position's avg_price.
        """
        if isinstance(current_price, dict):
            value = self.cash
            for sym, pos in self.positions.items():
                px = current_price.get(sym)
                if px is None:
                    px = pos.avg_price
                value += pos.qty * px
        else:
            value = self.cash + sum(pos.qty * current_price for pos in self.positions.values())

        # Append to the list buffer (fast amortized O(1)).
        self.value_history.append(float(value))

    def value_history_xp(self, prefer_gpu: object = "auto", gpu_min_size: int = 10000):
        """Return the value history as an array in the selected array module (np or cp).

        This performs a single conversion of the in-memory list to an `xp` array.
        """
        xp = select_array_module(prefer_gpu, len(self.value_history), gpu_min_size)
        return ensure_array(self.value_history, xp)

    def portfolio_value_snapshot(self, price: float) -> float:
        """
        Compute total portfolio equity using a single price for all symbols.

        This is mainly a convenience for single-asset experiments.
        """
        if not self.positions:
            return self.cash
        else:
            return self.cash + sum(pos.qty * price for pos in self.positions.values())

    def reset(self):
        """Reset portfolio to its initial state.
        
        Clears cash deviations, all positions, and value history.
        Useful for running multiple backtests sequentially without creating new objects.
        """
        self.cash = self.initial_cash
        self.positions = {}
        self.value_history = []