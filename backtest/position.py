"""backtest.position

Lightweight position record.

This module intentionally keeps the position model minimal:
- Long-only quantity tracking
- Average cost basis
- Market value computation given an external price
"""

from dataclasses import dataclass

@dataclass
class Position:
    """Represents an open position in a single symbol."""
    symbol: str
    qty: int
    avg_price: float

    def add(self, qty: int, price: float):
        """
        Increase position size and update average cost basis.

        Uses weighted-average cost:
        avg_price := (old_qty*old_avg + qty*price) / (old_qty + qty)
        """
        total_cost = self.qty * self.avg_price + qty * price
        self.qty += qty
        self.avg_price = total_cost / self.qty

    def remove(self, qty: int):
        """
        Decrease position size.

        Notes
        -----
        This does not realize PnL; realized PnL is computed in analytics by pairing
        buys/sells from the broker trade log.
        """
        assert qty <= self.qty
        self.qty -= qty

    def market_value(self, price: float) -> float:
        """
        Compute current market value.

        Parameters
        ----------
        price:
            Current unit price.
        """
        return self.qty * price