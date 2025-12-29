from dataclasses import dataclass
from typing import Literal

@dataclass
class Order:
    """
    Represents a trade order in the backtesting engine.
    Args:
        symbol (str): The asset symbol for the order.
        side (Literal["BUY", "SELL", "HOLD"]): The order side.
        qty (int): The quantity for the order.
        timestamp (int | None): Optional timestamp for the order.
    """
    symbol: str
    side: Literal["BUY", "SELL", "HOLD"]
    qty: int
    timestamp: int | None = None