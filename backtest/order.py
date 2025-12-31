"""backtest.order

Order representation.

Orders are produced by strategies and executed by the broker. The engine
schedules orders using a next-bar model (the order produced at time t is
executed on the next bar for that symbol).
"""

from dataclasses import dataclass
from typing import Literal

@dataclass
class Order:
    """
    Trade intent produced by a Strategy.

    Attributes
    ----------
    symbol:
        Asset identifier. In single-asset mode, the engine injects Symbol='SINGLE'
        into the dataset so strategies can always set a symbol.
    side:
        'BUY' | 'SELL' | 'HOLD'.
    qty:
        Quantity requested. The broker may log qty=0 if an order is rejected or
        unfilled.
    order_type:
        'MARKET' or 'LIMIT'.
    limit_price:
        Used for LIMIT orders.
    timestamp:
        Optional timestamp to attach to logs/analytics; if None, broker derives it
        from the event.
    """
    symbol: str
    side: Literal["BUY", "SELL", "HOLD"]
    qty: int
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: float | None = None
    timestamp: int | None = None