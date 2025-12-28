from dataclasses import dataclass
from typing import Literal

@dataclass
class Order:
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int
    timestamp: int | None = None