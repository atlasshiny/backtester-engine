from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float

    def add(self, qty: int, price: float):
        total_cost = self.qty * self.avg_price + qty * price
        self.qty += qty
        self.avg_price = total_cost / self.qty

    def remove(self, qty: int):
        assert qty <= self.qty
        self.qty -= qty

    def market_value(self, price: float) -> float:
        return self.qty * price