from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float

    def add(self, qty: int, price: float):
        """
        Add quantity to the position and update the average price.
        Args:
            qty (int): Quantity to add.
            price (float): Price at which to add.
        """
        total_cost = self.qty * self.avg_price + qty * price
        self.qty += qty
        self.avg_price = total_cost / self.qty

    def remove(self, qty: int):
        """
        Remove quantity from the position.
        Args:
            qty (int): Quantity to remove.
        """
        assert qty <= self.qty
        self.qty -= qty

    def market_value(self, price: float) -> float:
        """
        Calculate the market value of the position at a given price.
        Args:
            price (float): The current market price.
        Returns:
            float: The market value of the position.
        """
        return self.qty * price