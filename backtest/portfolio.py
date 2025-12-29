from backtest.position import Position

class Portfolio:
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.value_history = []

    def add_position(self, symbol: str, qty: int, price: float):
        if symbol in self.positions:
            self.positions[symbol].add(qty, price)
        else:
            self.positions[symbol] = Position(symbol=symbol, qty=qty, avg_price=price)

    def remove_position(self, symbol: str, qty: int):
        if symbol in self.positions:
            pos = self.positions[symbol]
            if qty > pos.qty:
                print(f"Cannot sell {qty} units of {symbol}; only {pos.qty} available.")
                return
            elif qty == pos.qty:
                del self.positions[symbol]
            else:
                pos.remove(qty)

    def update_value_history(self, current_price):
        value = self.cash + sum(pos.qty * current_price for pos in self.positions.values())
        self.value_history.append(value)

    def portfolio_value_snapshot(self, price: float) -> float:
        if not self.positions:
            return self.cash
        else:
            return self.cash + sum(pos.qty * price for pos in self.positions.values())