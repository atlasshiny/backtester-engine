from typing import Literal, Union, Tuple
class Portfolio():
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {} # change to a dict or list eventually
        self.trade_log = []
        self.value_history = []
        pass

    def add_position(self, symbol_or_name: str, amount: int, price: float):
        if symbol_or_name in self.positions:
            pos = self.positions[symbol_or_name]
            # Update average price and amount
            total_cost = pos['amount'] * pos['price'] + amount * price
            total_amount = pos['amount'] + amount
            avg_price = total_cost / total_amount
            self.positions[symbol_or_name] = {'amount': total_amount, 'price': avg_price}
        else:
            self.positions[symbol_or_name] = {'amount': amount, 'price': price}

    def remove_position(self, symbol: str, amount: int, price: float):
        if symbol in self.positions:
            pos = self.positions[symbol]
            if amount > pos['amount']:
                print(f"Cannot sell {amount} units of {symbol}; only {pos['amount']} available.")
                # Optionally, raise an exception or just return
                return
            elif amount == pos['amount']:
                del self.positions[symbol]
            else:
                remaining_amount = pos['amount'] - amount
                self.positions[symbol] = {'amount': remaining_amount, 'price': pos['price']}
        else:
            # print(f"No position found for symbol: {symbol}")
            pass

    def update_value_history(self, current_price):
        value = self.cash + sum(pos['amount'] * current_price for pos in self.positions.values())
        self.value_history.append(value)

    def log_trade(self, side: Literal["BUY", "SELL", "HOLD"], amount: int, symbol_or_name: str, price: int, comment: str):
        trade = {
            "symbol": symbol_or_name,
            "side": side,
            "qty": amount,
            "price": price,
            "comment": comment
        }
        self.trade_log.append(trade)

    def execute(self, event: tuple, action: Union[Tuple[Literal["BUY"], int], Tuple[Literal["SELL"], int], Literal["HOLD"]], symbol_or_name: str):
        price = event[4] # using the "high" of a day's worth of data
        match action[0]:
            case "BUY":
                total_cost = event[4] * action[1]
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.add_position(symbol_or_name=symbol_or_name, amount=action[1], price=price)
                    self.log_trade(side=action[0], amount=action[1], symbol_or_name=symbol_or_name, price=price)
                else: 
                    self.log_trade(side=action[0], amount=0, symbol_or_name=symbol_or_name, price=None, comment="Insufficint Funds")
            case "HOLD":
                self.log_trade(side=action[0], amount=action[1], symbol_or_name=symbol_or_name, price=price)
                pass
            case "SELL":
                amount_to_sell = action[1]
                self.cash += amount_to_sell * price
                self.remove_position(symbol=symbol_or_name, amount=amount_to_sell, price=price)
                self.log_trade(side=action[0], amount=action[1], symbol_or_name=symbol_or_name, price=price)
        self.update_value_history(event[4])

    def portfolio_value_snapshot(self, event: float) -> float:
        if not self.positions:
            return self.cash
        else:
            return self.cash + sum(pos['amount'] * event for pos in self.positions.values())