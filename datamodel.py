"""Mock datamodel module for backtesting IMC trading bot."""
from typing import Dict, List

class Order:
    def __init__(self, symbol: str, price: float, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[float, int] = {}
        self.sell_orders: Dict[float, int] = {}

class Trade:
    def __init__(self, buyer: str, seller: str, price: float, quantity: int):
        self.buyer = buyer
        self.seller = seller
        self.price = price
        self.quantity = quantity

class TradingState:
    def __init__(
        self,
        timestamp: int,
        order_depths: Dict[str, OrderDepth],
        market_trades: Dict[str, List[Trade]],
        position: Dict[str, int],
        observations: Dict[str, int],
        traderData: str
    ):
        self.timestamp = timestamp
        self.order_depths = order_depths
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        self.traderData = traderData