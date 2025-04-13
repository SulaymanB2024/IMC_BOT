#!/usr/bin/env python3
"""Backtesting script for FeatureBasedTrader using historical market data."""
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile
from collections import defaultdict

# Note: datamodel is provided by the IMC environment
from datamodel import OrderDepth, TradingState, Order, Trade
from trading_bot import FeatureBasedTrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_market_data(zip_path: str, day: int, symbol: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load price and trade data for a specific day from the zip archive."""
    with ZipFile(zip_path) as zf:
        # Load price data
        price_file = f"round-2-island-data-bottle/prices_round_2_day_{day}.csv"
        price_data = pd.read_csv(zf.open(price_file), sep=';')
        logger.info(f"Loaded {len(price_data)} price records")
        
        # Load trade data
        trade_file = f"round-2-island-data-bottle/trades_round_2_day_{day}.csv"
        trade_data = pd.read_csv(zf.open(trade_file), sep=';')
        logger.info(f"Loaded {len(trade_data)} trade records")
    
    # Filter by symbol if specified
    if symbol and symbol != "ALL":
        price_data = price_data[price_data['product'] == symbol]
        trade_data = trade_data[trade_data['symbol'] == symbol]
        logger.info(f"After filtering for {symbol}: {len(price_data)} price records, {len(trade_data)} trade records")
        
    # Convert timestamps to datetime for compatibility with features
    base_date = pd.Timestamp('2025-04-12')
    price_data['dt'] = base_date + pd.to_timedelta(price_data['timestamp'].astype(int), unit='s')
    
    logger.info(f"Price data time range: {price_data['dt'].min()} to {price_data['dt'].max()}")
    return price_data, trade_data

def create_order_depth(row: pd.Series) -> OrderDepth:
    """Create OrderDepth object from a row of market data.
    
    Args:
        row: Row from price data DataFrame containing bid/ask prices and volumes
        
    Returns:
        Populated OrderDepth object
    """
    order_depth = OrderDepth()
    
    # Add bid orders (using only level 1 for simplicity)
    if not pd.isna(row['bid_price_1']):
        order_depth.buy_orders[row['bid_price_1']] = row['bid_volume_1']
        
    # Add ask orders (using only level 1 for simplicity)
    if not pd.isna(row['ask_price_1']):
        order_depth.sell_orders[row['ask_price_1']] = row['ask_volume_1']
        
    return order_depth

def create_trading_state(
    timestamp: int,
    market_data: pd.DataFrame,
    trade_data: pd.DataFrame,
    positions: Dict[str, int]
) -> TradingState:
    """Create TradingState object for a specific timestamp.
    
    Args:
        timestamp: Current timestamp
        market_data: Price data DataFrame
        trade_data: Trade data DataFrame
        positions: Current positions dictionary
        
    Returns:
        Populated TradingState object
    """
    # Get market data for this timestamp
    current_market = market_data[market_data['timestamp'] == timestamp]
    
    # Create order depths for each product
    order_depths = {}
    for _, row in current_market.iterrows():
        product = row['product']
        order_depths[product] = create_order_depth(row)
    
    # Get trades that occurred at this timestamp
    current_trades = trade_data[trade_data['timestamp'] == timestamp]
    market_trades = {}
    for product in order_depths.keys():
        product_trades = current_trades[current_trades['symbol'] == product]
        market_trades[product] = [
            Trade(trade['buyer'], trade['seller'], trade['price'], trade['quantity'])
            for _, trade in product_trades.iterrows()
        ]
    
    return TradingState(
        timestamp=timestamp,
        order_depths=order_depths,
        market_trades=market_trades,
        position=positions.copy(),
        observations={},  # Empty for backtest
        traderData=""    # Empty for backtest
    )

def simulate_fills(
    orders: Dict[str, List[Order]],
    next_market_data: pd.DataFrame,
    positions: Dict[str, int],
    cash: float,
    trade_log: List[dict]
) -> Tuple[Dict[str, int], float]:
    """Simulate order fills using next timestamp's market data.
    
    Args:
        orders: Dictionary of orders from trader
        next_market_data: Market data for next timestamp
        positions: Current positions dictionary
        cash: Current cash balance
        trade_log: List to record trade executions
        
    Returns:
        Tuple of (updated positions dict, updated cash)
    """
    position_limits = defaultdict(lambda: 100)  # Default 100 position limit per symbol
    
    for product, product_orders in orders.items():
        market_row = next_market_data[next_market_data['product'] == product]
        if market_row.empty:
            continue
            
        next_bid = market_row['bid_price_1'].iloc[0]
        next_ask = market_row['ask_price_1'].iloc[0]
        
        for order in product_orders:
            # Skip if would exceed position limits
            if abs(positions.get(product, 0) + order.quantity) > position_limits[product]:
                logger.warning(f"Order for {product} would exceed position limit - skipping")
                continue
                
            # Simulate fill logic
            if order.quantity > 0:  # Buy order
                if order.price >= next_ask:
                    positions[product] = positions.get(product, 0) + order.quantity
                    cash -= order.quantity * next_ask
                    trade_log.append({
                        'timestamp': market_row['timestamp'].iloc[0],
                        'product': product,
                        'price': next_ask,
                        'quantity': order.quantity,
                        'side': 'BUY'
                    })
            else:  # Sell order
                if order.price <= next_bid:
                    positions[product] = positions.get(product, 0) + order.quantity  # quantity is negative
                    cash -= order.quantity * next_bid  # negative quantity makes this a cash addition
                    trade_log.append({
                        'timestamp': market_row['timestamp'].iloc[0],
                        'product': product,
                        'price': next_bid,
                        'quantity': order.quantity,
                        'side': 'SELL'
                    })
                    
    return positions, cash

def calculate_pnl(
    positions: Dict[str, int],
    market_data: pd.DataFrame,
    cash: float
) -> Dict[str, float]:
    """Calculate PnL per symbol using last available prices.
    
    Args:
        positions: Final positions dictionary
        market_data: Complete market data DataFrame
        cash: Final cash balance
        
    Returns:
        Dictionary of PnL per symbol
    """
    pnl = {'cash': cash}
    
    # Get last price for each product
    last_prices = {}
    for product in positions.keys():
        product_data = market_data[market_data['product'] == product]
        if not product_data.empty:
            last_prices[product] = product_data['mid_price'].iloc[-1]
            
    # Calculate PnL per symbol
    for product, position in positions.items():
        if product in last_prices:
            pnl[product] = position * last_prices[product]
            
    # Add total
    pnl['total'] = sum(pnl.values())
    
    return pnl

def main():
    parser = argparse.ArgumentParser(description="Backtest the FeatureBasedTrader")
    parser.add_argument("--day", type=int, required=True, help="Trading day to simulate (-1, 0, 1)")
    parser.add_argument("--symbol", default="ALL", help="Symbol to trade (default: ALL)")
    parser.add_argument(
        "--zip_file",
        default="round-2-island-data-bottle.zip",
        help="Path to data zip file"
    )
    parser.add_argument(
        "--features_file",
        default="outputs/trading_model/features.parquet",
        help="Path to pre-computed features file"
    )
    args = parser.parse_args()
    
    # Load market data
    logger.info(f"Loading market data for day {args.day}")
    market_data, trade_data = load_market_data(args.zip_file, args.day, args.symbol)
    
    # Initialize trader
    trader = FeatureBasedTrader()
    if trader.features_df is None:
        logger.error("Failed to load features - check features file path")
        return
    
    # Initialize simulation state
    positions = {}
    cash = 0
    trade_log = []
    
    # Get unique timestamps
    timestamps = sorted(market_data['timestamp'].unique())
    
    # Main simulation loop
    logger.info("Starting simulation")
    for i, t in enumerate(timestamps[:-1]):  # Exclude last timestamp as we need t+1 data
        # Create state object
        state = create_trading_state(t, market_data, trade_data, positions)
        
        # Get trader's orders
        try:
            orders_dict, _, _ = trader.run(state)
        except Exception as e:
            logger.error(f"Error running trader at t={t}: {str(e)}")
            continue
        
        # Simulate fills using t+1 market data
        next_market = market_data[market_data['timestamp'] == timestamps[i + 1]]
        positions, cash = simulate_fills(orders_dict, next_market, positions, cash, trade_log)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(timestamps)} timestamps")
    
    # Calculate final PnL
    pnl = calculate_pnl(positions, market_data, cash)
    
    # Print results
    logger.info("\n=== Backtest Results ===")
    logger.info(f"Processed {len(timestamps)} timestamps")
    logger.info(f"Total trades executed: {len(trade_log)}")
    logger.info("\nFinal positions:")
    for symbol, pos in positions.items():
        logger.info(f"{symbol}: {pos}")
    logger.info("\nPnL breakdown:")
    for key, value in pnl.items():
        logger.info(f"{key}: {value:,.2f}")

if __name__ == "__main__":
    main()