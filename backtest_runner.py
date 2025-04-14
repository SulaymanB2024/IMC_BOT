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
    """Load price and trade data for a specific day from the zip archive with caching."""
    # Define cache paths
    cache_dir = Path('data/processed')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    price_cache = cache_dir / f'prices_day_{day}.parquet'
    trade_cache = cache_dir / f'trades_day_{day}.parquet'
    
    # Try loading from cache first
    if price_cache.exists() and trade_cache.exists():
        try:
            price_data = pd.read_parquet(price_cache)
            trade_data = pd.read_parquet(trade_cache)
            logger.info(f"Loaded data from cache for day {day}")
            
            if symbol and symbol != "ALL":
                price_data = price_data[price_data['product'] == symbol]
                trade_data = trade_data[trade_data['symbol'] == symbol]
                
            return price_data, trade_data
        except Exception as e:
            logger.warning(f"Failed to load from cache: {str(e)}")
    
    # Load from zip if cache missing or invalid
    with ZipFile(zip_path) as zf:
        # Load price data efficiently
        price_file = f"round-2-island-data-bottle/prices_round_2_day_{day}.csv"
        price_data = pd.read_csv(
            zf.open(price_file),
            sep=';',
            engine='c',  # Use C engine for better performance
            dtype={  # Specify dtypes to avoid inference
                'timestamp': np.int64,
                'product': str,
                'bid_price_1': np.float32,
                'ask_price_1': np.float32,
                'bid_volume_1': np.int32,
                'ask_volume_1': np.int32
            }
        )
        
        # Load trade data efficiently
        trade_file = f"round-2-island-data-bottle/trades_round_2_day_{day}.csv"
        trade_data = pd.read_csv(
            zf.open(trade_file),
            sep=';',
            engine='c',
            dtype={
                'timestamp': np.int64,
                'symbol': str,
                'price': np.float32,
                'quantity': np.int32,
                'buyer': str,
                'seller': str
            }
        )
    
    logger.info(f"Loaded {len(price_data)} price records, {len(trade_data)} trade records")
    
    # Convert timestamps to datetime
    base_date = pd.Timestamp('2025-04-12')
    price_data['dt'] = base_date + pd.to_timedelta(price_data['timestamp'], unit='s')
    
    # Cache the full data
    try:
        price_data.to_parquet(price_cache, engine='fastparquet', compression='snappy')
        trade_data.to_parquet(trade_cache, engine='fastparquet', compression='snappy')
        logger.info(f"Cached data for day {day}")
    except Exception as e:
        logger.warning(f"Failed to cache data: {str(e)}")
    
    # Filter by symbol if specified
    if symbol and symbol != "ALL":
        price_data = price_data[price_data['product'] == symbol]
        trade_data = trade_data[trade_data['symbol'] == symbol]
        logger.info(f"Filtered data for {symbol}")
    
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
    """Simulate order fills using next timestamp's market data."""
    position_limits = defaultdict(lambda: 100)  # Default 100 position limit per symbol
    
    for product, product_orders in orders.items():
        market_row = next_market_data[next_market_data['product'] == product]
        if market_row.empty:
            logger.warning(f"No market data found for {product} - skipping orders")
            continue
            
        next_bid = market_row['bid_price_1'].iloc[0]
        next_ask = market_row['ask_price_1'].iloc[0]
        
        # Sort orders by price (best prices first)
        buy_orders = sorted([o for o in product_orders if o.quantity > 0], 
                          key=lambda x: x.price, reverse=True)  # Highest price first
        sell_orders = sorted([o for o in product_orders if o.quantity < 0], 
                           key=lambda x: x.price)  # Lowest price first
        
        # Track fills for this product in this iteration
        total_bought = 0
        total_sold = 0
        
        # Process buys
        for order in buy_orders:
            if order.price >= next_ask:  # Willing to pay the ask
                new_position = positions.get(product, 0) + total_bought + order.quantity
                if abs(new_position) > position_limits[product]:
                    logger.warning(f"Order for {product} would exceed position limit - skipping")
                    continue
                    
                total_bought += order.quantity
                cash -= order.quantity * next_ask
                
                trade_log.append({
                    'timestamp': market_row['timestamp'].iloc[0],
                    'product': product,
                    'price': next_ask,
                    'quantity': order.quantity,
                    'side': 'BUY',
                    'remaining_cash': cash
                })
        
        # Process sells
        for order in sell_orders:
            if order.price <= next_bid:  # Willing to hit the bid
                new_position = positions.get(product, 0) + total_sold + order.quantity
                if abs(new_position) > position_limits[product]:
                    logger.warning(f"Order for {product} would exceed position limit - skipping")
                    continue
                    
                total_sold += order.quantity
                cash -= order.quantity * next_bid  # quantity is negative for sells
                
                trade_log.append({
                    'timestamp': market_row['timestamp'].iloc[0],
                    'product': product,
                    'price': next_bid,
                    'quantity': order.quantity,
                    'side': 'SELL',
                    'remaining_cash': cash
                })
        
        # Update position
        if total_bought + total_sold != 0:
            positions[product] = positions.get(product, 0) + total_bought + total_sold
            logger.info(f"Updated {product} position to {positions[product]} after fills")
    
    return positions, cash

def calculate_pnl(
    positions: Dict[str, int],
    market_data: pd.DataFrame,
    cash: float,
    trades: List[dict]
) -> Dict[str, float]:
    """Calculate PnL per symbol using last available prices.
    
    Args:
        positions: Final positions dictionary
        market_data: Complete market data DataFrame
        cash: Final cash balance 
        trades: List of executed trades for realized PnL calculation
        
    Returns:
        Dictionary with unrealized PnL per symbol, realized PnL, and total PnL
    """
    pnl = {'cash': cash, 'realized': 0.0}
    
    # Calculate realized PnL from trades
    position_tracker = defaultdict(int)
    cost_basis = defaultdict(float)
    
    for trade in trades:
        product = trade['product']
        price = trade['price']
        quantity = trade['quantity']
        
        # Update average cost basis
        old_position = position_tracker[product]
        old_cost = cost_basis[product] * old_position
        
        if old_position == 0:
            cost_basis[product] = price
        else:
            # Weighted average for cost basis
            cost_basis[product] = (old_cost + price * quantity) / (old_position + quantity)
            
        # If position direction changes, realize PnL
        if old_position * (old_position + quantity) < 0:  # Direction changed
            realized_pnl = (price - cost_basis[product]) * min(abs(old_position), abs(quantity))
            pnl['realized'] += realized_pnl
            
        position_tracker[product] += quantity
    
    # Calculate unrealized PnL using last prices
    for product in positions.keys():
        product_data = market_data[market_data['product'] == product]
        if product_data.empty:
            logger.warning(f"No market data found for {product} - skipping PnL calculation")
            continue
            
        last_price = product_data['mid_price'].iloc[-1]
        position = positions[product]
        cost = cost_basis[product]
        
        unrealized_pnl = position * (last_price - cost)
        pnl[f'{product}_unrealized'] = unrealized_pnl
        
    # Calculate totals
    pnl['total_unrealized'] = sum(v for k, v in pnl.items() if k.endswith('_unrealized'))
    pnl['total'] = pnl['cash'] + pnl['realized'] + pnl['total_unrealized']
    
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
    pnl = calculate_pnl(positions, market_data, cash, trade_log)
    
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