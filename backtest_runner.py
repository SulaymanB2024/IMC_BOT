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
    
    # Convert timestamps to datetime with UTC timezone
    base_date = pd.Timestamp('2025-04-12', tz='UTC')
    price_data['dt'] = base_date + pd.to_timedelta(price_data['timestamp'], unit='s')
    price_data.set_index('dt', inplace=True)
    trade_data['dt'] = base_date + pd.to_timedelta(trade_data['timestamp'], unit='s')
    trade_data.set_index('dt', inplace=True)
    
    # Sort by index
    price_data.sort_index(inplace=True)
    trade_data.sort_index(inplace=True)
    
    # Log time range info
    logger.info(f"Price data time range: {price_data.index.min()} to {price_data.index.max()}")
    logger.info(f"Trade data time range: {trade_data.index.min()} to {trade_data.index.max()}")
    
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

def load_data_range(zip_path: str, start_day: int, end_day: int, symbol: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate market data for a range of days.
    
    Args:
        zip_path: Path to data zip file
        start_day: First day to load (inclusive)
        end_day: Last day to load (inclusive)
        symbol: Optional symbol to filter data
        
    Returns:
        Tuple of (price_data, trade_data) DataFrames
    """
    all_price_data = []
    all_trade_data = []
    
    for day in range(start_day, end_day + 1):
        price_data, trade_data = load_market_data(zip_path, day, symbol)
        all_price_data.append(price_data)
        all_trade_data.append(trade_data)
        logger.info(f"Loaded data for day {day}")
    
    combined_price_data = pd.concat(all_price_data, axis=0, ignore_index=True)
    combined_trade_data = pd.concat(all_trade_data, axis=0, ignore_index=True)
    
    # Sort by timestamp
    combined_price_data.sort_values('timestamp', inplace=True)
    combined_trade_data.sort_values('timestamp', inplace=True)
    
    logger.info(f"Combined data: {len(combined_price_data)} price records, {len(combined_trade_data)} trade records")
    return combined_price_data, combined_trade_data

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
    positions: Dict[str, int],
    cash: float
) -> TradingState:
    """Create TradingState object for a specific timestamp.
    
    Args:
        timestamp: Current timestamp
        market_data: Price data DataFrame
        positions: Current positions dictionary
        cash: Current cash balance
        
    Returns:
        Populated TradingState object
    """
    # Get market data for this timestamp by matching the index
    target_dt = pd.Timestamp('2025-04-12', tz='UTC') + pd.Timedelta(seconds=timestamp)
    current_market = market_data[market_data.index == target_dt]
    
    # Create order depths for each product
    order_depths = {}
    for _, row in current_market.iterrows():
        product = row['product']
        order_depths[product] = create_order_depth(row)
    
    return TradingState(
        timestamp=timestamp,
        order_depths=order_depths,
        market_trades={},  # Empty for this version
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
    """Calculate PnL per symbol using last available prices."""
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
        
        # Handle edge cases for cost basis calculation
        if old_position == 0:
            # New position, cost basis is trade price
            cost_basis[product] = price
        else:
            old_cost = cost_basis[product]
            new_position = old_position + quantity
            
            if new_position == 0:
                # Position flattened - cost basis becomes 0
                cost_basis[product] = 0
            else:
                # Regular case - weighted average for cost basis
                total_cost = old_cost * old_position + price * quantity
                # Avoid division by zero with epsilon
                cost_basis[product] = total_cost / (abs(new_position) + 1e-10)
            
            # Calculate realized PnL if position direction changes
            if old_position * new_position <= 0:  # Direction changed or flattened
                # Calculate realized PnL only on the amount that closed
                closed_quantity = min(abs(old_position), abs(quantity))
                realized_pnl = (price - old_cost) * closed_quantity
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
        
        # Only calculate unrealized PnL for non-zero positions
        if position != 0:
            unrealized_pnl = position * (last_price - cost)
            pnl[f'{product}_unrealized'] = unrealized_pnl
        else:
            pnl[f'{product}_unrealized'] = 0.0
    
    # Calculate totals
    pnl['total_unrealized'] = sum(v for k, v in pnl.items() if k.endswith('_unrealized'))
    pnl['total'] = pnl['cash'] + pnl['realized'] + pnl['total_unrealized']
    
    return pnl

def calculate_daily_pnl(trade_log: List[dict], timestamps: List[int]) -> pd.Series:
    """Calculate daily PnL from trade log.
    
    Args:
        trade_log: List of trade dictionaries with timestamps and PnL info
        timestamps: List of timestamps from the simulation
        
    Returns:
        pandas Series with daily PnL values
    """
    if not trade_log:
        return pd.Series()
        
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trade_log)
    trades_df['dt'] = pd.Timestamp('2025-04-12') + pd.to_timedelta(trades_df['timestamp'], unit='s')
    trades_df['date'] = trades_df['dt'].dt.date
    
    # Calculate PnL per trade
    trades_df['trade_pnl'] = -trades_df['price'] * trades_df['quantity']  # Negative because buy reduces cash
    
    # Group by date and sum
    daily_pnl = trades_df.groupby('date')['trade_pnl'].sum()
    return daily_pnl

def calculate_performance_metrics(daily_pnl: pd.Series, trade_log: List[dict]) -> Dict[str, float]:
    """Calculate various trading performance metrics.
    
    Args:
        daily_pnl: Series of daily PnL values
        trade_log: List of executed trades
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Skip metrics if no data
    if daily_pnl.empty:
        return {'warning': 'No trades executed - metrics unavailable'}
    
    # Basic PnL metrics
    metrics['total_pnl'] = daily_pnl.sum()
    metrics['daily_mean_pnl'] = daily_pnl.mean()
    metrics['daily_std_pnl'] = daily_pnl.std()
    
    # Risk-free rate (assuming 0 for simplicity)
    risk_free_rate = 0.0
    
    # Sharpe Ratio (annualized)
    if metrics['daily_std_pnl'] > 0:
        sharpe = np.sqrt(252) * (metrics['daily_mean_pnl'] - risk_free_rate) / metrics['daily_std_pnl']
        metrics['sharpe_ratio'] = sharpe
    else:
        metrics['sharpe_ratio'] = np.nan
    
    # Maximum Drawdown
    cumulative = daily_pnl.cumsum()
    running_max = cumulative.expanding().max()
    drawdowns = cumulative - running_max
    metrics['max_drawdown'] = drawdowns.min()
    
    # Win Rate and Profit Factor
    trades_df = pd.DataFrame(trade_log)
    trades_df['trade_pnl'] = -trades_df['price'] * trades_df['quantity']
    winning_trades = (trades_df['trade_pnl'] > 0).sum()
    total_trades = len(trades_df)
    metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
    
    # Profit Factor (absolute ratio of gains to losses)
    gains = trades_df[trades_df['trade_pnl'] > 0]['trade_pnl'].sum()
    losses = abs(trades_df[trades_df['trade_pnl'] < 0]['trade_pnl'].sum())
    metrics['profit_factor'] = gains / losses if losses != 0 else np.inf
    
    # Sortino Ratio (using only downside deviation)
    downside_returns = daily_pnl[daily_pnl < 0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.mean(downside_returns**2))
        metrics['sortino_ratio'] = np.sqrt(252) * (metrics['daily_mean_pnl'] - risk_free_rate) / downside_std
    else:
        metrics['sortino_ratio'] = np.inf
    
    # Total Return (assuming initial capital of 100,000 for percentage calculation)
    initial_capital = 100000
    metrics['total_return_pct'] = (metrics['total_pnl'] / initial_capital) * 100
    
    return metrics

def print_metrics_report(metrics: Dict[str, float]) -> None:
    """Print formatted performance metrics report.
    
    Args:
        metrics: Dictionary of calculated performance metrics
    """
    logger.info("\n=== Performance Metrics ===")
    
    # Format metrics for display
    logger.info("Risk-Adjusted Returns:")
    logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):,.2f}")
    logger.info(f"Sortino Ratio: {metrics.get('sortino_ratio', 'N/A'):,.2f}")
    
    logger.info("\nReturns:")
    logger.info(f"Total PnL: ${metrics.get('total_pnl', 0):,.2f}")
    logger.info(f"Total Return: {metrics.get('total_return_pct', 0):,.2f}%")
    logger.info(f"Average Daily PnL: ${metrics.get('daily_mean_pnl', 0):,.2f}")
    logger.info(f"Daily PnL Std Dev: ${metrics.get('daily_std_pnl', 0):,.2f}")
    
    logger.info("\nRisk Metrics:")
    logger.info(f"Maximum Drawdown: ${metrics.get('max_drawdown', 0):,.2f}")
    
    logger.info("\nTrade Statistics:")
    logger.info(f"Win Rate: {metrics.get('win_rate', 0)*100:,.1f}%")
    logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):,.2f}")

def process_orders(
    orders: Dict[str, List[Order]],
    market_data: pd.DataFrame,
    timestamp: int,
    positions: Dict[str, int]
) -> List[Trade]:
    """Process orders against market data to generate trades.
    
    Args:
        orders: Dictionary of orders by symbol
        market_data: Market data DataFrame
        timestamp: Current timestamp
        positions: Current positions dictionary
        
    Returns:
        List of executed trades
    """
    executed_trades = []
    target_dt = pd.Timestamp('2025-04-12', tz='UTC') + pd.Timedelta(seconds=timestamp)
    
    for symbol, symbol_orders in orders.items():
        # Get current market data for this symbol
        symbol_data = market_data[
            (market_data.index == target_dt) &
            (market_data['product'] == symbol)
        ]
        
        if symbol_data.empty:
            logger.warning(f"No market data found for {symbol} at {target_dt}")
            continue
        
        # Get best bid/ask
        bid_price = symbol_data['bid_price_1'].iloc[0]
        ask_price = symbol_data['ask_price_1'].iloc[0]
        bid_volume = symbol_data['bid_volume_1'].iloc[0]
        ask_volume = symbol_data['ask_volume_1'].iloc[0]
        
        for order in symbol_orders:
            # Skip invalid orders
            if order.quantity == 0:
                continue
                
            # Check if order can be executed
            if order.quantity > 0:  # Buy order
                if order.price >= ask_price and abs(order.quantity) <= ask_volume:
                    # Execute buy at ask price
                    executed_trades.append(Trade(
                        symbol=symbol,
                        price=ask_price,
                        quantity=order.quantity,
                        timestamp=timestamp
                    ))
            else:  # Sell order
                if order.price <= bid_price and abs(order.quantity) <= bid_volume:
                    # Execute sell at bid price
                    executed_trades.append(Trade(
                        symbol=symbol,
                        price=bid_price,
                        quantity=order.quantity,
                        timestamp=timestamp
                    ))
    
    return executed_trades

def calculate_metrics(trades: List[Trade], final_positions: Dict[str, int], final_cash: float) -> Dict:
    """Calculate trading performance metrics.
    
    Args:
        trades: List of executed trades
        final_positions: Final positions by symbol
        final_cash: Final cash balance
        
    Returns:
        Dictionary of performance metrics
    """
    if not trades:
        return {}
        
    # Convert trades to DataFrame for analysis
    trade_data = []
    for t in trades:
        trade_data.append({
            'timestamp': t.timestamp,
            'symbol': t.symbol,
            'price': t.price,
            'quantity': t.quantity,
            'value': t.price * t.quantity
        })
    df = pd.DataFrame(trade_data)
    
    # Calculate daily returns
    df['date'] = pd.to_datetime('2025-04-12') + pd.to_timedelta(df['timestamp'], unit='s')
    df['date'] = df['date'].dt.date
    daily_pnl = df.groupby('date')['value'].sum()
    
    # Basic metrics
    metrics = {}
    metrics['total_trades'] = len(trades)
    metrics['total_volume'] = df['quantity'].abs().sum()
    
    # Return metrics
    if len(daily_pnl) > 0:
        metrics['total_return'] = -daily_pnl.sum() / 100000  # Assuming 100k initial capital
        metrics['daily_returns_mean'] = daily_pnl.mean()
        metrics['return_std'] = daily_pnl.std() / 100000 if len(daily_pnl) > 1 else 0
        
        # Risk metrics (annualized)
        if metrics['return_std'] > 0:
            metrics['sharpe_ratio'] = np.sqrt(252) * (metrics['daily_returns_mean'] / daily_pnl.std())
            
            # Sortino ratio using downside deviation
            downside_returns = daily_pnl[daily_pnl < 0]
            if len(downside_returns) > 0:
                downside_std = np.sqrt(np.mean(downside_returns**2))
                metrics['sortino_ratio'] = np.sqrt(252) * (metrics['daily_returns_mean'] / downside_std)
            else:
                metrics['sortino_ratio'] = float('inf')
        else:
            metrics['sharpe_ratio'] = float('inf') if metrics['daily_returns_mean'] > 0 else float('-inf')
            metrics['sortino_ratio'] = float('inf') if metrics['daily_returns_mean'] > 0 else float('-inf')
        
        # Maximum drawdown
        cumulative = (-daily_pnl).cumsum()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / 100000  # Convert to percentage
        metrics['max_drawdown'] = drawdown.min()
        
        # Win rate and profit factor
        df['pnl'] = -df['value']  # Negative because value is cost
        winning_trades = (df['pnl'] > 0).sum()
        metrics['win_rate'] = winning_trades / len(trades)
        
        gains = df[df['pnl'] > 0]['pnl'].sum()
        losses = abs(df[df['pnl'] < 0]['pnl'].sum())
        metrics['profit_factor'] = gains / losses if losses != 0 else float('inf')
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Backtest the FeatureBasedTrader")
    parser.add_argument("--start_day", type=int, required=True, 
                       help="First day to simulate (inclusive)")
    parser.add_argument("--end_day", type=int, required=True,
                       help="Last day to simulate (inclusive)")
    parser.add_argument("--symbol", default="ALL", 
                       help="Symbol to trade (default: ALL)")
    parser.add_argument("--zip_file", default="round-2-island-data-bottle.zip",
                       help="Path to data zip file")
    parser.add_argument("--features_file", default="outputs/trading_model/features.parquet",
                       help="Path to pre-computed features file")
    args = parser.parse_args()
    
    # Load market data for date range
    logger.info(f"Loading market data for days {args.start_day} to {args.end_day}")
    market_data, trade_data = load_data_range(
        args.zip_file,
        args.start_day,
        args.end_day,
        args.symbol
    )
    
    # Initialize trader
    trader = FeatureBasedTrader()
    
    if trader.features_df is None:
        logger.error("Failed to load features - check features file path")
        return
        
    # Filter features to match date range (using UTC timestamps)
    base_date = pd.Timestamp('2025-04-12', tz='UTC')
    start_ts = base_date + pd.Timedelta(days=args.start_day)
    end_ts = base_date + pd.Timedelta(days=args.end_day + 1)  # Add 1 to make inclusive
    mask = (trader.features_df.index >= start_ts) & (trader.features_df.index < end_ts)
    trader.features_df = trader.features_df[mask]
    logger.info(f"Filtered features to {len(trader.features_df)} records")
    
    # Initialize simulation state
    positions = {}
    cash = 0
    trade_log = []
    
    # Get unique timestamps
    timestamps = sorted(market_data['timestamp'].unique())
    
    # Main simulation loop
    logger.info("Starting simulation")
    for i, t in enumerate(timestamps[:-1]):  # Exclude last timestamp as we need t+1 data
        state = create_trading_state(t, market_data, positions, cash)
        
        try:
            orders, _, _ = trader.run(state)
            
            # Process orders and update positions/cash
            trades = process_orders(orders, market_data, t, positions)
            for trade in trades:
                # Update position
                symbol = trade.symbol
                positions[symbol] = positions.get(symbol, 0) + trade.quantity
                cash -= trade.quantity * trade.price
                trade_log.append(trade)
                
        except Exception as e:
            logger.error(f"Error in trading loop at timestamp {t}: {str(e)}")
            continue
    
    # Calculate and print metrics
    print_metrics_report(trade_log, positions, cash)

def print_metrics_report(trades: List[Trade], final_positions: Dict[str, int], final_cash: float):
    """Print performance metrics with safe handling of N/A values."""
    if not trades:
        logger.info("No trades executed during simulation")
        return
        
    metrics = calculate_metrics(trades, final_positions, final_cash)
    
    # Helper function to format metric values
    def format_metric(value, format_spec='.2f'):
        if isinstance(value, (int, float)) and not pd.isna(value):
            if '%' in format_spec:
                return f"{value*100:{format_spec}}%"
            return f"{value:{format_spec}}"
        return str(value)  # Return string representation for non-numeric values
    
    # Print metrics with safe formatting
    logger.info("\n=== Performance Metrics ===")
    logger.info(f"Total Trades Executed: {len(trades)}")
    logger.info(f"Sharpe Ratio: {format_metric(metrics.get('sharpe_ratio', 'N/A'), ',.2f')}")
    logger.info(f"Sortino Ratio: {format_metric(metrics.get('sortino_ratio', 'N/A'), ',.2f')}")
    logger.info(f"Total Return: {format_metric(metrics.get('total_return', 'N/A'), '.2%')}")
    logger.info(f"Daily Return Std Dev: {format_metric(metrics.get('return_std', 'N/A'), '.2%')}")
    logger.info(f"Max Drawdown: {format_metric(metrics.get('max_drawdown', 'N/A'), '.2%')}")
    logger.info(f"Profit Factor: {format_metric(metrics.get('profit_factor', 'N/A'), ',.2f')}")
    logger.info(f"Final Cash: {format_metric(final_cash, ',.2f')}")
    
    # Print final positions
    logger.info("\n=== Final Positions ===")
    for symbol, pos in final_positions.items():
        logger.info(f"{symbol}: {pos:,d}")

if __name__ == "__main__":
    main()