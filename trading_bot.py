"""Feature-based trading bot implementing market making strategies."""
import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datamodel import OrderDepth, TradingState, Order

logger = logging.getLogger(__name__)

class FeatureBasedTrader:
    def __init__(self):
        """Initialize the trader with feature data."""
        self.features_df = None
        self._load_features()
        
        # Add memoization for timestamp lookup
        self._last_timestamp = None
        self._last_features = None
        
        # Market making parameters with safeguards
        self.base_spread = 0.5  # Default spread
        self.min_spread = 0.1   # Minimum allowed spread
        self.base_order_size = 10
        self.min_order_size = 1
        
        # Signal thresholds with validation
        self.strong_signal_threshold = min(max(40.0, 20.0), 80.0)  # Clamp between 20-80
        self.weak_signal_threshold = min(max(10.0, 5.0), 30.0)    # Clamp between 5-30
        
        # Volume multipliers with validation
        self.strong_volume_multiplier = min(max(1.5, 1.1), 2.0)  # Clamp between 1.1-2.0
        self.weak_volume_multiplier = min(max(1.2, 1.1), 1.5)    # Clamp between 1.1-1.5
        
    def _load_features(self) -> None:
        """Load pre-generated features from parquet file."""
        try:
            features_path = Path("outputs/trading_model/features.parquet")
            if not features_path.exists():
                logger.warning(f"Features file not found at {features_path}")
                return
                
            self.features_df = pd.read_parquet(features_path)
            
            # Ensure datetime index
            if not isinstance(self.features_df.index, pd.DatetimeIndex):
                logger.warning("Converting index to datetime")
                self.features_df.index = pd.to_datetime(self.features_df.index)
            
            self.features_df.sort_index(inplace=True)
            
            # Validate required columns
            required_cols = ['composite_signal', 'regime_volatility', 'regime_trend']
            missing_cols = [col for col in required_cols if col not in self.features_df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
            
            logger.info(f"Loaded features data with shape {self.features_df.shape}")
            logger.info(f"Features time range: {self.features_df.index.min()} to {self.features_df.index.max()}")
            
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            self.features_df = None
            
    def _get_features_for_timestamp(self, timestamp: int) -> Optional[pd.Series]:
        """Get features for a specific timestamp with memoization.
        
        Args:
            timestamp: Current timestamp from TradingState
            
        Returns:
            Series containing features or None if not found
        """
        if self.features_df is None:
            return None
            
        # Check memoized result
        if timestamp == self._last_timestamp and self._last_features is not None:
            return self._last_features
            
        try:
            # Convert timestamp to datetime
            current_timestamp = pd.Timestamp('2025-04-12') + pd.Timedelta(seconds=timestamp)
            
            # Use asof to get most recent features (robust to missing timestamps)
            features = self.features_df.asof(current_timestamp)
            
            # Validate feature freshness (no older than 5 minutes)
            if features is not None:
                feature_age = (current_timestamp - features.name).total_seconds()
                if feature_age > 300:  # 5 minutes
                    logger.warning(f"Features too old: {feature_age:.1f} seconds")
                    features = None
            
            # Update memoization
            self._last_timestamp = timestamp
            self._last_features = features
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting features: {str(e)}")
            return None
            
    def _calculate_skewed_volumes(self, composite_signal: float) -> tuple[int, int]:
        """Calculate bid and ask volumes based on composite signal strength.
        
        Args:
            composite_signal: The composite trading signal (-100 to +100)
            
        Returns:
            tuple: (bid_volume, ask_volume) as integers
        """
        # Ensure signal is within bounds
        composite_signal = max(min(composite_signal, 100), -100)
        
        # Start with base volume
        base_vol = max(self.base_order_size, self.min_order_size)
        
        # Strong bullish
        if composite_signal >= self.strong_signal_threshold:
            bid_volume = base_vol * self.strong_volume_multiplier
            ask_volume = base_vol / self.strong_volume_multiplier
            
        # Weak bullish    
        elif composite_signal >= self.weak_signal_threshold:
            bid_volume = base_vol * self.weak_volume_multiplier
            ask_volume = base_vol / self.weak_volume_multiplier
            
        # Strong bearish    
        elif composite_signal <= -self.strong_signal_threshold:
            bid_volume = base_vol / self.strong_volume_multiplier
            ask_volume = base_vol * self.strong_volume_multiplier
            
        # Weak bearish    
        elif composite_signal <= -self.weak_signal_threshold:
            bid_volume = base_vol / self.weak_volume_multiplier
            ask_volume = base_vol * self.weak_volume_multiplier
            
        # Neutral
        else:
            bid_volume = base_vol
            ask_volume = base_vol
            
        # Ensure volumes are positive integers
        return max(1, int(round(bid_volume))), max(1, int(round(ask_volume)))
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Main trading logic implementation.
        
        Args:
            state (TradingState): Current market state including order books
            
        Returns:
            tuple: (
                Dict[str, List[Order]]: Orders to be placed for each symbol
                int: Number of conversions
                str: Updated trader state data
            )
        """
        # Initialize empty orders dictionary
        orders = {}
        
        # Get current timestamp
        current_timestamp = state.timestamp
        latest_features = self._get_features_for_timestamp(current_timestamp)
        
        # Process each product
        for product in state.order_depths.keys():
            order_depth: OrderDepth = state.order_depths[product]
            orders[product] = []
            
            # Calculate reference price from order book
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            if best_bid is None or best_ask is None:
                logger.warning(f"Insufficient liquidity for {product} - missing bid or ask")
                continue
                
            mid_price = (best_bid + best_ask) / 2
            
            # Determine spread multiplier based on market regime
            spread_multiplier = 1.0  # Default multiplier
            if latest_features is not None:
                hmm_regime = latest_features.get('hmm_regime', 0)
                if hmm_regime == 0:  # Quiet state
                    spread_multiplier = 0.8  # Tighter spread in quiet markets
                elif hmm_regime == 1:  # Active state
                    spread_multiplier = 1.5  # Wider spread in volatile markets
            
            # Calculate adjusted spread and quote prices
            adjusted_spread = self.base_spread * spread_multiplier
            bid_price = mid_price - adjusted_spread / 2
            ask_price = mid_price + adjusted_spread / 2
            
            # Round prices to nearest tick (assuming tick size of 0.1)
            bid_price = round(bid_price, 1)
            ask_price = round(ask_price, 1)
            
            # Calculate volumes based on composite signal
            composite_signal = latest_features.get('composite_signal', 0.0) if latest_features is not None else 0.0
            bid_volume, ask_volume = self._calculate_skewed_volumes(composite_signal)
            
            logger.info(
                f"{product} - Mid: {mid_price:.1f}, "
                f"Spread: {adjusted_spread:.2f} ({spread_multiplier:.1f}x), "
                f"Signal: {composite_signal:.1f}, "
                f"Volumes [B/A]: {bid_volume}/{ask_volume}, "
                f"Quoting {bid_price:.1f}/{ask_price:.1f}"
            )
            
            # Place orders with skewed volumes
            orders[product] = [
                Order(product, bid_price, bid_volume),     # Buy order
                Order(product, ask_price, -ask_volume)    # Sell order (negative volume for sells)
            ]
        
        return orders, 0, "FEATURE_BASED_TRADER_V1"