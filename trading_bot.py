"""Feature-based trading bot implementing market making strategies."""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datamodel import OrderDepth, TradingState, Order

logger = logging.getLogger(__name__)

class FeatureBasedTrader:
    def __init__(self):
        """Initialize the trader with feature data and configuration."""
        # Feature data loading
        self.features_df = None
        self._load_features()
        
        # Base date for timestamp conversion with UTC timezone
        self.base_date = pd.Timestamp('2025-04-12', tz='UTC')
        
        # Memoization for timestamp lookup
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
        
        # Risk management parameters
        self.inventory_target = 0       # Target position for each symbol
        self.inventory_limit = 100      # Max absolute position allowed per symbol
        self.notional_limit = 10000    # Max notional value per symbol
        self.inventory_skew_intensity = 1.0  # Increased from 0.2 for stronger position control
        self.position_scale_power = 2.0     # Power for non-linear position scaling
        
        # Price rounding and safety
        self.tick_size = 0.1  # Minimum price increment
        self.max_position_notional = 10000  # Max notional position value
        
    def _load_features(self) -> None:
        """Load pre-generated features from parquet file."""
        try:
            features_path = Path("outputs/trading_model/features.parquet")
            if not features_path.exists():
                logger.warning(f"Features file not found at {features_path}")
                return
                
            self.features_df = pd.read_parquet(features_path)
            
            # Ensure datetime index and UTC timezone
            if not isinstance(self.features_df.index, pd.DatetimeIndex):
                logger.warning("Converting index to datetime")
                self.features_df.index = pd.to_datetime(self.features_df.index)
            
            # Ensure timezone consistency
            if self.features_df.index.tz is None:
                logger.info("Setting index timezone to UTC")
                self.features_df.index = self.features_df.index.tz_localize('UTC')
            elif self.features_df.index.tz.zone != 'UTC':
                logger.info(f"Converting index timezone from {self.features_df.index.tz} to UTC")
                self.features_df.index = self.features_df.index.tz_convert('UTC')
            
            # Sort and validate index
            self.features_df.sort_index(inplace=True)
            
            # Check for duplicate indices
            if self.features_df.index.duplicated().any():
                logger.warning("Found duplicate timestamps in features - keeping last value")
                self.features_df = self.features_df[~self.features_df.index.duplicated(keep='last')]
            
            # Validate required columns
            required_cols = ['composite_signal', 'regime_volatility', 'regime_trend']
            missing_cols = [col for col in required_cols if col not in self.features_df.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
            
            # Log feature data info
            logger.info(f"Loaded features data with shape {self.features_df.shape}")
            logger.info(f"Features time range: {self.features_df.index.min()} to {self.features_df.index.max()}")
            logger.info(f"Features index timezone: {self.features_df.index.tz}")
            logger.info(f"Available columns: {', '.join(self.features_df.columns)}")
            
            # Validate value ranges
            if 'composite_signal' in self.features_df.columns:
                signal_range = self.features_df['composite_signal'].agg(['min', 'max'])
                logger.info(f"Composite signal range: {signal_range['min']:.1f} to {signal_range['max']:.1f}")
                
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
            logger.error("Features DataFrame not loaded")
            return None
            
        # Check memoized result
        if timestamp == self._last_timestamp and self._last_features is not None:
            return self._last_features
            
        try:
            # Convert timestamp to UTC datetime
            current_timestamp = self.base_date + pd.Timedelta(seconds=timestamp)
            if current_timestamp.tz is None:
                current_timestamp = current_timestamp.tz_localize('UTC')
            
            logger.info(f"Looking up features for timestamp: {current_timestamp} (raw: {timestamp})")
            logger.info(f"Features index timezone: {self.features_df.index.tz}")
            
            # Use asof to get most recent features
            features = self.features_df.asof(current_timestamp)
            
            if features is not None:
                # Log the found timestamp and feature values
                logger.info(f"Found features at timestamp: {features.name}")
                logger.info(f"Composite signal: {features.get('composite_signal', 'NOT_FOUND')}")
                logger.info(f"Feature age: {(current_timestamp - features.name).total_seconds():.1f} seconds")
                
                # Validate feature freshness (no older than 5 minutes)
                feature_age = (current_timestamp - features.name).total_seconds()
                if feature_age > 300:  # 5 minutes
                    logger.warning(f"Features too old: {feature_age:.1f} seconds")
                    features = None
            else:
                logger.warning(f"No features found for timestamp {current_timestamp}")
            
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
        
    def _calculate_reference_price(self, order_depth: OrderDepth, symbol: str) -> Optional[float]:
        """Calculate reference price from order book.
        
        Args:
            order_depth: Order book for the symbol
            symbol: Trading symbol
            
        Returns:
            Reference price (usually mid-price) or None if insufficient liquidity
        """
        try:
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            if best_bid is None or best_ask is None:
                logger.warning(f"Insufficient liquidity for {symbol} - missing bid or ask")
                return None
                
            return (best_bid + best_ask) / 2
            
        except Exception as e:
            logger.error(f"Error calculating reference price for {symbol}: {str(e)}")
            return None
    
    def _calculate_adjusted_spread(self, latest_features: Optional[pd.Series]) -> float:
        """Calculate spread based on market regime.
        
        Args:
            latest_features: Current market features
            
        Returns:
            Adjusted spread value
        """
        # Start with base spread
        spread_multiplier = 1.0
        
        if latest_features is not None:
            # Adjust based on HMM regime
            hmm_regime = latest_features.get('hmm_regime', 0)
            if hmm_regime == 0:  # Quiet state
                spread_multiplier = 0.8  # Tighter spread in quiet markets
            elif hmm_regime == 1:  # Active state
                spread_multiplier = 1.5  # Wider spread in volatile markets
            
            # Additional adjustments based on volatility regime
            vol_regime = latest_features.get('regime_volatility', 'medium')
            if vol_regime == 'high':
                spread_multiplier *= 1.2
            elif vol_regime == 'low':
                spread_multiplier *= 0.9
        
        adjusted_spread = max(self.base_spread * spread_multiplier, self.min_spread)
        return adjusted_spread
    
    def _calculate_signal_skewed_volumes(self, latest_features: Optional[pd.Series]) -> Tuple[int, int]:
        """Calculate base volumes adjusted by market signals.
        
        Args:
            latest_features: Current market features
            
        Returns:
            Tuple of (bid_volume, ask_volume)
        """
        # Get composite signal with default of 0 (neutral)
        composite_signal = latest_features.get('composite_signal', 0.0) if latest_features is not None else 0.0
        return self._calculate_skewed_volumes(composite_signal)
    
    def _apply_inventory_adjustment(
        self,
        symbol_position: int,
        bid_volume: int,
        ask_volume: int,
        ref_price: float
    ) -> Tuple[int, int]:
        """Adjust volumes based on current inventory position.
        
        Args:
            symbol_position: Current position in the symbol
            bid_volume: Initial bid volume
            ask_volume: Initial ask volume
            ref_price: Current reference price for notional value calculation
            
        Returns:
            Tuple of adjusted (bid_volume, ask_volume)
        """
        # Calculate position deviation from target
        position_deviation = symbol_position - self.inventory_target
        
        # Calculate adjustment factor (-1 to 1) based on position relative to limits
        adjustment_factor = (position_deviation / self.inventory_limit) * self.inventory_skew_intensity
        adjustment_factor = max(min(adjustment_factor, 1.0), -1.0)
        
        # Check notional position value
        notional_position = abs(symbol_position * ref_price)
        if notional_position > self.max_position_notional:
            logger.warning(f"Position notional {notional_position:.0f} exceeds limit {self.max_position_notional}")
            # Reduce volumes more aggressively
            adjustment_factor *= 1.5
        
        # Adjust volumes - reduce buy volume when long, sell volume when short
        if adjustment_factor > 0:  # Long position
            bid_adj = 1 - adjustment_factor
            ask_adj = 1 + adjustment_factor
        else:  # Short position
            bid_adj = 1 - adjustment_factor
            ask_adj = 1 + adjustment_factor
        
        # Apply adjustments and ensure minimum volumes
        final_bid = max(1, int(round(bid_volume * bid_adj)))
        final_ask = max(1, int(round(ask_volume * ask_adj)))
        
        return final_bid, final_ask
    
    def _generate_orders(
        self,
        symbol: str,
        bid_price: float,
        ask_price: float,
        bid_volume: int,
        ask_volume: int,
        current_position: int
    ) -> List[Order]:
        """Generate buy and sell orders with proper rounding and notional limit enforcement.
        
        Args:
            symbol: Trading symbol
            bid_price: Bid quote price
            ask_price: Ask quote price
            bid_volume: Bid quote volume
            ask_volume: Ask quote volume
            current_position: Current position in the symbol
            
        Returns:
            List of Order objects
        """
        # Round prices to nearest tick
        bid_price = round(bid_price / self.tick_size) * self.tick_size
        ask_price = round(ask_price / self.tick_size) * self.tick_size
        
        # Check potential notional values and adjust volumes if needed
        potential_bid_position = current_position + bid_volume
        potential_ask_position = current_position - ask_volume  # Subtract because ask_volume is positive here
        
        bid_notional = abs(potential_bid_position * bid_price)
        ask_notional = abs(potential_ask_position * ask_price)
        
        # Adjust bid volume if it would breach notional limit
        if bid_notional > self.notional_limit:
            # Calculate maximum allowable volume
            max_bid_volume = int((self.notional_limit / abs(bid_price)) - abs(current_position))
            if max_bid_volume <= 0:
                bid_volume = 0
                logger.warning(f"{symbol}: Bid volume set to 0 - potential notional {bid_notional:.0f} would exceed limit {self.notional_limit}")
            else:
                bid_volume = min(bid_volume, max_bid_volume)
                logger.warning(f"{symbol}: Reduced bid volume to {bid_volume} to respect notional limit")
        
        # Adjust ask volume if it would breach notional limit
        if ask_notional > self.notional_limit:
            # Calculate maximum allowable volume
            max_ask_volume = int((self.notional_limit / abs(ask_price)) - abs(current_position))
            if max_ask_volume <= 0:
                ask_volume = 0
                logger.warning(f"{symbol}: Ask volume set to 0 - potential notional {ask_notional:.0f} would exceed limit {self.notional_limit}")
            else:
                ask_volume = min(ask_volume, max_ask_volume)
                logger.warning(f"{symbol}: Reduced ask volume to {ask_volume} to respect notional limit")
        
        # Ensure minimum volumes (only if we're placing orders at all)
        if bid_volume > 0:
            bid_volume = max(self.min_order_size, bid_volume)
        if ask_volume > 0:
            ask_volume = max(self.min_order_size, ask_volume)
        
        return [
            Order(symbol, bid_price, bid_volume),      # Buy order
            Order(symbol, ask_price, -ask_volume)     # Sell order (negative volume)
        ]
    
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
            mid_price = self._calculate_reference_price(order_depth, product)
            if mid_price is None:
                continue
            
            # Determine spread multiplier based on market regime
            adjusted_spread = self._calculate_adjusted_spread(latest_features)
            bid_price = mid_price - adjusted_spread / 2
            ask_price = mid_price + adjusted_spread / 2
            
            # Calculate volumes based on composite signal
            bid_volume, ask_volume = self._calculate_signal_skewed_volumes(latest_features)
            
            # Apply inventory adjustment using correct position attribute
            symbol_position = state.position.get(product, 0)
            bid_volume, ask_volume = self._apply_inventory_adjustment(symbol_position, bid_volume, ask_volume, mid_price)
            
            # Log state before generating orders
            notional_value = abs(symbol_position * mid_price)
            logger.info(
                f"{product} - Mid: {mid_price:.1f}, "
                f"Spread: {adjusted_spread:.2f}, "
                f"Signal: {latest_features.get('composite_signal', 0.0):.1f}, "
                f"Volumes [B/A]: {bid_volume}/{ask_volume}, "
                f"Position: {symbol_position} (Notional: {notional_value:.0f}), "
                f"Quoting {bid_price:.1f}/{ask_price:.1f}"
            )
            
            # Generate and place orders with current position for notional checks
            orders[product] = self._generate_orders(
                product, 
                bid_price, 
                ask_price, 
                bid_volume, 
                ask_volume,
                symbol_position
            )
        
        return orders, 0, "FEATURE_BASED_TRADER_V1"