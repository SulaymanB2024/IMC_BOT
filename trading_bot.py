"""Feature-based trading bot implementing market making strategies."""
import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order

logger = logging.getLogger(__name__)

class FeatureBasedTrader:
    def __init__(self):
        """Initialize the trader with feature data."""
        self.features_df = None
        self._load_features()
        
        # Market making parameters
        self.base_spread = 0.5  # Default spread
        self.base_order_size = 10  # Default order size
        
        # Signal thresholds for volume skewing
        self.strong_signal_threshold = 40.0
        self.weak_signal_threshold = 10.0
        
        # Volume skew multipliers
        self.strong_volume_multiplier = 1.5
        self.weak_volume_multiplier = 1.2
        
    def _load_features(self) -> None:
        """Load pre-generated features from parquet file."""
        try:
            features_path = Path("outputs/trading_model/features.parquet")
            if not features_path.exists():
                logger.warning(f"Features file not found at {features_path}")
                return
                
            self.features_df = pd.read_parquet(features_path)
            self.features_df.index = pd.to_datetime(self.features_df.index)
            self.features_df.sort_index(inplace=True)
            
            logger.info(f"Successfully loaded features data with shape {self.features_df.shape}")
            logger.info(f"Features time range: {self.features_df.index.min()} to {self.features_df.index.max()}")
            
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            self.features_df = None
            
    def _calculate_skewed_volumes(self, composite_signal: float) -> tuple[int, int]:
        """Calculate bid and ask volumes based on composite signal strength.
        
        Args:
            composite_signal: The composite trading signal (-100 to +100)
            
        Returns:
            tuple: (bid_volume, ask_volume) as integers
        """
        # Start with base volume
        base_vol = self.base_order_size
        
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
        current_timestamp = pd.Timestamp(state.timestamp)
        latest_features = None
        
        # Attempt to look up current features
        if self.features_df is not None:
            try:
                # Get most recent features relative to current timestamp
                latest_features = self.features_df.loc[self.features_df.index.asof(current_timestamp)]
                
                # Log key market state indicators
                logger.info(
                    f"T={current_timestamp}: Features found - "
                    f"HMM={latest_features.get('hmm_regime_label', 'N/A')}, "
                    f"Signal={latest_features.get('composite_signal', 'N/A'):.1f}, "
                    f"Volatility={latest_features.get('regime_volatility', 'N/A')}, "
                    f"Trend={latest_features.get('regime_trend', 'N/A')}, "
                    f"Anomaly={latest_features.get('anomaly_flag_iforest', 'N/A')}"
                )
                
            except KeyError as e:
                logger.warning(f"No features found for timestamp {current_timestamp}: {str(e)}")
            except Exception as e:
                logger.error(f"Error looking up features: {str(e)}")
        
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