"""Feature engineering module for generating trading model features."""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import ta  # Technical Analysis library

from .config import get_config

logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical analysis indicators."""
    result = df.copy()
    config = get_config()
    
    if 'price' not in df.columns:
        logger.error("Price column not found in data")
        return df
    
    try:
        # RSI
        if config['feature_config']['technical_indicators']['rsi']['window']:
            window = config['feature_config']['technical_indicators']['rsi']['window']
            result['rsi'] = ta.momentum.rsi(df['price'], window=window)
        
        # MACD
        if config['feature_config']['technical_indicators']['macd']:
            macd_config = config['feature_config']['technical_indicators']['macd']
            result['macd'] = ta.trend.macd(
                df['price'],
                window_slow=macd_config['slow'],
                window_fast=macd_config['fast'],
                window_sign=macd_config['signal']
            )
            result['macd_signal'] = ta.trend.macd_signal(
                df['price'],
                window_slow=macd_config['slow'],
                window_fast=macd_config['fast'],
                window_sign=macd_config['signal']
            )
            result['macd_diff'] = ta.trend.macd_diff(
                df['price'],
                window_slow=macd_config['slow'],
                window_fast=macd_config['fast'],
                window_sign=macd_config['signal']
            )
        
        # Bollinger Bands
        if config['feature_config']['technical_indicators']['bollinger']:
            bb_config = config['feature_config']['technical_indicators']['bollinger']
            result['bb_upper'] = ta.volatility.bollinger_hband(
                df['price'],
                window=bb_config['window'],
                window_dev=bb_config['num_std']
            )
            result['bb_lower'] = ta.volatility.bollinger_lband(
                df['price'],
                window=bb_config['window'],
                window_dev=bb_config['num_std']
            )
            result['bb_mid'] = ta.volatility.bollinger_mavg(
                df['price'],
                window=bb_config['window']
            )
        
        # Volatility (20-day rolling std)
        result['volatility_20'] = df['price'].rolling(window=20).std()
        
        logger.info("Technical indicators added successfully")
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        
    return result

def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime detection features."""
    result = df.copy()
    config = get_config()
    
    try:
        if config['feature_config']['regime_detection']['volatility']['enabled']:
            vol_config = config['feature_config']['regime_detection']['volatility']
            vol = df['price'].rolling(window=vol_config['window']).std()
            
            # Classify volatility regimes
            result['regime_volatility'] = pd.cut(
                vol,
                bins=[-np.inf, vol_config['thresholds']['low_vol'], 
                      vol_config['thresholds']['high_vol'], np.inf],
                labels=['low', 'medium', 'high']
            )
        
        if config['feature_config']['regime_detection']['trend']['enabled']:
            window = config['feature_config']['regime_detection']['trend']['window']
            ma = df['price'].rolling(window=window).mean()
            
            # Simple trend detection
            result['regime_trend'] = np.where(
                df['price'] > ma * 1.02, 'strong_up',
                np.where(df['price'] > ma * 1.01, 'weak_up',
                np.where(df['price'] < ma * 0.98, 'strong_down',
                np.where(df['price'] < ma * 0.99, 'weak_down', 'neutral')))
            )
            
        logger.info("Market regime features added successfully")
            
    except Exception as e:
        logger.error(f"Error calculating market regime features: {str(e)}")
        
    return result

def generate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals based on technical indicators."""
    result = df.copy()
    config = get_config()
    
    try:
        # MACD signals
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            result['signal_macd_cross'] = np.where(
                (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)),
                1,  # Bullish crossover
                np.where(
                    (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
                    -1,  # Bearish crossover
                    0   # No signal
                )
            )
        
        # RSI signals
        if 'rsi' in df.columns:
            rsi_config = config['feature_config']['signal_generation']['rsi']
            result['signal_rsi'] = np.where(
                df['rsi'] < rsi_config['oversold'],
                1,  # Oversold
                np.where(
                    df['rsi'] > rsi_config['overbought'],
                    -1,  # Overbought
                    0
                )
            )
        
        # Moving average crossover signals
        ma_config = config['feature_config']['signal_generation']['ma_crossover']
        if ma_config['enabled']:
            fast_ma = df['price'].rolling(window=ma_config['fast_ma']).mean()
            slow_ma = df['price'].rolling(window=ma_config['slow_ma']).mean()
            
            result['signal_ma_cross'] = np.where(
                (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1)),
                1,  # Bullish crossover
                np.where(
                    (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1)),
                    -1,  # Bearish crossover
                    0
                )
            )
        
        # Volume surge signals
        vol_config = config['feature_config']['signal_generation']['volume_surge']
        if vol_config['enabled'] and 'volume' in df.columns:
            vol_ma = df['volume'].rolling(window=vol_config['window']).mean()
            result['signal_volume_surge'] = np.where(
                df['volume'] > vol_ma * vol_config['threshold'],
                1,  # Volume surge
                0
            )
        
        # Combine signals into composite score (-100 to +100)
        signal_columns = [col for col in result.columns if col.startswith('signal_')]
        if signal_columns:
            # Weight and normalize the signals
            result['composite_signal'] = (
                result[signal_columns].sum(axis=1) * (100 / (len(signal_columns) * 1.5))
            ).clip(-100, 100)  # Clip to ensure bounds
            
        logger.info("Trading signals generated successfully")
            
    except Exception as e:
        logger.error(f"Error generating trading signals: {str(e)}")
        
    return result

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to generate all features."""
    logger.info("Starting feature generation")
    
    try:
        # Generate features sequentially
        result = df.copy()
        
        # Add technical indicators
        result = add_technical_indicators(result)
        logger.info("Technical indicators added")
        
        # Add market regime features
        result = add_market_regime_features(result)
        logger.info("Market regime features added")
        
        # Generate trading signals
        result = generate_trading_signals(result)
        logger.info("Trading signals generated")
        
        # Forward fill any NaN values from indicators
        result = result.ffill()
        
        # Save features
        output_path = Path('outputs/trading_model/features.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path)
        logger.info(f"Features saved to {output_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Feature generation failed: {str(e)}")
        raise