"""Feature engineering module for generating trading signals and indicators."""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict
import ta  # Technical Analysis library
import ta.trend as trend
import ta.momentum as momentum
import ta.volatility as volatility

from .config import get_config

logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """Add technical indicators to the DataFrame."""
    logger.info("Calculating technical indicators")
    config = get_config().feature_config['technical_indicators']
    
    result = df.copy()
    n_periods = len(df)
    
    # Adjust window sizes for small datasets
    max_window = max(n_periods // 5, 2)  # At least 2 periods
    
    # Simple Moving Averages with adjusted windows
    sma_windows = [min(w, max_window) for w in config['sma_windows']]
    for window in sma_windows:
        result[f'sma_{window}'] = ta.trend.sma_indicator(
            result[price_col], 
            window=window,
            fillna=True
        )
    
    # Exponential Moving Averages
    ema_windows = [min(w, max_window) for w in config['ema_windows']]
    for window in ema_windows:
        result[f'ema_{window}'] = ta.trend.ema_indicator(
            result[price_col], 
            window=window,
            fillna=True
        )
    
    # RSI with adjusted period
    rsi_period = min(config['rsi_period'], max_window)
    result['rsi'] = ta.momentum.rsi(
        result[price_col], 
        window=rsi_period,
        fillna=True
    )
    
    # MACD with adjusted periods
    macd_slow = min(config['macd']['slow_period'], max_window)
    macd_fast = min(config['macd']['fast_period'], macd_slow - 1)
    macd_signal = min(config['macd']['signal_period'], macd_fast - 1)
    
    macd = ta.trend.MACD(
        result[price_col],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
        fillna=True
    )
    result['macd'] = macd.macd()
    result['macd_signal'] = macd.macd_signal()
    result['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands with adjusted window
    bb_window = min(config['bollinger_bands']['window'], max_window)
    bb = ta.volatility.BollingerBands(
        result[price_col],
        window=bb_window,
        window_dev=config['bollinger_bands']['num_std'],
        fillna=True
    )
    result['bb_upper'] = bb.bollinger_hband()
    result['bb_lower'] = bb.bollinger_lband()
    result['bb_mid'] = bb.bollinger_mavg()
    
    logger.info(f"Technical indicators calculated with adjusted windows (max_window={max_window})")
    return result

def add_volatility_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """Add volatility-based features to the DataFrame."""
    logger.info("Calculating volatility features")
    config = get_config().feature_config['volatility']
    
    result = df.copy()
    n_periods = len(df)
    max_window = max(n_periods // 5, 2)  # At least 2 periods
    
    # Calculate returns and ensure no infinite values
    returns = np.log(result[price_col]).diff()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Rolling volatility with adjusted windows
    std_windows = [min(w, max_window) for w in config['std_windows']]
    for window in std_windows:
        result[f'volatility_{window}'] = returns.rolling(
            window=window,
            min_periods=1
        ).std().fillna(method='bfill')
    
    # Volatility regime using shorter window for small datasets
    vol_window = min(20, max_window)
    vol_20 = returns.rolling(window=vol_window, min_periods=1).std()
    
    # Simple regime classification for very small datasets
    if n_periods < 10:
        result['vol_regime'] = 'medium'  # Default regime for very small datasets
    else:
        try:
            vol_unique = vol_20.dropna().unique()
            if len(vol_unique) >= 3:
                # Use quantiles if we have enough unique values
                result['vol_regime'] = pd.qcut(
                    vol_20.fillna(vol_20.mean()),
                    q=3,
                    labels=['low', 'medium', 'high'],
                    duplicates='drop'
                )
            else:
                # Simple threshold-based classification
                vol_mean = vol_20.mean()
                vol_std = vol_20.std()
                
                # Create bins with exactly one fewer label than bin edges
                bins = [-np.inf, vol_mean - vol_std, vol_mean + vol_std, np.inf]  # 4 edges
                labels = ['low', 'medium', 'high']  # 3 labels
                
                result['vol_regime'] = pd.cut(
                    vol_20,
                    bins=bins,
                    labels=labels,
                    include_lowest=True
                )
        except Exception as e:
            logger.warning(f"Volatility regime classification failed: {str(e)}")
            result['vol_regime'] = 'medium'
    
    logger.info(f"Volatility features calculated with adjusted windows (max_window={max_window})")
    return result

def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features to the DataFrame.
    
    Args:
        df: DataFrame containing trading data
        
    Returns:
        DataFrame with added lagged features
    """
    logger.info("Calculating lagged features")
    config = get_config().feature_config['lags']
    
    result = df.copy()
    
    # Price lags
    price_cols = df.filter(like='price').columns
    for col in price_cols:
        for lag in config['price_lags']:
            result[f'{col}_lag_{lag}'] = result[col].shift(lag)
    
    # Volume lags
    volume_cols = df.filter(like='volume').columns
    for col in volume_cols:
        for lag in config['volume_lags']:
            result[f'{col}_lag_{lag}'] = result[col].shift(lag)
    
    logger.info("Lagged features calculated successfully")
    return result

def add_anomaly_detection_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """Add anomaly detection features to the DataFrame.
    
    Args:
        df: DataFrame containing trading data
        price_col: Name of the price column to use for calculations
        
    Returns:
        DataFrame with added anomaly features
    """
    logger.info("Calculating anomaly detection features")
    config = get_config().feature_config['anomaly_detection']
    
    result = df.copy()
    
    # Z-score based anomaly detection
    window = config['z_score_window']
    threshold = config['threshold']
    
    rolling_mean = result[price_col].rolling(window=window).mean()
    rolling_std = result[price_col].rolling(window=window).std()
    z_score = (result[price_col] - rolling_mean) / rolling_std
    
    result['price_zscore'] = z_score
    result['is_anomaly'] = (abs(z_score) > threshold).astype(int)
    
    logger.info("Anomaly detection features calculated successfully")
    return result

def add_rolling_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistical features (skewness, kurtosis) to the DataFrame.
    
    Args:
        df: DataFrame containing trading data
        
    Returns:
        DataFrame with added rolling statistical features
    """
    logger.info("Calculating rolling statistical features")
    config = get_config().feature_config['statistical_features']
    
    result = df.copy()
    window = config['rolling_window']
    
    for col in config['target_columns']:
        if col not in result.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
            
        if 'skew' in config['metrics']:
            result[f'{col}_rolling_skew_{window}'] = (
                result[col].rolling(window=window).skew()
            )
            
        if 'kurt' in config['metrics']:
            result[f'{col}_rolling_kurt_{window}'] = (
                result[col].rolling(window=window).kurt()
            )
    
    logger.info("Rolling statistical features calculated successfully")
    return result

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between specified column pairs.
    
    Args:
        df: DataFrame containing trading data
        
    Returns:
        DataFrame with added interaction features
    """
    logger.info("Calculating interaction features")
    config = get_config().feature_config['interaction_features']
    
    result = df.copy()
    
    for col1, col2 in config['pairs']:
        if col1 not in result.columns or col2 not in result.columns:
            logger.warning(f"Column pair {col1}, {col2} not found in DataFrame")
            continue
            
        if 'multiply' in config['types']:
            result[f'{col1}_{col2}_multiply'] = result[col1] * result[col2]
            
        if 'ratio' in config['types']:
            # Handle division by zero
            result[f'{col1}_{col2}_ratio'] = result[col1].div(result[col2].replace(0, np.nan))
            
    logger.info("Interaction features calculated successfully")
    return result

def calculate_atr(df: pd.DataFrame, window: int = 14, price_col: str = 'price') -> pd.Series:
    """Calculate Average True Range for volatility measurement.
    
    Args:
        df: DataFrame with OHLC or similar price data
        window: Period for ATR calculation
        price_col: Column name for price data
        
    Returns:
        Series containing ATR values
    """
    # For single price column, approximate high/low from adjacent periods
    high = df[price_col].rolling(2, min_periods=1).max()
    low = df[price_col].rolling(2, min_periods=1).min()
    close = df[price_col]
    
    # Calculate True Range components
    tr1 = abs(high - low)  # Current high - current low
    tr2 = abs(high - close.shift(1))  # Current high - previous close
    tr3 = abs(low - close.shift(1))   # Current low - previous close
    
    # True Range is the maximum of the three measures
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # First value will be NaN due to shift operation
    tr.iloc[0] = np.nan
    
    # Calculate ATR using simple moving average of TR
    atr = tr.rolling(window=window, min_periods=window).mean()
    
    return atr

def detect_market_regimes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Detect market regimes based on volatility and trend."""
    logger.info("Detecting market regimes")
    result = df.copy()
    n_periods = len(df)
    
    if not config['regime_detection']['enabled'] or n_periods < 3:
        return result
    
    # Adjust window sizes for small datasets
    max_window = max(n_periods // 5, 2)
    
    # Volatility regime
    vol_config = config['regime_detection']['volatility']
    vol_window = min(vol_config['window'], max_window)
    
    if vol_config['source'] == 'atr':
        volatility_metric = calculate_atr(
            result, 
            window=vol_window,
            price_col='price'
        )
    else:  # rolling_stddev
        volatility_metric = result['price'].pct_change().rolling(
            window=vol_window,
            min_periods=1
        ).std()
    
    # Classify volatility regime
    if n_periods < 10:
        result['regime_volatility'] = 'medium'
    else:
        vol_mean = volatility_metric.mean()
        vol_std = volatility_metric.std()
        
        # Create bins with exactly one fewer label than bin edges
        bins = [-np.inf, vol_mean - vol_std/2, vol_mean + vol_std/2, np.inf]  # 4 edges
        labels = ['low', 'medium', 'high']  # 3 labels
        
        result['regime_volatility'] = pd.cut(
            volatility_metric,
            bins=bins,
            labels=labels,
            include_lowest=True
        )
    
    # Trend regime
    if config['regime_detection']['trend']['enabled']:
        trend_config = config['regime_detection']['trend']
        ma_window = min(trend_config['ma_window'], max_window)
        
        # Simple trend detection for small datasets
        ma = result['price'].rolling(window=ma_window, min_periods=1).mean()
        ma_slope = ma.diff()
        
        # Trend strength based on slope magnitude
        slope_std = ma_slope.std() or 0.001  # Avoid division by zero
        result['trend_strength'] = np.where(
            abs(ma_slope) > slope_std,
            'strong',
            'weak'
        )
        
        # Trend direction
        result['trend_direction'] = np.where(
            ma_slope > 0, 'up',
            np.where(ma_slope < 0, 'down', 'sideways')
        )
        
        # Combine strength and direction
        result['regime_trend'] = result.apply(
            lambda x: f"{x['trend_strength']}_{x['trend_direction']}"
            if pd.notna(x['trend_strength']) and pd.notna(x['trend_direction'])
            else 'unknown',
            axis=1
        )
    
    logger.info(f"Market regime detection complete with adjusted windows (max_window={max_window})")
    return result

def generate_trading_signals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate trading signals based on technical indicators.
    
    Args:
        df: DataFrame containing price and indicator data
        config: Configuration dictionary for signal generation
        
    Returns:
        DataFrame with added signal columns
    """
    logger.info("Generating trading signals")
    result = df.copy()
    
    if not config['signal_generation']['enabled']:
        return result
        
    # MACD signals
    if config['signal_generation']['macd']['signal_cross_enabled']:
        macd_diff = result['macd'] - result['macd_signal']
        result['signal_macd_cross'] = np.where(
            (macd_diff > 0) & (macd_diff.shift(1) <= 0), 1,  # Bullish cross
            np.where(
                (macd_diff < 0) & (macd_diff.shift(1) >= 0), -1,  # Bearish cross
                0
            )
        )
        
        if config['signal_generation']['macd']['zero_cross_enabled']:
            result['signal_macd_zero'] = np.where(
                (result['macd'] > 0) & (result['macd'].shift(1) <= 0), 1,
                np.where(
                    (result['macd'] < 0) & (result['macd'].shift(1) >= 0), -1,
                    0
                )
            )
    
    # RSI signals
    if config['signal_generation']['rsi']['enabled']:
        rsi_config = config['signal_generation']['rsi']
        result['signal_rsi'] = np.where(
            result['rsi'] < rsi_config['oversold'], 1,  # Oversold
            np.where(
                result['rsi'] > rsi_config['overbought'], -1,  # Overbought
                0
            )
        )
    
    # Moving Average crossover signals
    if config['signal_generation']['ma_crossover']['enabled']:
        ma_config = config['signal_generation']['ma_crossover']
        fast_ma = result['price'].rolling(window=ma_config['fast_ma']).mean()
        slow_ma = result['price'].rolling(window=ma_config['slow_ma']).mean()
        
        result['signal_ma_cross'] = np.where(
            (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1)), 1,
            np.where(
                (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1)), -1,
                0
            )
        )
    
    # Volume surge signals
    if config['signal_generation']['volume_surge']['enabled']:
        vol_config = config['signal_generation']['volume_surge']
        vol_ma = result['volume'].rolling(window=vol_config['window']).mean()
        result['signal_volume_surge'] = np.where(
            result['volume'] > vol_ma * vol_config['threshold'], 1, 0
        )
    
    # Combine signals into composite score (-100 to +100)
    signal_columns = [col for col in result.columns if col.startswith('signal_')]
    if signal_columns:
        # Weight and sum all signals
        result['composite_signal'] = sum(
            result[col] for col in signal_columns
        ) * (100 / len(signal_columns))
    
    logger.info("Trading signal generation complete")
    return result

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to generate all features for the trading model.
    
    Args:
        df: DataFrame containing cleaned trading data
        
    Returns:
        DataFrame with all calculated features
    """
    logger.info("Starting feature generation process")
    config = get_config().feature_config
    
    # Generate features sequentially
    result = df.copy()
    
    # Technical indicators (preserving rows)
    result = add_technical_indicators(result)
    logger.info(f"After technical indicators: {result.shape}")
    
    # Volatility features
    result = add_volatility_features(result)
    logger.info(f"After volatility features: {result.shape}")
    
    # Market regimes
    result = detect_market_regimes(result, config)
    logger.info(f"After market regimes: {result.shape}")
    
    # Trading signals
    result = generate_trading_signals(result, config)
    logger.info(f"After trading signals: {result.shape}")
    
    # Lagged features
    result = add_lagged_features(result)
    logger.info(f"After lagged features: {result.shape}")
    
    # Anomaly detection
    result = add_anomaly_detection_features(result)
    logger.info(f"After anomaly detection: {result.shape}")
    
    # Rolling statistical features
    result = add_rolling_statistical_features(result)
    logger.info(f"After statistical features: {result.shape}")
    
    # Interaction features
    result = add_interaction_features(result)
    logger.info(f"After interaction features: {result.shape}")
    
    # Handle NaN values more carefully
    # 1. Forward fill then backward fill technical indicators and volatility features
    indicator_cols = [col for col in result.columns if any(x in col.lower() for x in 
                     ['sma', 'ema', 'rsi', 'macd', 'bb_', 'volatility'])]
    result[indicator_cols] = result[indicator_cols].ffill().bfill()
    
    # 2. Forward fill regime and signal columns
    regime_cols = [col for col in result.columns if 'regime' in col.lower()]
    signal_cols = [col for col in result.columns if 'signal' in col.lower()]
    result[regime_cols + signal_cols] = result[regime_cols + signal_cols].ffill()
    
    # 3. Drop rows only if critical columns (price, volume) are missing
    result.dropna(subset=['price', 'volume'], inplace=True)
    
    logger.info(f"Feature generation complete. Final shape: {result.shape}")
    return result