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
    """Add technical indicators to the DataFrame.
    
    Args:
        df: DataFrame containing trading data
        price_col: Name of the price column to use for calculations
        
    Returns:
        DataFrame with added technical indicators
    """
    logger.info("Calculating technical indicators")
    config = get_config().feature_config['technical_indicators']
    
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Simple Moving Averages
    for window in config['sma_windows']:
        result[f'sma_{window}'] = ta.trend.sma_indicator(result[price_col], window=window)
    
    # Exponential Moving Averages
    for window in config['ema_windows']:
        result[f'ema_{window}'] = ta.trend.ema_indicator(result[price_col], window=window)
    
    # RSI
    result['rsi'] = ta.momentum.rsi(result[price_col], window=config['rsi_period'])
    
    # MACD
    macd = ta.trend.MACD(
        result[price_col],
        window_slow=config['macd']['slow_period'],
        window_fast=config['macd']['fast_period'],
        window_sign=config['macd']['signal_period']
    )
    result['macd'] = macd.macd()
    result['macd_signal'] = macd.macd_signal()
    result['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        result[price_col],
        window=config['bollinger_bands']['window'],
        window_dev=config['bollinger_bands']['num_std']
    )
    result['bb_upper'] = bb.bollinger_hband()
    result['bb_lower'] = bb.bollinger_lband()
    result['bb_mid'] = bb.bollinger_mavg()
    
    logger.info("Technical indicators calculated successfully")
    return result

def add_volatility_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """Add volatility-based features to the DataFrame.
    
    Args:
        df: DataFrame containing trading data
        price_col: Name of the price column to use for calculations
        
    Returns:
        DataFrame with added volatility features
    """
    logger.info("Calculating volatility features")
    config = get_config().feature_config['volatility']
    
    result = df.copy()
    returns = np.log(result[price_col]).diff()
    
    # Rolling volatility for different windows
    for window in config['std_windows']:
        result[f'volatility_{window}'] = returns.rolling(window=window).std()
    
    # Volatility regime (using 20-period volatility)
    vol_20 = returns.rolling(window=20).std()
    
    # Only calculate regimes if we have enough non-NaN values
    if vol_20.notna().sum() >= 3:  # Need at least 3 points for 3 quantiles
        result['vol_regime'] = pd.qcut(
            vol_20.fillna(vol_20.mean()),  # Fill NaNs with mean for binning
            q=3, 
            labels=['low', 'medium', 'high'],
            duplicates='drop'  # Handle potential duplicate bin edges
        )
    else:
        # For small samples, assign medium volatility
        result['vol_regime'] = 'medium'
    
    logger.info("Volatility features calculated successfully")
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
    """Detect market regimes based on volatility and trend.
    
    Args:
        df: DataFrame containing price and indicator data
        config: Configuration dictionary for regime detection
        
    Returns:
        DataFrame with added regime columns
    """
    logger.info("Detecting market regimes")
    result = df.copy()
    
    if not config['regime_detection']['enabled']:
        return result
        
    # Volatility regime
    vol_config = config['regime_detection']['volatility']
    if vol_config['source'] == 'atr':
        volatility_metric = calculate_atr(
            result, 
            window=min(vol_config['window'], len(result) // 2)  # Adjust window if data is small
        )
    else:  # rolling_stddev
        volatility_metric = result['price'].pct_change().rolling(
            window=min(vol_config['window'], len(result) // 2)
        ).std()
    
    # Classify volatility regime
    result['regime_volatility'] = pd.cut(
        volatility_metric,
        bins=[-np.inf, vol_config['thresholds']['low'], 
              vol_config['thresholds']['high'], np.inf],
        labels=['low', 'medium', 'high']
    )
    
    # Trend regime
    if config['regime_detection']['trend']['enabled']:
        trend_config = config['regime_detection']['trend']
        
        # Adjust window size for small datasets
        window = min(trend_config['ma_window'], len(result) // 2)
        
        # Calculate trend using simpler method for small datasets
        if len(result) < 20:  # If dataset is very small
            ma = result['price'].rolling(window=window, min_periods=1).mean()
            ma_slope = ma.diff()
            
            # Simple trend strength based on slope magnitude
            slope_std = ma_slope.std()
            result['trend_strength'] = np.where(
                abs(ma_slope) > slope_std,
                'strong',
                'weak'
            )
            
            result['trend_direction'] = np.where(
                ma_slope > 0, 'up',
                np.where(ma_slope < 0, 'down', 'sideways')
            )
        else:
            # Use ADX for larger datasets
            adx = trend.ADXIndicator(
                high=result['price'].rolling(2, min_periods=1).max(),
                low=result['price'].rolling(2, min_periods=1).min(),
                close=result['price'],
                window=window
            )
            result['adx'] = adx.adx()
            
            # Determine trend strength
            result['trend_strength'] = pd.cut(
                result['adx'],
                bins=[-np.inf, trend_config['threshold'], np.inf],
                labels=['weak', 'strong']
            )
            
            # Determine trend direction using MA slope
            ma = result['price'].rolling(window=window, min_periods=1).mean()
            ma_slope = ma.diff()
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
    
    logger.info("Market regime detection complete")
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
    
    # Technical indicators
    result = add_technical_indicators(result)
    
    # Volatility features
    result = add_volatility_features(result)
    
    # Market regimes
    result = detect_market_regimes(result, config)
    
    # Trading signals
    result = generate_trading_signals(result, config)
    
    # Lagged features
    result = add_lagged_features(result)
    
    # Anomaly detection
    result = add_anomaly_detection_features(result)
    
    # Rolling statistical features
    result = add_rolling_statistical_features(result)
    
    # Interaction features
    result = add_interaction_features(result)
    
    # Remove any remaining NaN values from feature calculation
    result.dropna(inplace=True)
    
    logger.info(f"Feature generation complete. Final shape: {result.shape}")
    return result