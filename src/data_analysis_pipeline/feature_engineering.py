"""Feature engineering module for generating trading signals and indicators."""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import ta  # Technical Analysis library
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

from .config import get_config, PipelineConfig

logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """Add technical indicators to the DataFrame."""
    logger.info("Calculating technical indicators")
    config = get_config().feature_config['technical_indicators']
    
    result = df.copy()
    n_periods = len(df)
    
    # Adjust window sizes for small datasets
    max_window = max(n_periods // 5, 2)  # At least 2 periods
    
    # Forward fill price data for indicator calculation
    result[price_col] = result[price_col].ffill().bfill()
    
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
    
    # Handle missing values in technical indicators
    tech_cols = [col for col in result.columns if any(x in col for x in ['sma', 'ema', 'rsi'])]
    result[tech_cols] = result[tech_cols].ffill().bfill()
    
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
    
    # Handle missing values in returns
    returns = returns.ffill().bfill()
    
    # Rolling volatility with adjusted windows
    std_windows = [min(w, max_window) for w in config['std_windows']]
    for window in std_windows:
        result[f'volatility_{window}'] = returns.rolling(
            window=window,
            min_periods=1
        ).std().ffill().bfill()
    
    # Volatility regime using shorter window for small datasets
    vol_window = min(20, max_window)
    vol_20 = returns.rolling(window=vol_window, min_periods=1).std()
    
    # Improved volatility regime classification
    if n_periods < 10:
        result['vol_regime'] = 'medium'  # Default regime for very small datasets
    else:
        try:
            vol_unique = vol_20.dropna().unique()
            if len(vol_unique) >= 3:
                # Use quantile-based bins for more robust classification
                quartiles = vol_20.quantile([0.25, 0.75])
                result['vol_regime'] = pd.cut(
                    vol_20,
                    bins=[-np.inf, quartiles[0.25], quartiles[0.75], np.inf],
                    labels=['low', 'medium', 'high'],
                    include_lowest=True
                )
            else:
                # Fall back to standard deviation based classification
                vol_mean = vol_20.mean()
                vol_std = vol_20.std()
                result['vol_regime'] = pd.cut(
                    vol_20,
                    bins=[-np.inf, vol_mean - vol_std, vol_mean + vol_std, np.inf],
                    labels=['low', 'medium', 'high'],
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
    
    # Classify volatility regime using simple thresholds for small datasets
    if n_periods < 10:
        result['regime_volatility'] = 'medium'
    else:
        vol_mean = volatility_metric.mean()
        vol_std = volatility_metric.std()
        result['regime_volatility'] = pd.cut(
            volatility_metric,
            bins=[-np.inf, vol_mean - vol_std/2, vol_mean + vol_std/2, np.inf],
            labels=['low', 'medium', 'high'],
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
    """Generate all configured features."""
    logger.info("Starting feature generation")
    result = df.copy()
    config = get_config()
    
    # Get pipeline mode
    pipeline_mode = config.get('pipeline_mode', 'full_analysis')
    logger.info(f"Generating features in {pipeline_mode} mode")
    
    try:
        # Core Technical Indicators (always enabled)
        if config['feature_engineering']['technical_indicators']['core_indicators']['enabled']:
            result = add_core_technical_indicators(result)
        
        # Extended Technical Indicators (mode dependent)
        if (pipeline_mode == 'full_analysis' and 
            config['feature_engineering']['technical_indicators']['extended_indicators']['enabled']):
            result = add_extended_technical_indicators(result)
        
        # Feature Groups
        feature_groups = config['feature_engineering']['feature_groups']
        
        # Lagged Features (configurable)
        if feature_groups['lagged_features']['enabled']:
            result = add_lagged_features(
                result,
                max_lags=feature_groups['lagged_features']['max_lags']
            )
        
        # Rolling Statistics (configurable)
        if feature_groups['rolling_statistics']['enabled']:
            result = add_rolling_statistics(
                result,
                windows=feature_groups['rolling_statistics']['windows'],
                metrics=feature_groups['rolling_statistics']['metrics']
            )
        
        # Long-term Features (only in full analysis mode)
        if pipeline_mode == 'full_analysis' and feature_groups['long_term_features']['enabled']:
            result = add_long_term_features(
                result,
                windows=feature_groups['long_term_features']['windows'],
                metrics=feature_groups['long_term_features']['metrics']
            )
        
        # Interaction Features (mode dependent)
        if (pipeline_mode == 'full_analysis' and 
            feature_groups['interaction_features']['enabled']):
            result = add_interaction_features(
                result,
                pairs=feature_groups['interaction_features']['pairs']
            )
        
        logger.info("Feature generation completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in feature generation: {str(e)}")
        raise

def add_core_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add essential technical indicators."""
    result = df.copy()
    config = get_config()
    core_config = config['feature_engineering']['technical_indicators']['core_indicators']
    
    # SMA
    if core_config['sma']['enabled']:
        for window in core_config['sma']['windows']:
            result[f'sma_{window}'] = ta.trend.sma_indicator(result['price'], window=window)
    
    # EMA
    if core_config['ema']['enabled']:
        for window in core_config['ema']['windows']:
            result[f'ema_{window}'] = ta.trend.ema_indicator(result['price'], window=window)
    
    # RSI
    if core_config['rsi']['enabled']:
        result['rsi'] = ta.momentum.rsi(result['price'], window=core_config['rsi']['period'])
    
    # Volatility (essential)
    if core_config['volatility']['enabled']:
        for window in core_config['volatility']['windows']:
            result[f'volatility_{window}'] = ta.volatility.standard_deviation(
                result['price'],
                window=window
            )
    
    # Log returns (always calculated as needed by HMM)
    result['log_return'] = np.log(result['price']).diff()
    
    return result

def add_extended_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add additional technical indicators for full analysis."""
    result = df.copy()
    config = get_config()
    ext_config = config['feature_engineering']['technical_indicators']['extended_indicators']
    
    # Only proceed if extended indicators are enabled
    if not ext_config['enabled']:
        return result
    
    # Bollinger Bands
    if ext_config['bollinger_bands']['enabled']:
        bb_config = ext_config['bollinger_bands']
        result['bb_high'] = ta.volatility.bollinger_hband(
            result['price'],
            window=bb_config['window'],
            std=bb_config['num_std']
        )
        result['bb_low'] = ta.volatility.bollinger_lband(
            result['price'],
            window=bb_config['window'],
            std=bb_config['num_std']
        )
    
    # MACD
    if ext_config['macd']['enabled']:
        macd_config = ext_config['macd']
        result['macd'] = ta.trend.macd(
            result['price'],
            fast=macd_config['fast_period'],
            slow=macd_config['slow_period'],
            signal=macd_config['signal_period']
        )
        result['macd_signal'] = ta.trend.macd_signal(
            result['price'],
            fast=macd_config['fast_period'],
            slow=macd_config['slow_period'],
            signal=macd_config['signal_period']
        )
    
    # ADX
    if ext_config['adx']['enabled']:
        result['adx'] = ta.trend.adx(
            high=result['price'] * 1.0001,  # Synthetic high
            low=result['price'] * 0.9999,   # Synthetic low
            close=result['price'],
            window=ext_config['adx']['period']
        )
    
    # Stochastic
    if ext_config['stochastic']['enabled']:
        stoch_config = ext_config['stochastic']
        result['stoch_k'] = ta.momentum.stoch(
            high=result['price'] * 1.0001,
            low=result['price'] * 0.9999,
            close=result['price'],
            window=stoch_config['k_period'],
            smooth_window=stoch_config['d_period']
        )
    
    return result

def add_rolling_statistics(
    df: pd.DataFrame,
    windows: List[int],
    metrics: List[str]
) -> pd.DataFrame:
    """Add rolling statistical features."""
    result = df.copy()
    
    for window in windows:
        rolling = result['price'].rolling(window=window)
        
        if 'mean' in metrics:
            result[f'price_rolling_mean_{window}'] = rolling.mean()
        if 'std' in metrics:
            result[f'price_rolling_std_{window}'] = rolling.std()
        if 'skew' in metrics:
            result[f'price_rolling_skew_{window}'] = rolling.skew()
        if 'kurt' in metrics:
            result[f'price_rolling_kurt_{window}'] = rolling.kurt()
    
    return result

def add_long_term_features(
    df: pd.DataFrame,
    windows: List[int],
    metrics: List[str]
) -> pd.DataFrame:
    """Add long-term statistical features."""
    result = df.copy()
    
    for window in windows:
        rolling = result['price'].rolling(window=window)
        
        for metric in metrics:
            if metric == 'mean':
                result[f'price_long_mean_{window}'] = rolling.mean()
            elif metric == 'std':
                result[f'price_long_std_{window}'] = rolling.std()
            elif metric == 'skew':
                result[f'price_long_skew_{window}'] = rolling.skew()
            elif metric == 'kurt':
                result[f'price_long_kurt_{window}'] = rolling.kurt()
    
    return result

def add_interaction_features(
    df: pd.DataFrame,
    pairs: List[List[str]]
) -> pd.DataFrame:
    """Add interaction features between pairs of columns."""
    result = df.copy()
    
    for col1, col2 in pairs:
        if col1 in result.columns and col2 in result.columns:
            # Multiplication interaction
            result[f'{col1}_{col2}_multiply'] = result[col1] * result[col2]
            
            # Ratio interaction (with safety check)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = result[col1] / result[col2]
                result[f'{col1}_{col2}_ratio'] = np.where(
                    np.isfinite(ratio),
                    ratio,
                    0
                )
    
    return result

def add_lagged_features(
    df: pd.DataFrame,
    max_lags: int
) -> pd.DataFrame:
    """Add lagged versions of key features."""
    result = df.copy()
    
    # Add lags for price and volume
    for lag in range(1, max_lags + 1):
        result[f'price_lag_{lag}'] = result['price'].shift(lag)
        if 'volume' in result.columns:
            result[f'volume_lag_{lag}'] = result['volume'].shift(lag)
    
    return result

def calculate_volatility_regimes(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Calculate volatility regimes based on configured thresholds.
    
    Args:
        df: DataFrame with volatility column
        config: Pipeline configuration
        
    Returns:
        DataFrame with regime_volatility column added
    """
    low = config.feature_engineering['volatility_threshold_low']
    high = config.feature_engineering['volatility_threshold_high']
    
    # Validate thresholds
    if not (0 < low < high):
        raise ValueError(f"Invalid volatility thresholds: low={low}, high={high}")
    
    # Define bin edges including infinite bounds
    bins = [-np.inf, low, high, np.inf]
    
    # Labels must be one fewer than bin edges
    labels = ['low', 'medium', 'high']
    
    df['regime_volatility'] = pd.cut(
        df['volatility_20'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    return df