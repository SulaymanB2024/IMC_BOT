"""Unit tests for feature engineering functionality."""
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from src.data_analysis_pipeline.feature_engineering import (
    add_technical_indicators,
    add_volatility_features,
    add_lagged_features,
    detect_market_regimes,
    generate_trading_signals,
    calculate_atr,
    prepare_hmm_features
)

class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.feature_config = {
            'technical_indicators': {
                'sma_windows': [3],
                'ema_windows': [3],
                'rsi_period': 14,
                'bollinger_bands': {'window': 20, 'num_std': 2},
                'macd': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9
                }
            },
            'volatility': {
                'std_windows': [5]
            },
            'lags': {
                'price_lags': [1, 2],
                'volume_lags': [1]
            },
            'regime_detection': {
                'enabled': True,
                'volatility': {
                    'source': 'atr',
                    'window': 20,
                    'thresholds': {
                        'low': 0.01,
                        'high': 0.03
                    }
                },
                'trend': {
                    'enabled': True,
                    'method': 'adx',
                    'threshold': 25,
                    'ma_window': 20,
                    'ma_comparison': True
                }
            },
            'signal_generation': {
                'enabled': True,
                'macd': {
                    'signal_cross_enabled': True,
                    'zero_cross_enabled': True
                },
                'rsi': {
                    'enabled': True,
                    'oversold': 30,
                    'overbought': 70
                },
                'ma_crossover': {
                    'enabled': True,
                    'fast_ma': 5,
                    'slow_ma': 20
                },
                'volume_surge': {
                    'enabled': True,
                    'threshold': 2.0,
                    'window': 20
                }
            }
        }

@pytest.fixture
def mock_config(monkeypatch):
    """Fixture to provide mock configuration."""
    config = MockConfig()
    def mock_get_config():
        return config
    from src.data_analysis_pipeline.feature_engineering import get_config
    monkeypatch.setattr('src.data_analysis_pipeline.feature_engineering.get_config', mock_get_config)
    return config

@pytest.fixture
def sample_price_data():
    """Create sample price data with predictable patterns."""
    return pd.DataFrame({
        'price': [100, 102, 98, 104, 103, 105, 107, 108],
        'volume': [1000, 1200, 800, 1500, 1300, 1400, 1600, 1700]
    }, index=pd.date_range('2024-01-01', periods=8, freq='D'))

def test_simple_moving_average(sample_price_data, mock_config):
    """Test SMA calculation with window=3."""
    result = add_technical_indicators(sample_price_data)
    
    # Calculate expected SMA manually
    expected_sma = pd.Series([
        np.nan, np.nan,  # First two periods have insufficient data
        100.0,  # (100 + 102 + 98) / 3
        101.333333,  # (102 + 98 + 104) / 3
        101.666667,  # (98 + 104 + 103) / 3
        104.0,  # (104 + 103 + 105) / 3
        105.0,  # (103 + 105 + 107) / 3
        106.666667,  # (105 + 107 + 108) / 3
    ], index=sample_price_data.index, name='sma_3')
    
    pd.testing.assert_series_equal(
        result['sma_3'].round(6),
        expected_sma.round(6),
        check_names=False
    )

def test_rsi_calculation(sample_price_data, mock_config):
    """Test RSI calculation basics."""
    result = add_technical_indicators(sample_price_data)
    
    assert 'rsi' in result.columns, "RSI column should be created"
    # Check only non-NaN values for RSI bounds
    valid_rsi = result['rsi'].dropna()
    assert valid_rsi.between(0, 100).all(), "RSI values should be between 0 and 100"
    # RSI needs several periods to start calculating
    assert result['rsi'].isna().sum() > 0, "RSI should have some NaN values at start"

def test_bollinger_bands(sample_price_data, mock_config):
    """Test Bollinger Bands calculation."""
    result = add_technical_indicators(sample_price_data)
    
    # Check that all BB columns exist
    assert 'bb_upper' in result.columns, "Upper BB column missing"
    assert 'bb_lower' in result.columns, "Lower BB column missing"
    assert 'bb_mid' in result.columns, "Middle BB column missing"
    
    # Drop NaN rows before comparing
    valid_bb = result.dropna(subset=['bb_upper', 'bb_lower', 'bb_mid'])
    
    # Basic BB properties on valid data
    assert (valid_bb['bb_upper'] >= valid_bb['bb_mid']).all(), "Upper band should be >= middle band"
    assert (valid_bb['bb_lower'] <= valid_bb['bb_mid']).all(), "Lower band should be <= middle band"

def test_volatility_features(sample_price_data, mock_config):
    """Test volatility calculation."""
    result = add_volatility_features(sample_price_data)
    
    # Check volatility column exists
    vol_col = f"volatility_{mock_config.feature_config['volatility']['std_windows'][0]}"
    assert vol_col in result.columns, "Volatility column should be created"
    
    # Volatility should be non-negative where not NaN
    valid_volatility = result[vol_col].dropna()
    assert (valid_volatility >= 0).all(), "Volatility should be non-negative"
    
    # First few values should be NaN due to rolling window
    window = mock_config.feature_config['volatility']['std_windows'][0]
    assert result[vol_col].head(window-1).isna().all(), f"First {window-1} values should be NaN"

def test_lagged_features(sample_price_data, mock_config):
    """Test creation of lagged features."""
    result = add_lagged_features(sample_price_data)
    
    # Check price lags
    for lag in mock_config.feature_config['lags']['price_lags']:
        col_name = f'price_lag_{lag}'
        assert col_name in result.columns, f"Missing lag column {col_name}"
        
        # Verify lag calculation by creating expected series with same index
        expected = pd.Series(
            sample_price_data['price'].shift(lag).values,
            index=sample_price_data.index,
            name=col_name
        )
        pd.testing.assert_series_equal(result[col_name], expected)

    # Check volume lags
    for lag in mock_config.feature_config['lags']['volume_lags']:
        col_name = f'volume_lag_{lag}'
        assert col_name in result.columns, f"Missing lag column {col_name}"
        
        # Verify lag calculation by creating expected series with same index
        expected = pd.Series(
            sample_price_data['volume'].shift(lag).values,
            index=sample_price_data.index,
            name=col_name
        )
        pd.testing.assert_series_equal(result[col_name], expected)

def test_market_regime_detection(sample_price_data, mock_config):
    """Test market regime detection features."""
    result = detect_market_regimes(sample_price_data, mock_config.feature_config)
    
    # Check that regime columns exist
    assert 'regime_volatility' in result.columns, "Volatility regime column missing"
    if mock_config.feature_config['regime_detection']['trend']['enabled']:
        assert 'regime_trend' in result.columns, "Trend regime column missing"
    
    # Validate regime values
    valid_vol_regimes = result['regime_volatility'].dropna()
    assert valid_vol_regimes.isin(['low', 'medium', 'high']).all(), "Invalid volatility regime values"
    
    if 'regime_trend' in result.columns:
        valid_trend_regimes = result['regime_trend'].dropna()
        assert all(x.split('_')[0] in ['weak', 'strong'] for x in valid_trend_regimes), "Invalid trend strength"
        assert all(x.split('_')[1] in ['up', 'down', 'sideways'] for x in valid_trend_regimes), "Invalid trend direction"

def test_trading_signal_generation(sample_price_data, mock_config):
    """Test trading signal generation."""
    # First generate technical indicators needed for signals
    data_with_indicators = add_technical_indicators(sample_price_data)
    result = generate_trading_signals(data_with_indicators, mock_config.feature_config)
    
    # Check signal columns exist
    if mock_config.feature_config['signal_generation']['macd']['signal_cross_enabled']:
        assert 'signal_macd_cross' in result.columns, "MACD signal cross column missing"
        signal_values = result['signal_macd_cross'].dropna()
        assert signal_values.isin([-1, 0, 1]).all(), "Invalid MACD signal values"
    
    if mock_config.feature_config['signal_generation']['rsi']['enabled']:
        assert 'signal_rsi' in result.columns, "RSI signal column missing"
        signal_values = result['signal_rsi'].dropna()
        assert signal_values.isin([-1, 0, 1]).all(), "Invalid RSI signal values"
    
    if mock_config.feature_config['signal_generation']['ma_crossover']['enabled']:
        assert 'signal_ma_cross' in result.columns, "MA crossover signal column missing"
        signal_values = result['signal_ma_cross'].dropna()
        assert signal_values.isin([-1, 0, 1]).all(), "Invalid MA crossover signal values"
    
    # Check composite signal
    assert 'composite_signal' in result.columns, "Composite signal column missing"
    valid_signals = result['composite_signal'].dropna()
    assert valid_signals.between(-100, 100).all(), "Composite signal outside valid range"

def test_atr_calculation(sample_price_data):
    """Test ATR calculation."""
    window = 3
    atr = calculate_atr(sample_price_data, window=window)
    
    assert isinstance(atr, pd.Series), "ATR should be a pandas Series"
    assert not atr.empty, "ATR should not be empty"
    
    # First {window} values should be NaN due to rolling window
    assert atr.head(window-1).isna().all(), f"First {window-1} values should be NaN"
    
    # ATR should be non-negative for valid values
    valid_atr = atr.dropna()
    assert (valid_atr >= 0).all(), "ATR values should be non-negative"

def test_volatility_regime_calculation(sample_price_data):
    """Test improved volatility regime calculation with quantile-based bins."""
    result = add_volatility_features(sample_price_data)
    
    # Check regime column exists
    assert 'vol_regime' in result.columns, "Missing volatility regime column"
    
    # Check regime values are valid
    valid_regimes = ['low', 'medium', 'high']
    assert result['vol_regime'].dropna().isin(valid_regimes).all(), "Invalid regime values"
    
    # Check distribution makes sense (should have all three regimes)
    regime_counts = result['vol_regime'].value_counts()
    assert len(regime_counts) == 3, "Missing some regime classifications"
    
    # Verify no NaN values
    assert not result['vol_regime'].isna().any(), "Found NaN values in regime classification"

def test_hmm_feature_preparation(sample_price_data):
    """Test HMM feature preparation and scaling."""
    config = get_config().hmm_analysis
    
    # Prepare features
    X_scaled = prepare_hmm_features(sample_price_data, config)
    
    # Check output shape
    expected_features = ['log_return', 'volume', 'volatility_5', 'rsi', 'macd']
    assert X_scaled.shape[1] == len(expected_features), "Incorrect number of features"
    
    # Check scaling properties
    assert np.abs(X_scaled.mean(axis=0)).max() < 0.1, "Features not properly centered"
    assert np.abs(X_scaled.std(axis=0) - 1.0).max() < 0.1, "Features not properly scaled"
    
    # Check no NaN or infinite values
    assert not np.isnan(X_scaled).any(), "Found NaN values after preparation"
    assert not np.isinf(X_scaled).any(), "Found infinite values after preparation"

def test_technical_indicators_small_dataset():
    """Test technical indicator calculation with very small datasets."""
    # Create minimal test data
    small_data = pd.DataFrame({
        'price': [100, 101, 99, 102, 98],
        'volume': [1000, 1100, 900, 1200, 800]
    }, index=pd.date_range('2025-04-12', periods=5, freq='T'))
    
    result = add_technical_indicators(small_data)
    
    # Check that indicators were calculated despite small size
    assert 'sma_5' in result.columns, "Missing SMA indicator"
    assert 'rsi' in result.columns, "Missing RSI indicator"
    assert not result['sma_5'].isna().any(), "Found NaN values in SMA"
    assert not result['rsi'].isna().any(), "Found NaN values in RSI"

def test_market_regime_detection_edge_cases(sample_price_data):
    """Test market regime detection with various edge cases."""
    # Test with very small dataset
    small_data = sample_price_data.head(3)
    result_small = detect_market_regimes(small_data, get_config().feature_config)
    assert 'regime_volatility' in result_small.columns, "Missing volatility regime for small data"
    assert result_small['regime_volatility'].notna().all(), "NaN values in small data regime"
    
    # Test with constant price (zero volatility)
    const_data = sample_price_data.copy()
    const_data['price'] = 100
    result_const = detect_market_regimes(const_data, get_config().feature_config)
    assert 'regime_volatility' in result_const.columns, "Missing volatility regime for constant data"
    assert result_const['regime_volatility'].notna().all(), "NaN values in constant data regime"
    
    # Test with missing values
    missing_data = sample_price_data.copy()
    missing_data.loc[missing_data.index[1:3], 'price'] = np.nan
    result_missing = detect_market_regimes(missing_data, get_config().feature_config)
    assert 'regime_volatility' in result_missing.columns, "Missing volatility regime for data with NaN"