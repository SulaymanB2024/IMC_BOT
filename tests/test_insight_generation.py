"""Unit tests for market insight generation functionality."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.data_analysis_pipeline.insight_generation import (
    analyze_market_regime,
    analyze_signals,
    analyze_anomalies,
    generate_market_summary,
    generate_insights
)

@pytest.fixture
def sample_market_data():
    """Create sample market data with features for testing."""
    index = pd.date_range('2024-01-01', periods=10, freq='D')
    data = {
        'price': [100, 102, 98, 104, 103, 105, 107, 108, 106, 110],
        'regime_volatility': ['low', 'low', 'medium', 'medium', 'high', 
                            'high', 'medium', 'low', 'low', 'medium'],
        'regime_trend': ['weak_up', 'strong_up', 'weak_down', 'strong_up', 'weak_up',
                        'strong_up', 'strong_up', 'weak_up', 'weak_down', 'strong_up'],
        'signal_macd_cross': [0, 1, -1, 0, 1, 1, 0, -1, 0, 1],
        'signal_rsi': [0, 1, -1, 0, 0, 1, 1, -1, 0, 0],
        'composite_signal': [0, 75, -60, 10, 45, 80, 30, -40, 5, 65],
        'is_anomaly': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'price_zscore': [0.1, 0.5, 3.2, 0.8, 1.2, 3.5, 0.9, 0.3, 0.7, 1.1]
    }
    return pd.DataFrame(data, index=index)

def test_analyze_market_regime(sample_market_data):
    """Test market regime analysis."""
    result = analyze_market_regime(sample_market_data)
    
    assert 'volatility' in result, "Missing volatility analysis"
    assert 'trend' in result, "Missing trend analysis"
    
    vol_analysis = result['volatility']
    assert vol_analysis['dominant_regime'] in ['low', 'medium', 'high']
    assert isinstance(vol_analysis['regime_changes'], int)
    assert isinstance(vol_analysis['regime_distribution'], dict)
    
    trend_analysis = result['trend']
    assert 'weak' in trend_analysis['dominant_regime'] or 'strong' in trend_analysis['dominant_regime']
    assert isinstance(trend_analysis['regime_changes'], int)
    assert isinstance(trend_analysis['regime_distribution'], dict)

def test_analyze_signals(sample_market_data):
    """Test trading signal analysis."""
    result = analyze_signals(sample_market_data)
    
    assert 'signals' in result, "Missing individual signals analysis"
    assert 'composite_signal' in result, "Missing composite signal analysis"
    
    # Check individual signals
    for signal_name in ['signal_macd_cross', 'signal_rsi']:
        assert signal_name in result['signals']
        signal_info = result['signals'][signal_name]
        assert isinstance(signal_info['total_signals'], int)
        assert isinstance(signal_info['signal_distribution'], dict)
    
    # Check composite signal
    composite = result['composite_signal']
    assert all(isinstance(composite[key], (int, float)) for key in 
              ['mean', 'std', 'max', 'min', 'strong_buy_signals', 'strong_sell_signals'])

def test_analyze_anomalies(sample_market_data):
    """Test anomaly analysis."""
    result = analyze_anomalies(sample_market_data)
    
    assert 'anomalies' in result, "Missing anomaly analysis"
    assert 'zscore_analysis' in result, "Missing z-score analysis"
    
    anomalies = result['anomalies']
    assert isinstance(anomalies['total_count'], int)
    assert isinstance(anomalies['percentage'], float)
    assert isinstance(anomalies['periods'], list)
    assert len(anomalies['periods']) == anomalies['total_count']
    
    zscore = result['zscore_analysis']
    assert isinstance(zscore['extreme_values_count'], int)
    assert isinstance(zscore['max_deviation'], float)
    assert isinstance(zscore['mean_deviation'], float)

def test_generate_market_summary(sample_market_data):
    """Test market summary generation."""
    summary = generate_market_summary(sample_market_data)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    
    # Check for key components in summary
    assert "market trend" in summary.lower()
    assert "volatility regime" in summary.lower()
    assert "signal" in summary.lower()

def test_generate_insights(sample_market_data, tmp_path):
    """Test full insight generation and file output."""
    # Mock config to use temporary path
    class MockConfig:
        def __init__(self, tmp_path):
            self.output_config = {
                'trading_model': {
                    'path': str(tmp_path / 'features.parquet')
                }
            }
    
    # Patch config
    import src.data_analysis_pipeline.insight_generation as ig
    original_config = ig.get_config
    ig.get_config = lambda: MockConfig(tmp_path)
    
    try:
        # Generate insights
        generate_insights(sample_market_data)
        
        # Check that files were created
        assert (tmp_path / 'market_insights.json').exists()
        assert (tmp_path / 'market_summary.txt').exists()
        
        # Validate JSON content
        with open(tmp_path / 'market_insights.json') as f:
            insights = json.load(f)
            assert all(key in insights for key in 
                      ['market_regimes', 'trading_signals', 'anomaly_analysis', 
                       'market_summary', 'metadata'])
        
        # Validate summary content
        with open(tmp_path / 'market_summary.txt') as f:
            summary = f.read()
            assert len(summary) > 0
            
    finally:
        # Restore original config
        ig.get_config = original_config