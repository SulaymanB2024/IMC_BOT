"""Unit tests for LLM output generation functionality."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.data_analysis_pipeline.llm_output import (
    format_time_series_context,
    format_regime_context,
    format_signal_context,
    format_anomaly_context,
    format_pattern_context,
    generate_llm_output
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

def test_time_series_context(sample_market_data):
    """Test time series context formatting."""
    result = format_time_series_context(sample_market_data)
    
    assert 'time_series_characteristics' in result
    chars = result['time_series_characteristics']
    
    assert all(key in chars for key in 
              ['start_time', 'end_time', 'frequency', 'total_periods'])
    assert all(key in chars['price_statistics'] for key in 
              ['mean', 'std', 'min', 'max', 'last_value'])
    assert chars['total_periods'] == len(sample_market_data)
    assert chars['frequency'] == 'D'  # Daily frequency

def test_regime_context(sample_market_data):
    """Test market regime context formatting."""
    result = format_regime_context(sample_market_data)
    
    assert 'market_regimes' in result
    regimes = result['market_regimes']
    
    assert 'current_state' in regimes
    assert 'regime_transitions' in regimes
    
    state = regimes['current_state']
    assert state['current_volatility'] == sample_market_data['regime_volatility'].iloc[-1]
    assert state['current_trend'] == sample_market_data['regime_trend'].iloc[-1]
    
    assert 'volatility_transitions' in regimes['regime_transitions']
    assert 'trend_transitions' in regimes['regime_transitions']

def test_signal_context(sample_market_data):
    """Test trading signal context formatting."""
    result = format_signal_context(sample_market_data)
    
    assert 'trading_signals' in result
    signals = result['trading_signals']
    
    assert 'current_signals' in signals
    assert 'composite_signal' in signals
    
    # Check current signals
    for col in ['signal_macd_cross', 'signal_rsi']:
        assert col in signals['current_signals']
        assert signals['current_signals'][col] == sample_market_data[col].iloc[-1]
    
    # Check composite signal
    composite = signals['composite_signal']
    assert all(key in composite for key in ['value', 'interpretation', 'confidence'])
    assert isinstance(composite['value'], float)
    assert isinstance(composite['interpretation'], str)
    assert isinstance(composite['confidence'], float)

def test_anomaly_context(sample_market_data):
    """Test anomaly detection context formatting."""
    result = format_anomaly_context(sample_market_data)
    
    assert 'anomaly_detection' in result
    anomalies = result['anomaly_detection']
    
    assert all(key in anomalies for key in 
              ['recent_anomalies', 'total_anomalies', 'anomaly_rate'])
    
    assert anomalies['total_anomalies'] == sample_market_data['is_anomaly'].sum()
    assert isinstance(anomalies['anomaly_rate'], float)
    assert isinstance(anomalies['recent_anomalies'], list)

def test_pattern_context(sample_market_data):
    """Test market pattern context formatting."""
    result = format_pattern_context(sample_market_data)
    
    assert 'market_patterns' in result
    patterns = result['market_patterns']
    
    assert 'identified_patterns' in patterns
    assert 'pattern_confidence' in patterns
    
    assert isinstance(patterns['identified_patterns'], list)
    for pattern in patterns['identified_patterns']:
        assert all(key in pattern for key in 
                  ['pattern', 'occurrence_count', 'significance'])

def test_generate_llm_output(sample_market_data, tmp_path):
    """Test full LLM output generation and file output."""
    # Mock config
    class MockConfig:
        def __init__(self, tmp_path):
            self.output_config = {
                'trading_model': {
                    'path': str(tmp_path / 'features.parquet')
                }
            }
    
    # Patch config
    import src.data_analysis_pipeline.llm_output as llm
    original_config = llm.get_config
    llm.get_config = lambda: MockConfig(tmp_path)
    
    try:
        # Generate LLM output
        generate_llm_output(sample_market_data)
        
        # Check that file was created
        output_file = tmp_path / 'llm_context.json'
        assert output_file.exists()
        
        # Validate JSON content
        with open(output_file) as f:
            content = json.load(f)
            
        # Check all required sections
        assert all(section in content for section in [
            'time_series_characteristics',
            'market_regimes',
            'trading_signals',
            'anomaly_detection',
            'market_patterns',
            'nlg_templates'
        ])
        
        # Check templates
        templates = content['nlg_templates']
        assert all(template in templates for template in [
            'market_summary',
            'anomaly_alert',
            'pattern_insight'
        ])
            
    finally:
        # Restore original config
        llm.get_config = original_config