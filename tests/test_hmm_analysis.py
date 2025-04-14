"""Unit tests for HMM analysis functionality."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_analysis_pipeline.hmm_analysis import (
    prepare_hmm_features,
    initialize_hmm,
    get_state_durations,
    get_stationary_distribution,
    train_hmm
)

@pytest.fixture
def sample_market_data():
    """Create sample market data with known patterns."""
    return pd.DataFrame({
        'price': [100, 102, 98, 104, 103, 105, 107, 108],
        'volume': [1000, 1200, 800, 1500, 1300, 1400, 1600, 1700],
        'log_return': [0, 0.02, -0.04, 0.06, -0.01, 0.02, 0.02, 0.01],
        'volatility_5': [0.01, 0.015, 0.02, 0.025, 0.02, 0.015, 0.01, 0.01],
        'rsi': [50, 60, 40, 70, 65, 68, 72, 75],
        'macd': [-0.1, 0.1, -0.2, 0.3, 0.2, 0.1, 0.2, 0.3]
    }, index=pd.date_range('2024-01-01', periods=8, freq='D'))

@pytest.fixture
def mock_hmm_config():
    """Create mock HMM configuration."""
    return {
        'n_hidden_states': 2,
        'covariance_type': 'diag',
        'feature_scaling': {
            'method': 'standard'
        }
    }

def test_prepare_hmm_features(sample_market_data, mock_hmm_config):
    """Test feature preparation for HMM."""
    X_scaled = prepare_hmm_features(sample_market_data, mock_hmm_config)
    
    # Check output shape
    assert X_scaled.shape[0] == len(sample_market_data)
    assert X_scaled.shape[1] == 5  # Expected features
    
    # Check scaling properties
    assert np.abs(X_scaled.mean(axis=0)).max() < 0.1, "Features not properly centered"
    assert np.abs(X_scaled.std(axis=0) - 1.0).max() < 0.1, "Features not properly scaled"
    
    # Check for NaN/inf values
    assert not np.isnan(X_scaled).any(), "Found NaN values"
    assert not np.isinf(X_scaled).any(), "Found infinite values"

def test_initialize_hmm(sample_market_data, mock_hmm_config):
    """Test HMM initialization."""
    X_scaled = prepare_hmm_features(sample_market_data, mock_hmm_config)
    n_states = mock_hmm_config['n_hidden_states']
    n_features = X_scaled.shape[1]

    model = initialize_hmm(n_states, n_features, X_scaled, mock_hmm_config['covariance_type'])

    # Check model properties
    assert model.n_components == n_states
    # Note: HMM doesn't expose n_features until after fit is called
    assert model.means_.shape[1] == n_features, "Wrong number of features"
    assert model.covariance_type == mock_hmm_config['covariance_type']

    # Check transition matrix properties
    assert model.transmat_.shape == (n_states, n_states)
    assert np.allclose(model.transmat_.sum(axis=1), 1.0), "Invalid transition probabilities"

    # Check means and covariances
    assert model.means_.shape == (n_states, n_features)
    if mock_hmm_config['covariance_type'] == 'diag':
        # Verify shape is (n_states, n_features, n_features) as per hmmlearn implementation
        assert model.covars_.shape == (n_states, n_features, n_features), "Covars_ shape should be (N, D, D) even for diag type"
        
        for i in range(n_states):
            # Check if the matrix for state i is effectively diagonal
            diagonal_matrix = np.diag(np.diag(model.covars_[i]))
            assert np.allclose(model.covars_[i], diagonal_matrix), f"Covariance matrix for state {i} is not diagonal"
            # Verify variances are positive
            assert np.all(np.diag(model.covars_[i]) > 0), f"Variances (diagonal) for state {i} should be positive"

def test_get_state_durations():
    """Test state duration calculation."""
    # Case 1: Simple alternating states [1,1,1,0,0,1,1]
    states = np.array([1, 1, 1, 0, 0, 1, 1], dtype=bool)
    durations = get_state_durations(states)
    assert len(durations) == 2, "Should find 2 runs of True values"
    assert durations[0] == 3, "First run should be length 3"
    assert durations[1] == 2, "Second run should be length 2"
    
    # Case 2: Single state all True [1,1,1]
    states = np.array([1, 1, 1], dtype=bool)
    durations = get_state_durations(states)
    assert len(durations) == 1, "Should find 1 run"
    assert durations[0] == 3, "Run should be length 3"
    
    # Case 3: Single state all False [0,0,0]
    states = np.array([0, 0, 0], dtype=bool)
    durations = get_state_durations(states)
    assert len(durations) == 1, "Should return [0] for no True values"
    assert durations[0] == 0, "Duration should be 0 for all False"
    
    # Case 4: Starts with True [1,0,1,1]
    states = np.array([1, 0, 1, 1], dtype=bool)
    durations = get_state_durations(states)
    assert len(durations) == 2, "Should find 2 runs"
    assert durations[0] == 1, "First run should be length 1"
    assert durations[1] == 2, "Second run should be length 2"

def test_get_stationary_distribution():
    """Test stationary distribution calculation."""
    # Test case 1: Simple 2-state transition matrix
    P = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    pi = get_stationary_distribution(P)
    assert np.allclose(pi, [0.5, 0.5]), "Incorrect stationary distribution"
    assert np.allclose(pi.sum(), 1.0), "Distribution doesn't sum to 1"
    
    # Test case 2: More complex 3-state transition matrix
    P = np.array([[0.8, 0.1, 0.1],
                  [0.2, 0.6, 0.2],
                  [0.1, 0.1, 0.8]])
    pi = get_stationary_distribution(P)
    assert np.allclose(pi.sum(), 1.0), "Distribution doesn't sum to 1"
    assert all(p >= 0 for p in pi), "Negative probabilities found"

def test_train_hmm(sample_market_data, mock_hmm_config):
    """Test HMM training and results."""
    results = train_hmm(sample_market_data, mock_hmm_config)
    
    # Check basic structure
    assert 'model_info' in results
    assert 'states' in results
    assert 'transition_matrix' in results
    
    # Check model info
    assert results['model_info']['n_states'] == mock_hmm_config['n_hidden_states']
    assert results['model_info']['covariance_type'] == mock_hmm_config['covariance_type']
    
    # Check state sequence
    assert len(results['state_sequence']) == len(sample_market_data)
    assert all(isinstance(s, int) for s in results['state_sequence'])
    
    # Check state characteristics
    for state in results['states'].values():
        assert 'means' in state
        assert 'characteristics' in state
        assert isinstance(state['characteristics'], list)
        
    # Check transition matrix
    trans_mat = np.array(results['transition_matrix'])
    assert trans_mat.shape == (mock_hmm_config['n_hidden_states'],) * 2
    assert np.allclose(trans_mat.sum(axis=1), 1.0)