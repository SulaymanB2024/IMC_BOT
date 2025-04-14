"""Hidden Markov Model analysis for market state detection."""
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

from .config import get_config

logger = logging.getLogger(__name__)

def prepare_hmm_features(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Prepare and scale features for HMM training."""
    pipeline_mode = config.get('pipeline_mode', 'full_analysis')
    
    # Select features based on pipeline mode
    features = config['hmm_analysis']['features'][f'{pipeline_mode}_mode']
    logger.info(f"Using {len(features)} features for HMM in {pipeline_mode} mode: {features}")
    
    # Validate feature availability
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logger.warning(f"Missing features for HMM: {missing_features}")
        features = [f for f in features if f in df.columns]
        
    if not features:
        raise ValueError("No valid features available for HMM analysis")
    
    X = df[features].copy()
    
    # Handle missing values using newer pandas methods
    X = X.ffill().bfill()
    
    # Remove outliers using IQR method
    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        X[col] = X[col].clip(lower_bound, upper_bound)
    
    # Standard scaling for better HMM convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add small noise to break exact zeros
    X_scaled += np.random.normal(0, 1e-6, X_scaled.shape)
    
    return X_scaled

def initialize_hmm(n_states: int, n_features: int, x_data: np.ndarray, covariance_type: str = 'diag') -> hmm.GaussianHMM:
    """Initialize HMM with smart starting parameters."""
    hmm_model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=200,
        tol=1e-5,
        init_params='',  # Disable automatic initialization
        random_state=42
    )
    
    # Set required dimensions
    hmm_model.n_features = n_features
    
    # Initialize transition matrix with high self-transition probabilities
    transitions = np.ones((n_states, n_states)) * 0.1
    np.fill_diagonal(transitions, 0.9)
    transitions /= transitions.sum(axis=1, keepdims=True)
    hmm_model.transmat_ = transitions
    
    # Initialize means using k-means
    kmeans = KMeans(n_clusters=n_states, random_state=42)
    kmeans.fit(x_data)
    hmm_model.means_ = kmeans.cluster_centers_
    
    # Initialize covariances based on k-means clusters
    if covariance_type == 'diag':
        covars = np.zeros((n_states, n_features))
        for i in range(n_states):
            cluster_points = x_data[kmeans.labels_ == i]
            if len(cluster_points) > 1:
                covars[i] = np.var(cluster_points, axis=0) + 1e-3
            else:
                covars[i] = np.ones(n_features) * 0.1
    else:  # full covariance
        covars = np.zeros((n_states, n_features, n_features))
        for i in range(n_states):
            cluster_points = x_data[kmeans.labels_ == i]
            if len(cluster_points) > 1:
                cov = np.cov(cluster_points.T) + np.eye(n_features) * 1e-3
                covars[i] = cov
            else:
                covars[i] = np.eye(n_features) * 0.1
    
    hmm_model.covars_ = covars
    
    # Set startprob to uniform distribution
    hmm_model.startprob_ = np.ones(n_states) / n_states
    
    return hmm_model

def train_hmm(data: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """Train HMM model with improved convergence handling."""
    logger.info("Starting HMM analysis")
    
    pipeline_mode = config.get('pipeline_mode', 'full_analysis')
    hmm_config = config['hmm_analysis']
    
    # Use simpler model for trading mode
    if pipeline_mode == 'trading':
        hmm_config['n_iterations'] = min(hmm_config['n_iterations'], 1000)  # Limit iterations
        hmm_config['n_init'] = min(hmm_config['n_init'], 5)  # Fewer initialization attempts
    
    # Prepare features
    X_scaled = prepare_hmm_features(data, config)
    features_used = config['hmm_analysis']['features'][f'{pipeline_mode}_mode']
    logger.info(f"Prepared {len(features_used)} features for HMM: {features_used}")
    
    # Ensure data is 2D array (n_samples, n_features)
    if len(X_scaled.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {X_scaled.shape}")
    
    # Training parameters
    n_states = hmm_config['n_hidden_states']
    covariance_type = hmm_config['covariance_type']
    n_attempts = hmm_config['n_init']
    
    best_score = float('-inf')
    best_model = None
    
    for attempt in range(1, n_attempts + 3):
        logger.info(f"HMM training attempt {attempt}/{n_attempts}")
        try:
            # Initialize model with smart starting values
            hmm_model = initialize_hmm(n_states, X_scaled.shape[1], X_scaled, covariance_type)
            
            # Ensure all model attributes are properly set
            if not hasattr(hmm_model, 'n_features') or hmm_model.n_features != X_scaled.shape[1]:
                hmm_model.n_features = X_scaled.shape[1]
            
            # Fit model with sequences
            lengths = [len(X_scaled)]  # Single continuous sequence
            hmm_model.fit(X_scaled, lengths)
            
            score = hmm_model.score(X_scaled)
            logger.info(f"Attempt {attempt} score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_model = hmm_model
                
        except Exception as e:
            logger.warning(f"Training attempt {attempt} failed: {str(e)}")
            continue
    
    if best_model is None:
        raise RuntimeError("All HMM training attempts failed")
    
    logger.info(f"Selected best model with score: {best_score:.2f}")
    
    # Decode states
    states = best_model.predict(X_scaled)
    state_probs = best_model.predict_proba(X_scaled)
    
    # Calculate state characteristics
    state_stats = {}
    for state in range(n_states):
        mask = (states == state)
        state_data = data[mask]
        
        # Calculate mean characteristics for this state
        means = {col: float(state_data[col].mean()) for col in features_used}
        
        # Interpret state characteristics
        characteristics = []
        
        # Trend interpretation
        if means['log_return'] > 0.001:
            characteristics.append('bullish')
        elif means['log_return'] < -0.001:
            characteristics.append('bearish')
        else:
            characteristics.append('neutral')
            
        # Volatility interpretation
        vol = means['volatility_5']
        if vol < np.percentile(data['volatility_5'], 33):
            characteristics.append('low volatility')
        elif vol > np.percentile(data['volatility_5'], 66):
            characteristics.append('high volatility')
        else:
            characteristics.append('medium volatility')
            
        # RSI interpretation
        if means['rsi'] < 30:
            characteristics.append('oversold')
        elif means['rsi'] > 70:
            characteristics.append('overbought')
        else:
            characteristics.append('neutral RSI')
            
        # MACD interpretation
        if means['macd'] > 0:
            characteristics.append('strong momentum')
        else:
            characteristics.append('weak momentum')
        
        state_stats[f"state_{state}"] = {
            'means': means,
            'characteristics': characteristics,
            'probability': float(np.mean(state_probs[:, state])),
            'duration_stats': {
                'mean': float(get_state_durations(states == state).mean()),
                'std': float(get_state_durations(states == state).std()),
                'max': int(get_state_durations(states == state).max())
            }
        }
    
    # Save interpretations
    results = {
        'model_info': {
            'n_states': n_states,
            'covariance_type': covariance_type,
            'score': float(best_score),
            'features_used': features_used,
            'convergence_attempts': n_attempts
        },
        'states': state_stats,
        'state_sequence': states.tolist(),
        'state_probabilities': state_probs.tolist(),
        'transition_matrix': best_model.transmat_.tolist(),
        'means': best_model.means_.tolist(),
        'covars': [cov.tolist() for cov in best_model.covars_],
        'stationary_distribution': get_stationary_distribution(best_model.transmat_).tolist()
    }
    
    # Save to dashboard
    output_path = Path('outputs/web_dashboard/hmm_interpretations.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save HMM interpretations: {str(e)}")
    
    logger.info("HMM analysis completed successfully")
    return results

def get_state_durations(state_mask: np.ndarray) -> np.ndarray:
    """Calculate durations of consecutive True values in boolean array.
    
    Args:
        state_mask: Boolean array indicating state membership
        
    Returns:
        Array of state duration lengths
    """
    if len(state_mask) == 0:
        return np.array([0])
        
    # Convert to int array
    state_int = state_mask.astype(int)
    
    # Add sentinel values to handle edge cases
    padded = np.r_[0, state_int, 0]
    
    # Find boundaries between different states
    changes = np.where(np.diff(padded) != 0)[0]
    
    # If no True values found
    if len(changes) == 0:
        return np.array([0] if state_int[0] == 0 else [len(state_int)])
    
    # Convert change points to runs
    changes = changes.reshape(-1, 2)
    
    # Calculate durations of True runs
    durations = changes[:, 1] - changes[:, 0]
    
    return durations

def get_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Calculate the stationary distribution of a Markov chain."""
    eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
    stationary = eigenvecs[:, np.argmax(np.real(eigenvals))]
    return np.real(stationary / np.sum(stationary))

def run_hmm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run HMM analysis on market data."""
    result = df.copy()
    config = get_config()
    
    if not config['hmm_analysis']['enabled']:
        logger.info("HMM analysis disabled in config")
        return result
    
    try:
        # Train HMM model
        hmm_results = train_hmm(result, config)
        
        # Add state sequence to DataFrame
        result['hmm_regime'] = hmm_results['state_sequence']
        result['hmm_regime_prob'] = [
            max(probs) for probs in hmm_results['state_probabilities']
        ]
        
        # Add regime characteristics if in full_analysis mode
        if config.get('pipeline_mode') == 'full_analysis':
            regime_chars = {
                i: '_'.join(info['characteristics'])
                for i, (state, info) in enumerate(hmm_results['states'].items())
            }
            result['hmm_regime_char'] = result['hmm_regime'].map(regime_chars)
        
        logger.info("HMM regime detection completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"HMM analysis failed: {str(e)}")
        # Return original DataFrame if analysis fails
        return result