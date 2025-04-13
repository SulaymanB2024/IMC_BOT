"""Hidden Markov Model analysis for market state detection."""
import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import RobustScaler
from pathlib import Path

from .config import get_config

logger = logging.getLogger(__name__)

def prepare_features_for_hmm(df: pd.DataFrame, config: dict) -> tuple:
    """Prepare features for HMM analysis."""
    # Get features to use
    features = config['hmm_analysis']['features_to_use']
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        raise ValueError("No valid features available for HMM analysis")
    
    # Extract and scale features
    X = df[available_features].copy()
    X = X.fillna(method='ffill').fillna(method='bfill')
    
    if config['hmm_analysis']['feature_scaling']['enabled']:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    return X_scaled, available_features

def fit_hmm(X: np.ndarray, config: dict) -> hmm.GaussianHMM:
    """Fit HMM model to the data."""
    model = hmm.GaussianHMM(
        n_components=config['hmm_analysis']['n_hidden_states'],
        covariance_type=config['hmm_analysis']['covariance_type'],
        n_iter=config['hmm_analysis']['n_iterations'],
        random_state=config['hmm_analysis']['random_state']
    )
    
    # Try multiple initializations
    best_score = float('-inf')
    best_model = None
    
    for i in range(config['hmm_analysis']['n_init']):
        try:
            model.fit(X)
            score = model.score(X)
            
            if score > best_score:
                best_score = score
                best_model = model
                
        except Exception as e:
            logger.warning(f"HMM fitting attempt {i+1} failed: {str(e)}")
            continue
    
    if best_model is None:
        raise RuntimeError("Failed to fit HMM model")
    
    return best_model

def interpret_states(model: hmm.GaussianHMM, feature_names: list) -> dict:
    """Interpret the characteristics of each HMM state."""
    interpretations = {}
    
    for state in range(model.n_components):
        state_mean = model.means_[state]
        state_cov = model.covars_[state] if model.covariance_type == 'diag' else np.diag(model.covars_[state])
        
        # Create interpretation for each state
        state_chars = {}
        for i, feature in enumerate(feature_names):
            state_chars[feature] = {
                'mean': state_mean[i],
                'std': np.sqrt(state_cov[i])
            }
        
        interpretations[f'state_{state}'] = state_chars
    
    # Save interpretations
    output_path = Path('outputs/web_dashboard/hmm_interpretations.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(interpretations).to_json(output_path)
    
    return interpretations

def run_hmm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run HMM analysis on market data."""
    result = df.copy()
    config = get_config()
    
    if not config.get('hmm_analysis', {}).get('enabled', False):
        return result
    
    try:
        # Prepare features
        X_scaled, feature_names = prepare_features_for_hmm(result, config)
        logger.info(f"Prepared {len(feature_names)} features for HMM analysis")
        
        # Fit model
        model = fit_hmm(X_scaled, config)
        logger.info("HMM model fitted successfully")
        
        # Predict states
        states = model.predict(X_scaled)
        state_probs = model.predict_proba(X_scaled)
        
        # Add predictions to results
        result[config['hmm_analysis']['output_column_name']] = states
        
        # Add state probabilities
        for i in range(model.n_components):
            result[f'hmm_state_{i}_prob'] = state_probs[:, i]
        
        # Generate and save state interpretations
        interpretations = interpret_states(model, feature_names)
        logger.info("State interpretations generated")
        
        # Label states based on volatility and trend
        volatility_state = pd.Series(states).map({
            i: 'volatile' if interpretations[f'state_{i}']['volatility_20']['mean'] > 0 else 'calm'
            for i in range(model.n_components)
        })
        
        trend_state = pd.Series(states).map({
            i: 'bullish' if interpretations[f'state_{i}']['log_return']['mean'] > 0 else 'bearish'
            for i in range(model.n_components)
        })
        
        result['hmm_regime_label'] = volatility_state + '_' + trend_state
        
        n_states = len(np.unique(states))
        logger.info(f"HMM analysis complete with {n_states} states identified")
        
        return result
        
    except Exception as e:
        logger.error(f"HMM analysis failed: {str(e)}")
        return result