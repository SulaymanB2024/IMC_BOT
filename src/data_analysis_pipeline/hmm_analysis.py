"""Hidden Markov Model analysis module for market regime detection."""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .config import get_config

logger = logging.getLogger(__name__)

def prepare_features(df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Prepare and scale features for HMM analysis."""
    try:
        features_to_use = config['features_to_use']
        feature_matrix = pd.DataFrame(index=df.index)
        
        # Add log return if needed
        if 'log_return' in features_to_use:
            feature_matrix['log_return'] = np.log(df['price']).diff()
        
        # Add other requested features
        for feature in features_to_use:
            if feature != 'log_return':
                if feature in df.columns:
                    feature_matrix[feature] = df[feature]
                else:
                    logger.warning(f"Feature {feature} not found in data")
                    features_to_use.remove(feature)
        
        if not features_to_use:
            raise ValueError("No valid features available for HMM analysis")
        
        # Handle missing values
        feature_matrix = feature_matrix.ffill().bfill()
        
        # Remove any remaining NaN rows
        feature_matrix = feature_matrix.dropna()
        
        if len(feature_matrix) < config['n_hidden_states'] * 3:
            raise ValueError(f"Insufficient data points ({len(feature_matrix)}) for {config['n_hidden_states']} states")
        
        # Scale features
        if config['feature_scaling']['enabled']:
            method = config['feature_scaling']['method']
            scaler = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler()
            }.get(method)
            
            if scaler is None:
                raise ValueError(f"Unknown scaling method: {method}")
                
            scaled_features = scaler.fit_transform(feature_matrix)
        else:
            scaled_features = feature_matrix.values
        
        return scaled_features, features_to_use
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {str(e)}")
        return None

def train_hmm_model(X: np.ndarray, config: Dict[str, Any]) -> Optional[hmm.GaussianHMM]:
    """Train HMM model with multiple initialization attempts."""
    best_model = None
    best_score = float('-inf')
    n_attempts = 3
    
    for attempt in range(n_attempts):
        try:
            model = hmm.GaussianHMM(
                n_components=config['n_hidden_states'],
                covariance_type=config['covariance_type'],
                n_iter=config['n_iterations'],
                random_state=config['random_state'] + attempt
            )
            
            model.fit(X)
            score = model.score(X)
            
            if score > best_score:
                best_model = model
                best_score = score
                
            logger.info(f"HMM training attempt {attempt + 1}: score = {score:.2f}")
            
        except Exception as e:
            logger.warning(f"HMM training attempt {attempt + 1} failed: {str(e)}")
            continue
    
    if best_model is None:
        logger.error("All HMM training attempts failed")
        return None
        
    logger.info(f"Best HMM model achieved score: {best_score:.2f}")
    return best_model

def interpret_states(model: hmm.GaussianHMM, feature_names: List[str]) -> Dict[str, Any]:
    """Interpret the learned HMM states based on their parameters.
    
    Args:
        model: Trained HMM model
        feature_names: Names of features used in training
        
    Returns:
        Dictionary containing state interpretations
    """
    n_states = model.n_components
    interpretations = {}
    
    # Get standardized feature importance for each state
    means = model.means_
    covars = model.covars_
    
    # For each state
    for state in range(n_states):
        state_means = means[state]
        state_info = {
            'means': dict(zip(feature_names, state_means.tolist())),
            'characteristics': []
        }
        
        # Interpret state characteristics based on means
        characteristics = []
        for feat_idx, feat_name in enumerate(feature_names):
            mean_val = state_means[feat_idx]
            
            # Interpretation logic based on feature type
            if feat_name == 'log_return':
                if mean_val > 0.001:
                    characteristics.append('bullish')
                elif mean_val < -0.001:
                    characteristics.append('bearish')
                else:
                    characteristics.append('neutral')
                    
            elif 'volatility' in feat_name:
                if mean_val > 0.5:  # Assuming scaled values
                    characteristics.append('high volatility')
                elif mean_val < -0.5:
                    characteristics.append('low volatility')
                else:
                    characteristics.append('medium volatility')
                    
            elif feat_name == 'rsi':
                if mean_val > 0.7:  # Assuming scaled RSI
                    characteristics.append('overbought')
                elif mean_val < 0.3:
                    characteristics.append('oversold')
                    
            elif feat_name == 'macd':
                if mean_val > 0.2:
                    characteristics.append('strong momentum')
                elif mean_val < -0.2:
                    characteristics.append('weak momentum')
        
        state_info['characteristics'] = characteristics
        interpretations[f'state_{state}'] = state_info
    
    # Add transition probabilities and stationary distribution
    interpretations['transition_matrix'] = model.transmat_.tolist()
    
    # Calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(model.transmat_.T)
    stationary = eigenvecs[:, np.argmax(eigenvals)].real
    stationary = stationary / stationary.sum()
    interpretations['stationary_distribution'] = stationary.tolist()
    
    return interpretations

def run_hmm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run HMM analysis on the trading data."""
    logger.info("Starting HMM analysis")
    config = get_config().hmm_analysis
    
    if not config.get('enabled', False):
        logger.info("HMM analysis disabled in config")
        return df
    
    try:
        # Prepare features with error handling
        prep_result = prepare_features(df, config)
        if prep_result is None:
            logger.warning("Feature preparation failed, skipping HMM analysis")
            return df
            
        X, feature_names = prep_result
        logger.info(f"Prepared {len(feature_names)} features for HMM: {feature_names}")
        
        # Try multiple initializations
        best_model = None
        best_score = float('-inf')
        n_attempts = 3
        
        for attempt in range(n_attempts):
            try:
                logger.info(f"HMM training attempt {attempt + 1}/{n_attempts}")
                model = hmm.GaussianHMM(
                    n_components=config['n_hidden_states'],
                    covariance_type=config['covariance_type'],
                    n_iter=config['n_iterations'],
                    random_state=config['random_state'] + attempt
                )
                
                model.fit(X)
                score = model.score(X)
                
                if score > best_score:
                    best_model = model
                    best_score = score
                    
                logger.info(f"Attempt {attempt + 1} score: {score:.2f}")
                
                if model.monitor_.converged_:
                    logger.info(f"Converged in {model.monitor_.iter_} iterations")
                else:
                    logger.warning("Did not converge")
                
            except Exception as e:
                logger.warning(f"Training attempt {attempt + 1} failed: {str(e)}")
                continue
        
        if best_model is None:
            logger.error("All HMM training attempts failed")
            return df
            
        logger.info(f"Selected best model with score: {best_score:.2f}")
        
        # Generate predictions with error handling
        try:
            hidden_states = best_model.predict(X)
            
            # Create state mappings based on characteristics
            state_chars = interpret_states(best_model, feature_names)
            state_mapping = {}
            
            for state in range(config['n_hidden_states']):
                state_info = state_chars[f'state_{state}']
                chars = state_info.get('characteristics', [])
                
                # Map states to meaningful labels if possible
                if 'bullish' in chars and 'high volatility' in chars:
                    label = 'volatile_bullish'
                elif 'bearish' in chars and 'high volatility' in chars:
                    label = 'volatile_bearish'
                elif 'neutral' in chars and 'low volatility' in chars:
                    label = 'calm_neutral'
                else:
                    label = f"state_{state}"
                
                state_mapping[state] = label
            
            # Add predictions to DataFrame with meaningful labels
            result = df.copy()
            state_series = pd.Series(
                [state_mapping.get(state, f"state_{state}") for state in hidden_states],
                index=df.index[:len(hidden_states)]
            )
            
            # Add both numeric and labeled states
            result[config['output_column_name']] = hidden_states
            result[f"{config['output_column_name']}_label"] = state_series
            
            # Save detailed interpretations
            save_dir = Path(get_config().output_config['dashboard']['path'])
            save_dir.mkdir(parents=True, exist_ok=True)
            
            interpretations = {
                'model_info': {
                    'n_states': config['n_hidden_states'],
                    'covariance_type': config['covariance_type'],
                    'score': float(best_score),
                    'features_used': feature_names
                },
                'states': state_chars,
                'state_mapping': state_mapping,
                'transition_matrix': best_model.transmat_.tolist(),
                'means': best_model.means_.tolist(),
                'covars': best_model.covars_.tolist() if hasattr(best_model, 'covars_') else None
            }
            
            with open(save_dir / 'hmm_interpretations.json', 'w') as f:
                json.dump(interpretations, f, indent=2, cls=NumpyEncoder)
            
            logger.info("HMM analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction phase: {str(e)}")
            return df
            
    except Exception as e:
        logger.error(f"HMM analysis failed: {str(e)}")
        return df

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)