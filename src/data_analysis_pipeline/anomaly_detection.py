"""Anomaly detection module for identifying unusual market behavior."""
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import get_config

logger = logging.getLogger(__name__)

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect anomalies in the market data using Isolation Forest."""
    result = df.copy()
    config = get_config()
    
    if not config.get('anomaly_detection', {}).get('enabled', False):
        return result
    
    try:
        # Get features to use for anomaly detection
        features = config['anomaly_detection']['features_to_use']
        available_features = [f for f in features if f in result.columns]
        
        if not available_features:
            logger.warning("No valid features found for anomaly detection")
            return result
            
        logger.info(f"Running anomaly detection using features: {available_features}")
        
        # Prepare data
        X = result[available_features].copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize and fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=config['anomaly_detection'].get('contamination', 'auto'),
            n_estimators=config['anomaly_detection'].get('n_estimators', 100),
            max_samples=config['anomaly_detection'].get('max_samples', 'auto'),
            random_state=config['anomaly_detection'].get('random_state', 42)
        )
        
        # Fit and predict
        anomalies = iso_forest.fit_predict(X_scaled)
        
        # Convert predictions to binary flags (-1 for anomaly, 1 for normal)
        output_col = config['anomaly_detection'].get('output_column_name', 'anomaly_flag_iforest')
        result[output_col] = (anomalies == -1).astype(int)
        
        # Calculate anomaly scores (higher score = more anomalous)
        result[f'{output_col}_score'] = -iso_forest.score_samples(X_scaled)
        
        n_anomalies = result[output_col].sum()
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(result)*100:.2f}% of data)")
        
        return result
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        return result