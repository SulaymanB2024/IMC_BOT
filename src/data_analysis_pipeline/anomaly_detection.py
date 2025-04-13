"""Anomaly detection module for identifying unusual trading patterns."""
import logging
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.ensemble import IsolationForest

from .config import get_config

logger = logging.getLogger(__name__)

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect anomalies in the trading data using Isolation Forest.
    
    Args:
        df: DataFrame containing trading features
        
    Returns:
        DataFrame with added anomaly detection column (-1 for anomalies, 1 for normal)
    """
    logger.info("Starting anomaly detection process")
    config = get_config()
    anomaly_config = config.anomaly_detection
    
    # Check if anomaly detection is enabled
    if not anomaly_config.get('enabled', False):
        logger.info("Anomaly detection disabled in config")
        return df
    
    # Copy DataFrame to avoid modifying original
    result = df.copy()
    
    try:
        # Get features to use for anomaly detection
        features = anomaly_config['features_to_use']
        missing_features = [f for f in features if f not in result.columns]
        if missing_features:
            logger.warning(f"Missing features for anomaly detection: {missing_features}")
            features = [f for f in features if f in result.columns]
            
        if not features:
            logger.error("No valid features available for anomaly detection")
            return df
            
        # Prepare feature matrix
        X = result[features].copy()
        
        # Handle missing values in features
        X = X.interpolate().fillna(method='bfill').fillna(method='ffill')
        
        # Initialize and fit Isolation Forest
        iforest = IsolationForest(
            n_estimators=anomaly_config['n_estimators'],
            max_samples=anomaly_config['max_samples'],
            contamination=anomaly_config['contamination'],
            random_state=anomaly_config['random_state']
        )
        
        # Fit and predict
        anomaly_labels = iforest.fit_predict(X)
        
        # Add predictions to DataFrame
        output_col = anomaly_config['output_column_name']
        result[output_col] = anomaly_labels
        
        # Log statistics
        anomaly_count = (anomaly_labels == -1).sum()
        total_count = len(anomaly_labels)
        logger.info(
            f"Anomaly detection complete. Found {anomaly_count} anomalies "
            f"({(anomaly_count/total_count)*100:.2f}% of data)"
        )
        
    except Exception as e:
        logger.error(f"Error during anomaly detection: {str(e)}", exc_info=True)
        return df
        
    return result