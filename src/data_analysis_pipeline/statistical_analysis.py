"""Statistical analysis module for market data."""
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from typing import Dict, List, Tuple

from .config import get_config

logger = logging.getLogger(__name__)

def run_stationarity_tests(df: pd.DataFrame, config: dict) -> Dict:
    """Run Augmented Dickey-Fuller test for stationarity."""
    results = {}
    columns = config['statistical_analysis']['stationarity_tests']['columns_to_test']
    significance = config['statistical_analysis']['stationarity_tests']['adf_significance_level']
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in data")
            continue
            
        # Run ADF test
        try:
            series = df[col].dropna()
            adf_result = adfuller(series)
            
            results[col] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < significance
            }
            
        except Exception as e:
            logger.error(f"ADF test failed for {col}: {str(e)}")
            continue
    
    return results

def run_granger_causality_tests(df: pd.DataFrame, config: dict) -> Dict:
    """Run Granger causality tests between feature pairs."""
    results = {}
    pairs = config['statistical_analysis']['granger_causality_tests']['pairs_to_test']
    max_lag = config['statistical_analysis']['granger_causality_tests']['max_lag']
    significance = config['statistical_analysis']['granger_causality_tests']['granger_significance_level']
    
    for target, features in pairs:
        # Ensure all features exist
        if not all(f in df.columns for f in target + features):
            missing = [f for f in target + features if f not in df.columns]
            logger.warning(f"Skipping Granger test - missing columns: {missing}")
            continue
        
        key = f"{'+'.join(target)} ~ {'+'.join(features)}"
        results[key] = {}
        
        try:
            # Prepare data
            data = pd.concat([df[target], df[features]], axis=1).dropna()
            
            # Run Granger test
            granger_test = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            
            # Extract results for each lag
            for lag in range(1, max_lag + 1):
                test_stats = granger_test[lag][0]
                results[key][f'lag_{lag}'] = {
                    'f_stat': float(test_stats['ssr_ftest'][0]),
                    'p_value': float(test_stats['ssr_ftest'][1]),
                    'significant': test_stats['ssr_ftest'][1] < significance
                }
                
        except Exception as e:
            logger.error(f"Granger causality test failed for {key}: {str(e)}")
            continue
    
    return results

def calculate_correlations(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate correlation matrix for selected features."""
    columns = config['visualization'].get('correlation_heatmap_columns', [])
    if not columns:
        return None
        
    available_cols = [col for col in columns if col in df.columns]
    if not available_cols:
        logger.warning("No valid columns for correlation analysis")
        return None
        
    return df[available_cols].corr()

def run_statistical_tests(df: pd.DataFrame) -> None:
    """Run statistical analysis on market data."""
    config = get_config()
    
    if not config.get('statistical_analysis', {}).get('enabled', False):
        return
    
    try:
        results = {}
        
        # Run stationarity tests
        if config['statistical_analysis']['stationarity_tests'].get('run_adf', True):
            logger.info("Running stationarity tests...")
            results['stationarity'] = run_stationarity_tests(df, config)
        
        # Run Granger causality tests
        if config['statistical_analysis']['granger_causality_tests'].get('run_granger', True):
            logger.info("Running Granger causality tests...")
            results['granger_causality'] = run_granger_causality_tests(df, config)
        
        # Calculate correlations
        logger.info("Calculating correlations...")
        corr_matrix = calculate_correlations(df, config)
        if corr_matrix is not None:
            results['correlation_matrix'] = corr_matrix.to_dict()
        
        # Save results
        output_path = Path(config['statistical_analysis']['results_output_file'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Statistical analysis results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Statistical analysis failed: {str(e)}")
        raise