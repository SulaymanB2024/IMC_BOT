"""Statistical analysis module for testing time series properties."""
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from .config import get_config

logger = logging.getLogger(__name__)

def run_adf_test(series: pd.Series) -> Dict[str, Any]:
    """Run Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        
    Returns:
        Dictionary containing test results
    """
    try:
        # Run ADF test
        result = adfuller(series.dropna())
        
        # Extract results
        test_stat, p_value, n_lags, n_obs, critical_values = result
        
        # Determine if stationary at different confidence levels
        is_stationary = {
            '1%': test_stat < critical_values['1%'],
            '5%': test_stat < critical_values['5%'],
            '10%': test_stat < critical_values['10%']
        }
        
        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'n_lags': n_lags,
            'n_observations': n_obs,
            'critical_values': critical_values,
            'is_stationary': is_stationary
        }
        
    except Exception as e:
        logger.error(f"ADF test failed: {str(e)}")
        return {
            'error': str(e),
            'test_statistic': None,
            'p_value': None,
            'is_stationary': {'1%': None, '5%': None, '10%': None}
        }

def run_granger_test(data: pd.DataFrame, dependent: List[str], independent: List[str], 
                    max_lag: int) -> Dict[str, Any]:
    """Run Granger causality test.
    
    Args:
        data: DataFrame containing the variables
        dependent: List of dependent variable column names
        independent: List of independent variable column names
        max_lag: Maximum number of lags to test
        
    Returns:
        Dictionary containing test results for each lag
    """
    try:
        # Prepare data for test
        test_data = pd.concat([
            data[dependent].mean(axis=1),  # If multiple dependent vars, take mean
            data[independent].mean(axis=1)  # Same for independent vars
        ], axis=1)
        test_data = test_data.dropna()
        
        # Run test
        results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        
        # Extract p-values for each lag
        granger_results = {}
        for lag in range(1, max_lag + 1):
            # Get F-test results (index 0) and its p-value (index 1)
            f_stat = results[lag][0]['ssr_ftest'][0]
            p_value = results[lag][0]['ssr_ftest'][1]
            granger_results[str(lag)] = {
                'f_statistic': f_stat,
                'p_value': p_value
            }
            
        return {
            'results_by_lag': granger_results,
            'dependent_vars': dependent,
            'independent_vars': independent
        }
        
    except Exception as e:
        logger.error(f"Granger causality test failed: {str(e)}")
        return {
            'error': str(e),
            'results_by_lag': {},
            'dependent_vars': dependent,
            'independent_vars': independent
        }

def run_statistical_tests(df: pd.DataFrame) -> None:
    """Run all configured statistical tests on the data.
    
    Args:
        df: DataFrame containing features to analyze
    """
    logger.info("Starting statistical analysis")
    config = get_config()
    stat_config = config.statistical_analysis
    
    # Check if analysis is enabled
    if not stat_config.get('enabled', False):
        logger.info("Statistical analysis disabled in config")
        return
    
    results = {'stationarity_tests': {}, 'granger_causality_tests': {}}
    
    # Run stationarity tests
    if stat_config['stationarity_tests']['run_adf']:
        logger.info("Running stationarity tests")
        columns = stat_config['stationarity_tests']['columns_to_test']
        sig_level = stat_config['stationarity_tests']['adf_significance_level']
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping ADF test")
                continue
                
            logger.info(f"Testing stationarity of {col}")
            adf_result = run_adf_test(df[col])
            results['stationarity_tests'][col] = adf_result
            
            # Log result summary
            if adf_result.get('p_value') is not None:
                is_stationary = adf_result['p_value'] < sig_level
                logger.info(f"{col}: {'Stationary' if is_stationary else 'Non-stationary'} "
                          f"(p-value: {adf_result['p_value']:.4f})")
    
    # Run Granger causality tests
    if stat_config['granger_causality_tests']['run_granger']:
        logger.info("Running Granger causality tests")
        pairs = stat_config['granger_causality_tests']['pairs_to_test']
        max_lag = stat_config['granger_causality_tests']['max_lag']
        sig_level = stat_config['granger_causality_tests']['granger_significance_level']
        
        for pair_idx, (dependent, independent) in enumerate(pairs):
            # Check if all columns exist
            if not all(col in df.columns for col in dependent + independent):
                missing = [col for col in dependent + independent if col not in df.columns]
                logger.warning(f"Columns {missing} not found, skipping Granger test for pair {pair_idx}")
                continue
            
            logger.info(f"Testing Granger causality: {' & '.join(independent)} -> {' & '.join(dependent)}")
            granger_result = run_granger_test(df, dependent, independent, max_lag)
            results['granger_causality_tests'][f"pair_{pair_idx}"] = granger_result
            
            # Log result summary
            if 'error' not in granger_result:
                min_p_value = min(lag['p_value'] for lag in granger_result['results_by_lag'].values())
                significant = min_p_value < sig_level
                logger.info(f"Granger causality {'found' if significant else 'not found'} "
                          f"(minimum p-value across lags: {min_p_value:.4f})")
    
    # Save results
    output_path = Path(stat_config['results_output_file'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Statistical analysis results saved to {output_path}")