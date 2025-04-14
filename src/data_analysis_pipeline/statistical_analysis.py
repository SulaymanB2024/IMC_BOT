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

def prepare_for_granger(series: pd.Series) -> pd.Series:
    """Prepare a time series for Granger causality testing by ensuring stationarity.
    
    Args:
        series: Original time series
        
    Returns:
        Stationary version of the series
    """
    # First try: use the series as is
    adf_result = adfuller(series.dropna())[1]
    
    if adf_result < 0.05:
        return series
        
    # Second try: take first difference
    diff = series.diff().dropna()
    adf_result = adfuller(diff)[1]
    
    if adf_result < 0.05:
        logger.info(f"Using first difference for {series.name}")
        return diff
        
    # Third try: take log returns
    if (series > 0).all():
        log_ret = np.log(series).diff().dropna()
        adf_result = adfuller(log_ret)[1]
        
        if adf_result < 0.05:
            logger.info(f"Using log returns for {series.name}")
            return log_ret
            
    logger.warning(f"Could not achieve stationarity for {series.name}")
    return series

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
        # Prepare dependent variable (take mean if multiple)
        dep_series = data[dependent].mean(axis=1) if len(dependent) > 1 else data[dependent[0]]
        dep_series.name = '_'.join(dependent)
        
        # Prepare independent variable (take mean if multiple)
        ind_series = data[independent].mean(axis=1) if len(independent) > 1 else data[independent[0]]
        ind_series.name = '_'.join(independent)
        
        # Ensure stationarity
        dep_stationary = prepare_for_granger(dep_series)
        ind_stationary = prepare_for_granger(ind_series)
        
        # Combine and clean data
        test_data = pd.DataFrame({
            'dependent': dep_stationary,
            'independent': ind_stationary
        }).dropna()
        
        # Check if we have enough data
        min_observations = max_lag * 3  # Require at least 3 observations per lag
        if len(test_data) < min_observations:
            raise ValueError(f"Insufficient observations ({len(test_data)}) for max_lag={max_lag}")
            
        # Run test
        results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        
        # Extract results
        granger_results = {}
        for lag in range(1, max_lag + 1):
            # Get F-test results
            f_stat = results[lag][0]['ssr_ftest'][0]
            p_value = results[lag][0]['ssr_ftest'][1]
            granger_results[str(lag)] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
        return {
            'results_by_lag': granger_results,
            'dependent_vars': dependent,
            'independent_vars': independent,
            'n_observations': len(test_data),
            'transformation': {
                'dependent': 'original' if dep_series.equals(dep_stationary) else 
                            'diff' if dep_series.diff().dropna().equals(dep_stationary) else 'log_returns',
                'independent': 'original' if ind_series.equals(ind_stationary) else 
                             'diff' if ind_series.diff().dropna().equals(ind_stationary) else 'log_returns'
            }
        }
        
    except Exception as e:
        logger.error(f"Granger causality test failed: {str(e)}")
        return {
            'error': str(e),
            'dependent_vars': dependent,
            'independent_vars': independent
        }

def run_statistical_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Run statistical tests on the data."""
    logger.info("Running statistical tests")
    results = {
        'stationarity_tests': {},
        'granger_causality_tests': {}
    }
    
    try:
        config = get_config()
        if not hasattr(config, 'statistical_analysis'):
            logger.error("Missing statistical_analysis configuration")
            return results
            
        # Validate configuration
        stat_config = config.statistical_analysis
        if not isinstance(stat_config, dict):
            logger.error("Invalid statistical_analysis configuration format")
            return results
            
        required_keys = ['granger_causality_tests', 'pairs_to_test', 'max_lag']
        if not all(key in stat_config.get('granger_causality_tests', {}) for key in required_keys):
            logger.error("Missing required configuration keys for Granger causality tests")
            return results
        
        # ADF test for stationarity
        for col in ['price', 'volume']:
            if col not in df.columns:
                continue
                
            try:
                adf_result = adfuller(df[col].dropna())
                results['stationarity_tests'][col] = {
                    'statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05  # Standard significance level
                }
            except Exception as e:
                logger.warning(f"ADF test failed for {col}: {str(e)}")
                continue
        
        # Granger causality tests with improved error handling
        granger_config = stat_config['granger_causality_tests']
        test_pairs = granger_config['pairs_to_test']
        maxlag = min(
            granger_config['max_lag'],
            len(df) // 10  # More conservative max lag based on data size
        )
        
        logger.info(f"Running Granger causality tests with max_lag={maxlag}")
        
        for dep, ind in test_pairs:
            # Validate variable lists
            if not isinstance(dep, list) or not isinstance(ind, list):
                logger.warning(f"Invalid test pair format: {dep}->{ind}")
                continue
                
            # Skip if any variables are missing
            missing_vars = [x for x in dep + ind if x not in df.columns]
            if missing_vars:
                logger.warning(f"Skipping Granger test {ind}->{dep}: missing columns {missing_vars}")
                continue
                
            try:
                gc_result = run_granger_test(df, dep, ind, maxlag)
                if 'error' not in gc_result:
                    results['granger_causality_tests'][f"{'_'.join(ind)}_to_{'_'.join(dep)}"] = gc_result
                    logger.info(f"Completed Granger test for {ind}->{dep}")
                else:
                    logger.warning(f"Granger test failed for {ind}->{dep}: {gc_result['error']}")
                    
            except Exception as e:
                logger.warning(f"Granger test failed for {ind}->{dep}: {str(e)}")
        
        # Save results with proper error handling
        try:
            output_dir = Path(config.output_config['dashboard']['path'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'statistical_test_results.json', 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
                
            logger.info("Statistical tests completed and results saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save statistical test results: {str(e)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Statistical tests failed: {str(e)}")
        return results
        
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)