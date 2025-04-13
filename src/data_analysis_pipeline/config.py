"""Configuration loader for the data analysis pipeline."""
import logging
from pathlib import Path
from typing import Dict, Any, List
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Configuration validation constants
ALLOWED_COV_TYPES = ['spherical', 'diag', 'full', 'tied']
ALLOWED_SCALING_METHODS = ['standard', 'minmax', 'robust']

def validate_hmm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate HMM analysis configuration parameters.
    
    Args:
        config: Raw HMM configuration dictionary
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not config:
        return {'enabled': False}
        
    if not isinstance(config.get('n_hidden_states', 2), int) or config.get('n_hidden_states', 2) < 2:
        raise ValueError("n_hidden_states must be an integer >= 2")
        
    cov_type = config.get('covariance_type', 'diag')
    if cov_type not in ALLOWED_COV_TYPES:
        raise ValueError(f"covariance_type must be one of: {ALLOWED_COV_TYPES}")
        
    if 'feature_scaling' in config:
        scaling_method = config['feature_scaling'].get('method', 'standard')
        if scaling_method not in ALLOWED_SCALING_METHODS:
            raise ValueError(f"scaling method must be one of: {ALLOWED_SCALING_METHODS}")
            
    return config

def validate_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate LLM output generation configuration parameters.
    
    Args:
        config: Raw LLM configuration dictionary
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not config:
        return {'enabled': False}
        
    if not isinstance(config.get('summary_periods', []), list):
        raise ValueError("summary_periods must be a list of integers")
        
    if not all(isinstance(p, int) and p > 0 for p in config.get('summary_periods', [])):
        raise ValueError("summary_periods must contain positive integers")
        
    output_file = config.get('output_file', '')
    if not output_file.endswith('.md'):
        raise ValueError("output_file must have .md extension for Markdown format")
        
    return config

@dataclass
class PipelineConfig:
    """Data class to store pipeline configuration."""
    input_config: Dict[str, Any]
    processing_config: Dict[str, Any]
    feature_config: Dict[str, Any]
    output_config: Dict[str, Any]
    logging_config: Dict[str, Any]
    anomaly_detection: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    visualization: Dict[str, Any]
    hmm_analysis: Dict[str, Any]
    llm_output_generation: Dict[str, Any]

def load_config(config_path: str = "config.yaml") -> PipelineConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        PipelineConfig object containing all settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file: {e}")
            
    # Load and validate HMM config
    hmm_config = validate_hmm_config(config.get('hmm_analysis', {}))
    
    # Load and validate LLM config
    llm_config = validate_llm_config(config.get('llm_output_generation', {}))
            
    return PipelineConfig(
        input_config=config.get('input', {}),
        processing_config={
            'time_alignment': {
                'resample_freq': 'min',  # Resample frequency (e.g., 'min', '5min', etc.)
                'fill_method': 'ffill'   # Forward fill missing values
            },
            'feature_engineering': {
                'window_sizes': {
                    'short': 5,
                    'medium': 20,
                    'long': 50
                },
                'volatility': {
                    'window': 20,
                    'std_threshold': 2.0
                }
            },
            'regime_detection': {
                'enabled': True,
                'volatility': {
                    'source': 'atr',
                    'window': 14,
                    'thresholds': {
                        'low': 0.5,
                        'high': 1.5
                    }
                },
                'trend': {
                    'enabled': True,
                    'ma_window': 20,
                    'threshold': 25
                }
            }
        },
        feature_config=config.get('features', {}),
        output_config=config.get('output', {}),
        logging_config=config.get('logging', {}),
        anomaly_detection=config.get('anomaly_detection', {}),
        statistical_analysis=config.get('statistical_analysis', {}),
        visualization=config.get('visualization', {}),
        hmm_analysis=hmm_config,
        llm_output_generation=llm_config
    )

# Singleton instance for global access
_config: PipelineConfig = None

def get_config() -> PipelineConfig:
    """Get the global configuration instance.
    
    Returns:
        PipelineConfig object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config