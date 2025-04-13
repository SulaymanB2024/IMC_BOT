"""Configuration loader for the data analysis pipeline."""
from pathlib import Path
from typing import Dict, Any
import yaml
from dataclasses import dataclass
import os

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
        visualization=config.get('visualization', {})
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