"""Configuration management for the data analysis pipeline."""
import logging
from pathlib import Path
import yaml
from typing import Literal, Dict, Any

logger = logging.getLogger(__name__)

_config = None

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values."""
    # Validate pipeline_mode
    valid_modes = ('trading', 'full_analysis')
    pipeline_mode = config.get('pipeline_mode', 'full_analysis')
    if pipeline_mode not in valid_modes:
        logger.warning(f"Invalid pipeline_mode '{pipeline_mode}', defaulting to 'full_analysis'")
        config['pipeline_mode'] = 'full_analysis'

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set defaults and validate
        if 'pipeline_mode' not in config:
            config['pipeline_mode'] = 'full_analysis'
            logger.info("Pipeline mode not specified, defaulting to 'full_analysis'")
            
        validate_config(config)
        logger.info(f"Configuration loaded from {config_path} with mode: {config['pipeline_mode']}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        raise

def get_config(config_path: str = "config.yaml") -> dict:
    """Get configuration, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config