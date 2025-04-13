"""Configuration management for the data analysis pipeline."""
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

_config = None

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
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