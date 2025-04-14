"""Output generation module for saving pipeline results."""
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional

from .config import get_config

logger = logging.getLogger(__name__)

def validate_feature_data(df: pd.DataFrame) -> bool:
    """Validate feature DataFrame before saving.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if validation passes
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            logger.error("Feature DataFrame is empty")
            return False
            
        # Check for required columns
        required_cols = [
            'price', 'volume',  # Base columns
            'composite_signal', # Trading signals
            'regime_volatility', 'regime_trend'  # Market regimes
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Check for invalid values in key columns
        if df['composite_signal'].isnull().any():
            logger.error("NaN values found in composite_signal")
            return False
            
        if not df['composite_signal'].between(-100, 100).all():
            logger.error("composite_signal values outside valid range [-100, 100]")
            return False
            
        # Check index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("Index is not DatetimeIndex")
            return False
            
        if not df.index.is_monotonic_increasing:
            logger.error("Index is not sorted")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return False

def generate_market_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate basic market summary statistics.
    
    Args:
        df: Feature DataFrame
        
    Returns:
        Dictionary containing summary statistics
    """
    try:
        summary = {
            'time_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat(),
                'periods': len(df)
            },
            'price_stats': df['price'].describe().to_dict(),
            'volume_stats': df['volume'].describe().to_dict(),
            'regime_distribution': {
                'volatility': df['regime_volatility'].value_counts().to_dict(),
                'trend': df['regime_trend'].value_counts().to_dict()
            },
            'signal_stats': {
                'mean': float(df['composite_signal'].mean()),
                'std': float(df['composite_signal'].std()),
                'min': float(df['composite_signal'].min()),
                'max': float(df['composite_signal'].max()),
                'strong_buy_signals': int((df['composite_signal'] > 50).sum()),
                'strong_sell_signals': int((df['composite_signal'] < -50).sum())
            }
        }
        return summary
        
    except Exception as e:
        logger.error(f"Error generating market summary: {str(e)}")
        return {}

def save_outputs(feature_data: pd.DataFrame) -> bool:
    """Save pipeline outputs with validation and error handling.
    
    Args:
        feature_data: DataFrame containing features to save
        
    Returns:
        bool: True if save operation succeeds
    """
    try:
        config = get_config()
        
        # Validate feature data
        if not validate_feature_data(feature_data):
            logger.error("Feature data validation failed")
            return False
        
        # Create output directories
        output_dir = Path('outputs/trading_model')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features to parquet with proper error handling
        features_path = output_dir / 'features.parquet'
        try:
            feature_data.to_parquet(
                features_path,
                compression='snappy',
                index=True
            )
            logger.info(f"Features saved to {features_path}")
        except Exception as e:
            logger.error(f"Failed to save features: {str(e)}")
            return False
        
        # Generate and save market summary
        try:
            summary = generate_market_summary(feature_data)
            summary_path = output_dir / 'market_summary.txt'
            
            with open(summary_path, 'w') as f:
                f.write("Market Data Summary\n")
                f.write("=================\n\n")
                
                # Time range
                f.write(f"Time Range: {summary['time_range']['start']} to {summary['time_range']['end']}\n")
                f.write(f"Number of periods: {summary['time_range']['periods']}\n\n")
                
                # Price and volume stats
                f.write("Price Statistics:\n")
                for k, v in summary['price_stats'].items():
                    f.write(f"{k}: {v:.2f}\n")
                f.write("\nVolume Statistics:\n")
                for k, v in summary['volume_stats'].items():
                    f.write(f"{k}: {v:.2f}\n")
                    
                # Market regimes
                f.write("\nMarket Regime Distribution:\n")
                f.write("\nVolatility Regimes:\n")
                for regime, count in summary['regime_distribution']['volatility'].items():
                    f.write(f"{regime}: {count} periods\n")
                f.write("\nTrend Regimes:\n")
                for regime, count in summary['regime_distribution']['trend'].items():
                    f.write(f"{regime}: {count} periods\n")
                    
                # Signal statistics
                f.write("\nTrading Signal Statistics:\n")
                f.write(f"Mean Signal: {summary['signal_stats']['mean']:.2f}\n")
                f.write(f"Signal Std Dev: {summary['signal_stats']['std']:.2f}\n")
                f.write(f"Strong Buy Signals: {summary['signal_stats']['strong_buy_signals']}\n")
                f.write(f"Strong Sell Signals: {summary['signal_stats']['strong_sell_signals']}\n")
                
            logger.info(f"Market summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save market summary: {str(e)}")
            # Non-critical error, continue
        
        return True
        
    except Exception as e:
        logger.error(f"Error in save_outputs: {str(e)}")
        return False