"""Module for generating LLM-optimized output from market analysis."""
import logging
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from .config import get_config

logger = logging.getLogger(__name__)

def format_time_series_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Format time series metadata and characteristics.
    
    Args:
        df: DataFrame with time series data
        
    Returns:
        Dictionary with formatted time series context
    """
    return {
        "time_series_characteristics": {
            "start_time": str(df.index[0]),
            "end_time": str(df.index[-1]),
            "frequency": pd.infer_freq(df.index),
            "total_periods": len(df),
            "price_statistics": {
                "mean": float(df['price'].mean()),
                "std": float(df['price'].std()),
                "min": float(df['price'].min()),
                "max": float(df['price'].max()),
                "last_value": float(df['price'].iloc[-1])
            }
        }
    }

def format_regime_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Format market regime information for LLM consumption.
    
    Args:
        df: DataFrame with regime information
        
    Returns:
        Dictionary with formatted regime context
    """
    # Get current and historical regime information
    current_state = {
        "current_volatility": str(df['regime_volatility'].iloc[-1]),
        "current_trend": str(df['regime_trend'].iloc[-1]) if 'regime_trend' in df else "unknown"
    }
    
    # Calculate regime transitions
    regime_changes = {
        "volatility_transitions": [],
        "trend_transitions": []
    }
    
    if 'regime_volatility' in df.columns:
        vol_changes = df[df['regime_volatility'] != df['regime_volatility'].shift()]
        regime_changes["volatility_transitions"] = [
            {
                "timestamp": str(idx),
                "from_regime": str(df['regime_volatility'].shift().loc[idx]),
                "to_regime": str(row)
            }
            for idx, row in vol_changes['regime_volatility'].items()
        ]
    
    if 'regime_trend' in df.columns:
        trend_changes = df[df['regime_trend'] != df['regime_trend'].shift()]
        regime_changes["trend_transitions"] = [
            {
                "timestamp": str(idx),
                "from_regime": str(df['regime_trend'].shift().loc[idx]),
                "to_regime": str(row)
            }
            for idx, row in trend_changes['regime_trend'].items()
        ]
    
    return {
        "market_regimes": {
            "current_state": current_state,
            "regime_transitions": regime_changes
        }
    }

def format_signal_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Format trading signals for LLM consumption.
    
    Args:
        df: DataFrame with trading signals
        
    Returns:
        Dictionary with formatted signal context
    """
    # Get current signal states
    signal_cols = [col for col in df.columns if col.startswith('signal_')]
    current_signals = {
        col: float(df[col].iloc[-1])
        for col in signal_cols
    }
    
    # Get composite signal context
    if 'composite_signal' in df.columns:
        composite = df['composite_signal'].iloc[-1]
        signal_strength = abs(composite)
        signal_direction = "bullish" if composite > 0 else "bearish" if composite < 0 else "neutral"
        
        if signal_strength > 50:
            strength_desc = "strong"
        elif signal_strength > 25:
            strength_desc = "moderate"
        else:
            strength_desc = "weak"
    else:
        composite = 0
        strength_desc = "unknown"
        signal_direction = "unknown"
    
    return {
        "trading_signals": {
            "current_signals": current_signals,
            "composite_signal": {
                "value": float(composite),
                "interpretation": f"{strength_desc} {signal_direction}",
                "confidence": float(signal_strength) if 'signal_strength' in locals() else 0.0
            }
        }
    }

def format_anomaly_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Format anomaly detection results for LLM consumption.
    
    Args:
        df: DataFrame with anomaly detection results
        
    Returns:
        Dictionary with formatted anomaly context
    """
    recent_window = df.iloc[-5:]  # Look at last 5 periods
    recent_anomalies = []
    
    if 'is_anomaly' in df.columns:
        anomaly_periods = recent_window[recent_window['is_anomaly'] == 1].index
        recent_anomalies = [
            {
                "timestamp": str(idx),
                "price": float(df.loc[idx, 'price']),
                "zscore": float(df.loc[idx, 'price_zscore']) if 'price_zscore' in df else None
            }
            for idx in anomaly_periods
        ]
    
    return {
        "anomaly_detection": {
            "recent_anomalies": recent_anomalies,
            "total_anomalies": int(df['is_anomaly'].sum()) if 'is_anomaly' in df else 0,
            "anomaly_rate": float((df['is_anomaly'].sum() / len(df)) * 100) if 'is_anomaly' in df else 0.0
        }
    }

def format_pattern_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Format identified patterns and behaviors for LLM consumption.
    
    Args:
        df: DataFrame with market data and features
        
    Returns:
        Dictionary with formatted pattern context
    """
    patterns = []
    
    # Check for trend patterns
    if 'regime_trend' in df.columns:
        consecutive_trends = df['regime_trend'].value_counts()
        significant_trends = [
            {
                "pattern": trend,
                "occurrence_count": int(count),
                "significance": "high" if count > len(df) * 0.1 else "medium"
            }
            for trend, count in consecutive_trends.items()
            if count >= 3  # At least 3 periods to be considered a pattern
        ]
        patterns.extend(significant_trends)
    
    # Check for volatility patterns
    if 'regime_volatility' in df.columns:
        vol_patterns = df['regime_volatility'].value_counts()
        significant_vol = [
            {
                "pattern": f"{vol}_volatility",
                "occurrence_count": int(count),
                "significance": "high" if count > len(df) * 0.1 else "medium"
            }
            for vol, count in vol_patterns.items()
            if count >= 3
        ]
        patterns.extend(significant_vol)
    
    return {
        "market_patterns": {
            "identified_patterns": patterns,
            "pattern_confidence": "high" if len(patterns) > 0 else "low"
        }
    }

def generate_llm_output(df: pd.DataFrame) -> None:
    """Generate LLM-optimized output from market analysis.
    
    Args:
        df: DataFrame containing all features and analysis results
    """
    logger.info("Generating LLM-optimized output")
    
    # Combine all context components
    llm_context = {
        **format_time_series_context(df),
        **format_regime_context(df),
        **format_signal_context(df),
        **format_anomaly_context(df),
        **format_pattern_context(df)
    }
    
    # Add natural language summary templates
    llm_context["nlg_templates"] = {
        "market_summary": (
            "The market is currently in a {current_state[current_trend]} trend with "
            "{current_state[current_volatility]} volatility. Trading signals indicate a "
            "{trading_signals[composite_signal][interpretation]} stance with "
            "{trading_signals[composite_signal][confidence]:.1f}% confidence."
        ),
        "anomaly_alert": (
            "There have been {anomaly_detection[total_anomalies]} anomalous events "
            "detected, representing {anomaly_detection[anomaly_rate]:.1f}% of all "
            "observations. {recent_anomalies_text}"
        ),
        "pattern_insight": (
            "Analysis has identified {market_patterns[identified_patterns][0][occurrence_count]} "
            "instances of {market_patterns[identified_patterns][0][pattern]} behavior "
            "with {market_patterns[identified_patterns][0][significance]} significance."
        )
    }
    
    # Save LLM context
    output_dir = Path(get_config().output_config['trading_model']['path']).parent
    llm_output_file = output_dir / 'llm_context.json'
    
    with open(llm_output_file, 'w') as f:
        json.dump(llm_context, f, indent=2)
    
    logger.info(f"LLM-optimized output saved to {llm_output_file}")