"""Module for generating structured textual insights from market analysis."""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from pathlib import Path

from .config import get_config

logger = logging.getLogger(__name__)

def analyze_market_regime(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze market regime changes and characteristics.
    
    Args:
        df: DataFrame with regime columns
        
    Returns:
        Dictionary containing regime analysis
    """
    insights = {}
    
    # Analyze volatility regimes
    if 'regime_volatility' in df.columns:
        vol_regime_stats = df['regime_volatility'].value_counts()
        dominant_vol = vol_regime_stats.index[0] if not vol_regime_stats.empty else 'unknown'
        regime_changes = (df['regime_volatility'] != df['regime_volatility'].shift()).sum()
        
        insights['volatility'] = {
            'dominant_regime': dominant_vol,
            'regime_changes': int(regime_changes),
            'regime_distribution': vol_regime_stats.to_dict()
        }
    
    # Analyze trend regimes
    if 'regime_trend' in df.columns:
        trend_regime_stats = df['regime_trend'].value_counts()
        dominant_trend = trend_regime_stats.index[0] if not trend_regime_stats.empty else 'unknown'
        trend_changes = (df['regime_trend'] != df['regime_trend'].shift()).sum()
        
        insights['trend'] = {
            'dominant_regime': dominant_trend,
            'regime_changes': int(trend_changes),
            'regime_distribution': trend_regime_stats.to_dict()
        }
    
    return insights

def analyze_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze trading signals and their characteristics.
    
    Args:
        df: DataFrame with signal columns
        
    Returns:
        Dictionary containing signal analysis
    """
    insights = {'signals': {}}
    
    # Analyze individual signal columns
    signal_cols = [col for col in df.columns if col.startswith('signal_')]
    for col in signal_cols:
        signal_stats = df[col].value_counts()
        non_zero_signals = df[col].abs().sum()
        
        insights['signals'][col] = {
            'total_signals': int(non_zero_signals),
            'signal_distribution': signal_stats.to_dict()
        }
    
    # Analyze composite signal if available
    if 'composite_signal' in df.columns:
        composite = df['composite_signal'].dropna()
        insights['composite_signal'] = {
            'mean': float(composite.mean()),
            'std': float(composite.std()),
            'max': float(composite.max()),
            'min': float(composite.min()),
            'strong_buy_signals': int((composite > 50).sum()),
            'strong_sell_signals': int((composite < -50).sum())
        }
    
    return insights

def analyze_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze detected anomalies and their context.
    
    Args:
        df: DataFrame with anomaly detection columns
        
    Returns:
        Dictionary containing anomaly analysis
    """
    insights = {}
    
    # Analyze anomaly flags
    if 'is_anomaly' in df.columns:
        total_anomalies = df['is_anomaly'].sum()
        anomaly_periods = df[df['is_anomaly'] == 1].index
        
        insights['anomalies'] = {
            'total_count': int(total_anomalies),
            'percentage': float((total_anomalies / len(df)) * 100),
            'periods': [str(p) for p in anomaly_periods]
        }
    
    # Analyze z-scores if available
    if 'price_zscore' in df.columns:
        zscore = df['price_zscore'].dropna()
        insights['zscore_analysis'] = {
            'extreme_values_count': int((abs(zscore) > 3).sum()),
            'max_deviation': float(zscore.abs().max()),
            'mean_deviation': float(zscore.abs().mean())
        }
    
    return insights

def generate_market_summary(df: pd.DataFrame) -> str:
    """Generate a textual summary of market conditions.
    
    Args:
        df: DataFrame with all features and signals
        
    Returns:
        String containing market summary
    """
    summary_points = []
    
    # Overall trend
    if 'regime_trend' in df.columns:
        latest_trend = df['regime_trend'].iloc[-1]
        summary_points.append(f"Current market trend: {latest_trend}")
    
    # Volatility state
    if 'regime_volatility' in df.columns:
        latest_vol = df['regime_volatility'].iloc[-1]
        summary_points.append(f"Current volatility regime: {latest_vol}")
    
    # Signal consensus
    if 'composite_signal' in df.columns:
        latest_signal = df['composite_signal'].iloc[-1]
        signal_strength = abs(latest_signal)
        signal_direction = "bullish" if latest_signal > 0 else "bearish"
        if signal_strength > 50:
            strength_desc = "strong"
        elif signal_strength > 25:
            strength_desc = "moderate"
        else:
            strength_desc = "weak"
        
        summary_points.append(
            f"Current signal consensus: {strength_desc} {signal_direction} "
            f"(signal strength: {signal_strength:.1f})"
        )
    
    # Recent anomalies
    if 'is_anomaly' in df.columns:
        recent_window = df.iloc[-5:]  # Look at last 5 periods
        recent_anomalies = recent_window['is_anomaly'].sum()
        if recent_anomalies > 0:
            summary_points.append(
                f"Warning: {recent_anomalies} anomalous price movements "
                "detected in recent periods"
            )
    
    return "\n".join(summary_points)

def generate_insights(df: pd.DataFrame) -> None:
    """Generate and save all market insights.
    
    Args:
        df: DataFrame containing all features and analysis results
    """
    logger.info("Generating market insights")
    
    insights = {
        'market_regimes': analyze_market_regime(df),
        'trading_signals': analyze_signals(df),
        'anomaly_analysis': analyze_anomalies(df),
        'market_summary': generate_market_summary(df)
    }
    
    # Add metadata
    insights['metadata'] = {
        'analysis_period': {
            'start': str(df.index[0]),
            'end': str(df.index[-1])
        },
        'total_periods': len(df)
    }
    
    # Save insights
    output_dir = Path(get_config().output_config['trading_model']['path']).parent
    insights_file = output_dir / 'market_insights.json'
    
    with open(insights_file, 'w') as f:
        json.dump(insights, f, indent=2)
    
    logger.info(f"Market insights saved to {insights_file}")
    
    # Save human-readable summary separately
    summary_file = output_dir / 'market_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(insights['market_summary'])
    
    logger.info(f"Market summary saved to {summary_file}")