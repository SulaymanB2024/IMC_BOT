"""Module for generating LLM-optimized market analysis output."""
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Utility functions from llm_output_generator.py
def safe_rate_calculation(count: int, total: int) -> float:
    """Calculate percentage with safety checks for zero division."""
    return (count / total * 100) if total > 0 else 0.0

def analyze_state_characteristics(df: pd.DataFrame, hmm_col: str,
                                target_cols: List[str],
                                metrics: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analyze characteristic behaviors of each HMM state."""
    state_stats = {}
    metric_funcs = {
        'mean': np.mean,
        'std': np.std,
        'median': np.median,
        'min': np.min,
        'max': np.max
    }
    
    for state in df[hmm_col].unique():
        state_data = df[df[hmm_col] == state]
        state_stats[state] = {}
        for col in target_cols:
            if col in df.columns:
                state_stats[state][col] = {
                    metric: float(metric_funcs[metric](state_data[col]))
                    for metric in metrics if metric in metric_funcs
                }
    return state_stats

def analyze_state_persistence(df: pd.DataFrame, hmm_col: str) -> Tuple[pd.Series, pd.Series]:
    """Calculate duration statistics for each HMM state."""
    state_changes = df[hmm_col] != df[hmm_col].shift()
    state_starts = df.index[state_changes].tolist()
    state_ends = state_starts[1:] + [df.index[-1]]
    
    durations = pd.Series(index=df[hmm_col].unique(), dtype=float)
    transition_counts = pd.Series(index=df[hmm_col].unique(), dtype=int)
    
    for state in durations.index:
        state_mask = df[hmm_col] == state
        state_periods = df[state_mask].index.to_series().diff()
        durations[state] = state_periods.mean() if not state_periods.empty else pd.Timedelta('0s')
        transition_counts[state] = sum(state_changes & (df[hmm_col] == state))
    
    return durations, transition_counts

# Core formatting functions from llm_output.py
def format_time_series_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Format time series metadata and characteristics."""
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
    """Format market regime information."""
    current_state = {
        "current_volatility": str(df['regime_volatility'].iloc[-1]),
        "current_trend": str(df['regime_trend'].iloc[-1]) if 'regime_trend' in df else "unknown"
    }
    
    regime_changes = {"volatility_transitions": [], "trend_transitions": []}
    
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
    
    return {"market_regimes": {"current_state": current_state, "regime_transitions": regime_changes}}

def format_signal_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Format trading signals."""
    signal_cols = [col for col in df.columns if col.startswith('signal_')]
    current_signals = {col: float(df[col].iloc[-1]) for col in signal_cols}
    
    composite = df['composite_signal'].iloc[-1] if 'composite_signal' in df.columns else 0
    signal_strength = abs(composite)
    signal_direction = "bullish" if composite > 0 else "bearish" if composite < 0 else "neutral"
    strength_desc = "strong" if signal_strength > 50 else "moderate" if signal_strength > 25 else "weak"
    
    return {
        "trading_signals": {
            "current_signals": current_signals,
            "composite_signal": {
                "value": float(composite),
                "interpretation": f"{strength_desc} {signal_direction}",
                "confidence": float(signal_strength) if signal_strength else 0.0
            }
        }
    }

def generate_llm_output(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save LLM-optimized output from market analysis."""
    logger.info("Generating LLM-optimized output")
    
    llm_context = {
        **format_time_series_context(df),
        **format_regime_context(df),
        **format_signal_context(df)
    }
    
    # Add natural language summary templates
    llm_context["nlg_templates"] = {
        "market_summary": (
            "The market is currently in a {current_state[current_trend]} trend with "
            "{current_state[current_volatility]} volatility. Trading signals indicate a "
            "{trading_signals[composite_signal][interpretation]} stance with "
            "{trading_signals[composite_signal][confidence]:.1f}% confidence."
        )
    }
    
    # Save LLM context
    llm_output_file = output_dir / 'llm_context.json'
    with open(llm_output_file, 'w') as f:
        json.dump(llm_context, f, indent=2)
    
    logger.info(f"LLM-optimized output saved to {llm_output_file}")