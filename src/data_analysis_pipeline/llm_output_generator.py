"""Module for generating LLM-friendly output summaries."""
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from .config import get_config

logger = logging.getLogger(__name__)

def safe_rate_calculation(count: int, total: int) -> float:
    """Calculate percentage with safety checks for zero division.
    
    Args:
        count: Numerator count
        total: Denominator total
        
    Returns:
        Calculated percentage or 0.0 if total is zero
    """
    return (count / total * 100) if total > 0 else 0.0

def format_state_transition(current_state: int, previous_state: int, 
                          interpretations: Optional[Dict[str, Any]] = None) -> str:
    """Format state transition text with interpretations if available.
    
    Args:
        current_state: Current HMM state number
        previous_state: Previous HMM state number
        interpretations: Optional dictionary of state interpretations
        
    Returns:
        Formatted transition description string
    """
    if current_state == previous_state:
        return "State remained stable"
    
    def get_state_desc(state: int) -> str:
        if interpretations and f'state_{state}' in interpretations:
            chars = interpretations[f'state_{state}'].get('characteristics', [])
            if chars:
                return f"State {state} ({', '.join(chars)})"
        return f"State {state}"
    
    return f"Transition from {get_state_desc(previous_state)} to {get_state_desc(current_state)}"

def analyze_state_characteristics(df: pd.DataFrame, hmm_col: str,
                                target_cols: List[str],
                                metrics: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analyze characteristic behaviors of each HMM state.
    
    Args:
        df: DataFrame containing HMM states and features
        hmm_col: Name of HMM state column
        target_cols: List of columns to analyze
        metrics: List of metrics to calculate
        
    Returns:
        Nested dictionary of state characteristics
    """
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
                    for metric in metrics
                    if metric in metric_funcs
                }
    
    return state_stats

def analyze_state_persistence(df: pd.DataFrame, hmm_col: str) -> Tuple[pd.Series, pd.Series]:
    """Calculate duration statistics for each HMM state.
    
    Args:
        df: DataFrame containing HMM states
        hmm_col: Name of HMM state column
        
    Returns:
        Tuple of (average durations, transition counts)
    """
    # Calculate state durations
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

def summarize_data_quality(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """Generate summary of data quality metrics.
    
    Args:
        df: DataFrame to analyze
        config: Data quality configuration dictionary
        
    Returns:
        Markdown-formatted quality summary string
    """
    summary = ["## Data Quality Summary\n"]
    
    # Check completeness
    check_cols = config['check_columns']
    missing_rates = df[check_cols].isna().mean() * 100
    significant_missing = missing_rates[missing_rates > config['significant_missing_threshold']]
    
    if not significant_missing.empty:
        summary.append("### Missing Data Analysis")
        for col, rate in significant_missing.items():
            summary.append(f"- {col}: {rate:.1f}% missing values")
    
    # Check for time gaps
    time_gaps = df.index.to_series().diff()
    gap_threshold = pd.Timedelta(config['time_gap_threshold'])
    large_gaps = time_gaps[time_gaps > gap_threshold]
    
    if not large_gaps.empty:
        summary.append("\n### Time Gap Analysis")
        summary.append(f"Found {len(large_gaps)} gaps larger than {config['time_gap_threshold']}:")
        summary.append(f"- Largest gap: {large_gaps.max()}")
        summary.append(f"- Average gap size: {large_gaps.mean()}")
        
        if len(large_gaps) <= 3:  # Show details for small number of gaps
            summary.append("\nDetailed gaps:")
            for idx, gap in large_gaps.items():
                summary.append(f"- {idx}: {gap}")
    
    # Add data span information
    total_span = df.index[-1] - df.index[0]
    coverage = (len(df) * df.index.to_series().diff().median()) / total_span * 100
    
    summary.append("\n### Data Coverage")
    summary.append(f"- Time span: {total_span}")
    summary.append(f"- Effective coverage: {coverage:.1f}%")
    
    return "\n".join(summary)

def load_json_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load and parse a JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON content or None if loading fails
    """
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load JSON file {file_path}: {str(e)}")
        return None

def summarize_statistical_tests(results: Dict[str, Any]) -> str:
    """Generate summary of statistical test results.
    
    Args:
        results: Dictionary containing statistical test results
        
    Returns:
        Markdown-formatted summary string
    """
    summary = []
    summary.append("## Statistical Analysis\n")
    
    if not results:
        return '\n'.join(summary + ["No statistical test results available."])
    
    # Stationarity tests
    if 'stationarity_tests' in results:
        summary.append("### Stationarity Tests\n")
        for var, test in results['stationarity_tests'].items():
            if all(k in test for k in ['statistic', 'p_value']):
                is_stationary = test['p_value'] < 0.05
                summary.append(f"- {var}: {'Stationary' if is_stationary else 'Non-stationary'} "
                             f"(p-value: {test['p_value']:.4f})")
    
    # Granger causality
    if 'granger_causality_tests' in results and results['granger_causality_tests']:
        summary.append("\n### Granger Causality Tests\n")
        for pair, test in results['granger_causality_tests'].items():
            if all(k in test for k in ['independent_vars', 'dependent_vars', 'results_by_lag']):
                lag_results = test['results_by_lag']
                if lag_results:
                    min_p_value = min(result['p_value'] 
                                    for result in lag_results.values())
                    summary.append(f"- {test['independent_vars']} -> {test['dependent_vars']}: "
                                 f"{'Significant' if min_p_value < 0.05 else 'Not significant'} "
                                 f"(min p-value: {min_p_value:.4f})")
    
    return '\n'.join(summary)

def summarize_anomalies(df: pd.DataFrame, periods: List[int]) -> str:
    """Generate summary of anomaly detection results.
    
    Args:
        df: DataFrame containing anomaly flags
        periods: List of periods to analyze
        
    Returns:
        Markdown-formatted summary string
    """
    summary = []
    summary.append("## Anomaly Detection\n")
    
    if 'is_anomaly' not in df.columns:
        return '\n'.join(summary + ["No anomaly detection results available."])
    
    # Overall statistics
    total_anomalies = df['is_anomaly'].sum()
    total_periods = len(df)
    anomaly_rate = safe_rate_calculation(total_anomalies, total_periods)
    
    summary.append(f"Total anomalies detected: {total_anomalies} ({anomaly_rate:.1f}% of all observations)")
    
    # Recent periods analysis
    summary.append("\nRecent periods:")
    for period in sorted(periods):
        if period <= len(df):
            recent = df.iloc[-period:]
            anomaly_count = recent['is_anomaly'].sum()
            period_rate = safe_rate_calculation(anomaly_count, len(recent))
            summary.append(f"- Last {period} periods: {anomaly_count} anomalies ({period_rate:.1f}%)")
    
    return '\n'.join(summary)

def summarize_hmm_regimes(df: pd.DataFrame, periods: List[int], 
                         interpretations: Optional[Dict[str, Any]] = None) -> str:
    """Generate summary of HMM market regime states.
    
    Args:
        df: DataFrame containing HMM states
        periods: List of periods to analyze
        interpretations: Optional dictionary of HMM state interpretations
        
    Returns:
        Markdown-formatted summary string
    """
    summary = []
    summary.append("## HMM Market Regime Analysis\n")
    
    hmm_col = get_config().hmm_analysis['output_column_name']
    if hmm_col not in df.columns:
        return '\n'.join(summary + ["No HMM regime detection results available."])
    
    # Overall state distribution
    total_states = df[hmm_col].value_counts()
    total_periods = len(df)
    
    summary.append("### Overall State Distribution:")
    for state, count in total_states.items():
        state_pct = safe_rate_calculation(count, total_periods)
        state_desc = ""
        if interpretations and f'state_{state}' in interpretations:
            chars = interpretations[f'state_{state}'].get('characteristics', [])
            if chars:
                state_desc = f" ({', '.join(chars)})"
        summary.append(f"- State {state}{state_desc}: {count} periods ({state_pct:.1f}%)")
    
    # Recent periods analysis
    summary.append("\n### Recent State Analysis:")
    for period in sorted(periods):
        if period <= len(df):
            recent = df.iloc[-period:]
            state_counts = recent[hmm_col].value_counts()
            
            summary.append(f"\nLast {period} periods:")
            for state, count in state_counts.items():
                state_pct = safe_rate_calculation(count, len(recent))
                state_desc = ""
                if interpretations and f'state_{state}' in interpretations:
                    chars = interpretations[f'state_{state}'].get('characteristics', [])
                    if chars:
                        state_desc = f" ({', '.join(chars)})"
                summary.append(f"- State {state}{state_desc}: {count} periods ({state_pct:.1f}%)")
    
    # Most recent transition
    if len(df) >= 2:
        current_state = df[hmm_col].iloc[-1]
        previous_state = df[hmm_col].iloc[-2]
        transition_text = format_state_transition(current_state, previous_state, interpretations)
        summary.append(f"\nMost Recent: {transition_text}")
    
    return '\n'.join(summary)

def summarize_traditional_regimes(df: pd.DataFrame) -> str:
    """Generate summary of traditional volatility and trend regimes.
    
    Args:
        df: DataFrame containing regime indicators
        
    Returns:
        Markdown-formatted summary string
    """
    summary = []
    summary.append("## Traditional Market Regimes\n")
    
    has_regimes = False
    
    # Volatility regime
    if 'regime_volatility' in df.columns:
        has_regimes = True
        current_vol = df['regime_volatility'].iloc[-1]
        vol_dist = df['regime_volatility'].value_counts()
        total_periods = len(df)
        
        summary.append("### Volatility Regime")
        summary.append(f"Current: {current_vol}")
        summary.append("\nDistribution:")
        for regime, count in vol_dist.items():
            regime_pct = safe_rate_calculation(count, total_periods)
            summary.append(f"- {regime}: {count} periods ({regime_pct:.1f}%)")
    
    # Trend regime
    if 'regime_trend' in df.columns:
        has_regimes = True
        current_trend = df['regime_trend'].iloc[-1]
        trend_dist = df['regime_trend'].value_counts()
        
        summary.append("\n### Trend Regime")
        summary.append(f"Current: {current_trend}")
        summary.append("\nDistribution:")
        for regime, count in trend_dist.items():
            regime_pct = safe_rate_calculation(count, total_periods)
            summary.append(f"- {regime}: {count} periods ({regime_pct:.1f}%)")
    
    if not has_regimes:
        summary.append("No traditional regime detection results available.")
    
    return '\n'.join(summary)

def summarize_recent_signals(df: pd.DataFrame, periods: List[int]) -> str:
    """Generate summary of recent trading signals.
    
    Args:
        df: DataFrame containing trading signals
        periods: List of periods to analyze
        
    Returns:
        Markdown-formatted summary string
    """
    summary = []
    summary.append("## Trading Signals\n")
    
    if 'composite_signal' not in df.columns:
        return '\n'.join(summary + ["No trading signal results available."])
    
    # Current signal
    current_signal = df['composite_signal'].iloc[-1]
    signal_strength = abs(current_signal)
    signal_direction = "bullish" if current_signal > 0 else "bearish" if current_signal < 0 else "neutral"
    
    summary.append(f"Current Signal: {signal_direction.title()} (strength: {signal_strength:.1f})")
    
    # Recent signal analysis
    summary.append("\nRecent Signal Analysis:")
    for period in sorted(periods):
        if period <= len(df):
            recent = df.iloc[-period:]
            avg_signal = recent['composite_signal'].mean()
            max_signal = recent['composite_signal'].abs().max()
            signal_direction = "bullish" if avg_signal > 0 else "bearish" if avg_signal < 0 else "neutral"
            summary.append(
                f"- Last {period} periods: {signal_direction.title()} "
                f"(avg strength: {abs(avg_signal):.1f}, max: {max_signal:.1f})"
            )
    
    return '\n'.join(summary)

def summarize_state_characteristics(state_chars: Dict[str, Dict[str, Dict[str, float]]], 
                                  interpretations: Optional[Dict[str, Any]] = None) -> str:
    """Generate summary of HMM state characteristics analysis.
    
    Args:
        state_chars: Dictionary of state characteristics
        interpretations: Optional dictionary of state interpretations
        
    Returns:
        Markdown-formatted summary string
    """
    summary = []
    summary.append("## HMM State Characteristics\n")
    
    for state, features in state_chars.items():
        state_desc = ""
        if interpretations and f'state_{state}' in interpretations:
            chars = interpretations[f'state_{state}'].get('characteristics', [])
            if chars:
                state_desc = f" ({', '.join(chars)})"
        
        summary.append(f"### State {state}{state_desc}\n")
        for feature, metrics in features.items():
            summary.append(f"#### {feature}")
            for metric, value in metrics.items():
                summary.append(f"- {metric}: {value:.4f}")
        summary.append("")  # Add blank line between states
    
    return '\n'.join(summary)

def summarize_state_persistence(durations: pd.Series, 
                              transitions: pd.Series,
                              interpretations: Optional[Dict[str, Any]] = None) -> str:
    """Generate summary of HMM state persistence analysis.
    
    Args:
        durations: Series of average state durations
        transitions: Series of state transition counts
        interpretations: Optional dictionary of state interpretations
        
    Returns:
        Markdown-formatted summary string
    """
    summary = []
    summary.append("## HMM State Persistence\n")
    
    for state in durations.index:
        state_desc = ""
        if interpretations and f'state_{state}' in interpretations:
            chars = interpretations[f'state_{state}'].get('characteristics', [])
            if chars:
                state_desc = f" ({', '.join(chars)})"
        
        duration = durations[state]
        n_transitions = transitions[state]
        
        summary.append(f"### State {state}{state_desc}")
        summary.append(f"- Average duration: {duration}")
        summary.append(f"- Number of transitions: {n_transitions}")
        summary.append("")  # Add blank line between states
    
    return '\n'.join(summary)

def generate_llm_summary(df: pd.DataFrame) -> None:
    """Generate comprehensive summary for LLM consumption.
    
    Args:
        df: DataFrame containing all features and analysis results
    """
    logger.info("Generating LLM-optimized summary")
    
    config = get_config().llm_output_generation
    if not config.get('enabled', False):
        logger.info("LLM output generation disabled in config")
        return
    
    try:
        # Initialize summary sections
        sections = []
        sections.append("# Market Analysis Summary\n")
        
        # Add timestamp and data period
        sections.append(f"Analysis timestamp: {pd.Timestamp.now()}\n")
        sections.append(f"Data period: {df.index[0]} to {df.index[-1]}\n")
        
        # Load auxiliary data
        hmm_interpretations = None
        if config.get('hmm_interpretations_file'):
            hmm_interpretations = load_json_file(config['hmm_interpretations_file'])
        
        statistical_results = None
        if config.get('statistical_results_file'):
            statistical_results = load_json_file(config['statistical_results_file'])
        
        # Data quality summary
        if config.get('include_data_quality_summary', True):
            sections.append(summarize_data_quality(df, config['data_quality']))
        
        # Statistical analysis
        if config.get('include_statistical_summary', True) and statistical_results:
            sections.append(summarize_statistical_tests(statistical_results))
        
        # HMM analysis sections
        hmm_col = get_config().hmm_analysis['output_column_name']
        if hmm_col in df.columns:
            if config.get('include_hmm_summary', True):
                sections.append(summarize_hmm_regimes(
                    df, 
                    config['summary_periods'],
                    hmm_interpretations
                ))
            
            if config.get('include_state_characteristics', True):
                state_chars = analyze_state_characteristics(
                    df,
                    hmm_col,
                    config['state_characteristics']['target_columns'],
                    config['state_characteristics']['metrics']
                )
                sections.append(summarize_state_characteristics(
                    state_chars,
                    hmm_interpretations
                ))
            
            if config.get('include_state_persistence', True):
                durations, transitions = analyze_state_persistence(df, hmm_col)
                sections.append(summarize_state_persistence(
                    durations,
                    transitions,
                    hmm_interpretations
                ))
        
        # Traditional regime analysis
        if config.get('include_regime_summary', True):
            sections.append(summarize_traditional_regimes(df))
        
        # Anomaly detection
        if config.get('include_anomaly_summary', True):
            sections.append(summarize_anomalies(
                df,
                config['summary_periods']
            ))
        
        # Trading signals
        if config.get('include_recent_signal_summary', True):
            sections.append(summarize_recent_signals(
                df,
                config['summary_periods']
            ))
        
        # Save summary
        output_path = Path(config['output_file'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n\n'.join(sections))
        
        logger.info(f"LLM summary saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating LLM summary: {str(e)}", exc_info=True)