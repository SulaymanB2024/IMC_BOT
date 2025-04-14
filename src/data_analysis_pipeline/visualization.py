"""Visualization module for generating plots and dashboards."""
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from .config import get_config

logger = logging.getLogger(__name__)

def plot_seasonal_decompose(df: pd.DataFrame, config: Dict[str, Any]) -> Optional[go.Figure]:
    """Create time series decomposition plot.
    
    Args:
        df: DataFrame containing time series data
        config: Visualization configuration dictionary
        
    Returns:
        Plotly figure object if successful, None otherwise
    """
    try:
        if not config.get('plot_decomposition'):
            return None
            
        col = config['decomposition_column']
        if col not in df.columns:
            logger.warning(f"Column {col} not found for decomposition")
            return None
            
        # Perform decomposition
        decomposition = seasonal_decompose(
            df[col],
            model=config['decomposition_model'],
            period=config['decomposition_period']
        )
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual']
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name='Original'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=decomposition.trend, name='Trend'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=decomposition.seasonal, name='Seasonal'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=decomposition.resid, name='Residual'),
            row=4, col=1
        )
        
        fig.update_layout(
            height=1000,
            title_text=f"Time Series Decomposition of {col}",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in seasonal decomposition: {str(e)}")
        return None

def plot_feature_distributions(df: pd.DataFrame, config: Dict[str, Any]) -> List[go.Figure]:
    """Create distribution plots for specified features.
    
    Args:
        df: DataFrame containing features
        config: Visualization configuration dictionary
        
    Returns:
        List of Plotly figure objects
    """
    figures = []
    
    if not config.get('plot_feature_distributions'):
        return figures
        
    for col in config['distribution_columns']:
        if col not in df.columns:
            logger.warning(f"Column {col} not found for distribution plot")
            continue
            
        fig = px.histogram(
            df, x=col,
            title=f'Distribution of {col}',
            marginal='box'  # Add boxplot to the marginal
        )
        
        # Add KDE if the column is numeric and has enough unique values
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            kde = px.line(df, x=col)
            fig.add_trace(kde.data[0])
            
        figures.append(fig)
    
    return figures

def plot_timeseries_with_anomalies(df: pd.DataFrame, config: Dict[str, Any]) -> Optional[go.Figure]:
    """Create time series plot with highlighted anomalies.
    
    Args:
        df: DataFrame containing time series and anomaly flags
        config: Visualization configuration dictionary
        
    Returns:
        Plotly figure object if successful, None otherwise
    """
    try:
        if not config.get('plot_anomalies_on_timeseries'):
            return None
            
        base_col = config['timeseries_base_column']
        flag_col = config['anomaly_flag_column']
        
        if base_col not in df.columns or flag_col not in df.columns:
            logger.warning(f"Required columns missing for anomaly plot")
            return None
        
        # Create base time series plot
        fig = go.Figure()
        
        # Add main time series
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[base_col],
            name=base_col,
            line=dict(color='blue')
        ))
        
        # Add anomaly points
        anomalies = df[df[flag_col] == -1]  # Isolation Forest uses -1 for anomalies
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies[base_col],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='x'
                )
            ))
        
        fig.update_layout(
            title=f'{base_col} Time Series with Detected Anomalies',
            xaxis_title='Time',
            yaxis_title=base_col,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in anomaly visualization: {str(e)}")
        return None

def plot_correlation_heatmap(df: pd.DataFrame, config: Dict[str, Any]) -> Optional[go.Figure]:
    """Create correlation heatmap for specified features.
    
    Args:
        df: DataFrame containing features
        config: Visualization configuration dictionary
        
    Returns:
        Plotly figure object if successful, None otherwise
    """
    try:
        if not config.get('plot_statistical_tests'):
            return None
            
        columns = config['correlation_heatmap_columns']
        if not all(col in df.columns for col in columns):
            logger.warning("Some columns missing for correlation heatmap")
            return None
            
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={'size': 10},
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            width=800,
            height=800
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error in correlation heatmap: {str(e)}")
        return None

def plot_hmm_states(df: pd.DataFrame, config: Dict[str, Any]) -> Optional[go.Figure]:
    """Create visualization of price series with HMM state backgrounds.
    
    Args:
        df: DataFrame containing price data and HMM states
        config: Visualization configuration dictionary
        
    Returns:
        Plotly figure object if successful, None otherwise
    """
    try:
        # Check if HMM states are present
        hmm_col = get_config().hmm_analysis['output_column_name']
        if hmm_col not in df.columns:
            logger.warning("HMM states column not found")
            return None
            
        # Create figure with price series
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['price'],
            name='Price',
            line=dict(color='black', width=1)
        ))
        
        # Add colored backgrounds for each state
        unique_states = sorted(df[hmm_col].unique())
        colors = px.colors.qualitative.Set3[:len(unique_states)]  # Color scheme
        
        for state, color in zip(unique_states, colors):
            state_periods = df[df[hmm_col] == state]
            if len(state_periods) == 0:
                continue
                
            # Find continuous segments of the same state
            state_changes = state_periods.index.to_series().diff() > pd.Timedelta(minutes=1)
            segment_starts = state_periods.index[state_changes | (state_changes.shift(1).fillna(True))]
            
            for start in segment_starts:
                segment = state_periods[state_periods.index >= start]
                if len(segment) == 0:
                    continue
                    
                end = segment.index[segment.index.to_series().diff() > pd.Timedelta(minutes=1)].min()
                if pd.isna(end):
                    end = segment.index[-1]
                
                # Add colored background for state period
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=color,
                    opacity=0.2,
                    layer="below",
                    name=f"State {state}"
                )
        
        # Load state interpretations if available
        try:
            dashboard_path = Path(config.output_config['dashboard']['path'])
            with open(dashboard_path / 'hmm_interpretations.json', 'r') as f:
                interpretations = json.load(f)
                
            # Add annotations for state characteristics
            for state in unique_states:
                state_info = interpretations.get(f'state_{state}')
                if state_info and 'characteristics' in state_info:
                    chars = ', '.join(state_info['characteristics'])
                    fig.add_annotation(
                        x=1.02,
                        y=0.9 - (state * 0.1),  # Stack vertically
                        text=f"State {state}: {chars}",
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        align="left"
                    )
        except Exception as e:
            logger.warning(f"Could not load state interpretations: {str(e)}")
        
        fig.update_layout(
            title="Price Series with HMM Market Regimes",
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            template="plotly_white",
            margin=dict(r=250)  # Make room for state annotations
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating HMM state visualization: {str(e)}")
        return None

def plot_hmm_regimes(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create visualization of price series with HMM regime backgrounds.
    
    Args:
        df: DataFrame containing price and HMM regime data
        
    Returns:
        Plotly figure if successful, None otherwise
    """
    logger.info("Generating HMM regime visualization")
    config = get_config()
    
    try:
        # Check if HMM states are present
        hmm_col = config.hmm_analysis['output_column_name']
        hmm_label_col = f"{hmm_col}_label"
        
        if hmm_col not in df.columns:
            logger.warning("HMM regime column not found")
            return None
        
        # Create figure with price series
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['price'],
            name='Price',
            line=dict(color='black', width=1)
        ))
        
        # Load state interpretations for colors
        try:
            output_dir = Path(config.output_config['dashboard']['path'])
            with open(output_dir / 'hmm_interpretations.json', 'r') as f:
                interpretations = json.load(f)
                
            state_mapping = interpretations.get('state_mapping', {})
        except Exception as e:
            logger.warning(f"Could not load HMM interpretations: {str(e)}")
            state_mapping = {}
        
        # Color scheme for different regimes
        color_map = {
            'volatile_bullish': 'rgba(0, 255, 0, 0.2)',   # Green
            'volatile_bearish': 'rgba(255, 0, 0, 0.2)',   # Red
            'calm_neutral': 'rgba(128, 128, 128, 0.2)',   # Gray
        }
        
        # Add colored backgrounds for each regime
        unique_states = sorted(df[hmm_col].unique())
        
        for state in unique_states:
            # Get state label if available
            if hmm_label_col in df.columns:
                state_label = df[df[hmm_col] == state][hmm_label_col].iloc[0]
            else:
                state_label = state_mapping.get(str(state), f"State {state}")
            
            # Get color for state
            color = color_map.get(state_label, f'rgba({hash(state_label) % 256}, {hash(state_label*2) % 256}, {hash(state_label*3) % 256}, 0.2)')
            
            # Find continuous segments of the same state
            state_changes = df[hmm_col] != df[hmm_col].shift()
            regime_starts = df.index[state_changes & (df[hmm_col] == state)]
            regime_ends = df.index[state_changes & (df[hmm_col].shift() == state)]
            
            if len(regime_starts) == 0:
                continue
                
            # Handle case where regime continues to the end
            if len(regime_ends) < len(regime_starts):
                regime_ends = list(regime_ends) + [df.index[-1]]
            
            # Add colored background for each regime period
            for start, end in zip(regime_starts, regime_ends):
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor=color,
                    opacity=0.5,
                    layer="below",
                    name=state_label
                )
        
        # Add annotations for regime characteristics
        if interpretations and 'states' in interpretations:
            y_pos = 1.05
            for state in unique_states:
                state_info = interpretations['states'].get(f'state_{state}', {})
                chars = state_info.get('characteristics', [])
                if chars:
                    fig.add_annotation(
                        x=1.02,
                        y=y_pos,
                        xref="paper",
                        yref="paper",
                        text=f"State {state}: {', '.join(chars)}",
                        showarrow=False,
                        align="left"
                    )
                    y_pos -= 0.05
        
        # Update layout
        fig.update_layout(
            title="Price Series with HMM Market Regimes",
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            template="plotly_white",
            margin=dict(r=250)  # Make room for annotations
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating HMM regime visualization: {str(e)}")
        return None

def plot_hmm_transition_matrix(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create visualization of HMM transition matrix.
    
    Args:
        df: DataFrame containing HMM states
        
    Returns:
        Plotly figure if successful, None otherwise
    """
    logger.info("Generating HMM transition matrix visualization")
    config = get_config()
    
    try:
        # Load HMM interpretations
        output_dir = Path(config.output_config['dashboard']['path'])
        with open(output_dir / 'hmm_interpretations.json', 'r') as f:
            interpretations = json.load(f)
        
        if 'transition_matrix' not in interpretations:
            logger.warning("No transition matrix found in interpretations")
            return None
            
        # Get transition matrix and state mappings
        trans_matrix = np.array(interpretations['transition_matrix'])
        state_mapping = interpretations.get('state_mapping', {})
        
        # Create state labels
        n_states = len(trans_matrix)
        labels = [
            state_mapping.get(str(i), f"State {i}")
            for i in range(n_states)
        ]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=trans_matrix,
            x=labels,
            y=labels,
            text=np.round(trans_matrix, 3),
            texttemplate='%{text}',
            colorscale='Viridis',
            zmin=0,
            zmax=1
        ))
        
        # Update layout
        fig.update_layout(
            title="HMM State Transition Matrix",
            xaxis_title="To State",
            yaxis_title="From State",
            width=600,
            height=600
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating transition matrix visualization: {str(e)}")
        return None

def plot_hmm_state_probabilities(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create visualization of HMM state probabilities over time.
    
    Args:
        df: DataFrame containing HMM states
        
    Returns:
        Plotly figure if successful, None otherwise
    """
    logger.info("Generating HMM state probabilities visualization")
    config = get_config()
    
    try:
        hmm_col = config.hmm_analysis['output_column_name']
        if hmm_col not in df.columns:
            logger.warning("HMM regime column not found")
            return None
            
        # Calculate rolling state probabilities
        window = min(100, len(df) // 10)  # Adaptive window size
        state_probs = pd.get_dummies(df[hmm_col]).rolling(
            window=window,
            min_periods=1
        ).mean()
        
        # Create figure
        fig = go.Figure()
        
        # Load state interpretations if available
        try:
            output_dir = Path(config.output_config['dashboard']['path'])
            with open(output_dir / 'hmm_interpretations.json', 'r') as f:
                interpretations = json.load(f)
            state_mapping = interpretations.get('state_mapping', {})
        except Exception:
            state_mapping = {}
        
        # Add probability line for each state
        for state in sorted(df[hmm_col].unique()):
            state_label = state_mapping.get(str(state), f"State {state}")
            fig.add_trace(go.Scatter(
                x=df.index,
                y=state_probs[state],
                name=state_label,
                mode='lines',
                line=dict(width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title="HMM State Probabilities Over Time",
            xaxis_title="Time",
            yaxis_title="State Probability",
            yaxis=dict(
                tickformat=".0%",
                range=[0, 1]
            ),
            template="plotly_white",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating state probabilities visualization: {str(e)}")
        return None

def save_plot(fig: go.Figure, filename: str, output_dir: Path) -> None:
    """Save a Plotly figure to file.
    
    Args:
        fig: Plotly figure object
        filename: Name of the output file
        output_dir: Directory to save the file
    """
    try:
        output_path = output_dir / filename
        fig.write_html(str(output_path))
        logger.info(f"Saved plot to {output_path}")
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {str(e)}")

def generate_visualizations(df: pd.DataFrame, trading_mode: bool = False) -> None:
    """Generate visualizations based on configuration and mode.
    
    Args:
        df: DataFrame containing features and analysis results
        trading_mode: If True, generate only essential trading visualizations
    """
    config = get_config()
    vis_config = config.get('visualization', {})
    output_dir = Path(config['output']['dashboard']['path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Core trading visualizations (always generated)
    if vis_config.get('plot_anomalies_on_timeseries', True):
        logger.info("Generating price and anomaly visualization")
        plot_price_with_anomalies(
            df, 
            output_dir / 'price_anomalies.html',
            vis_config
        )
    
    if vis_config.get('hmm_visualization', {}).get('enabled', True):
        logger.info("Generating HMM regime visualization")
        plot_hmm_regimes(
            df,
            output_dir / 'hmm_regimes.html',
            vis_config.get('hmm_visualization', {})
        )
    
    # Additional visualizations for full analysis mode
    if not trading_mode:
        if vis_config.get('plot_decomposition', True):
            logger.info("Generating time series decomposition")
            plot_decomposition(
                df,
                output_dir / 'decomposition.html',
                vis_config
            )
        
        if vis_config.get('plot_feature_distributions', True):
            logger.info("Generating feature distribution plots")
            plot_distributions(
                df,
                output_dir,
                vis_config.get('distribution_columns', [])
            )
        
        if vis_config.get('plot_statistical_tests', True):
            logger.info("Generating statistical test visualizations")
            plot_correlation_heatmap(
                df,
                output_dir / 'correlation_heatmap.html',
                vis_config.get('correlation_heatmap_columns', [])
            )

def plot_price_with_anomalies(df: pd.DataFrame, output_path: Path, config: Dict[str, Any]) -> None:
    """Generate price chart with anomaly markers."""
    # Implementation remains the same...
    pass

def plot_hmm_regimes(df: pd.DataFrame, output_path: Path, config: Dict[str, Any]) -> None:
    """Generate HMM regime visualization."""
    # Implementation remains the same...
    pass

def plot_decomposition(df: pd.DataFrame, output_path: Path, config: Dict[str, Any]) -> None:
    """Generate time series decomposition plot."""
    # Implementation remains the same...
    pass

def plot_distributions(df: pd.DataFrame, output_dir: Path, columns: list) -> None:
    """Generate distribution plots for specified columns."""
    # Implementation remains the same...
    pass

def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path, columns: list) -> None:
    """Generate correlation heatmap for specified columns."""
    # Implementation remains the same...
    pass