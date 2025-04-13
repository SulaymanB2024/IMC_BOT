"""Visualization module for generating interactive plots and dashboards."""
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
from typing import Dict, Any, List, Optional

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

def generate_visualizations(df: pd.DataFrame) -> None:
    """Generate all configured visualizations.
    
    Args:
        df: DataFrame containing features and analysis results
    """
    logger.info("Starting visualization generation")
    
    config = get_config()
    viz_config = config.visualization
    
    if not viz_config:
        logger.info("No visualization configuration found")
        return
        
    # Create output directory
    output_dir = Path(config.output_config['dashboard']['path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate decomposition plot
    if fig := plot_seasonal_decompose(df, viz_config):
        save_plot(fig, 'decomposition.html', output_dir)
    
    # Generate distribution plots
    for idx, fig in enumerate(plot_feature_distributions(df, viz_config)):
        save_plot(fig, f'distribution_{idx+1}.html', output_dir)
    
    # Generate anomaly plot
    if fig := plot_timeseries_with_anomalies(df, viz_config):
        save_plot(fig, 'anomalies.html', output_dir)
    
    # Generate correlation heatmap
    if fig := plot_correlation_heatmap(df, viz_config):
        save_plot(fig, 'correlation_heatmap.html', output_dir)
        
    logger.info("Visualization generation complete")