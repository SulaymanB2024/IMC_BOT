"""Output generation module for saving processed data and features."""
import logging
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict, Any

from .config import get_config
from .visualization import generate_visualizations

logger = logging.getLogger(__name__)

def save_features_parquet(df: pd.DataFrame) -> None:
    """Save the feature DataFrame to parquet format.
    
    Args:
        df: DataFrame containing features to save
    """
    config = get_config()
    output_path = Path(config.output_config['trading_model']['path'])
    compression = config.output_config['trading_model']['compression']
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving features to {output_path}")
    df.to_parquet(output_path, compression=compression)
    logger.info("Features saved successfully")

def create_price_chart(df: pd.DataFrame, price_col: str = 'price') -> Dict[str, Any]:
    """Create an interactive price chart with indicators.
    
    Args:
        df: DataFrame containing price and indicator data
        price_col: Name of the price column
        
    Returns:
        Plotly figure as JSON
    """
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        name='Price',
        line=dict(color='blue')
    ))
    
    # Add Bollinger Bands if they exist
    if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_mid']):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bb_upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['bb_lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ))
    
    fig.update_layout(
        title='Price with Bollinger Bands',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white'
    )
    
    return fig.to_json()

def create_indicator_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """Create an interactive chart of key indicators.
    
    Args:
        df: DataFrame containing indicator data
        
    Returns:
        Plotly figure as JSON
    """
    fig = go.Figure()
    
    # Add RSI if it exists
    if 'rsi' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['rsi'],
            name='RSI',
            line=dict(color='purple')
        ))
        
        # Add RSI threshold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
    
    # Add MACD if it exists
    if all(col in df.columns for col in ['macd', 'macd_signal']):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['macd'],
            name='MACD',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['macd_signal'],
            name='Signal',
            line=dict(color='orange')
        ))
    
    fig.update_layout(
        title='Technical Indicators',
        xaxis_title='Time',
        yaxis_title='Value',
        template='plotly_white'
    )
    
    return fig.to_json()

def save_dashboard_outputs(df: pd.DataFrame) -> None:
    """Save dashboard visualizations and summaries.
    
    Args:
        df: DataFrame containing features and indicators
    """
    config = get_config()
    dashboard_path = Path(config.output_config['dashboard']['path'])
    dashboard_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating dashboard outputs")
    
    # Generate all visualizations using the new module
    generate_visualizations(df)
    
    # Save data summary
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'start_date': df.index.min().isoformat(),
        'end_date': df.index.max().isoformat(),
        'feature_stats': df.describe().to_dict()
    }
    with open(dashboard_path / 'data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Dashboard outputs saved successfully")

def save_outputs(df: pd.DataFrame) -> None:
    """Main function to save all pipeline outputs.
    
    Args:
        df: DataFrame containing features and indicators
    """
    # Save features for trading model
    save_features_parquet(df)
    
    # Save dashboard visualizations
    save_dashboard_outputs(df)
    
    logger.info("All outputs generated and saved successfully")