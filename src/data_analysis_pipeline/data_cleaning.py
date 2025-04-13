"""Data cleaning and validation module."""
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the input data."""
    df = df.copy()
    
    # Remove duplicates
    n_duplicates = df.index.duplicated().sum()
    if n_duplicates:
        logger.warning(f"Found {n_duplicates} duplicate timestamps - removing")
        df = df[~df.index.duplicated(keep='first')]
    
    # Handle missing values
    missing_by_col = df.isnull().sum()
    if missing_by_col.any():
        logger.warning("Missing values found:")
        for col, count in missing_by_col[missing_by_col > 0].items():
            logger.warning(f"  {col}: {count} missing values")
        
        # Forward fill price-based columns
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        df[price_cols] = df[price_cols].ffill()  # Using ffill() instead of fillna(method='ffill')
        
        # Forward fill volume with 0
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        df[volume_cols] = df[volume_cols].fillna(0)
        
        # Drop any remaining rows with missing values
        df = df.dropna()
    
    # Remove outliers (prices/volumes that are extreme multiples of median)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().all():  # Skip columns that are all NaN
            continue
            
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        
        if mad == 0:  # Skip if MAD is 0 (constant values)
            continue
            
        threshold = 10  # Number of MADs for outlier detection
        outliers = np.abs(df[col] - median) > threshold * mad
        n_outliers = outliers.sum()
        
        if n_outliers > 0:  # Check if there are any outliers
            logger.warning(f"Found {n_outliers} outliers in {col}")
            df.loc[outliers, col] = np.nan
            df[col] = df[col].ffill()  # Using ffill() instead of fillna(method='ffill')
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Add basic derived columns if price exists
    if 'price' in df.columns:
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        if 'volume' in df.columns:
            df['volume_notional'] = df['price'] * df['volume']
    
    logger.info(f"Data cleaning complete. Shape: {df.shape}")
    return df

def merge_data(prices_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """Merge price and order data."""
    # Remove duplicate timestamps first
    prices_df = prices_df[~prices_df.index.duplicated(keep='first')]
    orders_df = orders_df[~orders_df.index.duplicated(keep='first')]
    
    # Ensure both DataFrames have the same index frequency
    freq = pd.Timedelta(seconds=1)  # 1-second frequency
    
    # Generate a complete range of timestamps
    full_range = pd.date_range(
        start=min(prices_df.index.min(), orders_df.index.min()),
        end=max(prices_df.index.max(), orders_df.index.max()),
        freq=freq
    )
    
    # Reindex both DataFrames to the full range and forward fill
    prices_resampled = prices_df.reindex(full_range).ffill()  # Using ffill() instead of fillna(method='ffill')
    orders_resampled = orders_df.reindex(full_range).ffill()  # Using ffill() instead of fillna(method='ffill')
    
    # Merge the DataFrames
    merged = pd.concat([prices_resampled, orders_resampled], axis=1)
    
    # Drop any remaining rows with all missing values
    merged = merged.dropna(how='all')
    
    logger.info(f"Data merge complete. Shape: {merged.shape}")
    return merged