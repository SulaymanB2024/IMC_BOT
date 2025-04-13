"""Data ingestion module for loading and processing trading data."""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import glob
import os

from .config import get_config

logger = logging.getLogger(__name__)

def extract_data_files(zip_path: str, extract_dir: str) -> list:
    """Extract data files from zip archive.
    
    Args:
        zip_path: Path to zip file
        extract_dir: Directory to extract files to
        
    Returns:
        List of extracted file paths
    """
    logger.info(f"Extracting {zip_path} to {extract_dir}")
    
    # Create extract directory if it doesn't exist
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Find all CSV files in extracted directory, including subdirectories
    extracted_files = glob.glob(os.path.join(extract_dir, '**/*.csv'), recursive=True)
    logger.info(f"Extracted files: {extracted_files}")
    
    return extracted_files

def identify_data_files(file_list: list) -> tuple:
    """Identify price and order data files from list.
    
    Args:
        file_list: List of file paths
        
    Returns:
        Tuple of (price_files, order_files)
    """
    price_files = []
    order_files = []
    
    for file in file_list:
        basename = os.path.basename(file).lower()
        if 'price' in basename:
            price_files.append(file)
        elif 'trade' in basename or 'order' in basename:
            order_files.append(file)
    
    # Sort files to ensure consistent order
    price_files.sort()
    order_files.sort()
    
    return price_files, order_files

def load_and_process_data(file_list: list) -> tuple:
    """Load and process data from CSV files.
    
    Args:
        file_list: List of file paths
        
    Returns:
        Tuple of (prices_df, orders_df) with datetime indices
    """
    # Identify relevant files
    price_files, order_files = identify_data_files(file_list)
    
    if not price_files or not order_files:
        raise ValueError("Could not identify price and order data files")
    
    logger.info(f"Found {len(price_files)} price files and {len(order_files)} order files")
    
    # Load price data
    prices_dfs = []
    for file in price_files:
        df = pd.read_csv(file, sep=';')  # Use semicolon separator
        day = int(os.path.basename(file).split('_')[-1].split('.')[0])
        df['day'] = day
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            df.set_index('timestamp', inplace=True)
        
        prices_dfs.append(df)
    
    # Load order data
    orders_dfs = []
    for file in order_files:
        df = pd.read_csv(file, sep=';')  # Use semicolon separator
        day = int(os.path.basename(file).split('_')[-1].split('.')[0])
        df['day'] = day
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            df.set_index('timestamp', inplace=True)
        
        orders_dfs.append(df)
    
    # Combine data
    prices_df = pd.concat(prices_dfs)
    orders_df = pd.concat(orders_dfs)
    
    # Sort by timestamp
    prices_df.sort_index(inplace=True)
    orders_df.sort_index(inplace=True)
    
    return prices_df, orders_df

def merge_data(prices_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """Merge price and order data based on configuration settings.
    
    Args:
        prices_df: DataFrame containing price data with datetime index
        orders_df: DataFrame containing order data with datetime index
        
    Returns:
        Merged DataFrame
    """
    config = get_config()
    processing_config = config.processing_config
    
    logger.info("Merging price and order data")
    
    # Ensure we have datetime indices
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        raise ValueError("Price data must have a datetime index")
    if not isinstance(orders_df.index, pd.DatetimeIndex):
        raise ValueError("Order data must have a datetime index")
    
    # Get resample frequency from config
    freq = processing_config['time_alignment'].get('resample_freq', '1T')  # Default to 1 minute
    
    # Resample price data with proper aggregation methods
    prices_resampled = prices_df.resample(freq).agg({
        'price': 'last',  # Last price in the interval
        'day': 'last'     # Day should remain constant
    })
    
    # Resample order/trade data with proper aggregation methods
    # Assuming volume needs to be summed within each interval
    orders_resampled = orders_df.resample(freq).agg({
        'volume': 'sum',   # Sum volumes within interval
        'day': 'last'      # Day should remain constant
    })
    
    # Merge on index (timestamp) with outer join to keep all timepoints
    merged_df = pd.merge(
        prices_resampled,
        orders_resampled,
        left_index=True,
        right_index=True,
        how='outer',
        suffixes=('_price', '_trade')
    )
    
    # Forward fill missing values, then backward fill any remaining
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
    
    # Ensure the index is sorted
    merged_df.sort_index(inplace=True)
    
    # Clean up day columns if they were duplicated in the merge
    if 'day_price' in merged_df.columns and 'day_trade' in merged_df.columns:
        # Keep one day column
        merged_df['day'] = merged_df['day_price'].fillna(merged_df['day_trade'])
        merged_df.drop(['day_price', 'day_trade'], axis=1, inplace=True)
        
    logger.info(f"Merged data shape: {merged_df.shape}")
    return merged_df

def load_trading_data() -> tuple:
    """Load trading data from source files.
    
    Returns:
        Tuple of (prices_df, orders_df)
    """
    config = get_config()
    data_dir = Path('data/raw')
    
    # Try both zip files
    zip_files = [
        'round-2-island-data-bottle.zip',
        'round-2-island-data-bottle (2).zip'
    ]
    
    extracted_files = []
    for zip_file in zip_files:
        zip_path = Path(zip_file)
        if zip_path.exists():
            try:
                files = extract_data_files(str(zip_path), str(data_dir))
                if files:  # If we found CSV files, use them
                    extracted_files = files
                    break
            except Exception as e:
                logger.warning(f"Error extracting {zip_file}: {str(e)}")
    
    if not extracted_files:
        # If no files were extracted from zips, try looking for existing CSVs
        extracted_files = glob.glob(os.path.join(data_dir, '**/*.csv'), recursive=True)
    
    logger.info("Loading extracted data files")
    prices_df, orders_df = load_and_process_data(extracted_files)
    
    return prices_df, orders_df