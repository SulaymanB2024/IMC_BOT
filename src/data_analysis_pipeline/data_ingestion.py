"""Data ingestion module for loading and preprocessing trading data."""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import zipfile
import gzip
from typing import Tuple, List, Optional

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

def identify_file_type(file_path: str) -> str:
    """Identify the type of a data file based on extension and content.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        String indicating file type ('csv', 'csv.gz', 'parquet', or 'unknown')
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if path.name.endswith('.csv.gz'):
        return 'csv.gz'
    elif suffix == '.csv':
        return 'csv'
    elif suffix == '.parquet':
        return 'parquet'
    elif suffix == '.gz':
        # Check if it's a gzipped CSV
        try:
            with gzip.open(file_path, 'rt') as f:
                first_line = f.readline()
                if ',' in first_line or ';' in first_line or '\t' in first_line:
                    return 'csv.gz'
        except Exception:
            pass
    
    return 'unknown'

def try_read_file(file: str) -> Optional[pd.DataFrame]:
    """Attempt to read data file with appropriate method.
    
    Args:
        file: Path to data file
        
    Returns:
        DataFrame if successful, None otherwise
    """
    file_type = identify_file_type(file)
    
    if file_type == 'parquet':
        try:
            return pd.read_parquet(file)
        except Exception as e:
            logger.error(f"Failed to read parquet file {file}: {str(e)}")
            return None
            
    elif file_type in ['csv', 'csv.gz']:
        encodings = ['utf-8', 'latin1', 'cp1252']
        separators = [';', ',', '\t']
        
        # Function to try reading with specific settings
        def try_read(opener):
            for encoding in encodings:
                for sep in separators:
                    try:
                        with opener() as f:
                            df = pd.read_csv(f, sep=sep, encoding=encoding)
                            if len(df.columns) > 1:
                                logger.info(f"Successfully read {file} with separator '{sep}' and encoding '{encoding}'")
                                return df
                    except Exception as e:
                        logger.debug(f"Failed to read {file} with sep='{sep}' and encoding='{encoding}': {str(e)}")
                        continue
            return None
        
        # Try reading based on file type
        if file_type == 'csv':
            return try_read(lambda: open(file, 'r'))
        else:  # csv.gz
            return try_read(lambda: gzip.open(file, 'rt'))
    
    logger.error(f"Unsupported file type for {file}")
    return None

def identify_data_files(file_list: List[str]) -> Tuple[List[str], List[str]]:
    """Identify price and order data files from a list of files."""
    price_files = []
    order_files = []
    
    for file in file_list:
        filename = os.path.basename(file).lower()
        if 'price' in filename:
            price_files.append(file)
        elif any(x in filename for x in ['order', 'trade']):
            order_files.append(file)
    
    return sorted(price_files), sorted(order_files)

def try_read_csv(file: str) -> Optional[pd.DataFrame]:
    """Attempt to read CSV file with different settings.
    
    Args:
        file: Path to CSV file
        
    Returns:
        DataFrame if successful, None otherwise
    """
    encodings = ['utf-8', 'latin1', 'cp1252']
    separators = [';', ',', '\t']
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(file, sep=sep, encoding=encoding)
                
                # Verify we got more than just one column
                if len(df.columns) > 1:
                    logger.info(f"Successfully read {file} with separator '{sep}' and encoding '{encoding}'")
                    return df
                    
            except Exception as e:
                logger.debug(f"Failed to read {file} with sep='{sep}' and encoding='{encoding}': {str(e)}")
                continue
    
    logger.error(f"Could not read {file} with any combination of separators and encodings")
    return None

def load_price_data(file: str) -> Optional[pd.DataFrame]:
    """Load price data from a CSV file."""
    try:
        # Try reading with different settings
        df = try_read_file(file)
        if df is None:
            return None
            
        # Extract day from filename
        day = int(os.path.basename(file).split('_')[-1].split('.')[0])
        df['day'] = day
        
        # Handle timestamp conversion
        if 'timestamp' in df.columns:
            base_date = pd.Timestamp('2025-04-13') + pd.Timedelta(days=day)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            df['timestamp'] = base_date + pd.to_timedelta(df['timestamp'].dt.time.astype(str))
            df.set_index('timestamp', inplace=True)
            
            # Ensure timestamps are within expected range
            if df.index.min() < pd.Timestamp('2020-01-01') or df.index.max() > pd.Timestamp('2030-12-31'):
                logger.warning(f"Suspicious timestamps in {file}")
        
        # Verify data quality
        if 'mid_price' not in df.columns:
            logger.error(f"Required column 'mid_price' not found in {file}")
            return None
            
        if df['mid_price'].isna().all():
            logger.error(f"All price values are NaN in {file}")
            return None
            
        if df['mid_price'].min() < 0:
            logger.warning(f"Negative prices found in {file}")
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading price data from {file}: {str(e)}")
        return None

def load_order_data(file: str) -> Optional[pd.DataFrame]:
    """Load order/trade data from a CSV file."""
    try:
        # Try reading with different settings
        df = try_read_file(file)
        if df is None:
            return None
            
        # Extract day from filename
        day = int(os.path.basename(file).split('_')[-1].split('.')[0])
        df['day'] = day
        
        # Handle timestamp conversion
        if 'timestamp' in df.columns:
            base_date = pd.Timestamp('2025-04-13') + pd.Timedelta(days=day)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            df['timestamp'] = base_date + pd.to_timedelta(df['timestamp'].dt.time.astype(str))
            df.set_index('timestamp', inplace=True)
            
            # Ensure timestamps are within expected range
            if df.index.min() < pd.Timestamp('2020-01-01') or df.index.max() > pd.Timestamp('2030-12-31'):
                logger.warning(f"Suspicious timestamps in {file}")
        
        # Map quantity to volume
        if 'quantity' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['quantity'].abs()  # Use absolute value for volume
        
        # Verify data quality
        if 'volume' not in df.columns:
            logger.error(f"Required column 'volume' not found in {file}")
            return None
            
        if df['volume'].isna().all():
            logger.error(f"All volume values are NaN in {file}")
            return None
            
        if (df['volume'] < 0).any():
            logger.warning(f"Negative volumes found in {file}, taking absolute values")
            df['volume'] = df['volume'].abs()
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading order data from {file}: {str(e)}")
        return None

def merge_data(prices_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """Merge price and order data with proper time alignment."""
    logger.info("Merging price and order data")
    config = get_config()
    
    try:
        # Validate inputs
        if not isinstance(prices_df.index, pd.DatetimeIndex):
            raise ValueError("Price data must have datetime index")
        if not isinstance(orders_df.index, pd.DatetimeIndex):
            raise ValueError("Order data must have datetime index")
            
        # Log data ranges
        logger.info(f"Price data range: {prices_df.index.min()} to {prices_df.index.max()}")
        logger.info(f"Order data range: {orders_df.index.min()} to {orders_df.index.max()}")
        
        # Create standardized DataFrame with aligned timestamps
        freq = config.processing_config['time_alignment'].get('resample_freq', '1T')
        
        # Resample price data
        prices_resampled = prices_df.resample(freq).agg({
            'mid_price': 'last',
            'day': 'last'
        }).fillna(method='ffill')
        
        # Create standardized price column
        prices_resampled['price'] = prices_resampled['mid_price']
        
        # Resample order data, summing volumes
        orders_resampled = orders_df.resample(freq).agg({
            'volume': 'sum',
            'day': 'last'
        }).fillna(0)  # Fill missing volumes with 0
        
        # Merge on timestamp
        result = pd.merge(
            prices_resampled,
            orders_resampled,
            left_index=True,
            right_index=True,
            how='outer',
            suffixes=('_price', '_trade')
        )
        
        # Clean up day columns and handle missing values
        if 'day_price' in result.columns and 'day_trade' in result.columns:
            result['day'] = result['day_price'].combine_first(result['day_trade'])
            result.drop(['day_price', 'day_trade'], axis=1, inplace=True)
        
        # Handle missing values
        result['price'] = result['price'].fillna(method='ffill').fillna(method='bfill')
        result['volume'] = result['volume'].fillna(0)
        
        # Verify merged data quality
        if result.empty:
            raise ValueError("Merged DataFrame is empty")
            
        if result['price'].isna().any():
            raise ValueError("Missing values in price column after merge")
            
        # Log merge results
        logger.info(f"Merged data shape: {result.shape}")
        logger.info(f"Merged data range: {result.index.min()} to {result.index.max()}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error merging data: {str(e)}")
        raise

def load_trading_data() -> tuple:
    """Load trading data from source files.
    
    Returns:
        Tuple of (prices_df, orders_df)
    """
    config = get_config()
    data_dir = Path('data/raw/round-2-island-data-bottle')
    
    logger.info("Loading trading data")
    
    # Look for CSV files in the data directory
    extracted_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not extracted_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
    # Identify price and order files
    price_files, order_files = identify_data_files(extracted_files)
    
    if not price_files or not order_files:
        raise ValueError("Could not identify price and order data files")
    
    logger.info(f"Found {len(price_files)} price files and {len(order_files)} order files")
    
    # Load price data
    prices_dfs = []
    for file in price_files:
        df = load_price_data(file)
        if df is not None:
            prices_dfs.append(df)
    
    # Load order data    
    orders_dfs = []
    for file in order_files:
        df = load_order_data(file)
        if df is not None:
            orders_dfs.append(df)
            
    if not prices_dfs or not orders_dfs:
        raise ValueError("Failed to load price or order data")
        
    # Combine data
    prices_df = pd.concat(prices_dfs)
    orders_df = pd.concat(orders_dfs)
    
    # Sort by timestamp
    prices_df.sort_index(inplace=True)
    orders_df.sort_index(inplace=True)
    
    return prices_df, orders_df