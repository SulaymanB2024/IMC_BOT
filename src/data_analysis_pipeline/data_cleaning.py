"""Data cleaning and validation module for the trading pipeline."""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def validate_price_data(df: pd.DataFrame) -> List[str]:
    """Validate price-related columns in the dataset.
    
    Args:
        df: DataFrame containing trading data
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    # Check for negative prices
    price_columns = df.filter(like='price').columns
    for col in price_columns:
        if (df[col] < 0).any():
            issues.append(f"Negative values found in {col}")
    
    # Check for extreme price movements (z-score > 5)
    for col in price_columns:
        zscore = np.abs((df[col] - df[col].mean()) / df[col].std())
        extreme_points = df[zscore > 5].index
        if len(extreme_points) > 0:
            issues.append(f"Extreme price movements found in {col} at: {extreme_points}")
    
    return issues

def validate_volume_data(df: pd.DataFrame) -> List[str]:
    """Validate volume-related columns in the dataset.
    
    Args:
        df: DataFrame containing trading data
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    # Check for negative volumes
    volume_columns = df.filter(like='volume').columns
    for col in volume_columns:
        if (df[col] < 0).any():
            issues.append(f"Negative values found in {col}")
    
    return issues

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the trading data by handling missing values and outliers.
    
    Args:
        df: DataFrame containing trading data
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning process")
    
    # Store original shape for logging
    original_shape = df.shape
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # Fill missing values with forward fill, then backward fill
        df[col] = df[col].ffill().bfill()
        
        # For any remaining NaN (if both ffill and bfill failed),
        # fill with column median
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Handle outliers using winsorization with more aggressive thresholds
    # for better outlier control
    price_columns = df.filter(like='price').columns
    for col in price_columns:
        # Calculate median and IQR for robust outlier detection
        median = df[col].median()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds using IQR method (1.5 * IQR)
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        # Clip values to bounds
        df[col] = df[col].clip(lower=lower, upper=upper)
        
        # Log significant outliers
        outliers = df[col][(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            logger.info(f"Clipped {len(outliers)} outliers in {col}")
    
    # Log cleaning results
    logger.info(f"Data shape before cleaning: {original_shape}")
    logger.info(f"Data shape after cleaning: {df.shape}")
    
    return df

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to validate and clean the trading data.
    
    Args:
        df: DataFrame containing trading data
        
    Returns:
        Cleaned and validated DataFrame
    """
    # Run validations
    price_issues = validate_price_data(df)
    volume_issues = validate_volume_data(df)
    
    # Log validation issues
    all_issues = price_issues + volume_issues
    if all_issues:
        for issue in all_issues:
            logger.warning(f"Validation issue found: {issue}")
    else:
        logger.info("No validation issues found")
    
    # Clean data
    cleaned_df = clean_data(df)
    
    return cleaned_df