"""Unit tests for data cleaning functionality."""
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from src.data_analysis_pipeline.data_cleaning import (
    clean_data,
    validate_price_data,
    validate_volume_data
)

@pytest.fixture
def sample_data_with_missing_values():
    """Create sample DataFrame with missing values."""
    return pd.DataFrame({
        'price': [100, np.nan, 102, np.nan, 104],
        'volume': [1000, 2000, np.nan, 4000, 5000],
        'other_col': [1, 2, 3, np.nan, 5]
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))

@pytest.fixture
def sample_data_with_outliers():
    """Create sample DataFrame with outlier values."""
    return pd.DataFrame({
        'price': [100, 101, 1000, 102, 103],  # 1000 is an outlier
        'volume': [-100, 2000, 3000, 4000, 5000],  # -100 is invalid
    }, index=pd.date_range('2024-01-01', periods=5, freq='D'))

def test_clean_data_handles_missing_values(sample_data_with_missing_values):
    """Test that clean_data properly handles missing values."""
    result = clean_data(sample_data_with_missing_values)
    
    # Check no NaN values remain
    assert not result.isnull().any().any(), "Cleaned data should not contain NaN values"
    
    # Check forward fill worked as expected for price
    assert result['price'].iloc[1] == 100, "Missing price should be filled with previous value"
    
    # Check that all columns were handled
    for col in result.columns:
        assert not result[col].isnull().any(), f"Column {col} still contains NaN values"

def test_clean_data_handles_outliers(sample_data_with_outliers):
    """Test that clean_data properly handles outlier values."""
    result = clean_data(sample_data_with_outliers)
    
    # The extreme price (1000) should be winsorized
    assert result['price'].max() < 1000, "Outlier price should be capped"
    assert result['price'].min() > 90, "Lower bound for price should be reasonable"
    
    # Check that basic statistics are reasonable
    price_std = result['price'].std()
    assert price_std < 100, "After cleaning, price standard deviation should be reasonable"

def test_validate_price_data_negative_values():
    """Test that validate_price_data catches negative prices."""
    df = pd.DataFrame({
        'price': [100, -50, 102],
        'other_price': [200, 201, -10]
    })
    
    issues = validate_price_data(df)
    assert len(issues) == 2, "Should detect negative values in both price columns"
    assert any("Negative values found" in issue for issue in issues)

def test_validate_volume_data_negative_values():
    """Test that validate_volume_data catches negative volumes."""
    df = pd.DataFrame({
        'volume': [1000, -500, 2000],
        'other_volume': [100, 200, -50]
    })
    
    issues = validate_volume_data(df)
    assert len(issues) == 2, "Should detect negative values in both volume columns"
    assert any("Negative values found" in issue for issue in issues)

def test_clean_data_preserves_dtypes():
    """Test that clean_data preserves or corrects data types."""
    input_df = pd.DataFrame({
        'price': [100, 101, np.nan, 103],
        'volume': [1000, 2000, 3000, 4000],
    }, index=pd.date_range('2024-01-01', periods=4, freq='D'))
    
    result = clean_data(input_df)
    
    # Check datatypes are preserved
    assert result['price'].dtype == np.float64, "Price should remain float64"
    assert result['volume'].dtype == np.int64, "Volume should remain int64"