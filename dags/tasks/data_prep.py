#!/usr/bin/env python3
"""
data_prep.py - Module for data preparation and feature engineering
-----------------------------------------------------------------
This module provides functionality for preparing datasets for model training.
It includes data loading, cleaning, and feature engineering functions.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Import utility modules
from utils.storage import download as s3_download
from utils.cache import GLOBAL_CACHE, cache_result

logger = logging.getLogger(__name__)

@cache_result
def prepare_dataset(
    source_path: str,
    output_dir: str,
    apply_feature_engineering: bool = True
) -> str:
    """
    Load and prepare a dataset for model training
    
    Args:
        source_path: Path to the source data (can be local or S3)
        output_dir: Directory to save the processed data
        apply_feature_engineering: Whether to apply feature engineering
        
    Returns:
        Path to the processed dataset
    """
    logger.info(f"Preparing dataset from {source_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from source (handle S3 paths)
    if source_path.startswith('s3://'):
        # Download from S3
        local_path = os.path.join(output_dir, "raw_data.csv")
        s3_download(source_path, local_path)
        source_path = local_path
    
    # Load data based on file extension
    if source_path.endswith('.csv'):
        df = pd.read_csv(source_path)
    elif source_path.endswith('.parquet'):
        df = pd.read_parquet(source_path)
    else:
        raise ValueError(f"Unsupported file format: {source_path}")
    
    logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Basic cleaning
    # Handle missing values
    df = clean_dataset(df)
    
    # Apply feature engineering if requested
    if apply_feature_engineering:
        df = engineer_features(df)
    
    # Save processed data
    output_path = os.path.join(output_dir, "processed_data.parquet")
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Processed dataset saved to {output_path}")
    return output_path

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('unknown')
        elif df_clean[col].dtype in ['int64', 'float64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Handle outliers for numeric columns
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        # Calculate Q1, Q3 and IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers at bounds
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    # Make a copy to avoid modifying the original
    df_eng = df.copy()
    
    # Apply feature engineering based on available columns
    # These are common operations that may need to be customized for your dataset
    
    # Example: Convert categorical features to one-hot encoding
    cat_columns = df_eng.select_dtypes(include=['object']).columns
    for col in cat_columns:
        if df_eng[col].nunique() < 10:  # Only one-hot encode if few unique values
            dummies = pd.get_dummies(df_eng[col], prefix=col, drop_first=True)
            df_eng = pd.concat([df_eng, dummies], axis=1)
    
    # Example: Create interaction features for numeric columns
    num_columns = df_eng.select_dtypes(include=['int64', 'float64']).columns[:5]  # Limit to first 5 to avoid explosion
    for i, col1 in enumerate(num_columns):
        for col2 in num_columns[i+1:]:
            df_eng[f"{col1}_x_{col2}"] = df_eng[col1] * df_eng[col2]
    
    # Example: Create polynomial features for important numeric columns
    for col in num_columns[:3]:  # Only for first 3 numeric columns
        df_eng[f"{col}_squared"] = df_eng[col] ** 2
    
    # Example: Add date-based features if there are datetime columns
    date_columns = df_eng.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        df_eng[f"{col}_year"] = df_eng[col].dt.year
        df_eng[f"{col}_month"] = df_eng[col].dt.month
        df_eng[f"{col}_day"] = df_eng[col].dt.day
        df_eng[f"{col}_dayofweek"] = df_eng[col].dt.dayofweek
    
    return df_eng 