#!/usr/bin/env python3
"""
data_prep.py - Module for data preparation and feature engineering
-----------------------------------------------------------------
This module provides functionality for preparing datasets for model training.
It includes data loading, cleaning, and feature engineering functions.
Optimized for large datasets (1.6M+ records).
"""

import os
import logging
import pandas as pd
import numpy as np
import gc
import time
from typing import Optional, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq

# Import utility modules
from utils.storage import download as s3_download
from utils.cache import GLOBAL_CACHE, cache_result

logger = logging.getLogger(__name__)

@cache_result
def prepare_dataset(
    source_path: str,
    output_dir: str,
    apply_feature_engineering: bool = True,
    chunk_size: int = 500000  # Process large datasets in chunks
) -> str:
    """
    Load and prepare a dataset for model training - optimized for large datasets
    
    Args:
        source_path: Path to the source data (can be local or S3)
        output_dir: Directory to save the processed data
        apply_feature_engineering: Whether to apply feature engineering
        chunk_size: Number of rows to process at once (for chunked processing)
        
    Returns:
        Path to the processed dataset
    """
    start_time = time.time()
    logger.info(f"Preparing dataset from {source_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from source (handle S3 paths)
    if source_path.startswith('s3://'):
        # Download from S3
        local_path = os.path.join(output_dir, "raw_data.csv")
        s3_download(source_path, local_path)
        source_path = local_path
    
    # Determine if the dataset is large enough to need chunked processing
    # First check file size
    file_size_mb = os.path.getsize(source_path) / (1024 * 1024)
    use_chunks = file_size_mb > 500  # If file is larger than 500MB
    
    output_path = os.path.join(output_dir, "processed_data.parquet")
    
    if use_chunks:
        logger.info(f"Using chunked processing for large dataset ({file_size_mb:.1f} MB)")
        process_large_dataset(source_path, output_path, chunk_size, apply_feature_engineering)
    else:
        # Process normally for smaller datasets
        df = load_dataset(source_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Basic cleaning
        df = clean_dataset(df)
        
        # Apply feature engineering if requested
        if apply_feature_engineering:
            df = engineer_features(df)
        
        # Save processed data with optimized parquet settings
        save_optimized_parquet(df, output_path)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processed dataset saved to {output_path} in {elapsed_time:.2f} seconds")
    return output_path

def load_dataset(source_path: str) -> pd.DataFrame:
    """
    Load dataset with optimized settings based on file type
    """
    if source_path.endswith('.csv'):
        # Use optimized CSV reading for large files
        df = pd.read_csv(
            source_path,
            low_memory=False,
            dtype_backend="pyarrow"
        )
    elif source_path.endswith('.parquet'):
        # Use PyArrow for faster parquet reading
        table = pq.read_table(
            source_path,
            use_threads=True,
            memory_pool=pa.default_memory_pool()
        )
        df = table.to_pandas()
        del table
        gc.collect()
    else:
        raise ValueError(f"Unsupported file format: {source_path}")
    
    # Optimize memory usage
    return optimize_dtypes(df)

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage before optimization: {start_mem:.2f} MB")
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns to categorical if appropriate
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If cardinality is less than 50% of data
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage after optimization: {end_mem:.2f} MB")
    logger.info(f"Reduced by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df

def process_large_dataset(source_path: str, output_path: str, chunk_size: int, apply_feature_engineering: bool):
    """
    Process a large dataset in chunks to avoid memory issues
    """
    # Determine file format
    is_csv = source_path.endswith('.csv')
    
    # First pass - analyze data to get column types and statistics
    logger.info(f"First pass: analyzing data structure and statistics")
    
    if is_csv:
        # Read a sample to determine schema
        sample_df = pd.read_csv(source_path, nrows=10000)
        dtypes = sample_df.dtypes
        
        # Create chunk iterator
        chunks = pd.read_csv(source_path, chunksize=chunk_size, dtype_backend="pyarrow")
    else:
        # For parquet, read metadata to get schema
        parquet_file = pq.ParquetFile(source_path)
        table = parquet_file.read_row_group(0, use_threads=True)
        sample_df = table.to_pandas()
        dtypes = sample_df.dtypes
        
        # Create chunk iterator for parquet
        chunks = pq.ParquetFile(source_path).iter_batches(batch_size=chunk_size)
    
    # Initialize stats for numeric columns
    stats = {}
    for col in sample_df.select_dtypes(include=['number']).columns:
        stats[col] = {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0}
    
    # Process in chunks
    chunk_count = 0
    total_rows = 0
    
    # Process each chunk
    logger.info(f"Starting multi-pass processing with chunk size: {chunk_size}")
    
    # Open the output file for writing with appropriate schema
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Create an empty processed DataFrame with the proper schema
    processed_empty = clean_dataset(sample_df.iloc[0:0].copy())
    if apply_feature_engineering:
        processed_empty = engineer_features(processed_empty)
    
    # Create parquet writer
    schema = pa.Schema.from_pandas(processed_empty)
    writer = pq.ParquetWriter(output_path, schema)
    
    for chunk_num, chunk in enumerate(chunks):
        chunk_start = time.time()
        
        # Convert PyArrow batch to pandas if needed
        if not is_csv:
            chunk = chunk.to_pandas()
        
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        chunk_count += 1
        
        logger.info(f"Processing chunk {chunk_count}: {chunk_rows} rows")
        
        # Update statistics for numeric columns
        for col in stats:
            if col in chunk.columns and pd.api.types.is_numeric_dtype(chunk[col].dtype):
                chunk_stats = chunk[col].agg(['min', 'max', 'sum', 'count'])
                stats[col]['min'] = min(stats[col]['min'], chunk_stats['min'])
                stats[col]['max'] = max(stats[col]['max'], chunk_stats['max'])
                stats[col]['sum'] += chunk_stats['sum']
                stats[col]['count'] += chunk_stats['count']
        
        # Clean the chunk
        chunk = clean_dataset(chunk)
        
        # Apply feature engineering if needed
        if apply_feature_engineering:
            chunk = engineer_features(chunk)
        
        # Write the chunk to parquet file
        table = pa.Table.from_pandas(chunk)
        writer.write_table(table)
        
        # Free memory
        del chunk
        del table
        gc.collect()
        
        chunk_time = time.time() - chunk_start
        logger.info(f"Chunk {chunk_count} processed in {chunk_time:.2f} seconds")
    
    # Close the writer
    writer.close()
    
    logger.info(f"Completed processing {total_rows} rows in {chunk_count} chunks")
    
    # Log statistics
    logger.info(f"Dataset statistics summary:")
    for col, col_stats in list(stats.items())[:10]:  # Show first 10 columns only
        if col_stats['count'] > 0:
            mean = col_stats['sum'] / col_stats['count']
            logger.info(f"  {col}: min={col_stats['min']}, max={col_stats['max']}, mean={mean:.4f}")

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers
    Optimized for performance with large datasets
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values efficiently by column type
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # For numeric columns, use median (pre-calculate for efficiency)
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        else:
            # For non-numeric columns, use 'unknown'
            df_clean[col] = df_clean[col].fillna('unknown')
    
    # Handle outliers for numeric columns with vectorized operations
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    
    # Process in smaller batches if many numeric columns
    batch_size = 20
    for i in range(0, len(numeric_cols), batch_size):
        batch_cols = numeric_cols[i:i+batch_size]
        
        for col in batch_cols:
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

def engineer_features(df: pd.DataFrame, max_features: int = 100) -> pd.DataFrame:
    """
    Apply feature engineering to the dataset
    Optimized for large datasets with controls to prevent feature explosion
    
    Args:
        df: Input DataFrame
        max_features: Maximum number of new features to add
        
    Returns:
        DataFrame with engineered features
    """
    start_time = time.time()
    original_features = len(df.columns)
    
    # Make a copy to avoid modifying the original
    df_eng = df.copy()
    new_features_count = 0
    
    # One-hot encode selective categorical columns
    cat_columns = df_eng.select_dtypes(include=['object', 'category']).columns
    
    # Only encode columns with reasonable cardinality
    for col in cat_columns:
        n_unique = df_eng[col].nunique()
        # Only encode if few unique values and not too many new columns
        if n_unique < 10 and n_unique > 1 and new_features_count + n_unique < max_features:
            # Use get_dummies more efficiently - sparse=True for memory optimization
            dummies = pd.get_dummies(df_eng[col], prefix=col, drop_first=True, sparse=True)
            df_eng = pd.concat([df_eng, dummies], axis=1)
            new_features_count += len(dummies.columns)
    
    # Create polynomial features for selective numeric columns
    # Select only the most important numeric columns
    num_cols = df_eng.select_dtypes(include=['number']).columns
    if len(num_cols) > 5:
        # If many numeric columns, prioritize
        num_cols = num_cols[:5]  # Take first 5 as a simple heuristic
    
    # Create squared terms for selected numeric columns
    for col in num_cols:
        if new_features_count < max_features:
            df_eng[f"{col}_squared"] = df_eng[col].pow(2)
            new_features_count += 1
    
    # Create selective interaction terms (only if we have enough budget left)
    if new_features_count + 10 < max_features and len(num_cols) >= 2:
        # Create interaction features for pairs of important numeric columns
        for i, col1 in enumerate(num_cols[:3]):  # Limit to first 3
            for col2 in num_cols[i+1:min(i+3, len(num_cols))]:  # Limit interactions
                df_eng[f"{col1}_x_{col2}"] = df_eng[col1] * df_eng[col2]
                new_features_count += 1
                
                # Stop if we hit the feature limit
                if new_features_count >= max_features:
                    break
    
    # Handle datetime columns if any
    date_columns = [col for col in df_eng.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        try:
            # Try to convert to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(df_eng[col]):
                df_eng[col] = pd.to_datetime(df_eng[col], errors='coerce')
            
            # Extract useful datetime components
            if new_features_count < max_features:
                df_eng[f"{col}_year"] = df_eng[col].dt.year
                new_features_count += 1
            
            if new_features_count < max_features:
                df_eng[f"{col}_month"] = df_eng[col].dt.month
                new_features_count += 1
            
            # Stop if we hit the feature limit
            if new_features_count >= max_features:
                break
        except:
            logger.warning(f"Could not convert {col} to datetime")
    
    elapsed = time.time() - start_time
    logger.info(f"Feature engineering completed in {elapsed:.2f}s. Added {new_features_count} new features.")
    logger.info(f"Total features: {original_features} → {len(df_eng.columns)}")
    
    return df_eng

def save_optimized_parquet(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to parquet with optimized settings
    """
    # Convert to PyArrow Table first for better control
    table = pa.Table.from_pandas(df)
    
    # Write with optimized settings
    pq.write_table(
        table,
        output_path,
        compression='snappy',  # Good balance of speed and compression
        use_dictionary=True,
        write_statistics=True
    )
    
    logger.info(f"Saved optimized parquet file ({os.path.getsize(output_path) / (1024*1024):.2f} MB)") 