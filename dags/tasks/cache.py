#!/usr/bin/env python3
"""
cache.py - Centralized caching system for ML operations

This module provides a caching system for expensive operations in the ML pipeline.
It helps eliminate repetitive calculations across different task modules by:
1. Storing and retrieving DataFrame statistics
2. Caching transformed dataframes and columns
3. Preserving calculations like correlations and feature importance
4. Providing utilities for parallel processing with result sharing

Usage:
    from tasks.cache import DataFrameCache

    # Initialize cache
    cache = DataFrameCache()
    
    # Store DataFrame statistics
    cache.compute_statistics(df, 'my_dataframe')
    
    # Retrieve statistics
    mean_value = cache.get_statistic('my_dataframe', 'column_name', 'mean')
    
    # Cache transformed data
    cache.store_transformed(df, 'transformed_df')
    
    # Retrieve transformed data
    transformed = cache.get_transformed('transformed_df')
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Union
from functools import lru_cache
import time
import multiprocessing as mp
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

# Directory for temporary cache files
CACHE_DIR = "/tmp/ml_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class DataFrameCache:
    """Centralized cache for DataFrame operations to avoid redundant computations."""
    
    def __init__(self, use_disk: bool = True, max_memory_items: int = 100):
        """
        Initialize the cache system.
        
        Args:
            use_disk: Whether to use disk for caching large objects
            max_memory_items: Maximum number of items to keep in memory
        """
        self.memory_cache = {}
        self.statistics_cache = {}
        self.correlation_cache = {}
        self.transformation_cache = {}
        self.importance_cache = {}
        self.use_disk = use_disk
        self.max_memory_items = max_memory_items
        logger.info(f"Initialized DataFrameCache with disk caching {'enabled' if use_disk else 'disabled'}")
    
    def _get_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash for a DataFrame to use as cache key."""
        # Use shape, column names, and sampling of values to create a hash
        sample_values = df.sample(min(100, len(df))).values.flatten()[:1000]
        hash_input = f"{df.shape}_{list(df.columns)}_{sample_values.mean()}_{sample_values.std()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def compute_statistics(self, df: pd.DataFrame, df_name: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute and cache various statistics for a DataFrame in a single pass.
        
        Args:
            df: DataFrame to analyze
            df_name: Name identifier for the DataFrame
            
        Returns:
            Dictionary of statistics by column
        """
        logger.info(f"Computing statistics for DataFrame '{df_name}' with shape {df.shape}")
        start_time = time.time()
        
        # Initialize statistics dictionary
        self.statistics_cache[df_name] = {}
        
        # Calculate statistics by data type
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Process numeric columns efficiently
        for col in numeric_cols:
            # Get non-null values for accurate statistics
            values = df[col].dropna()
            
            # Skip if no valid values
            if len(values) == 0:
                self.statistics_cache[df_name][col] = {
                    'mean': np.nan,
                    'median': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'missing_pct': 1.0,
                    'zeros_pct': np.nan,
                    'unique_count': 0,
                    'skewness': np.nan,
                    'kurtosis': np.nan
                }
                continue
                
            # Calculate all statistics in one pass
            self.statistics_cache[df_name][col] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'missing_pct': float((df[col].isnull().sum() / len(df))),
                'zeros_pct': float((values == 0).sum() / len(values)),
                'unique_count': int(values.nunique()),
                'skewness': float(stats.skew(values)),
                'kurtosis': float(stats.kurtosis(values))
            }
        
        # Process categorical columns
        for col in categorical_cols:
            values = df[col].dropna()
            
            # Skip if no valid values
            if len(values) == 0:
                self.statistics_cache[df_name][col] = {
                    'missing_pct': 1.0,
                    'unique_count': 0,
                    'top_value': None,
                    'top_count': 0
                }
                continue
                
            # Calculate basic statistics for categorical columns
            value_counts = values.value_counts()
            top_value = value_counts.index[0] if len(value_counts) > 0 else None
            top_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
            
            self.statistics_cache[df_name][col] = {
                'missing_pct': float((df[col].isnull().sum() / len(df))),
                'unique_count': int(values.nunique()),
                'top_value': top_value,
                'top_count': int(top_count)
            }
        
        # Log completion
        elapsed = time.time() - start_time
        logger.info(f"Computed statistics for {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns in {elapsed:.2f}s")
        
        return self.statistics_cache[df_name]
    
    def get_statistic(self, df_name: str, column: str, stat_name: str) -> Any:
        """
        Retrieve a cached statistic for a DataFrame column.
        
        Args:
            df_name: DataFrame identifier
            column: Column name
            stat_name: Statistic name (mean, std, etc.)
            
        Returns:
            The requested statistic or None if not found
        """
        try:
            return self.statistics_cache[df_name][column][stat_name]
        except KeyError:
            logger.warning(f"Statistic {stat_name} for column {column} in DataFrame {df_name} not found in cache")
            return None
    
    def compute_correlations(self, df: pd.DataFrame, df_name: str, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Compute and cache correlations for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            df_name: Name identifier for the DataFrame
            target_col: Optional target column for focused correlation
            
        Returns:
            DataFrame of correlations
        """
        # Create a unique key based on the dataframe and target column
        key = f"{df_name}_{target_col if target_col else 'all'}_corr"
        
        # Skip if already cached
        if key in self.correlation_cache:
            logger.info(f"Using cached correlations for {key}")
            return self.correlation_cache[key]
        
        logger.info(f"Computing correlations for DataFrame '{df_name}'{' with target ' + target_col if target_col else ''}")
        start_time = time.time()
        
        # Get numeric columns to correlate
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # If dataset is large, sample it for correlation calculation
        if len(df) > 100000:
            logger.info(f"Large DataFrame detected, sampling for correlation calculation")
            df_sample = df.sample(min(100000, int(len(df) * 0.1)), random_state=42)
        else:
            df_sample = df
        
        # Calculate correlations
        if target_col:
            # Focused correlation with target column
            if target_col in df_sample.columns:
                correlations = pd.Series(
                    {col: df_sample[col].corr(df_sample[target_col]) 
                     for col in numeric_cols if col != target_col}
                ).sort_values(ascending=False)
            else:
                logger.warning(f"Target column {target_col} not found in DataFrame")
                correlations = pd.Series()
        else:
            # Full correlation matrix
            correlations = df_sample[numeric_cols].corr()
        
        # Cache the result
        self.correlation_cache[key] = correlations
        
        # Log completion
        elapsed = time.time() - start_time
        logger.info(f"Computed correlations in {elapsed:.2f}s")
        
        return correlations
    
    def get_correlations(self, df_name: str, target_col: Optional[str] = None) -> pd.DataFrame:
        """Retrieve cached correlations."""
        key = f"{df_name}_{target_col if target_col else 'all'}_corr"
        return self.correlation_cache.get(key)
    
    def store_transformed(self, df: pd.DataFrame, key: str) -> None:
        """Store a transformed DataFrame in the cache."""
        if self.use_disk:
            # Store large DataFrames on disk
            logger.info(f"Storing transformed DataFrame '{key}' to disk cache")
            cache_file = os.path.join(CACHE_DIR, f"{key}.parquet")
            df.to_parquet(cache_file, index=False)
        else:
            # Store in memory
            logger.info(f"Storing transformed DataFrame '{key}' in memory cache")
            self.transformation_cache[key] = df
    
    def get_transformed(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve a transformed DataFrame from the cache."""
        if self.use_disk:
            cache_file = os.path.join(CACHE_DIR, f"{key}.parquet")
            if os.path.exists(cache_file):
                logger.info(f"Loading transformed DataFrame '{key}' from disk cache")
                return pd.read_parquet(cache_file)
        else:
            if key in self.transformation_cache:
                logger.info(f"Loading transformed DataFrame '{key}' from memory cache")
                return self.transformation_cache[key]
        
        logger.warning(f"Transformed DataFrame '{key}' not found in cache")
        return None
    
    def compute_feature_importance(self, df: pd.DataFrame, df_name: str, target_col: str, method: str = 'correlation') -> pd.Series:
        """
        Compute and cache feature importance.
        
        Args:
            df: DataFrame with features
            df_name: Name identifier for the DataFrame
            target_col: Target column name
            method: Method to use ('correlation' or 'mutual_info')
            
        Returns:
            Series with feature importance scores
        """
        # Create a unique key
        key = f"{df_name}_{target_col}_{method}_importance"
        
        # Skip if already cached
        if key in self.importance_cache:
            logger.info(f"Using cached feature importance for {key}")
            return self.importance_cache[key]
        
        logger.info(f"Computing feature importance for DataFrame '{df_name}' with target {target_col} using {method}")
        start_time = time.time()
        
        # Get numeric columns excluding the target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        if method == 'correlation':
            # Get or compute correlations
            corr_key = f"{df_name}_{target_col}_corr"
            if corr_key in self.correlation_cache:
                importance = self.correlation_cache[corr_key].abs().sort_values(ascending=False)
            else:
                importance = self.compute_correlations(df, df_name, target_col).abs().sort_values(ascending=False)
        
        elif method == 'mutual_info':
            try:
                from sklearn.feature_selection import mutual_info_regression
                
                # If dataset is large, sample it
                if len(df) > 100000:
                    logger.info(f"Large DataFrame detected, sampling for mutual information calculation")
                    df_sample = df.sample(min(100000, int(len(df) * 0.1)), random_state=42)
                else:
                    df_sample = df
                
                # Drop rows with NaN in target
                valid_mask = ~df_sample[target_col].isnull()
                for col in numeric_cols:
                    valid_mask = valid_mask & ~df_sample[col].isnull()
                
                if valid_mask.sum() > 10:  # Need enough valid data points
                    X = df_sample.loc[valid_mask, numeric_cols]
                    y = df_sample.loc[valid_mask, target_col]
                    mi_values = mutual_info_regression(X, y)
                    importance = pd.Series(mi_values, index=numeric_cols).sort_values(ascending=False)
                else:
                    logger.warning("Not enough valid data points for mutual information calculation")
                    importance = pd.Series()
            except Exception as e:
                logger.error(f"Error calculating mutual information: {str(e)}")
                importance = pd.Series()
        else:
            logger.error(f"Unknown feature importance method: {method}")
            importance = pd.Series()
        
        # Cache the result
        self.importance_cache[key] = importance
        
        # Log completion
        elapsed = time.time() - start_time
        logger.info(f"Computed feature importance in {elapsed:.2f}s")
        
        return importance
    
    def get_feature_importance(self, df_name: str, target_col: str, method: str = 'correlation') -> pd.Series:
        """Retrieve cached feature importance."""
        key = f"{df_name}_{target_col}_{method}_importance"
        return self.importance_cache.get(key)
    
    def parallel_column_processing(self, df: pd.DataFrame, func, columns: List[str], n_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Process columns in parallel with shared results.
        
        Args:
            df: DataFrame to process
            func: Function to apply to each column (takes df and column name)
            columns: List of column names to process
            n_workers: Number of worker processes to use
            
        Returns:
            Dictionary of {column_name: result}
        """
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        
        # For small number of columns, don't parallelize
        if len(columns) <= 3:
            return {col: func(df, col) for col in columns}
        
        logger.info(f"Processing {len(columns)} columns in parallel with {n_workers} workers")
        start_time = time.time()
        
        # Create arguments for each worker
        args_list = [(df, col) for col in columns]
        
        # Use a wrapper function that accepts a single argument
        def process_wrapper(args):
            data_df, column = args
            return column, func(data_df, column)
        
        # Create a pool and process columns in parallel
        with mp.Pool(n_workers) as pool:
            results = pool.map(process_wrapper, args_list)
        
        # Convert results list to dictionary
        result_dict = dict(results)
        
        # Log completion
        elapsed = time.time() - start_time
        logger.info(f"Parallel column processing completed in {elapsed:.2f}s")
        
        return result_dict
    
    def clear_cache(self) -> None:
        """Clear all in-memory caches."""
        self.memory_cache.clear()
        self.statistics_cache.clear()
        self.correlation_cache.clear()
        self.transformation_cache.clear()
        self.importance_cache.clear()
        
        # Clear disk cache if used
        if self.use_disk:
            import shutil
            try:
                for file in os.listdir(CACHE_DIR):
                    os.remove(os.path.join(CACHE_DIR, file))
                logger.info(f"Cleared disk cache in {CACHE_DIR}")
            except Exception as e:
                logger.error(f"Error clearing disk cache: {str(e)}")
    
    def __str__(self) -> str:
        """String representation with cache statistics."""
        stats = {
            'statistics_cache': len(self.statistics_cache),
            'correlation_cache': len(self.correlation_cache),
            'transformation_cache': len(self.transformation_cache),
            'importance_cache': len(self.importance_cache),
        }
        return f"DataFrameCache(stats={stats})"

# Create a singleton instance for import by other modules
GLOBAL_CACHE = DataFrameCache()

# Decorator for caching function results with dataset fingerprinting
def cache_result(func):
    """Decorator to cache function results based on DataFrame fingerprint."""
    def wrapper(df, *args, **kwargs):
        if not isinstance(df, pd.DataFrame):
            # If first argument is not a DataFrame, simply call the function
            return func(df, *args, **kwargs)
        
        # Create a cache key based on function name, df fingerprint, and arguments
        df_hash = GLOBAL_CACHE._get_dataframe_hash(df)
        key = f"{func.__name__}_{df_hash}_{str(args)}_{str(sorted(kwargs.items()))}"
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        # Check if result is in cache
        if key_hash in GLOBAL_CACHE.memory_cache:
            logger.debug(f"Using cached result for {func.__name__}")
            return GLOBAL_CACHE.memory_cache[key_hash]
        
        # Call the function
        result = func(df, *args, **kwargs)
        
        # Cache the result
        GLOBAL_CACHE.memory_cache[key_hash] = result
        
        # Limit memory cache size
        if len(GLOBAL_CACHE.memory_cache) > GLOBAL_CACHE.max_memory_items:
            # Remove a random item (simple strategy)
            GLOBAL_CACHE.memory_cache.pop(next(iter(GLOBAL_CACHE.memory_cache)))
        
        return result
    
    return wrapper

if __name__ == "__main__":
    # Example usage (for testing only)
    test_bucket = "grange-seniordesign-bucket"
    test_key = "raw-data/ut_loss_history_1.csv"
    test_local = "/tmp/homeowner_data.csv"

    # Check cache validity
    valid = is_cache_valid(test_bucket, test_key, test_local)
    print(f"Cache valid: {valid}")

    # If not valid, update cache
    if not valid:
        update_cache(test_bucket, test_key, test_local)
        print("Cache updated.")
    
    # Optionally clear cache (uncomment for testing cleanup)
    # clear_cache(test_local)
    # print("Cache cleared.")
