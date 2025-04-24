#!/usr/bin/env python3
"""
cache.py

A centralized caching system for machine learning operations.
Provides utilities to:
- Store and retrieve DataFrame statistics
- Cache transformed DataFrames and columns
- Cache correlations, feature importance, etc.
- Provide utilities for parallel processing with result sharing

This helps eliminate redundant calculations across tasks.
"""

import os
import pickle
import hashlib
import logging
import functools
import time
from typing import Dict, List, Callable, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import joblib

# Configure logging
logger = logging.getLogger(__name__)

class DataFrameCache:
    """
    Global cache for DataFrame operations to avoid redundant computations.
    """
    def __init__(self, cache_dir: Optional[str] = None, max_cache_items: int = 100):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store persistent cache files (or None for in-memory only)
            max_cache_items: Maximum number of items to keep in memory cache
        """
        # In-memory caches
        self.statistics_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.correlation_cache: Dict[str, pd.Series] = {}
        self.column_cache: Dict[str, Dict[str, Any]] = {}
        self.transformed_cache: Dict[str, pd.DataFrame] = {}
        self.result_cache: Dict[str, Any] = {}
        
        # Cache settings
        self.max_cache_items = max_cache_items
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        logger.info(f"Initialized DataFrameCache with {'persistent storage' if cache_dir else 'in-memory only'}")
    
    def _generate_df_fingerprint(self, df: pd.DataFrame) -> str:
        """
        Generate a unique fingerprint for a DataFrame based on:
        - Column names
        - Number of rows
        - Selected data samples
        - Data types
        """
        # Create a hash of column names, shape, and sample values
        columns_str = ','.join(df.columns.astype(str))
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        
        # Sample a few values for fingerprinting
        sample_size = min(100, len(df))
        if sample_size > 0:
            try:
                # Take systematic samples through the dataframe
                indices = np.linspace(0, len(df)-1, num=sample_size, dtype=int)
                samples = df.iloc[indices].head(1)
                sample_str = samples.to_json(orient='records')
            except:
                # Fallback if sampling fails
                sample_str = df.head(1).to_json(orient='records')
        else:
            sample_str = "{}"
            
        # Add datatypes to the fingerprint
        dtypes_str = str(df.dtypes.to_dict())
        
        # Combine elements and create hash
        fingerprint_data = f"{columns_str}|{shape_str}|{sample_str}|{dtypes_str}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()
    
    def compute_statistics(self, df: pd.DataFrame, df_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Compute and cache various statistics for each column in the DataFrame.
        
        Args:
            df: DataFrame to analyze
            df_name: Unique name for this DataFrame in the cache
            
        Returns:
            Dictionary with column statistics
        """
        if df_name in self.statistics_cache:
            return self.statistics_cache[df_name]
        
        start_time = time.time()
        logger.info(f"Computing statistics for DataFrame '{df_name}' with {len(df.columns)} columns")
        
        # Initialize statistics dictionary
        stats = {}
        
        # Process each column
        for col in df.columns:
            col_stats = {}
            series = df[col]
            
            # Identify column type
            is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
            is_categorical = pd.api.types.is_categorical_dtype(series.dtype) or series.dtype == 'object'
            
            # Basic statistics for all columns
            col_stats['missing_count'] = series.isna().sum()
            col_stats['missing_pct'] = series.isna().mean()
            
            if is_numeric:
                # Statistics for numeric columns
                non_na = series.dropna()
                col_stats['mean'] = non_na.mean() if len(non_na) > 0 else np.nan
                col_stats['median'] = non_na.median() if len(non_na) > 0 else np.nan
                col_stats['std'] = non_na.std() if len(non_na) > 0 else np.nan
                col_stats['min'] = non_na.min() if len(non_na) > 0 else np.nan
                col_stats['max'] = non_na.max() if len(non_na) > 0 else np.nan
                
                # Calculate percentiles
                if len(non_na) > 0:
                    col_stats['q1'] = non_na.quantile(0.25)
                    col_stats['q3'] = non_na.quantile(0.75)
                    col_stats['iqr'] = col_stats['q3'] - col_stats['q1']
                
                # Calculate skewness and zeros percentage
                if len(non_na) > 2:  # Skewness requires at least 3 non-NA values
                    col_stats['skewness'] = non_na.skew()
                    col_stats['zeros_pct'] = (non_na == 0).mean()
            
            if is_categorical:
                # Statistics for categorical columns
                non_na = series.dropna()
                if len(non_na) > 0:
                    # Get value counts
                    val_counts = non_na.value_counts()
                    col_stats['unique_count'] = len(val_counts)
                    col_stats['top_value'] = val_counts.index[0] if not val_counts.empty else None
                    col_stats['top_count'] = val_counts.iloc[0] if not val_counts.empty else 0
                    col_stats['top_pct'] = col_stats['top_count'] / len(non_na) if len(non_na) > 0 else 0
            
            # Store column statistics
            stats[col] = col_stats
            
        # Store in cache
        self.statistics_cache[df_name] = stats
        
        elapsed = time.time() - start_time
        logger.info(f"Computed statistics for '{df_name}' in {elapsed:.2f} seconds")
        
        return stats
    
    def get_statistic(self, df_name: str, column: str, stat_name: str) -> Any:
        """
        Retrieve a specific statistic for a column from the cache.
        
        Args:
            df_name: Name of the DataFrame in cache
            column: Column name
            stat_name: Name of the statistic to retrieve
            
        Returns:
            The requested statistic value or None if not found
        """
        if df_name not in self.statistics_cache:
            return None
        
        if column not in self.statistics_cache[df_name]:
            return None
            
        return self.statistics_cache[df_name][column].get(stat_name)
    
    def compute_correlations(self, df: pd.DataFrame, df_name: str, target_col: Optional[str] = None) -> pd.Series:
        """
        Compute correlations between all columns and optionally a target column.
        
        Args:
            df: DataFrame to analyze
            df_name: Unique name for this DataFrame
            target_col: Target column for correlation analysis, or None for all-to-all
            
        Returns:
            Series with correlation values
        """
        cache_key = f"{df_name}_{target_col or 'all'}_corr"
        
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
        
        start_time = time.time()
        logger.info(f"Computing correlations for '{df_name}'{f' with target {target_col}' if target_col else ''}")
        
        # Only include numeric columns
        numeric_df = df.select_dtypes(include='number')
        
        if target_col:
            # Correlations with specific target
            if target_col not in numeric_df.columns:
                logger.warning(f"Target column '{target_col}' not found in numeric columns")
                return pd.Series(dtype=float)
                
            correlations = numeric_df.corrwith(numeric_df[target_col]).drop(target_col)
        else:
            # All-to-all correlations (upper triangle of correlation matrix)
            corr_matrix = numeric_df.corr()
            correlations = pd.Series()
            
            for i, col1 in enumerate(corr_matrix.columns):
                for col2 in corr_matrix.columns[i+1:]:
                    key = f"{col1}_{col2}"
                    correlations[key] = corr_matrix.loc[col1, col2]
        
        # Store in cache
        self.correlation_cache[cache_key] = correlations
        
        elapsed = time.time() - start_time
        logger.info(f"Computed {len(correlations)} correlations in {elapsed:.2f} seconds")
        
        return correlations
    
    def store_transformed(self, df: pd.DataFrame, transform_name: str) -> None:
        """
        Store a transformed DataFrame in the cache.
        
        Args:
            df: DataFrame to store
            transform_name: Name of the transformation
        """
        # Apply cache size limit - remove oldest entries if exceeding max size
        if len(self.transformed_cache) >= self.max_cache_items:
            # Get the oldest item (first key in dict)
            oldest_key = next(iter(self.transformed_cache))
            del self.transformed_cache[oldest_key]
            logger.debug(f"Removed oldest transform '{oldest_key}' from cache")
        
        # Store new transformation
        self.transformed_cache[transform_name] = df.copy()
        logger.debug(f"Stored transformed DataFrame '{transform_name}' in cache")
        
        # Optionally persist to disk
        if self.cache_dir:
            try:
                cache_path = os.path.join(self.cache_dir, f"{transform_name}.pkl")
                joblib.dump(df, cache_path)
                logger.debug(f"Persisted transform '{transform_name}' to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to persist transform '{transform_name}': {str(e)}")
    
    def get_transformed(self, transform_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a transformed DataFrame from the cache.
        
        Args:
            transform_name: Name of the transformation
            
        Returns:
            The cached DataFrame or None if not found
        """
        # First check in-memory cache
        if transform_name in self.transformed_cache:
            logger.debug(f"Retrieved transform '{transform_name}' from memory cache")
            return self.transformed_cache[transform_name]
        
        # Then check persistent storage
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{transform_name}.pkl")
            if os.path.exists(cache_path):
                try:
                    df = joblib.load(cache_path)
                    # Store in memory for faster future access
                    self.transformed_cache[transform_name] = df
                    logger.debug(f"Retrieved transform '{transform_name}' from disk cache")
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load transform '{transform_name}' from disk: {str(e)}")
        
        return None
    
    def store_column_result(self, df_name: str, column: str, operation: str, result: Any) -> None:
        """
        Store results of column-specific operations.
        
        Args:
            df_name: DataFrame identifier
            column: Column name
            operation: Name of the operation performed
            result: Result to cache
        """
        cache_key = f"{df_name}_{column}_{operation}"
        
        # Initialize column cache if needed
        if df_name not in self.column_cache:
            self.column_cache[df_name] = {}
            
        if column not in self.column_cache[df_name]:
            self.column_cache[df_name][column] = {}
            
        # Store result
        self.column_cache[df_name][column][operation] = result
        logger.debug(f"Stored column result for '{df_name}.{column}.{operation}'")
    
    def get_column_result(self, df_name: str, column: str, operation: str) -> Optional[Any]:
        """
        Retrieve cached column operation result.
        
        Args:
            df_name: DataFrame identifier
            column: Column name
            operation: Name of the operation
            
        Returns:
            Cached result or None if not found
        """
        if df_name not in self.column_cache:
            return None
            
        if column not in self.column_cache[df_name]:
            return None
            
        return self.column_cache[df_name][column].get(operation)
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear specific or all cache types.
        
        Args:
            cache_type: Type of cache to clear, or None for all
        """
        if cache_type is None or cache_type == 'statistics':
            self.statistics_cache.clear()
            logger.info("Cleared statistics cache")
            
        if cache_type is None or cache_type == 'correlations':
            self.correlation_cache.clear()
            logger.info("Cleared correlation cache")
            
        if cache_type is None or cache_type == 'columns':
            self.column_cache.clear()
            logger.info("Cleared column result cache")
            
        if cache_type is None or cache_type == 'transformed':
            self.transformed_cache.clear()
            logger.info("Cleared transformed DataFrame cache")
            
        if cache_type is None or cache_type == 'results':
            self.result_cache.clear()
            logger.info("Cleared general result cache")
            
        # Optionally clear disk cache too
        if self.cache_dir and (cache_type is None or cache_type == 'disk'):
            try:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                logger.info(f"Cleared disk cache at {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {str(e)}")
    
    def parallel_column_processing(self, df: pd.DataFrame, process_func: Callable, 
                                 columns: List[str], n_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Process columns in parallel with shared results.
        
        Args:
            df: DataFrame to process
            process_func: Function that takes (df, column) and returns a result
            columns: List of columns to process
            n_workers: Number of worker processes (None for auto)
            
        Returns:
            Dictionary mapping column names to their processing results
        """
        if not columns:
            return {}
            
        # Determine number of workers
        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(columns))
            
        # For very small tasks, process sequentially
        if len(columns) <= 2 or n_workers <= 1:
            return {col: process_func(df, col) for col in columns}
        
        start_time = time.time()
        logger.info(f"Processing {len(columns)} columns in parallel with {n_workers} workers")
        
        # Split columns into chunks for workers
        chunk_size = max(1, len(columns) // n_workers)
        chunks = [columns[i:i+chunk_size] for i in range(0, len(columns), chunk_size)]
        
        # Create shared dictionary to collect results
        with Manager() as manager:
            results_dict = manager.dict()
            
            # Function for each worker to process a chunk of columns
            def process_chunk(chunk_columns):
                chunk_results = {}
                for col in chunk_columns:
                    chunk_results[col] = process_func(df, col)
                return chunk_results
            
            # Launch workers using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_results = {executor.submit(process_chunk, chunk): i 
                                for i, chunk in enumerate(chunks)}
                
                # Collect results as they complete
                for future in future_results:
                    chunk_results = future.result()
                    # Update shared dictionary
                    for col, result in chunk_results.items():
                        results_dict[col] = result
        
        # Convert manager dict to regular dict
        results = dict(results_dict)
        
        elapsed = time.time() - start_time
        logger.info(f"Completed parallel processing of {len(columns)} columns in {elapsed:.2f} seconds")
        
        return results


# Create global instance
GLOBAL_CACHE = DataFrameCache(
    cache_dir=os.environ.get('CACHE_DIR', None),
    max_cache_items=int(os.environ.get('MAX_CACHE_ITEMS', 100))
)


def cache_result(func):
    """
    Decorator to cache function results based on DataFrame fingerprinting.
    
    For functions that take a DataFrame as the first argument,
    this will generate a fingerprint of the DataFrame and use it
    as part of the cache key.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Try to find DataFrame in arguments
        df = None
        if args and isinstance(args[0], pd.DataFrame):
            df = args[0]
        elif 'df' in kwargs and isinstance(kwargs['df'], pd.DataFrame):
            df = kwargs['df']
            
        if df is None:
            # No DataFrame found, just call the function
            return func(*args, **kwargs)
            
        # Generate cache key
        func_name = func.__name__
        df_fingerprint = GLOBAL_CACHE._generate_df_fingerprint(df)
        
        # Add other args to fingerprint (excluding DataFrame)
        other_args = []
        for i, arg in enumerate(args):
            if i == 0 and arg is df:
                continue
            try:
                # Try to make a hashable representation
                if isinstance(arg, (str, int, float, bool, type(None))):
                    other_args.append(str(arg))
                elif isinstance(arg, (list, tuple)):
                    other_args.append(str(sorted(arg) if all(isinstance(x, (str, int, float)) for x in arg) else arg))
                else:
                    # For complex objects, use their id
                    other_args.append(f"{type(arg).__name__}_{id(arg)}")
            except:
                other_args.append(f"unhashable_{i}")
                
        # Add kwargs to fingerprint
        kwargs_str = []
        for k, v in sorted(kwargs.items()):
            if k == 'df' and v is df:
                continue
            try:
                if isinstance(v, (str, int, float, bool, type(None))):
                    kwargs_str.append(f"{k}={v}")
                elif isinstance(v, (list, tuple)):
                    kwargs_str.append(f"{k}={sorted(v) if all(isinstance(x, (str, int, float)) for x in v) else v}")
                else:
                    kwargs_str.append(f"{k}={type(v).__name__}_{id(v)}")
            except:
                kwargs_str.append(f"{k}=unhashable")
                
        # Combine all elements for final cache key
        cache_key = f"{func_name}_{df_fingerprint}_{'_'.join(other_args)}_{'_'.join(kwargs_str)}"
        cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Check if result is in cache
        if cache_key_hash in GLOBAL_CACHE.result_cache:
            logger.debug(f"Cache hit for {func_name}")
            return GLOBAL_CACHE.result_cache[cache_key_hash]
            
        # Not in cache, compute result
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Store in cache
        GLOBAL_CACHE.result_cache[cache_key_hash] = result
        logger.debug(f"Cached result for {func_name} (took {elapsed:.2f}s)")
        
        return result
        
    return wrapper 