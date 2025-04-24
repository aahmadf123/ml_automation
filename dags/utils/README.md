# ML Automation Caching System

This directory contains utility modules for the ML Automation pipeline, including a powerful caching system for optimizing resource-intensive operations.

## Caching System Overview

The `cache.py` module provides a centralized caching system for machine learning operations to eliminate redundant calculations across different tasks in the ML pipeline. This helps significantly improve performance, especially for operations that are repeated across different DAG runs.

### Key Features

- **DataFrame Statistics Caching**: Store and retrieve basic statistics for DataFrames
- **Correlation Caching**: Cache correlation matrices and feature importance calculations
- **Transformation Caching**: Store transformed DataFrames to avoid recomputing them
- **Column-specific Results Caching**: Cache results of column-specific operations
- **Parallel Processing**: Utilities for parallel processing with efficient result sharing
- **Fingerprinting**: Intelligent tracking of DataFrame identity for accurate cache hits

## Usage

### Basic Usage

```python
from utils.cache import GLOBAL_CACHE, cache_result

# Cache statistics for a DataFrame
df_name = "my_dataframe"
GLOBAL_CACHE.compute_statistics(df, df_name)

# Retrieve a specific statistic
mean_value = GLOBAL_CACHE.get_statistic(df_name, 'column_name', 'mean')

# Store a transformed DataFrame
GLOBAL_CACHE.store_transformed(transformed_df, "transformed_data")

# Retrieve a transformed DataFrame
cached_df = GLOBAL_CACHE.get_transformed("transformed_data")
```

### Using the `@cache_result` Decorator

The simplest way to use the caching system is with the `@cache_result` decorator, which automatically caches function results based on the DataFrame fingerprint and function arguments:

```python
from utils.cache import cache_result

@cache_result
def process_dataframe(df, threshold=0.5):
    """This function's result will be cached based on df fingerprint and threshold."""
    # Expensive operations...
    return result
```

### Parallel Column Processing

For operations that need to be applied to multiple columns, the caching system provides parallel processing utilities:

```python
def process_column(df, column):
    # Process a single column...
    return result

# Process multiple columns in parallel with caching
results = GLOBAL_CACHE.parallel_column_processing(df, process_column, columns_list)
```

## Examples

### Caching Statistics

```python
from utils.cache import GLOBAL_CACHE

# Compute and cache statistics
df_name = f"data_quality_{id(df)}"
stats = GLOBAL_CACHE.compute_statistics(df, df_name)

# Later, retrieve a specific statistic
mean = GLOBAL_CACHE.get_statistic(df_name, 'column_name', 'mean')
```

### Caching Feature Importance

```python
# Check if we've already cached this calculation
model_hash = str(hash(model))
df_hash = str(id(X))
cache_key = f"{model_id}_{model_hash}_{df_hash}_importance"

# Look for feature importance in column cache
cached_result = GLOBAL_CACHE.get_column_result(
    df_name=f"{model_id}_df",
    column="all",
    operation=f"feature_importance_{model_hash}"
)

if cached_result is not None:
    return cached_result
    
# Calculate feature importance
# ...

# Store in cache for future use
GLOBAL_CACHE.store_column_result(
    df_name=f"{model_id}_df",
    column="all",
    operation=f"feature_importance_{model_hash}",
    result=result
)
```

### Optimizing Data Drift Detection

```python
@cache_result
def detect_data_drift(processed_data_path):
    """This function will cache its results based on file contents."""
    # Check if we have already computed statistics
    df_hash = os.path.basename(processed_data_path).replace('.parquet', '')
    df_name = f"drift_detection_{df_hash}"
    
    # Use cached data if available
    cached_df = GLOBAL_CACHE.get_transformed(df_name)
    if cached_df is not None:
        # Use cached data
        # ...
```

## Best Practices

1. **Use Meaningful Keys**: Choose descriptive identifiers for cached items
2. **Fingerprinting**: For DataFrames, use the built-in fingerprinting or generate your own based on key properties
3. **Cache Invalidation**: Use the `clear_cache()` method when needed
4. **Memory Management**: Set appropriate `max_cache_items` to avoid excessive memory usage
5. **Persistent Storage**: Enable persistent storage for larger cache items

## Notes

- In-memory cache has a default limit of 100 items, configurable via `max_cache_items`
- Persistent storage can be enabled by setting the `cache_dir` parameter
- Environment variables can be used to configure the cache:
  - `CACHE_DIRECTORY`: Path to persistent cache storage
  - `MAX_CACHE_ITEMS`: Maximum number of items to keep in memory

For more information, see the [full documentation](./cache.py). 