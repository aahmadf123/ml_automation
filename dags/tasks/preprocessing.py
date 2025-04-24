#!/usr/bin/env python3
"""
preprocessing.py

Handles:
  - Data loading from Parquet
  - Handling missing values
  - Outlier detection and capping
  - Categorical encoding
  - Generating pure_premium & sample_weight
  - Pandera schema validation (after pure_premium exists)
  - Selecting the correct loss-history features per MODEL_ID
  - Data profiling report + Slack notification via agent_actions
  - Integration of new UI components and endpoints
  - Skewness detection and transformation
  - Enhanced data profiling with Swifter

Requirements:
  - Python 3.11+
  - numpy>=1.24.0
  - pandas>=2.0.0
  - pyarrow>=14.0.0
  - scipy>=1.11.0
  - scikit-learn>=1.3.0
  - swifter[notebook,groupby]==1.4.0
  - ydata-profiling>=4.5.0
"""

import os
import json
import logging
import time
import gc
from typing import Dict, List, Tuple

import pandas as pd
from ydata_profiling import ProfileReport
from airflow.models import Variable
from tasks.schema_validation import validate_schema
from agent_actions import handle_function_call
import numpy as np
from datetime import datetime
from utils.storage import download as s3_download
from utils.storage import upload as s3_upload
from utils.slack import post as send_message
from tasks.data_quality import DataQualityMonitor
import multiprocessing as mp
from functools import partial
import swifter
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from tasks.cache import GLOBAL_CACHE, cache_result

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

PROFILE_REPORT_PATH = "/tmp/homeowner_profile_report.html"

# Prefix groups for loss-history features
RAW_PREFIXES = ["num_loss_3yr_", "num_loss_yrs45_", "num_loss_free_yrs_"]
DECAY_PREFIXES = {
    "model2": ["lhdwc_5y_1d_"],  # equal
    "model3": ["lhdwc_5y_2d_"],  # linear decay
    "model4": ["lhdwc_5y_3d_"],  # fast decay
    "model5": ["lhdwc_5y_4d_"],  # slow decay
}

# Set up logging
logger = logging.getLogger(__name__)

# Initialize data quality monitor
quality_monitor = DataQualityMonitor()

@cache_result
def load_data_to_dataframe(parquet_path: str) -> pd.DataFrame:
    """
    Read Parquet from local path into DataFrame with advanced pyarrow optimizations.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Use multi-threaded reading for better performance
    table = pq.read_table(parquet_path, use_threads=True, memory_pool=pa.default_memory_pool())
    
    # Convert to pandas
    df = table.to_pandas()
    
    # Downcast numeric columns to reduce memory usage
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Clean up the Arrow table to free memory
    del table
    import gc
    gc.collect()
    
    logging.info(f"Loaded data from {parquet_path}, shape={df.shape}")
    return df


@cache_result
def handle_missing_data(
    df: pd.DataFrame,
    strategy: str = "mean",
    missing_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Drop columns with > threshold missing, then impute remaining missing.
    Memory optimized version.
    """
    # Calculate null ratios efficiently
    null_ratio = df.isnull().mean()
    drop_cols = null_ratio[null_ratio > missing_threshold].index.tolist()
    
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logging.info(f"Dropped columns >{missing_threshold*100:.0f}% missing: {drop_cols}")

    # Store a copy for modification
    result_df = df.copy()
    
    # Cache dataset statistics to avoid recalculations
    df_name = f"preprocessing_{id(df)}"
    GLOBAL_CACHE.compute_statistics(df, df_name)
    
    # Handle remaining missing values based on strategy
    if strategy == "mean":
        # Calculate means only for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                # Get mean from cache instead of recalculating
                mean_val = GLOBAL_CACHE.get_statistic(df_name, col, 'mean')
                result_df[col] = result_df[col].fillna(mean_val)
    elif strategy == "zero":
        result_df = result_df.fillna(0)
    elif strategy == "ffill":
        result_df = result_df.fillna(method="ffill")
        # If still have nulls after ffill (at the beginning), use bfill
        result_df = result_df.fillna(method="bfill")
    else:
        raise ValueError(f"Unknown missing data strategy '{strategy}'")

    logging.info(f"Applied missing data strategy: {strategy}")
    return result_df


@cache_result
def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, int]:
    """
    Count outliers per numeric column using the IQR method.
    """
    counts: Dict[str, int] = {}
    # Process columns in parallel for large datasets
    if len(df) > 50000 and len(df.select_dtypes(include="number").columns) > 10:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        
        def process_col(df, col):
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (df[col] < q1 - factor * iqr) | (df[col] > q3 + factor * iqr)
            return int(mask.sum())
        
        # Use parallel processing with shared results
        counts = GLOBAL_CACHE.parallel_column_processing(df, process_col, numeric_cols)
    else:
        # Sequential processing for smaller datasets
        for col in df.select_dtypes(include="number"):
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (df[col] < q1 - factor * iqr) | (df[col] > q3 + factor * iqr)
            counts[col] = int(mask.sum())
    
    return counts


@cache_result
def cap_outliers(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.Series:
    """
    Clip numeric column values to within [Q1 - factor*IQR, Q3 + factor*IQR].
    Returns just the processed series for improved memory usage.
    """
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)


@cache_result
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One‑hot encode all object/categorical columns.
    """
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    logging.info(f"One‑hot encoded columns: {obj_cols}")
    return df


@cache_result
def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the loss‑history features needed for current MODEL_ID,
    plus all non–loss‑history columns.
    """
    model_id = (
        os.getenv("MODEL_ID")
        or Variable.get("MODEL_ID", default_var="model1")
    ).strip().lower()
    logging.info(f"Selecting features for MODEL_ID='{model_id}'")

    if model_id == "model1":
        keep_prefixes = RAW_PREFIXES
    elif model_id in DECAY_PREFIXES:
        keep_prefixes = DECAY_PREFIXES[model_id]
    else:
        raise ValueError(f"Unknown MODEL_ID '{model_id}'")

    all_loss_prefixes = RAW_PREFIXES + sum(DECAY_PREFIXES.values(), [])
    def is_loss_col(col: str) -> bool:
        return any(col.startswith(p) for p in all_loss_prefixes)

    keep_cols = [
        col for col in df.columns
        if any(col.startswith(p) for p in keep_prefixes) or not is_loss_col(col)
    ]
    removed = [col for col in df.columns if col not in keep_cols]
    logging.info(f"Dropped loss‑history cols for {model_id}: {removed}")
    return df[keep_cols]

@cache_result
def generate_profile_report(df: pd.DataFrame, output_path: str = PROFILE_REPORT_PATH) -> None:
    """
    Generate a profile report using ydata-profiling.
    """
    # Only sample if needed for large datasets
    if len(df) > 100000:
        sample = df.sample(min(100000, len(df) // 10))
        logging.info(f"Using {len(sample):,} rows for profiling (sampled from {len(df):,})")
    else:
        sample = df
        logging.info(f"Using all {len(df):,} rows for profiling")
    
    # For very wide datasets, limit columns to improve performance
    if len(df.columns) > 100:
        logging.info(f"Dataset has {len(df.columns)} columns, limiting profile to important columns")
        # Prioritize numeric columns and columns with fewer missing values
        col_info = []
        for col in df.columns:
            missing_pct = df[col].isna().mean()
            is_numeric = pd.api.types.is_numeric_dtype(df[col].dtype)
            col_info.append((col, missing_pct, is_numeric))
        
        # Sort by numeric first, then by missing percentage
        col_info.sort(key=lambda x: (-x[2], x[1]))
        selected_cols = [col for col, _, _ in col_info[:100]]
        sample = sample[selected_cols]
        logging.info(f"Selected {len(selected_cols)} columns for profiling")
    
    # Generate minimal report first, then expand if needed
    profile = ProfileReport(
        sample,
        title="Homeowner Loss History Data Profile",
        minimal=True,
        progress_bar=False,
        explorative=True
    )
    
    profile.to_file(output_path)
    logging.info(f"Profile report saved to {output_path}")


def process_column_group(df, columns, process_func):
    """
    Process a group of columns with the given function.
    This function is separate to make it picklable for multiprocessing.
    """
    result = {}
    for col in columns:
        result[col] = process_func(df, col)
    return result


@cache_result
def create_prior_loss_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a prior_loss_ind feature that indicates whether a policy has any prior losses.
    1 means no prior losses (loss-free), 0 means has prior losses.
    
    This is an important feature for insurance rating and risk segmentation.
    """
    model_id = (
        os.getenv("MODEL_ID")
        or Variable.get("MODEL_ID", default_var="model1")
    ).strip().lower()
    
    # Get model-specific loss history columns
    if model_id == "model1":
        model_columns = [col for col in df.columns if any(col.startswith(p) for p in RAW_PREFIXES)]
        # Special handling for model1 which often uses loss_free_yrs_total
        if 'loss_free_yrs_total' in df.columns:
            df['prior_loss_ind'] = (df['loss_free_yrs_total'] == 5).astype(int)
            logging.info("Created prior_loss_ind based on loss_free_yrs_total")
            return df
    elif model_id in DECAY_PREFIXES:
        prefix = DECAY_PREFIXES[model_id][0]
        model_columns = [col for col in df.columns if col.startswith(prefix)]
    else:
        # Try to find any loss history columns
        prefixes = RAW_PREFIXES + sum(DECAY_PREFIXES.values(), [])
        model_columns = [col for col in df.columns if any(col.startswith(p) for p in prefixes)]
    
    # If we found relevant columns, create the indicator
    if model_columns:
        df['prior_loss_ind'] = (df[model_columns].sum(axis=1) == 0).astype(int)
        logging.info(f"Created prior_loss_ind based on {len(model_columns)} loss history columns")
    else:
        # If no relevant columns found, check for summary columns
        total_cols = [col for col in df.columns if col.endswith('_total') and any(
            col.startswith(p) for p in prefixes)]
        if total_cols:
            df['prior_loss_ind'] = (df[total_cols].sum(axis=1) == 0).astype(int)
            logging.info(f"Created prior_loss_ind based on {len(total_cols)} summary columns")
        else:
            # Default to 0 if we can't determine
            df['prior_loss_ind'] = 0
            logging.warning("Could not find loss history columns to create prior_loss_ind, defaulting to 0")
    
    return df


def preprocess_data(data_path, output_path, force_reprocess=False):
    """
    Main preprocessing function that orchestrates all data transformation steps.
    
    Args:
        data_path: Path to input data file
        output_path: Path to save processed data
        force_reprocess: Whether to reprocess even if output exists
        
    Returns:
        Path to the processed file
    """
    start_time = time.time()
    logger.info(f"Starting preprocessing pipeline on {data_path}")
    
    # Check if output already exists
    if os.path.exists(output_path) and not force_reprocess:
        logger.info(f"Processed file already exists at {output_path}, skipping preprocessing")
        return output_path
    
    # 1. Load the data
    df = load_data_to_dataframe(data_path)
    orig_shape = df.shape
    logger.info(f"Loaded dataset with shape: {orig_shape}")
    
    # Store original data statistics for comparison
    df_name = "original_data"
    GLOBAL_CACHE.compute_statistics(df, df_name)
    
    # 2. Generate initial profile report
    logger.info("Generating initial profile report")
    initial_profile_path = "/tmp/homeowner_initial_profile.html"
    generate_profile_report(df, initial_profile_path)
    
    # 3. Handle missing data
    logger.info("Handling missing data")
    df = handle_missing_data(df, strategy="mean")
    
    # 4. Detect and handle outliers
    logger.info("Detecting outliers")
    outlier_counts = detect_outliers_iqr(df)
    total_outliers = sum(outlier_counts.values())
    logger.info(f"Found {total_outliers} outliers across {len(outlier_counts)} columns")
    
    # Only process columns with significant outliers
    outlier_threshold = 0.01 * len(df)  # 1% of data
    columns_to_process = [col for col, count in outlier_counts.items() 
                         if count > outlier_threshold]
    
    if columns_to_process:
        logger.info(f"Handling outliers in {len(columns_to_process)} columns")
        
        # For large datasets or many columns, use parallel processing
        if len(df) > 50000 and len(columns_to_process) > 10:
            # Split columns into groups for multiprocessing
            num_workers = min(mp.cpu_count(), len(columns_to_process))
            
            # Use the cache's parallel column processing
            processed_columns = GLOBAL_CACHE.parallel_column_processing(
                df, 
                cap_outliers, 
                columns_to_process,
                n_workers=num_workers
            )
            
            # Update dataframe with processed columns
            for col, processed_series in processed_columns.items():
                df[col] = processed_series
        else:
            # Process sequentially for smaller datasets
            for col in columns_to_process:
                df[col] = cap_outliers(df, col)
    
    # 5. Store intermediate results for reuse
    logger.info("Caching intermediate data")
    GLOBAL_CACHE.store_transformed(df, "after_outlier_handling")
    
    # 6. Detect and transform skewed features
    logger.info("Detecting and transforming skewed features")
    df = detect_and_transform_skewed_features(df)
    
    # 7. Check for data quality issues
    logger.info("Running data quality checks")
    quality_issues = quality_monitor.run_checks(df)
    if quality_issues:
        logger.warning(f"Found {len(quality_issues)} data quality issues")
        # Report issues but continue processing
        
    # 8. Encode categorical variables
    logger.info("Encoding categorical variables")
    df = encode_categoricals(df)
    
    # 9. Select features for the current model
    logger.info("Selecting model-specific features")
    df = select_model_features(df)
    
    # 10. Create prior loss indicator (insurance-specific feature)
    logger.info("Creating prior loss indicator")
    df = create_prior_loss_indicator(df)
    
    # 11. Generate final profile report
    logger.info("Generating final profile report")
    generate_profile_report(df, PROFILE_REPORT_PATH)
    
    # 12. Calculate correlations
    logger.info("Analyzing feature correlations")
    target_col = "pure_premium" if "pure_premium" in df.columns else None
    if target_col:
        correlations = GLOBAL_CACHE.compute_correlations(df, "final_data", target_col)
        top_correlations = correlations.abs().sort_values(ascending=False).head(10)
        logger.info(f"Top correlations with {target_col}: \n{top_correlations}")
    
    # 13. Save processed data
    logger.info(f"Saving processed data to {output_path}")
    
    # Get directory of output path
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Use efficient Parquet writing
    df.to_parquet(output_path, index=False, compression="snappy")
    
    # 14. Validate final schema
    try:
        logger.info("Validating final data schema")
        validate_schema(df)
        logger.info("Schema validation passed")
    except Exception as e:
        logger.warning(f"Schema validation failed: {str(e)}")
    
    # 15. Log summary statistics
    final_shape = df.shape
    elapsed_time = time.time() - start_time
    
    summary = {
        "original_rows": orig_shape[0],
        "original_columns": orig_shape[1],
        "final_rows": final_shape[0],
        "final_columns": final_shape[1],
        "processing_time_seconds": elapsed_time,
        "quality_issues": len(quality_issues) if quality_issues else 0
    }
    
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Summary: {summary}")
    
    # 16. Upload profile report to S3 if configured
    try:
        s3_bucket = os.getenv("REPORTS_BUCKET")
        if s3_bucket:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"reports/profile_{timestamp}.html"
            s3_upload(PROFILE_REPORT_PATH, s3_bucket, s3_key)
            logger.info(f"Uploaded profile report to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload profile report: {str(e)}")
    
    # 17. Send notification about completion
    try:
        message = f"✅ Preprocessing completed\n• Rows: {orig_shape[0]} → {final_shape[0]}\n• Columns: {orig_shape[1]} → {final_shape[1]}\n• Time: {elapsed_time:.2f}s"
        send_message(message, "#data-pipeline")
    except Exception as e:
        logger.error(f"Failed to send notification: {str(e)}")
    
    # Clean up temporary files
    for path in [initial_profile_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass
    
    # Clean up memory
    gc.collect()
    
    return output_path

@cache_result
def detect_and_transform_skewed_features(df, threshold=0.5):
    """
    Detect and transform skewed numeric features.
    """
    start_time = time.time()
    logger.info(f"Checking for skewness in {len(df.columns)} columns")
    
    # Only examine numeric columns with sufficient non-zero values
    numeric_cols = df.select_dtypes(include=["number"]).columns
    candidates = []
    
    # Cache statistics if not already cached
    df_name = f"skew_detection_{id(df)}"
    if df_name not in GLOBAL_CACHE.statistics_cache:
        GLOBAL_CACHE.compute_statistics(df, df_name)
    
    # Find columns to transform
    for col in numeric_cols:
        # Get statistics from cache
        skewness = GLOBAL_CACHE.get_statistic(df_name, col, 'skewness')
        non_zero_pct = 1.0 - GLOBAL_CACHE.get_statistic(df_name, col, 'zeros_pct')
        
        # Skip columns with mostly zeros or insufficient skewness
        if pd.isna(skewness) or abs(skewness) < threshold or non_zero_pct < 0.1:
            continue
            
        candidates.append((col, skewness))
    
    # Sort by absolute skewness for prioritization
    candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    logger.info(f"Found {len(candidates)} columns with significant skewness")
    
    # Only transform the most skewed columns (cap at 100 for performance)
    columns_to_transform = [col for col, _ in candidates[:100]]
    
    if not columns_to_transform:
        logger.info("No columns require transformation")
        return df
    
    # Transform skewed columns efficiently
    result_df = df.copy()
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    
    # For large datasets, process in batches
    if len(df) > 100000:
        batch_size = min(100000, len(df) // mp.cpu_count())
        logger.info(f"Using batch processing with size {batch_size} for large dataset")
        
        # Process in batches to avoid memory issues
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[i:batch_end, :]
            
            # Fit transform on batch
            transformed = transformer.fit_transform(batch_df[columns_to_transform])
            
            # Update result with transformed values
            for j, col in enumerate(columns_to_transform):
                result_df.iloc[i:batch_end, result_df.columns.get_loc(col)] = transformed[:, j]
                
            # Report progress
            if (i // batch_size) % 10 == 0:
                logger.info(f"Processed {batch_end}/{len(df)} rows for skewness transformation")
    else:
        # Process all at once for smaller datasets
        transformed = transformer.fit_transform(df[columns_to_transform])
        for i, col in enumerate(columns_to_transform):
            result_df[col] = transformed[:, i]
    
    elapsed = time.time() - start_time
    logger.info(f"Transformed {len(columns_to_transform)} skewed columns in {elapsed:.2f}s")
    
    # Store the transformed dataframe for reuse
    GLOBAL_CACHE.store_transformed(result_df, "after_skew_transform")
    return result_df

@cache_result
def analyze_correlations_with_swifter(df, target_col=None, threshold=0.7):
    """
    Analyze correlations between features, optionally with a target column.
    Using our new caching system.
    """
    # Use the global cache instead of recalculating
    df_name = f"correlation_analysis_{id(df)}"
    return GLOBAL_CACHE.compute_correlations(df, df_name, target_col)

@cache_result
def test_normality_with_swifter(df, columns=None, test_method='shapiro', sample_size=1000):
    """
    Test normality of numeric columns using Shapiro-Wilk or D'Agostino tests.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to test, or None for all numeric
        test_method: 'shapiro' or 'dagostino'
        sample_size: Maximum sample size for testing
        
    Returns:
        DataFrame with test results
    """
    if not columns:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Use our cache to avoid redundant sampling and calculations
    cache_key = f"normality_test_{test_method}_{sample_size}"
    cached_result = GLOBAL_CACHE.get_transformed(cache_key)
    if cached_result is not None:
        return cached_result
    
    logger.info(f"Testing normality of {len(columns)} columns using {test_method}")
    
    # Use parallel processing for multiple columns
    results = []
    
    def test_column(df, column):
        # Sample if needed
        series = df[column].dropna()
        if len(series) > sample_size:
            series = series.sample(sample_size, random_state=42)
            
        # Skip if not enough data
        if len(series) < 8:
            return {'column': column, 'test': test_method, 'statistic': None, 
                    'p_value': None, 'normal': None, 'n': len(series)}
            
        try:
            if test_method == 'shapiro':
                statistic, p_value = stats.shapiro(series)
            elif test_method == 'dagostino':
                statistic, p_value = stats.normaltest(series)
            else:
                raise ValueError(f"Unknown test method: {test_method}")
                
            return {'column': column, 'test': test_method, 'statistic': statistic, 
                    'p_value': p_value, 'normal': p_value > 0.05, 'n': len(series)}
        except Exception as e:
            logger.warning(f"Error testing normality for {column}: {str(e)}")
            return {'column': column, 'test': test_method, 'statistic': None, 
                    'p_value': None, 'normal': None, 'n': len(series), 'error': str(e)}
    
    # Use our cache's parallel column processing
    results_dict = GLOBAL_CACHE.parallel_column_processing(df, test_column, columns)
    results = [results_dict[col] for col in columns if col in results_dict]
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Cache for future use
    GLOBAL_CACHE.store_transformed(result_df, cache_key)
    
    return result_df

# Add more optimized functions here...

if __name__ == "__main__":
    # This allows the module to be run as a script for testing
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <parquet_input_path> [<output_path>]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/preprocessed_data.parquet"
    
    preprocess_data(input_path, output_path, force_reprocess=True)
