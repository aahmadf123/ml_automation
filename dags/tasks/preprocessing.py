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
        df.drop(columns=drop_cols, inplace=True)
        logging.info(f"Dropped columns >{missing_threshold*100:.0f}% missing: {drop_cols}")

    # Handle remaining missing values based on strategy
    if strategy == "mean":
        # Calculate means only for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
    elif strategy == "zero":
        df.fillna(0, inplace=True)
    elif strategy == "ffill":
        df.fillna(method="ffill", inplace=True)
        # If still have nulls after ffill (at the beginning), use bfill
        df.fillna(method="bfill", inplace=True)
    else:
        raise ValueError(f"Unknown missing data strategy '{strategy}'")

    logging.info(f"Applied missing data strategy: {strategy}")
    return df


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, int]:
    """
    Count outliers per numeric column using the IQR method.
    """
    counts: Dict[str, int] = {}
    for col in df.select_dtypes(include="number"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - factor * iqr) | (df[col] > q3 + factor * iqr)
        counts[col] = int(mask.sum())
    return counts


def cap_outliers(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Clip numeric column values to within [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One‑hot encode all object/categorical columns.
    """
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    logging.info(f"One‑hot encoded columns: {obj_cols}")
    return df


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


def generate_profile_report(df: pd.DataFrame, output_path: str = PROFILE_REPORT_PATH) -> None:
    """
    Generate a minimal profiling report, write to HTML, and notify Slack.
    """
    try:
        profile = ProfileReport(df, title="Homeowner Data Profile", minimal=True)
        profile.to_file(output_path)
        logging.info(f"Saved profiling report to {output_path}")
    except Exception as e:
        logging.warning(f"Failed to generate profile report: {e}")
        return

    try:
        send_message(
            channel="#agent_logs",
            title="📊 Profiling Summary",
            details=f"Profile saved to {output_path}",
            urgency="low"
        )
    except Exception as e:
        logging.warning(f"Slack notification failed: {e}")


def preprocess_data(data_path, output_path, force_reprocess=False):
    """
    Preprocess the data and perform quality checks.
    Uses advanced memory-efficient processing with pyarrow 14.0.2 optimizations.
    
    Features:
    - Optimized data loading with PyArrow
    - Memory-efficient processing using batch operations
    - Parallel processing with Swifter and multiprocessing
    - Enhanced data quality monitoring
    - Advanced normality testing with Shapiro-Wilk and D'Agostino's K^2 tests
    - Intelligent skewness detection and adaptive transformation selection
    - Multiple transformation methods (Yeo-Johnson, Box-Cox, Log, Square root)
    - Feature importance analysis with correlation and mutual information
    - Correlation analysis for exploratory data understanding
    - Comprehensive data profiling with detailed statistics
    
    Args:
        data_path: Path to input data file (parquet format)
        output_path: Path to write processed output file
        force_reprocess: If True, always reprocess even if output file exists
        
    Returns:
        Path to output file
    """
    try:
        # Import necessary dependencies
        import pandas as pd
        import numpy as np
        from datetime import datetime
        import os
        import json
        import gc
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.dataset as ds
        import pyarrow.compute as pc
        from utils.storage import upload as s3_upload
        from utils.slack import post as send_message
        from tasks.data_quality import DataQualityMonitor
        import multiprocessing as mp
        from functools import partial
        import swifter
        
        # Check if output file already exists and is valid
        if not force_reprocess and os.path.exists(output_path):
            try:
                # Check if the existing file is a valid parquet file with expected schema
                logger = logging.getLogger(__name__)
                logger.info(f"Output file {output_path} already exists. Checking if it's valid...")
                
                # Try to open the file with pyarrow to validate it
                existing_dataset = ds.dataset(output_path, format="parquet")
                existing_schema = existing_dataset.schema
                
                # Check if pure_premium column exists (indicates the file was properly processed)
                if 'pure_premium' in [field.name for field in existing_schema]:
                    logger.info(f"Valid preprocessed file already exists at {output_path} with pure_premium column. Skipping preprocessing.")
                    
                    # Just to make sure the file is readable, try to read a small sample
                    test_sample = pq.read_table(output_path, columns=['pure_premium'], nrows=5)
                    
                    # Create a minimal performance report for consistent return format
                    return output_path
            except Exception as e:
                # If there's any error reading the existing file, log it and continue with processing
                logger.warning(f"Existing file at {output_path} couldn't be validated: {str(e)}. Proceeding with preprocessing.")
        elif force_reprocess and os.path.exists(output_path):
            logger = logging.getLogger(__name__)
            logger.info(f"Force reprocessing enabled. Reprocessing data even though {output_path} exists.")
        
        # Get available cores, but reserve one for system operations
        n_cores = max(mp.cpu_count() - 1, 1)
        
        # Set pandas options for memory efficiency
        pd.options.mode.chained_assignment = None  # default='warn'
        
        # Initialize quality monitor
        quality_monitor = DataQualityMonitor()
        
        # Load data with memory optimization
        logger = logging.getLogger(__name__)
        logger.info(f"Loading data from {data_path} with advanced pyarrow 14.0.2 optimizations using {n_cores} cores")
        
        # First, explore the dataset schema
        dataset = ds.dataset(data_path, format="parquet")
        schema = dataset.schema
        
        # Log schema information
        logger.info(f"Dataset schema: {schema}")
        
        # Set batch processing parameters for optimal performance
        # Adjust batch_size based on available memory
        batch_size = 100000  # Increased from 50000 for better throughput
        
        # Create scanner with optimized parallelism - adjust batch_readahead based on batch_size
        scanner = ds.Scanner.from_dataset(
            dataset,
            use_threads=True,
            batch_size=batch_size,
            batch_readahead=2  # Read ahead 2 batches for better I/O utilization
        )
        
        # Process in batches to control memory usage
        processed_rows = 0
        start_time = time.time()
        
        # Define optimized batch processing function
        def process_batch(batch, batch_num):
            nonlocal processed_rows
            # Convert to pandas for processing
            df_batch = batch.to_pandas(split_blocks=True, self_destruct=True)  # Use pyarrow's memory optimization
            batch_rows = len(df_batch)
            processed_rows += batch_rows
            
            # Downcast numeric columns for memory efficiency
            for col in df_batch.select_dtypes(include=['float64']).columns:
                df_batch[col] = pd.to_numeric(df_batch[col], downcast='float')
            for col in df_batch.select_dtypes(include=['int64']).columns:
                df_batch[col] = pd.to_numeric(df_batch[col], downcast='integer')
            
            # Avoid logging for every batch to reduce overhead
            if batch_num % 5 == 0 or batch_rows < 10000:
                batch_time = time.time() - start_time
                rows_per_sec = processed_rows / max(batch_time, 0.001)
                logger.info(f"Processed batch {batch_num}: {batch_rows} rows, Total: {processed_rows} rows, Speed: {rows_per_sec:.0f} rows/sec")
            
            return df_batch
        
        # Read and process batches - using list comprehension for efficiency
        logger.info(f"Starting batch processing with size {batch_size}")
        data_batches = [process_batch(batch, i) for i, batch in enumerate(scanner.to_batches())]
        
        # Combine processed batches efficiently
        logger.info("Combining processed batches")
        data = pd.concat(data_batches, ignore_index=True, copy=False)
        del data_batches
        gc.collect()
        
        process_time = time.time() - start_time
        logger.info(f"Data loaded and preprocessed in {process_time:.1f}s, shape={data.shape}, speed={processed_rows/max(process_time, 0.001):.0f} rows/sec")
        
        # Generate initial data profile using swifter
        logger.info("Generating initial data profile")
        initial_profile = profile_dataframe_with_swifter(data, sample_size=min(50000, len(data)))
        
        # Save initial profile to a temporary JSON file for reference
        profile_path = "/tmp/initial_profile.json"
        with open(profile_path, 'w') as f:
            json.dump(initial_profile, f, indent=2, default=str)
        logger.info(f"Initial profile saved to {profile_path}")
        
        # Perform initial quality check
        logger.info("Performing initial quality check")
        quality_report = quality_monitor.run_quality_checks(data)
        if quality_report['status'] == 'fail':
            send_message(
                channel="#alerts",
                title="⚠️ Data Quality Issues Detected",
                details=f"Quality report: {quality_report}",
                urgency="medium"
            )
        
        # Optimize memory before heavy processing
        gc.collect()
        
        # Basic preprocessing - with timing for performance analysis
        start_time = time.time()
        logger.info("Handling missing data")
        data = handle_missing_data(data)
        missing_time = time.time() - start_time
        logger.info(f"Missing data handling completed in {missing_time:.1f}s")
        gc.collect()
        
        start_time = time.time()
        logger.info("Handling outliers")
        # Optimize outlier handling by processing in parallel for large datasets
        if len(data) > 100000:
            # Define a parallel version of outlier handling for large numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            # Filter out categorical columns that might be stored as numeric
            numeric_cols = [col for col in numeric_cols if data[col].nunique() > 20]
            
            if len(numeric_cols) > 10:  # Only parallelize if we have many numeric columns
                # Process in parallel groups for efficiency
                column_groups = [numeric_cols[i:i+max(1, len(numeric_cols)//n_cores)] 
                                 for i in range(0, len(numeric_cols), max(1, len(numeric_cols)//n_cores))]
                
                def process_column_group(cols):
                    result = data[cols].copy()
                    for col in cols:
                        # Calculate z-scores efficiently
                        mean_val = data[col].mean()
                        std_val = data[col].std()
                        if std_val > 0:
                            z_scores = np.abs((data[col] - mean_val) / std_val)
                            mask = z_scores > 3
                            if mask.any():
                                # Get the data type of the column
                                dtype = data[col].dtype
                                # Explicitly cast the mean value to the column's data type
                                if np.issubdtype(dtype, np.integer):
                                    mean_val = int(mean_val)
                                result.loc[mask, col] = mean_val
                    return result
                
                # Create a pool and process column groups in parallel
                with mp.Pool(n_cores) as pool:
                    results = pool.map(process_column_group, column_groups)
                
                # Update data with the processed results
                for i, cols in enumerate(column_groups):
                    data[cols] = results[i]
                
                del results
            else:
                data = handle_outliers(data)
        else:
            data = handle_outliers(data)
            
        outlier_time = time.time() - start_time
        logger.info(f"Outlier handling completed in {outlier_time:.1f}s")
        gc.collect()
        
        # NEW: Test normality of numeric features
        start_time = time.time()
        logger.info("Testing normality of numeric features")
        
        # Get numeric columns for testing
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        num_columns = len(numeric_cols)
        
        # Only run normality tests if we have enough data and numeric columns
        if len(data) > 1000 and num_columns > 0:
            # Use sample_size based on data size
            sample_size = min(5000, len(data))  # Limit to 5000 samples for performance
            
            # For large datasets with many columns, focus on potentially important columns
            if len(data) > 100000 and num_columns > 20:
                logger.info(f"Large dataset detected with {num_columns} numeric columns. Testing a subset.")
                # Sample columns or use some heuristic to select important ones
                test_columns = list(numeric_cols[:20])  # Take first 20 columns as an example
                logger.info(f"Testing normality of {len(test_columns)} columns")
            else:
                test_columns = list(numeric_cols)
                logger.info(f"Testing normality of all {len(test_columns)} numeric columns")
            
            # Run different normality tests
            shapiro_results = test_normality_with_swifter(
                data, 
                columns=test_columns, 
                test_method='shapiro',
                sample_size=sample_size
            )
            
            # For larger datasets, use D'Agostino's test which is more suitable for larger samples
            if len(data) > 5000:
                dagostino_results = test_normality_with_swifter(
                    data, 
                    columns=test_columns, 
                    test_method='normaltest',
                    sample_size=sample_size
                )
            
            # Save normality test results
            if not shapiro_results.empty:
                norm_path = "/tmp/normality_test_results.csv"
                shapiro_results.to_csv(norm_path, index=False)
                logger.info(f"Normality test results saved to {norm_path}")
                
                # Identify columns that need transformation
                non_normal_cols = shapiro_results[~shapiro_results['is_normal']]['column'].tolist()
                if non_normal_cols:
                    logger.info(f"Non-normal distributions detected in {len(non_normal_cols)} columns")
                    # These columns will be candidates for skewness transformation
                    skew_candidates = non_normal_cols
                else:
                    logger.info("All tested columns passed normality tests")
                    skew_candidates = []
            else:
                logger.info("No valid normality test results")
                skew_candidates = []
        else:
            logger.info("Dataset too small or no numeric columns for normality testing")
            skew_candidates = []
            
        normality_time = time.time() - start_time
        logger.info(f"Normality testing completed in {normality_time:.1f}s")
        gc.collect()
        
        # NEW: Detect and transform skewed features
        start_time = time.time()
        logger.info("Detecting and transforming skewed features")
        skew_threshold = 0.5  # Configurable threshold for skewness detection
        
        # Only run skewness detection/transformation if we have enough data
        if len(data) > 1000:
            # If we have normality test results, prioritize those columns
            if 'skew_candidates' in locals() and skew_candidates:
                logger.info(f"Using {len(skew_candidates)} candidates from normality tests for skewness transformation")
                # Calculate skewness only for non-normal columns
                skew_values = data[skew_candidates].swifter.progress_bar(False).apply(lambda x: stats.skew(x.dropna()))
                skewed_features = skew_values[abs(skew_values) > skew_threshold].index.tolist()
                logger.info(f"Found {len(skewed_features)} skewed features among {len(skew_candidates)} non-normal candidates")
            else:
                # Perform regular skewness detection and transformation
                data, skewed_features = detect_and_transform_skewed_features(data, threshold=skew_threshold)
            
            # Apply transformations if skewed features were found
            if skewed_features:
                logger.info(f"Applying transformations to {len(skewed_features)} skewed features")
                
                # Track transformations for reporting
                transformation_results = {}
                
                # Process each skewed feature
                for col in skewed_features:
                    # Select the best transformation method
                    method, _ = select_best_transformation(data, col)
                    
                    if method and method != "none":
                        logger.info(f"Selected {method} transformation for {col}")
                        
                        # Apply the chosen transformation to the full dataset
                        if method == "yeo-johnson":
                            try:
                                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                                # Process in batches for large datasets
                                if len(data) > 100000:
                                    result = np.zeros(len(data))
                                    batch_size = 100000
                                    num_batches = len(data) // batch_size + 1
                                    
                                    for i in range(num_batches):
                                        start_idx = i * batch_size
                                        end_idx = min((i + 1) * batch_size, len(data))
                                        batch = data[col].iloc[start_idx:end_idx].values.reshape(-1, 1)
                                        mask = ~np.isnan(batch).flatten()
                                        if mask.any():
                                            transform = pt.fit(batch[mask])
                                            batch_result = np.array(batch, copy=True).flatten()
                                            batch_result[mask] = transform.transform(batch[mask]).flatten()
                                            result[start_idx:end_idx] = batch_result
                                        
                                        data[f"{col}_transformed"] = result
                                        transformation_results[col] = "yeo-johnson"
                                    else:
                                        values = data[[col]].values
                                        mask = ~np.isnan(values).flatten()
                                        if mask.any():
                                            transform = pt.fit(values[mask].reshape(-1, 1))
                                            result = np.array(values, copy=True).flatten()
                                            result[mask] = transform.transform(values[mask].reshape(-1, 1)).flatten()
                                            data[f"{col}_transformed"] = result
                                            transformation_results[col] = "yeo-johnson"
                            except Exception as e:
                                logger.warning(f"Yeo-Johnson transformation failed for {col}: {str(e)}")
                        
                        elif method == "box-cox":
                            try:
                                # Check if all values are positive
                                if data[col].min() <= 0:
                                    # Add a constant to make all values positive
                                    shift = abs(data[col].min()) + 1.0
                                    data[f"{col}_shifted"] = data[col] + shift
                                    col = f"{col}_shifted"
                                    
                                    pt = PowerTransformer(method='box-cox', standardize=False)
                                    # Process in batches for large datasets
                                    if len(data) > 100000:
                                        result = np.zeros(len(data))
                                        batch_size = 100000
                                        num_batches = len(data) // batch_size + 1
                                        
                                        for i in range(num_batches):
                                            start_idx = i * batch_size
                                            end_idx = min((i + 1) * batch_size, len(data))
                                            batch = data[col].iloc[start_idx:end_idx].values.reshape(-1, 1)
                                            mask = ~np.isnan(batch).flatten()
                                            if mask.any():
                                                transform = pt.fit(batch[mask])
                                                batch_result = np.array(batch, copy=True).flatten()
                                                batch_result[mask] = transform.transform(batch[mask]).flatten()
                                                result[start_idx:end_idx] = batch_result
                                            
                                        # Apply the full transformation after all batches
                                        orig_col = col.replace("_shifted", "")
                                        data[f"{orig_col}_transformed"] = result
                                        transformation_results[orig_col] = "box-cox"
                                    else:
                                        # For smaller datasets
                                        values = data[[col]].values
                                        mask = ~np.isnan(values).flatten()
                                        if mask.any():
                                            transform = pt.fit(values[mask].reshape(-1, 1))
                                            result = np.array(values, copy=True).flatten()
                                            result[mask] = transform.transform(values[mask].reshape(-1, 1)).flatten()
                                            orig_col = col.replace("_shifted", "")
                                            data[f"{orig_col}_transformed"] = result
                                            transformation_results[orig_col] = "box-cox"
                            except Exception as e:
                                logger.warning(f"Box-Cox transformation failed for {col}: {str(e)}")
                        
                        elif method == "log":
                            try:
                                # Check if all values are non-negative
                                if data[col].min() < 0:
                                    # Add a constant to make all values positive
                                    shift = abs(data[col].min()) + 1.0
                                    data[f"{col}_shifted"] = data[col] + shift
                                    col = f"{col}_shifted"
                                    
                                # Apply log1p transformation (handles zeros gracefully)
                                data[f"{col.replace('_shifted', '')}_transformed"] = data[col].swifter.progress_bar(False).apply(np.log1p)
                                transformation_results[col.replace("_shifted", "")] = "log"
                            except Exception as e:
                                logger.warning(f"Log transformation failed for {col}: {str(e)}")
                        
                        elif method == "sqrt":
                            try:
                                # Check if all values are non-negative
                                if data[col].min() < 0:
                                    # Add a constant to make all values positive
                                    shift = abs(data[col].min()) + 1.0
                                    data[f"{col}_shifted"] = data[col] + shift
                                    col = f"{col}_shifted"
                                    
                                # Apply sqrt transformation
                                data[f"{col.replace('_shifted', '')}_transformed"] = data[col].swifter.progress_bar(False).apply(np.sqrt)
                                transformation_results[col.replace("_shifted", "")] = "sqrt"
                            except Exception as e:
                                logger.warning(f"Square root transformation failed for {col}: {str(e)}")
                
                # Log the results
                logger.info(f"Transformed {len(transformation_results)} skewed features using best methods")
                
                # Analyze if transformations improved the skewness
                transformed_cols = [f"{col}_transformed" for col in transformation_results.keys() 
                                 if f"{col}_transformed" in data.columns]
                
                if transformed_cols:
                    # Calculate skewness for original and transformed columns
                    orig_skew = data[skewed_features].swifter.progress_bar(False).apply(lambda x: abs(stats.skew(x.dropna())))
                    trans_skew = data[transformed_cols].swifter.progress_bar(False).apply(lambda x: abs(stats.skew(x.dropna())))
                    
                    # Compare and log the improvement
                    improvement = pd.DataFrame({
                        'original_skew': orig_skew.values,
                        'transformed_skew': trans_skew.values,
                        'improvement_pct': (orig_skew.values - trans_skew.values) / orig_skew.values * 100
                    }, index=[col.replace('_transformed', '') for col in transformed_cols])
                    
                    logger.info(f"Skewness transformation results:\n{improvement.to_string()}")
                    
                    # Replace original columns with transformed ones if significantly improved
                    improved_cols = improvement[improvement['improvement_pct'] > 20].index.tolist()
                    for col in improved_cols:
                        data[col] = data[f"{col}_transformed"]
                        logger.info(f"Replaced {col} with its transformed version (improvement: {improvement.loc[col, 'improvement_pct']:.1f}%)")
                    
                    # Drop the temporary transformed columns
                    data.drop(columns=transformed_cols, inplace=True)
        else:
            logger.info("Dataset too small for reliable skewness detection, skipping transformation")
            
        skew_time = time.time() - start_time
        logger.info(f"Skewness handling completed in {skew_time:.1f}s")
        gc.collect()
        
        # NEW: Analyze feature correlations with the target (if available)
        if 'pure_premium' in data.columns or 'il_total' in data.columns:
            start_time = time.time()
            target_col = 'pure_premium' if 'pure_premium' in data.columns else 'il_total'
            logger.info(f"Analyzing feature correlations with target: {target_col}")
            
            # Get correlations
            correlations = analyze_correlations_with_swifter(data, target_col=target_col, threshold=0.3)
            
            # Log top correlated features
            if not correlations.empty:
                top_n = min(10, len(correlations))
                logger.info(f"Top {top_n} correlated features with {target_col}:\n{correlations.head(top_n).to_string()}")
                
                # Save correlations to temp file
                corr_path = "/tmp/feature_correlations.json"
                correlations.to_json(corr_path)
                logger.info(f"Feature correlations saved to {corr_path}")
            
            corr_time = time.time() - start_time
            logger.info(f"Correlation analysis completed in {corr_time:.1f}s")
            gc.collect()
            
        start_time = time.time()
        logger.info("Encoding categorical variables")
        data = encode_categorical_variables(data)
        encoding_time = time.time() - start_time
        logger.info(f"Categorical encoding completed in {encoding_time:.1f}s")
        gc.collect()
        
        start_time = time.time()
        logger.info("Scaling numeric features")
        # Use swifter for parallelized column operations if dataset is large
        if len(data) > 100000:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                min_val = data[col].min()
                range_val = data[col].max() - min_val
                # Avoid division by zero
                if range_val > 0:
                    data[col] = data[col].swifter.progress_bar(False).apply(lambda x: (x - min_val) / range_val)
                else:
                    data[col] = 0
        else:
            data = scale_numeric_features(data)
            
        scaling_time = time.time() - start_time
        logger.info(f"Feature scaling completed in {scaling_time:.1f}s")
        gc.collect()
        
        # Calculate pure_premium as required by schema validation
        start_time = time.time()
        logger.info("Calculating pure_premium and sample weights")
        if 'il_total' in data.columns and 'eey' in data.columns:
            # Create target variable (Pure Premium)
            data['pure_premium'] = data['il_total'] / data['eey']
            # Create sample weight variable based on earned exposure years
            data['wt'] = data['eey']
            logger.info("Successfully added pure_premium and wt columns")
            
            # NEW: Calculate feature importance if target is available
            logger.info("Calculating feature importance")
            importance = calculate_feature_importance_parallel(
                data, 
                target_col='pure_premium', 
                method='correlation', 
                n_features=20
            )
            
            # Log top important features
            if not importance.empty:
                logger.info(f"Top important features for pure_premium:\n{importance.to_string()}")
                
                # Save importance to temp file
                imp_path = "/tmp/feature_importance.json"
                importance.to_json(imp_path)
                logger.info(f"Feature importance saved to {imp_path}")
        else:
            missing_cols = []
            if 'il_total' not in data.columns:
                missing_cols.append('il_total')
            if 'eey' not in data.columns:
                missing_cols.append('eey')
            error_msg = f"Cannot calculate pure_premium: missing columns {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        calc_time = time.time() - start_time
        logger.info(f"Pure premium calculation completed in {calc_time:.1f}s")
        gc.collect()
        
        # Final quality check after preprocessing
        logger.info("Performing final quality check")
        final_quality_report = quality_monitor.run_quality_checks(data)
        if final_quality_report['status'] == 'fail':
            send_message(
                channel="#alerts",
                title="⚠️ Post-Preprocessing Quality Issues",
                details=f"Quality report: {final_quality_report}",
                urgency="medium"
            )
        
        # Generate final profile
        logger.info("Generating final data profile")
        final_profile = profile_dataframe_with_swifter(data, sample_size=min(50000, len(data)))
        
        # Save final profile to a temporary JSON file for reference
        final_profile_path = "/tmp/final_profile.json"
        with open(final_profile_path, 'w') as f:
            json.dump(final_profile, f, indent=2, default=str)
        logger.info(f"Final profile saved to {final_profile_path}")
        
        # Save preprocessed data using advanced PyArrow features
        start_time = time.time()
        logger.info(f"Saving preprocessed data to {output_path} with ZSTD compression")
        
        # Convert pandas to pyarrow Table with optimized settings
        arrow_table = pa.Table.from_pandas(data)
        
        # Write with optimized settings
        pq.write_table(
            arrow_table, 
            output_path,
            compression='zstd',  # Using ZSTD for better compression ratio and speed
            compression_level=3,  # Balanced compression level for speed/size
            use_dictionary=True,
            write_statistics=True,
            row_group_size=100000  # Optimize for reading performance
        )
        
        save_time = time.time() - start_time
        logger.info(f"Data saved to parquet in {save_time:.1f}s")
        
        # Save quality reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quality_reports = {
            'initial': {'status': quality_report['status']},
            'final': {'status': final_quality_report['status']},
            'timestamp': timestamp,
            'performance': {
                'missing_handling_time': missing_time,
                'outlier_handling_time': outlier_time,
                'normality_handling_time': normality_time,
                'skewness_handling_time': skew_time,
                'categorical_encoding_time': encoding_time,
                'feature_scaling_time': scaling_time,
                'pure_premium_calc_time': calc_time,
                'save_time': save_time
            }
        }
        
        # Free memory before returning
        del data, arrow_table
        gc.collect()
        
        return output_path
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in preprocess_data: {str(e)}")
        from utils.slack import post as send_message
        send_message(
            channel="#alerts",
            title="❌ Preprocessing Error",
            details=str(e),
            urgency="high"
        )
        raise

def handle_outliers(data, threshold=3):
    """
    Handle outliers using z-score method with memory optimization.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Calculate z-scores efficiently without creating a new large array
        mean_val = data[col].mean()
        std_val = data[col].std()
        z_scores = np.abs((data[col] - mean_val) / std_val)
        mask = z_scores > threshold
        
        # Get the data type of the column
        dtype = data[col].dtype
        
        # Explicitly cast the mean value to the column's data type to avoid dtype warnings
        if np.issubdtype(dtype, np.integer):
            mean_val = int(mean_val)
        
        data.loc[mask, col] = mean_val
        
        # Clean up intermediates to free memory
        del z_scores, mask
        
    return data

def encode_categorical_variables(data):
    """
    Encode categorical variables efficiently.
    """
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        data[col] = pd.Categorical(data[col]).codes
    
    return data

def scale_numeric_features(data):
    """
    Scale numeric features using min-max scaling with memory optimization.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        min_val = data[col].min()
        range_val = data[col].max() - min_val
        # Avoid division by zero
        if range_val > 0:
            data[col] = (data[col] - min_val) / range_val
        else:
            data[col] = 0
    
    return data

def detect_and_transform_skewed_features(df, threshold=0.5):
    """
    Detect and transform skewed numeric features using swifter.
    
    Args:
        df: DataFrame to analyze
        threshold: Absolute skewness threshold to identify skewed features
        
    Returns:
        Tuple of (transformed dataframe, list of skewed features)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate skewness using swifter for parallelization
    logger.info(f"Calculating skewness for {len(numeric_cols)} numeric columns with swifter")
    skew_values = df[numeric_cols].swifter.progress_bar(False).apply(lambda x: stats.skew(x.dropna()))
    
    # Identify skewed columns
    skewed_features = skew_values[abs(skew_values) > threshold].index.tolist()
    
    # Log skewed features
    logger.info(f"Detected {len(skewed_features)} skewed features with threshold {threshold}")
    
    # Apply Yeo-Johnson transformation to skewed features
    if len(skewed_features) > 0:
        logger.info(f"Applying Yeo-Johnson transformation to {len(skewed_features)} skewed features")
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        
        # For large dataframes, process in batches
        if len(df) > 100000:
            batch_size = 100000
            num_batches = len(df) // batch_size + 1
            
            for col in skewed_features:
                # Process column in batches to save memory
                result = np.zeros(len(df))
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(df))
                    batch = df[col].iloc[start_idx:end_idx].values.reshape(-1, 1)
                    # Handle potential NaNs
                    mask = ~np.isnan(batch).flatten()
                    if mask.any():
                        transform = pt.fit(batch[mask])
                        batch_result = np.array(batch, copy=True).flatten()
                        batch_result[mask] = transform.transform(batch[mask]).flatten()
                        result[start_idx:end_idx] = batch_result
                    else:
                        result[start_idx:end_idx] = batch.flatten()
                
                # Create a new column for the transformed values
                df[f"{col}_transformed"] = result
                
                # Clear memory
                del result
                gc.collect()
        else:
            # For smaller dataframes, transform each column directly
            for col in skewed_features:
                # Handle column in isolation to preserve memory
                values = df[[col]].values
                mask = ~np.isnan(values).flatten()
                if mask.any():
                    transform = pt.fit(values[mask].reshape(-1, 1))
                    result = np.array(values, copy=True).flatten()
                    result[mask] = transform.transform(values[mask].reshape(-1, 1)).flatten()
                    df[f"{col}_transformed"] = result
    
    return df, skewed_features

def profile_dataframe_with_swifter(df, sample_size=10000):
    """
    Profile dataframe statistics with swifter for performance.
    
    Args:
        df: DataFrame to profile
        sample_size: Size of sample to use for profiling
        
    Returns:
        Dictionary with profile information
    """
    logger.info(f"Profiling dataframe with shape {df.shape} using swifter")
    
    # Sample if dataframe is large
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
        logger.info(f"Using sample of {sample_size} rows for profiling")
    else:
        df_sample = df
    
    # Basic profile info
    profile = {
        'timestamp': datetime.now().isoformat(),
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'column_types': df.dtypes.value_counts().to_dict()
    }
    
    # Missing values - use swifter for parallel processing
    logger.info("Calculating missing value percentages")
    profile['missing_percentages'] = df.isnull().mean().swifter.progress_bar(False).apply(
        lambda x: round(x*100, 2)
    ).to_dict()
    
    # Unique values count
    logger.info("Calculating unique value counts")
    profile['unique_values'] = df_sample.swifter.progress_bar(False).apply(
        lambda x: int(x.nunique())
    ).to_dict()
    
    # Add skewness for numeric columns
    logger.info("Calculating skewness for numeric columns")
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        profile['skewness'] = df_sample[numeric_cols].swifter.progress_bar(False).apply(
            lambda x: float(stats.skew(x.dropna()))
        ).to_dict()
    
    # Add top values for categorical columns
    logger.info("Identifying top values for categorical columns")
    cat_cols = df_sample.select_dtypes(include=['object', 'category']).columns
    top_values = {}
    for col in cat_cols:
        try:
            value_counts = df_sample[col].value_counts(dropna=False)
            if len(value_counts) > 0:
                top_values[col] = value_counts.index[0]
        except:
            pass
    profile['top_values'] = top_values
    
    # Add basic statistics for numeric columns
    logger.info("Calculating statistics for numeric columns")
    if len(numeric_cols) > 0:
        profile['numeric_stats'] = {}
        for col in numeric_cols:
            try:
                profile['numeric_stats'][col] = {
                    'mean': float(df_sample[col].mean()),
                    'std': float(df_sample[col].std()),
                    'min': float(df_sample[col].min()),
                    'max': float(df_sample[col].max()),
                    'median': float(df_sample[col].median())
                }
            except:
                pass
    
    logger.info("Profiling complete")
    return profile

def analyze_correlations_with_swifter(df, target_col=None, threshold=0.7):
    """
    Efficiently analyze correlations using swifter for parallel processing.
    
    Args:
        df: DataFrame to analyze
        target_col: Optional target column to calculate correlations against
        threshold: Correlation threshold for filtering high correlations
        
    Returns:
        DataFrame or Series of correlations, depending on target_col parameter
    """
    logger.info(f"Analyzing correlations with swifter {'for target: '+target_col if target_col else ''}")
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Limit columns if there are too many to avoid memory issues
    if len(numeric_cols) > 30:
        logger.info(f"Limiting correlation analysis to 30 columns out of {len(numeric_cols)}")
        numeric_cols = list(numeric_cols)[:30]
    
    if target_col and target_col in df.columns:
        # For large datasets, calculate correlation in batches
        if len(df) > 100000:
            logger.info("Using batch processing for correlation calculation")
            correlations = {}
            
            # Process in batches of columns
            batch_size = 5  # Number of columns per batch
            column_batches = [numeric_cols[i:i+batch_size] for i in range(0, len(numeric_cols), batch_size)]
            
            for batch in column_batches:
                # Calculate correlation for each column in the batch with the target
                for col in batch:
                    if col != target_col:
                        # Calculate correlation efficiently
                        valid_mask = ~(df[col].isnull() | df[target_col].isnull())
                        if valid_mask.sum() > 1:  # Need at least 2 valid data points
                            correlations[col] = float(df.loc[valid_mask, col].corr(df.loc[valid_mask, target_col]))
                        else:
                            correlations[col] = float('nan')
                
                # Free memory after each batch
                gc.collect()
            
            # Convert to Series
            correlations = pd.Series(correlations)
        else:
            # For smaller datasets, use swifter for parallel processing
            correlations = df[numeric_cols].swifter.progress_bar(False).apply(
                lambda x: x.corr(df[target_col]) if x.name != target_col else 0
            )
        
        # Filter by threshold
        high_corr = correlations[abs(correlations) > threshold]
        logger.info(f"Found {len(high_corr)} columns with correlation > {threshold} with {target_col}")
        return high_corr.sort_values(ascending=False)
    else:
        # If no target column specified, or it doesn't exist, calculate full correlation matrix
        if len(df) > 100000:
            logger.info("Dataset too large for full correlation matrix. Using sampling.")
            sample_size = min(100000, len(df) // 10)
            df_sample = df.sample(sample_size, random_state=42)
            return df_sample[numeric_cols].corr()
        else:
            return df[numeric_cols].corr()

def calculate_feature_importance_parallel(df, target_col, method='correlation', n_features=None):
    """
    Calculate feature importance using swifter for parallel processing.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
        method: Method to use ('correlation' or 'mutual_info')
        n_features: Number of top features to return (None for all)
        
    Returns:
        Series with feature importance scores
    """
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in dataframe")
        return pd.Series()
    
    # Get numeric columns excluding the target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) == 0:
        logger.warning("No numeric feature columns found for importance calculation")
        return pd.Series()
    
    logger.info(f"Calculating feature importance using {method} method for {len(numeric_cols)} features")
    
    if method == 'correlation':
        # Use the correlation function we already defined
        importance = abs(analyze_correlations_with_swifter(df, target_col, threshold=0))
    
    elif method == 'mutual_info':
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # For large datasets, process in batches
            if len(df) > 100000:
                logger.info("Using batch processing for mutual information calculation")
                
                # Sample the data for efficiency
                sample_size = min(100000, len(df) // 2)
                df_sample = df.sample(sample_size, random_state=42)
                
                # Calculate mutual information for chunks of features
                def calc_mi_for_chunk(chunk_cols):
                    # Drop rows with NaN in target or features
                    valid_mask = ~df_sample[target_col].isnull()
                    for col in chunk_cols:
                        valid_mask = valid_mask & ~df_sample[col].isnull()
                    
                    if valid_mask.sum() > 10:  # Need enough valid data points
                        X = df_sample.loc[valid_mask, chunk_cols]
                        y = df_sample.loc[valid_mask, target_col]
                        mi_values = mutual_info_regression(X, y)
                        return pd.Series(mi_values, index=chunk_cols)
                    else:
                        return pd.Series(0, index=chunk_cols)
                
                # Split features into chunks for parallel processing
                n_cores = max(1, mp.cpu_count() - 1)
                chunk_size = max(1, len(numeric_cols) // n_cores)
                feature_chunks = [numeric_cols[i:i+chunk_size] for i in range(0, len(numeric_cols), chunk_size)]
                
                # Process chunks and combine results
                chunk_results = []
                for chunk in feature_chunks:
                    chunk_results.append(calc_mi_for_chunk(chunk))
                
                importance = pd.concat(chunk_results)
            else:
                # For smaller datasets, process all at once
                # Drop rows with NaN in target
                valid_mask = ~df[target_col].isnull()
                for col in numeric_cols:
                    valid_mask = valid_mask & ~df[col].isnull()
                
                if valid_mask.sum() > 10:  # Need enough valid data points
                    X = df.loc[valid_mask, numeric_cols]
                    y = df.loc[valid_mask, target_col]
                    mi_values = mutual_info_regression(X, y)
                    importance = pd.Series(mi_values, index=numeric_cols)
                else:
                    logger.warning("Not enough valid data points for mutual information calculation")
                    importance = pd.Series(0, index=numeric_cols)
        except Exception as e:
            logger.error(f"Error calculating mutual information: {str(e)}")
            importance = pd.Series(0, index=numeric_cols)
    
    else:
        logger.error(f"Unknown feature importance method: {method}")
        return pd.Series()
    
    # Sort by importance
    importance = importance.sort_values(ascending=False)
    
    # Limit to top N features if requested
    if n_features is not None and n_features < len(importance):
        importance = importance.head(n_features)
        logger.info(f"Returning top {n_features} features by importance")
    
    return importance

def test_normality_with_swifter(df, columns=None, test_method='shapiro', sample_size=1000):
    """
    Perform normality tests on numeric columns using swifter for parallel processing.
    
    Args:
        df: DataFrame with data to test
        columns: Specific columns to test (None for all numeric)
        test_method: Test to use ('shapiro', 'ks', 'normaltest', or 'anderson')
        sample_size: Sample size to use for testing (for large datasets)
        
    Returns:
        DataFrame with test results including p-values and test statistics
    """
    logger.info(f"Testing normality using {test_method} test")
    
    # Get the columns to test
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(columns, str):
        columns = [columns]
    
    # Convert to list if it's an Index
    if not isinstance(columns, list):
        columns = list(columns)
    
    # Sample data if it's very large
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
        logger.info(f"Using sample of {sample_size} rows for normality testing")
    else:
        df_sample = df
    
    # Initialize results dictionary
    results = []
    
    # Process in batches to avoid memory issues
    batch_size = min(10, len(columns))  # Process at most 10 columns at a time
    column_batches = [columns[i:i+batch_size] for i in range(0, len(columns), batch_size)]
    
    for batch in column_batches:
        batch_results = []
        
        for col in batch:
            # Skip columns with too few unique values
            if df_sample[col].nunique() < 5:
                logger.info(f"Skipping {col} - too few unique values for normality testing")
                continue
                
            # Get non-null values
            values = df_sample[col].dropna().values
            
            # Skip if too few values
            if len(values) < 8:  # Shapiro needs at least 3, but we use a higher threshold
                logger.info(f"Skipping {col} - too few non-null values for normality testing")
                continue
            
            # Run the specified test
            try:
                if test_method == 'shapiro':
                    # Shapiro-Wilk test
                    stat, p_value = stats.shapiro(values)
                    is_normal = p_value > 0.05
                    batch_results.append({
                        'column': col,
                        'test': 'Shapiro-Wilk',
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': is_normal
                    })
                
                elif test_method == 'normaltest':
                    # D'Agostino's K^2 test
                    stat, p_value = stats.normaltest(values)
                    is_normal = p_value > 0.05
                    batch_results.append({
                        'column': col,
                        'test': "D'Agostino's K^2",
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': is_normal
                    })
                
                elif test_method == 'ks':
                    # Kolmogorov-Smirnov test
                    # Compare with normal distribution with same mean and std
                    mean, std = np.mean(values), np.std(values)
                    if std > 0:
                        stat, p_value = stats.kstest(
                            values, 
                            'norm', 
                            args=(mean, std)
                        )
                        is_normal = p_value > 0.05
                        batch_results.append({
                            'column': col,
                            'test': 'Kolmogorov-Smirnov',
                            'statistic': stat,
                            'p_value': p_value,
                            'is_normal': is_normal
                        })
                
                elif test_method == 'anderson':
                    # Anderson-Darling test
                    result = stats.anderson(values, dist='norm')
                    # No direct p-value, use critical value at 5% significance
                    stat = result.statistic
                    critical_value = result.critical_values[2]  # 5% significance level
                    is_normal = stat < critical_value
                    batch_results.append({
                        'column': col,
                        'test': 'Anderson-Darling',
                        'statistic': stat,
                        'critical_value': critical_value,
                        'significance': 5.0,  # 5%
                        'is_normal': is_normal
                    })
                    
                else:
                    logger.warning(f"Unknown normality test method: {test_method}")
                    
            except Exception as e:
                logger.warning(f"Error testing normality for column {col}: {str(e)}")
        
        # Add batch results to overall results
        results.extend(batch_results)
        gc.collect()  # Free memory after each batch
    
    # Convert to DataFrame
    if results:
        result_df = pd.DataFrame(results)
        
        # Log summary
        normal_cols = result_df[result_df['is_normal']]['column'].tolist()
        non_normal_cols = result_df[~result_df['is_normal']]['column'].tolist()
        
        logger.info(f"Normality test results ({test_method}):")
        logger.info(f"  - Normal distributions: {len(normal_cols)} columns")
        logger.info(f"  - Non-normal distributions: {len(non_normal_cols)} columns")
        
        return result_df
    else:
        logger.warning("No columns were successfully tested for normality")
        return pd.DataFrame()

def select_best_transformation(data, column, sample_size=5000):
    """
    Select the best transformation method for a skewed column.
    
    Args:
        data: DataFrame containing the column
        column: Name of the column to transform
        sample_size: Sample size to use for testing
        
    Returns:
        Tuple of (method_name, transformed_values)
    """
    # Sample data if it's very large
    if len(data) > sample_size:
        values = data[column].sample(sample_size, random_state=42).dropna().values
    else:
        values = data[column].dropna().values
    
    if len(values) < 10:
        return None, None
    
    # Get original skewness and check if transformation is needed
    original_skew = stats.skew(values)
    logger.info(f"Column {column} has skewness: {original_skew:.4f}")
    
    # If abs skewness < 0.5, it's already approximately normal
    if abs(original_skew) < 0.5:
        return "none", None
    
    # Try different transformations and evaluate results
    transformations = {}
    
    # 1. Yeo-Johnson (works with negative values)
    try:
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        yj_values = pt.fit_transform(values.reshape(-1, 1)).flatten()
        yj_skew = stats.skew(yj_values)
        transformations['yeo-johnson'] = {
            'skew': yj_skew,
            'values': yj_values,
            'improvement': abs(original_skew) - abs(yj_skew)
        }
    except Exception as e:
        logger.warning(f"Yeo-Johnson transformation failed for {column}: {str(e)}")
    
    # 2. Box-Cox (only for strictly positive data)
    if np.all(values > 0):
        try:
            pt = PowerTransformer(method='box-cox', standardize=False)
            bc_values = pt.fit_transform(values.reshape(-1, 1)).flatten()
            bc_skew = stats.skew(bc_values)
            transformations['box-cox'] = {
                'skew': bc_skew,
                'values': bc_values,
                'improvement': abs(original_skew) - abs(bc_skew)
            }
        except Exception as e:
            logger.warning(f"Box-Cox transformation failed for {column}: {str(e)}")
    
    # 3. Log transformation (only for strictly positive data)
    if np.all(values > 0):
        try:
            log_values = np.log1p(values)  # log(1+x) to handle values close to 0
            log_skew = stats.skew(log_values)
            transformations['log'] = {
                'skew': log_skew,
                'values': log_values,
                'improvement': abs(original_skew) - abs(log_skew)
            }
        except Exception as e:
            logger.warning(f"Log transformation failed for {column}: {str(e)}")
    
    # 4. Square root (only for non-negative data)
    if np.all(values >= 0):
        try:
            sqrt_values = np.sqrt(values)
            sqrt_skew = stats.skew(sqrt_values)
            transformations['sqrt'] = {
                'skew': sqrt_skew,
                'values': sqrt_values,
                'improvement': abs(original_skew) - abs(sqrt_skew)
            }
        except Exception as e:
            logger.warning(f"Square root transformation failed for {column}: {str(e)}")
    
    # Find the transformation with the best improvement
    if transformations:
        best_method = max(transformations.items(), key=lambda x: x[1]['improvement'])
        best_method_name, best_info = best_method
        
        # Only consider it an improvement if skewness reduced significantly
        if best_info['improvement'] > 0.2:  # 20% improvement in skewness
            logger.info(f"Best transformation for {column}: {best_method_name} (original skew: {original_skew:.4f}, new skew: {best_info['skew']:.4f})")
            return best_method_name, best_info['values']
        else:
            logger.info(f"No significant improvement found for {column} - best was {best_method_name} with {best_info['improvement']:.4f} reduction")
            return "none", None
    else:
        logger.warning(f"No successful transformations found for {column}")
        return "none", None
