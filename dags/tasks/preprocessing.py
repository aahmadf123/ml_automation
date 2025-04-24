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
"""

import os
import json
import logging
import time
from typing import Dict

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
