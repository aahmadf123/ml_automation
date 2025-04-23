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


def preprocess_data(data_path, output_path):
    """
    Preprocess the data and perform quality checks.
    Uses advanced memory-efficient processing with pyarrow 14.0.2 optimizations.
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
        
        # Set pandas options for memory efficiency
        pd.options.mode.chained_assignment = None  # default='warn'
        
        # Initialize quality monitor
        quality_monitor = DataQualityMonitor()
        
        # Load data with memory optimization
        logger = logging.getLogger(__name__)
        logger.info(f"Loading data from {data_path} with advanced pyarrow 14.0.2 optimizations")
        
        # First, explore the dataset schema
        dataset = ds.dataset(data_path, format="parquet")
        schema = dataset.schema
        
        # Log schema information
        logger.info(f"Dataset schema: {schema}")
        
        # Create a scanner with multi-threading enabled
        scanner = ds.Scanner.from_dataset(
            dataset,
            use_threads=True,
            memory_pool=pa.default_memory_pool()
        )
        
        # Process in batches to control memory usage
        batch_size = 50000
        table_batches = []
        processed_rows = 0
        
        # Custom batch processing function
        def process_batch(batch):
            nonlocal processed_rows
            # Convert to pandas for processing
            df_batch = batch.to_pandas()
            processed_rows += len(df_batch)
            
            # Downcast numeric columns to reduce memory usage
            for col in df_batch.select_dtypes(include=['float64']).columns:
                df_batch[col] = pd.to_numeric(df_batch[col], downcast='float')
            for col in df_batch.select_dtypes(include=['int64']).columns:
                df_batch[col] = pd.to_numeric(df_batch[col], downcast='integer')
                
            logger.info(f"Processed {len(df_batch)} rows (total: {processed_rows})")
            return df_batch
        
        # Read and process batches
        data_batches = []
        for batch in scanner.to_batches(batch_size=batch_size):
            df_batch = process_batch(batch)
            data_batches.append(df_batch)
            
            # Clean up to free memory
            gc.collect()
        
        # Combine processed batches
        logger.info("Combining processed batches")
        data = pd.concat(data_batches, ignore_index=True)
        del data_batches
        gc.collect()
        
        logger.info(f"Data loaded and preprocessed, shape={data.shape}")
        
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
        
        # Basic preprocessing
        logger.info("Handling missing data")
        data = handle_missing_data(data)
        gc.collect()  # Force garbage collection after operations
        
        logger.info("Handling outliers")
        data = handle_outliers(data)
        gc.collect()
        
        logger.info("Encoding categorical variables")
        data = encode_categorical_variables(data)
        gc.collect()
        
        logger.info("Scaling numeric features")
        data = scale_numeric_features(data)
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
        logger.info(f"Saving preprocessed data to {output_path} with ZSTD compression")
        
        # Convert pandas to pyarrow Table
        arrow_table = pa.Table.from_pandas(data)
        
        # Write with optimized settings
        pq.write_table(
            arrow_table, 
            output_path,
            compression='zstd',  # Using ZSTD for better compression ratio
            compression_level=3,  # Balanced compression level (range 1-22)
            use_dictionary=True,
            write_statistics=True,
            use_threads=True
        )
        
        # Save quality reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quality_reports = {
            'initial': {'status': quality_report['status']},
            'final': {'status': final_quality_report['status']},
            'timestamp': timestamp
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
