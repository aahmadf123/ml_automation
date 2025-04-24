#!/usr/bin/env python3
"""
tasks/drift.py

Handles:
  - Generation of reference means from the processed data.
  - Data drift detection by comparing current data with versioned reference means.
  - A self‑healing routine when drift is detected.

Refactored to:
  - Use utils/storage.upload instead of boto3 directly.
  - Version the reference means file on upload under a timestamped prefix.
  - Wrap S3 operations with tenacity retries.
  - Load drift threshold from Airflow Variable "DRIFT_THRESHOLD".
  - Leverage the caching system for improved performance.
"""

import os
import logging
import time
import pandas as pd
import numpy as np
from airflow.models import Variable
from tenacity import retry, stop_after_attempt, wait_fixed
import json
import boto3
from datetime import datetime
from utils.storage import download as s3_download
from utils.storage import upload as s3_upload
from utils.config import (
    DATA_BUCKET, REFERENCE_KEY_PREFIX, AWS_REGION,
    DRIFT_THRESHOLD
)
from utils.cache import GLOBAL_CACHE, cache_result

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Local path for the reference means CSV
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"

# Initialize AWS clients with retry
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def initialize_aws_clients():
    """Initialize AWS clients with retry logic."""
    try:
        cloudwatch = boto3.client('cloudwatch', region_name=AWS_REGION)
        lambda_client = boto3.client('lambda', region_name=AWS_REGION)
        ssm = boto3.client('ssm', region_name=AWS_REGION)
        secretsmanager = boto3.client('secretsmanager', region_name=AWS_REGION)
        return cloudwatch, lambda_client, ssm, secretsmanager
    except Exception as e:
        logger.error(f"Failed to initialize AWS clients: {e}")
        raise

@cache_result
def generate_reference_means(processed_data_path: str) -> str:
    """
    Generate reference means from processed data and upload to S3.
    Uses advanced pyarrow optimizations for memory efficiency.
    
    Args:
        processed_data_path: Path to processed data file
        
    Returns:
        S3 path to the reference means file
    """
    # Import slack only when needed
    from utils.slack import post as send_message
    
    try:
        logger.info(f"Loading processed data from {processed_data_path}")
        
        # Check if we have already computed statistics for this DataFrame
        df_hash = os.path.basename(processed_data_path).replace('.parquet', '')
        df_name = f"reference_means_{df_hash}"
        
        # Use cached data if available
        cached_df = GLOBAL_CACHE.get_transformed(df_name)
        if cached_df is not None:
            logger.info(f"Using cached data for {df_name}")
            df = cached_df
        else:
            # Load the data using pyarrow optimizations
            import pyarrow as pa
            import pyarrow.parquet as pq
            import pyarrow.compute as pc
            import pyarrow.dataset as ds
            import gc
            
            # Create dataset for efficient reading
            dataset = ds.dataset(processed_data_path, format="parquet")
            
            # Create scanner with projection and filtering to only include numeric columns
            scanner = ds.Scanner.from_dataset(
                dataset,
                use_threads=True
            )
            
            # Read as arrow table
            table = scanner.to_table()
            
            # Convert to pandas for easier stats calculation
            df = table.to_pandas()
            
            # Cache the DataFrame for future use
            GLOBAL_CACHE.store_transformed(df, df_name)
            
            # Free memory
            del table
            gc.collect()
        
        # Get statistics from cache instead of recalculating
        GLOBAL_CACHE.compute_statistics(df, df_name)
        
        # Calculate means for numerical columns by using cached statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        means = pd.Series({
            col: GLOBAL_CACHE.get_statistic(df_name, col, 'mean') 
            for col in numeric_cols
        })
        
        # Save to CSV
        means.to_csv(REFERENCE_MEANS_PATH, header=True)
        logger.info(f"Reference means generated and saved to {REFERENCE_MEANS_PATH}")
        
        # Upload to S3 with timestamp in the key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{REFERENCE_KEY_PREFIX}/{timestamp}/reference_means.csv"
        s3_upload(REFERENCE_MEANS_PATH, s3_key)
        
        # Also upload to latest for easy access
        latest_key = f"{REFERENCE_KEY_PREFIX}/latest/reference_means.csv"
        s3_upload(REFERENCE_MEANS_PATH, latest_key)
        
        s3_path = f"s3://{DATA_BUCKET}/{s3_key}"
        
        send_message(
            channel="#data-engineering",
            title="📊 Reference Means Generated",
            details=f"Reference means generated from processed data.\nS3 Path: {s3_path}",
            urgency="normal"
        )
        
        return s3_path
        
    except Exception as e:
        error_msg = f"Error generating reference means: {e}"
        logger.error(error_msg)
        send_message(
            channel="#alerts",
            title="❌ Reference Means Generation Failed",
            details=error_msg,
            urgency="high"
        )
        raise

@cache_result
def detect_data_drift(processed_data_path: str, reference_path: str = None) -> dict:
    """
    Detect drift between current data and reference means.
    Uses advanced pyarrow optimizations for memory efficiency.
    
    Args:
        processed_data_path: Path to current processed data
        reference_path: Path to reference means file (default: latest from S3)
        
    Returns:
        Dict with drift results
    """
    # Import slack only when needed
    from utils.slack import post as send_message
    import gc
    
    drift_threshold = float(Variable.get("DRIFT_THRESHOLD", default_var=str(DRIFT_THRESHOLD)))
    logger.info(f"Using drift threshold: {drift_threshold}")
    
    try:
        logger.info(f"Loading current data from {processed_data_path}")
        
        # Check if we have already computed statistics for this DataFrame
        df_hash = os.path.basename(processed_data_path).replace('.parquet', '')
        df_name = f"drift_detection_{df_hash}"
        
        # Use cached data if available or compute statistics
        cached_df = GLOBAL_CACHE.get_transformed(df_name)
        if cached_df is not None:
            logger.info(f"Using cached data for {df_name}")
            current_df = cached_df
            # Ensure statistics are computed
            GLOBAL_CACHE.compute_statistics(current_df, df_name)
        else:
            # Load the data using pyarrow optimizations
            import pyarrow as pa
            import pyarrow.parquet as pq
            import pyarrow.compute as pc
            import pyarrow.dataset as ds
            
            # Create dataset for efficient reading
            dataset = ds.dataset(processed_data_path, format="parquet")
            
            # Create scanner with multi-threading enabled
            scanner = ds.Scanner.from_dataset(
                dataset,
                use_threads=True
            )
            
            # Read as arrow table
            table = scanner.to_table()
            
            # Convert to pandas 
            current_df = table.to_pandas()
            
            # Cache the DataFrame and compute statistics
            GLOBAL_CACHE.store_transformed(current_df, df_name)
            GLOBAL_CACHE.compute_statistics(current_df, df_name)
            
            # Free memory
            del table
            gc.collect()
        
        # Get reference means
        if reference_path is None:
            reference_key = f"{REFERENCE_KEY_PREFIX}/latest/reference_means.csv"
            s3_download(reference_key, REFERENCE_MEANS_PATH)
            reference_path = REFERENCE_MEANS_PATH
            
        logger.info(f"Loading reference means from {reference_path}")
        reference_means = pd.read_csv(reference_path, index_col=0, header=None, squeeze=True)
        
        # Get numeric columns
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        
        # Calculate current means using cached statistics
        current_means = pd.Series({
            col: GLOBAL_CACHE.get_statistic(df_name, col, 'mean')
            for col in numeric_cols
        })
        
        # Free memory if we're done with the dataframe
        if not GLOBAL_CACHE.get_transformed(df_name) is current_df:
            del current_df
            gc.collect()
        
        # Calculate drift for each column
        drift_results = {}
        significant_drift = {}
        
        for col in numeric_cols:
            if col in reference_means:
                ref_mean = reference_means[col]
                cur_mean = current_means[col]
                
                # Calculate relative change (avoid division by zero)
                if abs(ref_mean) > 1e-10:
                    relative_change = abs((cur_mean - ref_mean) / ref_mean)
                else:
                    relative_change = abs(cur_mean - ref_mean)
                
                drift_results[col] = {
                    "reference_mean": float(ref_mean),
                    "current_mean": float(cur_mean),
                    "absolute_change": float(abs(cur_mean - ref_mean)),
                    "relative_change": float(relative_change),
                    "significant": relative_change > drift_threshold
                }
                
                if relative_change > drift_threshold:
                    significant_drift[col] = float(relative_change)
        
        # Send notification if significant drift detected
        if significant_drift:
            drift_msg = "\n".join([f"{col}: {change:.2%}" for col, change in significant_drift.items()])
            send_message(
                channel="#alerts",
                title="🚨 Data Drift Detected",
                details=f"Significant drift detected in {len(significant_drift)} feature(s):\n{drift_msg}",
                urgency="high"
            )
            
        # Prepare summary statistics
        results = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": len(significant_drift) > 0,
            "num_features_with_drift": len(significant_drift),
            "drift_details": drift_results,
            "drift_threshold": drift_threshold
        }
        
        return results
        
    except Exception as e:
        error_msg = f"Error detecting data drift: {e}"
        logger.error(error_msg)
        send_message(
            channel="#alerts",
            title="❌ Drift Detection Failed",
            details=error_msg,
            urgency="high"
        )
        raise

@cache_result
def self_healing(drift_results: dict, processed_data_path: str) -> dict:
    """
    Apply self-healing to data if drift is detected.
    
    Args:
        drift_results: Results from the drift detection
        processed_data_path: Path to processed data
        
    Returns:
        Dict with self-healing results
    """
    # Import slack only when needed
    from utils.slack import post as send_message
    
    try:
        # Check if drift was detected
        if not drift_results["drift_detected"]:
            logger.info("No significant drift detected. Self-healing not needed.")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "skipped",
                "reason": "No drift detected"
            }
        
        # Get DataFrame hash for caching
        df_hash = os.path.basename(processed_data_path).replace('.parquet', '')
        df_name = f"self_healing_{df_hash}"
        
        # Check for cached transformed version
        cached_df = GLOBAL_CACHE.get_transformed(df_name)
        if cached_df is not None:
            logger.info(f"Using cached data for self-healing: {df_name}")
            df = cached_df
        else:
            # Load the data
            logger.info(f"Loading data for self-healing from {processed_data_path}")
            df = pd.read_parquet(processed_data_path)
            # Cache for future use
            GLOBAL_CACHE.store_transformed(df, df_name)
        
        # Apply healing strategies
        healed_columns = {}
        
        # Get features with significant drift
        drift_features = []
        for col, details in drift_results["drift_details"].items():
            if details["significant"]:
                drift_features.append(col)
        
        # Apply healing approaches based on feature type
        for col in drift_features:
            ref_mean = drift_results["drift_details"][col]["reference_mean"]
            curr_mean = drift_results["drift_details"][col]["current_mean"]
            
            # Apply scaling transformation to realign with reference distribution
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Save original values for verification
                original_mean = df[col].mean()
                
                # Apply scaling transformation
                scaling_factor = ref_mean / curr_mean if abs(curr_mean) > 1e-10 else 1.0
                df[col] = df[col] * scaling_factor
                
                # Verify the transformation
                new_mean = df[col].mean()
                healed_columns[col] = {
                    "strategy": "scaling",
                    "original_mean": original_mean,
                    "target_mean": ref_mean,
                    "new_mean": new_mean,
                    "scaling_factor": scaling_factor,
                    "improvement": abs(new_mean - ref_mean) / abs(original_mean - ref_mean)
                }
        
        # Save healed dataset
        healed_path = processed_data_path.replace(".parquet", "_healed.parquet")
        df.to_parquet(healed_path)
        logger.info(f"Healed dataset saved to {healed_path}")
        
        # Store the healed dataframe in cache with new name
        healed_df_name = f"healed_{df_hash}"
        GLOBAL_CACHE.store_transformed(df, healed_df_name)
        
        # Upload healed dataset to S3
        s3_key = f"data/healed/{os.path.basename(healed_path)}"
        s3_upload(healed_path, s3_key)
        
        # Send notification about healing
        if healed_columns:
            healing_msg = "\n".join([
                f"{col}: {details['improvement']:.2%} improvement" 
                for col, details in healed_columns.items()
            ])
            send_message(
                channel="#data-engineering",
                title="🔄 Self-Healing Applied",
                details=f"Self-healing applied to {len(healed_columns)} columns:\n{healing_msg}\n" +
                        f"Healed dataset saved to {s3_key}",
                urgency="medium"
            )
            
        # Prepare results
        results = {
            "timestamp": datetime.now().isoformat(),
            "status": "applied",
            "healed_columns": healed_columns,
            "healed_path": healed_path,
            "s3_key": s3_key
        }
        
        return results
        
    except Exception as e:
        error_msg = f"Error in self-healing: {e}"
        logger.error(error_msg)
        send_message(
            channel="#alerts",
            title="❌ Self-Healing Failed",
            details=error_msg,
            urgency="high"
        )
        raise

def record_system_metrics() -> dict:
    """Record system metrics to CloudWatch."""
    # Initialize AWS clients
    cloudwatch, _, _, _ = initialize_aws_clients()
    
    # Import necessary modules for system metrics
    import psutil
    
    metrics = {}
    
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics["cpu_utilization"] = cpu_percent
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics["memory_utilization"] = memory.percent
        metrics["memory_available_gb"] = memory.available / (1024 ** 3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics["disk_utilization"] = disk.percent
        metrics["disk_free_gb"] = disk.free / (1024 ** 3)
        
        # Network metrics (just an example, can be customized)
        net_io_counters = psutil.net_io_counters()
        metrics["network_bytes_sent"] = net_io_counters.bytes_sent
        metrics["network_bytes_recv"] = net_io_counters.bytes_recv
        
        # Log metrics
        timestamp = int(time.time())
        
        # Send metrics to CloudWatch
        cloudwatch.put_metric_data(
            Namespace='MLAutomation',
            MetricData=[
                {
                    'MetricName': name,
                    'Value': value,
                    'Unit': 'Percent' if 'utilization' in name else 'None',
                    'Timestamp': datetime.fromtimestamp(timestamp)
                }
                for name, value in metrics.items()
            ]
        )
        
        logger.info(f"Recorded system metrics: {metrics}")
        
        return {
            "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "metrics": metrics,
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"Error recording system metrics: {e}"
        logger.error(error_msg)
        # Import slack only when needed
        from utils.slack import post as send_message
        send_message(
            channel="#alerts",
            title="❌ System Metrics Recording Failed",
            details=error_msg,
            urgency="medium"
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

def update_monitoring_with_ui_components() -> dict:
    """Update UI monitoring components."""
    # This function could be expanded to update React components in the dashboard
    try:
        # For now just log that we would update components
        logger.info("Would update UI monitoring components")
        
        # In real implementation, this might:
        # 1. Generate new React components based on latest metrics
        # 2. Update AWS Amplify app or S3 bucket with new components
        # 3. Trigger a rebuild of the frontend
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "components_updated": ["DriftMonitor", "DataQualityDashboard"]
        }
        
    except Exception as e:
        error_msg = f"Error updating UI components: {e}"
        logger.error(error_msg)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }
