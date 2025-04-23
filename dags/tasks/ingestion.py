"""
Data ingestion task for ML Automation.

This module handles the ingestion of raw data from S3, including:
- Checking file existence in S3
- Downloading and caching data
- Converting data formats
- Error handling and notifications
"""

import logging
import os
from typing import Optional, Dict, Any

# Setup logging
log = logging.getLogger(__name__)

# Constants
S3_DATA_FOLDER = "raw-data"
LOCAL_CSV_PATH = "/tmp/homeowner_data.csv"
LOCAL_PARQUET_PATH = "/tmp/homeowner_data.parquet"

def check_s3_file_exists(bucket: str, key: str) -> bool:
    """
    Check if a file exists in S3.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        bool: True if file exists, False otherwise
        
    Raises:
        ClientError: If there's an error accessing S3
    """
    import boto3
    from botocore.exceptions import ClientError
    
    s3_client = boto3.client("s3")
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        log.error(f"Error checking S3 file: {str(e)}")
        raise

def ingest_data_from_s3() -> str:
    """
    Download raw CSV file from S3, convert to Parquet, and return the path.
    
    This function:
    1. Checks if the file exists in S3
    2. Downloads and caches the data if needed
    3. Converts CSV to Parquet format
    4. Handles errors and sends notifications
    
    Returns:
        str: Path to the local Parquet file
        
    Raises:
        FileNotFoundError: If the file doesn't exist in S3
        Exception: For other errors during ingestion
    """
    from utils.config import DATA_BUCKET
    from utils.storage import download as s3_download
    from .cache import is_cache_valid, update_cache
    
    s3_key = f"{S3_DATA_FOLDER}/ut_loss_history_1.csv"
    s3_bucket = DATA_BUCKET
    
    # Check if file exists in S3
    if not check_s3_file_exists(s3_bucket, s3_key):
        error_msg = f"Data file not found in S3: s3://{s3_bucket}/{s3_key}"
        log.error(error_msg)
        # Import slack only when needed
        from utils.slack import post as send_message
        send_message(
            channel="#alerts",
            title="❌ Data Ingestion Failed",
            details=error_msg,
            urgency="high"
        )
        raise FileNotFoundError(error_msg)
    
    try:
        # Download and cache data if needed
        if not is_cache_valid(s3_bucket, s3_key, LOCAL_CSV_PATH):
            log.info(f"Downloading and caching: {s3_key}")
            update_cache(s3_bucket, s3_key, LOCAL_CSV_PATH)
        else:
            log.info(f"Cache hit. Skipping download for: {s3_key}")
        
        # Convert to Parquet
        import pandas as pd
        try:
            df = pd.read_csv(LOCAL_CSV_PATH)
            df.to_parquet(LOCAL_PARQUET_PATH, index=False)
            log.info(f"Converted to Parquet: {LOCAL_PARQUET_PATH}")
            
            # Verify the Parquet file was created
            if not os.path.exists(LOCAL_PARQUET_PATH):
                raise FileNotFoundError(f"Parquet file not created: {LOCAL_PARQUET_PATH}")
                
            return LOCAL_PARQUET_PATH
            
        except Exception as e:
            error_msg = f"Error converting to Parquet: {str(e)}"
            log.error(error_msg)
            # Import slack only when needed
            from utils.slack import post as send_message
            send_message(
                channel="#alerts",
                title="❌ Data Conversion Failed",
                details=error_msg,
                urgency="high"
            )
            raise
            
    except Exception as e:
        error_msg = f"Error ingesting data: {str(e)}"
        log.error(error_msg)
        # Import slack only when needed
        from utils.slack import post as send_message
        send_message(
            channel="#alerts",
            title="❌ Data Ingestion Failed",
            details=error_msg,
            urgency="high"
        )
        raise
