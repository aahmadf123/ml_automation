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

def list_s3_objects(bucket: str, prefix: str) -> list:
    """
    List objects in S3 bucket with the given prefix for debugging.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 object prefix
        
    Returns:
        list: List of object keys
    """
    import boto3
    
    s3_client = boto3.client("s3")
    log.info(f"Listing objects in bucket '{bucket}' with prefix '{prefix}'")
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' in response:
            objects = []
            for obj in response['Contents']:
                log.info(f"Found: {obj['Key']} ({obj['Size']} bytes)")
                objects.append(obj['Key'])
            return objects
        else:
            log.warning(f"No objects found with prefix '{prefix}'")
            return []
    except Exception as e:
        log.error(f"Error listing S3 objects: {str(e)}")
        return []

def ingest_data_from_s3(bucket_name: Optional[str] = None, key: Optional[str] = None, local_path: Optional[str] = None) -> str:
    """
    Efficiently process large CSV file from S3 using advanced pyarrow 14.0.2 features.
    
    Args:
        bucket_name: S3 bucket name (optional, uses DATA_BUCKET from config if not provided)
        key: S3 object key (optional, uses default path if not provided)
        local_path: Local path to save the file (optional, uses default if not provided)
    
    This function:
    1. Checks if the file exists in S3
    2. Uses pyarrow-native CSV parsing for memory efficiency
    3. Employs ZSTD compression and dictionary encoding
    4. Handles errors and sends notifications
    
    Returns:
        str: Path to the local Parquet file
        
    Raises:
        FileNotFoundError: If the file doesn't exist in S3
        Exception: For other errors during ingestion
    """
    try:
        # Import dependencies
        import boto3
        import pandas as pd
        from utils.config import DATA_BUCKET, AWS_REGION
        import gc
        
        # Make sure required packages are installed
        try:
            import pyarrow as pa
            import pyarrow.csv as csv
            import pyarrow.parquet as pq
            import pyarrow.dataset as ds
            import pyarrow.compute as pc
            import smart_open
        except ImportError:
            log.warning("Required packages not installed, installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "smart_open[s3] pyarrow>=14.0.0"])
            import pyarrow as pa
            import pyarrow.csv as csv
            import pyarrow.parquet as pq
            import pyarrow.dataset as ds
            import pyarrow.compute as pc
            import smart_open
        
        # Use provided parameters or defaults
        s3_bucket = bucket_name or DATA_BUCKET
        s3_key = key or f"{S3_DATA_FOLDER}/ut_loss_history_1.csv"
        output_path = local_path or LOCAL_PARQUET_PATH
        
        s3_uri = f"s3://{s3_bucket}/{s3_key}"
        log.info(f"Ingesting data from: {s3_uri}")
        log.info(f"Output path: {output_path}")
        
        # Debug: List objects in both raw_data and raw-data folders to find the file
        raw_data_objects = list_s3_objects(s3_bucket, "raw_data/")
        raw_hyphen_objects = list_s3_objects(s3_bucket, "raw-data/")
        
        # If file not found in specified path but found in another folder, update the key
        target_filename = os.path.basename(s3_key)
        
        if not check_s3_file_exists(s3_bucket, s3_key):
            log.warning(f"File not found at {s3_uri}, searching in alternate locations")
            
            # Try to find the file in raw_data/
            if not s3_key.startswith("raw_data/"):
                for obj_key in raw_data_objects:
                    if obj_key.endswith(target_filename):
                        s3_key = obj_key
                        log.info(f"Found file in raw_data/ folder: {s3_key}")
                        break
                        
            # Try to find the file in raw-data/
            if not check_s3_file_exists(s3_bucket, s3_key) and not s3_key.startswith("raw-data/"):
                for obj_key in raw_hyphen_objects:
                    if obj_key.endswith(target_filename):
                        s3_key = obj_key
                        log.info(f"Found file in raw-data/ folder: {s3_key}")
                        break
        
        # Final check if file exists
        if not check_s3_file_exists(s3_bucket, s3_key):
            error_msg = f"Data file not found in S3: {s3_uri}"
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
        
        log.info(f"Processing file from s3://{s3_bucket}/{s3_key} with advanced pyarrow 14.0.2 optimizations")
        
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Download to a local temporary file first for more efficient processing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_csv_path = temp_file.name
            log.info(f"Downloading CSV to temporary file: {temp_csv_path}")
            s3_client.download_file(s3_bucket, s3_key, temp_csv_path)
        
        try:
            # Use PyArrow's native CSV reader for memory efficiency
            log.info("Using PyArrow's native CSV reader for memory-efficient parsing")
            
            # First just read the header to understand the schema
            read_options = csv.ReadOptions(
                use_threads=True,
                block_size=2**25,  # 32MB blocks for memory efficiency
            )
            
            # Configure conversion options for data type inference
            convert_options = csv.ConvertOptions(
                strings_can_be_null=True,
                null_values=['null', 'NULL', 'Null', 'NA', 'N/A', 'nan', ''],
                timestamp_parsers=['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y']
            )
            
            # Parse CSV directly to Arrow Table
            log.info("Parsing CSV to Arrow Table...")
            arrow_table = csv.read_csv(
                temp_csv_path,
                read_options=read_options,
                parse_options=csv.ParseOptions(newlines_in_values=False),
                convert_options=convert_options
            )
            
            log.info(f"CSV parsed to Arrow Table with {len(arrow_table)} rows and {len(arrow_table.column_names)} columns")
            
            # We don't need a scanner since we already have the full arrow_table
            # Just write directly to parquet using advanced features of pyarrow 14.0.2
            log.info(f"Writing to Parquet with ZSTD compression: {output_path}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            pq.write_table(
                arrow_table,
                output_path,
                compression='zstd',
                compression_level=3,
                use_dictionary=True,
                write_statistics=True,
                use_deprecated_int96_timestamps=False,
                coerce_timestamps='ms',
                allow_truncated_timestamps=False
            )
            
            # Clean up
            del arrow_table
            gc.collect()
            
            # Verify the file was created
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Parquet file not created: {output_path}")
            
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            log.info(f"Successfully converted to Parquet: {output_path} ({file_size_mb:.2f} MB)")
            
            # Remove temporary CSV file
            os.unlink(temp_csv_path)
            
            return output_path
            
        except Exception as e:
            log.error(f"Error in PyArrow processing: {str(e)}")
            
            # Clean up temp file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
                
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
