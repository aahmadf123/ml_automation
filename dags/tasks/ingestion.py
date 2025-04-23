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
    Efficiently process large CSV file from S3 using streaming and convert to Parquet.
    
    This function:
    1. Checks if the file exists in S3
    2. Processes the data in a memory-efficient way using smart_open and dask
    3. Converts to Parquet format using a streaming approach
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
        from utils.config import DATA_BUCKET, AWS_REGION
        
        try:
            # First try to use dask for distributed processing
            import dask.dataframe as dd
            use_dask = True
            log.info("Using dask for distributed processing")
        except ImportError:
            use_dask = False
            log.info("Dask not available, using smart_open for streaming")
            # Make sure smart_open is installed
            try:
                import smart_open
            except ImportError:
                log.warning("smart_open not installed, installing...")
                import subprocess
                subprocess.check_call(["pip", "install", "smart_open[s3]"])
                import smart_open
        
        # S3 paths
        s3_key = f"{S3_DATA_FOLDER}/ut_loss_history_1.csv"
        s3_bucket = DATA_BUCKET
        s3_uri = f"s3://{s3_bucket}/{s3_key}"
        
        # Check if file exists in S3
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
        
        log.info(f"Processing large file from {s3_uri}")
        
        if use_dask:
            # Process with Dask - more efficient for very large files
            # Create S3 client with proper credentials
            s3_client = boto3.client('s3', region_name=AWS_REGION)
            
            # Set up storage options for dask
            storage_options = {
                'client': s3_client,
            }
            
            # Read CSV directly from S3 using dask
            log.info("Reading CSV from S3 using Dask")
            ddf = dd.read_csv(
                s3_uri,
                storage_options=storage_options,
                assume_missing=True,
                dtype_backend='pyarrow',  # Use PyArrow for better memory efficiency
                blocksize="500MB"  # Adjust based on your memory constraints
            )
            
            # Write to parquet with proper partitioning
            log.info(f"Writing to Parquet: {LOCAL_PARQUET_PATH}")
            ddf.to_parquet(
                LOCAL_PARQUET_PATH,
                engine='pyarrow',
                compression='snappy',  # Good balance of compression and speed
                write_index=False
            )
            
        else:
            # Process with smart_open - good for streaming without dependencies
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
            from smart_open import open
            
            # Initialize PyArrow schema and writer
            log.info("Initializing PyArrow writer")
            
            # Process the CSV in chunks to avoid memory issues
            chunksize = 400000  # Adjust based on your memory constraints
            
            # Open the S3 file for streaming
            s3_reader = open(s3_uri, 'r', transport_params={'client': boto3.client('s3', region_name=AWS_REGION)})
            
            # Create a schema writer after reading the first chunk
            log.info(f"Reading CSV in chunks and writing to {LOCAL_PARQUET_PATH}")
            
            # Use first chunk to infer schema
            first_chunk = pd.read_csv(s3_reader, nrows=chunksize)
            s3_reader.seek(0)  # Reset position to start of file
            
            # Create the schema based on the first chunk
            schema = pa.Schema.from_pandas(first_chunk)
            
            # Create the parquet writer
            with pq.ParquetWriter(LOCAL_PARQUET_PATH, schema, compression='snappy') as writer:
                # Process in chunks
                for chunk_i, chunk in enumerate(pd.read_csv(s3_reader, chunksize=chunksize)):
                    table = pa.Table.from_pandas(chunk, schema=schema)
                    writer.write_table(table)
                    log.info(f"Processed chunk {chunk_i+1} ({chunksize} rows)")
        
        # Verify the Parquet file was created
        if not os.path.exists(LOCAL_PARQUET_PATH):
            raise FileNotFoundError(f"Parquet file not created: {LOCAL_PARQUET_PATH}")
        
        log.info(f"Successfully converted large file to Parquet: {LOCAL_PARQUET_PATH}")
        
        return LOCAL_PARQUET_PATH
        
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
