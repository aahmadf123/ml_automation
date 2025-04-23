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
    2. Processes the data in a memory-efficient way using smart_open and pyarrow
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
        import pandas as pd
        from utils.config import DATA_BUCKET, AWS_REGION
        
        # Make sure smart_open and pyarrow are installed
        try:
            import smart_open
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            log.warning("Required packages not installed, installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "smart_open[s3] pyarrow"])
            import smart_open
            import pyarrow as pa
            import pyarrow.parquet as pq
        
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
        
        # Process with smart_open - good for streaming large files
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Process the CSV in chunks to avoid memory issues
        chunksize = 400000  # Using larger chunks as adjusted
        
        # Open the S3 file for streaming
        try:
            s3_reader = smart_open.open(s3_uri, 'r', 
                                       transport_params={'client': s3_client})
        except Exception as e:
            log.error(f"Error opening S3 file with smart_open: {str(e)}")
            raise
        
        # Create a schema writer after reading the first chunk
        log.info(f"Reading CSV in chunks and writing to {LOCAL_PARQUET_PATH}")
        
        try:
            # Use first chunk to infer schema
            first_chunk = pd.read_csv(s3_reader, nrows=chunksize)
            log.info(f"Read first chunk with {len(first_chunk)} rows and {len(first_chunk.columns)} columns")
            
            # Reset position to start of file
            s3_reader.seek(0)
            
            # Create the schema based on the first chunk
            schema = pa.Schema.from_pandas(first_chunk)
            log.info(f"Created schema with {len(schema.names)} fields")
            
            # Create the parquet writer with compression
            with pq.ParquetWriter(LOCAL_PARQUET_PATH, schema, compression='snappy') as writer:
                # Process in chunks
                chunk_count = 0
                total_rows = 0
                
                for chunk in pd.read_csv(s3_reader, chunksize=chunksize):
                    chunk_count += 1
                    total_rows += len(chunk)
                    
                    # Convert chunk to PyArrow table
                    table = pa.Table.from_pandas(chunk, schema=schema)
                    
                    # Write the table to the parquet file
                    writer.write_table(table)
                    
                    log.info(f"Processed chunk {chunk_count} with {len(chunk)} rows (total: {total_rows} rows)")
            
            log.info(f"Completed processing {total_rows} rows in {chunk_count} chunks")
        
        except Exception as e:
            log.error(f"Error in chunked processing: {str(e)}")
            raise
        finally:
            # Close the S3 reader
            s3_reader.close()
        
        # Verify the Parquet file was created
        if not os.path.exists(LOCAL_PARQUET_PATH):
            raise FileNotFoundError(f"Parquet file not created: {LOCAL_PARQUET_PATH}")
        
        file_size_mb = os.path.getsize(LOCAL_PARQUET_PATH) / (1024 * 1024)
        log.info(f"Successfully converted large file to Parquet: {LOCAL_PARQUET_PATH} ({file_size_mb:.2f} MB)")
        
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
