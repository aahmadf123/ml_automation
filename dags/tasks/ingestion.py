# tasks/ingestion.py

import os
import logging
import boto3
import pandas as pd
from airflow.models import Variable
from tasks.cache import is_cache_valid, update_cache
from botocore.exceptions import ClientError
from dags.utils.config import DATA_BUCKET, AWS_REGION
from dags.utils.storage import download as s3_download
from dags.utils.slack import post as send_message

S3_BUCKET = DATA_BUCKET
S3_DATA_FOLDER = "raw-data"
LOCAL_CSV_PATH = "/tmp/homeowner_data.csv"
LOCAL_PARQUET_PATH = "/tmp/homeowner_data.parquet"

s3_client = boto3.client("s3")

def check_s3_file_exists(bucket: str, key: str) -> bool:
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise

def ingest_data_from_s3() -> str:
    """
    Downloads the raw CSV file from S3 (with caching), 
    converts it to Parquet, and returns the Parquet path.
    """
    s3_key = f"{S3_DATA_FOLDER}/ut_loss_history_1.csv"
    
    # Check if file exists in S3
    if not check_s3_file_exists(S3_BUCKET, s3_key):
        error_msg = f"Data file not found in S3: s3://{S3_BUCKET}/{s3_key}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        if not is_cache_valid(S3_BUCKET, s3_key, LOCAL_CSV_PATH):
            update_cache(S3_BUCKET, s3_key, LOCAL_CSV_PATH)
            logging.info(f"Downloaded and cached: {s3_key}")
        else:
            logging.info(f"Cache hit. Skipping download for: {s3_key}")
        
        # Convert CSV → Parquet
        df = pd.read_csv(LOCAL_CSV_PATH)
        df.to_parquet(LOCAL_PARQUET_PATH, index=False)
        return LOCAL_PARQUET_PATH
        
    except Exception as e:
        logging.error(f"Error in data ingestion: {str(e)}")
        raise
