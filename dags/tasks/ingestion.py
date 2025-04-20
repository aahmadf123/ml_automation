# tasks/ingestion.py

import os
import logging
import boto3
import pandas as pd
from airflow.models import Variable
from tasks.cache import is_cache_valid, update_cache

S3_BUCKET = os.environ.get("S3_BUCKET", Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket"))
S3_DATA_FOLDER = "raw-data"
LOCAL_CSV_PATH = "/tmp/homeowner_data.csv"
LOCAL_PARQUET_PATH = "/tmp/homeowner_data.parquet"

s3_client = boto3.client("s3")

def ingest_data_from_s3() -> str:
    """
    Downloads the raw CSV file from S3 (with caching), 
    converts it to Parquet, and returns the Parquet path.
    """
    s3_key = f"{S3_DATA_FOLDER}/ut_loss_history_1.csv"
    try:
        if not is_cache_valid(S3_BUCKET, s3_key, LOCAL_CSV_PATH):
            update_cache(S3_BUCKET, s3_key, LOCAL_CSV_PATH)
            logging.info(f"Downloaded and cached: {s3_key}")
        else:
            logging.info(f"Cache hit. Skipping download for: {s3_key}")
        # Convert CSV → Parquet
        df = pd.read_csv(LOCAL_CSV_PATH)
        df.to_parquet(LOCAL_PARQUET_PATH, index=False)
        logging.info(f"Converted CSV to Parquet at {LOCAL_PARQUET_PATH}")
        return LOCAL_PARQUET_PATH
    except Exception as e:
        logging.error(f"Error in ingest_data_from_s3: {e}")
        raise
