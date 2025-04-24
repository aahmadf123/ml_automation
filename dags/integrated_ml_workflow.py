"""
integrated_ml_workflow.py - A comprehensive ML workflow DAG integrating multiple systems:
- Uses AWS MWAA for orchestration
- Logs to MLflow (running on EC2) for experiment tracking
- Integrates with ClearML for experiment management
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateObjectOperator
from airflow.models import Variable
import os
import boto3
import pandas as pd
import logging
import json
import requests
import tempfile
import mlflow
import sys
from botocore.exceptions import ClientError
from pathlib import Path
import glob
import re

# Use direct imports for modules
import utils.config as config
import utils.clearml_config as clearml_config
import tasks.training as training
import tasks.data_quality as data_quality
import tasks.drift as drift
import utils.slack as slack
import tasks.ingestion as ingestion

# Set up logging
logger = logging.getLogger(__name__)

# Default args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'integrated_ml_workflow',
    default_args=default_args,
    description='Integrated ML workflow with MWAA, ClearML, and MLflow',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['ml', 'integration', 'clearml', 'mlflow'],
)

# Functions for tasks
def download_data(**context):
    """Download data from S3 or reuse existing ingested data"""
    from tempfile import NamedTemporaryFile
    import os
    
    logger.info("Starting download_data task")
    
    # Get bucket name from Airflow Variable to ensure it's up to date
    bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
    key = config.RAW_DATA_KEY
    
    logger.info(f"Using bucket from Airflow Variables: {bucket}")
    logger.info(f"Attempting to access data at s3://{bucket}/{key}")
    
    try:
        # Create airflow data directory if it doesn't exist (for persistence between tasks)
        data_dir = '/opt/airflow/data'
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Ensuring data directory exists: {data_dir}")
        
        # Generate a persistent filename based on the S3 key
        filename = os.path.basename(key)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_path = os.path.join(data_dir, f"raw_{timestamp}_{filename}")
        logger.info(f"Will download to persistent location: {local_path}")
        
        # Use the ingestion function from the full pipeline if available
        try:
            logger.info("Attempting to use pipeline ingestion function")
            data_path = ingestion.ingest_data_from_s3(
                bucket_name=bucket,
                key=key,
                local_path=local_path  # Pass the persistent path to the function
            )
            logger.info(f"Successfully ingested data using pipeline function: {data_path}")
        except Exception as e:
            # Fall back to direct S3 download if ingestion function fails
            logger.warning(f"Pipeline ingestion failed, falling back to direct download: {str(e)}")
            
            # Download directly to the persistent location
            s3_client = boto3.client('s3', region_name=config.AWS_REGION)
            logger.info(f"Downloading to {local_path}")
            s3_client.download_file(bucket, key, local_path)
            data_path = local_path
            logger.info(f"Successfully downloaded to {local_path}")
        
        # Also create a backup copy in /tmp as fallback
        try:
            tmp_path = f"/tmp/raw_data_backup_{timestamp}.csv"
            import shutil
            shutil.copy2(local_path, tmp_path)
            logger.info(f"Created backup copy at {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup copy: {str(e)}")
            
        # Test file exists and is accessible
        if not os.path.exists(data_path):
            logger.error(f"Downloaded file not found at {data_path}")
            raise FileNotFoundError(f"Downloaded file not found at {data_path}")
            
        file_size = os.path.getsize(data_path)
        if file_size == 0:
            logger.error(f"Downloaded file is empty: {data_path}")
            raise ValueError(f"Downloaded file is empty: {data_path}")
            
        logger.info(f"Verified file exists with size {file_size} bytes")
                
        # Log with ClearML if enabled
        try:
            clearml_task = clearml_config.init_clearml("Data_Download")
            if clearml_task:
                clearml_task.set_parameter("s3_bucket", bucket)
                clearml_task.set_parameter("s3_key", key)
                clearml_config.log_dataset_to_clearml(
                    dataset_name="Raw_Data",
                    dataset_path=data_path,
                    dataset_tags=["raw", "csv"]
                )
                clearml_task.close()
        except Exception as e:
            logger.warning(f"Error logging to ClearML: {str(e)}")
        
        # Return local path for downstream tasks
        context['ti'].xcom_push(key='data_path', value=data_path)
        
        try:
            slack.post(f":white_check_mark: Data accessed from s3://{bucket}/{key}")
        except Exception as e:
            logger.warning(f"Error sending Slack notification: {str(e)}")
            
        return data_path
        
    except Exception as e:
        logger.error(f"Error in download_data task: {str(e)}")
        
        try:
            slack.post(f":x: Failed to access data: {str(e)}")
        except:
            pass
            
        raise

def verify_file_persistence(**context):
    """Utility task to verify file persistence between tasks and fix issues if possible"""
    import os
    import glob
    from pathlib import Path
    
    logger.info("Starting verify_file_persistence task")
    
    try:
        # Get raw data path from previous task
        raw_data_path = context['ti'].xcom_pull(task_ids='download_data', key='data_path')
        
        if not raw_data_path:
            logger.error("Failed to get data_path from previous task")
            raise ValueError("No data path provided from download_data task")
            
        # Check if the file exists
        if not os.path.exists(raw_data_path):
            logger.warning(f"File not found at expected location: {raw_data_path}")
            
            # Try to locate the file
            original_filename = os.path.basename(raw_data_path)
            search_locations = [
                "/opt/airflow/data",
                "/tmp"
            ]
            
            found_file = False
            for location in search_locations:
                if not os.path.exists(location):
                    logger.warning(f"Search location does not exist: {location}")
                    continue
                    
                logger.info(f"Searching in: {location}")
                # First try exact name
                if os.path.exists(os.path.join(location, original_filename)):
                    new_path = os.path.join(location, original_filename)
                    logger.info(f"Found exact file at: {new_path}")
                    found_file = True
                    context['ti'].xcom_push(key='data_path', value=new_path)
                    break
                    
                # Then try pattern matching
                pattern = f"{location}/*{original_filename.split('_')[-1]}"
                matches = glob.glob(pattern)
                if matches:
                    # Use the most recently modified file
                    newest_file = max(matches, key=os.path.getmtime)
                    logger.info(f"Found similar file at: {newest_file}")
                    found_file = True
                    context['ti'].xcom_push(key='data_path', value=newest_file)
                    break
            
            if not found_file:
                # Last resort: look for any CSV file
                for location in search_locations:
                    if not os.path.exists(location):
                        continue
                        
                    csv_files = glob.glob(f"{location}/*.csv")
                    if csv_files:
                        newest_csv = max(csv_files, key=os.path.getmtime)
                        logger.info(f"Found CSV file at: {newest_csv}")
                        found_file = True
                        context['ti'].xcom_push(key='data_path', value=newest_csv)
                        break
            
            if not found_file:
                # If still not found, download from S3
                try:
                    logger.warning("No existing files found, attempting direct S3 download")
                    bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
                    key = config.RAW_DATA_KEY
                    s3_client = boto3.client('s3', region_name=config.AWS_REGION)
                    
                    # Create a persistent path
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = os.path.basename(key)
                    data_dir = '/opt/airflow/data'
                    os.makedirs(data_dir, exist_ok=True)
                    recovery_path = os.path.join(data_dir, f"recovery_{timestamp}_{filename}")
                    
                    logger.info(f"Downloading to recovery location: {recovery_path}")
                    s3_client.download_file(bucket, key, recovery_path)
                    context['ti'].xcom_push(key='data_path', value=recovery_path)
                    logger.info(f"Successfully downloaded recovery file")
                    found_file = True
                except Exception as e:
                    logger.error(f"Recovery download failed: {str(e)}")
            
            if not found_file:
                logger.error("Could not find or recover data file")
                raise FileNotFoundError(f"Data file not found and recovery attempts failed")
        else:
            logger.info(f"Verified file exists at: {raw_data_path}")
            
            # Check file size
            file_size = os.path.getsize(raw_data_path)
            logger.info(f"File size: {file_size} bytes")
            
            if file_size == 0:
                logger.error(f"File exists but is empty: {raw_data_path}")
                raise ValueError(f"File exists but is empty: {raw_data_path}")
                
            # Create a backup for extra safety
            try:
                data_dir = '/opt/airflow/data'
                os.makedirs(data_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.basename(raw_data_path)
                backup_path = os.path.join(data_dir, f"verified_{timestamp}_{filename}")
                
                import shutil
                shutil.copy2(raw_data_path, backup_path)
                logger.info(f"Created verified backup at: {backup_path}")
                
                # Push both paths to XCom
                context['ti'].xcom_push(key='verified_data_path', value=backup_path)
            except Exception as e:
                logger.warning(f"Failed to create backup: {str(e)}")
        
        # Always return the (possibly updated) data path
        return context['ti'].xcom_pull(key='data_path')
        
    except Exception as e:
        logger.error(f"Error in verify_file_persistence task: {str(e)}")
        raise

def process_data(**context):
    """Process the data and prepare for model training"""
    import pandas as pd
    import numpy as np
    from tempfile import NamedTemporaryFile
    import os
    import mlflow
    from pathlib import Path
    
    logger.info("Starting process_data task")
    
    try:
        # Get raw data path from previous task with validation
        raw_data_path = context['ti'].xcom_pull(task_ids='download_data', key='data_path')
        
        if not raw_data_path:
            logger.error("Failed to get data_path from previous task")
            raise ValueError("No data path provided from download_data task")
            
        # Verify the file exists and has content
        data_file = Path(raw_data_path)
        if not data_file.exists():
            logger.warning(f"Data file does not exist at primary location: {raw_data_path}")
            
            # Try to find the file in alternative locations
            potential_locations = [
                # Check for backup in /tmp
                Path("/tmp").glob("raw_data_backup_*.csv"),
                # Check for files in data directory
                Path("/opt/airflow/data").glob("raw_*.csv"),
                # Check for any CSV files in data directory
                Path("/opt/airflow/data").glob("*.csv"),
                # Last resort - check for any parquet files
                Path("/opt/airflow/data").glob("*.parquet")
            ]
            
            # Try each location
            for pattern in potential_locations:
                files = sorted(pattern, key=os.path.getmtime, reverse=True)
                if files:
                    # Found files - use most recent
                    raw_data_path = str(files[0])
                    data_file = Path(raw_data_path)
                    logger.info(f"Found alternative file at: {raw_data_path}")
                    break
            
            # If still not found, try to list S3 and download directly
            if not data_file.exists():
                logger.warning("Attempting to download data directly from S3")
                try:
                    s3_client = boto3.client('s3', region_name=config.AWS_REGION)
                    bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
                    key = config.RAW_DATA_KEY
                    
                    # Create a new local path
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = os.path.basename(key)
                    local_path = f"/opt/airflow/data/recovery_{timestamp}_{filename}"
                    
                    # Download the file
                    logger.info(f"Downloading from s3://{bucket}/{key} to {local_path}")
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    s3_client.download_file(bucket, key, local_path)
                    raw_data_path = local_path
                    data_file = Path(raw_data_path)
                    logger.info(f"Successfully downloaded recovery file to {local_path}")
                except Exception as e:
                    logger.error(f"Recovery download failed: {str(e)}")
            
            # If still not found, raise error
            if not data_file.exists():
                logger.error(f"Data file not found at primary or alternative locations")
                raise FileNotFoundError(f"Data file not found: {raw_data_path}")
            
        file_size = data_file.stat().st_size
        logger.info(f"Found data file {raw_data_path} with size {file_size} bytes")
        
        if file_size == 0:
            logger.error(f"Data file is empty: {raw_data_path}")
            raise ValueError(f"Data file is empty: {raw_data_path}")
        
        # Initialize MLflow with robust error handling
        try:
            mlflow_uri = config.MLFLOW_URI
            mlflow_experiment = config.MLFLOW_EXPERIMENT
            
            logger.info(f"Connecting to MLflow at {mlflow_uri}")
            mlflow.set_tracking_uri(mlflow_uri)
            
            # Check if experiment exists, create if not
            experiment = mlflow.get_experiment_by_name(mlflow_experiment)
            if not experiment:
                logger.info(f"Creating new MLflow experiment: {mlflow_experiment}")
                mlflow.create_experiment(mlflow_experiment)
                
            mlflow.set_experiment(mlflow_experiment)
            logger.info(f"Successfully connected to MLflow experiment: {mlflow_experiment}")
        except Exception as e:
            logger.warning(f"Unable to configure MLflow: {str(e)}. Will continue without MLflow tracking.")
        
        # Start ClearML task
        clearml_task = None
        try:
            clearml_task = clearml_config.init_clearml("Data_Processing")
            logger.info("Successfully initialized ClearML task")
        except Exception as e:
            logger.warning(f"Error initializing ClearML: {str(e)}. Will continue without ClearML tracking.")
        
        # Load the data with robust error handling for different file formats
        logger.info(f"Loading data from {raw_data_path}")
        try:
            if raw_data_path.endswith('.csv'):
                # Try different encoding options if default fails
                try:
                    df = pd.read_csv(raw_data_path)
                except UnicodeDecodeError:
                    logger.warning("Default encoding failed, trying UTF-8")
                    df = pd.read_csv(raw_data_path, encoding='utf-8')
                except Exception:
                    logger.warning("UTF-8 encoding failed, trying latin1")
                    df = pd.read_csv(raw_data_path, encoding='latin1')
            elif raw_data_path.endswith('.parquet'):
                df = pd.read_parquet(raw_data_path)
            else:
                logger.error(f"Unsupported file format: {raw_data_path}")
                raise ValueError(f"Unsupported file format: {raw_data_path}")
                
            # Confirm data was loaded
            if df.empty:
                logger.error("Loaded DataFrame is empty")
                raise ValueError("Loaded DataFrame is empty")
                
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
        
        # Start MLflow run for data processing
        mlflow_run = None
        try:
            mlflow_run = mlflow.start_run(run_name="data_processing")
            run_id = mlflow_run.info.run_id
            logger.info(f"Started MLflow run with ID: {run_id}")
            
            # Log raw data stats
            mlflow.log_param("raw_data_rows", len(df))
            mlflow.log_param("raw_data_columns", len(df.columns))
            
            logger.info("Logged initial data stats to MLflow")
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {str(e)}. Will continue without MLflow tracking.")
            run_id = None
        
        # Basic preprocessing
        logger.info("Starting data preprocessing")
        
        # Keep track of preprocessing operations for logging
        preprocessing_ops = []
        
        # Handle missing values
        missing_before = df.isna().sum().sum()
        logger.info(f"Missing values before imputation: {missing_before}")
        
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].isna().any():
                    df[col] = df[col].fillna('unknown')
                    preprocessing_ops.append(f"Filled missing values in '{col}' with 'unknown'")
            else:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    preprocessing_ops.append(f"Filled missing values in '{col}' with median ({median_val})")
        
        missing_after = df.isna().sum().sum()
        logger.info(f"Missing values after imputation: {missing_after}")
        logger.info(f"Removed {missing_before - missing_after} missing values")
        
        # Log with ClearML if available
        if clearml_task:
            clearml_task.set_parameter("processed_rows", len(df))
            clearml_task.set_parameter("processed_columns", len(df.columns))
            clearml_task.set_parameter("missing_values_removed", missing_before - missing_after)
            logger.info("Logged preprocessing stats to ClearML")
        
        # Save processed data to a more persistent location
        try:
            # Create a more descriptive and persistent filename
            # Use airflow data directory instead of /tmp to ensure it persists between tasks
            data_dir = os.path.join('/opt/airflow/data')
            # Create directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_filename = f"processed_data_{timestamp}.parquet"
            processed_path = os.path.join(data_dir, processed_filename)
            
            logger.info(f"Saving processed data to {processed_path}")
            df.to_parquet(processed_path, index=False)
                
            # Verify the file was created and has content
            if not os.path.exists(processed_path) or os.path.getsize(processed_path) == 0:
                logger.error(f"Failed to create processed data file or file is empty: {processed_path}")
                raise IOError(f"Failed to create processed data file or file is empty: {processed_path}")
                
            logger.info(f"Successfully saved processed data to {processed_path}")
            
            # Also save a backup copy in case the primary location fails
            backup_path = f"/tmp/{processed_filename}"
            logger.info(f"Saving backup copy to {backup_path}")
            df.to_parquet(backup_path, index=False)
            logger.info(f"Backup copy saved to {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
            # Fallback to using tempfile if the main approach fails
            try:
                with NamedTemporaryFile(delete=False, suffix=f'_{timestamp}.parquet') as temp_file:
                    processed_path = temp_file.name
                    logger.info(f"Fallback: Saving processed data to {processed_path}")
                    df.to_parquet(processed_path, index=False)
                    
                if not os.path.exists(processed_path) or os.path.getsize(processed_path) == 0:
                    logger.error(f"Fallback failed: File is empty or doesn't exist: {processed_path}")
                    raise IOError(f"Fallback failed: File is empty or doesn't exist: {processed_path}")
                    
                logger.info(f"Successfully saved processed data to fallback location: {processed_path}")
            except Exception as inner_e:
                logger.error(f"Fallback also failed: {str(inner_e)}")
                raise
        
        # Log processed data stats to MLflow
        if mlflow_run:
            try:
                mlflow.log_param("processed_data_rows", len(df))
                mlflow.log_metric("missing_values_count", df.isna().sum().sum())
                mlflow.log_metric("missing_values_removed", missing_before - missing_after)
                
                # Log preprocessing operations as a text artifact
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                    ops_file = f.name
                    f.write("\n".join(preprocessing_ops))
                mlflow.log_artifact(ops_file, "preprocessing_operations")
                os.unlink(ops_file)  # Clean up temp file
                
                logger.info("Logged final data stats to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log processed data stats to MLflow: {str(e)}")
        
        # Upload to S3
        processed_s3_key = None
        try:
            # Get bucket directly from Variables
            bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
            
            s3_client = boto3.client('s3', region_name=config.AWS_REGION)
            processed_s3_key = f"processed/processed_data_{timestamp}.parquet"
            
            logger.info(f"Uploading processed data to S3: s3://{bucket}/{processed_s3_key}")
            s3_client.upload_file(processed_path, bucket, processed_s3_key)
            logger.info(f"Successfully uploaded processed data to S3")
            
            # Log S3 location in MLflow
            if mlflow_run:
                try:
                    mlflow.log_param("processed_data_s3_path", f"s3://{bucket}/{processed_s3_key}")
                except Exception as e:
                    logger.warning(f"Failed to log S3 path to MLflow: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to upload processed data to S3: {str(e)}")
            logger.info("Will continue with local processed data file")
        
        # Close MLflow run
        if mlflow_run:
            try:
                mlflow.end_run()
                logger.info("Successfully ended MLflow run")
            except Exception as e:
                logger.warning(f"Failed to properly end MLflow run: {str(e)}")
        
        # Test file exists and is accessible before proceeding
        if not os.path.exists(processed_path):
            logger.error(f"File verification failed - file doesn't exist: {processed_path}")
            raise FileNotFoundError(f"File verification failed - file doesn't exist: {processed_path}")
            
        file_size = os.path.getsize(processed_path)
        if file_size == 0:
            logger.error(f"File verification failed - file is empty: {processed_path}")
            raise ValueError(f"File verification failed - file is empty: {processed_path}")
            
        logger.info(f"File verification passed: {processed_path} exists with size {file_size} bytes")
        
        # Push the processed data path and S3 key to Xcom
        context['ti'].xcom_push(key='processed_data_path', value=processed_path)
        context['ti'].xcom_push(key='processed_data_filename', value=processed_filename)
        if processed_s3_key:
            context['ti'].xcom_push(key='processed_s3_key', value=processed_s3_key)
        
        # Cleanup raw data file
        try:
            os.remove(raw_data_path)
            logger.info(f"Cleaned up raw data file: {raw_data_path}")
        except Exception as e:
            logger.warning(f"Error removing raw data file: {str(e)}")
        
        # Send Slack notification
        try:
            slack.post(f":white_check_mark: Data processed successfully. Rows: {len(df)}, Columns: {len(df.columns)}")
            logger.info("Sent success notification to Slack")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return processed_path
        
    except Exception as e:
        logger.error(f"Error in process_data task: {str(e)}")
        
        # Send failure notification
        try:
            slack.post(f":x: Failed to process data: {str(e)}")
        except:
            pass
            
        raise

def check_data_quality(**context):
    """
    Run data quality checks on processed data.
    Uses aggressive memory management for large datasets.
    """
    import tempfile
    import gc
    import json
    import os
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datetime import datetime
    import mlflow
    from psutil import Process, virtual_memory
    from dags.tasks.data_quality import DataQualityMonitor
    import dags.config as config
    import dags.utils.clearml_config as clearml_config
    
    process = Process(os.getpid())
    logger = context["ti"].log

    # Get processed data path from previous task
    try:
        processed_data_path = context["ti"].xcom_pull(task_ids="process_data", key="processed_data_path")
        processed_data_filename = context["ti"].xcom_pull(task_ids="process_data", key="processed_data_filename")
        
        logger.info(f"Retrieved processed data path: {processed_data_path}")
        
        # Verify file exists and check size
        if not os.path.exists(processed_data_path):
            logger.error(f"Processed data file not found at {processed_data_path}")
            
            # Try to recover from alternative locations
            alternative_paths = [
                f"/tmp/{processed_data_filename}",
                f"./data/processed/{processed_data_filename}",
                f"/opt/airflow/data/processed/{processed_data_filename}"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Found file at alternative location: {alt_path}")
                    processed_data_path = alt_path
                    break
            
            # If still not found, try to download from S3
            if not os.path.exists(processed_data_path):
                try:
                    from dags.utils.s3 import download_file
                    s3_path = f"processed/{processed_data_filename}"
                    local_path = f"/tmp/{processed_data_filename}"
                    logger.info(f"Attempting to download from S3: {s3_path}")
                    download_file(s3_path, local_path)
                    if os.path.exists(local_path):
                        processed_data_path = local_path
                        logger.info(f"Successfully downloaded file from S3")
                    else:
                        logger.error(f"Failed to download file from S3")
                except Exception as e:
                    logger.error(f"Error downloading from S3: {str(e)}")
            
            # If still not found, raise error
            if not os.path.exists(processed_data_path):
                raise FileNotFoundError(f"Processed data file not found at {processed_data_path} or any alternative locations")
        
        # Check file size and log
        file_size_mb = os.path.getsize(processed_data_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Initialize monitor
        logger.info("Initializing DataQualityMonitor")
        monitor = DataQualityMonitor()
        
        # Log available memory
        try:
            mem = virtual_memory()
            logger.info(f"Available memory: {mem.available/(1024*1024*1024):.2f} GB out of {mem.total/(1024*1024*1024):.2f} GB")
        except Exception as e:
            logger.warning(f"Could not get memory info: {str(e)}")
            
        # Get basic dataset info without loading entire dataset
        try:
            # Use pyarrow to examine dataset without loading everything
            parquet_file = pq.ParquetFile(processed_data_path)
            num_rows = parquet_file.metadata.num_rows
            num_columns = len(parquet_file.schema.names)
            logger.info(f"Dataset has {num_rows:,} rows and {num_columns} columns")
            
            # Determine sample size based on total rows - be aggressive for very large datasets
            if num_rows > 5_000_000:
                sample_size = 50_000  # Extra small sample for extremely large datasets
                sample_fraction = None
            elif num_rows > 1_000_000:
                sample_size = 100_000  # Small sample for very large datasets
                sample_fraction = None
            elif num_rows > 500_000:
                sample_size = 200_000
                sample_fraction = None
            elif num_rows > 100_000:
                sample_fraction = 0.2  # 20% for medium datasets
                sample_size = int(num_rows * sample_fraction)
            else:
                sample_fraction = 1.0  # Use full dataset for small datasets
                sample_size = num_rows
                
            logger.info(f"Will use {sample_size:,} rows ({sample_size/num_rows:.1%} of data) for quality checks")
            
            # Load data with sampling for extremely efficient memory usage
            if sample_fraction is not None and sample_fraction < 1.0:
                # For medium datasets, read with pandas sampling
                df = pd.read_parquet(
                    processed_data_path, 
                    engine='pyarrow',
                    filters=None  # No filtering
                ).sample(frac=sample_fraction, random_state=42)
                logger.info(f"Loaded {len(df):,} rows using pandas sampling")
            elif sample_size < num_rows:
                # For large datasets, use pyarrow to read and sample more efficiently
                # First, determine row groups and calculate how many we need
                row_groups = parquet_file.num_row_groups
                rows_per_group = num_rows / row_groups if row_groups > 0 else num_rows
                
                # If we have enough row groups, we can sample at the row group level
                if row_groups >= 5 and rows_per_group < sample_size:
                    # Select random row groups to read
                    groups_needed = max(1, int(sample_size / rows_per_group))
                    groups_to_read = sorted(np.random.choice(
                        range(row_groups), 
                        size=min(groups_needed, row_groups), 
                        replace=False
                    ))
                    
                    # Read selected row groups
                    table = pa.concat_tables([
                        parquet_file.read_row_group(i) 
                        for i in groups_to_read
                    ])
                    
                    # Convert to pandas and sample if needed
                    df = table.to_pandas()
                    if len(df) > sample_size:
                        df = df.sample(sample_size, random_state=42)
                    
                    logger.info(f"Loaded {len(df):,} rows using row group sampling from {len(groups_to_read)} row groups")
                    del table
                    gc.collect()
                else:
                    # Read the first N rows for a quick approximation
                    # Not ideal statistically but good for memory efficiency
                    df = next(pq.read_table(
                        processed_data_path,
                        use_threads=True,
                        memory_map=True
                    ).to_batches(sample_size)).to_pandas()
                    
                    logger.info(f"Loaded first {len(df):,} rows for initial assessment")
                    
                    # If we have enough memory, try proper random sampling
                    try:
                        mem = virtual_memory()
                        if mem.available > (num_rows * num_columns * 8 * 0.1):  # Rough estimate: 10% of full size
                            # Create row indices for random sampling
                            indices = sorted(np.random.choice(
                                num_rows, 
                                size=sample_size, 
                                replace=False
                            ))
                            
                            # Use pyarrow's filter to select specific rows
                            # This is more memory efficient than pandas
                            table = pq.read_table(
                                processed_data_path,
                                filters=[('__index_level_0__', 'in', indices)],
                                use_threads=True,
                                memory_map=True
                            )
                            
                            # Replace our dataframe with the proper random sample
                            del df
                            gc.collect()
                            df = table.to_pandas()
                            del table
                            gc.collect()
                            
                            logger.info(f"Replaced with properly randomized sample of {len(df):,} rows")
                        else:
                            logger.info("Not enough memory for proper random sampling, using initial rows")
                    except Exception as e:
                        logger.warning(f"Error during advanced sampling, using initial rows: {str(e)}")
            else:
                # For manageable files, load all data
                df = pd.read_parquet(processed_data_path, engine='pyarrow')
                logger.info(f"Loaded full dataset with {len(df):,} rows")
            
            # For very large datasets, reduce column count if needed
            if len(df.columns) > 30 and len(df) > 100000:
                # Calculate memory usage per column
                try:
                    memory_usage = df.memory_usage(deep=True)
                    total_memory_mb = memory_usage.sum() / (1024 * 1024)
                    logger.info(f"Current DataFrame memory usage: {total_memory_mb:.2f} MB")
                    
                    # If memory usage is too high, keep only essential columns
                    if total_memory_mb > 500:  # If using more than 500MB
                        # Sort columns by memory usage
                        columns_by_size = memory_usage.sort_values(ascending=False)
                        
                        # Strategy: Keep all categorical, datetime and ID columns (usually important)
                        # but limit numeric columns by their memory usage
                        cat_cols = df.select_dtypes(include=['category', 'object', 'datetime64']).columns.tolist()
                        # Add columns that look like IDs
                        id_cols = [col for col in df.columns if col.lower().endswith('id') or col.lower() == 'id']
                        # Prioritize these columns
                        priority_cols = list(set(cat_cols + id_cols))
                        
                        # Calculate how many more columns we can include
                        remaining_slots = max(10, 30 - len(priority_cols))
                        
                        # Find small numeric columns
                        numeric_cols = df.select_dtypes(include=['number']).columns.difference(priority_cols)
                        small_numeric_cols = memory_usage[numeric_cols].nsmallest(remaining_slots).index.tolist()
                        
                        # Combine all columns we want to keep
                        columns_to_keep = priority_cols + small_numeric_cols
                        
                        # Only keep these columns
                        df = df[columns_to_keep]
                        logger.info(f"Reduced columns from {len(memory_usage)} to {len(columns_to_keep)} to save memory")
                        
                        # Report new memory usage
                        new_memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                        logger.info(f"New DataFrame memory usage: {new_memory_mb:.2f} MB")
                except Exception as e:
                    logger.warning(f"Could not optimize columns: {str(e)}")
            
            # Verify data was loaded successfully
            if df.empty:
                logger.error("Loaded DataFrame is empty")
                raise ValueError("Loaded DataFrame is empty")
                
            logger.info(f"Successfully loaded processed data with shape: {df.shape}")
            
            # Log memory usage 
            try:
                memory_info = process.memory_info()
                memory_usage_mb = memory_info.rss / (1024 * 1024)
                logger.info(f"Memory usage after loading data: {memory_usage_mb:.2f} MB")
            except:
                pass
                
        except Exception as e:
            logger.error(f"Failed to load processed data: {str(e)}")
            raise
        
        # Run data quality checks with error handling, using small batches for memory efficiency
        try:
            logger.info("Running data quality checks")
            
            # For very large datasets, process in chunks
            if len(df) > 100000:
                logger.info(f"Using chunked processing for {len(df):,} rows")
                
                # Split into smaller chunks for processing
                chunk_size = min(50000, max(10000, len(df) // 5))  # Adaptive chunk size
                num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size > 0 else 0)
                
                # Initialize empty results
                quality_results = {
                    'timestamp': datetime.now().isoformat(),
                    'missing_values': {},
                    'outliers': {},
                    'data_drift': {},
                    'correlation_changes': {},
                    'total_issues': 0,
                    'missing_value_issues': 0,
                    'outlier_issues': 0
                }
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(df))
                    if start_idx >= len(df):
                        break
                        
                    logger.info(f"Processing chunk {i+1}/{num_chunks} (rows {start_idx:,} to {end_idx:,})")
                    chunk_df = df.iloc[start_idx:end_idx].copy()
                    
                    # Force garbage collection before processing
                    gc.collect()
                    
                    # Get memory usage before processing
                    try:
                        memory_info = process.memory_info()
                        memory_usage_mb = memory_info.rss / (1024 * 1024)
                        logger.info(f"Memory usage before processing chunk {i+1}: {memory_usage_mb:.2f} MB")
                    except:
                        pass
                    
                    # Run quality checks on this chunk with try/except to prevent entire job failure
                    try:
                        chunk_results = monitor.run_quality_checks(chunk_df)
                        
                        # Merge results from this chunk
                        for key in ['missing_values', 'outliers', 'data_drift', 'correlation_changes']:
                            quality_results[key].update(chunk_results.get(key, {}))
                    except Exception as e:
                        logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    
                    # Free memory
                    del chunk_df
                    gc.collect()
                    
                    # Get memory usage after processing
                    try:
                        memory_info = process.memory_info()
                        memory_usage_mb = memory_info.rss / (1024 * 1024)
                        logger.info(f"Memory usage after processing chunk {i+1}: {memory_usage_mb:.2f} MB")
                    except:
                        pass
                
                # Recalculate total issues
                for key in ['missing_values', 'outliers']:
                    if key == 'missing_values':
                        quality_results['missing_value_issues'] = len(quality_results[key])
                    elif key == 'outliers':
                        quality_results['outlier_issues'] = len(quality_results[key])
                
                quality_results['total_issues'] = sum(len(quality_results[key]) for key in ['missing_values', 'outliers', 'data_drift', 'correlation_changes'])
                logger.info(f"Combined results from {num_chunks} chunks, found {quality_results['total_issues']} total issues")
                
            else:
                # For smaller datasets, process all at once
                quality_results = monitor.run_quality_checks(df)
                logger.info(f"Completed quality checks, found {quality_results.get('total_issues', 0)} issues")
            
            # Clean up dataframe reference to free memory
            del df
            gc.collect()
            
            # Log memory usage again
            try:
                memory_info = process.memory_info()
                memory_usage_mb = memory_info.rss / (1024 * 1024)
                logger.info(f"Memory usage after quality checks: {memory_usage_mb:.2f} MB")
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error during quality checks: {str(e)}")
            # Try to clean up memory before raising
            try:
                del df
                gc.collect()
            except:
                pass
            raise
        
        # Log results to MLflow
        if mlflow_run:
            try:
                # Log parameters
                logger.info("Logging quality check parameters to MLflow")
                mlflow.log_params({
                    "missing_threshold": monitor.config["missing_threshold"],
                    "outlier_threshold": monitor.config["outlier_threshold"],
                    "correlation_threshold": monitor.config["correlation_threshold"],
                    "sample_size": sample_size,
                    "total_rows": num_rows
                })
                
                # Log metrics
                logger.info("Logging quality check metrics to MLflow")
                # Make sure these metrics exist, otherwise use defaults
                mlflow.log_metric("total_issues", quality_results.get("total_issues", 0))
                mlflow.log_metric("missing_value_issues", quality_results.get("missing_value_issues", 0))
                mlflow.log_metric("outlier_issues", quality_results.get("outlier_issues", 0))
                
                # Create and log a simplified report to reduce memory usage
                logger.info("Creating quality report artifact")
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                    report_path = temp_file.name
                    
                    # Create a simplified version of the results to reduce size
                    simplified_results = {
                        'timestamp': quality_results.get('timestamp', datetime.now().isoformat()),
                        'total_issues': quality_results.get('total_issues', 0),
                        'missing_value_issues': quality_results.get('missing_value_issues', 0),
                        'outlier_issues': quality_results.get('outlier_issues', 0),
                        'missing_values_count': len(quality_results.get('missing_values', {})),
                        'outliers_count': len(quality_results.get('outliers', {})),
                        'data_drift_count': len(quality_results.get('data_drift', {})),
                        'correlation_changes_count': len(quality_results.get('correlation_changes', {})),
                        # Only include the first 50 items of each type to limit size
                        'top_missing_values': dict(list(quality_results.get('missing_values', {}).items())[:50]),
                        'top_outliers': dict(list(quality_results.get('outliers', {}).items())[:50]),
                        'top_data_drift': dict(list(quality_results.get('data_drift', {}).items())[:50]),
                        'top_correlation_changes': dict(list(quality_results.get('correlation_changes', {}).items())[:50])
                    }
                    
                    json.dump(simplified_results, temp_file, indent=2)
                
                mlflow.log_artifact(report_path)
                logger.info(f"Logged quality report to MLflow: {report_path}")
                
                # Clean up temp file
                try:
                    os.unlink(report_path)
                except:
                    pass
            except Exception as e:
                logger.warning(f"Failed to log quality results to MLflow: {str(e)}")
        
        # Log to ClearML if available
        if clearml_task:
            try:
                logger.info("Logging quality check results to ClearML")
                for issue_type, count in quality_results.items():
                    if isinstance(count, (int, float)):
                        clearml_task.get_logger().report_scalar(
                            title="Data Quality",
                            series=issue_type,
                            value=count,
                            iteration=0
                        )
            except Exception as e:
                logger.warning(f"Failed to log quality results to ClearML: {str(e)}")
            
            try:
                clearml_task.close()
                logger.info("Successfully closed ClearML task")
            except Exception as e:
                logger.warning(f"Failed to properly close ClearML task: {str(e)}")
        
        # Close MLflow run
        if mlflow_run:
            try:
                mlflow.end_run()
                logger.info("Successfully ended MLflow run")
            except Exception as e:
                logger.warning(f"Failed to properly end MLflow run: {str(e)}")
        
        # Continue only if below threshold or override is set
        data_quality_threshold = int(Variable.get("DATA_QUALITY_THRESHOLD", default_var="10"))
        force_continue = Variable.get("FORCE_CONTINUE", default_var="false").lower() == "true"
        
        # Make sure total_issues exists, otherwise use a default value
        total_issues = quality_results.get("total_issues", 0)
        logger.info(f"Quality issues: {total_issues}, Threshold: {data_quality_threshold}, Force continue: {force_continue}")
        
        if total_issues > data_quality_threshold and not force_continue:
            error_msg = f"Data quality issues ({total_issues}) exceed threshold ({data_quality_threshold})"
            logger.error(error_msg)
            
            try:
                slack.post(f":x: {error_msg}. Workflow stopped.")
                logger.info("Sent quality failure notification to Slack")
            except Exception as e:
                logger.warning(f"Failed to send Slack notification: {str(e)}")
                
            raise ValueError(error_msg)
        
        # Push quality results for downstream tasks
        # Slim down the quality_results to avoid XCom size issues
        simplified_results = {
            'timestamp': quality_results.get('timestamp', datetime.now().isoformat()),
            'total_issues': quality_results.get('total_issues', 0),
            'missing_value_issues': quality_results.get('missing_value_issues', 0),
            'outlier_issues': quality_results.get('outlier_issues', 0),
            'has_drift': len(quality_results.get('data_drift', {})) > 0,
            'has_correlation_changes': len(quality_results.get('correlation_changes', {})) > 0
        }
        context['ti'].xcom_push(key='quality_results', value=simplified_results)
        context['ti'].xcom_push(key='processed_data_path', value=processed_data_path)  # Pass the verified path
        
        # Send success notification
        try:
            slack.post(f":white_check_mark: Data quality check completed. Issues: {total_issues}")
            logger.info("Sent quality success notification to Slack")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return simplified_results
        
    except Exception as e:
        logger.error(f"Error in check_data_quality task: {str(e)}", exc_info=True)
        
        # Send failure notification
        try:
            slack.post(f":x: Failed to check data quality: {str(e)}")
        except:
            pass
            
        raise

def train_model(**context):
    """Train the model using processed data"""
    logger.info("Starting train_model task")
    
    try:
        # Get processed data path with validation
        processed_data_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        
        if not processed_data_path:
            logger.error("Failed to get processed_data_path from previous task")
            raise ValueError("No processed data path provided from process_data task")
            
        # Verify the file exists and has content
        if not os.path.exists(processed_data_path):
            logger.error(f"Processed data file does not exist: {processed_data_path}")
            raise FileNotFoundError(f"Processed data file not found: {processed_data_path}")
            
        file_size = os.path.getsize(processed_data_path)
        logger.info(f"Found processed data file {processed_data_path} with size {file_size} bytes")
        
        if file_size == 0:
            logger.error(f"Processed data file is empty: {processed_data_path}")
            raise ValueError(f"Processed data file is empty: {processed_data_path}")
        
        # Generate model ID with timestamp
        model_id = f"homeowner_loss_model_{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Starting training for model: {model_id}")
        
        # Call training function with error handling
        try:
            # Train model - this function handles MLflow and ClearML logging
            logger.info(f"Calling training.train_and_compare_fn with model_id={model_id}")
            result = training.train_and_compare_fn(model_id, processed_data_path)
            logger.info(f"Training completed successfully: {result}")
            
            # Push model ID for downstream tasks
            context['ti'].xcom_push(key='model_id', value=model_id)
            
            # Send success notification
            try:
                slack.post(f":white_check_mark: Model {model_id} trained successfully")
                logger.info("Sent training success notification to Slack")
            except Exception as e:
                logger.warning(f"Failed to send Slack notification: {str(e)}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            
            # Send failure notification
            try:
                slack.post(f":x: Model training failed: {str(e)}")
                logger.info("Sent training failure notification to Slack")
            except Exception as slack_error:
                logger.warning(f"Failed to send Slack notification: {str(slack_error)}")
                
            raise
            
    except Exception as e:
        logger.error(f"Error in train_model task: {str(e)}")
        raise

def check_for_drift(**context):
    """Check for data drift between reference and new data"""
    logger.info("Starting check_for_drift task")
    
    try:
        # Get processed data path with validation
        processed_data_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        
        if not processed_data_path:
            logger.error("Failed to get processed_data_path from previous task")
            raise ValueError("No processed data path provided from process_data task")
            
        # Verify the file exists and has content
        if not os.path.exists(processed_data_path):
            logger.error(f"Processed data file does not exist: {processed_data_path}")
            raise FileNotFoundError(f"Processed data file not found: {processed_data_path}")
            
        file_size = os.path.getsize(processed_data_path)
        logger.info(f"Found processed data file {processed_data_path} with size {file_size} bytes")
        
        if file_size == 0:
            logger.error(f"Processed data file is empty: {processed_data_path}")
            raise ValueError(f"Processed data file is empty: {processed_data_path}")
        
        # Load current data for logging purposes
        try:
            logger.info(f"Loading data from {processed_data_path}")
            current_data = pd.read_parquet(processed_data_path)
            logger.info(f"Loaded data with shape: {current_data.shape}")
        except Exception as e:
            logger.warning(f"Failed to load current data for logging: {str(e)}")
            logger.info("Will continue with drift detection using file path")
        
        # Initialize MLflow with robust error handling
        mlflow_run = None
        try:
            mlflow_uri = config.MLFLOW_URI
            mlflow_experiment = config.MLFLOW_EXPERIMENT
            
            logger.info(f"Connecting to MLflow at {mlflow_uri}")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(mlflow_experiment)
            
            mlflow_run = mlflow.start_run(run_name="drift_detection")
            run_id = mlflow_run.info.run_id
            logger.info(f"Started MLflow run with ID: {run_id}")
        except Exception as e:
            logger.warning(f"Unable to configure MLflow: {str(e)}. Will continue without MLflow tracking.")
            run_id = None
        
        # Start ClearML task
        clearml_task = None
        try:
            clearml_task = clearml_config.init_clearml("Drift_Detection")
            logger.info("Successfully initialized ClearML task")
        except Exception as e:
            logger.warning(f"Error initializing ClearML: {str(e)}. Will continue without ClearML tracking.")
        
        # Run drift detection with error handling
        try:
            logger.info(f"Calling drift.detect_data_drift with {processed_data_path}")
            drift_results = drift.detect_data_drift(processed_data_path)
            logger.info(f"Drift detection completed: {drift_results}")
        except Exception as e:
            logger.error(f"Error during drift detection: {str(e)}")
            raise
        
        # Get parameters for logging
        drift_threshold = float(Variable.get("DRIFT_THRESHOLD", default_var="0.1"))
        force_continue = Variable.get("FORCE_CONTINUE", default_var="false").lower() == "true"
        
        # Log to MLflow if available
        if mlflow_run:
            try:
                # Log parameters
                logger.info("Logging drift parameters to MLflow")
                mlflow.log_params({
                    "drift_threshold": drift_threshold,
                    "reference_data_date": datetime.now().strftime("%Y-%m-%d")
                })
                
                # Log metrics
                logger.info("Logging drift metrics to MLflow")
                mlflow.log_metric("overall_drift_score", drift_results["overall_drift_score"])
                mlflow.log_metric("drifted_features_count", len(drift_results["drifted_features"]))
                
                # Create and log report
                logger.info("Creating drift report artifact")
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                    report_path = temp_file.name
                    json.dump(drift_results, temp_file, indent=2)
                
                mlflow.log_artifact(report_path)
                logger.info(f"Logged drift report to MLflow: {report_path}")
                
                # Clean up temp file
                try:
                    os.unlink(report_path)
                except:
                    pass
            except Exception as e:
                logger.warning(f"Failed to log drift results to MLflow: {str(e)}")
        
        # Log to ClearML if available
        if clearml_task:
            try:
                logger.info("Logging drift metrics to ClearML")
                clearml_task.get_logger().report_scalar(
                    title="Drift Detection",
                    series="overall_drift_score",
                    value=drift_results["overall_drift_score"],
                    iteration=0
                )
                
                # Log individual feature scores
                for feature, score in drift_results.get("feature_drift_scores", {}).items():
                    clearml_task.get_logger().report_scalar(
                        title="Feature Drift",
                        series=feature,
                        value=score,
                        iteration=0
                    )
                logger.info("Successfully logged feature drift scores to ClearML")
            except Exception as e:
                logger.warning(f"Failed to log drift results to ClearML: {str(e)}")
                
            try:
                clearml_task.close()
                logger.info("Successfully closed ClearML task")
            except Exception as e:
                logger.warning(f"Failed to properly close ClearML task: {str(e)}")
        
        # Close MLflow run
        if mlflow_run:
            try:
                mlflow.end_run()
                logger.info("Successfully ended MLflow run")
            except Exception as e:
                logger.warning(f"Failed to properly end MLflow run: {str(e)}")
        
        # Check if drift exceeds threshold
        logger.info(f"Drift score: {drift_results['overall_drift_score']}, Threshold: {drift_threshold}, Force continue: {force_continue}")
        
        if drift_results["overall_drift_score"] > drift_threshold and not force_continue:
            # Send alert but continue workflow
            warning_msg = f"Data drift detected! Score: {drift_results['overall_drift_score']:.4f}, Threshold: {drift_threshold}"
            logger.warning(warning_msg)
            
            try:
                slack.post(f":warning: {warning_msg}")
                logger.info("Sent drift warning notification to Slack")
            except Exception as e:
                logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        # Push drift results for downstream tasks
        context['ti'].xcom_push(key='drift_results', value=drift_results)
        
        # Send success notification
        try:
            slack.post(f":white_check_mark: Drift detection completed. Score: {drift_results['overall_drift_score']:.4f}")
            logger.info("Sent drift success notification to Slack")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return drift_results
        
    except Exception as e:
        logger.error(f"Error in check_for_drift task: {str(e)}")
        
        # Send failure notification
        try:
            slack.post(f":x: Failed to detect drift: {str(e)}")
        except:
            pass
            
        raise

# Add a test function for checking environment and connectivity
def test_execution(**context):
    """Test task to verify environment settings and connections"""
    logger.info("Starting test_execution task")
    
    # Store test results
    test_results = {
        "status": "success",
        "tests": {},
        "errors": []
    }
    
    try:
        # Test 1: Check environment variables and Airflow variables
        logger.info("Testing environment variables and Airflow variables")
        test_results["tests"]["env_vars"] = {}
        
        # Check AWS region
        region = config.AWS_REGION
        test_results["tests"]["env_vars"]["aws_region"] = region
        logger.info(f"AWS region: {region}")
        
        # Check S3 bucket
        try:
            bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
            test_results["tests"]["env_vars"]["s3_bucket"] = bucket
            logger.info(f"S3 bucket: {bucket}")
        except Exception as e:
            error = f"Failed to get S3 bucket variable: {str(e)}"
            test_results["errors"].append(error)
            logger.error(error)
        
        # Test 2: Check AWS connectivity
        logger.info("Testing AWS connectivity")
        test_results["tests"]["aws_conn"] = {}
        
        try:
            # Create S3 client
            s3_client = boto3.client('s3', region_name=region)
            
            # List buckets
            response = s3_client.list_buckets()
            buckets = [b['Name'] for b in response['Buckets']]
            test_results["tests"]["aws_conn"]["buckets_accessible"] = True
            test_results["tests"]["aws_conn"]["buckets_count"] = len(buckets)
            logger.info(f"Successfully listed {len(buckets)} S3 buckets")
            
            # Check if our bucket exists
            bucket_exists = bucket in buckets
            test_results["tests"]["aws_conn"]["bucket_exists"] = bucket_exists
            logger.info(f"Bucket {bucket} exists: {bucket_exists}")
            
            if bucket_exists:
                # List objects in bucket (limited to 10)
                objects = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=10)
                object_count = objects.get('KeyCount', 0)
                test_results["tests"]["aws_conn"]["objects_count"] = object_count
                logger.info(f"Found {object_count} objects in bucket {bucket}")
        except Exception as e:
            error = f"AWS connectivity test failed: {str(e)}"
            test_results["errors"].append(error)
            logger.error(error)
            test_results["tests"]["aws_conn"]["success"] = False
        else:
            test_results["tests"]["aws_conn"]["success"] = True
        
        # Test 3: Check MLflow connectivity
        logger.info("Testing MLflow connectivity")
        test_results["tests"]["mlflow_conn"] = {}
        
        try:
            mlflow_uri = config.MLFLOW_URI
            test_results["tests"]["mlflow_conn"]["uri"] = mlflow_uri
            
            # Try to connect to MLflow
            mlflow.set_tracking_uri(mlflow_uri)
            client = mlflow.tracking.MlflowClient()
            
            # List experiments
            experiments = client.list_experiments()
            test_results["tests"]["mlflow_conn"]["experiments_count"] = len(experiments)
            logger.info(f"Successfully connected to MLflow at {mlflow_uri}")
            logger.info(f"Found {len(experiments)} experiments")
        except Exception as e:
            error = f"MLflow connectivity test failed: {str(e)}"
            test_results["errors"].append(error)
            logger.error(error)
            test_results["tests"]["mlflow_conn"]["success"] = False
        else:
            test_results["tests"]["mlflow_conn"]["success"] = True
        
        # Test 4: Test file system access
        logger.info("Testing file system access")
        test_results["tests"]["filesystem"] = {}
        
        try:
            # Create a test file
            test_dir = tempfile.mkdtemp()
            test_file = os.path.join(test_dir, "test_file.txt")
            
            with open(test_file, 'w') as f:
                f.write("This is a test file")
            
            # Read the test file
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Clean up
            os.remove(test_file)
            os.rmdir(test_dir)
            
            test_results["tests"]["filesystem"]["write_success"] = True
            test_results["tests"]["filesystem"]["read_success"] = True
            logger.info("Successfully tested file system access")
        except Exception as e:
            error = f"File system access test failed: {str(e)}"
            test_results["errors"].append(error)
            logger.error(error)
            test_results["tests"]["filesystem"]["success"] = False
        else:
            test_results["tests"]["filesystem"]["success"] = True
        
        # Overall status
        if test_results["errors"]:
            test_results["status"] = "warning"
            logger.warning(f"Test completed with {len(test_results['errors'])} warnings/errors")
        else:
            logger.info("All tests completed successfully")
        
        # Push test results to XCom
        context['ti'].xcom_push(key='test_results', value=test_results)
        
        return test_results
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        test_results["status"] = "error"
        test_results["errors"].append(str(e))
        
        # Push test results to XCom even if failed
        context['ti'].xcom_push(key='test_results', value=test_results)
        
        raise

# Define tasks
download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_data,
    provide_context=True,
    retries=3,
    retry_delay=timedelta(minutes=3),
    execution_timeout=timedelta(minutes=30),
    dag=dag,
)

verify_task = PythonOperator(
    task_id='verify_file_persistence',
    python_callable=verify_file_persistence,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=timedelta(minutes=15),
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=5),
    execution_timeout=timedelta(hours=1),
    dag=dag,
)

quality_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=3),
    execution_timeout=timedelta(minutes=45),
    dag=dag,
)

drift_task = PythonOperator(
    task_id='check_for_drift',
    python_callable=check_for_drift,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=3),
    execution_timeout=timedelta(minutes=45),
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    retries=1,
    retry_delay=timedelta(minutes=10),
    execution_timeout=timedelta(hours=4),
    dag=dag,
)

# Add test task
test_task = PythonOperator(
    task_id='test_execution',
    python_callable=test_execution,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=timedelta(minutes=10),
    dag=dag,
)

# Define workflow
test_task >> download_task >> verify_task >> process_task >> [quality_task, drift_task] >> train_task 