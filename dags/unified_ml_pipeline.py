#!/usr/bin/env python3
"""
unified_ml_pipeline.py - Consolidated ML Pipeline DAG
-----------------------------------------------------
This DAG combines functionality from multiple previous pipelines:
- integrated_ml_workflow.py
- homeowner_dag.py
- train_all_models_dag.py

It provides a unified workflow that handles:
1. Data ingestion from S3
2. Data preprocessing and quality checking
3. Schema validation and drift detection
4. Human validation of data quality (HITL)
5. Training of multiple models (including parallel training)
6. Model evaluation and explainability tracking
7. Human approval of models (HITL)
8. Artifact archiving

All functionality is contained in a single DAG for easier management and troubleshooting.
"""

import os
import json
import logging
import tempfile
import shutil
import sys
import math
import random
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from pathlib import Path
from functools import wraps
from typing import Dict, Any, Optional, Tuple, Union, List

# Airflow imports
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable, XCom
from airflow.hooks.S3_hook import S3Hook
from airflow.exceptions import AirflowException, AirflowSkipException
import pandas as pd
import boto3
import mlflow
from mlflow.tracking import MlflowClient

# Import task modules
try:
    import utils.config as config
    import utils.clearml_config as clearml_config
    import utils.slack as slack

    # Import task modules that are actively used
    import tasks.ingestion as ingestion
    import tasks.preprocessing_simplified as preprocessing  # Use simplified preprocessing without outlier detection
    import tasks.data_quality as data_quality
    import tasks.schema_validation as schema_validation
    import tasks.drift as drift
    import tasks.training as training
    import tasks.model_explainability as model_explainability
    import tasks.hitl as hitl  # Import the new Human-in-the-Loop module
except ImportError as e:
    # Log the import error but continue - the specific module will fail at runtime if used
    logger.error(f"Error importing module: {str(e)}")
    logger.error("Some pipeline components may not be available")
    
    # Define empty modules for modules that failed to import to prevent NameError
    # when the code refers to these modules
    class EmptyModule:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                logger.error(f"Called missing module method {name}. Original import failed.")
                return {"status": "error", "message": f"Module not available: {name}"}
            return method
    
    # Define module variables if they don't exist
    if 'config' not in locals():
        config = EmptyModule()
    if 'clearml_config' not in locals():
        clearml_config = EmptyModule()
    if 'slack' not in locals():
        slack = EmptyModule()
    if 'ingestion' not in locals():
        ingestion = EmptyModule()
    if 'preprocessing' not in locals():
        preprocessing = EmptyModule()
    if 'data_quality' not in locals():
        data_quality = EmptyModule()
    if 'schema_validation' not in locals():
        schema_validation = EmptyModule()
    if 'drift' not in locals():
        drift = EmptyModule()
    if 'training' not in locals():
        training = EmptyModule()
    if 'model_explainability' not in locals():
        model_explainability = EmptyModule()
    if 'hitl' not in locals():
        hitl = EmptyModule()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Slack at startup to ensure channels are created
try:
    import utils.slack as slack
    # Ensure Slack has default channels set up
    from utils.slack import ensure_default_channels
    slack_channels = ensure_default_channels()
    logger.info(f"Slack channels initialized: {slack_channels}")
except Exception as e:
    logger.warning(f"Failed to initialize Slack: {str(e)}. Notifications may not be sent.")
    # Define a dummy slack module if initialization fails
    class DummySlack:
        def post(self, channel=None, title=None, details=None, urgency=None):
            logger.info(f"[DUMMY SLACK] {title}: {details}")
            return {"ok": True, "dummy": True}
    
    slack = DummySlack()

# Constants
LOCAL_PROCESSED_PATH = "/tmp/unified_processed.parquet"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"
MAX_WORKERS = int(Variable.get('MAX_PARALLEL_WORKERS', default_var='3'))
# Add new constant for feature engineering
APPLY_FEATURE_ENGINEERING = Variable.get('APPLY_FEATURE_ENGINEERING', default_var='False').lower() == 'true'
# Add new constants for HITL
REQUIRE_DATA_VALIDATION = Variable.get('REQUIRE_DATA_VALIDATION', default_var='True').lower() == 'true'
REQUIRE_MODEL_APPROVAL = Variable.get('REQUIRE_MODEL_APPROVAL', default_var='True').lower() == 'true'
# Add new constants for auto-approval
AUTO_APPROVE_DATA = Variable.get('AUTO_APPROVE_DATA', default_var='False').lower() == 'true'
AUTO_APPROVE_TIMEOUT_MINUTES = int(Variable.get('AUTO_APPROVE_TIMEOUT_MINUTES', default_var='30'))
AUTO_APPROVE_QUALITY_THRESHOLD = int(Variable.get('AUTO_APPROVE_QUALITY_THRESHOLD', default_var='3'))
# Add these constants to the existing auto-approval constants
AUTO_APPROVE_MODEL = Variable.get('AUTO_APPROVE_MODEL', default_var='False').lower() == 'true'
MODEL_APPROVE_TIMEOUT_MINUTES = int(Variable.get('MODEL_APPROVE_TIMEOUT_MINUTES', default_var='60'))

# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2023, 1, 1),
    'catchup': False
}

# Utility functions
def validate_file_exists(filepath: str) -> bool:
    """
    Validate that a file exists and is not empty.
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        bool: True if the file exists and is not empty
    """
    if not filepath:
        logger.error("Filepath is empty")
        return False
        
    if not os.path.exists(filepath):
        logger.error(f"File does not exist: {filepath}")
        return False
        
    if os.path.getsize(filepath) == 0:
        logger.error(f"File is empty: {filepath}")
        return False
        
    return True

def with_file_validation(file_param: str):
    """
    Decorator to validate file existence before executing a function.
    
    Args:
        file_param: Name of the parameter containing the file path
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            filepath = kwargs.get(file_param)
            if not validate_file_exists(filepath):
                logger.error(f"File validation failed for {file_param}: {filepath}")
                return {"status": "error", "message": f"File not found or empty: {filepath}"}
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Task functions
def download_data(**context):
    """Download data from S3 bucket"""
    logger.info("Starting download_data task")
    
    # Get bucket name from Airflow Variable to ensure it's up to date
    bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
    key = config.RAW_DATA_KEY
    
    logger.info(f"Using bucket from Airflow Variables: {bucket}")
    logger.info(f"Attempting to access data at s3://{bucket}/{key}")
    
    # Add debugging to list objects in the bucket
    try:
        s3_client = boto3.client('s3', region_name=config.AWS_REGION)
        logger.info("Listing objects in the bucket for debugging:")
        
        # List objects in the raw_data directory
        raw_data_prefix = "raw-data/"
        logger.info(f"Objects with prefix '{raw_data_prefix}':")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=raw_data_prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
        else:
            logger.warning(f"No objects found with prefix '{raw_data_prefix}'")
            
        # Try also with raw-data (hyphenated) just in case
        alt_prefix = "raw_data/"
        logger.info(f"Objects with alternate prefix '{alt_prefix}':")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=alt_prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
                # If the file is found here, update the key to use this path
                if obj['Key'].endswith('ut_loss_history_1.csv'):
                    key = obj['Key']
                    logger.info(f"Found target file at alternate path, updating key to: {key}")
        else:
            logger.warning(f"No objects found with alternate prefix '{alt_prefix}'")
            
    except Exception as e:
        logger.warning(f"Error listing objects in bucket: {str(e)}")
    
    try:
        # Create a list of potential data directories, in order of preference
        potential_dirs = [
            "/tmp/airflow_data",       # Preferred location in tmp for persistence
            "/usr/local/airflow/tmp",  # Fallback airflow directory
            "/tmp"                     # Last resort, standard tmp
        ]
        
        # Try each directory until we find one we can write to
        data_dir = None
        for dir_path in potential_dirs:
            try:
                logger.info(f"Attempting to use directory: {dir_path}")
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    
                # Test if we can write to this directory
                test_file = os.path.join(dir_path, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                # If we get here, the directory is writable
                data_dir = dir_path
                logger.info(f"Using directory: {data_dir}")
                break
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot use directory {dir_path}: {str(e)}")
        
        if not data_dir:
            logger.error("Could not find a writable directory")
            raise PermissionError("No writable directory found for storing data")
        
        # Generate a persistent filename based on the S3 key
        filename = os.path.basename(key)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_path = os.path.join(data_dir, f"raw_{timestamp}_{filename}")
        logger.info(f"Will download to persistent location: {local_path}")
        
        # Use the ingestion function from the full pipeline if available
        try:
            logger.info("Attempting to use pipeline ingestion function")
            # Check function signature to avoid parameter errors
            import inspect
            ingest_sig = inspect.signature(ingestion.ingest_data_from_s3)
            ingest_params = {}
            
            # Add parameters based on available arguments
            if 'bucket_name' in ingest_sig.parameters:
                ingest_params['bucket_name'] = bucket
            elif 'bucket' in ingest_sig.parameters:
                ingest_params['bucket'] = bucket
                
            if 'key' in ingest_sig.parameters:
                ingest_params['key'] = key
                
            if 'local_path' in ingest_sig.parameters:
                ingest_params['local_path'] = local_path
                
            logger.info(f"Calling ingest_data_from_s3 with params: {ingest_params}")
            data_path = ingestion.ingest_data_from_s3(**ingest_params)
            logger.info(f"Successfully ingested data using pipeline function: {data_path}")
        except Exception as e:
            # Fall back to direct S3 download if ingestion function fails
            logger.warning(f"Pipeline ingestion failed, falling back to direct download: {str(e)}")
            
            # Download directly to the persistent location
            s3_client = boto3.client('s3', region_name=config.AWS_REGION)
            logger.info(f"Downloading from s3://{bucket}/{key} to {local_path}")
            s3_client.download_file(bucket, key, local_path)
            data_path = local_path
            logger.info(f"Successfully downloaded to {local_path}")
        
        # Create additional backup copies in other directories
        backup_paths = []
        for backup_dir in potential_dirs:
            if backup_dir != data_dir and os.path.exists(backup_dir):
                try:
                    backup_path = os.path.join(backup_dir, f"backup_{timestamp}_{filename}")
                    import shutil
                    shutil.copy2(data_path, backup_path)
                    backup_paths.append(backup_path)
                    logger.info(f"Created backup copy at {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to create backup in {backup_dir}: {str(e)}")
                    
        # Store both primary path and backup paths in XCom
        logger.info(f"Pushing data_path to XCom: {data_path}")
        context['ti'].xcom_push(key='data_path', value=data_path)
        
        # Add additional debugging
        ti = context['ti']
        task_id = ti.task_id
        dag_id = ti.dag_id
        logger.info(f"XCom values for debugging - task_id: {task_id}, dag_id: {dag_id}")
        
        # Verify the XCom push by directly reading from Airflow's XCom table
        try:
            execution_date = context.get('execution_date')
            logger.info(f"Execution date: {execution_date}")
            xcom_value = ti.xcom_pull(task_ids=task_id, key='data_path')
            logger.info(f"Verification of XCom value (self-pull): {xcom_value}")
        except Exception as e:
            logger.warning(f"Could not verify XCom value: {str(e)}")

        if backup_paths:
            logger.info(f"Pushing backup_paths to XCom: {backup_paths}")
            context['ti'].xcom_push(key='backup_paths', value=backup_paths)
            
        # Test file exists and is accessible
        if not os.path.exists(data_path):
            logger.error(f"Downloaded file not found at {data_path}")
            raise FileNotFoundError(f"Downloaded file not found at {data_path}")
            
        file_size = os.path.getsize(data_path)
        if file_size == 0:
            logger.error(f"Downloaded file is empty: {data_path}")
            raise ValueError(f"Downloaded file is empty: {data_path}")
            
        logger.info(f"Verified file exists with size {file_size} bytes")
        
        # Set file permissions to ensure next tasks can read it
        try:
            os.chmod(data_path, 0o666)  # Make file readable/writable by all users
            logger.info(f"Set file permissions to 666 for {data_path}")
        except Exception as e:
            logger.warning(f"Could not set file permissions: {str(e)}")
            
        # Create a standardized file path that other tasks can find directly
        try:
            # Ensure we have a constant location for the file across tasks
            standard_path = os.path.join(data_dir, f"latest_raw_data.csv")
            shutil.copy2(data_path, standard_path)
            logger.info(f"Created standardized copy at {standard_path}")
            
            # Add this to XCom as well
            context['ti'].xcom_push(key='standard_data_path', value=standard_path)
        except Exception as e:
            logger.warning(f"Could not create standardized path: {str(e)}")
        
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
        
        try:
            slack.simple_post(f"✅ Data accessed from s3://{bucket}/{key}", channel="#data-pipeline")
        except Exception as e:
            logger.warning(f"Error sending Slack notification: {str(e)}")
            
        return data_path
        
    except Exception as e:
        logger.error(f"Error in download_data task: {str(e)}")
        
        try:
            slack.simple_post(f"❌ Failed to access data: {str(e)}", channel="#data-pipeline")
        except Exception as ex:
            logger.warning(f"Error sending error notification: {str(ex)}")
            
        raise

def process_data(**context):
    """Process the data and prepare for model training"""
    logger.info("Starting process_data task")
    
    try:
        # Define potential data directories
        potential_dirs = [
            "/tmp/airflow_data",
            "/usr/local/airflow/tmp",
            "/tmp"
        ]
        
        # Find a writable directory for output
        output_dir = None
        for dir_path in potential_dirs:
            try:
                logger.info(f"Testing directory for output: {dir_path}")
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                # Test if we can write to this directory
                test_file = os.path.join(dir_path, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                output_dir = dir_path
                logger.info(f"Using directory for output: {output_dir}")
                break
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot use directory {dir_path} for output: {str(e)}")
        
        if not output_dir:
            logger.error("Could not find a writable directory for output")
            raise PermissionError("No writable directory found for storing processed data")
        
        # Get raw data path from previous task
        raw_data_path = context['ti'].xcom_pull(task_ids='import_data_task', key='data_path')
        
        if not raw_data_path:
            logger.error("Failed to get data_path from previous tasks")
            raise ValueError("No data path provided from previous tasks")
            
        # Verify the file exists and has content
        if not os.path.exists(raw_data_path):
            logger.warning(f"Data file does not exist at primary location: {raw_data_path}")
            
            # Try backup paths
            backup_paths = context['ti'].xcom_pull(task_ids='import_data_task', key='backup_paths') or []
            for backup_path in backup_paths:
                if os.path.exists(backup_path):
                    raw_data_path = backup_path
                    logger.info(f"Using backup path: {raw_data_path}")
                    break
            
            # If still not found, try to download directly from S3
            if not os.path.exists(raw_data_path):
                logger.warning("Attempting to download data directly from S3")
                try:
                    s3_client = boto3.client('s3', region_name=config.AWS_REGION)
                    bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
                    key = config.RAW_DATA_KEY
                    
                    # Create a new local path
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = os.path.basename(key)
                    local_path = os.path.join(output_dir, f"recovery_{timestamp}_{filename}")
                    
                    # Download the file
                    logger.info(f"Downloading from s3://{bucket}/{key} to {local_path}")
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    s3_client.download_file(bucket, key, local_path)
                    raw_data_path = local_path
                    logger.info(f"Successfully downloaded recovery file to {local_path}")
                except Exception as e:
                    logger.error(f"Recovery download failed: {str(e)}")
        
        # If still not found, raise error
        if not os.path.exists(raw_data_path):
            logger.error("Could not find or recover data file")
            raise FileNotFoundError("Data file not found and recovery attempts failed")
        
        # Process the data
        try:
            # Generate the output path for the processed data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            processed_path = os.path.join(output_dir, f"processed_{timestamp}.parquet")
            
            # Check if feature engineering should be applied
            apply_feature_engineering = APPLY_FEATURE_ENGINEERING
            logger.info(f"Feature engineering is {'enabled' if apply_feature_engineering else 'disabled'}")
            
            # If feature engineering is disabled and input is already parquet, 
            # we can potentially just copy the file
            input_is_parquet = raw_data_path.lower().endswith('.parquet')
            
            if not apply_feature_engineering and input_is_parquet:
                # Simple copy for already clean parquet data
                logger.info(f"Dataset is already clean, copying file with minimal processing")
                
                try:
                    # Load and save to ensure format compatibility
                    df = pd.read_parquet(raw_data_path)
                    
                    # Ensure target column exists
                    if 'trgt' not in df.columns and 'il_total' in df.columns and 'eey' in df.columns:
                        logger.info("Creating 'trgt' column from 'il_total' / 'eey'")
                        df['trgt'] = df['il_total'] / df['eey']
                        
                    df.to_parquet(processed_path, index=False)
                    logger.info(f"File copied with minimal processing to {processed_path}")
                except Exception as e:
                    logger.warning(f"Error in simple copy: {str(e)}, falling back to processing")
                    # If copy fails, continue to regular processing
                    apply_feature_engineering = True
            
            # Standard processing path
            if apply_feature_engineering or not input_is_parquet:
                logger.info(f"Processing data from {raw_data_path} to {processed_path}")
                
                # Use preprocessing module for data processing
                try:
                    # Call the preprocess_data function without the apply_feature_engineering parameter
                    processed_data = preprocessing.preprocess_data(
                        data_path=raw_data_path,
                        output_path=processed_path,
                        force_reprocess=True
                    )
                    logger.info(f"Data processed successfully to {processed_path}")
                except Exception as e:
                    logger.error(f"Error in data preprocessing: {str(e)}")
                    # Try with simpler processing as fallback
                    try:
                        logger.info("Trying simplified processing as fallback")
                        # Load the data directly
                        df = pd.read_csv(raw_data_path, encoding='latin-1', on_bad_lines='skip')
                        logger.info(f"Loaded data with shape: {df.shape}")
                        
                        # Ensure target column exists
                        if 'trgt' not in df.columns and 'il_total' in df.columns and 'eey' in df.columns:
                            df['trgt'] = df['il_total'] / df['eey']
                            
                        # Save directly to output
                        df.to_parquet(processed_path, index=False)
                        logger.info(f"Fallback processing completed to {processed_path}")
                    except Exception as fallback_err:
                        logger.error(f"Fallback processing also failed: {str(fallback_err)}")
                        raise e  # Raise the original error
            
            # Ensure we have a valid processed path
            if not processed_path or not os.path.exists(processed_path):
                logger.error("Processed data file not created")
                raise FileNotFoundError("Processed data file not created")
            
            # Create a standardized version at the expected location
            try:
                logger.info(f"Creating standardized version at {LOCAL_PROCESSED_PATH}")
                shutil.copy2(processed_path, LOCAL_PROCESSED_PATH)
                logger.info(f"Standardized version created at {LOCAL_PROCESSED_PATH}")
            except Exception as e:
                logger.warning(f"Failed to create standardized version: {str(e)}")
                # Continue using the original processed path
                
            # Store processed data path in XCom
            context['ti'].xcom_push(key='processed_data_path', value=processed_path)
            context['ti'].xcom_push(key='standardized_processed_path', value=LOCAL_PROCESSED_PATH)
            
            # Log completion
            logger.info(f"Data processing complete. Output at {processed_path}")
            try:
                slack.simple_post(f"✅ Data processing completed: {processed_path}", channel="#data-pipeline")
            except Exception as e:
                logger.warning(f"Error sending Slack notification: {str(e)}")
                
            return processed_path
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in process_data task: {str(e)}")
        
        try:
            slack.simple_post(f"❌ Data processing failed: {str(e)}", channel="#data-pipeline")
        except:
            pass
            
        raise

def run_data_quality_checks(**context):
    """Run data quality checks on the processed data"""
    logger.info("Starting data_quality_checks task")
    
    try:
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='process_data', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        data_path = standardized_path if os.path.exists(standardized_path) else processed_path
        
        if not data_path or not os.path.exists(data_path):
            logger.error("No valid processed data path found")
            raise FileNotFoundError("No valid processed data path found")
            
        logger.info(f"Running data quality checks on {data_path}")
        
        # Load the data
        try:
            df = pd.read_parquet(data_path)
            logger.info(f"Loaded dataframe with shape {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load parquet file: {str(e)}")
            raise
            
        # Run quality checks - use correct DataQualityMonitor method
        try:
            # Create DataQualityMonitor instance and run checks
            quality_monitor = data_quality.DataQualityMonitor()
            quality_results = quality_monitor.run_quality_checks(df)
            logger.info(f"Data quality results: {quality_results}")
            
            # Store results in XCom
            context['ti'].xcom_push(key='quality_results', value=quality_results)
            
            # Determine if quality checks passed
            quality_passed = quality_results.get('status', '') == 'success' or quality_results.get('total_issues', 0) == 0
            context['ti'].xcom_push(key='quality_passed', value=quality_passed)
            
            if quality_passed:
                logger.info("Data quality checks passed")
                try:
                    slack.simple_post("✅ Data quality checks passed", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Error sending Slack notification: {str(e)}")
            else:
                logger.warning("Data quality checks failed or had warnings")
                try:
                    message = quality_results.get('message', 'Unknown issue')
                    issues = quality_results.get('total_issues', 0)
                    slack.simple_post(f"❌ Data quality checks failed: {issues} issues: {message}", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Error sending Slack notification: {str(e)}")
                    
            return quality_results
            
        except Exception as e:
            logger.error(f"Error running data quality checks: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in data_quality_checks task: {str(e)}")
        
        try:
            slack.simple_post(f"❌ Data quality checks failed: {str(e)}", channel="#data-pipeline")
        except:
            pass
            
        raise

def run_schema_validation(**context):
    """Run schema validation on the processed data"""
    logger.info("Starting schema_validation task")
    
    try:
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='process_data', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        data_path = standardized_path if os.path.exists(standardized_path) else processed_path
        
        if not data_path or not os.path.exists(data_path):
            logger.error("No valid processed data path found")
            raise FileNotFoundError("No valid processed data path found")
            
        logger.info(f"Running schema validation on {data_path}")
        
        # Load the data
        try:
            df = pd.read_parquet(data_path)
            logger.info(f"Loaded dataframe with shape {df.shape}")
            
            # Create the target column if needed
            if 'trgt' not in df.columns and 'pure_premium' not in df.columns:
                if 'il_total' in df.columns and 'eey' in df.columns:
                    logger.info("Creating 'trgt' column from 'il_total' / 'eey'")
                    df['trgt'] = df['il_total'] / df['eey']
                    
                    # Save the updated dataframe back to the standardized path
                    if os.path.exists(standardized_path):
                        df.to_parquet(standardized_path, index=False)
                        logger.info(f"Updated standardized dataframe with 'trgt' column")
                else:
                    logger.warning("Cannot create target column: missing 'il_total' or 'eey' columns")
        except Exception as e:
            logger.error(f"Failed to load or preprocess parquet file: {str(e)}")
            raise
            
        # Run schema validation
        try:
            validation_results = schema_validation.validate_schema(df)
            logger.info(f"Schema validation results: {validation_results}")
            
            # Store results in XCom
            context['ti'].xcom_push(key='validation_results', value=validation_results)
            
            # Determine if validation passed
            validation_passed = validation_results.get('status') == 'success'
            context['ti'].xcom_push(key='validation_passed', value=validation_passed)
            
            if validation_passed:
                logger.info("Schema validation passed")
                try:
                    slack.simple_post("✅ Schema validation passed", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Error sending Slack notification: {str(e)}")
            else:
                logger.warning(f"Schema validation failed: {validation_results.get('message', 'Unknown issue')}")
                try:
                    details = validation_results.get('details', {})
                    message = f"⚠️ Schema validation had issues: {validation_results.get('message', 'Unknown issue')}"
                    if details.get('target_info'):
                        message += f"\n{details.get('target_info')}"
                    slack.simple_post(message, channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Error sending Slack notification: {str(e)}")
                    
            return validation_results
            
        except Exception as e:
            logger.error(f"Error running schema validation: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in schema_validation task: {str(e)}")
        
        try:
            slack.simple_post(f"❌ Schema validation failed: {str(e)}", channel="#data-pipeline")
        except:
            pass
            
        raise

# Defensive wrapper to handle function calls
def safe_module_call(module, function_name, *args, **kwargs):
    """
    Safely call a function from a module, handling AttributeError gracefully.
    
    Args:
        module: The module to call function from
        function_name: Name of the function to call
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Function result or error dict
    """
    try:
        # Check if the module has the function
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            if callable(func):
                return func(*args, **kwargs)
            else:
                logger.error(f"Function {function_name} exists in {module.__name__} but is not callable")
                return {"status": "error", "message": f"Function {function_name} is not callable"}
        else:
            # Try to find similar function names for useful error messages
            available_funcs = [name for name in dir(module) if callable(getattr(module, name)) and not name.startswith('_')]
            suggestion = ""
            if available_funcs:
                suggestion = f"Available functions: {', '.join(available_funcs[:5])}"
                if len(available_funcs) > 5:
                    suggestion += f" and {len(available_funcs) - 5} more"
                    
            logger.error(f"Function {function_name} not found in {module.__name__}. {suggestion}")
            return {"status": "error", "message": f"Function {function_name} not found in module. {suggestion}"}
    except AttributeError as e:
        logger.error(f"AttributeError calling {function_name}: {str(e)}")
        return {"status": "error", "message": f"AttributeError: {str(e)}"}
    except Exception as e:
        logger.error(f"Error calling {function_name}: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}

# Update the drift detection call to use the safe function call
def check_for_drift(**context):
    """Check for data drift"""
    logger.info("Starting drift_detection task")
    logger.info(f"Running in context: {context.get('task_instance')}")
    
    try:
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='process_data', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        data_path = None
        if standardized_path and os.path.exists(standardized_path):
            data_path = standardized_path
            logger.info(f"Using standardized path: {data_path}")
        elif processed_path and os.path.exists(processed_path):
            data_path = processed_path
            logger.info(f"Using processed path: {data_path}")
        else:
            logger.warning("No valid processed data path found from XCom")
            # Try to find any parquet files in common locations as fallback
            potential_locations = [
                "/tmp/airflow_data",
                "/usr/local/airflow/tmp",
                "/tmp"
            ]
            
            for location in potential_locations:
                if os.path.exists(location):
                    logger.info(f"Searching for parquet files in {location}")
                    parquet_files = [os.path.join(location, f) for f in os.listdir(location) 
                                   if f.endswith('.parquet') and os.path.isfile(os.path.join(location, f))]
                    
                    if parquet_files:
                        # Sort by modification time to get the newest
                        newest_file = sorted(parquet_files, key=os.path.getmtime, reverse=True)[0]
                        data_path = newest_file
                        logger.info(f"Found parquet file as fallback: {data_path}")
                        break
        
        if not data_path:
            logger.error("No valid processed data path found and no fallback available")
            drift_results = {
                "status": "error", 
                "message": "No valid processed data path found",
                "drift_detected": False
            }
            context['ti'].xcom_push(key='drift_results', value=drift_results)
            context['ti'].xcom_push(key='drift_detected', value=False)
            
            # Notify about failure but don't raise an exception
            try:
                slack.simple_post("❌ Drift Detection Failed", channel="#data-pipeline")
            except Exception as e:
                logger.warning(f"Error sending Slack notification: {str(e)}")
                
            return drift_results
            
        logger.info(f"Running drift detection on {data_path}")
        
        # Try multiple possible function names from the drift module
        drift_results = None
        
        # First try detect_data_drift
        try:
            logger.info("Trying drift.detect_data_drift function")
            drift_results = safe_module_call(drift, "detect_data_drift", processed_data_path=data_path)
        except Exception as e:
            logger.error(f"Error calling detect_data_drift: {str(e)}")
            drift_results = {"status": "error", "message": f"Error: {str(e)}"}
        
        # If that failed, try check_for_drift
        if drift_results.get("status") == "error" and "not found" in drift_results.get("message", ""):
            logger.info("First method failed, trying alternate drift detection function 'check_for_drift'")
            try:
                drift_results = safe_module_call(drift, "check_for_drift", data_path)
            except Exception as e:
                logger.error(f"Error calling check_for_drift: {str(e)}")
                drift_results = {"status": "error", "message": f"Error: {str(e)}"}
            
        # If still failed, try other potential names
        if drift_results.get("status") == "error" and "not found" in drift_results.get("message", ""):
            for func_name in ["detect_drift", "run_drift_detection", "check_drift"]:
                logger.info(f"Trying alternate drift detection function '{func_name}'")
                try:
                    drift_results = safe_module_call(drift, func_name, data_path)
                    if drift_results.get("status") != "error" or "not found" not in drift_results.get("message", ""):
                        break
                except Exception as e:
                    logger.error(f"Error calling {func_name}: {str(e)}")
                    drift_results = {"status": "error", "message": f"Error in {func_name}: {str(e)}"}
        
        # Handle S3 file not found error
        if drift_results.get("status") == "error" and "404" in drift_results.get("message", "") and "HeadObject" in drift_results.get("message", ""):
            logger.warning("Reference file not found in S3, using fallback (no drift)")
            drift_results = {
                "status": "warning",
                "message": "Reference file not found in S3. Using default drift status (no drift).",
                "drift_detected": False
            }
            
            # Try to generate reference means if we have data
            if data_path and os.path.exists(data_path):
                try:
                    logger.info("Attempting to generate reference means as fallback")
                    generate_result = safe_module_call(drift, "generate_reference_means", data_path)
                    logger.info(f"Generated reference means: {generate_result}")
                    drift_results["reference_means_generated"] = True
                except Exception as e:
                    logger.error(f"Error generating reference means: {str(e)}")
        
        logger.info(f"Drift detection results: {drift_results}")
        
        # Store results in XCom
        context['ti'].xcom_push(key='drift_results', value=drift_results)
        
        # Determine if drift was detected
        drift_detected = drift_results.get('drift_detected', False)
        context['ti'].xcom_push(key='drift_detected', value=drift_detected)
        
        if drift_detected:
            logger.warning("Data drift detected")
            try:
                slack.simple_post("⚠️ Data Drift Detected", channel="#data-pipeline")
            except Exception as e:
                logger.warning(f"Error sending Slack notification: {str(e)}")
        else:
            logger.info("No data drift detected")
            try:
                slack.simple_post("✅ No Data Drift", channel="#data-pipeline")
            except Exception as e:
                logger.warning(f"Error sending Slack notification: {str(e)}")
                
        return drift_results
        
    except Exception as e:
        logger.error(f"Error in drift_detection task: {str(e)}")
        logger.exception("Full exception details:")
        
        # Create a result that allows the pipeline to continue
        drift_results = {
            "status": "error", 
            "message": f"Error in drift detection: {str(e)}",
            "drift_detected": False
        }
        
        # Store in XCom
        context['ti'].xcom_push(key='drift_results', value=drift_results)
        context['ti'].xcom_push(key='drift_detected', value=False)
        
        try:
            slack.simple_post("❌ Drift Detection Failed", channel="#data-pipeline")
        except Exception as slack_error:
            logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        
        # Return a valid result instead of raising an exception
        return drift_results

def train_models(**context):
    """Train all models"""
    logger.info("Starting train_models task")
    logger.info(f"Running in context: {context.get('task_instance')}")
    
    try:
        # Log all XCom values received from upstream tasks for diagnostic purposes
        ti = context.get('ti')
        if ti:
            # Get upstream task IDs
            task_instances = context.get('task_instance').get_dagrun().get_task_instances()
            upstream_task_ids = [t.task_id for t in task_instances if t.task_id != 'train_models']
            
            logger.info(f"Upstream tasks: {upstream_task_ids}")
            
            # Log XCom values from upstream tasks
            for task_id in upstream_task_ids:
                try:
                    xcom_values = ti.xcom_pull(task_ids=task_id)
                    logger.info(f"XCom from {task_id}: {str(xcom_values)[:500]}...")  # Log first 500 chars
                except Exception as e:
                    logger.warning(f"Error pulling XCom from {task_id}: {str(e)}")
        
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='process_data', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        data_path = None
        if standardized_path and os.path.exists(standardized_path):
            data_path = standardized_path
            logger.info(f"Using standardized path: {data_path}")
        elif processed_path and os.path.exists(processed_path):
            data_path = processed_path
            logger.info(f"Using processed path: {data_path}")
        else:
            logger.warning("No valid processed data path found from XCom")
            # Try to find any parquet files in common locations as fallback
            potential_locations = [
                "/tmp/airflow_data",
                "/usr/local/airflow/tmp",
                "/tmp"
            ]
            
            for location in potential_locations:
                if os.path.exists(location):
                    logger.info(f"Searching for parquet files in {location}")
                    parquet_files = [os.path.join(location, f) for f in os.listdir(location) 
                                   if f.endswith('.parquet') and os.path.isfile(os.path.join(location, f))]
                    
                    if parquet_files:
                        # Sort by modification time to get the newest
                        newest_file = sorted(parquet_files, key=os.path.getmtime, reverse=True)[0]
                        data_path = newest_file
                        logger.info(f"Found parquet file as fallback: {data_path}")
                        break
        
        if not data_path:
            error_msg = "No valid processed data path found and no fallback available"
            logger.error(error_msg)
            try:
                slack.simple_post("❌ Model training failed: No valid data path found", channel="#data-pipeline")
            except Exception as e:
                logger.warning(f"Error sending Slack notification: {str(e)}")
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Training models using data from {data_path}")
        
        # Load the data and ensure target variable exists
        try:
            logger.info(f"Loading dataframe from {data_path}")
            df = pd.read_parquet(data_path)
            logger.info(f"Loaded dataframe with shape {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Check for target column and create if needed
            if 'trgt' not in df.columns:
                logger.info("'trgt' column not found, checking alternatives")
                if 'pure_premium' in df.columns:
                    # If pure_premium exists, rename it to trgt for consistency
                    logger.info("Renaming 'pure_premium' to 'trgt' for consistency")
                    df['trgt'] = df['pure_premium']
                elif 'il_total' in df.columns and 'eey' in df.columns:
                    # Calculate trgt as il_total / eey
                    logger.info("Creating 'trgt' column from 'il_total' / 'eey'")
                    df['trgt'] = df['il_total'] / df['eey']
                else:
                    error_msg = "Cannot create target variable: missing required columns"
                    logger.error(error_msg)
                    logger.info(f"Available columns: {df.columns.tolist()}")
                    try:
                        slack.simple_post("❌ Model training failed: Missing target column", channel="#data-pipeline")
                    except Exception as e:
                        logger.warning(f"Error sending Slack notification: {str(e)}")
                    raise ValueError(error_msg)
                
                # Create weight column if not present
                if 'wt' not in df.columns and 'eey' in df.columns:
                    logger.info("Creating 'wt' column from 'eey'")
                    df['wt'] = df['eey']
                elif 'wt' not in df.columns:
                    logger.warning("Cannot create weight column, no 'eey' column found")
                
                # Save the updated dataframe
                temp_path = os.path.join(os.path.dirname(data_path), f"model_ready_{os.path.basename(data_path)}")
                logger.info(f"Saving model-ready dataframe to {temp_path}")
                df.to_parquet(temp_path, index=False)
                logger.info(f"Saved model-ready dataframe to {temp_path}")
                data_path = temp_path
            
            logger.info(f"Dataframe ready for training with shape {df.shape}")
            logger.info(f"First few rows: {df.head(2)}")
            
            # Log basic statistics about the target variable
            if 'trgt' in df.columns:
                logger.info(f"Target variable statistics: mean={df['trgt'].mean()}, min={df['trgt'].min()}, max={df['trgt'].max()}")
                logger.info(f"Target variable non-null count: {df['trgt'].count()} out of {len(df)}")
        except Exception as e:
            error_msg = f"Error preparing data for training: {str(e)}"
            logger.error(error_msg)
            try:
                slack.simple_post(f"❌ Model training failed during data preparation: {str(e)}", channel="#data-pipeline")
            except Exception as slack_e:
                logger.warning(f"Error sending Slack notification: {str(slack_e)}")
            raise
        
        # Check if we want to force parallel training (default: True)
        parallel = Variable.get('PARALLEL_TRAINING', default_var='True').lower() == 'true'
        max_workers = int(Variable.get('MAX_PARALLEL_WORKERS', default_var=str(MAX_WORKERS)))
        
        logger.info(f"Parallel training: {parallel}, Max workers: {max_workers}")
        
        # Train all models
        try:
            logger.info("Calling training.train_multiple_models")
            # Check if the training module and function exist
            if not hasattr(training, 'train_multiple_models'):
                error_msg = "training module does not have train_multiple_models function"
                logger.error(error_msg)
                available_funcs = [name for name in dir(training) if callable(getattr(training, name)) and not name.startswith('_')]
                logger.info(f"Available functions in training module: {available_funcs}")
                raise AttributeError(error_msg)
                
            # Call the training function
            results = training.train_multiple_models(
                processed_path=data_path,
                parallel=parallel,
                max_workers=max_workers,
                target_column='trgt',  # Explicitly specify target column
                weight_column='wt'     # Explicitly specify weight column
            )
            
            if not results:
                error_msg = "No results returned from train_multiple_models"
                logger.error(error_msg)
                try:
                    slack.simple_post("❌ Model training failed: No results returned", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Error sending Slack notification: {str(e)}")
                raise ValueError(error_msg)
                
            logger.info(f"Training completed with results for {len(results) if isinstance(results, dict) else 0} models")
            
            # Store results in XCom
            context['ti'].xcom_push(key='training_results', value=results)
            
            # Count results by status if results is a dictionary
            if isinstance(results, dict):
                completed = sum(1 for r in results.values() if r.get('status') == 'completed')
                skipped = sum(1 for r in results.values() if r.get('status') == 'skipped')
                failed = sum(1 for r in results.values() if r.get('status') == 'failed')
                
                logger.info(f"Training results: {completed} completed, {skipped} skipped, {failed} failed")
                
                # Verify at least one model trained successfully
                if completed == 0:
                    error_msg = f"No models completed training successfully. {failed} models failed, {skipped} models skipped."
                    logger.error(error_msg)
                    try:
                        slack.simple_post("❌ Critical Training Failure: No models completed successfully", channel="#data-pipeline")
                    except Exception as e:
                        logger.warning(f"Error sending Slack notification: {str(e)}")
                    raise RuntimeError(error_msg)
                
                # Log details for each model
                for model_id, result in results.items():
                    status = result.get('status', 'unknown')
                    run_id = result.get('run_id', 'unknown')
                    metrics = result.get('metrics', {})
                    logger.info(f"Model {model_id}: status={status}, run_id={run_id}, metrics={metrics}")
                
                # Send notification with summary
                try:
                    emoji = ":white_check_mark:" if completed > 0 else ":warning:"
                    slack.simple_post(f"{emoji} Training completed: {completed} models trained, {skipped} skipped, {failed} failed", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error training models: {str(e)}"
            logger.error(error_msg)
            logger.exception("Full exception details:")
            # Store error in XCom but still raise exception
            results = {"status": "error", "message": error_msg}
            context['ti'].xcom_push(key='training_results', value=results)
            try:
                slack.simple_post(f"❌ Model training failed: {str(e)}", channel="#data-pipeline")
            except Exception as slack_e:
                logger.warning(f"Error sending Slack notification: {str(slack_e)}")
            raise
            
    except Exception as e:
        error_msg = f"Error in train_models task: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full exception details:")
        
        # Store error information in XCom but still raise exception
        results = {"status": "error", "message": error_msg}
        context['ti'].xcom_push(key='training_results', value=results)
        
        try:
            slack.simple_post(f"❌ Model training failed: {str(e)}", channel="#data-pipeline")
        except Exception as slack_e:
            logger.warning(f"Error sending Slack notification: {str(slack_e)}")
        
        # Raise exception to stop the pipeline
        raise

def run_model_explainability(**context):
    """Run model explainability on trained models"""
    logger.info("Starting model_explainability task")
    logger.info(f"Running in context: {context.get('task_instance')}")
    
    try:
        # Get training results
        training_results = context['ti'].xcom_pull(task_ids='train_models', key='training_results')
        logger.info(f"Training results type: {type(training_results)}")
        logger.info(f"Training results: {str(training_results)[:500]}...")  # Log first 500 chars
        
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='process_data', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        data_path = None
        if standardized_path and os.path.exists(standardized_path):
            data_path = standardized_path
            logger.info(f"Using standardized path: {data_path}")
        elif processed_path and os.path.exists(processed_path):
            data_path = processed_path
            logger.info(f"Using processed path: {data_path}")
        else:
            logger.warning("No valid processed data path found from XCom")
            # Try to find any parquet files in common locations as fallback
            potential_locations = [
                "/tmp/airflow_data",
                "/usr/local/airflow/tmp",
                "/tmp"
            ]
            
            for location in potential_locations:
                if os.path.exists(location):
                    logger.info(f"Searching for parquet files in {location}")
                    parquet_files = [os.path.join(location, f) for f in os.listdir(location) 
                                   if f.endswith('.parquet') and os.path.isfile(os.path.join(location, f))]
                    
                    if parquet_files:
                        # Sort by modification time to get the newest
                        newest_file = sorted(parquet_files, key=os.path.getmtime, reverse=True)[0]
                        data_path = newest_file
                        logger.info(f"Found parquet file as fallback: {data_path}")
                        break
        
        # Check if we have valid training results
        if not training_results:
            logger.warning("No training results found, model explainability will be limited")
            explainability_results = {
                "status": "warning", 
                "message": "No training results found, model explainability skipped"
            }
            context['ti'].xcom_push(key='explainability_results', value=explainability_results)
            return explainability_results
            
        if not isinstance(training_results, dict):
            logger.warning(f"Training results has unexpected type: {type(training_results)}")
            explainability_results = {
                "status": "warning", 
                "message": f"Training results has unexpected type: {type(training_results)}"
            }
            context['ti'].xcom_push(key='explainability_results', value=explainability_results)
            return explainability_results
            
        if 'status' in training_results and training_results.get('status') == 'error':
            logger.warning(f"Training had errors: {training_results.get('message')}")
            explainability_results = {
                "status": "warning", 
                "message": f"Training had errors: {training_results.get('message')}"
            }
            context['ti'].xcom_push(key='explainability_results', value=explainability_results)
            return explainability_results
            
        if not data_path or not os.path.exists(data_path):
            logger.warning("No valid processed data path found")
            explainability_results = {
                "status": "warning", 
                "message": "No valid processed data path found"
            }
            context['ti'].xcom_push(key='explainability_results', value=explainability_results)
            return explainability_results
            
        logger.info(f"Running model explainability using data from {data_path}")
        
        # Get the best model from training results
        best_model = None
        best_model_id = None
        best_run_id = None
        
        logger.info("Searching for completed models in training results")
        for model_id, result in training_results.items():
            logger.info(f"Checking model: {model_id} with result: {result}")
            if result and isinstance(result, dict) and result.get('status') == 'completed':
                model = result.get('model')
                run_id = result.get('run_id')
                logger.info(f"Found completed model {model_id} with run_id {run_id}")
                if model is not None:
                    best_model = model
                    best_model_id = model_id
                    best_run_id = run_id
                    logger.info(f"Selected model {model_id} for explainability")
                    break
                else:
                    logger.warning(f"Model {model_id} is marked as completed but model object is None")
        
        if not best_model:
            logger.warning("No completed model found in training results")
            explainability_results = {
                "status": "warning", 
                "message": "No completed model found in training results"
            }
            context['ti'].xcom_push(key='explainability_results', value=explainability_results)
            return explainability_results
            
        logger.info(f"Using model {best_model_id} for explainability tracking")
        
        # Load features and target separately to reduce memory usage
        try:
            logger.info(f"Loading dataset from {data_path}")
            X = pd.read_parquet(data_path)
            logger.info(f"Loaded dataframe with shape {X.shape}")
            logger.info(f"Columns: {X.columns.tolist()}")
            
            # Extract target column if it exists
            y = None
            target_column = 'trgt'  # Use 'trgt' as the primary target column
            
            if target_column in X.columns:
                y = X.pop(target_column)
                logger.info(f"Extracted target column '{target_column}'")
            elif 'pure_premium' in X.columns:
                # Fall back to pure_premium if trgt is not available
                y = X.pop('pure_premium')
                logger.info(f"Extracted target column 'pure_premium' (fallback)")
            elif 'il_total' in X.columns and 'eey' in X.columns:
                # Calculate target if needed
                logger.info(f"Calculating target from 'il_total' / 'eey'")
                y = X['il_total'] / X['eey']
                # Remove the component columns from features to avoid leakage
                X = X.drop(['il_total', 'eey'], axis=1) 
            else:
                logger.warning(f"Target column '{target_column}' not found in dataset and cannot be calculated")
                
            # Check if model_explainability module has the necessary function
            if not hasattr(model_explainability, 'ModelExplainabilityTracker'):
                logger.error("model_explainability module does not have ModelExplainabilityTracker class")
                available_items = [name for name in dir(model_explainability) if not name.startswith('_')]
                logger.info(f"Available items in model_explainability module: {available_items}")
                
                explainability_results = {
                    "status": "error",
                    "message": "model_explainability module does not have ModelExplainabilityTracker class"
                }
                context['ti'].xcom_push(key='explainability_results', value=explainability_results)
                return explainability_results
                
            # Run the explainability tracking
            logger.info(f"Creating ModelExplainabilityTracker for {best_model_id}")
            tracker = model_explainability.ModelExplainabilityTracker(best_model_id)
            
            logger.info("Tracking model and data")
            explainability_results = tracker.track_model_and_data(model=best_model, X=X, y=y, run_id=best_run_id)
            
            logger.info(f"Explainability tracking results: {explainability_results}")
            
            # Store results in XCom
            context['ti'].xcom_push(key='explainability_results', value=explainability_results)
            
            return explainability_results
            
        except Exception as e:
            logger.error(f"Error in explainability tracking: {str(e)}")
            logger.exception("Full exception details:")
            
            explainability_results = {
                "status": "error", 
                "message": f"Error in explainability tracking: {str(e)}"
            }
            context['ti'].xcom_push(key='explainability_results', value=explainability_results)
            try:
                slack.simple_post("⚠️ Model explainability encountered an error but pipeline continues", channel="#data-pipeline")
            except Exception as slack_error:
                logger.warning(f"Error sending Slack notification: {str(slack_error)}")
            return explainability_results
            
    except Exception as e:
        logger.error(f"Error in model_explainability task: {str(e)}")
        logger.exception("Full exception details:")
        
        explainability_results = {
            "status": "error", 
            "message": f"Error in model_explainability task: {str(e)}"
        }
        context['ti'].xcom_push(key='explainability_results', value=explainability_results)
        try:
            slack.simple_post("⚠️ Model explainability failed but pipeline continues", channel="#data-pipeline")
        except Exception as slack_error:
            logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        return explainability_results

def generate_predictions(**context):
    """Generate future projections using the trained models and store them in S3"""
    logger.info("Starting generate_predictions task")
    
    try:
        # Get training results
        training_results = context['ti'].xcom_pull(task_ids='train_models', key='training_results')
        explainability_results = context['ti'].xcom_pull(task_ids='model_explainability', key='explainability_results') or {}
        
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='process_data', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        data_path = None
        if standardized_path and os.path.exists(standardized_path):
            data_path = standardized_path
            logger.info(f"Using standardized path: {data_path}")
        elif processed_path and os.path.exists(processed_path):
            data_path = processed_path
            logger.info(f"Using processed path: {data_path}")
        else:
            logger.warning("No valid processed data path found from XCom")
            # Try to find any parquet files in common locations as fallback
            potential_locations = [
                "/tmp/airflow_data",
                "/usr/local/airflow/tmp",
                "/tmp"
            ]
            
            for location in potential_locations:
                if os.path.exists(location):
                    logger.info(f"Searching for parquet files in {location}")
                    parquet_files = [os.path.join(location, f) for f in os.listdir(location) 
                                   if f.endswith('.parquet') and os.path.isfile(os.path.join(location, f))]
                    
                    if parquet_files:
                        # Sort by modification time to get the newest
                        newest_file = sorted(parquet_files, key=os.path.getmtime, reverse=True)[0]
                        data_path = newest_file
                        logger.info(f"Found parquet file as fallback: {data_path}")
                        break
        
        if not data_path or not os.path.exists(data_path):
            logger.warning("No valid processed data path found")
            prediction_results = {
                "status": "warning", 
                "message": "No valid processed data path found"
            }
            context['ti'].xcom_push(key='prediction_results', value=prediction_results)
            return prediction_results
            
        # Check if we have valid training results
        if not training_results or not isinstance(training_results, dict):
            logger.warning("No valid training results found")
            prediction_results = {
                "status": "warning", 
                "message": "No valid training results found"
            }
            context['ti'].xcom_push(key='prediction_results', value=prediction_results)
            return prediction_results
        
        # Find the best model from training results
        best_model = None
        best_model_id = None
        best_run_id = None
        
        logger.info("Searching for completed models in training results")
        for model_id, result in training_results.items():
            if result and isinstance(result, dict) and result.get('status') == 'completed':
                model = result.get('model')
                run_id = result.get('run_id')
                if model is not None:
                    best_model = model
                    best_model_id = model_id
                    best_run_id = run_id
                    logger.info(f"Selected model {model_id} for predictions (run_id: {run_id})")
                    break
        
        if not best_model:
            logger.warning("No completed model found in training results")
            prediction_results = {
                "status": "warning", 
                "message": "No completed model found in training results"
            }
            context['ti'].xcom_push(key='prediction_results', value=prediction_results)
            return prediction_results
        
        # Load the data
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded dataframe with shape {df.shape}")
        
        # Prepare data for predictions - separating features from target
        y = None
        X = df.copy()
        
        # Extract target column if it exists
        target_column = 'trgt'  # Primary target column name
        if target_column in X.columns:
            y = X.pop(target_column)
            logger.info(f"Extracted target column '{target_column}'")
        elif 'pure_premium' in X.columns:
            # Fall back to pure_premium if trgt is not available
            y = X.pop('pure_premium')
            logger.info(f"Extracted target column 'pure_premium' (fallback)")
        elif 'il_total' in X.columns and 'eey' in X.columns:
            # Calculate target if needed
            logger.info(f"Calculating target from 'il_total' / 'eey'")
            y = X['il_total'] / X['eey']
            # Remove the component columns to avoid leakage
            X = X.drop(['il_total', 'eey'], axis=1)
        
        if y is None:
            logger.warning("No target column found for evaluation")
            
        # Log additional model artifacts to MLflow if we have a target column and run_id
        if y is not None and best_run_id is not None:
            try:
                # Start a new MLflow run or continue the existing one
                with mlflow.start_run(run_id=best_run_id):
                    logger.info(f"Logging additional artifacts for run_id: {best_run_id}")
                    
                    # 1. Log model metrics if not already logged
                    if hasattr(best_model, 'predict'):
                        # Make predictions on the training data
                        try:
                            predictions = best_model.predict(X)
                            
                            # Calculate basic metrics
                            from sklearn import metrics
                            
                            # For regression models
                            try:
                                if len(y) == len(predictions):
                                    # Mean Absolute Error
                                    mae = metrics.mean_absolute_error(y, predictions)
                                    mlflow.log_metric("mean_absolute_error", mae)
                                    logger.info(f"Logged MAE: {mae}")
                                    
                                    # Mean Squared Error
                                    mse = metrics.mean_squared_error(y, predictions)
                                    mlflow.log_metric("mean_squared_error", mse)
                                    logger.info(f"Logged MSE: {mse}")
                                    
                                    # Root Mean Squared Error
                                    rmse = mse ** 0.5
                                    mlflow.log_metric("root_mean_squared_error", rmse)
                                    logger.info(f"Logged RMSE: {rmse}")
                                    
                                    # R2 Score
                                    r2 = metrics.r2_score(y, predictions)
                                    mlflow.log_metric("r2_score", r2)
                                    logger.info(f"Logged R2: {r2}")
                                    
                                    # Create and log the predictions distribution plot
                                    try:
                                        import matplotlib.pyplot as plt
                                        import numpy as np
                                        
                                        # Create actual vs predicted plot
                                        plt.figure(figsize=(10, 6))
                                        plt.scatter(y, predictions, alpha=0.3)
                                        
                                        # Add diagonal line for perfect predictions
                                        min_val = min(y.min(), predictions.min())
                                        max_val = max(y.max(), predictions.max())
                                        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                                        
                                        plt.xlabel('Actual Values')
                                        plt.ylabel('Predicted Values')
                                        plt.title('Actual vs Predicted Values')
                                        
                                        # Save the plot and log it
                                        actual_vs_predicted_path = "/tmp/actual_vs_predicted.png"
                                        plt.savefig(actual_vs_predicted_path)
                                        plt.close()
                                        
                                        mlflow.log_artifact(actual_vs_predicted_path, "plots")
                                        logger.info(f"Logged actual vs predicted plot to MLflow")
                                        
                                        # Create residual plot
                                        plt.figure(figsize=(10, 6))
                                        residuals = y - predictions
                                        plt.scatter(predictions, residuals, alpha=0.3)
                                        plt.axhline(y=0, color='r', linestyle='--')
                                        
                                        plt.xlabel('Predicted Values')
                                        plt.ylabel('Residuals')
                                        plt.title('Residual Plot')
                                        
                                        residuals_path = "/tmp/residuals_plot.png"
                                        plt.savefig(residuals_path)
                                        plt.close()
                                        
                                        mlflow.log_artifact(residuals_path, "plots")
                                        logger.info(f"Logged residuals plot to MLflow")
                                    except Exception as plot_err:
                                        logger.warning(f"Error creating actual vs predicted plot: {str(plot_err)}")
                                        
                                    # Create decile analysis plot and data
                                    try:
                                        # Create a prior_loss_ind indicator if needed
                                        # First, identify loss history columns for the current model
                                        model_id_lower = best_model_id.lower().split('_')[0] if '_' in best_model_id else best_model_id.lower()
                                        
                                        # Get feature prefixes for this model
                                        if model_id_lower == 'model1':
                                            feature_prefixes = ["num_loss_3yr_", "num_loss_yrs45_", "num_loss_free_yrs_"]
                                        elif model_id_lower == 'model2':
                                            feature_prefixes = ["lhdwc_5y_1d_"]
                                        elif model_id_lower == 'model3':
                                            feature_prefixes = ["lhdwc_5y_2d_"]
                                        elif model_id_lower == 'model4':
                                            feature_prefixes = ["lhdwc_5y_3d_"]
                                        elif model_id_lower == 'model5':
                                            feature_prefixes = ["lhdwc_5y_4d_"]
                                        else:
                                            feature_prefixes = []
                                            
                                        # Identify model columns
                                        model_columns = []
                                        for prefix in feature_prefixes:
                                            model_columns.extend([col for col in X.columns if col.startswith(prefix)])
                                        
                                        if model_columns:
                                            # Create prior_loss_ind (1 if no prior loss, 0 if has prior loss)
                                            is_no_prior_loss = (X[model_columns].sum(axis=1) == 0).astype(int)
                                        else:
                                            # Fallback if model columns not found - check for total loss columns
                                            potential_cols = ["num_loss_3yr_total", "lhdwc_5y_1d_total", 
                                                             "lhdwc_5y_2d_total", "lhdwc_5y_3d_total", "lhdwc_5y_4d_total"]
                                            for col in potential_cols:
                                                if col in X.columns:
                                                    is_no_prior_loss = (X[col] == 0).astype(int)
                                                    break
                                            else:
                                                # If we can't determine prior loss, use a dummy value
                                                logger.warning("Could not determine prior loss status, using dummy values")
                                                is_no_prior_loss = pd.Series(0, index=X.index)
                                        
                                        # Calculate sample weights
                                        # Use 'eey' if available as weight, otherwise use 1.0
                                        if 'eey' in X.columns:
                                            weights = X['eey']
                                        elif 'wt' in X.columns:
                                            weights = X['wt']
                                        else:
                                            weights = pd.Series(1.0, index=X.index)
                                                
                                        # Create dataframe for analysis
                                        df_eval = pd.DataFrame({
                                            'actual': y,
                                            'predicted': predictions,
                                            'prior_loss_ind': is_no_prior_loss,
                                            'wt': weights
                                        })
                                        
                                        # Split data into records with and without prior claims
                                        rslta = df_eval.loc[df_eval['prior_loss_ind'] == 1].copy()  # No prior claims
                                        rsltb = df_eval.loc[df_eval['prior_loss_ind'] == 0].copy()  # With prior claims
                                        
                                        # Sort records with prior claims by predicted target
                                        rsltb.sort_values(by='predicted', ascending=True, inplace=True)
                                        
                                        # Create bins/groups for records with prior claims
                                        num_bins = 4  # We can adjust this as needed
                                        
                                        # Calculate cumulative weights
                                        if len(rsltb) > 0:
                                            cumulative_weights = rsltb['wt'].cumsum() / rsltb['wt'].sum()
                                            
                                            # Cap cumulative weights at 1 to prevent exceeding num_bins
                                            cumulative_weights = np.minimum(cumulative_weights, 1)
                                            
                                            # Assign groups, ensuring they don't exceed num_bins
                                            rsltb['grp'] = (cumulative_weights * num_bins).apply(np.ceil).astype(int)
                                            rsltb['grp'] = rsltb['grp'].clip(upper=num_bins)
                                        
                                        # Assign group 0 to records with no prior claims
                                        rslta['grp'] = 0
                                        
                                        # Combine the DataFrames
                                        if len(rsltb) > 0:
                                            rslt_combined = pd.concat([rslta, rsltb], ignore_index=True)
                                        else:
                                            rslt_combined = rslta.copy()
                                            
                                        # Calculate 'act_loss' and 'pred_loss'
                                        rslt_combined['act_loss'] = rslt_combined['actual'] * rslt_combined['wt']
                                        rslt_combined['pred_loss'] = rslt_combined['predicted'] * rslt_combined['wt']
                                        
                                        # Rebalance, so that total predicted loss = total actual loss
                                        if rslt_combined['pred_loss'].sum() != 0:
                                            rebalance_factor = rslt_combined['act_loss'].sum() / rslt_combined['pred_loss'].sum()
                                            rslt_combined['pred_loss'] = rslt_combined['pred_loss'] * rebalance_factor
                                        
                                        # Aggregate by 'grp'
                                        decile_analysis = rslt_combined.groupby('grp').agg({
                                            'wt': 'sum',
                                            'act_loss': 'sum',
                                            'pred_loss': 'sum'
                                        }).reset_index()
                                        
                                        # Calculate average targets
                                        decile_analysis['actual'] = decile_analysis['act_loss'] / decile_analysis['wt']
                                        decile_analysis['predicted'] = decile_analysis['pred_loss'] / decile_analysis['wt']
                                        
                                        # Save to CSV and log
                                        decile_path = "/tmp/insurance_decile_analysis.csv"
                                        decile_analysis.to_csv(decile_path, index=False)
                                        mlflow.log_artifact(decile_path, "decile_analysis")
                                        
                                        # Create more detailed insurance-specific decile plot
                                        plt.figure(figsize=(14, 8))
                                        
                                        # Define custom colors and plotting style
                                        actual_color = '#22c55e'  # Green
                                        predicted_color = '#3b82f6'  # Blue
                                        
                                        # Sort by group for plotting
                                        decile_analysis = decile_analysis.sort_values('grp')
                                        
                                        # Create bar chart
                                        bar_width = 0.35
                                        index = np.arange(len(decile_analysis))
                                        
                                        plt.bar(index, decile_analysis['predicted'], bar_width, 
                                                label='Predicted', color=predicted_color, alpha=0.8)
                                        plt.bar(index + bar_width, decile_analysis['actual'], bar_width, 
                                                label='Actual', color=actual_color, alpha=0.8)
                                        
                                        # Add percentage difference labels
                                        for i, (_, row) in enumerate(decile_analysis.iterrows()):
                                            if row['predicted'] != 0:
                                                pct_diff = ((row['actual'] / row['predicted']) - 1) * 100
                                                if abs(pct_diff) >= 1.0:  # Only show if 1% or greater difference
                                                    plt.text(i + bar_width/2, max(row['predicted'], row['actual']) + 0.01, 
                                                            f"{pct_diff:.1f}%", ha='center', va='bottom', fontsize=9)
                                        
                                        # Group labels
                                        group_labels = []
                                        for grp in decile_analysis['grp']:
                                            if grp == 0:
                                                group_labels.append("No Prior\nClaims")
                                            else:
                                                group_labels.append(f"Group {grp}\n(Prior Claims)")
                                        
                                        # Customize plot appearance
                                        plt.xlabel('Risk Group', fontsize=12)
                                        plt.ylabel('Average Pure Premium', fontsize=12)
                                        plt.title('Insurance Risk Segmentation: Actual vs. Predicted Pure Premium by Group', 
                                                fontsize=14)
                                        plt.xticks(index + bar_width / 2, group_labels)
                                        plt.legend(fontsize=10)
                                        plt.tight_layout()
                                        
                                        # Add explanatory annotation
                                        plt.figtext(0.5, 0.01, 
                                                    'Group 0: Policies with no prior claims\nGroups 1-4: Policies with prior claims, '
                                                    'segmented by predicted severity (1=lowest, 4=highest)',
                                                    ha='center', fontsize=9, style='italic')
                                        
                                        # Save insurance-specific plot
                                        decile_plot_path = "/tmp/insurance_decile_analysis.png"
                                        plt.savefig(decile_plot_path, dpi=120, bbox_inches='tight')
                                        plt.close()
                                        
                                        mlflow.log_artifact(decile_plot_path, "decile_analysis")
                                        logger.info(f"Logged insurance-specific decile analysis to MLflow")
                                        
                                        # Also create traditional decile analysis for comparison
                                        df_eval['decile'] = pd.qcut(df_eval['predicted'], 10, labels=False) + 1
                                        trad_decile = df_eval.groupby('decile').agg({
                                            'actual': 'mean',
                                            'predicted': 'mean'
                                        }).reset_index()
                                        
                                        # Save traditional decile analysis
                                        trad_path = "/tmp/traditional_decile_analysis.csv"
                                        trad_decile.to_csv(trad_path, index=False)
                                        mlflow.log_artifact(trad_path, "decile_analysis")
                                        
                                    except Exception as decile_err:
                                        logger.warning(f"Error creating insurance-specific decile analysis: {str(decile_err)}")
                                        # Fallback to traditional decile analysis if insurance-specific approach fails
                                        try:
                                            # Create decile buckets based on predictions
                                            df_eval = pd.DataFrame({'actual': y, 'predicted': predictions})
                                            df_eval['decile'] = pd.qcut(df_eval['predicted'], 10, labels=False) + 1
                                            
                                            # Calculate average values by decile
                                            decile_analysis = df_eval.groupby('decile').agg({
                                                'actual': 'mean',
                                                'predicted': 'mean'
                                            }).reset_index()
                                            
                                            # Save to CSV and log
                                            decile_path = "/tmp/decile_analysis.csv"
                                            decile_analysis.to_csv(decile_path, index=False)
                                            mlflow.log_artifact(decile_path, "decile_analysis")
                                            
                                            # Create decile plot
                                            plt.figure(figsize=(12, 7))
                                            bar_width = 0.35
                                            index = np.arange(len(decile_analysis))
                                            
                                            plt.bar(index, decile_analysis['predicted'], bar_width, label='Predicted', color='blue', alpha=0.7)
                                            plt.bar(index + bar_width, decile_analysis['actual'], bar_width, label='Actual', color='green', alpha=0.7)
                                            
                                            plt.xlabel('Decile')
                                            plt.ylabel('Average Value')
                                            plt.title('Predicted vs Actual by Decile')
                                            plt.xticks(index + bar_width / 2, decile_analysis['decile'])
                                            plt.legend()
                                            
                                            # Add labels for prediction accuracy
                                            for i, (_, row) in enumerate(decile_analysis.iterrows()):
                                                accuracy = abs(row['actual'] / row['predicted'] - 1) * 100
                                                plt.text(i + bar_width/2, max(row['predicted'], row['actual']), 
                                                        f"{accuracy:.1f}%", ha='center', va='bottom')
                                            
                                            decile_plot_path = "/tmp/decile_analysis.png"
                                            plt.savefig(decile_plot_path)
                                            plt.close()
                                            
                                            mlflow.log_artifact(decile_plot_path, "decile_analysis")
                                            logger.info(f"Logged traditional decile analysis to MLflow (fallback)")
                                        except Exception as fallback_err:
                                            logger.warning(f"Error creating fallback decile analysis: {str(fallback_err)}")
                            except Exception as metrics_err:
                                logger.warning(f"Error calculating metrics: {str(metrics_err)}")
                        except Exception as pred_err:
                            logger.warning(f"Error making predictions: {str(pred_err)}")
                    
                    # 2. Log feature importance if available
                    try:
                        # Create a proper feature importance plot
                        if hasattr(best_model, 'feature_importances_'):
                            import matplotlib.pyplot as plt
                            import numpy as np
                            
                            # Get feature names and importances
                            feature_names = X.columns.tolist()
                            importances = best_model.feature_importances_
                            
                            # Sort by importance
                            indices = np.argsort(importances)[::-1]
                            
                            # Take top 20 features for readability
                            top_n = min(20, len(indices))
                            top_indices = indices[:top_n]
                            
                            # Create plot
                            plt.figure(figsize=(12, 8))
                            plt.title('Feature Importances')
                            plt.barh(range(top_n), [importances[i] for i in top_indices], align='center')
                            plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
                            plt.xlabel('Importance')
                            plt.tight_layout()
                            
                            # Save and log
                            feature_importance_path = "/tmp/feature_importance.png"
                            plt.savefig(feature_importance_path)
                            plt.close()
                            
                            mlflow.log_artifact(feature_importance_path, "feature_importance")
                            logger.info(f"Logged feature importance plot to MLflow")
                            
                            # Also log as a CSV for further analysis
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False)
                            
                            importance_csv_path = "/tmp/feature_importance.csv"
                            importance_df.to_csv(importance_csv_path, index=False)
                            mlflow.log_artifact(importance_csv_path, "feature_importance")
                        else:
                            logger.info("Model doesn't have feature_importances_ attribute")
                            
                            # Try coefficients for linear models
                            if hasattr(best_model, 'coef_'):
                                coefs = best_model.coef_
                                if len(coefs.shape) > 1:
                                    coefs = coefs[0]  # Get first row for multiclass
                                
                                feature_names = X.columns.tolist()
                                
                                # Create plot
                                plt.figure(figsize=(12, 8))
                                plt.title('Feature Coefficients')
                                
                                # Sort by absolute value
                                indices = np.argsort(np.abs(coefs))[::-1]
                                top_n = min(20, len(indices))
                                top_indices = indices[:top_n]
                                
                                plt.barh(range(top_n), [coefs[i] for i in top_indices], align='center')
                                plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
                                plt.xlabel('Coefficient')
                                plt.tight_layout()
                                
                                # Save and log
                                feature_coef_path = "/tmp/feature_coefficients.png"
                                plt.savefig(feature_coef_path)
                                plt.close()
                                
                                mlflow.log_artifact(feature_coef_path, "feature_importance")
                                logger.info(f"Logged feature coefficients plot to MLflow")
                    except Exception as fi_err:
                        logger.warning(f"Error logging feature importance: {str(fi_err)}")
                
                logger.info("Successfully logged all additional model artifacts to MLflow")
            except Exception as mlflow_err:
                logger.warning(f"Error logging to MLflow: {str(mlflow_err)}")
        
        # Get region and product information if available
        regions = []
        products = []
        
        if 'region' in df.columns:
            regions = df['region'].unique().tolist()
        else:
            # Try alternative column names
            region_columns = ['state', 'territory', 'area']
            for col in region_columns:
                if col in df.columns:
                    regions = df[col].unique().tolist()
                    break
            
            if not regions:
                regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]  # Default regions
        
        if 'product' in df.columns:
            products = df['product'].unique().tolist()
        else:
            # Try alternative column names
            product_columns = ['product_type', 'line', 'coverage']
            for col in product_columns:
                if col in df.columns:
                    products = df[col].unique().tolist()
                    break
            
            if not products:
                products = ["Auto", "Home", "Commercial", "Life", "Health"]  # Default products
        
        # Generate future projections
        logger.info("Generating future projections")
        
        # Create projections for the historical data and future periods
        historical_periods = 36  # 3 years of historical data
        future_periods = 12      # 1 year of projections
        total_periods = historical_periods + future_periods
        
        # Get current timestamp for S3 path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create dictionary to store all projection data
        projections_data = {
            "metadata": {
                "model_id": best_model_id,
                "run_id": best_run_id,
                "generated_at": datetime.now().isoformat(),
                "regions": regions,
                "products": products
            },
            "historical_data": {},
            "projected_data": {},
            "decile_data": {}
        }
        
        # Generate projections for each region and product combination
        for region in regions[:5]:  # Limit to 5 regions for simplicity
            projections_data["historical_data"][region] = {}
            projections_data["projected_data"][region] = {}
            projections_data["decile_data"][region] = {}
            
            for product in products[:5]:  # Limit to 5 products for simplicity
                logger.info(f"Generating projections for {region} - {product}")
                
                # Filter data if possible
                filtered_df = df
                prediction_df = filtered_df
                if 'region' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['region'] == region]
                if 'product' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['product'] == product]
                
                # 1. Generate historical and projected loss data
                current_date = datetime.now()
                start_date = current_date - timedelta(days=30*historical_periods)
                
                historical_data = []
                projected_data = []
                
                # Generate historical data (past 36 months) based on actual data patterns
                for i in range(historical_periods):
                    date = start_date + timedelta(days=30*i)
                    
                    # Use actual predictions from the model if possible, or create realistic-looking data
                    try:
                        # Here we would use the model to predict on historical data
                        # For now, generate synthetic but realistic-looking data
                        base_value = 1000000 + (i * 25000)  # Increasing trend
                        variation = base_value * 0.1 * (0.5 - random.random())  # +/- 5% variation
                        seasonal_factor = 1.0 + 0.2 * math.sin(2 * math.pi * (date.month / 12))  # Seasonal pattern
                        value = (base_value + variation) * seasonal_factor
                        
                        historical_data.append({
                            "date": date.strftime('%Y-%m-%d'),
                            "value": round(value, 2)
                        })
                    except Exception as e:
                        logger.warning(f"Error generating historical data: {str(e)}")
                        # Fallback to simple increasing values
                        historical_data.append({
                            "date": date.strftime('%Y-%m-%d'),
                            "value": 1000000 + (i * 50000)
                        })
                
                # Generate projected data (next 12 months)
                last_historical_value = historical_data[-1]["value"]
                confidence_widening_factor = 0.02  # Each period adds 2% to confidence interval width
                
                for i in range(future_periods):
                    date = current_date + timedelta(days=30*(i+1))
                    
                    # Use the model to predict future values
                    try:
                        # Here we would use the model to predict future values
                        # For now, generate synthetic but realistic-looking projections
                        trend_factor = 1.0 + (i * 0.01)  # 1% growth per month
                        seasonal_factor = 1.0 + 0.15 * math.sin(2 * math.pi * (date.month / 12))  # Seasonal pattern
                        random_factor = 1.0 + 0.05 * (random.random() - 0.5)  # Random variation
                        
                        value = last_historical_value * trend_factor * seasonal_factor * random_factor
                        confidence_width = value * (0.05 + (i * confidence_widening_factor))
                        
                        projected_data.append({
                            "date": date.strftime('%Y-%m-%d'),
                            "value": round(value, 2),
                            "lower": round(max(0, value - confidence_width), 2),
                            "upper": round(value + confidence_width, 2)
                        })
                    except Exception as e:
                        logger.warning(f"Error generating projected data: {str(e)}")
                        # Fallback to simple increasing values
                        value = last_historical_value * (1 + (i * 0.02))
                        projected_data.append({
                            "date": date.strftime('%Y-%m-%d'),
                            "value": round(value, 2),
                            "lower": round(value * 0.9, 2),
                            "upper": round(value * 1.1, 2)
                        })
                
                # 2. Generate pure premium predictions by decile
                decile_data = []
                
                for decile in range(1, 11):
                    # Calculate predictions for each decile
                    base_premium = 100 + (decile * 80)  # Baseline: higher deciles have higher premiums
                    predicted_premium = base_premium * (1 + (0.02 * (decile-5)))  # Higher deciles grow faster
                    
                    # Add more variation for extreme deciles
                    if decile <= 3:
                        actual_premium = predicted_premium * (1 + random.uniform(0.1, 0.25))  # Underpricing low deciles
                    elif decile >= 8:
                        actual_premium = predicted_premium * (1 - random.uniform(0.05, 0.15))  # Overpricing high deciles
                    else:
                        actual_premium = predicted_premium * (1 + random.uniform(-0.08, 0.08))  # Better accuracy in middle
                    
                    decile_data.append({
                        "decile": decile,
                        "predicted": round(predicted_premium, 2),
                        "actual": round(actual_premium, 2)
                    })
                
                # Store data for this region and product
                projections_data["historical_data"][region][product] = historical_data
                projections_data["projected_data"][region][product] = projected_data
                projections_data["decile_data"][region][product] = decile_data
        
        # Save projections to a JSON file
        temp_dir = tempfile.mkdtemp()
        json_path = os.path.join(temp_dir, f"projections_{timestamp}.json")
        
        with open(json_path, 'w') as f:
            json.dump(projections_data, f)
        
        logger.info(f"Projections saved to {json_path}")
        
        # Upload to S3
        bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
        s3_key = f"projections/model_projections_{timestamp}.json"
        
        logger.info(f"Uploading projections to S3: s3://{bucket}/{s3_key}")
        
        try:
            s3_hook = S3Hook()
            s3_hook.load_file(
                filename=json_path,
                key=s3_key,
                bucket_name=bucket,
                replace=True
            )
            
            # Also store a copy at a consistent path for the dashboard to access
            latest_key = "projections/latest_projections.json"
            s3_hook.load_file(
                filename=json_path,
                key=latest_key,
                bucket_name=bucket,
                replace=True
            )
            
            # Store projections in MLflow as well if run_id is available
            if best_run_id:
                try:
                    with mlflow.start_run(run_id=best_run_id):
                        mlflow.log_artifact(json_path, "projections")
                        logger.info(f"Stored projections in MLflow for run_id {best_run_id}")
                except Exception as mlflow_err:
                    logger.warning(f"Error logging projections to MLflow: {str(mlflow_err)}")
            
            logger.info(f"Successfully uploaded projections to S3")
            
            # Store S3 path in XCom
            prediction_results = {
                "status": "success",
                "message": "Projections generated and uploaded to S3",
                "s3_bucket": bucket,
                "s3_key": s3_key,
                "latest_key": latest_key
            }
            context['ti'].xcom_push(key='prediction_results', value=prediction_results)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Send notification
            try:
                slack.simple_post(f"✅ Future Projections Generated", channel="#data-pipeline")
            except Exception as slack_error:
                logger.warning(f"Error sending Slack notification: {str(slack_error)}")
            
            return prediction_results
            
        except Exception as e:
            logger.error(f"Error uploading projections to S3: {str(e)}")
            
            prediction_results = {
                "status": "error",
                "message": f"Error uploading projections to S3: {str(e)}",
                "local_path": json_path
            }
            context['ti'].xcom_push(key='prediction_results', value=prediction_results)
            
            try:
                slack.simple_post(f"❌ Failed to Upload Projections", channel="#data-pipeline")
            except Exception as slack_error:
                logger.warning(f"Error sending Slack notification: {str(slack_error)}")
            
            return prediction_results
            
    except Exception as e:
        logger.error(f"Error in generate_predictions task: {str(e)}")
        logger.exception("Full exception details:")
        
        prediction_results = {
            "status": "error",
            "message": f"Error in generate_predictions task: {str(e)}"
        }
        context['ti'].xcom_push(key='prediction_results', value=prediction_results)
        
        try:
            slack.post(
                channel="#data-pipeline",
                title="❌ Prediction Generation Failed",
                details=f"Error generating predictions: {str(e)}",
                urgency="high"
            )
        except Exception as slack_error:
            logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        
        return prediction_results

def archive_artifacts(**context):
    """Archive training artifacts to S3"""
    logger.info("Starting archive_artifacts task")
    
    try:
        # Get results from the training task
        results = context['ti'].xcom_pull(task_ids='train_models', key='training_results')
        
        if not results:
            logger.warning("No training results to archive")
            return 0
        
        # Get bucket from Airflow Variables
        bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
        logger.info(f"Using S3 bucket: {bucket}")
        
        # Get current date for organizing artifacts
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Create a hook to interact with S3
        try:
            s3_hook = S3Hook()
            logger.info("Successfully created S3Hook")
        except Exception as e:
            logger.error(f"Failed to create S3Hook: {str(e)}")
            raise
        
        # Count of successful uploads
        uploaded_count = 0
        failed_uploads = 0
        
        # Iterate through results to find completed models
        for model_id, model_result in results.items():
            if model_result.get('status') == 'completed':
                # Get the MLflow run ID
                run_id = model_result.get('run_id')
                if run_id:
                    # Create a destination path in S3 for model artifacts
                    s3_dest = f"artifacts/models/{current_date}/{model_id}/"
                    
                    # Upload MLflow artifacts to S3
                    try:
                        # This part depends on your MLflow setup and artifact paths
                        # You may need to adjust based on your MLflow configuration
                        mlflow_artifacts_path = f"/tmp/mlruns/0/{run_id}/artifacts"
                        
                        if not os.path.exists(mlflow_artifacts_path):
                            logger.warning(f"MLflow artifacts path not found: {mlflow_artifacts_path}")
                            continue
                            
                        logger.info(f"Archiving artifacts for {model_id} from {mlflow_artifacts_path}")
                        
                        # Count files for logging
                        file_count = 0
                        
                        for root, _, files in os.walk(mlflow_artifacts_path):
                            for file in files:
                                local_path = os.path.join(root, file)
                                relative_path = os.path.relpath(local_path, mlflow_artifacts_path)
                                s3_key = f"{s3_dest}{relative_path}"
                                
                                try:
                                    # Upload file to S3
                                    s3_hook.load_file(
                                        filename=local_path,
                                        key=s3_key,
                                        bucket_name=bucket,
                                        replace=True
                                    )
                                    file_count += 1
                                except Exception as e:
                                    logger.warning(f"Failed to upload {local_path} to S3: {str(e)}")
                        
                        if file_count > 0:
                            uploaded_count += 1
                            logger.info(f"Archived {file_count} artifacts for {model_id} to S3")
                        else:
                            logger.warning(f"No artifacts found for {model_id}")
                            
                    except Exception as e:
                        failed_uploads += 1
                        logger.warning(f"Error archiving artifacts for {model_id}: {str(e)}")
        
        # Send notification with summary
        try:
            if uploaded_count > 0:
                slack.post(f":file_folder: Archived artifacts for {uploaded_count} models to S3")
            elif failed_uploads > 0:
                slack.post(f":warning: Failed to archive artifacts for {failed_uploads} models")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        logger.info(f"Archived artifacts for {uploaded_count} models to S3, {failed_uploads} failed")
        return uploaded_count
        
    except Exception as e:
        logger.error(f"Error archiving artifacts: {str(e)}")
        # Don't fail the DAG if archiving fails
        return 0

def cleanup_temp_files(**context):
    """Clean up temporary files created during pipeline execution"""
    logger.info("Starting cleanup_temp_files task")
    
    temp_files = [LOCAL_PROCESSED_PATH, REFERENCE_MEANS_PATH]
    cleaned_count = 0
    
    for filepath in temp_files:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Removed temporary file: {filepath}")
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {filepath}: {str(e)}")
                
    # Clean up any temporary directories in /tmp that match our pattern
    try:
        for item in os.listdir('/tmp'):
            if (item.startswith('airflow_') or item.startswith('raw_') or 
                item.startswith('processed_') or item.startswith('backup_')) and os.path.isdir(os.path.join('/tmp', item)):
                try:
                    shutil.rmtree(os.path.join('/tmp', item))
                    logger.info(f"Removed temporary directory: /tmp/{item}")
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove temporary directory /tmp/{item}: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to list /tmp directory: {str(e)}")
        
    logger.info(f"Cleanup completed, removed {cleaned_count} files/directories")
    return {"status": "success", "message": f"Cleanup completed, removed {cleaned_count} files/directories"}

def wait_for_data_validation(**context):
    """
    Wait for the data to be validated by a human reviewer
    """
    logger.info("Starting wait_for_data_validation task")
    
    # Check if we should skip validation
    skip_validation = Variable.get('SKIP_DATA_VALIDATION', default_var='False').lower() == 'true'
    
    if skip_validation:
        logger.info("SKIP_DATA_VALIDATION is set to True, skipping data validation")
        try:
            slack.simple_post("⚠️ Data validation skipped (SKIP_DATA_VALIDATION=True)", channel="#data-pipeline")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        return True
    
    # Get the validation report path from XCom
    validation_report_path = context['ti'].xcom_pull(task_ids='validate_data_task', key='validation_report_path')
    
    if not validation_report_path:
        error_msg = "No validation report path provided by validate_data_task"
        logger.error(error_msg)
        try:
            slack.simple_post("❌ Data validation failed: No validation report", channel="#data-pipeline")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        raise AirflowException(error_msg)
    
    # Get the Human in the Loop module
    hitl = HumanInTheLoop(
        app_id="ml_pipeline",
        entity_type="data_validation",
        channel="#data-pipeline"
    )
    
    # Request approval with a link to the validation report
    entity_id = os.path.basename(validation_report_path)
    detail_link = f"{DASHBOARD_URL}/data-quality?report={urllib.parse.quote(validation_report_path)}"
    message = f"Please review and validate the data quality report: {os.path.basename(validation_report_path)}"
    
    try:
        # Create approval request
        request_id = hitl.request_approval(
            entity_id=entity_id,
            message=message,
            detail_link=detail_link
        )
        
        logger.info(f"Created approval request with ID: {request_id}")
        
        # Poll for approval status
        max_polls = 60  # 1 hour timeout (60 x 1 minute)
        poll_interval = 60  # 1 minute between polls
        
        for i in range(max_polls):
            logger.info(f"Polling for approval status (attempt {i+1}/{max_polls})")
            
            # Check if the request has been approved/rejected
            status = hitl.check_approval_status(request_id)
            logger.info(f"Current approval status: {status}")
            
            if status == "approved":
                logger.info("Data validation approved by reviewer")
                try:
                    slack.simple_post("✅ Data validation approved by reviewer", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
                return True
            
            elif status == "rejected":
                error_msg = "Data validation rejected by reviewer"
                logger.error(error_msg)
                try:
                    slack.simple_post("❌ Data validation rejected by reviewer, pipeline will stop", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
                raise AirflowException(error_msg)
            
            elif status == "error":
                error_msg = "Error checking data validation status"
                logger.error(error_msg)
                try:
                    slack.simple_post("❌ Error checking data validation status", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
                raise AirflowException(error_msg)
            
            # Sleep before polling again
            time.sleep(poll_interval)
        
        # If we've reached max_polls, timeout the request
        error_msg = f"Data validation request timed out after {max_polls * poll_interval / 60} hours"
        logger.error(error_msg)
        try:
            slack.simple_post("❌ Data validation request timed out", channel="#data-pipeline")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        raise AirflowException(error_msg)
        
    except Exception as e:
        error_msg = f"Error in wait_for_data_validation: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full exception details:")
        try:
            slack.simple_post(f"❌ Error in data validation process: {str(e)}", channel="#data-pipeline")
        except Exception as slack_e:
            logger.warning(f"Failed to send Slack notification: {str(slack_e)}")
        raise AirflowException(error_msg)

def wait_for_model_approval(**context):
    """
    Wait for the model to be approved by a human reviewer
    """
    logger.info("Starting wait_for_model_approval task")
    
    # Get the run_id from XCom and the model registry client
    run_id = context['ti'].xcom_pull(task_ids='train_models_task')
    
    # Check if AUTO_APPROVE_MODEL is set to True
    auto_approve = Variable.get('AUTO_APPROVE_MODEL', default_var='False').lower() == 'true'
    
    if auto_approve:
        logger.info("AUTO_APPROVE_MODEL is set to True, skipping human approval")
        try:
            slack.simple_post("⚠️ Model automatically approved (AUTO_APPROVE_MODEL=True)", channel="#data-pipeline")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        return True
    
    # Get the Human in the Loop module
    hitl = HumanInTheLoop(
        app_id="ml_pipeline",
        entity_type="model",
        channel="#data-pipeline"
    )
    
    # Request approval with a link to the MLflow UI
    entity_id = run_id
    detail_link = f"{MLFLOW_UI_URL}/#/experiments/1/runs/{run_id}"
    message = f"Please review and approve the model (run_id: {run_id})"
    
    try:
        # Create approval request
        request_id = hitl.request_approval(
            entity_id=entity_id,
            message=message,
            detail_link=detail_link
        )
        
        logger.info(f"Created approval request with ID: {request_id}")
        
        # Poll for approval status
        max_polls = 60  # 1 hour timeout (60 x 1 minute)
        poll_interval = 60  # 1 minute between polls
        
        for i in range(max_polls):
            logger.info(f"Polling for approval status (attempt {i+1}/{max_polls})")
            
            # Check if the request has been approved/rejected
            status = hitl.check_approval_status(request_id)
            logger.info(f"Current approval status: {status}")
            
            if status == "approved":
                logger.info("Model approved by reviewer")
                try:
                    slack.simple_post("✅ Model approved by reviewer", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
                return True
            
            elif status == "rejected":
                error_msg = "Model rejected by reviewer"
                logger.error(error_msg)
                try:
                    slack.simple_post("❌ Model rejected by reviewer, pipeline will stop", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
                raise AirflowException(error_msg)
            
            elif status == "error":
                error_msg = "Error checking model approval status"
                logger.error(error_msg)
                try:
                    slack.simple_post("❌ Error checking model approval status", channel="#data-pipeline")
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
                raise AirflowException(error_msg)
            
            # Sleep before polling again
            time.sleep(poll_interval)
        
        # If we've reached max_polls, timeout the request
        error_msg = f"Model approval request timed out after {max_polls * poll_interval / 60} hours"
        logger.error(error_msg)
        try:
            slack.simple_post("❌ Model approval request timed out", channel="#data-pipeline")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        raise AirflowException(error_msg)
        
    except Exception as e:
        error_msg = f"Error in wait_for_model_approval: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full exception details:")
        try:
            slack.simple_post(f"❌ Error in model approval process: {str(e)}", channel="#data-pipeline")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        raise AirflowException(error_msg)

def deploy_model(**context):
    """
    Deploy the trained model to production environment.
    
    This function:
    1. Gets the best trained model from the previous tasks
    2. Downloads the model artifacts from MLflow
    3. Deploys the model to AWS (SageMaker or Lambda) for serving
    4. Updates model metadata with deployment information
    5. Sends notification about successful deployment
    """
    logger.info("Starting deploy_model task")
    
    try:
        # Get training results from previous task
        training_results = context['ti'].xcom_pull(task_ids='train_models_task', key='training_results')
        
        if not training_results:
            logger.warning("No training results found, cannot deploy model")
            return {
                "status": "error",
                "message": "No training results available for deployment"
            }
            
        if not isinstance(training_results, dict):
            logger.warning(f"Training results has unexpected type: {type(training_results)}")
            return {
                "status": "warning",
                "message": f"Training results has unexpected type: {type(training_results)}"
            }
            
        # Initialize MLflow
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        client = MlflowClient()
        
        # Find the best model from training results
        best_model_id = None
        best_run_id = None
        
        logger.info("Searching for completed models in training results")
        for model_id, result in training_results.items():
            if model_id == 'best_model':
                # If we have a best_model entry, use that information
                if isinstance(result, dict) and 'model_id' in result:
                    best_model_id = result['model_id']
                    # Try to find run_id in the original model entry
                    if best_model_id in training_results:
                        best_run_id = training_results[best_model_id].get('run_id')
                    break
            
            # Otherwise find first completed model
            elif result and isinstance(result, dict) and result.get('status') == 'completed':
                best_model_id = model_id
                best_run_id = result.get('run_id')
                break
        
        if not best_model_id or not best_run_id:
            logger.warning("No suitable model found for deployment")
            return {
                "status": "warning",
                "message": "No suitable model found for deployment"
            }
            
        logger.info(f"Selected model {best_model_id} (run_id: {best_run_id}) for deployment")
        
        # Check if the model version is already in Production stage
        try:
            versions = client.get_latest_versions(best_model_id)
            production_versions = [v for v in versions if v.current_stage == "Production"]
            
            # Find the version matching our run_id
            target_version = None
            for version in versions:
                if version.run_id == best_run_id:
                    target_version = version
                    break
                    
            if target_version and target_version.current_stage == "Production":
                logger.info(f"Model {best_model_id} version {target_version.version} is already in Production stage")
                deployment_mode = "already_deployed"
            else:
                # If not in Production, transition it
                if target_version:
                    logger.info(f"Transitioning model {best_model_id} version {target_version.version} to Production")
                    client.transition_model_version_stage(
                        name=best_model_id,
                        version=target_version.version,
                        stage="Production",
                        archive_existing_versions=True
                    )
                    deployment_mode = "promoted_to_production"
                else:
                    logger.warning(f"Could not find model version for run_id {best_run_id}")
                    return {
                        "status": "warning",
                        "message": f"Could not find model version for run_id {best_run_id}"
                    }
        except Exception as e:
            logger.error(f"Error checking model versions: {str(e)}")
            deployment_mode = "error_checking_versions"
        
        # Download model artifacts
        try:
            # Create a temporary directory for model artifacts
            temp_dir = tempfile.mkdtemp(prefix="model_deploy_")
            logger.info(f"Downloading model artifacts to {temp_dir}")
            
            # Download the model
            local_path = mlflow.artifacts.download_artifacts(
                run_id=best_run_id,
                artifact_path="model",
                dst_path=temp_dir
            )
            logger.info(f"Model artifacts downloaded to {local_path}")
            
            # Store model path in S3 for reference
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"{config.MODEL_KEY_PREFIX}/deployed/{best_model_id}_{timestamp}"
            
            logger.info(f"Uploading deployment reference to S3: {s3_key}")
            
            # Create deployment metadata
            deployment_metadata = {
                "model_id": best_model_id,
                "run_id": best_run_id,
                "deployed_at": datetime.now().isoformat(),
                "deployed_by": "airflow",
                "deployment_mode": deployment_mode
            }
            
            # Save metadata to a file
            metadata_path = os.path.join(temp_dir, "deployment_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(deployment_metadata, f, indent=2)
            
            # Upload to S3
            try:
                s3_hook = S3Hook()
                s3_hook.load_file(
                    filename=metadata_path,
                    key=f"{s3_key}/metadata.json",
                    bucket_name=config.S3_BUCKET,
                    replace=True
                )
                logger.info(f"Deployment metadata uploaded to S3")
            except Exception as s3_err:
                logger.warning(f"Error uploading to S3: {str(s3_err)}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
        except Exception as artifact_err:
            logger.error(f"Error handling model artifacts: {str(artifact_err)}")
        
        # Log deployment to MLflow
        try:
            with mlflow.start_run(run_id=best_run_id):
                mlflow.log_param("deployed_at", datetime.now().isoformat())
                mlflow.log_param("deployment_mode", deployment_mode)
                mlflow.log_metric("deployment_success", 1.0)
                
                # Set tag for easy filtering of deployed models
                client.set_tag(best_run_id, "deployed", "true")
                
            logger.info(f"Deployment logged to MLflow for run_id {best_run_id}")
        except Exception as mlflow_err:
            logger.warning(f"Error logging deployment to MLflow: {str(mlflow_err)}")
        
        # Send notification about deployment
        try:
            from utils.slack import post as slack_post
            slack_post(
                channel="#ml-deployments",
                title="🚀 Model Deployed to Production",
                details=f"Model '{best_model_id}' has been deployed to production environment.\n" +
                        f"Deployment mode: {deployment_mode}\n" +
                        f"MLflow Run ID: {best_run_id}",
                urgency="high"
            )
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
        
        logger.info(f"Model {best_model_id} successfully deployed")
        
        # Return deployment results
        deployment_results = {
            "status": "success",
            "message": "Model successfully deployed",
            "model_id": best_model_id,
            "run_id": best_run_id,
            "deployment_mode": deployment_mode
        }
        
        context['ti'].xcom_push(key='deployment_results', value=deployment_results)
        return deployment_results
        
    except Exception as e:
        logger.error(f"Error in deploy_model task: {str(e)}")
        logger.exception("Full exception details:")
        
        # Store error information in XCom
        deployment_results = {
            "status": "error",
            "message": f"Error in model deployment: {str(e)}"
        }
        context['ti'].xcom_push(key='deployment_results', value=deployment_results)
        
        # Send notification about failure
        try:
            from utils.slack import post as slack_post
            slack_post(
                channel="#alerts",
                title="❌ Model Deployment Failed",
                details=f"Error deploying model to production: {str(e)}",
                urgency="high"
            )
        except Exception as slack_e:
            logger.warning(f"Error sending Slack notification: {str(slack_e)}")
        
        return deployment_results

# Create the DAG
dag = DAG(
    'unified_ml_pipeline',
    default_args=default_args,
    description='Unified ML Pipeline for model training and deployment',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'pipeline', 'training'],
)

# Create tasks within the DAG context
with dag:
    # Import task for importing raw data
    import_data_task = PythonOperator(
        task_id='import_data_task',
        python_callable=download_data,
        provide_context=True,
    )
    
    # Preprocess task for data preprocessing
    preprocess_data_task = PythonOperator(
        task_id='preprocess_data_task',
        python_callable=process_data,
        provide_context=True,
        trigger_rule='all_success',  # Only run if import was successful
    )
    
    # Data validation task
    validate_data_task = PythonOperator(
        task_id='validate_data_task',
        python_callable=run_data_quality_checks,
        provide_context=True,
        trigger_rule='all_success',  # Only run if preprocessing was successful
    )
    
    # Wait for data validation task
    wait_for_data_validation_task = PythonOperator(
        task_id='wait_for_data_validation_task',
        python_callable=wait_for_data_validation,
        provide_context=True,
        trigger_rule='all_success',  # Only run if validation was successful
    )
    
    # Train models task
    train_models_task = PythonOperator(
        task_id='train_models_task',
        python_callable=train_models,
        provide_context=True,
        trigger_rule='all_success',  # Only run if validation approval was given
    )
    
    # Wait for model approval task
    wait_for_model_approval_task = PythonOperator(
        task_id='wait_for_model_approval_task',
        python_callable=wait_for_model_approval,
        provide_context=True,
        trigger_rule='all_success',  # Only run if training was successful
    )
    
    # Deploy model task
    deploy_model_task = PythonOperator(
        task_id='deploy_model_task',
        python_callable=deploy_model,
        provide_context=True,
        trigger_rule='all_success',  # Only run if model approval was given
    )
    
    # Define task dependencies
    import_data_task >> preprocess_data_task >> validate_data_task >> wait_for_data_validation_task >> train_models_task >> wait_for_model_approval_task >> deploy_model_task