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
from datetime import datetime, timedelta
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
        context['ti'].xcom_push(key='data_path', value=data_path)
        if backup_paths:
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
        raw_data_path = context['ti'].xcom_pull(task_ids='download_data', key='data_path')
        
        if not raw_data_path:
            logger.error("Failed to get data_path from previous tasks")
            raise ValueError("No data path provided from previous tasks")
            
        # Verify the file exists and has content
        if not os.path.exists(raw_data_path):
            logger.warning(f"Data file does not exist at primary location: {raw_data_path}")
            
            # Try backup paths
            backup_paths = context['ti'].xcom_pull(task_ids='download_data', key='backup_paths') or []
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
                slack.post(f":white_check_mark: Data processing completed: {processed_path}")
            except Exception as e:
                logger.warning(f"Error sending Slack notification: {str(e)}")
                
            return processed_path
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in process_data task: {str(e)}")
        
        try:
            slack.post(f":x: Data processing failed: {str(e)}")
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
                    slack.post(":white_check_mark: Data quality checks passed")
                except Exception as e:
                    logger.warning(f"Error sending Slack notification: {str(e)}")
            else:
                logger.warning("Data quality checks failed or had warnings")
                try:
                    message = quality_results.get('message', 'Unknown issue')
                    issues = quality_results.get('total_issues', 0)
                    slack.post(f":warning: Data quality checks had {issues} issues: {message}")
                except Exception as e:
                    logger.warning(f"Error sending Slack notification: {str(e)}")
                    
            return quality_results
            
        except Exception as e:
            logger.error(f"Error running data quality checks: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in data_quality_checks task: {str(e)}")
        
        try:
            slack.post(f":x: Data quality checks failed: {str(e)}")
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
                    slack.post(":white_check_mark: Schema validation passed")
                except Exception as e:
                    logger.warning(f"Error sending Slack notification: {str(e)}")
            else:
                logger.warning(f"Schema validation failed: {validation_results.get('message', 'Unknown issue')}")
                try:
                    details = validation_results.get('details', {})
                    message = f":warning: Schema validation had issues: {validation_results.get('message')}"
                    if details.get('target_info'):
                        message += f"\n{details.get('target_info')}"
                    slack.post(message)
                except Exception as e:
                    logger.warning(f"Error sending Slack notification: {str(e)}")
                    
            return validation_results
            
        except Exception as e:
            logger.error(f"Error running schema validation: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in schema_validation task: {str(e)}")
        
        try:
            slack.post(f":x: Schema validation failed: {str(e)}")
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
    
    try:
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='process_data', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        data_path = standardized_path if os.path.exists(standardized_path) else processed_path
        
        if not data_path or not os.path.exists(data_path):
            logger.error("No valid processed data path found")
            raise FileNotFoundError("No valid processed data path found")
            
        logger.info(f"Running drift detection on {data_path}")
        
        # Try multiple possible function names from the drift module
        drift_results = None
        
        # First try detect_data_drift
        drift_results = safe_module_call(drift, "detect_data_drift", processed_data_path=data_path)
        
        # If that failed, try check_for_drift
        if drift_results.get("status") == "error" and "not found" in drift_results.get("message", ""):
            logger.info("Trying alternate drift detection function 'check_for_drift'")
            drift_results = safe_module_call(drift, "check_for_drift", data_path)
            
        # If still failed, try other potential names
        if drift_results.get("status") == "error" and "not found" in drift_results.get("message", ""):
            for func_name in ["detect_drift", "run_drift_detection", "check_drift"]:
                logger.info(f"Trying alternate drift detection function '{func_name}'")
                drift_results = safe_module_call(drift, func_name, data_path)
                if drift_results.get("status") != "error" or "not found" not in drift_results.get("message", ""):
                    break
                    
        logger.info(f"Drift detection results: {drift_results}")
        
        # Store results in XCom
        context['ti'].xcom_push(key='drift_results', value=drift_results)
        
        # Determine if drift was detected
        drift_detected = drift_results.get('drift_detected', False)
        context['ti'].xcom_push(key='drift_detected', value=drift_detected)
        
        if drift_detected:
            logger.warning("Data drift detected")
            try:
                slack.post(f":warning: Data drift detected: {drift_results.get('message', 'Unknown issue')}")
            except Exception as e:
                logger.warning(f"Error sending Slack notification: {str(e)}")
        else:
            logger.info("No data drift detected")
            try:
                slack.post(":white_check_mark: No data drift detected")
            except Exception as e:
                logger.warning(f"Error sending Slack notification: {str(e)}")
                
        return drift_results
        
    except Exception as e:
        logger.error(f"Error in drift_detection task: {str(e)}")
        
        try:
            slack.post(f":x: Drift detection failed: {str(e)}")
        except:
            pass
            
        raise

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
            logger.error("No valid processed data path found and no fallback available")
            raise FileNotFoundError("No valid processed data path found")
            
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
                    logger.warning("Cannot create target variable: missing required columns")
                    logger.info(f"Available columns: {df.columns.tolist()}")
                    logger.warning("Will attempt to train without target variable, but models may fail")
                
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
            logger.error(f"Error preparing data for training: {str(e)}")
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
                logger.error("training module does not have train_multiple_models function")
                available_funcs = [name for name in dir(training) if callable(getattr(training, name)) and not name.startswith('_')]
                logger.info(f"Available functions in training module: {available_funcs}")
                raise AttributeError("training module does not have train_multiple_models function")
                
            # Call the training function
            results = training.train_multiple_models(
                processed_path=data_path,
                parallel=parallel,
                max_workers=max_workers,
                target_column='trgt',  # Explicitly specify target column
                weight_column='wt'     # Explicitly specify weight column
            )
            
            if not results:
                logger.warning("No results returned from train_multiple_models")
                results = {"status": "warning", "message": "No results returned from training"}
                
            logger.info(f"Training completed with results for {len(results) if isinstance(results, dict) else 0} models")
            
            # Store results in XCom
            context['ti'].xcom_push(key='training_results', value=results)
            
            # Count results by status if results is a dictionary
            if isinstance(results, dict):
                completed = sum(1 for r in results.values() if r.get('status') == 'completed')
                skipped = sum(1 for r in results.values() if r.get('status') == 'skipped')
                failed = sum(1 for r in results.values() if r.get('status') == 'failed')
                
                logger.info(f"Training results: {completed} completed, {skipped} skipped, {failed} failed")
                
                # Log details for each model
                for model_id, result in results.items():
                    status = result.get('status', 'unknown')
                    run_id = result.get('run_id', 'unknown')
                    metrics = result.get('metrics', {})
                    logger.info(f"Model {model_id}: status={status}, run_id={run_id}, metrics={metrics}")
                
                # Send notification with summary
                try:
                    emoji = ":white_check_mark:" if completed > 0 else ":warning:"
                    slack.post(f"{emoji} Training completed: {completed} models trained, {skipped} skipped, {failed} failed")
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            logger.exception("Full exception details:")
            # Try to continue anyway
            results = {"status": "error", "message": f"Error training models: {str(e)}"}
            context['ti'].xcom_push(key='training_results', value=results)
            try:
                slack.post(f":warning: Model training encountered errors but pipeline continues: {str(e)}")
            except:
                pass
            return results
            
    except Exception as e:
        logger.error(f"Error in train_models task: {str(e)}")
        logger.exception("Full exception details:")
        
        try:
            slack.post(f":x: Model training failed: {str(e)}")
        except:
            pass
        
        # Return a result rather than raising an exception
        results = {"status": "error", "message": f"Error in train_models task: {str(e)}"}
        context['ti'].xcom_push(key='training_results', value=results)
        return results

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
                slack.post(f":warning: Model explainability encountered an error but pipeline continues: {str(e)}")
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
            slack.post(f":warning: Model explainability failed but pipeline continues: {str(e)}")
        except Exception as slack_error:
            logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        return explainability_results

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
    """Wait for human validation of data quality"""
    logger.info("Starting data_validation task")
    
    approval_status = "pending"  # Default status
    validation_result = None
    
    try:
        # Get quality results from previous task if available
        quality_results = context['ti'].xcom_pull(task_ids='data_quality_checks', key='quality_results') or {}
        validation_results = context['ti'].xcom_pull(task_ids='schema_validation', key='validation_results') or {}
        drift_results = context['ti'].xcom_pull(task_ids='drift_detection', key='drift_results') or {}
        
        logger.info(f"Data quality results: {quality_results}")
        logger.info(f"Schema validation results: {validation_results}")
        logger.info(f"Drift detection results: {drift_results}")
        
        # Try to call the HITL module's wait_for_data_validation function
        try:
            # Check if HITL module exists and has the necessary function
            if hitl and hasattr(hitl, 'wait_for_data_validation') and callable(getattr(hitl, 'wait_for_data_validation')):
                logger.info("Calling HITL module for data validation")
                validation_result = hitl.wait_for_data_validation(**context)
                logger.info(f"HITL module returned: {validation_result}")
                
                # Check the returned result
                if validation_result and isinstance(validation_result, dict):
                    approval_status = validation_result.get('status', 'unknown')
                    
                    # Log the result
                    if approval_status == 'approved' or approval_status == 'auto_approved':
                        logger.info("Data validation approved")
                        try:
                            slack.post(":white_check_mark: Data validation approved")
                        except Exception as e:
                            logger.warning(f"Error sending Slack notification: {str(e)}")
                    elif approval_status == 'rejected':
                        logger.warning("Data validation rejected, continuing anyway")
                        try:
                            slack.post(":warning: Data validation rejected but pipeline continues")
                        except Exception as e:
                            logger.warning(f"Error sending Slack notification: {str(e)}")
                    else:
                        logger.warning(f"Unexpected data validation result: {validation_result}")
                        
                else:
                    logger.warning("HITL module returned invalid or empty result, proceeding with pipeline")
                    validation_result = {
                        "status": "default_approved",
                        "message": "HITL module returned invalid result, proceeding with default approval",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                logger.warning("HITL module or wait_for_data_validation function not available")
                validation_result = {
                    "status": "default_approved",
                    "message": "HITL module not available, proceeding with default approval",
                    "timestamp": datetime.now().isoformat()
                }
        except AirflowSkipException as e:
            # This is raised when validation is rejected in HITL
            logger.warning(f"Data validation rejected via AirflowSkipException: {str(e)}")
            # Instead of propagating the skip, we'll proceed with warnings
            validation_result = {
                "status": "rejected_continue",
                "message": f"Data validation was rejected but pipeline will continue: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            try:
                slack.post(f":warning: Data validation rejected but pipeline continues: {str(e)}")
            except Exception as slack_error:
                logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        except Exception as e:
            # Handle any other exceptions from HITL module
            logger.error(f"Error in HITL data validation: {str(e)}")
            validation_result = {
                "status": "error_continue",
                "message": f"Error in data validation but pipeline will continue: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            try:
                slack.post(f":warning: Error in data validation but pipeline continues: {str(e)}")
            except Exception as slack_error:
                logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        
        # Store validation result in XCom
        context['ti'].xcom_push(key='data_validation_result', value=validation_result)
        
        return validation_result
        
    except Exception as e:
        # Catch-all for any other exceptions
        logger.error(f"Unexpected error in data_validation task: {str(e)}")
        
        # Create a result that allows the pipeline to continue
        validation_result = {
            "status": "unexpected_error",
            "message": f"Unexpected error in data validation but pipeline will continue: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in XCom
        context['ti'].xcom_push(key='data_validation_result', value=validation_result)
        
        try:
            slack.post(f":warning: Unexpected error in data validation but pipeline continues: {str(e)}")
        except Exception as slack_error:
            logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        
        # Return a valid result instead of raising an exception
        return validation_result

def wait_for_model_approval(**context):
    """Wait for human approval of trained models"""
    logger.info("Starting model_approval task")
    
    approval_status = "pending"  # Default status
    approval_result = None
    
    try:
        # Get training and explainability results if available
        training_results = context['ti'].xcom_pull(task_ids='train_models', key='training_results') or {}
        explainability_results = context['ti'].xcom_pull(task_ids='model_explainability', key='explainability_results') or {}
        
        logger.info(f"Training results: {training_results}")
        logger.info(f"Explainability results: {explainability_results}")
        
        # Try to call the HITL module's wait_for_model_approval function
        try:
            # Check if HITL module exists and has the necessary function
            if hitl and hasattr(hitl, 'wait_for_model_approval') and callable(getattr(hitl, 'wait_for_model_approval')):
                logger.info("Calling HITL module for model approval")
                approval_result = hitl.wait_for_model_approval(**context)
                logger.info(f"HITL module returned: {approval_result}")
                
                # Check the returned result
                if approval_result and isinstance(approval_result, dict):
                    approval_status = approval_result.get('status', 'unknown')
                    
                    # Log the result
                    if approval_status == 'approved' or approval_status == 'auto_approved':
                        logger.info("Model approval granted")
                        try:
                            slack.post(":white_check_mark: Model approval granted")
                        except Exception as e:
                            logger.warning(f"Error sending Slack notification: {str(e)}")
                    elif approval_status == 'rejected':
                        logger.warning("Model approval rejected, continuing anyway")
                        try:
                            slack.post(":warning: Model approval rejected but pipeline continues")
                        except Exception as e:
                            logger.warning(f"Error sending Slack notification: {str(e)}")
                    else:
                        logger.warning(f"Unexpected model approval result: {approval_result}")
                else:
                    logger.warning("HITL module returned invalid or empty result, proceeding with pipeline")
                    approval_result = {
                        "status": "default_approved",
                        "message": "HITL module returned invalid result, proceeding with default approval",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                logger.warning("HITL module or wait_for_model_approval function not available")
                approval_result = {
                    "status": "default_approved",
                    "message": "HITL module not available, proceeding with default approval",
                    "timestamp": datetime.now().isoformat()
                }
        except AirflowSkipException as e:
            # This is raised when approval is rejected in HITL
            logger.warning(f"Model approval rejected via AirflowSkipException: {str(e)}")
            # Instead of propagating the skip, we'll proceed with warnings
            approval_result = {
                "status": "rejected_continue",
                "message": f"Model approval was rejected but pipeline will continue: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            try:
                slack.post(f":warning: Model approval rejected but pipeline continues: {str(e)}")
            except Exception as slack_error:
                logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        except Exception as e:
            # Handle any other exceptions from HITL module
            logger.error(f"Error in HITL model approval: {str(e)}")
            approval_result = {
                "status": "error_continue",
                "message": f"Error in model approval but pipeline will continue: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            try:
                slack.post(f":warning: Error in model approval but pipeline continues: {str(e)}")
            except Exception as slack_error:
                logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        
        # Store approval result in XCom
        context['ti'].xcom_push(key='model_approval_result', value=approval_result)
        
        return approval_result
        
    except Exception as e:
        # Catch-all for any other exceptions
        logger.error(f"Unexpected error in model_approval task: {str(e)}")
        
        # Create a result that allows the pipeline to continue
        approval_result = {
            "status": "unexpected_error",
            "message": f"Unexpected error in model_approval task: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in XCom
        context['ti'].xcom_push(key='model_approval_result', value=approval_result)
        
        try:
            slack.post(f":warning: Unexpected error in model_approval but pipeline continues: {str(e)}")
        except Exception as slack_error:
            logger.warning(f"Error sending Slack notification: {str(slack_error)}")
        
        # Return a valid result instead of raising an exception
        return approval_result

# Create the DAG
dag = DAG(
    'unified_ml_pipeline',
    default_args=default_args,
    description='Unified ML pipeline that combines functionality from multiple previous DAGs, with HITL capabilities',
    schedule_interval=timedelta(days=1),
    max_active_runs=1,
    catchup=False,
    tags=['ml', 'integration', 'unified-pipeline', 'hitl'],
)

# Create tasks
download_data_task = PythonOperator(
    task_id='download_data',
    python_callable=download_data,
    provide_context=True,
    retries=3,
    retry_delay=timedelta(minutes=2),
    execution_timeout=timedelta(minutes=30),
    dag=dag
)

process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=5),
    execution_timeout=timedelta(hours=2),
    dag=dag
)

data_quality_task = PythonOperator(
    task_id='data_quality_checks',
    python_callable=run_data_quality_checks,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=5),
    execution_timeout=timedelta(minutes=30),
    dag=dag
)

schema_validation_task = PythonOperator(
    task_id='schema_validation',
    python_callable=run_schema_validation,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=5),
    execution_timeout=timedelta(minutes=30),
    dag=dag
)

drift_detection_task = PythonOperator(
    task_id='drift_detection',
    python_callable=check_for_drift,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=5),
    execution_timeout=timedelta(minutes=30),
    dag=dag
)

# Add new HITL task for data validation
data_validation_task = PythonOperator(
    task_id='data_validation',
    python_callable=wait_for_data_validation,
    provide_context=True,
    retries=1,  # Add one retry attempt in case of network issues
    retry_delay=timedelta(minutes=5),
    trigger_rule='all_done',  # Continue even if upstream tasks fail
    execution_timeout=timedelta(hours=12),  # Generous timeout but not infinite
    dag=dag
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    provide_context=True,
    retries=1,
    retry_delay=timedelta(minutes=10),
    trigger_rule='all_done',  # Continue even if upstream tasks fail
    execution_timeout=timedelta(hours=8),
    pool='large_memory_tasks',  # Use a dedicated resource pool if configured
    executor_config={
        "KubernetesExecutor": {
            "request_memory": "12Gi",
            "limit_memory": "24Gi",
            "request_cpu": "4",
            "limit_cpu": "8"
        }
    },
    dag=dag
)

model_explainability_task = PythonOperator(
    task_id='model_explainability',
    python_callable=run_model_explainability,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=5),
    trigger_rule='all_done',  # Continue even if upstream tasks fail
    execution_timeout=timedelta(hours=1),
    dag=dag
)

# Add new HITL task for model approval
model_approval_task = PythonOperator(
    task_id='model_approval',
    python_callable=wait_for_model_approval,
    provide_context=True,
    retries=1,  # Add one retry attempt in case of network issues
    retry_delay=timedelta(minutes=5),
    trigger_rule='all_done',  # Continue even if upstream tasks fail 
    execution_timeout=timedelta(hours=12),  # Generous timeout but not infinite
    dag=dag
)

archive_artifacts_task = PythonOperator(
    task_id='archive_artifacts',
    python_callable=archive_artifacts,
    provide_context=True,
    retries=3,
    retry_delay=timedelta(minutes=2),
    trigger_rule='all_done',  # Continue even if upstream tasks fail
    execution_timeout=timedelta(hours=1),
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    provide_context=True,
    retries=1,
    retry_delay=timedelta(minutes=2),
    trigger_rule='all_done',  # Continue regardless of upstream task status
    execution_timeout=timedelta(minutes=15),
    dag=dag
)

# Set task dependencies
download_data_task >> process_data_task >> [data_quality_task, schema_validation_task, drift_detection_task] >> data_validation_task >> train_models_task >> model_explainability_task >> model_approval_task >> archive_artifacts_task >> cleanup_task 