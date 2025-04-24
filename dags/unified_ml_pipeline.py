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
4. Training of multiple models (including parallel training)
5. Model evaluation and explainability tracking
6. Artifact archiving

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
import utils.config as config
import utils.clearml_config as clearml_config
import utils.slack as slack

# Import task modules that are actively used
import tasks.ingestion as ingestion
import tasks.preprocessing as preprocessing  # Primary data processing module
import tasks.data_quality as data_quality
import tasks.schema_validation as schema_validation
import tasks.drift as drift
import tasks.training as training
import tasks.model_explainability as model_explainability

# Additional modules that may be used in specific scenarios
# import tasks.data_prep as data_prep  # Alternative data processing - redundant with preprocessing
# import tasks.ab_testing as ab_testing  # Not directly used in the current pipeline
# import tasks.model_comparison as model_comparison  # Not directly used in the current pipeline

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
LOCAL_PROCESSED_PATH = "/tmp/unified_processed.parquet"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"
MAX_WORKERS = int(Variable.get('MAX_PARALLEL_WORKERS', default_var='3'))
# Add new constant for feature engineering
APPLY_FEATURE_ENGINEERING = Variable.get('APPLY_FEATURE_ENGINEERING', default_var='False').lower() == 'true'

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
                    # Call the preprocess_data function with appropriate parameters
                    processed_data = preprocessing.preprocess_data(
                        data_path=raw_data_path,
                        output_path=processed_path,
                        force_reprocess=True,
                        apply_feature_engineering=apply_feature_engineering
                    )
                    logger.info(f"Data processed successfully to {processed_path}")
                except Exception as e:
                    logger.error(f"Error in data preprocessing: {str(e)}")
                    raise
            
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
            
        # Run quality checks
        try:
            quality_results = data_quality.DataQualityMonitor().run_quality_checks(df)
            logger.info(f"Data quality results: {quality_results}")
            
            # Store results in XCom
            context['ti'].xcom_push(key='quality_results', value=quality_results)
            
            # Determine if quality checks passed
            quality_passed = quality_results.get('status') == 'success'
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
                    slack.post(f":warning: Data quality checks had issues: {quality_results.get('message', 'Unknown issue')}")
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
        
        # Run drift detection
        try:
            drift_results = drift.DriftDetector().check_for_drift(data_path)
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
            logger.error(f"Error running drift detection: {str(e)}")
            raise
            
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
    
    try:
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='process_data', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        data_path = standardized_path if os.path.exists(standardized_path) else processed_path
        
        if not data_path or not os.path.exists(data_path):
            logger.error("No valid processed data path found")
            raise FileNotFoundError("No valid processed data path found")
            
        logger.info(f"Training models using data from {data_path}")
        
        # Load the data and ensure target variable exists
        try:
            df = pd.read_parquet(data_path)
            
            # Check for target column and create if needed
            if 'trgt' not in df.columns:
                if 'pure_premium' in df.columns:
                    # If pure_premium exists, rename it to trgt for consistency
                    logger.info("Renaming 'pure_premium' to 'trgt' for consistency")
                    df['trgt'] = df['pure_premium']
                elif 'il_total' in df.columns and 'eey' in df.columns:
                    # Calculate trgt as il_total / eey
                    logger.info("Creating 'trgt' column from 'il_total' / 'eey'")
                    df['trgt'] = df['il_total'] / df['eey']
                else:
                    logger.error("Cannot create target variable: missing required columns")
                    raise ValueError("Missing columns needed to create target variable")
                
                # Create weight column if not present
                if 'wt' not in df.columns and 'eey' in df.columns:
                    logger.info("Creating 'wt' column from 'eey'")
                    df['wt'] = df['eey']
                
                # Save the updated dataframe
                temp_path = os.path.join(os.path.dirname(data_path), f"model_ready_{os.path.basename(data_path)}")
                df.to_parquet(temp_path, index=False)
                logger.info(f"Saved model-ready dataframe to {temp_path}")
                data_path = temp_path
            
            logger.info(f"Dataframe ready for training with shape {df.shape}")
        except Exception as e:
            logger.error(f"Error preparing data for training: {str(e)}")
            raise
        
        # Check if we want to force parallel training (default: True)
        parallel = Variable.get('PARALLEL_TRAINING', default_var='True').lower() == 'true'
        max_workers = int(Variable.get('MAX_PARALLEL_WORKERS', default_var=str(MAX_WORKERS)))
        
        logger.info(f"Parallel training: {parallel}, Max workers: {max_workers}")
        
        # Train all models
        try:
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
                
                # Send notification with summary
                try:
                    emoji = ":white_check_mark:" if completed > 0 else ":warning:"
                    slack.post(f"{emoji} Training completed: {completed} models trained, {skipped} skipped, {failed} failed")
                except Exception as e:
                    logger.warning(f"Failed to send Slack notification: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in train_models task: {str(e)}")
        
        try:
            slack.post(f":x: Model training failed: {str(e)}")
        except:
            pass
            
        raise

def run_model_explainability(**context):
    """Run model explainability on trained models"""
    logger.info("Starting model_explainability task")
    
    try:
        # Get training results
        training_results = context['ti'].xcom_pull(task_ids='train_models', key='training_results')
        
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='process_data', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        data_path = standardized_path if os.path.exists(standardized_path) else processed_path
        
        if not training_results or not isinstance(training_results, dict):
            logger.warning("No valid training results found")
            return {"status": "warning", "message": "No valid training results found"}
            
        if not data_path or not os.path.exists(data_path):
            logger.error("No valid processed data path found")
            return {"status": "error", "message": "No valid processed data path found"}
            
        logger.info(f"Running model explainability using data from {data_path}")
        
        # Get the best model from training results
        best_model = None
        best_model_id = None
        best_run_id = None
        
        for model_id, result in training_results.items():
            if result.get('status') == 'completed':
                model = result.get('model')
                run_id = result.get('run_id')
                if model:
                    best_model = model
                    best_model_id = model_id
                    best_run_id = run_id
                    break
        
        if not best_model:
            logger.warning("No completed model found in training results")
            return {"status": "warning", "message": "No completed model found in training results"}
            
        logger.info(f"Using model {best_model_id} for explainability tracking")
        
        # Load features and target separately to reduce memory usage
        try:
            X = pd.read_parquet(data_path)
            logger.info(f"Loaded dataframe with shape {X.shape}")
            
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
                
            # Run the explainability tracking
            tracker = model_explainability.ModelExplainabilityTracker(best_model_id)
            explainability_results = tracker.track_model_and_data(model=best_model, X=X, y=y, run_id=best_run_id)
            
            logger.info(f"Explainability tracking results: {explainability_results}")
            
            # Store results in XCom
            context['ti'].xcom_push(key='explainability_results', value=explainability_results)
            
            return explainability_results
            
        except Exception as e:
            logger.error(f"Error in explainability tracking: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    except Exception as e:
        logger.error(f"Error in model_explainability task: {str(e)}")
        return {"status": "error", "message": str(e)}

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

# Create the DAG
dag = DAG(
    'unified_ml_pipeline',
    default_args=default_args,
    description='Unified ML pipeline that combines functionality from multiple previous DAGs',
    schedule_interval=timedelta(days=1),
    max_active_runs=1,
    catchup=False,
    tags=['ml', 'integration', 'unified-pipeline'],
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

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    provide_context=True,
    retries=1,
    retry_delay=timedelta(minutes=10),
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
    execution_timeout=timedelta(hours=1),
    dag=dag
)

archive_artifacts_task = PythonOperator(
    task_id='archive_artifacts',
    python_callable=archive_artifacts,
    provide_context=True,
    retries=3,
    retry_delay=timedelta(minutes=2),
    execution_timeout=timedelta(hours=1),
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    provide_context=True,
    retries=1,
    retry_delay=timedelta(minutes=2),
    execution_timeout=timedelta(minutes=15),
    dag=dag
)

# Set task dependencies
download_data_task >> process_data_task >> [data_quality_task, schema_validation_task, drift_detection_task] >> train_models_task >> model_explainability_task >> archive_artifacts_task >> cleanup_task 