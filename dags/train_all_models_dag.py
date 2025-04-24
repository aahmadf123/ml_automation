#!/usr/bin/env python3
"""
train_all_models_dag.py - Main DAG for training multiple models
--------------------------------------------------------------
This DAG efficiently trains five models in parallel, shares data preparation steps,
and implements intelligent training skip based on historical performance.
"""

import os
import json
import logging
import tempfile
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Adjust import path to include the dags directory
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable, XCom
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.hooks.S3_hook import S3Hook
from airflow.providers.amazon.aws.operators.s3 import S3CopyObjectOperator

# Import necessary modules - use direct imports
import tasks.data_prep as data_prep
import tasks.training as training
import tasks.model_comparison as model_comparison
import utils.config as config
import utils.slack as slack

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

# Task: Fetch and prepare data
def prepare_data_fn(**context):
    """Fetch and prepare data for all models"""
    logger.info("Starting prepare_data task")
    
    try:
        # Get data path from Airflow variable or use default
        data_path = Variable.get('DATA_SOURCE_PATH', default_var='s3://data/latest.csv')
        bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
        
        logger.info(f"Using bucket from Airflow Variables: {bucket}")
        logger.info(f"Fetching data from {data_path}")
        
        # Create a temporary directory for outputs
        temp_dir = tempfile.mkdtemp(prefix="airflow_data_")
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Validate data path
        if not data_path:
            error_msg = "DATA_SOURCE_PATH is empty"
            logger.error(error_msg)
            try:
                slack.post(f":x: Training preparation failed: {error_msg}")
            except:
                pass
            raise ValueError(error_msg)
        
        # Generate processed dataframe
        try:
            logger.info(f"Calling data_prep.prepare_dataset with source_path={data_path}")
            processed_path = data_prep.prepare_dataset(
                source_path=data_path, 
                output_dir=temp_dir,
                apply_feature_engineering=True
            )
            
            # Verify the processed file exists and has content
            if not processed_path or not os.path.exists(processed_path):
                error_msg = f"Processed file not created at expected path: {processed_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            file_size = os.path.getsize(processed_path)
            if file_size == 0:
                error_msg = f"Processed file is empty: {processed_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.info(f"Data prepared and stored at {processed_path} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            try:
                slack.post(f":x: Data preparation failed: {str(e)}")
            except:
                pass
            raise
        
        # Push output path to XCom for next task
        context['ti'].xcom_push(key='processed_data_path', value=processed_path)
        
        # Send success notification
        try:
            slack.post(f":white_check_mark: Data preparation completed successfully: {processed_path}")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return processed_path
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        # Cleanup temp directory if it exists and there was an error
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                for root, dirs, files in os.walk(temp_dir, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                os.rmdir(temp_dir)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temporary directory: {str(cleanup_error)}")
            
        raise

# Task: Train all models in parallel
def train_all_models_fn(**context):
    """Train all 5 models in parallel using shared data"""
    logger.info("Starting train_all_models task")
    
    try:
        # Get processed data path from previous task
        processed_path = context['ti'].xcom_pull(task_ids='prepare_data', key='processed_data_path')
        
        if not processed_path:
            error_msg = "No processed_data_path provided from prepare_data task"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Verify the file exists and has content
        if not os.path.exists(processed_path):
            error_msg = f"Processed data file does not exist: {processed_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        file_size = os.path.getsize(processed_path)
        logger.info(f"Found processed data file {processed_path} with size {file_size} bytes")
        
        if file_size == 0:
            error_msg = f"Processed data file is empty: {processed_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check if we want to force parallel training (default: True)
        parallel = Variable.get('PARALLEL_TRAINING', default_var='True').lower() == 'true'
        max_workers = int(Variable.get('MAX_PARALLEL_WORKERS', default_var=str(MAX_WORKERS)))
        
        logger.info(f"Training all models using data from {processed_path}")
        logger.info(f"Parallel training: {parallel}, Max workers: {max_workers}")
        
        # Train all models
        try:
            results = training.train_multiple_models(
                processed_path=processed_path,
                parallel=parallel,
                max_workers=max_workers
            )
            
            if not results:
                error_msg = "No results returned from train_multiple_models"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.info(f"Training completed with results for {len(results)} models")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            try:
                slack.post(f":x: Model training failed: {str(e)}")
            except:
                pass
            raise
        
        # Push results to XCom for subsequent tasks
        context['ti'].xcom_push(key='training_results', value=results)
        
        # Count results by status
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
        
        # Fail the task if all models failed
        if completed == 0 and skipped == 0:
            error_msg = f"All {len(results)} models failed training"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        return results
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

# Task: Archive artifacts to S3
def archive_artifacts_fn(**context):
    """Archive training artifacts to S3"""
    logger.info("Starting archive_artifacts task")
    
    try:
        # Get results from the training task
        results = context['ti'].xcom_pull(task_ids='train_all_models', key='training_results')
        
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

# Create a test task for validating environment
def test_environment(**context):
    """Test task for validating AWS and MLflow connectivity"""
    logger.info("Starting test_environment task")
    
    test_results = {"status": "success", "errors": []}
    
    try:
        # Test AWS connectivity
        logger.info("Testing AWS connectivity")
        try:
            import boto3
            region = Variable.get("AWS_REGION", default_var="us-east-2")
            s3 = boto3.client('s3', region_name=region)
            response = s3.list_buckets()
            buckets = [b['Name'] for b in response['Buckets']]
            logger.info(f"AWS S3 connectivity OK - found {len(buckets)} buckets")
        except Exception as e:
            error_msg = f"AWS connectivity test failed: {str(e)}"
            test_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Test data_prep module
        logger.info("Testing data_prep module")
        try:
            # Just test module import
            import tasks.data_prep
            logger.info("data_prep module import successful")
        except Exception as e:
            error_msg = f"data_prep module import failed: {str(e)}"
            test_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Test training module
        logger.info("Testing training module")
        try:
            # Just test module import
            import tasks.training
            logger.info("training module import successful")
        except Exception as e:
            error_msg = f"training module import failed: {str(e)}"
            test_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Check for required Airflow variables
        logger.info("Checking required Airflow variables")
        required_vars = ["DATA_BUCKET", "MAX_PARALLEL_WORKERS", "PARALLEL_TRAINING"]
        missing_vars = []
        
        for var_name in required_vars:
            try:
                value = Variable.get(var_name)
                logger.info(f"Airflow variable {var_name} = {value}")
            except:
                missing_vars.append(var_name)
        
        if missing_vars:
            error_msg = f"Missing required Airflow variables: {', '.join(missing_vars)}"
            test_results["errors"].append(error_msg)
            logger.warning(error_msg)
        
        # Update status based on errors
        if test_results["errors"]:
            test_results["status"] = "warning"
            logger.warning(f"Environment test completed with {len(test_results['errors'])} warnings")
        else:
            logger.info("Environment test completed successfully")
        
        # Save results to XCom
        context['ti'].xcom_push(key='test_environment_results', value=test_results)
        
        return test_results
        
    except Exception as e:
        logger.error(f"Environment test failed: {str(e)}")
        test_results["status"] = "error"
        test_results["errors"].append(str(e))
        
        context['ti'].xcom_push(key='test_environment_results', value=test_results)
        
        # Return rather than raise to avoid DAG failure
        return test_results

# Number of parallel worker processes - get dynamically from Variable
MAX_WORKERS = int(Variable.get('MAX_PARALLEL_WORKERS', default_var=3))

# Create DAG
dag = DAG(
    'train_all_models',
    default_args=default_args,
    description='Train all 5 models in parallel with shared data preparation',
    schedule_interval='0 0 * * 0',  # Weekly on Sunday at midnight
    max_active_runs=1,
    catchup=False,
    tags=['training', 'optimization', 'all-models']
)

# Define environment test task
test_environment_task = PythonOperator(
    task_id='test_environment',
    python_callable=test_environment,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=1),
    execution_timeout=timedelta(minutes=10),
    dag=dag
)

# Define tasks with improved configurations
prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data_fn,
    provide_context=True,
    retries=2,
    retry_delay=timedelta(minutes=5),
    execution_timeout=timedelta(hours=4),  # Extended timeout for large datasets
    pool='large_memory_tasks',  # Use a dedicated resource pool
    executor_config={
        "KubernetesExecutor": {
            "request_memory": "8Gi",
            "limit_memory": "16Gi",
            "request_cpu": "2",
            "limit_cpu": "4"
        }
    },
    dag=dag
)

train_all_models_task = PythonOperator(
    task_id='train_all_models',
    python_callable=train_all_models_fn,
    provide_context=True,
    retries=1,
    retry_delay=timedelta(minutes=10),
    execution_timeout=timedelta(hours=8),  # Extended timeout for parallel training
    pool='large_memory_tasks',  # Use a dedicated resource pool
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

archive_artifacts_task = PythonOperator(
    task_id='archive_artifacts',
    python_callable=archive_artifacts_fn,
    provide_context=True,
    retries=3,
    retry_delay=timedelta(minutes=2),
    execution_timeout=timedelta(hours=1),
    dag=dag
)

# Import cross-DAG dependencies utility
from cross_dag_dependencies import wait_for_dag

# Add wait for homeowner DAG
wait_for_homeowner = wait_for_dag(
    dag=dag,
    upstream_dag_id="homeowner_loss_history_full_pipeline",
    timeout=7200,  # 2 hours timeout
    mode="reschedule"
)

# Set updated task dependencies with test_environment first
test_environment_task >> wait_for_homeowner >> prepare_data_task >> train_all_models_task >> archive_artifacts_task 