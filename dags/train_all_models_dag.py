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

# Adjust import path to include the dags directory
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable, XCom
from airflow.exceptions import AirflowException
from airflow.hooks.S3_hook import S3Hook
from airflow.providers.amazon.aws.operators.s3 import S3CopyObjectOperator

# Import necessary modules - use direct imports
import tasks.data_prep as data_prep
import tasks.training as training
import tasks.model_comparison as model_comparison
import utils.config as config

logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Number of parallel worker processes (adjust based on memory/CPU constraints)
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

# Task: Fetch and prepare data
def prepare_data_fn(**context):
    """Fetch and prepare data for all models"""
    try:
        # Get data path from Airflow variable or use default
        data_path = Variable.get('DATA_SOURCE_PATH', default_var='s3://data/latest.csv')
        
        logger.info(f"Fetching data from {data_path}")
        
        # Create a temporary directory for outputs
        temp_dir = tempfile.mkdtemp(prefix="airflow_data_")
        
        # Generate processed dataframe
        processed_path = data_prep.prepare_dataset(
            source_path=data_path, 
            output_dir=temp_dir,
            apply_feature_engineering=True
        )
        
        logger.info(f"Data prepared and stored at {processed_path}")
        
        # Push output path to XCom for next task
        context['ti'].xcom_push(key='processed_data_path', value=processed_path)
        
        return processed_path
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise

# Task: Train all models in parallel
def train_all_models_fn(**context):
    """Train all 5 models in parallel using shared data"""
    try:
        # Get processed data path from previous task
        processed_path = context['ti'].xcom_pull(task_ids='prepare_data', key='processed_data_path')
        
        # Check if we want to force parallel training (default: True)
        parallel = Variable.get('PARALLEL_TRAINING', default_var='True').lower() == 'true'
        
        logger.info(f"Training all models using data from {processed_path}")
        logger.info(f"Parallel training: {parallel}, Max workers: {MAX_WORKERS}")
        
        # Train all models
        results = training.train_multiple_models(
            processed_path=processed_path,
            parallel=parallel,
            max_workers=MAX_WORKERS
        )
        
        # Push results to XCom for subsequent tasks
        context['ti'].xcom_push(key='training_results', value=results)
        
        # Count results by status
        completed = sum(1 for r in results.values() if r.get('status') == 'completed')
        skipped = sum(1 for r in results.values() if r.get('status') == 'skipped')
        failed = sum(1 for r in results.values() if r.get('status') == 'failed')
        
        logger.info(f"Training results: {completed} completed, {skipped} skipped, {failed} failed")
        
        # Fail the task if all models failed
        if completed == 0 and skipped == 0:
            raise ValueError(f"All {len(results)} models failed training")
            
        return results
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise

# Task: Archive artifacts to S3
def archive_artifacts_fn(**context):
    """Archive training artifacts to S3"""
    try:
        # Get results from the training task
        results = context['ti'].xcom_pull(task_ids='train_all_models', key='training_results')
        
        # Get current date for organizing artifacts
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Create a hook to interact with S3
        s3_hook = S3Hook()
        
        # Count of successful uploads
        uploaded_count = 0
        
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
                        
                        if os.path.exists(mlflow_artifacts_path):
                            for root, _, files in os.walk(mlflow_artifacts_path):
                                for file in files:
                                    local_path = os.path.join(root, file)
                                    relative_path = os.path.relpath(local_path, mlflow_artifacts_path)
                                    s3_key = f"{s3_dest}{relative_path}"
                                    
                                    # Upload file to S3
                                    s3_hook.load_file(
                                        filename=local_path,
                                        key=s3_key,
                                        bucket_name=config.S3_BUCKET,
                                        replace=True
                                    )
                            
                            uploaded_count += 1
                            logger.info(f"Archived artifacts for {model_id} to S3")
                    except Exception as e:
                        logger.warning(f"Error archiving artifacts for {model_id}: {e}")
        
        logger.info(f"Archived artifacts for {uploaded_count} models to S3")
        return uploaded_count
    except Exception as e:
        logger.error(f"Error archiving artifacts: {e}")
        # Don't fail the DAG if archiving fails
        return 0

# Define tasks
prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data_fn,
    provide_context=True,
    dag=dag
)

train_all_models_task = PythonOperator(
    task_id='train_all_models',
    python_callable=train_all_models_fn,
    provide_context=True,
    dag=dag
)

archive_artifacts_task = PythonOperator(
    task_id='archive_artifacts',
    python_callable=archive_artifacts_fn,
    provide_context=True,
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

# Set task dependencies
wait_for_homeowner >> prepare_data_task >> train_all_models_task >> archive_artifacts_task 