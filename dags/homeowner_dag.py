#!/usr/bin/env python3
"""
Homeowner Loss History Pipeline DAG.

This DAG manages the complete pipeline for processing homeowner loss history data:
1. Data ingestion and preprocessing
2. Schema validation and monitoring
3. Model training and evaluation
4. Testing and deployment

The pipeline includes automated monitoring, drift detection, and self-healing capabilities.
"""

# Core Python imports
import os
import json
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, Union
from functools import wraps
import sys

# Airflow imports
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable, XCom
from airflow.utils.dates import days_ago
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.hooks.S3_hook import S3Hook
from airflow.sensors.external_task import ExternalTaskSensor
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Constants
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.parquet"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"

# Default arguments for the DAG
default_args: Dict[str, Any] = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
    'catchup': False
}

def validate_file_exists(filepath: str) -> bool:
    """
    Validate that a file exists and is not empty.
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        bool: True if the file exists and is not empty
    """
    if not filepath:
        log.error("Filepath is empty")
        return False
        
    if not os.path.exists(filepath):
        log.error(f"File does not exist: {filepath}")
        return False
        
    if os.path.getsize(filepath) == 0:
        log.error(f"File is empty: {filepath}")
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
                log.error(f"File validation failed for {file_param}: {filepath}")
                return {"status": "error", "message": f"File not found or empty: {filepath}"}
            return func(*args, **kwargs)
        return wrapper
    return decorator

def load_data(filepath: str, columns: Optional[list] = None, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Centralized function to load data from a parquet file.
    
    Args:
        filepath: Path to the parquet file
        columns: Specific columns to load, or None for all columns
        nrows: Number of rows to load, or None for all rows
        
    Returns:
        pd.DataFrame or None: The loaded dataframe or None if the file does not exist
    """
    try:
        if not validate_file_exists(filepath):
            return None
            
        # Log memory usage before loading
        log.info(f"Loading data from {filepath} with columns={columns}, nrows={nrows}")
        
        # Load the data
        df = pd.read_parquet(filepath, engine='pyarrow', columns=columns)
        
        # Apply row limit if specified
        if nrows is not None and nrows > 0:
            df = df.head(nrows)
            
        log.info(f"Successfully loaded dataframe with shape {df.shape}")
        return df
        
    except Exception as e:
        log.error(f"Error loading data from {filepath}: {str(e)}")
        return None

def setup_dag() -> DAG:
    """
    Create and configure the Homeowner Loss History DAG.
    
    Returns:
        DAG: Configured Airflow DAG instance
    """
    return DAG(
        'homeowner_loss_history_full_pipeline',
        default_args=default_args,
        description='Full pipeline for homeowner loss history processing',
        schedule_interval='0 0 * * *',  # Daily at midnight
        max_active_runs=1,
        tags=['ml', 'homeowner', 'loss_history'],
        on_failure_callback=notify_failure,
    )

def notify_failure(context):
    """
    Send a notification when the DAG fails.
    
    Args:
        context: Airflow context containing task instance and other info
    """
    task_instance = context.get('task_instance')
    task_id = task_instance.task_id
    dag_id = task_instance.dag_id
    execution_date = context.get('execution_date')
    
    log.error(f"DAG {dag_id} task {task_id} failed on {execution_date}")
    
    # Try to send a Slack notification
    try:
        from utils import slack
        slack.post(f":x: DAG {dag_id} task {task_id} failed on {execution_date}")
    except Exception as e:
        log.warning(f"Could not send failure notification: {str(e)}")

def validate_processed_file(**context) -> str:
    """
    Validate that the processed file exists before continuing.
    
    Args:
        context: Airflow context
        
    Returns:
        str: Next task to execute based on validation result
    """
    if validate_file_exists(LOCAL_PROCESSED_PATH):
        return 'continue_pipeline'
    else:
        return 'handle_missing_data'

def handle_missing_data(**context):
    """
    Handle the case where processed data is missing.
    
    Args:
        context: Airflow context
    """
    log.error(f"Processed data file missing or empty: {LOCAL_PROCESSED_PATH}")
    
    try:
        from utils import slack
        slack.post(f":x: Pipeline failed: Processed data file missing or empty")
    except Exception as e:
        log.warning(f"Failed to send Slack notification: {str(e)}")
        
    raise AirflowException(f"Processed data file missing or empty: {LOCAL_PROCESSED_PATH}")

def run_schema_validation(**context) -> dict:
    """
    Run schema validation on the processed data.
    
    Args:
        context: Airflow context
        
    Returns:
        dict: Validation results
    """
    import tasks.schema_validation as schema_validation
    
    try:
        if not validate_file_exists(LOCAL_PROCESSED_PATH):
            return {"status": "error", "message": "Data file not found"}
            
        df = load_data(LOCAL_PROCESSED_PATH)
        if df is None:
            return {"status": "error", "message": "Failed to load data"}
            
        return schema_validation.validate_schema(df)
        
    except Exception as e:
        log.error(f"Error in schema validation: {str(e)}")
        return {"status": "error", "message": str(e)}

def run_schema_snapshot(**context) -> dict:
    """
    Create a snapshot of the current schema.
    
    Args:
        context: Airflow context
        
    Returns:
        dict: Snapshot results
    """
    import tasks.schema_validation as schema_validation
    
    try:
        # We only need the column dtypes, so load with a small sample
        df = load_data(LOCAL_PROCESSED_PATH, nrows=1000)
        if df is None:
            return {"status": "error", "message": "Failed to load data"}
            
        return schema_validation.snapshot_schema(df)
        
    except Exception as e:
        log.error(f"Error in schema snapshot: {str(e)}")
        return {"status": "error", "message": str(e)}

def run_data_quality_checks(**context) -> dict:
    """
    Run data quality checks on the processed data.
    
    Args:
        context: Airflow context
        
    Returns:
        dict: Quality check results
    """
    import tasks.data_quality as data_quality
    
    try:
        df = load_data(LOCAL_PROCESSED_PATH)
        if df is None:
            return {"status": "error", "message": "Failed to load data"}
            
        return data_quality.DataQualityMonitor().run_quality_checks(df)
        
    except Exception as e:
        log.error(f"Error in data quality checks: {str(e)}")
        return {"status": "error", "message": str(e)}

def run_explainability_tracking(**context):
    """
    Run explainability tracking on the trained model.
    
    Args:
        context: Airflow context
        
    Returns:
        dict: Explainability tracking results
    """
    import tasks.model_explainability as model_explainability
    
    try:
        # Get the trained model from XCom
        ti = context['ti']
        model = ti.xcom_pull(task_ids='train_compare_model1')
        run_id = ti.xcom_pull(task_ids='train_compare_model1', key='run_id')
        
        if not model:
            log.warning("No model available from train_compare_model1 task")
            return {"status": "warning", "message": "No model available"}
            
        # Load features and target separately to reduce memory usage
        X = load_data(LOCAL_PROCESSED_PATH)
        if X is None:
            return {"status": "error", "message": "Failed to load feature data"}
            
        # Remove the target column
        y = X.pop('claim_amount') if 'claim_amount' in X.columns else None
        
        if y is None:
            log.warning("Target column 'claim_amount' not found in dataset")
            return {"status": "warning", "message": "Target column not found"}
            
        # Run the explainability tracking
        tracker = model_explainability.ModelExplainabilityTracker('homeowner_model')
        return tracker.track_model_and_data(model=model, X=X, y=y, run_id=run_id)
        
    except Exception as e:
        log.error(f"Error in explainability tracking: {str(e)}")
        return {"status": "error", "message": str(e)}

def run_ab_testing(**context):
    """
    Run A/B testing on the trained model.
    
    Args:
        context: Airflow context
        
    Returns:
        dict: A/B testing results
    """
    import tasks.ab_testing as ab_testing
    
    try:
        # Get the trained model from XCom
        ti = context['ti']
        new_model = ti.xcom_pull(task_ids='train_compare_model1')
        
        if not new_model:
            log.warning("No model available from train_compare_model1 task")
            return {"status": "warning", "message": "No model available"}
            
        # Load features and target separately to reduce memory usage
        X = load_data(LOCAL_PROCESSED_PATH)
        if X is None:
            return {"status": "error", "message": "Failed to load feature data"}
            
        # Remove the target column
        y = X.pop('claim_amount') if 'claim_amount' in X.columns else None
        
        if y is None:
            log.warning("Target column 'claim_amount' not found in dataset")
            return {"status": "warning", "message": "Target column not found"}
            
        # Run the A/B testing
        ab_pipeline = ab_testing.ABTestingPipeline('homeowner_model')
        return ab_pipeline.run_ab_test(new_model=new_model, X_test=X, y_test=y)
        
    except Exception as e:
        log.error(f"Error in A/B testing: {str(e)}")
        return {"status": "error", "message": str(e)}

def cleanup_temp_files(**context):
    """
    Clean up temporary files created during pipeline execution.
    
    Args:
        context: Airflow context
    """
    temp_files = [LOCAL_PROCESSED_PATH, REFERENCE_MEANS_PATH]
    
    for filepath in temp_files:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                log.info(f"Removed temporary file: {filepath}")
            except Exception as e:
                log.warning(f"Failed to remove temporary file {filepath}: {str(e)}")
                
    # Clean up any temporary directories in /tmp that match our pattern
    try:
        for item in os.listdir('/tmp'):
            if item.startswith('airflow_') and os.path.isdir(os.path.join('/tmp', item)):
                try:
                    shutil.rmtree(os.path.join('/tmp', item))
                    log.info(f"Removed temporary directory: /tmp/{item}")
                except Exception as e:
                    log.warning(f"Failed to remove temporary directory /tmp/{item}: {str(e)}")
    except Exception as e:
        log.warning(f"Failed to list /tmp directory: {str(e)}")
        
    return {"status": "success", "message": "Cleanup completed"}

def create_tasks(dag: DAG) -> Dict[str, Union[PythonOperator, BranchPythonOperator, EmptyOperator, ExternalTaskSensor]]:
    """
    Create all tasks for the DAG.
    
    Args:
        dag: The DAG instance to add tasks to
        
    Returns:
        Dict[str, Union[PythonOperator, BranchPythonOperator, EmptyOperator, ExternalTaskSensor]]: Dictionary of task operators
    """
    # Import task modules - moving them here to avoid circular imports
    from tasks import ingestion, preprocessing, drift, training, data_quality, monitoring
    
    # Create a sensor to wait for the integrated_ml_workflow DAG
    wait_for_integrated = ExternalTaskSensor(
        task_id='wait_for_integrated_workflow',
        external_dag_id='integrated_ml_workflow',
        external_task_id=None,  # Wait for the entire DAG
        timeout=7200,  # 2 hours
        mode='reschedule',
        poke_interval=60,  # Check every minute
        retries=3,
        retry_delay=timedelta(minutes=5),
        dag=dag
    )
    
    # Task to mark the start of our pipeline after dependency is met
    start_task = EmptyOperator(
        task_id='start_pipeline',
        dag=dag
    )
    
    # Data ingestion and preprocessing tasks
    ingest_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingestion.ingest_data_from_s3,
        provide_context=True,
        retries=3,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(minutes=30),
        dag=dag
    )
    
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocessing.preprocess_data,
        op_kwargs={
            'data_path': "{{ ti.xcom_pull(task_ids='ingest_data') }}",
            'output_path': LOCAL_PROCESSED_PATH,
            'force_reprocess': False
        },
        provide_context=True,
        retries=2,
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(hours=2),
        executor_config={
            "KubernetesExecutor": {
                "request_memory": "8Gi",
                "limit_memory": "12Gi",
                "request_cpu": "2",
                "limit_cpu": "4"
            }
        },
        dag=dag
    )
    
    # Branching task to validate processed file exists
    validate_file_task = BranchPythonOperator(
        task_id='validate_processed_file',
        python_callable=validate_processed_file,
        provide_context=True,
        dag=dag
    )
    
    # Task to handle missing data
    handle_missing_data_task = PythonOperator(
        task_id='handle_missing_data',
        python_callable=handle_missing_data,
        provide_context=True,
        dag=dag
    )
    
    # Task to continue pipeline after validation
    continue_pipeline = EmptyOperator(
        task_id='continue_pipeline',
        dag=dag
    )
    
    # Schema validation tasks with proper functions instead of lambdas
    validate_task = PythonOperator(
        task_id='validate_schema',
        python_callable=run_schema_validation,
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=20),
        executor_config={
            "KubernetesExecutor": {
                "request_memory": "2Gi",
                "limit_memory": "4Gi"
            }
        },
        dag=dag
    )
    
    snapshot_task = PythonOperator(
        task_id='snapshot_schema',
        python_callable=run_schema_snapshot,
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=15),
        executor_config={
            "KubernetesExecutor": {
                "request_memory": "1Gi",
                "limit_memory": "2Gi"
            }
        },
        dag=dag
    )
    
    # Monitoring tasks with proper functions instead of lambdas
    quality_monitor = PythonOperator(
        task_id='monitor_data_quality',
        python_callable=run_data_quality_checks,
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=30),
        executor_config={
            "KubernetesExecutor": {
                "request_memory": "2Gi",
                "limit_memory": "4Gi"
            }
        },
        dag=dag
    )
    
    reference_task = PythonOperator(
        task_id='generate_reference_means',
        python_callable=drift.generate_reference_means,
        op_kwargs={
            'processed_data_path': LOCAL_PROCESSED_PATH
        },
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=20),
        dag=dag
    )
    
    drift_task = PythonOperator(
        task_id='detect_drift',
        python_callable=drift.detect_data_drift,
        op_kwargs={
            'processed_data_path': LOCAL_PROCESSED_PATH
        },
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=30),
        dag=dag
    )
    
    healing_task = PythonOperator(
        task_id='self_healing',
        python_callable=drift.self_healing,
        op_kwargs={
            'drift_results': "{{ ti.xcom_pull(task_ids='detect_drift') }}",
            'processed_data_path': LOCAL_PROCESSED_PATH
        },
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=40),
        dag=dag
    )
    
    metrics_task = PythonOperator(
        task_id='record_metrics',
        python_callable=monitoring.record_system_metrics,
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=15),
        dag=dag
    )
    
    ui_task = PythonOperator(
        task_id='update_ui',
        python_callable=monitoring.update_monitoring_with_ui_components,
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=15),
        dag=dag
    )
    
    # Model tasks with proper functions instead of lambdas
    explainability_tracker = PythonOperator(
        task_id='track_explainability',
        python_callable=run_explainability_tracking,
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=60),
        executor_config={
            "KubernetesExecutor": {
                "request_memory": "4Gi",
                "limit_memory": "6Gi"
            }
        },
        dag=dag
    )
    
    ab_testing_task = PythonOperator(
        task_id='ab_testing',
        python_callable=run_ab_testing,
        provide_context=True,
        retries=2,
        execution_timeout=timedelta(minutes=60),
        executor_config={
            "KubernetesExecutor": {
                "request_memory": "4Gi",
                "limit_memory": "6Gi"
            }
        },
        dag=dag
    )
    
    train_task = PythonOperator(
        task_id='train_compare_model1',
        python_callable=training.train_and_compare_fn,
        op_kwargs={
            'model_id': 'homeowner_model',
            'processed_path': LOCAL_PROCESSED_PATH
        },
        provide_context=True,
        retries=1,
        execution_timeout=timedelta(hours=4),
        executor_config={
            "KubernetesExecutor": {
                "request_memory": "8Gi",
                "limit_memory": "12Gi",
                "request_cpu": "2",
                "limit_cpu": "4"
            }
        },
        dag=dag
    )
    
    # Cleanup task to run at the end or on failure
    cleanup_task = PythonOperator(
        task_id='cleanup_temp_files',
        python_callable=cleanup_temp_files,
        provide_context=True,
        retries=3,
        retry_delay=timedelta(minutes=1),
        trigger_rule=TriggerRule.ALL_DONE,  # Run this task even if upstream tasks fail
        dag=dag
    )
    
    # Manual override task
    override_task = PythonOperator(
        task_id='manual_override',
        python_callable=data_quality.manual_override,
        op_kwargs={
            'model_id': 'homeowner_model',
            'override_action': 'No action required'
        },
        provide_context=True,
        dag=dag
    )
    
    return {
        'wait_for_integrated': wait_for_integrated,
        'start_pipeline': start_task,
        'ingest': ingest_task,
        'preprocess': preprocess_task,
        'validate_file': validate_file_task,
        'handle_missing_data': handle_missing_data_task,
        'continue_pipeline': continue_pipeline,
        'validate': validate_task,
        'snapshot': snapshot_task,
        'quality_monitor': quality_monitor,
        'reference': reference_task,
        'drift': drift_task,
        'healing': healing_task,
        'metrics': metrics_task,
        'ui': ui_task,
        'explainability': explainability_tracker,
        'ab_testing': ab_testing_task,
        'train': train_task,
        'cleanup': cleanup_task,
        'override': override_task
    }

def setup_dependencies(tasks: Dict[str, Union[PythonOperator, BranchPythonOperator, EmptyOperator, ExternalTaskSensor]]) -> None:
    """
    Set up task dependencies in the DAG.
    
    Args:
        tasks: Dictionary of task operators
    """
    # Simple linear dependency for cross-DAG waiting
    tasks['wait_for_integrated'] >> tasks['start_pipeline'] >> tasks['ingest']
    
    # Data pipeline flow
    tasks['ingest'] >> tasks['preprocess'] >> tasks['validate_file']
    tasks['validate_file'] >> [tasks['continue_pipeline'], tasks['handle_missing_data']]
    tasks['continue_pipeline'] >> tasks['validate']
    tasks['validate'] >> [tasks['snapshot'], tasks['quality_monitor']]
    
    # Monitoring flow
    tasks['quality_monitor'] >> tasks['reference']
    tasks['reference'] >> tasks['drift']
    tasks['drift'] >> tasks['healing']
    
    # Metrics and UI updates
    tasks['healing'] >> tasks['metrics']
    tasks['metrics'] >> tasks['ui']
    
    # Model pipeline
    tasks['validate'] >> tasks['explainability']
    tasks['explainability'] >> tasks['ab_testing']
    tasks['ab_testing'] >> tasks['train']
    
    # Cleanup at the end
    tasks['ui'] >> tasks['cleanup']
    tasks['train'] >> tasks['cleanup']
    
    # Manual override can be triggered at any point
    tasks['override'].set_upstream([
        tasks['ingest'],
        tasks['validate'],
        tasks['drift'],
        tasks['train']
    ])

# Create the DAG
dag = setup_dag()

# Create and configure tasks
tasks = create_tasks(dag)

# Set up task dependencies
setup_dependencies(tasks)
