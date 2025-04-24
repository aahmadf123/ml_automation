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
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sys

# Airflow imports
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable
from airflow.utils.dates import days_ago
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
        tags=['ml', 'homeowner', 'loss_history']
    )

def create_tasks(dag: DAG) -> Dict[str, PythonOperator]:
    """
    Create all tasks for the DAG.
    
    Args:
        dag: The DAG instance to add tasks to
        
    Returns:
        Dict[str, PythonOperator]: Dictionary of task operators
    """
    # Import task modules directly
    import tasks.ingestion as ingestion
    import tasks.preprocessing as preprocessing
    import tasks.schema_validation as schema_validation
    import tasks.data_quality as data_quality
    import tasks.monitoring as monitoring
    import tasks.drift as drift
    import tasks.model_explainability as model_explainability
    import tasks.ab_testing as ab_testing
    import tasks.training as training
    
    # Data ingestion and preprocessing tasks
    ingest_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingestion.ingest_data_from_s3,
        dag=dag
    )
    
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocessing.preprocess_data,
        op_kwargs={
            'data_path': "{{ ti.xcom_pull(task_ids='ingest_data') }}",
            'output_path': LOCAL_PROCESSED_PATH,
            'force_reprocess': False  # Set to True to force reprocessing even if file exists
        },
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
    
    # Schema validation tasks
    validate_task = PythonOperator(
        task_id='validate_schema',
        python_callable=lambda **kwargs: schema_validation.validate_schema(
            pd.read_parquet(LOCAL_PROCESSED_PATH, engine='pyarrow')
        ) if os.path.exists(LOCAL_PROCESSED_PATH) else {"status": "error", "message": "Data file not found"},
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
        python_callable=lambda **kwargs: schema_validation.snapshot_schema(
            # Just need dtypes, so read a small sample with only 1000 rows for memory efficiency
            pd.read_parquet(LOCAL_PROCESSED_PATH, engine='pyarrow', columns=None)
        ) if os.path.exists(LOCAL_PROCESSED_PATH) else {"status": "error", "message": "Data file not found"},
        executor_config={
            "KubernetesExecutor": {
                "request_memory": "1Gi",
                "limit_memory": "2Gi"
            }
        },
        dag=dag
    )
    
    # Monitoring tasks
    quality_monitor = PythonOperator(
        task_id='monitor_data_quality',
        python_callable=lambda **kwargs: data_quality.DataQualityMonitor().run_quality_checks(
            pd.read_parquet(LOCAL_PROCESSED_PATH, engine='pyarrow', columns=None)
        ) if os.path.exists(LOCAL_PROCESSED_PATH) else {"status": "error", "message": "Data file not found"},
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
        python_callable=monitoring.generate_reference_means,
        op_kwargs={
            'processed_data_path': LOCAL_PROCESSED_PATH
        },
        dag=dag
    )
    
    drift_task = PythonOperator(
        task_id='detect_drift',
        python_callable=drift.detect_data_drift,
        op_kwargs={
            'processed_data_path': LOCAL_PROCESSED_PATH
        },
        dag=dag
    )
    
    healing_task = PythonOperator(
        task_id='self_healing',
        python_callable=drift.self_healing,
        op_kwargs={
            'drift_results': "{{ ti.xcom_pull(task_ids='detect_drift') }}",
            'processed_data_path': LOCAL_PROCESSED_PATH
        },
        dag=dag
    )
    
    metrics_task = PythonOperator(
        task_id='record_metrics',
        python_callable=monitoring.record_system_metrics,
        dag=dag
    )
    
    ui_task = PythonOperator(
        task_id='update_ui',
        python_callable=monitoring.update_monitoring_with_ui_components,
        dag=dag
    )
    
    # Model tasks
    explainability_tracker = PythonOperator(
        task_id='track_explainability',
        python_callable=lambda **kwargs: model_explainability.ModelExplainabilityTracker('homeowner_model').track_model_and_data(
            model=kwargs.get('ti').xcom_pull(task_ids='train_compare_model1'),
            X=pd.read_parquet(LOCAL_PROCESSED_PATH, engine='pyarrow').drop(columns=['claim_amount'], errors='ignore'),
            y=pd.read_parquet(LOCAL_PROCESSED_PATH, engine='pyarrow', columns=['claim_amount'])['claim_amount'],
            run_id=kwargs.get('ti').xcom_pull(task_ids='train_compare_model1', key='run_id')
        ) if os.path.exists(LOCAL_PROCESSED_PATH) else {"status": "error", "message": "Data file not found"},
        executor_config={
            "KubernetesExecutor": {
                "request_memory": "4Gi",
                "limit_memory": "6Gi"
            }
        },
        dag=dag
    )
    
    ab_testing = PythonOperator(
        task_id='ab_testing',
        python_callable=lambda **kwargs: ab_testing.ABTestingPipeline('homeowner_model').run_ab_test(
            new_model=kwargs.get('ti').xcom_pull(task_ids='train_compare_model1'),
            X_test=pd.read_parquet(LOCAL_PROCESSED_PATH, engine='pyarrow').drop(columns=['claim_amount'], errors='ignore'),
            y_test=pd.read_parquet(LOCAL_PROCESSED_PATH, engine='pyarrow', columns=['claim_amount'])['claim_amount']
        ) if os.path.exists(LOCAL_PROCESSED_PATH) else {"status": "error", "message": "Data file not found"},
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
    
    # Manual override task
    override_task = PythonOperator(
        task_id='manual_override',
        python_callable=data_quality.manual_override,
        op_kwargs={
            'model_id': 'homeowner_model',
            'override_action': 'No action required'
        },
        dag=dag
    )
    
    return {
        'ingest': ingest_task,
        'preprocess': preprocess_task,
        'validate': validate_task,
        'snapshot': snapshot_task,
        'quality_monitor': quality_monitor,
        'reference': reference_task,
        'drift': drift_task,
        'healing': healing_task,
        'metrics': metrics_task,
        'ui': ui_task,
        'explainability': explainability_tracker,
        'ab_testing': ab_testing,
        'train': train_task,
        'override': override_task
    }

def setup_dependencies(tasks: Dict[str, PythonOperator]) -> None:
    """
    Set up task dependencies in the DAG.
    
    Args:
        tasks: Dictionary of task operators
    """
    # Data pipeline flow
    tasks['ingest'] >> tasks['preprocess'] >> tasks['validate']
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
