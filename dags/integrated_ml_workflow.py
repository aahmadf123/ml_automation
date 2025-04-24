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
from utils.config import (
    MLFLOW_URI, MLFLOW_EXPERIMENT, DATA_BUCKET,
    S3_BUCKET, RAW_DATA_KEY
)
from utils.clearml_config import init_clearml, log_dataset_to_clearml
from tasks.training import train_and_compare_fn
from tasks.data_quality import DataQualityMonitor
from tasks.drift_detection import DriftDetector
from utils.slack import post as slack_post

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
    """Download data from S3 and return local path"""
    import boto3
    from tempfile import NamedTemporaryFile
    
    s3_client = boto3.client('s3')
    
    # Get data from S3
    with NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        local_path = temp_file.name
        s3_client.download_file(
            S3_BUCKET,
            RAW_DATA_KEY,
            local_path
        )
    
    # Log with ClearML if enabled
    try:
        clearml_task = init_clearml("Data_Download")
        if clearml_task:
            clearml_task.set_parameter("s3_bucket", S3_BUCKET)
            clearml_task.set_parameter("s3_key", RAW_DATA_KEY)
            log_dataset_to_clearml(
                dataset_name="Raw_Data",
                dataset_path=local_path,
                dataset_tags=["raw", "csv"]
            )
            clearml_task.close()
    except Exception as e:
        logger.warning(f"Error logging to ClearML: {str(e)}")
    
    # Return local path for downstream tasks
    context['ti'].xcom_push(key='data_path', value=local_path)
    slack_post(f":white_check_mark: Data downloaded from s3://{S3_BUCKET}/{RAW_DATA_KEY}")
    return local_path

def process_data(**context):
    """Process the data and prepare for model training"""
    import pandas as pd
    import numpy as np
    from tempfile import NamedTemporaryFile
    import os
    import mlflow
    
    # Get raw data path from previous task
    raw_data_path = context['ti'].xcom_pull(task_ids='download_data', key='data_path')
    
    # Initialize MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    # Start ClearML task
    clearml_task = None
    try:
        clearml_task = init_clearml("Data_Processing")
    except Exception as e:
        logger.warning(f"Error initializing ClearML: {str(e)}")
    
    # Load the data
    df = pd.read_csv(raw_data_path)
    
    # Start MLflow run for data processing
    with mlflow.start_run(run_name="data_processing") as run:
        run_id = run.info.run_id
        
        # Log raw data stats
        mlflow.log_param("raw_data_rows", len(df))
        mlflow.log_param("raw_data_columns", len(df.columns))
        
        # Basic preprocessing
        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Log with ClearML if available
        if clearml_task:
            clearml_task.set_parameter("processed_rows", len(df))
            clearml_task.set_parameter("processed_columns", len(df.columns))
        
        # Save processed data
        with NamedTemporaryFile(delete=False, suffix='.parquet') as temp_file:
            processed_path = temp_file.name
            df.to_parquet(processed_path, index=False)
        
        # Log processed data stats
        mlflow.log_param("processed_data_rows", len(df))
        mlflow.log_metric("missing_values_count", df.isna().sum().sum())
        
        # Upload to S3
        s3_client = boto3.client('s3')
        processed_s3_key = f"processed/processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        s3_client.upload_file(processed_path, S3_BUCKET, processed_s3_key)
        
        # Log location in MLflow
        mlflow.log_param("processed_data_s3_path", f"s3://{S3_BUCKET}/{processed_s3_key}")
    
    # Close ClearML task if available
    if clearml_task:
        clearml_task.close()
    
    # Push the processed data path
    context['ti'].xcom_push(key='processed_data_path', value=processed_path)
    context['ti'].xcom_push(key='processed_s3_key', value=processed_s3_key)
    
    # Cleanup raw data file
    try:
        os.remove(raw_data_path)
    except Exception as e:
        logger.warning(f"Error removing raw data file: {str(e)}")
    
    slack_post(f":white_check_mark: Data processed successfully. Rows: {len(df)}")
    return processed_path

def check_data_quality(**context):
    """Run data quality checks on the processed data"""
    # Get processed data path
    processed_data_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
    
    # Import your existing DataQualityMonitor
    from tasks.data_quality import DataQualityMonitor
    
    # Initialize monitor
    monitor = DataQualityMonitor()
    
    # Load data
    df = pd.read_parquet(processed_data_path)
    
    # Initialize MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    # Start ClearML task
    clearml_task = None
    try:
        clearml_task = init_clearml("Data_Quality_Check")
    except Exception as e:
        logger.warning(f"Error initializing ClearML: {str(e)}")
    
    # Run data quality checks
    with mlflow.start_run(run_name="data_quality") as run:
        run_id = run.info.run_id
        
        # Run quality checks
        quality_results = monitor.run_quality_checks(df)
        
        # Log results to MLflow
        mlflow.log_params({
            "missing_threshold": monitor.config["missing_threshold"],
            "outlier_threshold": monitor.config["outlier_threshold"],
            "correlation_threshold": monitor.config["correlation_threshold"]
        })
        
        # Log metrics
        mlflow.log_metric("total_issues", quality_results["total_issues"])
        mlflow.log_metric("missing_value_issues", quality_results["missing_value_issues"])
        mlflow.log_metric("outlier_issues", quality_results["outlier_issues"])
        
        # Create and log report
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            report_path = temp_file.name
            json.dump(quality_results, temp_file, indent=2)
        
        mlflow.log_artifact(report_path)
        
        # Log to ClearML if available
        if clearml_task:
            for issue_type, count in quality_results.items():
                if isinstance(count, (int, float)):
                    clearml_task.get_logger().report_scalar(
                        title="Data Quality",
                        series=issue_type,
                        value=count,
                        iteration=0
                    )
            clearml_task.close()
    
    # Continue only if below threshold or override is set
    data_quality_threshold = int(Variable.get("DATA_QUALITY_THRESHOLD", default_var="10"))
    force_continue = Variable.get("FORCE_CONTINUE", default_var="false").lower() == "true"
    
    if quality_results["total_issues"] > data_quality_threshold and not force_continue:
        slack_post(f":x: Data quality issues ({quality_results['total_issues']}) exceed threshold ({data_quality_threshold}). Workflow stopped.")
        raise ValueError(f"Data quality issues ({quality_results['total_issues']}) exceed threshold ({data_quality_threshold})")
    
    # Push quality results for downstream tasks
    context['ti'].xcom_push(key='quality_results', value=quality_results)
    slack_post(f":white_check_mark: Data quality check completed. Issues: {quality_results['total_issues']}")
    return quality_results

def train_model(**context):
    """Train the model using processed data"""
    # Get processed data path
    processed_data_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
    
    # Use the imported training function
    model_id = f"homeowner_loss_model_{datetime.now().strftime('%Y%m%d')}"
    
    try:
        # Train model - this function already handles both MLflow and ClearML logging
        train_and_compare_fn(model_id, processed_data_path)
        
        # Push model ID for downstream tasks
        context['ti'].xcom_push(key='model_id', value=model_id)
        slack_post(f":white_check_mark: Model {model_id} trained successfully")
        return model_id
    except Exception as e:
        slack_post(f":x: Model training failed: {str(e)}")
        raise

def check_for_drift(**context):
    """Check for data drift between reference and new data"""
    # Get processed data path
    processed_data_path = context['ti'].xcom_pull(task_ids='process_data', key='processed_data_path')
    
    # Import your existing DriftDetector
    from tasks.drift_detection import DriftDetector
    
    # Initialize detector
    detector = DriftDetector()
    
    # Load current data
    current_data = pd.read_parquet(processed_data_path)
    
    # Initialize MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    # Start ClearML task
    clearml_task = None
    try:
        clearml_task = init_clearml("Drift_Detection")
    except Exception as e:
        logger.warning(f"Error initializing ClearML: {str(e)}")
    
    # Run drift detection
    with mlflow.start_run(run_name="drift_detection") as run:
        run_id = run.info.run_id
        
        # Detect drift
        drift_results = detector.detect_drift(current_data)
        
        # Log results to MLflow
        mlflow.log_params({
            "drift_threshold": detector.threshold,
            "reference_data_date": detector.reference_date.strftime("%Y-%m-%d") if detector.reference_date else "None"
        })
        
        # Log metrics
        mlflow.log_metric("overall_drift_score", drift_results["overall_drift_score"])
        mlflow.log_metric("drifted_features_count", len(drift_results["drifted_features"]))
        
        # Create and log report
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            report_path = temp_file.name
            json.dump(drift_results, temp_file, indent=2)
        
        mlflow.log_artifact(report_path)
        
        # Log to ClearML if available
        if clearml_task:
            clearml_task.get_logger().report_scalar(
                title="Drift Detection",
                series="overall_drift_score",
                value=drift_results["overall_drift_score"],
                iteration=0
            )
            
            for feature, score in drift_results.get("feature_drift_scores", {}).items():
                clearml_task.get_logger().report_scalar(
                    title="Feature Drift",
                    series=feature,
                    value=score,
                    iteration=0
                )
            
            clearml_task.close()
    
    # Check if drift exceeds threshold
    drift_threshold = float(Variable.get("DRIFT_THRESHOLD", default_var="0.1"))
    force_continue = Variable.get("FORCE_CONTINUE", default_var="false").lower() == "true"
    
    if drift_results["overall_drift_score"] > drift_threshold and not force_continue:
        # Send alert but continue workflow
        slack_post(f":warning: Data drift detected! Score: {drift_results['overall_drift_score']:.4f}, Threshold: {drift_threshold}")
    
    # Push drift results for downstream tasks
    context['ti'].xcom_push(key='drift_results', value=drift_results)
    slack_post(f":white_check_mark: Drift detection completed. Score: {drift_results['overall_drift_score']:.4f}")
    return drift_results

# Define tasks
download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_data,
    provide_context=True,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    provide_context=True,
    dag=dag,
)

quality_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    provide_context=True,
    dag=dag,
)

drift_task = PythonOperator(
    task_id='check_for_drift',
    python_callable=check_for_drift,
    provide_context=True,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

# Define workflow
download_task >> process_task >> [quality_task, drift_task] >> train_task 