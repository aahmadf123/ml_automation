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
    
    logger.info("Starting download_data task")
    
    # Get bucket name from Airflow Variable to ensure it's up to date
    bucket = Variable.get("DATA_BUCKET", default_var="grange-seniordesign-bucket")
    key = config.RAW_DATA_KEY
    
    logger.info(f"Using bucket from Airflow Variables: {bucket}")
    logger.info(f"Attempting to access data at s3://{bucket}/{key}")
    
    try:
        # Use the ingestion function from the full pipeline if available
        try:
            logger.info("Attempting to use pipeline ingestion function")
            data_path = ingestion.ingest_data_from_s3(
                bucket_name=bucket,
                key=key
            )
            logger.info(f"Successfully ingested data using pipeline function: {data_path}")
        except Exception as e:
            # Fall back to direct S3 download if ingestion function fails
            logger.warning(f"Pipeline ingestion failed, falling back to direct download: {str(e)}")
            
            s3_client = boto3.client('s3', region_name=config.AWS_REGION)
            with NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                local_path = temp_file.name
                logger.info(f"Downloading to {local_path}")
                s3_client.download_file(bucket, key, local_path)
                data_path = local_path
                logger.info(f"Successfully downloaded to {local_path}")
                
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
            logger.error(f"Data file does not exist: {raw_data_path}")
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
        
        # Save processed data
        processed_path = None
        try:
            # Create a more descriptive filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with NamedTemporaryFile(delete=False, suffix=f'_{timestamp}.parquet') as temp_file:
                processed_path = temp_file.name
                logger.info(f"Saving processed data to {processed_path}")
                df.to_parquet(processed_path, index=False)
                
            # Verify the file was created and has content
            if not os.path.exists(processed_path) or os.path.getsize(processed_path) == 0:
                logger.error(f"Failed to create processed data file or file is empty: {processed_path}")
                raise IOError(f"Failed to create processed data file or file is empty: {processed_path}")
                
            logger.info(f"Successfully saved processed data to {processed_path}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
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
        
        # Close ClearML task if available
        if clearml_task:
            try:
                clearml_task.close()
                logger.info("Successfully closed ClearML task")
            except Exception as e:
                logger.warning(f"Failed to properly close ClearML task: {str(e)}")
        
        # Push the processed data path and S3 key to Xcom
        context['ti'].xcom_push(key='processed_data_path', value=processed_path)
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
    """Run data quality checks on the processed data"""
    logger.info("Starting check_data_quality task")
    
    try:
        # Get processed data path from previous task with validation
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
        
        # Import the DataQualityMonitor with error handling
        try:
            from tasks.data_quality import DataQualityMonitor
            monitor = DataQualityMonitor()
            logger.info("Successfully initialized DataQualityMonitor")
        except Exception as e:
            logger.error(f"Failed to import or initialize DataQualityMonitor: {str(e)}")
            raise
        
        # Load data with robust error handling
        try:
            logger.info(f"Loading processed data from {processed_data_path}")
            
            # For very large files, use optimized loading
            if file_size > 50_000_000:  # 50MB
                logger.info(f"Large file detected ({file_size} bytes). Using optimized loading strategy.")
                
                # Try to sample the file if it's too large for memory
                try:
                    # First load just a sample to check shape
                    import pyarrow.parquet as pq
                    metadata = pq.read_metadata(processed_data_path)
                    num_rows = metadata.num_rows
                    
                    if num_rows > 1_000_000:
                        logger.info(f"Large dataset detected with {num_rows} rows. Will load sample first.")
                        # Load 10% of data or 100k rows, whichever is smaller
                        sample_rows = min(num_rows // 10, 100_000)
                        df_sample = pd.read_parquet(processed_data_path, engine='pyarrow')
                        df = df_sample.sample(sample_rows, random_state=42)
                        logger.info(f"Loaded {sample_rows} rows sample instead of full dataset")
                    else:
                        df = pd.read_parquet(processed_data_path, engine='pyarrow')
                        logger.info(f"Dataset has {num_rows} rows, loading full dataset")
                        
                except Exception as e:
                    logger.warning(f"Error during optimized loading, falling back to standard load: {str(e)}")
                    df = pd.read_parquet(processed_data_path)
            else:
                # Standard loading for smaller files
                df = pd.read_parquet(processed_data_path)
            
            # Verify data was loaded successfully
            if df.empty:
                logger.error("Loaded DataFrame is empty")
                raise ValueError("Loaded DataFrame is empty")
                
            logger.info(f"Successfully loaded processed data with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load processed data: {str(e)}")
            raise
        
        # Initialize MLflow with robust error handling
        mlflow_run = None
        try:
            mlflow_uri = config.MLFLOW_URI
            mlflow_experiment = config.MLFLOW_EXPERIMENT
            
            logger.info(f"Connecting to MLflow at {mlflow_uri}")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(mlflow_experiment)
            
            mlflow_run = mlflow.start_run(run_name="data_quality")
            run_id = mlflow_run.info.run_id
            logger.info(f"Started MLflow run with ID: {run_id}")
        except Exception as e:
            logger.warning(f"Unable to configure MLflow: {str(e)}. Will continue without MLflow tracking.")
            run_id = None
        
        # Start ClearML task
        clearml_task = None
        try:
            clearml_task = clearml_config.init_clearml("Data_Quality_Check")
            logger.info("Successfully initialized ClearML task")
        except Exception as e:
            logger.warning(f"Error initializing ClearML: {str(e)}. Will continue without ClearML tracking.")
        
        # Run data quality checks with error handling
        try:
            logger.info("Running data quality checks")
            quality_results = monitor.run_quality_checks(df)
            logger.info(f"Quality check results: {quality_results}")
            
            # Clean up dataframe reference to free memory
            del df
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during quality checks: {str(e)}")
            raise
        
        # Log results to MLflow
        if mlflow_run:
            try:
                # Log parameters
                logger.info("Logging quality check parameters to MLflow")
                mlflow.log_params({
                    "missing_threshold": monitor.config["missing_threshold"],
                    "outlier_threshold": monitor.config["outlier_threshold"],
                    "correlation_threshold": monitor.config["correlation_threshold"]
                })
                
                # Log metrics
                logger.info("Logging quality check metrics to MLflow")
                # Make sure these metrics exist, otherwise use defaults
                mlflow.log_metric("total_issues", quality_results.get("total_issues", 0))
                mlflow.log_metric("missing_value_issues", quality_results.get("missing_value_issues", 0))
                mlflow.log_metric("outlier_issues", quality_results.get("outlier_issues", 0))
                
                # Create and log report
                logger.info("Creating quality report artifact")
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                    report_path = temp_file.name
                    json.dump(quality_results, temp_file, indent=2)
                
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
        context['ti'].xcom_push(key='quality_results', value=quality_results)
        
        # Send success notification
        try:
            slack.post(f":white_check_mark: Data quality check completed. Issues: {total_issues}")
            logger.info("Sent quality success notification to Slack")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return quality_results
        
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
test_task >> download_task >> process_task >> [quality_task, drift_task] >> train_task 