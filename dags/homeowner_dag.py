#!/usr/bin/env python3
"""
homeowner_dag.py

Main DAG for homeowner loss history project:
  - Data processing pipeline
  - Model training and evaluation
  - A/B testing
  - Monitoring and drift detection
"""

import os
import logging
import time
import json
import pandas as pd

from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

from tasks.ingestion import ingest_data_from_s3
from tasks.preprocessing import preprocess_data
from tasks.schema_validation import validate_schema, snapshot_schema
from tasks.data_quality import DataQualityMonitor
from tasks.drift import (
    generate_reference_means,
    detect_data_drift,
    self_healing as drift_self_heal
)
from tasks.monitoring import record_system_metrics, update_monitoring_with_ui_components
from tasks.model_explainability import ModelExplainabilityTracker
from tasks.ab_testing import ABTestingPipeline
from tasks.training import train_and_compare_fn, manual_override
from utils.slack import post as send_message
from utils.storage import upload as upload_to_s3
from utils.logging_config import get_logger, setup_logging
from utils.config import (
    DEFAULT_START_DATE, SCHEDULE_CRON, AIRFLOW_DAG_BASE_CONF,
    AWS_REGION, S3_BUCKET, MODEL_KEY_PREFIX, DRIFT_THRESHOLD,
    Config
)
from utils.security import SecurityUtils, validate_input

# Setup logging
setup_logging()
log = get_logger(__name__)

# Constants
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.parquet"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"
MODEL_IDS = Config.MODEL_IDS  # Use model IDs from config

# Initialize monitors
data_quality_monitor = DataQualityMonitor()
model_explainability_tracker = ModelExplainabilityTracker("model1")  # Will be updated per model
ab_testing_pipeline = ABTestingPipeline("model1", test_duration_days=7)  # Will be updated per model

# Initialize AWS clients with region
import boto3
s3 = boto3.client('s3', region_name=AWS_REGION)
cloudwatch = boto3.client('cloudwatch', region_name=AWS_REGION)

def _default_args():
    return {
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": datetime(2025, 1, 1),
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=2),
    }

@dag(
    dag_id="homeowner_loss_history_full_pipeline",
    default_args=_default_args(),
    description='DAG for homeowner loss history prediction pipeline',
    schedule='0 0 * * *',  # Daily at midnight
    catchup=False,
    tags=["homeowner", "loss_history"],
)
def homeowner_pipeline():

    # 1️⃣ Ingest raw CSV
    raw_path = ingest_data_from_s3()

    # 2️⃣ Preprocess → Parquet
    @task()
    def _preprocess(path: str) -> str:
        df = preprocess_data(path)
        validate_schema(df)
        snapshot_schema(df)
        df.to_parquet(LOCAL_PROCESSED_PATH, index=False)
        return LOCAL_PROCESSED_PATH
    processed_path = _preprocess(raw_path)

    # 2a Data Quality Check
    @task()
    def check_data_quality(path: str) -> str:
        df = pd.read_parquet(path)
        quality_report = data_quality_monitor.monitor_data_quality(df, is_baseline=True)
        
        if quality_report['issues']:
            send_message(
                channel="#alerts",
                title="🔍 Data Quality Issues Detected",
                details="\n".join([
                    f"{issue['type']}: {len(issue.get('details', []))} issues found"
                    for issue in quality_report['issues']
                ]),
                urgency="medium"
            )
        
        return path
    quality_checked_path = check_data_quality(processed_path)

    # 3️⃣ Drift‑check: existing reference?
    @task.branch()
    def _branch(path: str) -> str:
        if os.path.exists(REFERENCE_MEANS_PATH):
            ref = generate_reference_means(path, REFERENCE_MEANS_PATH)
            flag = detect_data_drift(path, ref)
            return "healing_task" if flag == "self_healing" else "override_or_train"
        return "override_or_train"
    branch = _branch(quality_checked_path)

    # 3a Self‑healing path
    @task(task_id="healing_task")
    def healing_task():
        send_message(
            channel="#alerts",
            title="⚠️ Drift Detected",
            details="Self‑healing routine executed.",
            urgency="medium",
        )
        drift_self_heal()
    heal = healing_task()

    # 4️⃣ Manual‑Override vs Start‑Training
    @task.branch(task_id="override_or_train")
    def override_branch() -> str:
        # Check if this is a retraining trigger from the dashboard
        try:
            conf = Variable.get("DAG_RUN_CONF", default_var="{}")
            if isinstance(conf, str):
                conf = json.loads(conf)
            
            if conf.get("action") == "retrain" and conf.get("model_id"):
                log.info(f"Retraining triggered for model: {conf.get('model_id')}")
                return "retrain_model"
        except Exception as e:
            log.error(f"Error checking retraining trigger: {e}")
        
        # Default behavior
        return "apply_override" if manual_override() else "start_training"

    @task(task_id="apply_override")
    def apply_override():
        params = manual_override()
        send_message(
            channel="#alerts",
            title="🛠️ Manual Override Applied",
            details=f"Custom hyperparams: {params}",
            urgency="low",
        )
        # training task will consume manual_override internally

    @task(task_id="retrain_model")
    def retrain_model():
        try:
            conf = Variable.get("DAG_RUN_CONF", default_var="{}")
            if isinstance(conf, str):
                conf = json.loads(conf)
            
            model_id = conf.get("model_id")
            if not model_id:
                log.error("No model_id provided for retraining")
                return
            
            log.info(f"Retraining model: {model_id}")
            send_message(
                channel="#alerts",
                title=f"🔄 Retraining {model_id}",
                details="Retraining triggered from dashboard",
                urgency="medium",
            )
            
            # Set the model_id for the training task
            Variable.set("TARGET_MODEL_ID", model_id)
            
        except Exception as e:
            log.error(f"Error in retrain_model task: {e}")

    start_training = EmptyOperator(task_id="start_training")

    # 5️⃣ Training per model → join
    join_after_training = EmptyOperator(
        task_id="join_after_training",
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    # Check if we're retraining a specific model
    @task.branch()
    def training_branch() -> str:
        try:
            target_model = Variable.get("TARGET_MODEL_ID", default_var=None)
            if target_model:
                return f"train_compare_{target_model}"
        except Exception as e:
            log.error(f"Error in training_branch: {e}")
        return "train_all_models"
    
    # Assign the training branch task to a variable
    training_branch_task = training_branch()
    
    train_all_models = EmptyOperator(task_id="train_all_models")
    
    # Create training tasks for each model
    training_tasks = {}
    for m in MODEL_IDS:
        # Create task for normal training path
        training_tasks[m] = PythonOperator(
            task_id=f"train_compare_{m}",
            python_callable=train_and_compare_fn,
            op_kwargs={
                "model_id": m,
                "processed_path": LOCAL_PROCESSED_PATH,
                "data_quality_monitor": data_quality_monitor,
                "model_explainability_tracker": ModelExplainabilityTracker(m),
                "ab_testing_pipeline": ABTestingPipeline(m, test_duration_days=7)
            },
        )
        # Set downstream relationship
        training_tasks[m].set_downstream(join_after_training)
    
    # Wire the training branch
    training_branch_task >> train_all_models
    train_all_models >> [training_tasks[m] for m in MODEL_IDS]
    
    # Wire drift → heal → override/train → training
    branch >> heal >> override_branch()
    branch >> override_branch()
    override_branch() >> apply_override() >> training_branch_task
    override_branch() >> start_training >> training_branch_task
    override_branch() >> retrain_model() >> training_branch_task

    # 6️⃣ Record metrics and notify
    @task()
    def record_metrics_task():
        record_system_metrics()
        update_monitoring_with_ui_components()

    @task()
    def notify_complete():
        send_message(
            channel="#alerts",
            title="✅ Pipeline Complete",
            details="All models trained and evaluated successfully.",
            urgency="low",
        )

    # 7️⃣ Archive artifacts
    @task()
    def archive():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f"archive/{timestamp}"
        
        # Archive processed data
        upload_to_s3(LOCAL_PROCESSED_PATH, f"{archive_path}/processed.parquet")
        
        # Archive reference means if exists
        if os.path.exists(REFERENCE_MEANS_PATH):
            upload_to_s3(REFERENCE_MEANS_PATH, f"{archive_path}/reference_means.csv")
        
        # Archive data quality report
        quality_summary = data_quality_monitor.get_quality_summary()
        if quality_summary:
            with open("/tmp/quality_summary.json", "w") as f:
                json.dump(quality_summary, f)
            upload_to_s3("/tmp/quality_summary.json", f"{archive_path}/quality_summary.json")

    # 8️⃣ Start WebSocket server for real-time updates
    @task()
    def start_websocket_server():
        update_monitoring_with_ui_components()

    # Set up task dependencies
    join_after_training >> record_metrics_task() >> notify_complete() >> archive()
    join_after_training >> start_websocket_server()

    return homeowner_pipeline()

# Create DAG instance
dag = homeowner_pipeline()
