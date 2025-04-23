#!/usr/bin/env python3
"""
homeowner_dag.py

Main DAG for homeowner loss history project:
  - Data processing pipeline
  - Model training and evaluation
  - A/B testing
  - Monitoring and drift detection
"""

# Core Python imports
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Airflow imports
from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

# Local imports
from utils.config import (
    DEFAULT_START_DATE, SCHEDULE_CRON, AIRFLOW_DAG_BASE_CONF,
    AWS_REGION, DATA_BUCKET, MODEL_KEY_PREFIX, DRIFT_THRESHOLD,
    Config
)
from utils.logging_config import get_logger, setup_logging

# Setup logging
setup_logging()
log = get_logger(__name__)

# Constants
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.parquet"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"
MODEL_IDS = Config.MODEL_IDS

def _default_args() -> Dict[str, Any]:
    """Get default arguments for the DAG."""
    return {
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": DEFAULT_START_DATE,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(hours=2),
    }

def _get_dag_config() -> Dict[str, Any]:
    """Get DAG configuration from Airflow variables."""
    try:
        conf = Variable.get("DAG_RUN_CONF", default_var="{}")
        return json.loads(conf) if isinstance(conf, str) else conf
    except Exception as e:
        log.error(f"Error getting DAG config: {e}")
        return {}

@dag(
    dag_id="homeowner_loss_history_full_pipeline",
    default_args=_default_args(),
    description='DAG for homeowner loss history prediction pipeline',
    schedule=SCHEDULE_CRON,
    catchup=False,
    tags=["homeowner", "loss_history"],
)
def homeowner_pipeline():
    """Main DAG function that defines the pipeline structure."""
    
    # Initialize monitors (lazy imports)
    from tasks.data_quality import DataQualityMonitor
    from tasks.model_explainability import ModelExplainabilityTracker
    from tasks.ab_testing import ABTestingPipeline
    
    data_quality_monitor = DataQualityMonitor()
    model_explainability_tracker = ModelExplainabilityTracker("model1")
    ab_testing_pipeline = ABTestingPipeline("model1", test_duration_days=7)

    # 1️⃣ Data Ingestion
    @task()
    def ingest_data():
        """Task to ingest data from S3."""
        from tasks.ingestion import ingest_data_from_s3
        return ingest_data_from_s3()

    # 2️⃣ Data Processing
    @task()
    def preprocess_data(raw_path: str) -> str:
        """Task to preprocess and validate data."""
        from tasks.preprocessing import preprocess_data
        from tasks.schema_validation import validate_schema, snapshot_schema
        import pandas as pd
        
        df = preprocess_data(raw_path)
        validate_schema(df)
        snapshot_schema(df)
        df.to_parquet(LOCAL_PROCESSED_PATH, index=False)
        return LOCAL_PROCESSED_PATH

    # 3️⃣ Data Quality Check
    @task()
    def check_data_quality(path: str) -> str:
        """Task to check data quality."""
        from utils.slack import post as send_message
        import pandas as pd
        
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

    # 4️⃣ Drift Detection
    @task.branch()
    def check_drift(path: str) -> str:
        """Task to check for data drift."""
        from tasks.drift import (
            generate_reference_means,
            detect_data_drift
        )
        
        if os.path.exists(REFERENCE_MEANS_PATH):
            ref = generate_reference_means(path, REFERENCE_MEANS_PATH)
            flag = detect_data_drift(path, ref)
            return "healing_task" if flag == "self_healing" else "override_or_train"
        return "override_or_train"

    # 5️⃣ Self-healing
    @task(task_id="healing_task")
    def healing_task():
        """Task to handle drift self-healing."""
        from utils.slack import post as send_message
        from tasks.drift import self_healing as drift_self_heal
        
        send_message(
            channel="#alerts",
            title="⚠️ Drift Detected",
            details="Self‑healing routine executed.",
            urgency="medium",
        )
        drift_self_heal()

    # 6️⃣ Training Control
    @task.branch(task_id="override_or_train")
    def override_branch() -> str:
        """Task to determine training path."""
        from tasks.training import manual_override
        
        conf = _get_dag_config()
        if conf.get("action") == "retrain" and conf.get("model_id"):
            log.info(f"Retraining triggered for model: {conf.get('model_id')}")
            return "retrain_model"
            
        return "apply_override" if manual_override() else "start_training"

    # 7️⃣ Training Tasks
    @task(task_id="apply_override")
    def apply_override():
        """Task to apply manual override."""
        from utils.slack import post as send_message
        from tasks.training import manual_override
        
        params = manual_override()
        send_message(
            channel="#alerts",
            title="🛠️ Manual Override Applied",
            details=f"Custom hyperparams: {params}",
            urgency="low",
        )

    @task(task_id="retrain_model")
    def retrain_model():
        """Task to handle model retraining."""
        from utils.slack import post as send_message
        
        conf = _get_dag_config()
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
        
        Variable.set("TARGET_MODEL_ID", model_id)

    # 8️⃣ Training Execution
    @task.branch()
    def training_branch() -> str:
        """Task to determine which model to train."""
        try:
            target_model = Variable.get("TARGET_MODEL_ID", default_var=None)
            if target_model:
                return f"train_compare_{target_model}"
        except Exception as e:
            log.error(f"Error in training_branch: {e}")
        return "train_all_models"

    # 9️⃣ Post-Training Tasks
    @task()
    def record_metrics():
        """Task to record system metrics."""
        from tasks.monitoring import record_system_metrics, update_monitoring_with_ui_components
        record_system_metrics()
        update_monitoring_with_ui_components()

    @task()
    def notify_complete():
        """Task to send completion notification."""
        from utils.slack import post as send_message
        send_message(
            channel="#alerts",
            title="✅ Pipeline Complete",
            details="All models trained and evaluated successfully.",
            urgency="low",
        )

    @task()
    def archive():
        """Task to archive artifacts."""
        from utils.storage import upload as upload_to_s3
        from utils.slack import post as send_message
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = f"archive/{timestamp}"
        
        try:
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
        except Exception as e:
            log.error(f"Error in archive task: {e}")
            send_message(
                channel="#alerts",
                title="❌ Archive Failed",
                details=str(e),
                urgency="high"
            )

    # Define task dependencies
    raw_data = ingest_data()
    processed_data = preprocess_data(raw_data)
    quality_checked = check_data_quality(processed_data)
    
    branch = check_drift(quality_checked)
    heal = healing_task()
    
    override = override_branch()
    apply_override_task = apply_override()
    retrain = retrain_model()
    start_training = EmptyOperator(task_id="start_training")
    
    training_branch_task = training_branch()
    train_all_models = EmptyOperator(task_id="train_all_models")
    
    join_after_training = EmptyOperator(
        task_id="join_after_training",
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    
    # Wire up the pipeline
    branch >> heal >> override
    branch >> override
    override >> apply_override_task >> training_branch_task
    override >> start_training >> training_branch_task
    override >> retrain >> training_branch_task
    
    training_branch_task >> train_all_models
    train_all_models.set_downstream(join_after_training)
    
    join_after_training >> record_metrics() >> notify_complete() >> archive()

    return homeowner_pipeline()

# Create DAG instance
homeowner_loss_history_full_pipeline = homeowner_pipeline()
