#!/usr/bin/env python3
"""
homeowner_dag.py

Home‑owner Loss‑History full pipeline DAG
• Ingest → Preprocess → Drift‑Check → (Self‑Heal ⧸ Manual‑Override → Train)
• Human‑in‑the‑Loop hooks for manual override
• Metrics, Notifications, Archive
• MLflow integration for experiment tracking
• WebSocket events for real-time dashboard updates
• Retraining triggers from dashboard
"""

import os
import logging
import time
import json

from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

from tasks.ingestion import ingest_data_from_s3
from tasks.preprocessing import preprocess_data
from tasks.schema_validation import validate_schema, snapshot_schema
from tasks.drift import (
    generate_reference_means,
    detect_data_drift,
    self_healing as drift_self_heal
)
from tasks.monitoring import record_system_metrics, update_monitoring_with_ui_components
from tasks.training import train_and_compare_fn, manual_override
from utils.slack import post as send_message
from utils.storage import upload as upload_to_s3

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Constants
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.parquet"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"
MODEL_IDS = ["model1", "model2", "model3", "model4", "model5"]


def _default_args():
    return {
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": datetime(2025, 1, 1),
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
        "execution_timeout": timedelta(hours=2),
    }

@dag(
    dag_id="homeowner_loss_history_full_pipeline",
    default_args=_default_args(),
    schedule_interval="0 10 * * *",  # daily at 10 AM
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

    # 3️⃣ Drift‑check: existing reference?
    @task.branch()
    def _branch(path: str) -> str:
        if os.path.exists(REFERENCE_MEANS_PATH):
            ref = generate_reference_means(path, REFERENCE_MEANS_PATH)
            flag = detect_data_drift(path, ref)
            return "healing_task" if flag == "self_healing" else "override_or_train"
        return "override_or_train"
    branch = _branch(processed_path)

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
    
    train_all_models = EmptyOperator(task_id="train_all_models")
    for m in MODEL_IDS:
        PythonOperator(
            task_id=f"train_compare_{m}",
            python_callable=train_and_compare_fn,
            op_kwargs={"model_id": m, "processed_path": LOCAL_PROCESSED_PATH},
        ).set_downstream(join_after_training)
    
    # Wire the training branch
    training_branch() >> train_all_models
    for m in MODEL_IDS:
        training_branch() >> PythonOperator(
            task_id=f"train_compare_{m}_direct",
            python_callable=train_and_compare_fn,
            op_kwargs={"model_id": m, "processed_path": LOCAL_PROCESSED_PATH},
        ).set_downstream(join_after_training)

    # wire drift → heal → override/train → training
    branch >> heal >> override_branch()
    branch >> override_branch()
    override_branch() >> apply_override() >> training_branch()
    override_branch() >> start_training >> training_branch()
    override_branch() >> retrain_model() >> training_branch()

    # 6️⃣ Metrics, Notify, Archive
    @task()
    def record_metrics_task():
        record_system_metrics(runtime=time.time())

    @task()
    def notify_complete():
        send_message(
            channel="#alerts",
            title="✅ Pipeline Complete",
            details="Training & SHAP logging finished.",
            urgency="low",
        )

    @task()
    def archive():
        upload_to_s3("/home/airflow/logs/homeowner_dag.log", "logs/homeowner_dag.log")
        upload_to_s3(LOCAL_PROCESSED_PATH, "archive/homeowner_processed.parquet")
        for f in (LOCAL_PROCESSED_PATH, REFERENCE_MEANS_PATH):
            try:
                os.remove(f)
            except OSError:
                pass

    join_after_training >> record_metrics_task() >> notify_complete() >> archive()

    # 7️⃣ Start WebSocket server for real-time updates
    @task()
    def start_websocket_server():
        update_monitoring_with_ui_components()
        send_message(
            channel="#alerts",
            title="🔄 WebSocket Server Started",
            details="WebSocket server started for real-time dashboard updates",
            urgency="low",
        )

    # Define dependencies for new tasks
    join_after_training >> start_websocket_server()

dag = homeowner_pipeline()
