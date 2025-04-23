#!/usr/bin/env python3
"""
tasks/drift.py

Handles:
  - Generation of reference means from the processed data.
  - Data drift detection by comparing current data with versioned reference means.
  - A self‑healing routine when drift is detected.

Refactored to:
  - Use utils/storage.upload instead of boto3 directly.
  - Version the reference means file on upload under a timestamped prefix.
  - Wrap S3 operations with tenacity retries.
  - Load drift threshold from Airflow Variable "DRIFT_THRESHOLD".
"""

import os
import logging
import time
import pandas as pd
import numpy as np
from airflow.models import Variable
from tenacity import retry, stop_after_attempt, wait_fixed
import json
import boto3
from datetime import datetime
from plugins.utils.slack import post as send_message
from plugins.utils.storage import download as s3_download
from plugins.utils.storage import upload as s3_upload
from plugins.utils.config import (
    DATA_BUCKET, REFERENCE_KEY_PREFIX, AWS_REGION,
    DRIFT_THRESHOLD
)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Local path for the reference means CSV
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"

# Initialize AWS clients with retry
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def initialize_aws_clients():
    """Initialize AWS clients with retry logic."""
    try:
        cloudwatch = boto3.client('cloudwatch', region_name=AWS_REGION)
        lambda_client = boto3.client('lambda', region_name=AWS_REGION)
        ssm = boto3.client('ssm', region_name=AWS_REGION)
        secretsmanager = boto3.client('secretsmanager', region_name=AWS_REGION)
        return cloudwatch, lambda_client, ssm, secretsmanager
    except Exception as e:
        logger.error(f"Failed to initialize AWS clients: {e}")
        raise

try:
    cloudwatch, lambda_client, ssm, secretsmanager = initialize_aws_clients()
except Exception as e:
    logger.error(f"Could not initialize AWS clients after retries: {e}")
    raise

# Get drift threshold from SSM Parameter Store or use default
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_drift_threshold():
    """Get drift threshold from SSM with retry logic."""
    try:
        drift_threshold_param = ssm.get_parameter(Name='/ml-automation/drift/threshold')
        threshold = float(drift_threshold_param['Parameter']['Value'])
        logger.info(f"Using drift threshold from SSM: {threshold}")
        return threshold
    except Exception as e:
        logger.warning(f"Failed to get drift threshold from SSM: {e}")
        try:
            threshold = float(Variable.get("DRIFT_THRESHOLD", default_var="0.1"))
            logger.info(f"Using drift threshold from Airflow Variable: {threshold}")
            return threshold
        except Exception:
            logger.warning("Using default drift threshold 0.1")
            return 0.1

DRIFT_THRESHOLD = get_drift_threshold()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_reference_means(
    processed_path: str,
    local_ref: str = REFERENCE_MEANS_PATH
) -> str:
    """
    Generates a timestamped reference means CSV and uploads it to S3 under the configured prefix.

    Args:
        processed_path: Path to the latest processed data parquet.
        local_ref: Local path to write the reference means.

    Returns:
        The local path to the generated reference means CSV.
    """
    df = pd.read_parquet(processed_path)
    means = (
        df.select_dtypes(include=[np.number])
          .mean()
          .reset_index()
          .rename(columns={"index": "column_name", 0: "mean_value"})
    )
    means.to_csv(local_ref, index=False)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    s3_key = f"{REFERENCE_KEY_PREFIX}/reference_means_{ts}.csv"
    s3_upload(local_ref, s3_key)
    logger.info(f"Uploaded reference means to s3://{DATA_BUCKET}/{s3_key}")
    
    # Log reference means generation to CloudWatch
    cloudwatch.put_metric_data(
        Namespace='MLAutomation',
        MetricData=[
            {
                'MetricName': 'ReferenceMeansGenerated',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': datetime.now()
            }
        ]
    )
    
    return local_ref

def calculate_drift(current_data, reference_data, columns):
    """
    Calculate drift metrics for specified columns.
    """
    drift_metrics = {}
    
    for col in columns:
        if col in current_data.columns and col in reference_data.columns:
            current_mean = current_data[col].mean()
            reference_mean = reference_data[col].mean()
            
            drift_metrics[col] = {
                'current_mean': current_mean,
                'reference_mean': reference_mean,
                'drift': abs(current_mean - reference_mean) / reference_mean if reference_mean != 0 else 0
            }
    
    return drift_metrics

def detect_drift(current_data_path, reference_data_path, drift_threshold=0.1):
    """
    Detect data drift between current and reference datasets.
    """
    try:
        # Load datasets
        current_data = pd.read_parquet(current_data_path)
        reference_data = pd.read_parquet(reference_data_path)
        
        # Calculate drift for numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        drift_metrics = calculate_drift(current_data, reference_data, numeric_cols)
        
        # Identify significant drift
        significant_drift = {
            col: metrics for col, metrics in drift_metrics.items()
            if metrics['drift'] > drift_threshold
        }
        
        # Prepare drift report
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': drift_metrics,
            'significant_drift': significant_drift,
            'threshold': drift_threshold
        }
        
        # Save drift report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"drift_reports/drift_{timestamp}.json"
        s3_upload(drift_report, report_path)
        
        # Log drift metrics to CloudWatch
        for col, metrics in drift_metrics.items():
            cloudwatch.put_metric_data(
                Namespace='MLAutomation',
                MetricData=[
                    {
                        'MetricName': 'FeatureDrift',
                        'Value': metrics['drift'],
                        'Unit': 'None',
                        'Timestamp': datetime.now(),
                        'Dimensions': [
                            {
                                'Name': 'Feature',
                                'Value': col
                            }
                        ]
                    }
                ]
            )
        
        # Log significant drift count
        cloudwatch.put_metric_data(
            Namespace='MLAutomation',
            MetricData=[
                {
                    'MetricName': 'SignificantDriftCount',
                    'Value': len(significant_drift),
                    'Unit': 'Count',
                    'Timestamp': datetime.now()
                }
            ]
        )
        
        # Trigger WebSocket notification if significant drift detected
        if significant_drift:
            # Get WebSocket API endpoint from SSM
            try:
                websocket_endpoint = ssm.get_parameter(Name='/ml-automation/websocket/endpoint')['Parameter']['Value']
                lambda_client.invoke(
                    FunctionName='notify_websocket',
                    InvocationType='Event',
                    Payload=json.dumps({
                        'type': 'drift_alert',
                        'data': drift_report,
                        'endpoint': websocket_endpoint
                    })
                )
            except Exception as e:
                logger.error(f"Failed to trigger WebSocket notification: {e}")
            
            # Send Slack notification
            send_message(
                channel="#alerts",
                title="⚠️ Significant Data Drift Detected",
                details=f"Drift report: {json.dumps(drift_report, indent=2)}",
                urgency="high"
            )
            
            # Log alert to CloudWatch
            cloudwatch.put_metric_data(
                Namespace='MLAutomation',
                MetricData=[
                    {
                        'MetricName': 'DriftAlert',
                        'Value': 1,
                        'Unit': 'Count',
                        'Timestamp': datetime.now()
                    }
                ]
            )
        
        return drift_report
        
    except Exception as e:
        logger.error(f"Error in detect_drift: {str(e)}")
        send_message(
            channel="#alerts",
            title="❌ Drift Detection Error",
            details=str(e),
            urgency="high"
        )
        raise

def should_retrain(drift_report, retrain_threshold=0.2):
    """
    Determine if model retraining is needed based on drift metrics.
    """
    if not drift_report.get('significant_drift'):
        return False
    
    # Check if any feature has drift above retrain threshold
    for metrics in drift_report['significant_drift'].values():
        if metrics['drift'] > retrain_threshold:
            return True
    
    return False

def detect_data_drift(
    current_data_path: str,
    reference_means_path: str,
    threshold: float = None
) -> str:
    """
    Compares current data means vs. the reference means to detect drift.

    Args:
        current_data_path: Path to the current processed parquet.
        reference_means_path: Path to a versioned reference means CSV.
        threshold: Fractional threshold for drift; if None uses DRIFT_THRESHOLD.

    Returns:
        "self_healing" if drift detected, else "train_xgboost_hyperopt".
    """
    if threshold is None:
        threshold = DRIFT_THRESHOLD

    df_current = pd.read_parquet(current_data_path)
    df_ref     = pd.read_csv(reference_means_path)
    ref_map    = dict(zip(df_ref["column_name"], df_ref["mean_value"]))

    drifted = False
    for col in df_current.select_dtypes(include=[np.number]).columns:
        if col not in ref_map:
            logging.warning(f"No reference for '{col}', skipping.")
            continue
        ref_val = ref_map[col]
        curr_val = df_current[col].mean()
        if ref_val != 0:
            ratio = abs(curr_val - ref_val) / abs(ref_val)
            if ratio > threshold:
                logging.error(
                    f"Drift in '{col}': current={curr_val:.2f}, ref={ref_val:.2f}, drift={ratio:.2%}"
                )
                drifted = True
        else:
            logging.warning(f"Reference mean for '{col}' is zero, skipping drift check.")

    return "self_healing" if drifted else "train_xgboost_hyperopt"


def self_healing() -> str:
    """
    Simulates a self‑healing routine when drift is detected.
    In production, this could trigger remediation workflows.

    Returns:
        A marker string once self‑healing completes.
    """
    logging.info("Drift detected: starting self‑healing (await manual approval)...")
    
    # Log self-healing event to CloudWatch
    cloudwatch.put_metric_data(
        Namespace='MLAutomation',
        MetricData=[
            {
                'MetricName': 'SelfHealingTriggered',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': datetime.now()
            }
        ]
    )
    
    time.sleep(5)
    logging.info("Self‑healing complete; manual override confirmed.")
    return "override_done"

def update_drift_detection_with_ui_components():
    """
    Placeholder function to update the drift detection process with new UI components and endpoints.
    """
    logging.info("Updating drift detection process with new UI components and endpoints.")
    # Placeholder for actual implementation
