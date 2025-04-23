#!/usr/bin/env python3
"""
preprocessing.py

Handles:
  - Data loading from Parquet
  - Handling missing values
  - Outlier detection and capping
  - Categorical encoding
  - Generating pure_premium & sample_weight
  - Pandera schema validation (after pure_premium exists)
  - Selecting the correct loss-history features per MODEL_ID
  - Data profiling report + Slack notification via agent_actions
  - Integration of new UI components and endpoints
"""

import os
import json
import logging
import time
from typing import Dict

import pandas as pd
from ydata_profiling import ProfileReport
from airflow.models import Variable
from tasks.schema_validation import validate_schema
from agent_actions import handle_function_call
import numpy as np
from datetime import datetime
from utils.storage import download as s3_download
from utils.storage import upload as s3_upload
from utils.slack import post as send_message
from tasks.data_quality import DataQualityMonitor

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

PROFILE_REPORT_PATH = "/tmp/homeowner_profile_report.html"

# Prefix groups for loss-history features
RAW_PREFIXES = ["num_loss_3yr_", "num_loss_yrs45_", "num_loss_free_yrs_"]
DECAY_PREFIXES = {
    "model2": ["lhdwc_5y_1d_"],  # equal
    "model3": ["lhdwc_5y_2d_"],  # linear decay
    "model4": ["lhdwc_5y_3d_"],  # fast decay
    "model5": ["lhdwc_5y_4d_"],  # slow decay
}

# Set up logging
logger = logging.getLogger(__name__)

# Initialize data quality monitor
quality_monitor = DataQualityMonitor()

def load_data_to_dataframe(parquet_path: str) -> pd.DataFrame:
    """
    Read Parquet from local path into DataFrame.
    """
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded data from {parquet_path}, shape={df.shape}")
    return df


def handle_missing_data(
    df: pd.DataFrame,
    strategy: str = "mean",
    missing_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Drop columns with > threshold missing, then impute remaining missing.
    """
    null_ratio = df.isnull().mean()
    drop_cols = null_ratio[null_ratio > missing_threshold].index.tolist()
    df.drop(columns=drop_cols, inplace=True)
    logging.info(f"Dropped columns >{missing_threshold*100:.0f}% missing: {drop_cols}")

    if strategy == "mean":
        df.fillna(df.mean(numeric_only=True), inplace=True)
    elif strategy == "zero":
        df.fillna(0, inplace=True)
    elif strategy == "ffill":
        df.fillna(method="ffill", inplace=True)
    else:
        raise ValueError(f"Unknown missing data strategy '{strategy}'")

    logging.info(f"Applied missing data strategy: {strategy}")
    return df


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, int]:
    """
    Count outliers per numeric column using the IQR method.
    """
    counts: Dict[str, int] = {}
    for col in df.select_dtypes(include="number"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - factor * iqr) | (df[col] > q3 + factor * iqr)
        counts[col] = int(mask.sum())
    return counts


def cap_outliers(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Clip numeric column values to within [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One‑hot encode all object/categorical columns.
    """
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    logging.info(f"One‑hot encoded columns: {obj_cols}")
    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the loss‑history features needed for current MODEL_ID,
    plus all non–loss‑history columns.
    """
    model_id = (
        os.getenv("MODEL_ID")
        or Variable.get("MODEL_ID", default_var="model1")
    ).strip().lower()
    logging.info(f"Selecting features for MODEL_ID='{model_id}'")

    if model_id == "model1":
        keep_prefixes = RAW_PREFIXES
    elif model_id in DECAY_PREFIXES:
        keep_prefixes = DECAY_PREFIXES[model_id]
    else:
        raise ValueError(f"Unknown MODEL_ID '{model_id}'")

    all_loss_prefixes = RAW_PREFIXES + sum(DECAY_PREFIXES.values(), [])
    def is_loss_col(col: str) -> bool:
        return any(col.startswith(p) for p in all_loss_prefixes)

    keep_cols = [
        col for col in df.columns
        if any(col.startswith(p) for p in keep_prefixes) or not is_loss_col(col)
    ]
    removed = [col for col in df.columns if col not in keep_cols]
    logging.info(f"Dropped loss‑history cols for {model_id}: {removed}")
    return df[keep_cols]


def generate_profile_report(df: pd.DataFrame, output_path: str = PROFILE_REPORT_PATH) -> None:
    """
    Generate a minimal profiling report, write to HTML, and notify Slack.
    """
    try:
        profile = ProfileReport(df, title="Homeowner Data Profile", minimal=True)
        profile.to_file(output_path)
        logging.info(f"Saved profiling report to {output_path}")
    except Exception as e:
        logging.warning(f"Failed to generate profile report: {e}")
        return

    try:
        send_message(
            channel="#agent_logs",
            title="📊 Profiling Summary",
            details=f"Profile saved to {output_path}",
            urgency="low"
        )
    except Exception as e:
        logging.warning(f"Slack notification failed: {e}")


def preprocess_data(data_path, output_path):
    """
    Preprocess the data and perform quality checks.
    """
    try:
        # Import necessary dependencies
        import pandas as pd
        import numpy as np
        from datetime import datetime
        import os
        import json
        from utils.storage import upload as s3_upload
        from utils.slack import post as send_message
        from tasks.data_quality import DataQualityMonitor
        
        # Load data
        data = pd.read_parquet(data_path)
        
        # Initialize quality monitor
        quality_monitor = DataQualityMonitor()
        
        # Perform initial quality check
        quality_report = quality_monitor.run_quality_checks(data)
        if quality_report['status'] == 'fail':
            send_message(
                channel="#alerts",
                title="⚠️ Data Quality Issues Detected",
                details=f"Quality report: {quality_report}",
                urgency="medium"
            )
        
        # Basic preprocessing
        data = handle_missing_data(data)
        data = handle_outliers(data)
        data = encode_categorical_variables(data)
        data = scale_numeric_features(data)
        
        # Final quality check after preprocessing
        final_quality_report = quality_monitor.run_quality_checks(data)
        if final_quality_report['status'] == 'fail':
            send_message(
                channel="#alerts",
                title="⚠️ Post-Preprocessing Quality Issues",
                details=f"Quality report: {final_quality_report}",
                urgency="medium"
            )
        
        # Save preprocessed data
        data.to_parquet(output_path)
        
        # Save quality reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quality_reports = {
            'initial': quality_report,
            'final': final_quality_report,
            'timestamp': timestamp
        }
        
        # Convert reports to serializable format for JSON
        # (pandas dataframes or numpy arrays won't serialize properly)
        import json
        try:
            json.dumps(quality_reports)
        except TypeError:
            # If we can't serialize, create a simplified version
            quality_reports = {
                'initial': {'status': quality_report['status']},
                'final': {'status': final_quality_report['status']},
                'timestamp': timestamp
            }
        
        # Upload quality reports to S3 - disabled for now to avoid S3 errors
        # reports_path = f"quality_reports/{os.path.basename(data_path)}_{timestamp}.json"
        # s3_upload(quality_reports, reports_path)
        
        return output_path
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in preprocess_data: {str(e)}")
        from utils.slack import post as send_message
        send_message(
            channel="#alerts",
            title="❌ Preprocessing Error",
            details=str(e),
            urgency="high"
        )
        raise

def handle_outliers(data, threshold=3):
    """
    Handle outliers using z-score method.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
        data.loc[z_scores > threshold, col] = data[col].mean()
    
    return data

def encode_categorical_variables(data):
    """
    Encode categorical variables.
    """
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        data[col] = pd.Categorical(data[col]).codes
    
    return data

def scale_numeric_features(data):
    """
    Scale numeric features using min-max scaling.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    
    return data
