"""
tasks

A package containing all the task implementations for the ML Automation DAGs.
"""

import os
import sys

# Add the dags directory to the Python path
dags_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dags_dir not in sys.path:
    sys.path.append(dags_dir)

from .ingestion import ingest_data_from_s3
from .preprocessing import preprocess_data
from .schema_validation import validate_schema, snapshot_schema
from .data_quality import DataQualityMonitor
from .drift import generate_reference_means, detect_data_drift, self_healing
from .monitoring import record_system_metrics, update_monitoring_with_ui_components
from .model_explainability import ModelExplainabilityTracker
from .ab_testing import ABTestingPipeline
from .training import train_and_compare_fn, manual_override

__all__ = [
    'ingest_data_from_s3',
    'preprocess_data',
    'validate_schema',
    'snapshot_schema',
    'DataQualityMonitor',
    'generate_reference_means',
    'detect_data_drift',
    'self_healing',
    'record_system_metrics',
    'update_monitoring_with_ui_components',
    'ModelExplainabilityTracker',
    'ABTestingPipeline',
    'train_and_compare_fn',
    'manual_override'
]
