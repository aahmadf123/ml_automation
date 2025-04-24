"""
Tasks package for ML Automation.

This package contains all task implementations for the ML Automation DAGs.
Each task is designed to be a self-contained unit of work that can be
executed independently within the Airflow DAG.

Tasks are organized by functionality:
- Data ingestion and preprocessing
- Model training and evaluation
- Monitoring and validation
- Testing and deployment
"""

import os
import sys
from typing import List

# Add the dags directory to the Python path
dags_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dags_dir not in sys.path:
    sys.path.append(dags_dir)

# Import all task functions and classes to properly expose them
from .ingestion import ingest_data_from_s3
from .preprocessing import preprocess_data
from .schema_validation import validate_schema, snapshot_schema
from .training import train_and_compare_fn
from .model_explainability import ModelExplainabilityTracker
from .ab_testing import ABTestingPipeline
from .data_quality import DataQualityMonitor, manual_override
from .drift import generate_reference_means, detect_data_drift, self_healing
from .monitoring import record_system_metrics, update_monitoring_with_ui_components

# Import the caching system for use across modules
from utils.cache import GLOBAL_CACHE, cache_result, DataFrameCache

# Define exports
__all__: List[str] = [
    # Data ingestion and preprocessing
    'ingest_data_from_s3',
    'preprocess_data',
    'validate_schema',
    'snapshot_schema',
    
    # Model training and evaluation
    'train_and_compare_fn',
    'ModelExplainabilityTracker',
    'ABTestingPipeline',
    
    # Monitoring and validation
    'DataQualityMonitor',
    'generate_reference_means',
    'detect_data_drift',
    'self_healing',
    'record_system_metrics',
    'update_monitoring_with_ui_components',
    
    # Testing and deployment
    'manual_override',
    
    # Caching system
    'GLOBAL_CACHE',
    'cache_result',
    'DataFrameCache'
]

# Version information
__version__ = "1.0.0"
