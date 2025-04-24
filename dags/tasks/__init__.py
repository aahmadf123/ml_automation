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

# Ensure the correct Python paths are set up
# In a container environment like Airflow, the dags directory is already in the path
# but we need to handle local development environment as well
current_dir = os.path.dirname(os.path.abspath(__file__))
dags_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(dags_dir)

# Add paths if they don't exist
for path in [current_dir, dags_dir, parent_dir]:
    if path not in sys.path:
        sys.path.append(path)

# Define exports - these are exposed when doing 'from tasks import X'
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

# Import all task functions and classes to properly expose them
# These imports need to be AFTER the path setup and AFTER the __all__ definition
from tasks.ingestion import ingest_data_from_s3
from tasks.preprocessing import preprocess_data
from tasks.schema_validation import validate_schema, snapshot_schema
from tasks.training import train_and_compare_fn
from tasks.model_explainability import ModelExplainabilityTracker
from tasks.ab_testing import ABTestingPipeline
from tasks.data_quality import DataQualityMonitor, manual_override
from tasks.drift import generate_reference_means, detect_data_drift, self_healing
from tasks.monitoring import record_system_metrics, update_monitoring_with_ui_components

# Import the caching system for use across modules
from utils.cache import GLOBAL_CACHE, cache_result, DataFrameCache

# Version information
__version__ = "1.0.0"
