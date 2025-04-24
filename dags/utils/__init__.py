"""
Utils package for ML Automation DAGs.

This package contains common utilities used across DAGs:
- config: Configuration management
- storage: S3 storage operations
- slack: Slack notifications
- security: Security utilities
- metrics: ML metrics calculations
- s3: S3 utility functions
- notifications: Notification utilities
- cache: Caching system
"""

import os
import sys
from typing import List

# Ensure the correct Python paths are set up
current_dir = os.path.dirname(os.path.abspath(__file__))
dags_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(dags_dir)

# Add paths if they don't exist
for path in [current_dir, dags_dir, parent_dir]:
    if path not in sys.path:
        sys.path.append(path)

# Define exports
__all__ = [
    # Config
    'DATA_BUCKET',
    'AWS_REGION',
    'REFERENCE_KEY_PREFIX',
    'DRIFT_THRESHOLD',
    'ConfigManager',
    'S3_BUCKET',
    'MLFLOW_URI',
    'MLFLOW_EXPERIMENT',
    
    # Storage
    'upload',
    'download',
    'upload_to_s3',
    'download_from_s3',
    
    # Notifications
    'post',
    'send_slack_notification',
    
    # Security
    'SecurityUtils',
    'validate_input',
    
    # Metrics utilities
    'calculate_metrics',
    'compare_models',
    'create_comparison_plots',
    'should_retrain',
    
    # Cache system
    'GLOBAL_CACHE',
    'cache_result',
    'DataFrameCache'
]

# Import utils modules
from utils.config import (
    DATA_BUCKET,
    AWS_REGION,
    REFERENCE_KEY_PREFIX,
    DRIFT_THRESHOLD,
    S3_BUCKET,
    MLFLOW_URI,
    MLFLOW_EXPERIMENT,
    ConfigManager
)
from utils.storage import upload, download
from utils.slack import post
from utils.security import SecurityUtils, validate_input
from utils.metrics import calculate_metrics, compare_models, create_comparison_plots, should_retrain
from utils.s3 import upload_to_s3, download_from_s3
from utils.notifications import send_slack_notification
from utils.cache import GLOBAL_CACHE, cache_result, DataFrameCache

"""
Utils package for ML Automation DAGs.
"""
