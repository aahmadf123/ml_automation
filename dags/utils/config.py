# utils/config.py

from dotenv import load_dotenv
load_dotenv()

import os
import boto3
import logging
import time
from functools import lru_cache
from typing import Dict, Any, Optional, Union
from airflow.models import Variable

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize AWS clients
ssm = boto3.client('ssm')

# Cache for SSM parameters
PARAM_CACHE = {}
CACHE_TTL = 300  # 5 minutes

@lru_cache(maxsize=100)
def get_ssm_parameter(param_name: str, default_value: Optional[Any] = None) -> Any:
    """
    Get a parameter from SSM Parameter Store with caching and robust error handling.
    Falls back to environment variable or Airflow Variable if not found.
    
    Args:
        param_name: Name of the parameter to retrieve
        default_value: Default value to return if parameter not found
        
    Returns:
        Parameter value from SSM, environment variable, or Airflow Variable
    """
    # Check cache first
    cache_key = f"{param_name}:{default_value}"
    if cache_key in PARAM_CACHE:
        cached_value, timestamp = PARAM_CACHE[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            logger.debug(f"Retrieved parameter {param_name} from cache")
            return cached_value
    
    try:
        # Try to get from SSM
        response = ssm.get_parameter(
            Name=f'/ml-automation/{param_name}',
            WithDecryption=True  # For SecureString parameters
        )
        value = response['Parameter']['Value']
        logger.info(f"Retrieved parameter {param_name} from SSM")
        
        # Cache the value
        PARAM_CACHE[cache_key] = (value, time.time())
        return value
        
    except ssm.exceptions.ParameterNotFound:
        logger.warning(f"Parameter {param_name} not found in SSM")
        # Fall back to environment variable or Airflow Variable
        if default_value is not None:
            value = os.getenv(param_name) or Variable.get(param_name, default_var=default_value)
        else:
            value = os.getenv(param_name) or Variable.get(param_name)
            
        # Cache the fallback value
        PARAM_CACHE[cache_key] = (value, time.time())
        return value
        
    except Exception as e:
        logger.error(f"Error retrieving parameter {param_name}: {str(e)}")
        # Fall back to environment variable or Airflow Variable
        if default_value is not None:
            value = os.getenv(param_name) or Variable.get(param_name, default_var=default_value)
        else:
            value = os.getenv(param_name) or Variable.get(param_name)
            
        # Cache the fallback value
        PARAM_CACHE[cache_key] = (value, time.time())
        return value

def validate_numeric_parameter(param_name: str, value: Union[int, float], min_value: float, max_value: float) -> None:
    """
    Validate a numeric parameter is within acceptable range.
    
    Args:
        param_name: Name of the parameter for error messages
        value: The parameter value to validate
        min_value: Minimum acceptable value
        max_value: Maximum acceptable value
        
    Raises:
        ValueError: If value is outside acceptable range
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"Parameter {param_name} must be numeric, got {type(value)}")
    
    if value < min_value or value > max_value:
        raise ValueError(f"Parameter {param_name} must be between {min_value} and {max_value}, got {value}")

# ─── S3 CONFIG ────────────────────────────────────────────────────────────────
S3_BUCKET           = get_ssm_parameter('S3_BUCKET')
RAW_DATA_KEY        = "raw-data/ut_loss_history_1.csv"
REFERENCE_KEY       = "reference/reference_means.csv"
REFERENCE_KEY_PREFIX= "reference"
MODELS_FOLDER       = "models"
LOGS_FOLDER         = "logs"
ARCHIVE_FOLDER      = get_ssm_parameter('S3_ARCHIVE_FOLDER', 'archive')

# ─── MLFLOW CONFIG ────────────────────────────────────────────────────────────
MLFLOW_URI          = get_ssm_parameter('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT   = Variable.get(
    "MLFLOW_EXPERIMENT_NAME",
    default_var="Homeowner_Loss_Hist_Proj"
)

# ─── AIRFLOW / DAG CONFIG ────────────────────────────────────────────────────
DEFAULT_START_DATE   = "2025-01-01"
SCHEDULE_CRON        = "0 10 * * *"  # daily at 10 AM
AIRFLOW_DAG_BASE_CONF= {}

# ─── SLACK CONFIG ────────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL    = get_ssm_parameter('SLACK_WEBHOOK_URL')
SLACK_CHANNEL_DEFAULT= get_ssm_parameter('SLACK_CHANNEL_DEFAULT', '#alerts')

# ─── HYPEROPT CONFIG ─────────────────────────────────────────────────────────
MAX_EVALS           = int(get_ssm_parameter('HYPEROPT_MAX_EVALS', '20'))
validate_numeric_parameter('HYPEROPT_MAX_EVALS', MAX_EVALS, 1, 1000)

# ─── UI COMPONENTS CONFIG ────────────────────────────────────────────────────
UI_COMPONENTS = {
    "sidebar": {
        "title": "ML Automation Dashboard",
        "logo": "assets/logo.png",
        "menu_items": [
            {"label": "Dashboard", "icon": "dashboard", "path": "/"},
            {"label": "Models", "icon": "model_training", "path": "/models"},
            {"label": "Data Quality", "icon": "data_object", "path": "/data-quality"},
            {"label": "Drift Detection", "icon": "trending_up", "path": "/drift"},
            {"label": "Alerts", "icon": "notifications", "path": "/alerts"},
            {"label": "Settings", "icon": "settings", "path": "/settings"}
        ]
    },
    "header": {
        "title": "ML Automation Dashboard",
        "subtitle": "Monitor and manage your ML models",
        "actions": [
            {"label": "Refresh", "icon": "refresh", "action": "refresh_data"},
            {"label": "Settings", "icon": "settings", "action": "open_settings"}
        ]
    },
    "dashboard": {
        "layout": "grid",
        "widgets": [
            {"type": "metrics", "title": "Model Performance", "size": "large"},
            {"type": "drift", "title": "Data Drift", "size": "medium"},
            {"type": "alerts", "title": "Recent Alerts", "size": "small"},
            {"type": "quality", "title": "Data Quality", "size": "medium"}
        ]
    }
}

# ─── MODEL CONFIG ────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "model1": {
        "name": "Baseline Model",
        "description": "Standard XGBoost model with default parameters",
        "features": ["num_loss_3yr_", "num_loss_yrs45_", "num_loss_free_yrs_"],
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100
        }
    },
    "model2": {
        "name": "Equal Weight Model",
        "description": "XGBoost model with equal weighting of loss history",
        "features": ["lhdwc_5y_1d_"],
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100
        }
    },
    "model3": {
        "name": "Linear Decay Model",
        "description": "XGBoost model with linear decay weighting of loss history",
        "features": ["lhdwc_5y_2d_"],
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100
        }
    },
    "model4": {
        "name": "Fast Decay Model",
        "description": "XGBoost model with fast decay weighting of loss history",
        "features": ["lhdwc_5y_3d_"],
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100
        }
    },
    "model5": {
        "name": "Slow Decay Model",
        "description": "XGBoost model with slow decay weighting of loss history",
        "features": ["lhdwc_5y_4d_"],
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100
        }
    }
}

# ─── DATA QUALITY CONFIG ────────────────────────────────────────────────────
DATA_QUALITY_CONFIG = {
    "missing_threshold": float(get_ssm_parameter('MISSING_THRESHOLD', '0.05')),
    "outlier_threshold": float(get_ssm_parameter('OUTLIER_THRESHOLD', '3.0')),
    "drift_threshold": float(get_ssm_parameter('DRIFT_THRESHOLD', '0.1')),
    "correlation_threshold": float(get_ssm_parameter('CORRELATION_THRESHOLD', '0.7'))
}

# Validate data quality thresholds
validate_numeric_parameter('MISSING_THRESHOLD', DATA_QUALITY_CONFIG['missing_threshold'], 0, 1)
validate_numeric_parameter('OUTLIER_THRESHOLD', DATA_QUALITY_CONFIG['outlier_threshold'], 1, 10)
validate_numeric_parameter('DRIFT_THRESHOLD', DATA_QUALITY_CONFIG['drift_threshold'], 0, 1)
validate_numeric_parameter('CORRELATION_THRESHOLD', DATA_QUALITY_CONFIG['correlation_threshold'], 0, 1)

# ─── WEBSOCKET CONFIG ────────────────────────────────────────────────────────
WEBSOCKET_CONFIG = {
    "host": get_ssm_parameter('WS_HOST', 'localhost'),
    "port": int(get_ssm_parameter('WS_PORT', '8765')),
    "ping_interval": int(get_ssm_parameter('WS_PING_INTERVAL', '30'))
}

# Validate WebSocket config
validate_numeric_parameter('WS_PORT', WEBSOCKET_CONFIG['port'], 1, 65535)
validate_numeric_parameter('WS_PING_INTERVAL', WEBSOCKET_CONFIG['ping_interval'], 5, 300)

# ─── API CONFIG ──────────────────────────────────────────────────────────────
API_CONFIG = {
    "host": get_ssm_parameter('API_HOST', 'localhost'),
    "port": int(get_ssm_parameter('API_PORT', '3000')),
    "rate_limit": int(get_ssm_parameter('API_RATE_LIMIT', '100'))
}

# Validate API config
validate_numeric_parameter('API_PORT', API_CONFIG['port'], 1, 65535)
validate_numeric_parameter('API_RATE_LIMIT', API_CONFIG['rate_limit'], 1, 10000)

# ─── SECURITY CONFIG ────────────────────────────────────────────────────────
SECURITY_CONFIG = {
    "jwt_secret": get_ssm_parameter('JWT_SECRET', 'your-secret-key'),
    "jwt_expiry": int(get_ssm_parameter('JWT_EXPIRY', '3600')),
    "cors_origins": get_ssm_parameter('CORS_ORIGINS', '*').split(','),
    "rate_limit_window": int(get_ssm_parameter('RATE_LIMIT_WINDOW', '3600')),
    "rate_limit_max_requests": int(get_ssm_parameter('RATE_LIMIT_MAX_REQUESTS', '1000'))
}

# Validate security config
validate_numeric_parameter('JWT_EXPIRY', SECURITY_CONFIG['jwt_expiry'], 60, 86400)
validate_numeric_parameter('RATE_LIMIT_WINDOW', SECURITY_CONFIG['rate_limit_window'], 60, 86400)
validate_numeric_parameter('RATE_LIMIT_MAX_REQUESTS', SECURITY_CONFIG['rate_limit_max_requests'], 1, 10000)

# ─── AIRFLOW CONFIG ────────────────────────────────────────────────────────
AIRFLOW_CONFIG = {
    "api_url": get_ssm_parameter('AIRFLOW_API_URL', 'http://localhost:8080'),
    "username": get_ssm_parameter('AIRFLOW_USERNAME', 'admin'),
    "password": get_ssm_parameter('AIRFLOW_PASSWORD', 'admin'),
    "dag_folder": get_ssm_parameter('AIRFLOW_DAG_FOLDER', '/opt/airflow/dags')
}

# ─── FEATURE CONFIG ────────────────────────────────────────────────────────
FEATURE_CONFIG = {
    "numeric_features": [
        "num_loss_3yr_",
        "num_loss_yrs45_",
        "num_loss_free_yrs_",
        "lhdwc_5y_1d_",
        "lhdwc_5y_2d_",
        "lhdwc_5y_3d_",
        "lhdwc_5y_4d_"
    ],
    "categorical_features": [
        "state",
        "zipcode",
        "policy_type",
        "construction_type",
        "roof_type",
        "occupancy_type"
    ],
    "target_feature": "pure_premium"
}

# ─── CLASS CONFIG ──────────────────────────────────────────────────────────
class Config:
    """Centralized configuration management."""
    
    # S3 Configuration
    S3_BUCKET = get_ssm_parameter('S3_BUCKET')
    S3_DATA_FOLDER = "raw-data"
    S3_ARCHIVE_FOLDER = get_ssm_parameter('S3_ARCHIVE_FOLDER', 'archive')
    S3_REFERENCE_KEY_PREFIX = "reference"
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = get_ssm_parameter('MLFLOW_TRACKING_URI')
    MLFLOW_EXPERIMENT_NAME = "homeowner_loss_history"
    
    # Model Configuration
    MODEL_IDS = ["model1", "model2", "model3", "model4", "model5"]
    MODEL_METRICS = ["rmse", "mse", "mae", "r2"]
    MODEL_THRESHOLDS = {
        "rmse": float(get_ssm_parameter('RMSE_THRESHOLD', '0.1')),
        "drift": float(get_ssm_parameter('DRIFT_THRESHOLD', '0.1')),
        "correlation": float(get_ssm_parameter('CORRELATION_THRESHOLD', '0.7'))
    }
    
    # Data Quality Configuration
    DATA_QUALITY_CONFIG = DATA_QUALITY_CONFIG
    
    # WebSocket Configuration
    WS_HOST = get_ssm_parameter('WS_HOST', 'localhost')
    WS_PORT = int(get_ssm_parameter('WS_PORT', '8765'))
    WS_PING_INTERVAL = int(get_ssm_parameter('WS_PING_INTERVAL', '30'))
    
    # API Configuration
    API_HOST = get_ssm_parameter('API_HOST', 'localhost')
    API_PORT = int(get_ssm_parameter('API_PORT', '3000'))
    API_RATE_LIMIT = int(get_ssm_parameter('API_RATE_LIMIT', '100'))
    
    # Security Configuration
    SECURITY_CONFIG = SECURITY_CONFIG
    
    # Slack Configuration
    SLACK_WEBHOOK_URL = get_ssm_parameter('SLACK_WEBHOOK_URL')
    SLACK_CHANNELS = {
        "alerts": "#alerts",
        "logs": "#agent_logs",
        "metrics": "#metrics"
    }
    
    # Airflow Configuration
    AIRFLOW_CONFIG = AIRFLOW_CONFIG
    
    # Feature Configuration
    FEATURE_CONFIG = FEATURE_CONFIG
    
    @classmethod
    def get_model_config(cls, model_id: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_id in MODEL_CONFIG:
            return MODEL_CONFIG[model_id]
        return {}
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate the configuration."""
        # Required parameters
        required_params = {
            'S3_BUCKET': str,
            'MLFLOW_TRACKING_URI': str,
            'SLACK_WEBHOOK_URL': str
        }
        
        for param, param_type in required_params.items():
            value = getattr(cls, param)
            if not value:
                raise ValueError(f"Missing required parameter: {param}")
            if not isinstance(value, param_type):
                raise TypeError(f"Invalid type for {param}: expected {param_type}")
        
        # Validate model IDs
        for model_id in cls.MODEL_IDS:
            if model_id not in MODEL_CONFIG:
                raise ValueError(f"Model ID {model_id} not found in MODEL_CONFIG")
        
        # Validate feature configuration
        if not cls.FEATURE_CONFIG.get('numeric_features'):
            raise ValueError("Missing numeric_features in FEATURE_CONFIG")
        if not cls.FEATURE_CONFIG.get('categorical_features'):
            raise ValueError("Missing categorical_features in FEATURE_CONFIG")
        if not cls.FEATURE_CONFIG.get('target_feature'):
            raise ValueError("Missing target_feature in FEATURE_CONFIG")
        
        # Validate thresholds
        for threshold_name, threshold_value in cls.MODEL_THRESHOLDS.items():
            validate_numeric_parameter(threshold_name, threshold_value, 0, 1)
        
        # Validate data quality config
        for key, value in cls.DATA_QUALITY_CONFIG.items():
            validate_numeric_parameter(key, value, 0, 1)
        
        # Validate WebSocket config
        validate_numeric_parameter('WS_PORT', cls.WS_PORT, 1, 65535)
        validate_numeric_parameter('WS_PING_INTERVAL', cls.WS_PING_INTERVAL, 5, 300)
        
        # Validate API config
        validate_numeric_parameter('API_PORT', cls.API_PORT, 1, 65535)
        validate_numeric_parameter('API_RATE_LIMIT', cls.API_RATE_LIMIT, 1, 10000)
        
        # Validate security config
        validate_numeric_parameter('JWT_EXPIRY', cls.SECURITY_CONFIG['jwt_expiry'], 60, 86400)
        validate_numeric_parameter('RATE_LIMIT_WINDOW', cls.SECURITY_CONFIG['rate_limit_window'], 60, 86400)
        validate_numeric_parameter('RATE_LIMIT_MAX_REQUESTS', cls.SECURITY_CONFIG['rate_limit_max_requests'], 1, 10000)

# Validate configuration on import
Config.validate_config()
