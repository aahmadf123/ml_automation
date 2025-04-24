# utils/config.py

from dotenv import load_dotenv
load_dotenv()

import os
import boto3
import logging
import time
from functools import lru_cache
from typing import Dict, Any, Optional, Union, List
from airflow.models import Variable
from pathlib import Path
import json
import yaml

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get AWS region from environment variable or use default
AWS_REGION = os.getenv('AWS_REGION', 'us-east-2')

# Initialize AWS clients with region
ssm = boto3.client('ssm', region_name=AWS_REGION)

# Cache for SSM parameters
PARAM_CACHE = {}
CACHE_TTL = 300  # 5 minutes

# Get bucket name from environment or Airflow variable
DATA_BUCKET = os.getenv('DATA_BUCKET', Variable.get('DATA_BUCKET', default_var='grange-seniordesign-bucket'))

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
S3_BUCKET           = DATA_BUCKET
RAW_DATA_KEY        = "raw-data/ut_loss_history_1.csv"
REFERENCE_KEY       = "reference/reference_means.csv"
REFERENCE_KEY_PREFIX= "reference"
MODELS_FOLDER       = "models"
MODEL_KEY_PREFIX    = "models"  # Prefix for model files in S3
LOGS_FOLDER         = "logs"
ARCHIVE_FOLDER      = get_ssm_parameter('S3_ARCHIVE_FOLDER', 'archive')

# ─── DRIFT CONFIG ────────────────────────────────────────────────────────────
DRIFT_THRESHOLD = float(get_ssm_parameter('DRIFT_THRESHOLD', '0.1'))
validate_numeric_parameter('DRIFT_THRESHOLD', DRIFT_THRESHOLD, 0, 1)

# ─── MLFLOW CONFIG ────────────────────────────────────────────────────────────
# Get MLflow tracking URI with fallback to local storage
try:
    MLFLOW_URI = get_ssm_parameter('MLFLOW_TRACKING_URI')
    # Test if the URI is valid
    if not MLFLOW_URI or not MLFLOW_URI.strip():
        logger.warning("MLflow tracking URI is empty, using local storage")
        MLFLOW_URI = "file:/tmp/mlruns"
except Exception as e:
    logger.warning(f"Failed to get MLflow tracking URI: {str(e)}, using local storage")
    MLFLOW_URI = "file:/tmp/mlruns"

# Get artifact store URI with fallback to S3 bucket path
MLFLOW_ARTIFACT_URI = get_ssm_parameter(
    'MLFLOW_ARTIFACT_URI', 
    's3://grange-seniordesign-bucket/mlflow-artifacts/'
)

# Get database URI with fallback to SQLite
MLFLOW_DB_URI = get_ssm_parameter(
    'MLFLOW_DB_URI', 
    'sqlite:////tmp/mlflow.db'
)

# Get model registry URI, defaults to tracking URI if not specified
MODEL_REGISTRY_URI = get_ssm_parameter('MODEL_REGISTRY_URI', MLFLOW_URI)

MLFLOW_EXPERIMENT = Variable.get(
    "MLFLOW_EXPERIMENT_NAME",
    default_var="Homeowner_Loss_Hist_Proj"
)

# ─── AIRFLOW / DAG CONFIG ────────────────────────────────────────────────────
DEFAULT_START_DATE   = "2025-01-01"
SCHEDULE_CRON        = "0 10 * * *"  # daily at 10 AM
AIRFLOW_DAG_BASE_CONF= {}

# MODEL APPROVAL CONFIGURATION
# Note: Set only one of these to True based on your desired workflow
AUTO_APPROVE_MODEL = Variable.get("AUTO_APPROVE_MODEL", default_var="True").lower() == "true"
REQUIRE_MODEL_APPROVAL = not AUTO_APPROVE_MODEL  # Inverse of auto approve

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
class ConfigManager:
    """
    Manages configuration for the ML Automation system.
    
    This class handles:
    - Environment variable management
    - Configuration file loading
    - Configuration validation
    - Default value handling
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self._config: Dict[str, Any] = {}
        self._config_path = config_path
        self._load_config()
        
    def _load_config(self) -> None:
        """
        Load configuration from file and environment variables.
        
        Raises:
            RuntimeError: If configuration loading fails
        """
        try:
            # Load from file if specified
            if self._config_path and os.path.exists(self._config_path):
                self._load_from_file()
                
            # Load from environment variables
            self._load_from_env()
            
            # Validate configuration
            self.validate_config()
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise RuntimeError("Configuration loading failed") from e
            
    def _load_from_file(self) -> None:
        """
        Load configuration from file.
        
        Supports JSON and YAML formats.
        """
        try:
            with open(self._config_path, 'r') as f:
                if self._config_path.endswith('.json'):
                    self._config.update(json.load(f))
                elif self._config_path.endswith(('.yaml', '.yml')):
                    self._config.update(yaml.safe_load(f))
                else:
                    logger.warning(f"Unsupported config file format: {self._config_path}")
                    
        except Exception as e:
            logger.error(f"Failed to load config file: {str(e)}")
            raise
            
    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        """
        try:
            # AWS Configuration
            self._config['aws'] = {
                'region': os.getenv('AWS_REGION', 'us-east-1'),
                'profile': os.getenv('AWS_PROFILE'),
                'bucket': os.getenv('AWS_BUCKET', 'ml-automation-data')
            }
            
            # Slack Configuration
            self._config['slack'] = {
                'token': os.getenv('SLACK_TOKEN'),
                'channel': os.getenv('SLACK_CHANNEL', '#ml-automation'),
                'username': os.getenv('SLACK_USERNAME', 'ML Automation')
            }
            
            # Airflow Configuration
            self._config['airflow'] = {
                'host': os.getenv('AIRFLOW_HOST', 'localhost'),
                'port': int(os.getenv('AIRFLOW_PORT', '8080')),
                'username': os.getenv('AIRFLOW_USERNAME', 'airflow'),
                'password': os.getenv('AIRFLOW_PASSWORD')
            }
            
            # Model Configuration
            self._config['model'] = {
                'path': os.getenv('MODEL_PATH', 'models'),
                'version': os.getenv('MODEL_VERSION', '1.0.0'),
                'threshold': float(os.getenv('MODEL_THRESHOLD', '0.5'))
            }
            
        except Exception as e:
            logger.error(f"Failed to load environment variables: {str(e)}")
            raise
            
    def validate_config(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Validate AWS configuration
            if not self._config['aws']['bucket']:
                raise ValueError("AWS bucket name is required")
                
            # Validate Slack configuration
            if not self._config['slack']['token']:
                logger.warning("Slack token not set")
                
            # Validate Airflow configuration
            if not self._config['airflow']['password']:
                logger.warning("Airflow password not set")
                
            # Validate Model configuration
            if not os.path.exists(self._config['model']['path']):
                os.makedirs(self._config['model']['path'], exist_ok=True)
                
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise ValueError("Invalid configuration") from e
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        try:
            return self._config.get(key, default)
        except Exception as e:
            logger.error(f"Failed to get config value for {key}: {str(e)}")
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        try:
            self._config[key] = value
        except Exception as e:
            logger.error(f"Failed to set config value for {key}: {str(e)}")
            raise
            
    def save(self) -> None:
        """
        Save configuration to file.
        
        Raises:
            RuntimeError: If configuration saving fails
        """
        if not self._config_path:
            logger.warning("No config file path specified")
            return
            
        try:
            with open(self._config_path, 'w') as f:
                if self._config_path.endswith('.json'):
                    json.dump(self._config, f, indent=2)
                elif self._config_path.endswith(('.yaml', '.yml')):
                    yaml.dump(self._config, f, default_flow_style=False)
                    
            logger.info(f"Configuration saved to {self._config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise RuntimeError("Configuration saving failed") from e

# Create a default instance
config = ConfigManager()
