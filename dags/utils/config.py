# utils/config.py

from dotenv import load_dotenv
load_dotenv()

import os
from airflow.models import Variable
from typing import Dict, Any

# ─── S3 CONFIG ────────────────────────────────────────────────────────────────
S3_BUCKET           = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET")
RAW_DATA_KEY        = "raw-data/ut_loss_history_1.csv"
REFERENCE_KEY       = "reference/reference_means.csv"
REFERENCE_KEY_PREFIX= "reference"
MODELS_FOLDER       = "models"
LOGS_FOLDER         = "logs"
ARCHIVE_FOLDER      = os.getenv("S3_ARCHIVE_FOLDER", "archive")

# ─── MLFLOW CONFIG ────────────────────────────────────────────────────────────
MLFLOW_URI          = (
    os.getenv("MLFLOW_TRACKING_URI")
    or Variable.get("MLFLOW_TRACKING_URI")
)
MLFLOW_EXPERIMENT   = Variable.get(
    "MLFLOW_EXPERIMENT_NAME",
    default_var="Homeowner_Loss_Hist_Proj"
)

# ─── AIRFLOW / DAG CONFIG ────────────────────────────────────────────────────
DEFAULT_START_DATE   = "2025-01-01"
SCHEDULE_CRON        = "0 10 * * *"  # daily at 10 AM
AIRFLOW_DAG_BASE_CONF= {}

# ─── SLACK CONFIG ────────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL    = (
    os.getenv("SLACK_WEBHOOK_URL")
    or Variable.get("SLACK_WEBHOOK_URL", default_var=None)
    or Variable.get("SLACK_WEBHOOK", default_var=None)
)
SLACK_CHANNEL_DEFAULT= (
    os.getenv("SLACK_CHANNEL_DEFAULT")
    or Variable.get("SLACK_CHANNEL_DEFAULT", default_var="#alerts")
)

# ─── HYPEROPT CONFIG ─────────────────────────────────────────────────────────
MAX_EVALS           = int(os.getenv("HYPEROPT_MAX_EVALS", 20))

# ─── UI COMPONENTS CONFIG ────────────────────────────────────────────────────
UI_COMPONENTS = {
    "sidebar": {
        "items": [
            "Home",
            "Drift Monitor",
            "Model Metrics",
            "Overrides",
            "Artifacts",
            "Data Quality",
            "Incidents",
            "Settings"
        ],
        "collapsible": True,
        "tooltips": True
    },
    "home_screen": {
        "top_bar": {
            "date_selector": True,
            "environment_toggle": ["Prod", "Dev"],
            "light_dark_mode_switch": True
        },
        "pipeline_health_card": {
            "last_run_timestamp": True,
            "status_icon": ["✅", "⚠️", "❌"],
            "runtime": True
        },
        "error_fix_chat_widget": {
            "enabled": True,
            "position": "bottom_right"
        },
        "quick_action_buttons": [
            "Trigger Retrain",
            "Run Self-Heal",
            "Generate Fix Proposal",
            "Open Code Console"
        ]
    },
    "drift_monitor": {
        "feature_drift_tiles": {
            "grid_format": True,
            "current_drift_percentages": True,
            "thresholds": True,
            "neon_accent_on_warning": True
        },
        "sparklines": True,
        "details_panel": {
            "full_line_chart": True,
            "context_info": True,
            "buttons": ["Self-Heal", "Propose Fix"]
        }
    },
    "model_metrics": {
        "model_selector": ["model1", "model2", "model3", "model4", "model5"],
        "metric_cards": ["RMSE", "MSE", "MAE", "R²"],
        "charts": [
            "RMSE over last N runs",
            "Actual vs. Predicted scatter",
            "SHAP summary"
        ],
        "download_icons": True
    },
    "overrides": {
        "pending_proposals_list": {
            "table_columns": ["problem_summary", "confidence", "timestamp"]
        },
        "proposal_detail_panel": {
            "code_snippets": True,
            "drift_context": True,
            "logs": True
        },
        "actions": ["Approve", "Reject", "Edit then Approve"]
    },
    "artifacts": {
        "s3_tree_view": "visuals/",
        "image_previews": ["SHAP", "AVS"],
        "actions": ["Copy S3 Path", "Download"]
    },
    "data_quality": {
        "schema_report": {
            "table_columns": ["expected", "actual", "type_mismatches"]
        },
        "null_duplicate_stats": True,
        "generate_fix_plan_button": True
    },
    "incidents": {
        "open_tickets_list": {
            "columns": ["status", "severity", "assignee"]
        },
        "create_ticket_form": {
            "fields": ["issue_summary", "severity_dropdown", "assignee_field"]
        },
        "sync_button": True
    },
    "settings": {
        "configuration_options": [
            "drift_thresholds",
            "HyperOpt_settings",
            "MLflow_experiment_name"
        ],
        "channel_mapping": {
            "alert_types": "Slack_channels"
        },
        "user_roles": ["admin", "viewer"]
    },
    "code_interpreter_console": {
        "floating_panel": True,
        "repl_editor": {
            "code_cell": True,
            "run_button": True
        },
        "file_browser": "/tmp",
        "output_panel": {
            "stdout": True,
            "charts": True
        },
        "history_list": True
    },
    "visual_style_ux": {
        "typography": {
            "headings": ["#", "##", "###"],
            "body_text": "sans-serif"
        },
        "colors": {
            "dark_mode": {
                "base": "#131313",
                "accent": "#4ECDC4"
            },
            "light_mode": {
                "base": "#FFFFFF",
                "accent": "#1A535C"
            }
        },
        "components": {
            "rounded_cards": "2xl",
            "soft_shadows": True,
            "spacing": "p-4"
        },
        "feedback": {
            "subtle_animations": True,
            "toast_notifications": True
        }
    },
    "integration_points": {
        "frontend": "React + Shadcn/UI",
        "charts": "Recharts",
        "api": "OpenAI Assistant endpoints",
        "realtime": "WebSockets"
    },
    "security_permissions": {
        "rbac": {
            "roles": ["admin", "viewer"]
        },
        "sandbox": {
            "restricted_directory": "/tmp"
        },
        "audit_trail": True
    }
}

class Config:
    """Centralized configuration management."""
    
    # S3 Configuration
    S3_BUCKET = os.getenv("S3_BUCKET", Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket"))
    S3_DATA_FOLDER = "raw-data"
    S3_ARCHIVE_FOLDER = "archive"
    S3_REFERENCE_KEY_PREFIX = "reference"
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", Variable.get("MLFLOW_TRACKING_URI", default_var="http://localhost:5000"))
    MLFLOW_EXPERIMENT_NAME = "homeowner_loss_history"
    
    # Model Configuration
    MODEL_IDS = ["model1", "model2", "model3", "model4", "model5"]
    MODEL_METRICS = ["rmse", "mse", "mae", "r2"]
    MODEL_THRESHOLDS = {
        "rmse": float(Variable.get("RMSE_THRESHOLD", default_var="0.1")),
        "drift": float(Variable.get("DRIFT_THRESHOLD", default_var="0.1")),
        "correlation": float(Variable.get("CORRELATION_THRESHOLD", default_var="0.7"))
    }
    
    # Data Quality Configuration
    DATA_QUALITY_CONFIG = {
        "missing_threshold": float(Variable.get("MISSING_THRESHOLD", default_var="0.05")),
        "outlier_threshold": float(Variable.get("OUTLIER_THRESHOLD", default_var="3.0")),
        "drift_threshold": float(Variable.get("DRIFT_THRESHOLD", default_var="0.1")),
        "correlation_threshold": float(Variable.get("CORRELATION_THRESHOLD", default_var="0.7"))
    }
    
    # WebSocket Configuration
    WS_HOST = os.getenv("WS_HOST", "localhost")
    WS_PORT = int(os.getenv("WS_PORT", "8765"))
    WS_PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "30"))
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "3000"))
    API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))
    
    # Security Configuration
    SECURITY_CONFIG = {
        "jwt_secret": os.getenv("JWT_SECRET", Variable.get("JWT_SECRET", default_var="your-secret-key")),
        "jwt_expiry": int(os.getenv("JWT_EXPIRY", "3600")),
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", "3600")),
        "rate_limit_max_requests": int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "1000"))
    }
    
    # Slack Configuration
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", Variable.get("SLACK_WEBHOOK_URL"))
    SLACK_CHANNELS = {
        "alerts": "#alerts",
        "logs": "#agent_logs",
        "metrics": "#metrics"
    }
    
    # Airflow Configuration
    AIRFLOW_CONFIG = {
        "api_url": os.getenv("AIRFLOW_API_URL", Variable.get("AIRFLOW_API_URL", default_var="http://localhost:8080")),
        "username": os.getenv("AIRFLOW_USERNAME", Variable.get("AIRFLOW_USERNAME", default_var="admin")),
        "password": os.getenv("AIRFLOW_PASSWORD", Variable.get("AIRFLOW_PASSWORD", default_var="admin")),
        "dag_folder": os.getenv("AIRFLOW_DAG_FOLDER", "/opt/airflow/dags")
    }
    
    # Feature Configuration
    FEATURE_CONFIG = {
        "raw_prefixes": ["num_loss_3yr_", "num_loss_yrs45_", "num_loss_free_yrs_"],
        "decay_prefixes": {
            "model2": ["lhdwc_5y_1d_"],  # equal
            "model3": ["lhdwc_5y_2d_"],  # linear decay
            "model4": ["lhdwc_5y_3d_"],  # fast decay
            "model5": ["lhdwc_5y_4d_"]   # slow decay
        }
    }
    
    @classmethod
    def get_model_config(cls, model_id: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return {
            "id": model_id,
            "metrics": cls.MODEL_METRICS,
            "thresholds": cls.MODEL_THRESHOLDS,
            "features": cls.FEATURE_CONFIG["decay_prefixes"].get(model_id, [])
        }
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration values."""
        required_vars = [
            "S3_BUCKET",
            "MLFLOW_TRACKING_URI",
            "SLACK_WEBHOOK_URL"
        ]
        
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(f"Missing required configuration variables: {', '.join(missing_vars)}")
        
        # Validate numeric thresholds
        for threshold_name, threshold_value in cls.MODEL_THRESHOLDS.items():
            if not isinstance(threshold_value, (int, float)) or threshold_value <= 0:
                raise ValueError(f"Invalid threshold value for {threshold_name}: {threshold_value}")
        
        # Validate data quality config
        for key, value in cls.DATA_QUALITY_CONFIG.items():
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"Invalid data quality config value for {key}: {value}")

# Validate configuration on import
Config.validate_config()
