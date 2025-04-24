import os
import logging
from typing import Dict, Any, Optional
from clearml import Task, Dataset, Model
from utils.config import get_ssm_parameter, AWS_REGION

# Set up logging
logger = logging.getLogger(__name__)

# ClearML credentials from SSM or environment variables
CLEARML_CONFIG = {
    "api_server": get_ssm_parameter('CLEARML_API_SERVER', os.getenv('CLEARML_API_HOST', 'https://app.clear.ml')),
    "web_server": get_ssm_parameter('CLEARML_WEB_SERVER', os.getenv('CLEARML_WEB_HOST', 'https://app.clear.ml')),
    "files_server": get_ssm_parameter('CLEARML_FILES_SERVER', os.getenv('CLEARML_FILES_HOST', 'https://files.clear.ml')),
    "key": get_ssm_parameter('CLEARML_KEY', os.getenv('CLEARML_API_ACCESS_KEY')),
    "secret": get_ssm_parameter('CLEARML_SECRET', os.getenv('CLEARML_API_SECRET_KEY')),
    "project_name": get_ssm_parameter('CLEARML_PROJECT', os.getenv('CLEARML_PROJECT', 'HomeownerLossHistoryProject'))
}

def init_clearml(task_name: str, task_type: str = Task.TaskTypes.training) -> Task:
    """
    Initialize ClearML task for experiment tracking
    
    Args:
        task_name: Name of the task
        task_type: Type of task (training, testing, inference, etc)
        
    Returns:
        ClearML Task object
    """
    try:
        task = Task.init(
            project_name=CLEARML_CONFIG["project_name"],
            task_name=task_name,
            task_type=task_type,
            reuse_last_task_id=False
        )
        
        # Connect to AWS
        task.set_credentials(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_region=AWS_REGION
        )
        
        logger.info(f"Initialized ClearML task: {task.id} - {task_name}")
        return task
    except Exception as e:
        logger.error(f"Failed to initialize ClearML task: {str(e)}")
        # Continue execution without ClearML
        return None

def log_dataset_to_clearml(dataset_name: str, dataset_path: str, dataset_tags: Optional[list] = None) -> Optional[str]:
    """
    Log a dataset to ClearML
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset
        dataset_tags: Tags for the dataset
        
    Returns:
        Dataset ID if successful, None otherwise
    """
    try:
        dataset = Dataset.create(
            dataset_name=dataset_name, 
            dataset_project=CLEARML_CONFIG["project_name"],
            dataset_tags=dataset_tags or []
        )
        
        # Add files
        dataset.add_files(dataset_path)
        
        # Upload
        dataset.upload()
        
        # Finalize
        dataset.finalize()
        
        logger.info(f"Logged dataset to ClearML: {dataset.id} - {dataset_name}")
        return dataset.id
    except Exception as e:
        logger.error(f"Failed to log dataset to ClearML: {str(e)}")
        return None

def log_model_to_clearml(task: Optional[Task], model, model_name: str, framework: str = "xgboost") -> Optional[str]:
    """
    Log a model to ClearML
    
    Args:
        task: Associated ClearML task
        model: Model object to log
        model_name: Name of the model
        framework: Model framework
        
    Returns:
        Model ID if successful, None otherwise
    """
    try:
        if task is None:
            logger.warning("No ClearML task provided, creating a new task for model logging")
            task = init_clearml(f"{model_name}_model_logging", Task.TaskTypes.inference)
            
        # In ClearML 1.14.2, we need to use the OutputModel API for XGBoost models
        output_model = task.register_output_model(
            model=model,
            name=model_name,
            framework=framework
        )
        
        model_id = output_model.id if hasattr(output_model, 'id') else None
        logger.info(f"Logged model to ClearML: {model_id} - {model_name}")
        return model_id
    except Exception as e:
        logger.error(f"Failed to log model to ClearML: {str(e)}")
        return None 