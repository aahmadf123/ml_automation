#!/usr/bin/env python3
"""
model_retuning.py - Model Retuning Triggers
--------------------------------------------
This module provides functions for triggering model retuning
based on both automated metrics and human feedback.

Features:
- Re-tuning triggers based on human feedback
- Performance-based auto triggers
- Integration with MLflow for experiment tracking
- Hyperparameter optimization restart with tuned parameters
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from airflow.models import Variable, XCom
from airflow.exceptions import AirflowSkipException
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Import utility modules
from utils.slack import post as slack_post

# Set up logging
logger = logging.getLogger(__name__)

# Constants
RETUNING_REQUEST_PREFIX = "RETUNING_REQUEST_"
RETUNING_PARAM_PREFIX = "RETUNING_PARAM_"
DEFAULT_IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement threshold

class ModelRetuning:
    """
    Main class for handling model retuning requests and triggers
    """
    
    @staticmethod
    def get_retuning_key(model_id: str) -> str:
        """Generate a unique key for retuning requests"""
        return f"{RETUNING_REQUEST_PREFIX}{model_id}"
    
    @staticmethod
    def get_param_key(model_id: str, param_name: str) -> str:
        """Generate a unique key for retuning parameter overrides"""
        return f"{RETUNING_PARAM_PREFIX}{model_id}_{param_name}"
    
    @staticmethod
    def request_retuning(
        model_id: str,
        requestor: str,
        reason: str,
        param_overrides: Optional[Dict[str, Any]] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Request model retuning.
        
        Args:
            model_id: Identifier for the model
            requestor: Who requested the retuning
            reason: Reason for retuning request
            param_overrides: Optional parameter overrides for retuning
            priority: Priority level (high, normal, low)
            
        Returns:
            Dict with request details
        """
        timestamp = datetime.now().isoformat()
        
        # Create retuning request record
        request_data = {
            "model_id": model_id,
            "requestor": requestor,
            "reason": reason,
            "timestamp": timestamp,
            "status": "pending",
            "priority": priority,
            "param_overrides": param_overrides or {}
        }
        
        # Store in Airflow Variable
        request_key = ModelRetuning.get_retuning_key(model_id)
        Variable.set(request_key, json.dumps(request_data))
        
        # If parameter overrides provided, store them
        if param_overrides:
            for param_name, param_value in param_overrides.items():
                param_key = ModelRetuning.get_param_key(model_id, param_name)
                Variable.set(param_key, json.dumps({"value": param_value, "timestamp": timestamp}))
        
        # Send notification to slack
        slack_message = f"""
        :arrows_counterclockwise: *Model Retuning Request* :arrows_counterclockwise:
        
        *Model*: {model_id}
        *Requestor*: {requestor}
        *Reason*: {reason}
        *Priority*: {priority}
        *Parameter Overrides*: {len(param_overrides or {})} parameters
        """
        
        try:
            slack_post(slack_message, channel="#model-tuning")
            logger.info(f"Retuning request for {model_id} sent to Slack")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return request_data
    
    @staticmethod
    def check_for_retuning_request(model_id: str) -> Dict[str, Any]:
        """
        Check if a retuning request exists for the model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Dict with request details if found, empty dict otherwise
        """
        request_key = ModelRetuning.get_retuning_key(model_id)
        
        try:
            request_json = Variable.get(request_key, default_var="{}")
            request_data = json.loads(request_json)
            
            # If we have an active request, return it
            if request_data and request_data.get('status') == 'pending':
                return request_data
                
            return {}
        except Exception as e:
            logger.warning(f"Error checking retuning request: {str(e)}")
            return {}
    
    @staticmethod
    def get_parameter_overrides(model_id: str) -> Dict[str, Any]:
        """
        Get parameter overrides for a model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Dict with parameter overrides
        """
        # Try to get from retuning request first
        request_key = ModelRetuning.get_retuning_key(model_id)
        param_overrides = {}
        
        try:
            request_json = Variable.get(request_key, default_var="{}")
            request_data = json.loads(request_json)
            
            if request_data and 'param_overrides' in request_data:
                param_overrides = request_data.get('param_overrides', {})
        except Exception as e:
            logger.warning(f"Error getting parameter overrides from request: {str(e)}")
        
        # Add any individually set parameters
        try:
            # Get all variables that might contain parameter overrides
            all_variables = {k: v for k, v in Variable.get_all().items() 
                            if k.startswith(f"{RETUNING_PARAM_PREFIX}{model_id}_")}
            
            for var_key, var_value in all_variables.items():
                try:
                    # Extract parameter name from key
                    param_name = var_key.replace(f"{RETUNING_PARAM_PREFIX}{model_id}_", "")
                    param_data = json.loads(var_value)
                    param_overrides[param_name] = param_data.get('value')
                except:
                    pass
        except Exception as e:
            logger.warning(f"Error getting individual parameter overrides: {str(e)}")
        
        return param_overrides
    
    @staticmethod
    def mark_retuning_complete(
        model_id: str,
        status: str = "completed",
        message: str = "",
        metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Mark a retuning request as complete.
        
        Args:
            model_id: Identifier for the model
            status: Status of retuning (completed, failed)
            message: Optional status message
            metrics: Optional performance metrics
            
        Returns:
            Dict with updated request details
        """
        request_key = ModelRetuning.get_retuning_key(model_id)
        
        try:
            request_json = Variable.get(request_key, default_var="{}")
            request_data = json.loads(request_json)
            
            if not request_data:
                logger.warning(f"No retuning request found for {model_id}")
                return {}
            
            # Update the request
            request_data['status'] = status
            request_data['completion_time'] = datetime.now().isoformat()
            if message:
                request_data['message'] = message
            if metrics:
                request_data['metrics'] = metrics
            
            # Save the updated request
            Variable.set(request_key, json.dumps(request_data))
            
            # Send notification to slack
            emoji = ":white_check_mark:" if status == "completed" else ":x:"
            slack_message = f"""
            {emoji} *Model Retuning {status.capitalize()}* {emoji}
            
            *Model*: {model_id}
            *Requestor*: {request_data.get('requestor', 'Unknown')}
            *Status*: {status.capitalize()}
            {f"*Message*: {message}" if message else ""}
            """
            
            if metrics:
                metrics_text = "\n*Metrics*:\n" + "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
                slack_message += metrics_text
            
            try:
                slack_post(slack_message, channel="#model-tuning")
                logger.info(f"Retuning completion for {model_id} sent to Slack")
            except Exception as e:
                logger.warning(f"Failed to send Slack notification: {str(e)}")
            
            return request_data
        except Exception as e:
            logger.warning(f"Error marking retuning complete: {str(e)}")
            return {}

def check_needs_retuning(
    model_id: str,
    current_metrics: Dict[str, float],
    metric_name: str = "rmse",
    improvement_threshold: Optional[float] = None,
    **context
) -> bool:
    """
    Check if a model needs retuning based on performance or human request.
    
    Args:
        model_id: Identifier for the model to check
        current_metrics: Current model metrics
        metric_name: Primary metric name to use for comparison
        improvement_threshold: Threshold for improvement to trigger retuning
        context: Airflow context (for XCom)
        
    Returns:
        bool: True if retuning is needed
    """
    logger.info(f"Checking if model {model_id} needs retuning")
    
    # First, check for manual retuning request
    retuning = ModelRetuning()
    request = retuning.check_for_retuning_request(model_id)
    
    if request:
        logger.info(f"Manual retuning request found for {model_id}: {request.get('reason', 'No reason provided')}")
        return True
    
    # If no manual request, check performance
    if not improvement_threshold:
        improvement_threshold = float(Variable.get("RMSE_IMPROVEMENT_THRESHOLD", 
                                                 default_var=str(DEFAULT_IMPROVEMENT_THRESHOLD)))
    
    try:
        # Get current metric
        current_value = current_metrics.get(metric_name)
        if current_value is None:
            logger.warning(f"Metric {metric_name} not found in current metrics")
            return False
        
        # Connect to MLflow and get production model
        client = MlflowClient()
        try:
            production_versions = client.get_latest_versions(model_id, stages=["Production"])
            if not production_versions:
                logger.info(f"No production model found for {model_id}. Retuning not needed.")
                return False
                
            prod_version = production_versions[0]
            run = client.get_run(prod_version.run_id)
            prod_value = run.data.metrics.get(metric_name)
            
            if prod_value is None:
                logger.warning(f"Metric {metric_name} not found in production model")
                return False
            
            # For metrics like RMSE where lower is better
            is_lower_better = metric_name in ['rmse', 'mae', 'mse', 'loss', 'error']
            
            if is_lower_better:
                potential_improvement = (prod_value - current_value) / prod_value
            else:
                potential_improvement = (current_value - prod_value) / prod_value
            
            # If we can improve by threshold%, suggest retuning
            if potential_improvement >= improvement_threshold:
                logger.info(f"Potential improvement of {potential_improvement:.2%} detected, suggesting retuning")
                return True
            else:
                logger.info(f"Current model performance similar to production (improvement: {potential_improvement:.2%}), retuning not needed")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking production model: {str(e)}")
            return False
            
    except Exception as e:
        logger.warning(f"Error in retuning check: {str(e)}")
        return False

def get_tuning_parameters(model_id: str, base_params: Dict[str, Any], **context) -> Dict[str, Any]:
    """
    Get tuning parameters combining base parameters with any overrides.
    
    Args:
        model_id: Identifier for the model
        base_params: Base parameters to use
        context: Airflow context
        
    Returns:
        Dict with final parameters
    """
    # Start with base parameters
    final_params = base_params.copy()
    
    # Check for parameter overrides
    retuning = ModelRetuning()
    overrides = retuning.get_parameter_overrides(model_id)
    
    if overrides:
        logger.info(f"Applying {len(overrides)} parameter overrides for {model_id}")
        for param_name, param_value in overrides.items():
            if param_name in final_params:
                logger.info(f"Overriding {param_name}: {final_params[param_name]} -> {param_value}")
                final_params[param_name] = param_value
            else:
                logger.warning(f"Override for unknown parameter {param_name}")
    
    return final_params

def trigger_retuning(
    dag_id: str,
    run_id: str,
    model_id: str,
    conf: Optional[Dict[str, Any]] = None,
    **context
) -> Dict[str, Any]:
    """
    Trigger retuning DAG for a model.
    
    Args:
        dag_id: Airflow DAG ID to trigger
        run_id: Current run ID
        model_id: Model ID to retune
        conf: Optional configuration to pass to the DAG
        context: Airflow context
        
    Returns:
        Dict with trigger status
    """
    from airflow.api.common.trigger_dag import trigger_dag as trigger
    
    logger.info(f"Triggering retuning for model {model_id}")
    
    # Prepare configuration
    dag_conf = conf or {}
    dag_conf.update({
        "model_id": model_id,
        "parent_run_id": run_id,
        "is_retuning": True,
        "timestamp": datetime.now().isoformat()
    })
    
    # Add any parameter overrides
    retuning = ModelRetuning()
    param_overrides = retuning.get_parameter_overrides(model_id)
    if param_overrides:
        dag_conf["param_overrides"] = param_overrides
    
    # Trigger the DAG
    try:
        triggered_run_id = trigger(dag_id=dag_id, run_id=None, conf=dag_conf, replace_microseconds=True)
        
        result = {
            "status": "triggered",
            "triggered_dag_id": dag_id,
            "triggered_run_id": triggered_run_id,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Successfully triggered retuning DAG {dag_id} for model {model_id}")
        
        # Send notification
        try:
            slack_message = f"""
            :rocket: *Model Retuning Triggered* :rocket:
            
            *Model*: {model_id}
            *DAG*: {dag_id}
            *Run ID*: {triggered_run_id}
            *Parameters*: {len(param_overrides)} overridden
            """
            slack_post(slack_message, channel="#model-tuning")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return result
    except Exception as e:
        logger.error(f"Failed to trigger retuning DAG: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }

# Airflow task for retuning decision
def decide_retuning(**context) -> Dict[str, Any]:
    """
    Airflow task to decide if retuning is needed based on metrics and requests.
    
    Args:
        context: Airflow task context
        
    Returns:
        Dict with decision details
    """
    # Get DAG run ID from context
    dag_run = context.get('dag_run')
    if not dag_run:
        raise ValueError("No DAG run information in context")
    
    run_id = dag_run.run_id
    
    # Get training and evaluation results
    ti = context.get('ti')
    training_results = ti.xcom_pull(task_ids='train_models', key='training_results')
    
    if not training_results or not isinstance(training_results, dict):
        logger.warning("No valid training results found")
        return {"status": "skipped", "reason": "No valid training results found"}
    
    # Check each model for retuning needs
    retuning_candidates = []
    
    for model_id, result in training_results.items():
        if result.get('status') != 'completed':
            continue
            
        metrics = result.get('metrics', {})
        if not metrics:
            continue
            
        # Check if this model needs retuning
        needs_retuning = check_needs_retuning(
            model_id=model_id,
            current_metrics=metrics,
            context=context
        )
        
        if needs_retuning:
            retuning_candidates.append({
                "model_id": model_id,
                "metrics": metrics,
                "reason": "Performance improvement or manual request"
            })
    
    # If we have candidates, prepare retuning
    if retuning_candidates:
        logger.info(f"Found {len(retuning_candidates)} models that need retuning")
        
        # For now, just pick the first candidate
        candidate = retuning_candidates[0]
        model_id = candidate["model_id"]
        
        # Trigger retuning DAG
        result = trigger_retuning(
            dag_id="model_hyperparameter_tuning",  # This should match your tuning DAG ID
            run_id=run_id,
            model_id=model_id,
            context=context
        )
        
        # Store the retuning decision in XCom
        context['ti'].xcom_push(key='retuning_decision', value={
            "needs_retuning": True,
            "candidates": retuning_candidates,
            "selected": model_id,
            "trigger_result": result
        })
        
        return {
            "status": "retuning",
            "model_id": model_id,
            "trigger_result": result
        }
    else:
        logger.info("No models need retuning")
        
        # Store the retuning decision in XCom
        context['ti'].xcom_push(key='retuning_decision', value={
            "needs_retuning": False,
            "reason": "No candidates for retuning"
        })
        
        return {
            "status": "skipped",
            "reason": "No models need retuning"
        }

def raise_question_if_better(new_model_metrics: Dict[str, float], production_model_metrics: Dict[str, float]) -> bool:
    """
    Raise a question if the new model performs better than the production model.
    
    Args:
        new_model_metrics: Metrics of the new model
        production_model_metrics: Metrics of the production model
        
    Returns:
        bool: True if the new model performs better, False otherwise
    """
    # Define the primary metric for comparison
    primary_metric = "rmse"
    
    new_model_performance = new_model_metrics.get(primary_metric)
    production_model_performance = production_model_metrics.get(primary_metric)
    
    if new_model_performance is None or production_model_performance is None:
        logger.warning("Primary metric not found in one of the models")
        return False
    
    # For RMSE, lower is better
    if new_model_performance < production_model_performance:
        # Raise a question to the human-in-the-loop
        question = f"New model with RMSE {new_model_performance} performs better than production model with RMSE {production_model_performance}. Should we make the new model the production model?"
        logger.info(question)
        
        # Send the question to Slack
        try:
            slack_post(question, channel="#model-tuning")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return True
    
    return False

def handle_human_in_the_loop(new_model_metrics: Dict[str, float], production_model_metrics: Dict[str, float]) -> None:
    """
    Handle the human-in-the-loop part by raising a question if the new model performs better.
    
    Args:
        new_model_metrics: Metrics of the new model
        production_model_metrics: Metrics of the production model
    """
    if raise_question_if_better(new_model_metrics, production_model_metrics):
        logger.info("Question raised to human-in-the-loop for model promotion decision")
    else:
        logger.info("New model does not perform better than production model, no question raised")

def decide_retuning_with_hitl(**context) -> Dict[str, Any]:
    """
    Airflow task to decide if retuning is needed based on metrics, requests, and human-in-the-loop.
    
    Args:
        context: Airflow task context
        
    Returns:
        Dict with decision details
    """
    # Get DAG run ID from context
    dag_run = context.get('dag_run')
    if not dag_run:
        raise ValueError("No DAG run information in context")
    
    run_id = dag_run.run_id
    
    # Get training and evaluation results
    ti = context.get('ti')
    training_results = ti.xcom_pull(task_ids='train_models', key='training_results')
    
    if not training_results or not isinstance(training_results, dict):
        logger.warning("No valid training results found")
        return {"status": "skipped", "reason": "No valid training results found"}
    
    # Check each model for retuning needs
    retuning_candidates = []
    
    for model_id, result in training_results.items():
        if result.get('status') != 'completed':
            continue
            
        metrics = result.get('metrics', {})
        if not metrics:
            continue
            
        # Check if this model needs retuning
        needs_retuning = check_needs_retuning(
            model_id=model_id,
            current_metrics=metrics,
            context=context
        )
        
        if needs_retuning:
            retuning_candidates.append({
                "model_id": model_id,
                "metrics": metrics,
                "reason": "Performance improvement or manual request"
            })
    
    # If we have candidates, prepare retuning
    if retuning_candidates:
        logger.info(f"Found {len(retuning_candidates)} models that need retuning")
        
        # For now, just pick the first candidate
        candidate = retuning_candidates[0]
        model_id = candidate["model_id"]
        
        # Trigger retuning DAG
        result = trigger_retuning(
            dag_id="model_hyperparameter_tuning",  # This should match your tuning DAG ID
            run_id=run_id,
            model_id=model_id,
            context=context
        )
        
        # Store the retuning decision in XCom
        context['ti'].xcom_push(key='retuning_decision', value={
            "needs_retuning": True,
            "candidates": retuning_candidates,
            "selected": model_id,
            "trigger_result": result
        })
        
        # Handle human-in-the-loop part
        production_model_metrics = {}  # Retrieve production model metrics from MLflow or other source
        handle_human_in_the_loop(candidate["metrics"], production_model_metrics)
        
        return {
            "status": "retuning",
            "model_id": model_id,
            "trigger_result": result
        }
    else:
        logger.info("No models need retuning")
        
        # Store the retuning decision in XCom
        context['ti'].xcom_push(key='retuning_decision', value={
            "needs_retuning": False,
            "reason": "No candidates for retuning"
        })
        
        return {
            "status": "skipped",
            "reason": "No models need retuning"
        }
