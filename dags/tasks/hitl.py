#!/usr/bin/env python3
"""
hitl.py - Human-in-the-Loop (HITL) functionality
-------------------------------------------------
This module provides functions for human validation and approval 
steps in the ML pipeline, enabling manual oversight of critical
ML processes.

Features:
- Data validation waiting task 
- Model approval interfaces
- Override tracking and management
- Slack notifications for human review requests
- Status checking and timeout handling
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

# Import utility modules
from utils.slack import post as slack_post

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_APPROVAL_TIMEOUT_HOURS = 24
DEFAULT_AUTO_APPROVE_TIME_HOURS = 48  # Auto-approve after this time if no action
HITL_STATUS_PREFIX = "HITL_STATUS_"
HITL_ACTION_PREFIX = "HITL_ACTION_"
HITL_COMMENT_PREFIX = "HITL_COMMENT_"

# HITL Status values
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_TIMEOUT = "timeout"
STATUS_AUTO_APPROVED = "auto_approved"

class HumanInTheLoop:
    """
    Main class for managing Human-in-the-Loop processes.
    """
    
    @staticmethod
    def get_status_key(task_id: str, run_id: str) -> str:
        """Generate a unique key for storing HITL status."""
        return f"{HITL_STATUS_PREFIX}{task_id}_{run_id}"
        
    @staticmethod
    def get_action_key(task_id: str, run_id: str) -> str:
        """Generate a unique key for storing HITL action."""
        return f"{HITL_ACTION_PREFIX}{task_id}_{run_id}"
        
    @staticmethod
    def get_comment_key(task_id: str, run_id: str) -> str:
        """Generate a unique key for storing HITL comment."""
        return f"{HITL_COMMENT_PREFIX}{task_id}_{run_id}"
    
    @staticmethod
    def request_approval(
        task_id: str,
        run_id: str,
        context: Dict[str, Any],
        artifacts: Dict[str, Any],
        message: str,
        approval_timeout_hours: int = DEFAULT_APPROVAL_TIMEOUT_HOURS,
    ) -> Dict[str, Any]:
        """
        Request human approval for a task.
        
        Args:
            task_id: Identifier for the task
            run_id: Airflow run ID
            context: Airflow context dictionary
            artifacts: Dictionary of relevant artifacts for human review
            message: Message explaining what needs approval
            approval_timeout_hours: Hours before approval times out
            
        Returns:
            Dict with request status and details
        """
        timestamp = datetime.now().isoformat()
        expiration = (datetime.now() + timedelta(hours=approval_timeout_hours)).isoformat()
        
        # Create status record
        status_data = {
            "task_id": task_id,
            "run_id": run_id,
            "status": STATUS_PENDING,
            "timestamp": timestamp,
            "expiration": expiration,
            "message": message,
            "artifacts": artifacts
        }
        
        # Store in Airflow Variable
        status_key = HumanInTheLoop.get_status_key(task_id, run_id)
        Variable.set(status_key, json.dumps(status_data))
        
        # Send notification to slack
        dashboard_url = Variable.get("DASHBOARD_URL", default_var="")
        approval_url = f"{dashboard_url}/approvals?task={task_id}&run={run_id}" if dashboard_url else "Check the ML Dashboard"
        
        slack_message = f"""
        :warning: *Human Review Required* :warning:
        
        *Task*: {task_id}
        *Run ID*: {run_id}
        *Message*: {message}
        *Expires*: In {approval_timeout_hours} hours
        
        To approve or reject, go to: {approval_url}
        """
        
        try:
            slack_post(slack_message, channel="#ml-approvals")
            logger.info(f"Approval request sent to Slack for {task_id} in run {run_id}")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return {
            "status": "pending",
            "task_id": task_id,
            "run_id": run_id,
            "timestamp": timestamp,
            "expiration": expiration
        }
    
    @staticmethod
    def check_approval_status(
        task_id: str,
        run_id: str,
        auto_approve_on_timeout: bool = False
    ) -> Dict[str, Any]:
        """
        Check the status of an approval request.
        
        Args:
            task_id: Identifier for the task
            run_id: Airflow run ID
            auto_approve_on_timeout: Whether to auto-approve if timed out
            
        Returns:
            Dict with approval status and details
        """
        status_key = HumanInTheLoop.get_status_key(task_id, run_id)
        action_key = HumanInTheLoop.get_action_key(task_id, run_id)
        comment_key = HumanInTheLoop.get_comment_key(task_id, run_id)
        
        try:
            # Check for existing status
            status_json = Variable.get(status_key, default_var="{}")
            status_data = json.loads(status_json)
            
            # Check for recorded action
            action = None
            try:
                action = Variable.get(action_key)
            except:
                pass
                
            # Check for comment
            comment = None
            try:
                comment = Variable.get(comment_key)
            except:
                pass
            
            # If we have an action, use it
            if action:
                status_data["status"] = action
                status_data["action_timestamp"] = datetime.now().isoformat()
                if comment:
                    status_data["comment"] = comment
                
                # Update the status
                Variable.set(status_key, json.dumps(status_data))
                
                # Clear action and comment to avoid double-processing
                try:
                    Variable.delete(action_key)
                    if comment:
                        Variable.delete(comment_key)
                except:
                    pass
                    
                return status_data
            
            # If still pending, check for timeout
            if status_data.get("status") == STATUS_PENDING:
                expiration = datetime.fromisoformat(status_data.get("expiration", "2099-12-31T00:00:00"))
                
                if datetime.now() > expiration:
                    # Check for auto-approve after longer timeout
                    auto_approve_expiration = datetime.fromisoformat(status_data.get("timestamp", "2000-01-01T00:00:00")) + timedelta(hours=DEFAULT_AUTO_APPROVE_TIME_HOURS)
                    
                    if auto_approve_on_timeout and datetime.now() > auto_approve_expiration:
                        status_data["status"] = STATUS_AUTO_APPROVED
                        status_data["action_timestamp"] = datetime.now().isoformat()
                        status_data["comment"] = "Automatically approved due to timeout"
                        
                        # Send notification
                        try:
                            slack_post(
                                f"⏰ Task `{task_id}` from run `{run_id}` was auto-approved after timeout.",
                                channel="#ml-approvals"
                            )
                        except:
                            pass
                    else:
                        status_data["status"] = STATUS_TIMEOUT
                        status_data["action_timestamp"] = datetime.now().isoformat()
                    
                    # Update the status
                    Variable.set(status_key, json.dumps(status_data))
            
            return status_data
            
        except Exception as e:
            logger.warning(f"Error checking approval status: {str(e)}")
            return {
                "status": "error",
                "task_id": task_id,
                "run_id": run_id,
                "error": str(e)
            }
    
    @staticmethod
    def record_approval(
        task_id: str,
        run_id: str,
        approved: bool,
        approver: str,
        comment: str = ""
    ) -> Dict[str, Any]:
        """
        Record an approval decision.
        
        Args:
            task_id: Identifier for the task
            run_id: Airflow run ID
            approved: Whether the task was approved
            approver: Who approved/rejected
            comment: Optional comment
            
        Returns:
            Dict with approval status and details
        """
        status_key = HumanInTheLoop.get_status_key(task_id, run_id)
        action_key = HumanInTheLoop.get_action_key(task_id, run_id)
        comment_key = HumanInTheLoop.get_comment_key(task_id, run_id)
        
        # Set the action
        action = STATUS_APPROVED if approved else STATUS_REJECTED
        Variable.set(action_key, action)
        
        # Set the comment if provided
        if comment:
            Variable.set(comment_key, comment)
        
        # Update status record
        try:
            status_json = Variable.get(status_key, default_var="{}")
            status_data = json.loads(status_json)
            
            status_data["status"] = action
            status_data["approver"] = approver
            status_data["action_timestamp"] = datetime.now().isoformat()
            if comment:
                status_data["comment"] = comment
                
            Variable.set(status_key, json.dumps(status_data))
        except Exception as e:
            logger.warning(f"Failed to update status record: {str(e)}")
        
        # Send notification
        action_text = "approved ✅" if approved else "rejected ❌"
        try:
            slack_message = f"""
            :memo: *Human Review Complete* :memo:
            
            *Task*: {task_id}
            *Run ID*: {run_id}
            *Decision*: {action_text}
            *Reviewer*: {approver}
            *Comment*: {comment if comment else "None provided"}
            """
            slack_post(slack_message, channel="#ml-approvals")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return {
            "status": action,
            "task_id": task_id,
            "run_id": run_id,
            "approver": approver,
            "timestamp": datetime.now().isoformat(),
            "comment": comment
        }

def wait_for_data_validation(**context) -> Dict[str, Any]:
    """
    Airflow task that waits for human validation of data quality and processing.
    
    Args:
        context: Airflow task context
        
    Returns:
        Dict with validation status and details
    """
    # Get DAG run ID from context
    dag_run = context.get('dag_run')
    if not dag_run:
        raise ValueError("No DAG run information in context")
    
    run_id = dag_run.run_id
    task_id = "data_validation"
    
    # Get data quality and preprocessing results
    ti = context.get('ti')
    quality_results = ti.xcom_pull(task_ids='data_quality_checks', key='quality_results')
    drift_results = ti.xcom_pull(task_ids='drift_detection', key='drift_results')
    processed_path = ti.xcom_pull(task_ids='process_data', key='processed_data_path')
    
    # Determine if this should be auto-approved
    validation_required = Variable.get("REQUIRE_DATA_VALIDATION", default_var="True").lower() == "true"
    
    if not validation_required:
        logger.info("Data validation is disabled, auto-approving")
        return {
            "status": "auto_approved",
            "message": "Data validation step skipped (disabled in configuration)"
        }
    
    # Check if there are any warnings or errors that require human review
    auto_approve = True
    validation_message = "Data processing completed successfully."
    
    # Check quality results
    if quality_results and quality_results.get('status') != 'success':
        auto_approve = False
        validation_message = f"Data quality issues found: {quality_results.get('message', 'Unknown issues')}"
    
    # Check drift results
    drift_detected = drift_results and drift_results.get('drift_detected', False)
    if drift_detected:
        auto_approve = False
        validation_message = f"Data drift detected: {drift_results.get('message', 'Unknown drift')}"
    
    # If everything looks good, auto-approve
    if auto_approve:
        logger.info("Auto-approving data validation (no issues detected)")
        return {
            "status": "auto_approved",
            "message": "Data automatically validated (no issues detected)"
        }
    
    # Request human approval
    artifacts = {
        "quality_results": quality_results,
        "drift_results": drift_results,
        "processed_data_path": processed_path
    }
    
    hitl = HumanInTheLoop()
    hitl.request_approval(
        task_id=task_id,
        run_id=run_id,
        context=context,
        artifacts=artifacts,
        message=validation_message
    )
    
    # Wait for the decision
    max_checks = 5  # For quick testing, in production use higher number
    approval_data = None
    
    for i in range(max_checks):
        approval_data = hitl.check_approval_status(
            task_id=task_id,
            run_id=run_id,
            auto_approve_on_timeout=True
        )
        
        status = approval_data.get('status')
        
        if status == STATUS_APPROVED or status == STATUS_AUTO_APPROVED:
            logger.info(f"Data validation approved: {approval_data}")
            return {
                "status": "approved",
                "details": approval_data
            }
        elif status == STATUS_REJECTED:
            logger.warning(f"Data validation rejected: {approval_data}")
            # Raise exception to stop the DAG
            raise AirflowSkipException("Data validation was rejected by a human reviewer")
        
        # If still pending, wait before checking again
        logger.info(f"Waiting for human validation (check {i+1}/{max_checks})")
        time.sleep(10)  # For testing - in production, use longer intervals
    
    # If we reach here without a decision, default behavior
    auto_approve_timeouts = Variable.get("AUTO_APPROVE_TIMEOUTS", default_var="False").lower() == "true"
    
    if auto_approve_timeouts:
        logger.info("Auto-approving after timeout (as configured)")
        return {
            "status": "auto_approved",
            "message": "Auto-approved after timeout"
        }
    else:
        logger.warning("Validation timed out and auto-approve is disabled")
        raise AirflowSkipException("Data validation timed out waiting for human review")

def wait_for_model_approval(**context) -> Dict[str, Any]:
    """
    Airflow task that waits for human approval of trained models.
    
    Args:
        context: Airflow task context
        
    Returns:
        Dict with approval status and details
    """
    # Get DAG run ID from context
    dag_run = context.get('dag_run')
    if not dag_run:
        raise ValueError("No DAG run information in context")
    
    run_id = dag_run.run_id
    task_id = "model_approval"
    
    # Get training and explainability results
    ti = context.get('ti')
    training_results = ti.xcom_pull(task_ids='train_models', key='training_results')
    explainability_results = ti.xcom_pull(task_ids='model_explainability', key='explainability_results')
    
    # Determine if this should be auto-approved
    approval_required = Variable.get("REQUIRE_MODEL_APPROVAL", default_var="True").lower() == "true"
    
    if not approval_required:
        logger.info("Model approval is disabled, auto-approving")
        return {
            "status": "auto_approved",
            "message": "Model approval step skipped (disabled in configuration)"
        }
    
    # Check if there are any issues that require human review
    auto_approve = True
    approval_message = "Model training completed successfully."
    
    # Check training results
    if not training_results:
        auto_approve = False
        approval_message = "No training results found"
    elif isinstance(training_results, dict):
        # Count failed models
        failed_count = sum(1 for r in training_results.values() if r.get('status') == 'failed')
        
        if failed_count > 0:
            auto_approve = False
            approval_message = f"Training failed for {failed_count} models"
    
    # If everything looks good, auto-approve
    if auto_approve:
        logger.info("Auto-approving models (no issues detected)")
        return {
            "status": "auto_approved",
            "message": "Models automatically approved (no issues detected)"
        }
    
    # Request human approval
    artifacts = {
        "training_results": training_results,
        "explainability_results": explainability_results
    }
    
    hitl = HumanInTheLoop()
    hitl.request_approval(
        task_id=task_id,
        run_id=run_id,
        context=context,
        artifacts=artifacts,
        message=approval_message
    )
    
    # Wait for the decision
    max_checks = 5  # For quick testing, in production use higher number
    approval_data = None
    
    for i in range(max_checks):
        approval_data = hitl.check_approval_status(
            task_id=task_id,
            run_id=run_id,
            auto_approve_on_timeout=True
        )
        
        status = approval_data.get('status')
        
        if status == STATUS_APPROVED or status == STATUS_AUTO_APPROVED:
            logger.info(f"Model approved: {approval_data}")
            return {
                "status": "approved",
                "details": approval_data
            }
        elif status == STATUS_REJECTED:
            logger.warning(f"Model rejected: {approval_data}")
            # Raise exception to stop the DAG
            raise AirflowSkipException("Model was rejected by a human reviewer")
        
        # If still pending, wait before checking again
        logger.info(f"Waiting for model approval (check {i+1}/{max_checks})")
        time.sleep(10)  # For testing - in production, use longer intervals
    
    # If we reach here without a decision, default behavior
    auto_approve_timeouts = Variable.get("AUTO_APPROVE_TIMEOUTS", default_var="False").lower() == "true"
    
    if auto_approve_timeouts:
        logger.info("Auto-approving after timeout (as configured)")
        return {
            "status": "auto_approved",
            "message": "Auto-approved after timeout"
        }
    else:
        logger.warning("Approval timed out and auto-approve is disabled")
        raise AirflowSkipException("Model approval timed out waiting for human review") 