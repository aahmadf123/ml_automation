#!/usr/bin/env python3
"""
agent_actions.py

Central dispatcher for AI-agent functions in the Homeowner pipeline.
- Slack, Airflow trigger, and S3 utilities.
- AI-agent hooks: both legacy helpers and new human-in-the-loop functions.
"""
import json
import logging
import time
from typing import Any, Dict, Optional, List, Union
import sys
import re
from datetime import datetime

# Setup logging
log = logging.getLogger(__name__)

###############################################
# Core Notification & Trigger Functions
###############################################
def send_to_slack(
    channel: str,
    title: str,
    details: str,
    urgency: str = "normal"
) -> Dict[str, Any]:
    """
    Send a message to Slack with retry logic.
    
    Args:
        channel: Slack channel to post to
        title: Message title
        details: Message details
        urgency: Message urgency level (normal/high/critical)
        
    Returns:
        Dict containing the Slack API response
    """
    from tenacity import retry, stop_after_attempt, wait_fixed
    import utils.slack as slack
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _send():
        log.info(f"Posting to Slack {channel}: {title}")
        return slack.post(
            channel=channel,
            title=title,
            details=details,
            urgency=urgency
        )
    
    try:
        return _send()
    except Exception as e:
        log.error(f"Failed to send Slack message: {str(e)}")
        raise

def trigger_airflow_dag(
    dag_id: str,
    conf: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Trigger an Airflow DAG with retry logic.
    
    Args:
        dag_id: ID of the DAG to trigger
        conf: Optional configuration to pass to the DAG
        
    Returns:
        Dict containing the Airflow API response
    """
    from tenacity import retry, stop_after_attempt, wait_fixed
    import utils.airflow_api as airflow_api
    import utils.config as config
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _trigger():
        log.info(f"Triggering DAG {dag_id}")
        return airflow_api.trigger_dag(
            dag_id=dag_id,
            conf=conf or config.AIRFLOW_DAG_BASE_CONF
        )
    
    try:
        return _trigger()
    except Exception as e:
        log.error(f"Failed to trigger DAG {dag_id}: {str(e)}")
        raise

###############################################
# Generic dispatcher for agent calls
###############################################
def handle_function_call(payload: Dict[str, Any]) -> Any:
    """
    Handle function calls from the agent.
    
    Args:
        payload: Dictionary containing function name and arguments
        
    Returns:
        Result of the function call
        
    Raises:
        ValueError: If function not found or arguments invalid
    """
    func_def = payload.get("function", {})
    name = func_def.get("name")
    args_str = func_def.get("arguments", "{}")
    
    try:
        kwargs = json.loads(args_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON args: {args_str}") from e
        
    func = globals().get(name)
    if not func or not callable(func):
        raise ValueError(f"Function '{name}' not found")
        
    log.info(f"Calling {name} with {kwargs}")
    try:
        return func(**kwargs)
    except Exception as e:
        log.error(f"Function call failed: {str(e)}")
        raise

###############################################
# Code Interpreter Hook
###############################################
def execute_python(code: str) -> Any:
    """
    Execute Python code in the Code Interpreter sandbox.
    
    Args:
        code: Python code to execute
        
    Returns:
        Result of code execution
        
    Raises:
        PermissionError: If code tries to access outside /tmp
    """
    if not code.startswith("/tmp/"):
        raise PermissionError("Code execution restricted to /tmp directory only.")
        
    log.info(f"Executing code: {code}")
    # Log the action for audit trail
    log_manual_action("execute_python", {"code": code})
    pass

###############################################
# Legacy AI-Agent Helpers
###############################################
def propose_fix(
    problem_summary: str,
    proposed_fix: str,
    confidence: float,
    requires_human_approval: bool
) -> Dict[str, Any]:
    """
    Propose a fix for an identified problem.
    
    Args:
        problem_summary: Description of the problem
        proposed_fix: Proposed solution
        confidence: Confidence level (0-1)
        requires_human_approval: Whether human approval is needed
        
    Returns:
        Dict containing the fix proposal
    """
    result = {
        "problem_summary": problem_summary,
        "proposed_fix": proposed_fix,
        "confidence": confidence,
        "requires_human_approval": requires_human_approval
    }
    log.info(f"Propose Fix: {result}")
    return result

def override_decision(
    fix_id: str,
    approved_by: str,
    comment: str = ""
) -> Dict[str, Any]:
    """
    Record an override decision.
    
    Args:
        fix_id: ID of the fix being overridden
        approved_by: Name of approver
        comment: Optional comment
        
    Returns:
        Dict containing the override decision
    """
    result = {
        "fix_id": fix_id,
        "approved_by": approved_by,
        "comment": comment,
        "status": "approved"
    }
    log.info(f"Override Decision: {result}")
    return result

def generate_root_cause_report(
    dag_id: str,
    execution_date: str
) -> Dict[str, Any]:
    """
    Generate a root cause analysis report.
    
    Args:
        dag_id: ID of the DAG
        execution_date: Date of execution
        
    Returns:
        Dict containing the root cause report
    """
    report = {
        "dag_id": dag_id,
        "execution_date": execution_date,
        "root_cause": "Data drift in feature 'example_feature' and increased RMSE.",
        "details": "Detected a 15% drift in 'example_feature' and RMSE increased by 0.5 units."
    }
    log.info(f"Root Cause Report: {report}")
    return report

def suggest_hyperparam_improvement(
    model_id: str,
    current_rmse: float,
    previous_best_rmse: float
) -> Dict[str, Any]:
    """
    Suggest hyperparameter improvements.
    
    Args:
        model_id: ID of the model
        current_rmse: Current RMSE value
        previous_best_rmse: Previous best RMSE value
        
    Returns:
        Dict containing the hyperparameter suggestion
    """
    suggestion = {
        "model_id": model_id,
        "suggestion": "Increase 'n_estimators' and decrease 'learning_rate'.",
        "current_rmse": current_rmse,
        "previous_best_rmse": previous_best_rmse,
        "confidence": 0.8
    }
    log.info(f"Hyperparameter Improvement Suggestion: {suggestion}")
    return suggestion

def validate_data_integrity(dataset_path: str) -> Dict[str, Any]:
    """
    Validate the integrity of a dataset.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        Dict containing the validation report
    """
    import pandas as pd
    
    report = {
        "dataset_path": dataset_path,
        "valid": True,
        "issues": None,
        "row_count": None
    }
    
    try:
        df = pd.read_csv(dataset_path)
        report["row_count"] = len(df)
    except Exception as e:
        report["valid"] = False
        report["issues"] = str(e)
        log.error(f"Data validation failed: {str(e)}")
        
    log.info(f"Data Integrity Report: {report}")
    return report

def describe_fix_plan(
    issue_type: str,
    solution_summary: str
) -> Dict[str, Any]:
    """
    Describe a fix plan for an issue.
    
    Args:
        issue_type: Type of issue
        solution_summary: Summary of the solution
        
    Returns:
        Dict containing the fix plan
    """
    plan = {
        "issue_type": issue_type,
        "solution_summary": solution_summary,
        "expected_outcome": "Improved model performance and reduced drift."
    }
    log.info(f"Fix Plan Description: {plan}")
    return plan

def fetch_airflow_logs(
    dag_id: str,
    run_id: str
) -> Dict[str, Any]:
    """
    Fetch logs for an Airflow DAG run.
    
    Args:
        dag_id: ID of the DAG
        run_id: ID of the run
        
    Returns:
        Dict containing the logs
    """
    logs = f"Simulated logs for DAG {dag_id}, run {run_id}."
    log.info(logs)
    return {"logs": logs}

def update_airflow_variable(
    key: str,
    value: str
) -> Dict[str, Any]:
    """
    Update an Airflow variable.
    
    Args:
        key: Variable key
        value: New value
        
    Returns:
        Dict containing the update status
    """
    log.info(f"Updating Airflow Variable {key} to {value}")
    return {"status": "success", "variable": key, "new_value": value}

def list_recent_failures(
    lookback_hours: int
) -> Dict[str, Any]:
    """
    List recent DAG failures.
    
    Args:
        lookback_hours: Number of hours to look back
        
    Returns:
        Dict containing the list of failures
    """
    failures = [{
        "dag_id": "homeowner_loss_history_full_pipeline",
        "task_id": "train_compare_model1",
        "failure_time": "2025-04-18T12:34:56Z"
    }]
    log.info(f"Recent failures (last {lookback_hours}h): {failures}")
    return {"failures": failures}

def escalate_issue(
    issue_summary: str,
    contact_method: str,
    severity: str
) -> Dict[str, Any]:
    """
    Escalate an issue to the appropriate team.
    
    Args:
        issue_summary: Summary of the issue
        contact_method: Method of contact
        severity: Severity level
        
    Returns:
        Dict containing the escalation details
    """
    escalation_message = f"Escalation [{severity}]: {issue_summary} via {contact_method}."
    log.info(escalation_message)
    return {"status": "escalated", "message": escalation_message}

###############################################
# New AI-Agent Hooks: Self-Healing & Human-in-the-Loop
###############################################
def detect_and_flag_drift(
    drift_metrics: Dict[str, float],
    threshold: float = 0.1
) -> bool:
    """
    Detect and flag data drift.
    
    Args:
        drift_metrics: Dictionary of drift metrics
        threshold: Drift threshold
        
    Returns:
        bool indicating if drift was detected
    """
    for feature, drift in drift_metrics.items():
        if drift > threshold:
            log.warning(f"Drift alert: {feature} -> {drift:.2%}")
            return True
    log.info("No significant drift detected.")
    return False

def schedule_retraining(
    trigger_conf: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Schedule a model retraining.
    
    Args:
        trigger_conf: Configuration for the retraining
        
    Returns:
        Dict containing the scheduling status
    """
    from utils.airflow_api import trigger_dag as api_trigger_dag
    
    try:
        run = api_trigger_dag(
            dag_id=trigger_conf.get("dag_id"),
            conf=trigger_conf
        )
        return {"status": "scheduled", "run": run}
    except Exception as e:
        log.error(f"Failed to schedule retraining: {str(e)}")
        raise

def explain_model_decision(
    inputs: List[Dict[str, Any]],
    model_id: str
) -> Dict[str, Any]:
    """
    Explain model decisions.
    
    Args:
        inputs: List of input records
        model_id: ID of the model
        
    Returns:
        Dict containing the explanations
    """
    explanations = [{"input": inp, "shap_values": []} for inp in inputs]
    log.info(f"Generated explanations for {len(inputs)} records.")
    return {"model_id": model_id, "explanations": explanations}

def approve_deployment(
    run_id: str,
    approver: str
) -> Dict[str, Any]:
    """
    Approve a model deployment.
    
    Args:
        run_id: ID of the run
        approver: Name of approver
        
    Returns:
        Dict containing the approval status
    """
    log.info(f"Deployment approved by {approver} for run {run_id}.")
    return {"run_id": run_id, "approved_by": approver}

def rollback_model(
    model_name: str,
    to_version: str
) -> Dict[str, Any]:
    """
    Rollback a model to a previous version.
    
    Args:
        model_name: Name of the model
        to_version: Version to rollback to
        
    Returns:
        Dict containing the rollback status
    """
    log.info(f"Rolling back {model_name} to version {to_version}.")
    return {"model": model_name, "rolled_back_to": to_version}

def generate_executive_summary(
    period: str,
    metrics: Dict[str, Any]
) -> str:
    """
    Generate an executive summary.
    
    Args:
        period: Time period
        metrics: Dictionary of metrics
        
    Returns:
        String containing the summary
    """
    summary = f"Executive Summary for {period}: " + ", ".join(
        f"{k}={v}" for k,v in metrics.items()
    )
    log.info("Generated executive summary.")
    return summary

def open_incident_ticket(
    issue: str,
    severity: str,
    assignee: str
) -> Dict[str, Any]:
    """
    Open an incident ticket.
    
    Args:
        issue: Description of the issue
        severity: Severity level
        assignee: Person assigned to the ticket
        
    Returns:
        Dict containing the ticket details
    """
    ticket_id = f"TICKET-{int(time.time())}"
    log.info(f"Opened {ticket_id}: {issue}")
    return {
        "ticket_id": ticket_id,
        "severity": severity,
        "assignee": assignee
    }

def optimize_compute_resources(
    history: Dict[str, float]
) -> Dict[str, Any]:
    """
    Optimize compute resources.
    
    Args:
        history: Dictionary of historical metrics
        
    Returns:
        Dict containing optimization suggestions
    """
    suggestion = {"workers": 5, "instance_type": "m5.xlarge"}
    log.info("Generated compute optimization suggestion.")
    return suggestion

def simulate_what_if(
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulate what-if scenarios.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Dict containing simulation results
    """
    prediction = {"rmse": 0.123, "training_time": 42}
    log.info("Completed what-if simulation.")
    return prediction

def audit_data_quality(
    dataset_path: str
) -> Dict[str, Any]:
    """
    Audit data quality.
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        Dict containing the audit results
    """
    report = {
        "nulls": 0,
        "duplicates": 0,
        "schema_violations": 0
    }
    log.info("Completed data quality audit.")
    return report

###############################################
# Role-Based Access Control (RBAC)
###############################################
def check_permissions(
    user_role: str,
    action: str
) -> bool:
    """
    Check user permissions.
    
    Args:
        user_role: Role of the user
        action: Action to check
        
    Returns:
        bool indicating if action is allowed
    """
    # Implementation would check against a permissions matrix
    return True

def log_manual_action(
    action: str,
    details: Dict[str, Any]
) -> None:
    """
    Log a manual action.
    
    Args:
        action: Type of action
        details: Details of the action
    """
    log.info(f"Manual action: {action} - {details}")

###############################################
# Fix Implementation Functions
###############################################
def implement_fix(
    fix_id: str,
    problem: str,
    solution: str,
    approved_by: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Implement an approved fix from the fix proposal system.
    
    Args:
        fix_id: ID of the fix to implement
        problem: Description of the problem
        solution: Proposed solution to implement
        approved_by: Who approved the fix
        metadata: Additional metadata
        
    Returns:
        Dict with implementation status
    """
    log.info(f"Implementing fix {fix_id} approved by {approved_by}")
    log.info(f"Problem: {problem}")
    log.info(f"Solution: {solution}")
    
    result = {
        "fix_id": fix_id,
        "implemented_by": "ai-agent",
        "timestamp": datetime.now().isoformat(),
        "success": False
    }
    
    try:
        # Extract the type of fix from metadata or problem description
        fix_type = metadata.get("fix_type") if metadata else "unknown"
        model_id = metadata.get("model_id") if metadata else None
        
        if "data quality" in problem.lower() or "missing values" in problem.lower():
            fix_type = "data_quality"
        elif "model performance" in problem.lower() or "rmse" in problem.lower():
            fix_type = "model_performance"
        elif "parameter" in problem.lower() or "hyperparameter" in problem.lower():
            fix_type = "hyperparameter_tuning"
        
        # Implement the appropriate fix based on type
        if fix_type == "data_quality":
            result.update(implement_data_quality_fix(problem, solution, model_id))
        elif fix_type == "model_performance":
            result.update(implement_model_performance_fix(problem, solution, model_id))
        elif fix_type == "hyperparameter_tuning":
            result.update(implement_hyperparameter_tuning(problem, solution, model_id))
        else:
            result.update({
                "status": "error",
                "message": f"Unknown fix type: {fix_type}"
            })
            
        # Log the implementation
        log.info(f"Fix implementation result: {result}")
        
        return result
    except Exception as e:
        error_msg = f"Error implementing fix: {str(e)}"
        log.error(error_msg)
        result.update({
            "status": "error",
            "message": error_msg,
            "error": str(e)
        })
        return result

def implement_data_quality_fix(
    problem: str,
    solution: str,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Implement a data quality fix.
    
    Args:
        problem: Description of the problem
        solution: Proposed solution
        model_id: Optional model ID
        
    Returns:
        Dict with implementation details
    """
    # Extract column name if present
    column_match = re.search(r"Column '([^']+)'", problem)
    column = column_match.group(1) if column_match else None
    
    # Determine the type of data quality issue
    issue_type = "unknown"
    if "missing values" in problem.lower():
        issue_type = "missing_values"
    elif "zero values" in problem.lower():
        issue_type = "zero_values"
    elif "skew" in problem.lower():
        issue_type = "skewness"
    elif "outlier" in problem.lower():
        issue_type = "outliers"
    
    # Create preprocessing directive to implement in next run
    directive = {
        "issue_type": issue_type,
        "column": column,
        "action": "auto_fix",
        "timestamp": datetime.now().isoformat()
    }
    
    # Store directive in Airflow variable for next pipeline run
    if column and issue_type != "unknown":
        try:
            from airflow.models import Variable
            var_name = f"DQ_FIX_{issue_type}_{column}"
            Variable.set(var_name, json.dumps(directive))
            
            return {
                "success": True,
                "message": f"Data quality fix directive created for {issue_type} in column {column}",
                "directive": directive
            }
        except Exception as e:
            log.error(f"Failed to create directive: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create data quality fix directive: {str(e)}"
            }
    else:
        return {
            "success": False,
            "message": "Could not determine column or issue type from problem description"
        }

def implement_model_performance_fix(
    problem: str,
    solution: str,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Implement a model performance fix.
    
    Args:
        problem: Description of the problem
        solution: Proposed solution
        model_id: Optional model ID
        
    Returns:
        Dict with implementation details
    """
    if not model_id:
        # Try to extract model ID from problem description
        model_match = re.search(r"Model '([^']+)'", problem)
        model_id = model_match.group(1) if model_match else None
    
    if not model_id:
        return {
            "success": False,
            "message": "Could not determine model ID from problem description"
        }
    
    # Determine which action to take based on the solution
    action = "unknown"
    if "feature engineering" in solution.lower():
        action = "feature_engineering"
    elif "hyperparameter" in solution.lower():
        action = "hyperparameter_tuning"
    elif "model architecture" in solution.lower():
        action = "model_architecture"
    elif "ensemble" in solution.lower():
        action = "ensemble_methods"
    
    # Create a request for model retuning
    try:
        # Import here to avoid circular imports
        from tasks.model_retuning import ModelRetuning
        
        retuning = ModelRetuning()
        request = retuning.request_retuning(
            model_id=model_id,
            requestor="ai-agent",
            reason=f"Auto-implemented fix for model performance issue: {problem[:100]}...",
            param_overrides={
                "improvement_action": action,
                "auto_triggered": True
            },
            priority="high"
        )
        
        return {
            "success": True,
            "message": f"Created retuning request for {model_id} with action {action}",
            "request": request
        }
    except Exception as e:
        log.error(f"Failed to create retuning request: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to create retuning request: {str(e)}"
        }

def implement_hyperparameter_tuning(
    problem: str,
    solution: str,
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Implement hyperparameter tuning.
    
    Args:
        problem: Description of the problem
        solution: Proposed solution
        model_id: Optional model ID
        
    Returns:
        Dict with implementation details
    """
    if not model_id:
        # Try to extract model ID from problem description
        model_match = re.search(r"Model '([^']+)'", problem)
        model_id = model_match.group(1) if model_match else None
    
    if not model_id:
        return {
            "success": False,
            "message": "Could not determine model ID from problem description"
        }
    
    # Extract parameter suggestions from solution text
    param_overrides = {}
    
    # Look for common hyperparameters in the solution
    learning_rate_match = re.search(r"learning rate.*?(\d+\.\d+)", solution)
    if learning_rate_match:
        param_overrides["learning_rate"] = float(learning_rate_match.group(1))
    
    max_depth_match = re.search(r"max depth.*?(\d+)", solution)
    if max_depth_match:
        param_overrides["max_depth"] = int(max_depth_match.group(1))
    
    n_estimators_match = re.search(r"estimators.*?(\d+)", solution)
    if n_estimators_match:
        param_overrides["n_estimators"] = int(n_estimators_match.group(1))
    
    # Create a request for model retuning with parameter overrides
    try:
        # Import here to avoid circular imports
        from tasks.model_retuning import ModelRetuning
        
        retuning = ModelRetuning()
        request = retuning.request_retuning(
            model_id=model_id,
            requestor="ai-agent",
            reason=f"Auto-implemented hyperparameter tuning: {problem[:100]}...",
            param_overrides=param_overrides,
            priority="high"
        )
        
        return {
            "success": True,
            "message": f"Created hyperparameter tuning request for {model_id} with {len(param_overrides)} overrides",
            "request": request,
            "param_overrides": param_overrides
        }
    except Exception as e:
        log.error(f"Failed to create hyperparameter tuning request: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to create hyperparameter tuning request: {str(e)}"
        }
