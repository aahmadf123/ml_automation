#!/usr/bin/env python3
"""
fix_proposals.py - AI-Generated Fix Proposals
---------------------------------------------
This module provides functions for generating and managing
fix proposals that can be reviewed and approved by humans.

Features:
- ML issue detection
- Automated fix proposal generation
- Human review workflow
- Fix implementation tracking
"""

import os
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from airflow.models import Variable, XCom
from airflow.exceptions import AirflowSkipException
import pandas as pd
import numpy as np

# Import utility modules
from utils.slack import post as slack_post
from agent_actions import handle_function_call

# Set up logging
logger = logging.getLogger(__name__)

# Constants
FIX_PROPOSAL_PREFIX = "FIX_PROPOSAL_"
FIX_DECISION_PREFIX = "FIX_DECISION_"
PROPOSAL_RETENTION_DAYS = 30  # How long to keep proposals in DB

class FixProposal:
    """Main class for managing fix proposals"""
    
    @staticmethod
    def get_proposal_key(proposal_id: str) -> str:
        """Generate a unique key for fix proposals"""
        return f"{FIX_PROPOSAL_PREFIX}{proposal_id}"
    
    @staticmethod
    def get_decision_key(proposal_id: str) -> str:
        """Generate a unique key for fix decisions"""
        return f"{FIX_DECISION_PREFIX}{proposal_id}"
    
    @staticmethod
    def generate_proposal_id() -> str:
        """Generate a unique ID for a fix proposal"""
        return f"fix_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    @staticmethod
    def propose_fix(
        problem_type: str,
        problem_description: str,
        proposed_solution: str,
        model_id: Optional[str] = None,
        run_id: Optional[str] = None,
        confidence: float = 0.0,
        severity: str = "medium",
        context: Optional[Dict[str, Any]] = None,
        requires_approval: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a fix proposal to be reviewed.
        
        Args:
            problem_type: Type of problem (data, model, system, etc.)
            problem_description: Detailed description of the problem
            proposed_solution: Proposed solution to implement
            model_id: Optional model ID if related to a specific model
            run_id: Optional run ID if related to a specific run
            confidence: Confidence level (0-1)
            severity: Problem severity (critical, high, medium, low)
            context: Additional context about the issue
            requires_approval: Whether human approval is required
            metadata: Additional metadata about the proposal
            
        Returns:
            Dict with proposal details
        """
        proposal_id = FixProposal.generate_proposal_id()
        timestamp = datetime.now().isoformat()
        
        # Create the proposal
        proposal = {
            "proposal_id": proposal_id,
            "problem_type": problem_type,
            "problem_description": problem_description,
            "proposed_solution": proposed_solution,
            "model_id": model_id,
            "run_id": run_id,
            "confidence": confidence,
            "severity": severity,
            "status": "pending" if requires_approval else "auto_approved",
            "timestamp": timestamp,
            "requires_approval": requires_approval,
            "context": context or {},
            "metadata": metadata or {},
            "expiration": (datetime.now() + timedelta(days=PROPOSAL_RETENTION_DAYS)).isoformat()
        }
        
        # Store the proposal
        proposal_key = FixProposal.get_proposal_key(proposal_id)
        Variable.set(proposal_key, json.dumps(proposal))
        
        # Send notification
        severity_emoji = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }.get(severity.lower(), "⚪")
        
        slack_message = f"""
        {severity_emoji} *New Fix Proposal* {severity_emoji}
        
        *ID*: {proposal_id}
        *Problem*: {problem_type}
        *Severity*: {severity}
        *Confidence*: {confidence:.0%}
        
        *Description*:
        {problem_description}
        
        *Proposed Solution*:
        {proposed_solution}
        
        {f"*Model*: {model_id}" if model_id else ""}
        {f"*Run*: {run_id}" if run_id else ""}
        
        *Status*: {proposal["status"]}
        """
        
        try:
            channel = "#fix-proposals"
            if severity.lower() == "critical":
                channel = "#incidents"
                
            slack_post(slack_message, channel=channel)
            logger.info(f"Fix proposal {proposal_id} sent to Slack")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {str(e)}")
        
        return proposal
    
    @staticmethod
    def get_proposal(proposal_id: str) -> Dict[str, Any]:
        """
        Retrieve a fix proposal.
        
        Args:
            proposal_id: ID of the proposal to retrieve
            
        Returns:
            Dict with proposal details, or empty dict if not found
        """
        proposal_key = FixProposal.get_proposal_key(proposal_id)
        
        try:
            proposal_json = Variable.get(proposal_key, default_var="{}")
            proposal = json.loads(proposal_json)
            
            # Check if decision exists
            try:
                decision_key = FixProposal.get_decision_key(proposal_id)
                decision_json = Variable.get(decision_key, default_var="{}")
                decision = json.loads(decision_json)
                
                if decision:
                    proposal["decision"] = decision
            except:
                pass
                
            return proposal
        except Exception as e:
            logger.warning(f"Error retrieving fix proposal: {str(e)}")
            return {}
    
    @staticmethod
    def record_decision(
        proposal_id: str,
        approved: bool,
        reviewer: str,
        comment: str = "",
        implementation_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record a decision on a fix proposal.
        
        Args:
            proposal_id: ID of the proposal
            approved: Whether the proposal was approved
            reviewer: Name of the reviewer
            comment: Optional comment
            implementation_details: Optional details about implementation
            
        Returns:
            Dict with decision details
        """
        # Get the proposal
        proposal_key = FixProposal.get_proposal_key(proposal_id)
        decision_key = FixProposal.get_decision_key(proposal_id)
        
        try:
            proposal_json = Variable.get(proposal_key, default_var="{}")
            proposal = json.loads(proposal_json)
            
            if not proposal:
                logger.warning(f"Proposal {proposal_id} not found")
                return {"status": "error", "message": "Proposal not found"}
                
            # Create decision record
            timestamp = datetime.now().isoformat()
            
            decision = {
                "proposal_id": proposal_id,
                "approved": approved,
                "reviewer": reviewer,
                "comment": comment,
                "timestamp": timestamp,
                "implementation_details": implementation_details or {}
            }
            
            # Save the decision
            Variable.set(decision_key, json.dumps(decision))
            
            # Update the proposal status
            proposal["status"] = "approved" if approved else "rejected"
            proposal["reviewed_by"] = reviewer
            proposal["reviewed_at"] = timestamp
            
            # Save the updated proposal
            Variable.set(proposal_key, json.dumps(proposal))
            
            # Send notification
            decision_emoji = "✅" if approved else "❌"
            slack_message = f"""
            {decision_emoji} *Fix Proposal {proposal["status"].capitalize()}* {decision_emoji}
            
            *ID*: {proposal_id}
            *Problem*: {proposal.get("problem_type")}
            *Reviewer*: {reviewer}
            *Decision*: {proposal["status"].capitalize()}
            
            {f"*Comment*: {comment}" if comment else ""}
            
            *Original Description*:
            {proposal.get("problem_description", "Not provided")}
            
            *Original Solution*:
            {proposal.get("proposed_solution", "Not provided")}
            """
            
            try:
                channel = "#fix-proposals"
                if proposal.get("severity", "").lower() == "critical":
                    channel = "#incidents"
                    
                slack_post(slack_message, channel=channel)
                logger.info(f"Fix proposal decision for {proposal_id} sent to Slack")
            except Exception as e:
                logger.warning(f"Failed to send Slack notification: {str(e)}")
            
            # If approved and we have implementation details, trigger agent
            if approved and implementation_details and implementation_details.get("auto_implement", False):
                try:
                    logger.info(f"Auto-implementing fix for proposal {proposal_id}")
                    FixProposal.implement_fix(proposal_id, proposal, decision)
                except Exception as e:
                    logger.error(f"Failed to auto-implement fix: {str(e)}")
            
            return decision
        except Exception as e:
            logger.error(f"Error recording fix proposal decision: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    def implement_fix(proposal_id: str, proposal: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement an approved fix using the agent.
        
        Args:
            proposal_id: ID of the proposal
            proposal: Proposal details
            decision: Decision details
            
        Returns:
            Dict with implementation details
        """
        try:
            # Extract details
            solution = proposal.get("proposed_solution", "")
            problem = proposal.get("problem_description", "")
            
            # Call agent actions to implement the fix
            function_call = {
                "name": "implement_fix",
                "arguments": {
                    "fix_id": proposal_id,
                    "problem": problem,
                    "solution": solution,
                    "approved_by": decision.get("reviewer", "Unknown"),
                    "metadata": proposal.get("metadata", {})
                }
            }
            
            # Use the agent to implement the fix
            result = handle_function_call(function_call)
            
            # Update the implementation status
            updated_decision = decision.copy()
            updated_decision["implementation"] = {
                "status": "completed" if result.get("success", False) else "failed",
                "timestamp": datetime.now().isoformat(),
                "details": result
            }
            
            # Save the updated decision
            decision_key = FixProposal.get_decision_key(proposal_id)
            Variable.set(decision_key, json.dumps(updated_decision))
            
            return result
        except Exception as e:
            logger.error(f"Error implementing fix: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    def list_pending_proposals() -> List[Dict[str, Any]]:
        """
        List all pending fix proposals.
        
        Returns:
            List of pending fix proposals
        """
        pending_proposals = []
        
        try:
            # Get all variables that might be fix proposals
            all_variables = {k: v for k, v in Variable.get_all().items() 
                           if k.startswith(FIX_PROPOSAL_PREFIX)}
            
            for var_key, var_value in all_variables.items():
                try:
                    proposal = json.loads(var_value)
                    
                    # Only include pending proposals
                    if proposal.get("status") == "pending":
                        pending_proposals.append(proposal)
                except:
                    pass
                    
            # Sort by timestamp (most recent first)
            pending_proposals.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
            
            return pending_proposals
        except Exception as e:
            logger.error(f"Error listing pending proposals: {str(e)}")
            return []

def analyze_model_metrics(
    model_id: str,
    metrics: Dict[str, float],
    previous_metrics: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, Dict[str, float]]] = None
) -> List[Dict[str, Any]]:
    """
    Analyze model metrics and generate fix proposals for issues.
    
    Args:
        model_id: ID of the model
        metrics: Current model metrics
        previous_metrics: Previous model metrics for comparison
        thresholds: Dictionary of metric thresholds by severity
        
    Returns:
        List of generated fix proposals
    """
    proposals = []
    
    # Default thresholds if not provided
    default_thresholds = {
        "rmse": {"critical": 0.5, "high": 0.3, "medium": 0.2, "low": 0.1},
        "mae": {"critical": 0.4, "high": 0.25, "medium": 0.15, "low": 0.05},
        "r2": {"critical": 0.5, "high": 0.6, "medium": 0.7, "low": 0.8},
        "accuracy": {"critical": 0.7, "high": 0.8, "medium": 0.9, "low": 0.95}
    }
    
    thresholds = thresholds or default_thresholds
    
    # Check each metric against thresholds
    for metric_name, metric_value in metrics.items():
        if metric_name not in thresholds:
            continue
            
        # For metrics where higher is better (like accuracy, r2)
        is_higher_better = metric_name in ['accuracy', 'r2', 'auc', 'precision', 'recall', 'f1']
        
        # Check against thresholds
        for severity, threshold in thresholds[metric_name].items():
            if (is_higher_better and metric_value < threshold) or \
               (not is_higher_better and metric_value > threshold):
                
                # Compare with previous metrics if available
                comparison = ""
                if previous_metrics and metric_name in previous_metrics:
                    prev_value = previous_metrics[metric_name]
                    change = metric_value - prev_value
                    change_pct = (change / prev_value) * 100 if prev_value != 0 else float('inf')
                    
                    if is_higher_better:
                        direction = "decreased" if change < 0 else "increased"
                    else:
                        direction = "increased" if change > 0 else "decreased"
                        
                    comparison = f"This has {direction} by {abs(change):.4f} ({abs(change_pct):.2f}%) compared to the previous value of {prev_value:.4f}."
                
                # Generate a problem description
                problem_description = f"Model '{model_id}' has a {metric_name} of {metric_value:.4f}, which is worse than the {severity} threshold of {threshold:.4f}. {comparison}"
                
                # Generate proposed solutions based on the metric
                if metric_name == 'rmse' or metric_name == 'mae':
                    proposed_solution = f"""
                    To improve the {metric_name} for model '{model_id}', consider:
                    
                    1. Feature engineering - Add more predictive features or transform existing ones
                    2. Hyperparameter tuning - Adjust learning rate, regularization, or model complexity
                    3. Data quality - Check for outliers or improve data preprocessing
                    4. Model architecture - Consider a more complex model or ensemble methods
                    5. Sample weighting - Adjust the weights of training examples
                    """
                elif metric_name == 'r2':
                    proposed_solution = f"""
                    To improve the R² score for model '{model_id}', consider:
                    
                    1. Feature selection - The model may be missing important predictive features
                    2. Non-linearity - Try adding polynomial features or using a non-linear model
                    3. Data transformation - Apply log or power transformations to target or features
                    4. Segmentation - The data might benefit from building separate models for different segments
                    5. Ensemble methods - Combine multiple models to capture different patterns
                    """
                elif metric_name in ['accuracy', 'precision', 'recall', 'f1']:
                    proposed_solution = f"""
                    To improve the {metric_name} for model '{model_id}', consider:
                    
                    1. Class imbalance - Apply resampling techniques or adjust class weights
                    2. Threshold tuning - Adjust the classification threshold
                    3. Feature importance - Focus on features with higher discrimination power
                    4. Advanced models - Try models specifically designed for classification (e.g., XGBoost)
                    5. Hyperparameter tuning - Optimize for {metric_name} specifically
                    """
                else:
                    proposed_solution = f"""
                    To improve the {metric_name} for model '{model_id}', consider:
                    
                    1. Data quality - Review the data for issues or outliers
                    2. Feature engineering - Add more predictive features
                    3. Hyperparameter tuning - Optimize model parameters
                    4. Model selection - Try a different model architecture
                    5. Regularization - Adjust to prevent overfitting
                    """
                
                # Create a fix proposal
                fix = FixProposal.propose_fix(
                    problem_type="model_performance",
                    problem_description=problem_description,
                    proposed_solution=proposed_solution,
                    model_id=model_id,
                    confidence=0.8,
                    severity=severity,
                    context={"metrics": metrics, "thresholds": thresholds[metric_name]},
                    requires_approval=True
                )
                
                proposals.append(fix)
                
                # Only create one proposal per metric (use the highest severity)
                break
    
    return proposals

def analyze_data_quality(
    df: pd.DataFrame,
    thresholds: Optional[Dict[str, Dict[str, float]]] = None
) -> List[Dict[str, Any]]:
    """
    Analyze data quality and generate fix proposals for issues.
    
    Args:
        df: DataFrame to analyze
        thresholds: Dictionary of quality thresholds by severity
        
    Returns:
        List of generated fix proposals
    """
    proposals = []
    
    # Default thresholds
    default_thresholds = {
        "missing_pct": {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.05},
        "zero_pct": {"critical": 0.9, "high": 0.8, "medium": 0.7, "low": 0.5},
        "unique_pct": {"critical": 0.95, "high": 0.9, "medium": 0.8, "low": 0.7},
        "skewness": {"critical": 5.0, "high": 3.0, "medium": 2.0, "low": 1.0}
    }
    
    thresholds = thresholds or default_thresholds
    
    # Basic statistics
    num_rows = len(df)
    num_cols = len(df.columns)
    
    # Missing values check
    missing_counts = df.isnull().sum()
    missing_pct = missing_counts / num_rows
    
    for col in missing_pct.index:
        pct = missing_pct[col]
        
        for severity, threshold in thresholds["missing_pct"].items():
            if pct > threshold:
                problem_description = f"Column '{col}' has {pct:.1%} missing values, which exceeds the {severity} threshold of {threshold:.1%}."
                
                proposed_solution = f"""
                To address the high number of missing values in column '{col}', consider:
                
                1. Imputation - Replace missing values with mean, median, or mode
                2. Dropping - If the column has too many missing values, consider dropping it
                3. Derived features - Create a binary indicator for missingness
                4. Advanced imputation - Use models to predict missing values
                5. Domain expertise - Consult domain experts for specialized imputation
                """
                
                fix = FixProposal.propose_fix(
                    problem_type="data_quality",
                    problem_description=problem_description,
                    proposed_solution=proposed_solution,
                    confidence=0.9,
                    severity=severity,
                    context={"column": col, "missing_pct": pct, "threshold": threshold},
                    requires_approval=True
                )
                
                proposals.append(fix)
                break
    
    # Zero values check (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        zero_pct = (df[col] == 0).sum() / num_rows
        
        for severity, threshold in thresholds["zero_pct"].items():
            if zero_pct > threshold:
                problem_description = f"Column '{col}' has {zero_pct:.1%} zero values, which exceeds the {severity} threshold of {threshold:.1%}."
                
                proposed_solution = f"""
                To address the high number of zero values in column '{col}', consider:
                
                1. Transformation - Apply a transformation (e.g., log(x+1)) to handle zeros
                2. Feature engineering - Create a binary indicator for zero/non-zero
                3. Data investigation - Determine if zeros represent missing data or actual measurements
                4. Segmentation - Split modeling into zero and non-zero groups
                5. Alternative encodings - Use different techniques for sparse features
                """
                
                fix = FixProposal.propose_fix(
                    problem_type="data_quality",
                    problem_description=problem_description,
                    proposed_solution=proposed_solution,
                    confidence=0.85,
                    severity=severity,
                    context={"column": col, "zero_pct": zero_pct, "threshold": threshold},
                    requires_approval=True
                )
                
                proposals.append(fix)
                break
    
    # Skewness check
    for col in numeric_cols:
        try:
            skewness = df[col].skew()
            
            for severity, threshold in thresholds["skewness"].items():
                if abs(skewness) > threshold:
                    direction = "positively" if skewness > 0 else "negatively"
                    problem_description = f"Column '{col}' is highly {direction} skewed (skewness = {skewness:.2f}), which exceeds the {severity} threshold of {threshold:.2f}."
                    
                    proposed_solution = f"""
                    To address the high skewness in column '{col}', consider:
                    
                    1. Transformation - Apply {'log, square root, or Box-Cox' if skewness > 0 else 'power transformations'}
                    2. Binning - Convert to categorical bins to reduce impact of skewness
                    3. Outlier treatment - {'Cap or remove extreme high values' if skewness > 0 else 'Cap or remove extreme low values'}
                    4. Normalization - Use robust scalers like MinMaxScaler
                    5. Non-parametric models - Consider models less affected by feature distributions
                    """
                    
                    fix = FixProposal.propose_fix(
                        problem_type="data_quality",
                        problem_description=problem_description,
                        proposed_solution=proposed_solution,
                        confidence=0.85,
                        severity=severity,
                        context={"column": col, "skewness": skewness, "threshold": threshold},
                        requires_approval=True
                    )
                    
                    proposals.append(fix)
                    break
        except:
            # Skip columns where skewness can't be calculated
            pass
    
    return proposals

def get_pending_fixes(**context) -> Dict[str, Any]:
    """
    Airflow task to get pending fix proposals.
    
    Args:
        context: Airflow task context
        
    Returns:
        Dict with pending fix proposals
    """
    # Get pending proposals
    fix_manager = FixProposal()
    pending_proposals = fix_manager.list_pending_proposals()
    
    # Store in XCom
    context['ti'].xcom_push(key='pending_fixes', value=pending_proposals)
    
    logger.info(f"Found {len(pending_proposals)} pending fix proposals")
    
    return {
        "status": "success",
        "pending_count": len(pending_proposals),
        "proposals": pending_proposals
    }

def analyze_metrics_for_issues(**context) -> Dict[str, Any]:
    """
    Airflow task to analyze model metrics for issues.
    
    Args:
        context: Airflow task context
        
    Returns:
        Dict with generated fix proposals
    """
    # Get training results from XCom
    ti = context.get('ti')
    training_results = ti.xcom_pull(task_ids='train_models', key='training_results')
    
    if not training_results or not isinstance(training_results, dict):
        logger.warning("No valid training results found")
        return {"status": "skipped", "reason": "No valid training results found"}
    
    # Analyze metrics for each model
    all_proposals = []
    
    for model_id, result in training_results.items():
        if result.get('status') != 'completed':
            continue
            
        metrics = result.get('metrics', {})
        if not metrics:
            continue
            
        # Get previous metrics if available
        previous_metrics = None
        try:
            previous_results = ti.xcom_pull(key=f'previous_metrics_{model_id}')
            if previous_results:
                previous_metrics = previous_results
        except:
            pass
            
        # Analyze metrics
        proposals = analyze_model_metrics(
            model_id=model_id,
            metrics=metrics,
            previous_metrics=previous_metrics
        )
        
        all_proposals.extend(proposals)
    
    # Store proposals in XCom
    context['ti'].xcom_push(key='fix_proposals', value=all_proposals)
    
    logger.info(f"Generated {len(all_proposals)} fix proposals from metrics analysis")
    
    return {
        "status": "success",
        "proposal_count": len(all_proposals),
        "proposals": all_proposals
    }

def auto_implement_approved_fixes(**context) -> Dict[str, Any]:
    """
    Airflow task to auto-implement approved fixes.
    
    Args:
        context: Airflow task context
        
    Returns:
        Dict with implementation results
    """
    # Check for any recently approved fixes that need implementation
    fix_manager = FixProposal()
    
    try:
        # Get all variables that might be fix decisions
        all_variables = {k: v for k, v in Variable.get_all().items() 
                       if k.startswith(FIX_DECISION_PREFIX)}
        
        implemented_count = 0
        failed_count = 0
        
        for var_key, var_value in all_variables.items():
            try:
                decision = json.loads(var_value)
                
                # Only process approved decisions without implementation
                if decision.get("approved", False) and "implementation" not in decision:
                    proposal_id = decision.get("proposal_id")
                    
                    if not proposal_id:
                        continue
                        
                    # Get the proposal
                    proposal = fix_manager.get_proposal(proposal_id)
                    
                    if not proposal:
                        logger.warning(f"Proposal {proposal_id} not found for approved decision")
                        continue
                        
                    # Check if this is a candidate for auto-implementation
                    auto_implement = decision.get("implementation_details", {}).get("auto_implement", False)
                    
                    if auto_implement:
                        logger.info(f"Auto-implementing fix for proposal {proposal_id}")
                        
                        # Implement the fix
                        result = fix_manager.implement_fix(proposal_id, proposal, decision)
                        
                        if result.get("status") != "error":
                            implemented_count += 1
                        else:
                            failed_count += 1
            except Exception as e:
                logger.warning(f"Error processing decision: {str(e)}")
        
        logger.info(f"Auto-implemented {implemented_count} fixes, {failed_count} failed")
        
        return {
            "status": "success",
            "implemented_count": implemented_count,
            "failed_count": failed_count
        }
    except Exception as e:
        logger.error(f"Error auto-implementing fixes: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        } 