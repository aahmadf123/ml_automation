#!/usr/bin/env python3
"""
model_comparison.py - Module for comparing multiple trained models
-----------------------------------------------------------------
This module provides functionality to compare the performance of multiple
trained models and generate reports for visualization.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

# Import Airflow specific modules
from airflow.models import Variable
from airflow.exceptions import AirflowSkipException
from airflow.decorators import task
import mlflow

# Fix import paths - use absolute imports
import utils.metrics as metrics
import utils.s3 as s3_utils
import utils.notifications as notifications

logger = logging.getLogger(__name__)

# Default S3 locations
DEFAULT_RESULTS_BUCKET = "ml-model-results"
DEFAULT_REPORTS_PREFIX = "model-comparison-reports"
DEFAULT_ARTIFACTS_PREFIX = "model-artifacts"


@task(multiple_outputs=True)
def compare_model_results(
    model_results: Dict[str, Dict[str, Any]],
    task_type: str = "regression",
    baseline_model: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    notify_slack: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare metrics across multiple models and generate a comparison report
    
    Args:
        model_results: Dictionary of model results, keyed by model_id
        task_type: Type of ML task ('regression')
        baseline_model: Model ID to use as baseline for comparison
        s3_bucket: S3 bucket to store comparison results (optional)
        s3_prefix: S3 prefix for comparison results (optional)
        notify_slack: Whether to send Slack notification with results
        
    Returns:
        Dictionary with comparison results and report paths
    """
    if not model_results:
        logger.warning("No model results provided for comparison")
        raise AirflowSkipException("No model results to compare")
    
    # Filter model results to include only Model1 and Model4
    model_results = {k: v for k, v in model_results.items() if k in ["Model1", "Model4"]}
    
    # Count number of models with complete results
    complete_models = sum(1 for result in model_results.values() 
                         if result.get("status") == "completed")
    
    if complete_models < 2:
        logger.warning(f"Only {complete_models} models completed successfully. Need at least 2 for comparison.")
        if complete_models == 1:
            # Just return the one model result without comparison
            model_id = next(model_id for model_id, result in model_results.items() 
                           if result.get("status") == "completed")
            return {
                "status": "single_model",
                "best_model": model_id,
                "comparison_performed": False,
                "model_results": model_results
            }
        else:
            raise AirflowSkipException("Less than 2 models completed successfully")
    
    # Create timestamp for this comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory for plots
    run_id = kwargs.get("run_id", timestamp)
    output_dir = f"/tmp/model_comparison_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare models
    logger.info(f"Comparing {complete_models} models")
    comparison_results = metrics.compare_models(
        model_results=model_results,
        task_type=task_type,
        baseline_model=baseline_model
    )
    
    # Save comparison results to JSON
    comparison_json_path = os.path.join(output_dir, "comparison_results.json")
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Create comparison plots
    logger.info("Generating comparison plots")
    plot_paths = metrics.create_comparison_plots(
        model_results=model_results,
        output_dir=output_dir,
        task_type=task_type
    )
    
    # Create summary report with key findings
    best_model = comparison_results.get("overall_best", {}).get("model_id")
    best_by_metric = comparison_results.get("best_by_metric", {})
    
    summary = {
        "timestamp": timestamp,
        "run_id": run_id,
        "num_models_compared": complete_models,
        "overall_best_model": best_model,
        "best_by_metric": best_by_metric,
        "generated_plots": list(plot_paths.keys()),
        "comparison_data_file": comparison_json_path,
    }
    
    # Get improvement over baseline if applicable
    if baseline_model and "improvements" in comparison_results:
        improvements = comparison_results["improvements"]
        if best_model in improvements:
            best_model_improvements = improvements[best_model]
            summary["baseline_comparison"] = {
                "baseline_model": baseline_model,
                "improvements": best_model_improvements
            }
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Upload to S3 if configured
    s3_locations = {}
    if s3_bucket:
        bucket = s3_bucket or DEFAULT_RESULTS_BUCKET
        prefix = s3_prefix or f"{DEFAULT_REPORTS_PREFIX}/{run_id}"
        
        # Upload summary
        summary_s3_key = f"{prefix}/comparison_summary.json"
        s3_utils.upload_to_s3(summary_path, bucket, summary_s3_key)
        s3_locations["summary"] = f"s3://{bucket}/{summary_s3_key}"
        
        # Upload plots
        for plot_name, plot_path in plot_paths.items():
            plot_filename = os.path.basename(plot_path)
            plot_s3_key = f"{prefix}/{plot_filename}"
            s3_utils.upload_to_s3(plot_path, bucket, plot_s3_key)
            s3_locations[plot_name] = f"s3://{bucket}/{plot_s3_key}"
        
        # Upload comparison results
        comparison_s3_key = f"{prefix}/comparison_results.json"
        s3_utils.upload_to_s3(comparison_json_path, bucket, comparison_s3_key)
        s3_locations["comparison_results"] = f"s3://{bucket}/{comparison_s3_key}"
    
    # Send notification if configured
    if notify_slack and best_model:
        try:
            webhook_url = Variable.get("slack_webhook_url", default_var=None)
            if webhook_url:
                message = f"Model Comparison Results\n"
                message += f"Best overall model: {best_model}\n"
                message += f"Models compared: {complete_models}\n"
                
                if "baseline_comparison" in summary:
                    improvements = summary["baseline_comparison"]["improvements"]
                    imp_text = ", ".join([f"{m}: {v:.2f}%" for m, v in improvements.items()])
                    message += f"Improvement over baseline: {imp_text}\n"
                
                if s3_locations:
                    message += f"Full report: {s3_locations.get('summary')}"
                
                notifications.send_slack_notification(webhook_url, message)
                logger.info("Slack notification sent with comparison results")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")
    
    # Return results
    return {
        "status": "success",
        "comparison_performed": True,
        "best_model": best_model,
        "output_dir": output_dir,
        "summary_path": summary_path,
        "plot_paths": plot_paths,
        "s3_locations": s3_locations,
        "model_results": model_results
    }


@task(multiple_outputs=True)
def identify_models_for_retraining(
    model_results: Dict[str, Dict[str, Any]],
    historical_metrics: Optional[Dict[str, List[Dict[str, float]]]] = None,
    threshold: float = 0.05,
    min_improvement: float = 0.02
) -> Dict[str, Any]:
    """
    Identify which models should be retrained based on performance
    
    Args:
        model_results: Dictionary of current model results
        historical_metrics: Dictionary of historical metrics by model ID
        threshold: Threshold for performance drop to trigger retraining
        min_improvement: Minimum improvement required to consider a retrain successful
        
    Returns:
        Dictionary with models to retrain and reasons
    """
    # If no historical metrics provided, try to load from MLflow or don't recommend any retrains
    if not historical_metrics:
        historical_metrics = {}
        try:
            # Try to get historical metrics from MLflow if available
            for model_id in model_results.keys():
                try:
                    runs = mlflow.search_runs(
                        filter_string=f"tags.model_id = '{model_id}'",
                        order_by=["start_time DESC"],
                        max_results=5
                    )
                    if not runs.empty:
                        # Extract metrics from each run
                        metrics_list = []
                        for _, run in runs.iterrows():
                            run_metrics = {}
                            for col in run.index:
                                if col.startswith("metrics."):
                                    metric_name = col.replace("metrics.", "")
                                    run_metrics[metric_name] = run[col]
                            if run_metrics:
                                metrics_list.append(run_metrics)
                        
                        if metrics_list:
                            historical_metrics[model_id] = metrics_list
                except Exception as e:
                    logger.warning(f"Failed to get historical metrics for {model_id}: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to load historical metrics: {str(e)}")
    
    # Check which models need retraining
    models_to_retrain = {}
    
    for model_id, result in model_results.items():
        if result.get("status") != "completed" or "metrics" not in result:
            # Skip models that didn't complete or don't have metrics
            continue
        
        current_metrics = result["metrics"]
        model_history = historical_metrics.get(model_id, [])
        
        # Skip if we're evaluating a model for the first time
        if not model_history:
            logger.info(f"No historical data for {model_id}, skipping retraining check")
            continue
        
        # Check if model should be retrained
        should_retrain_flag, reason = metrics.should_retrain(
            model_id=model_id,
            current_metrics=current_metrics,
            historical_metrics=model_history,
            threshold=threshold,
            min_improvement=min_improvement
        )
        
        if should_retrain_flag:
            models_to_retrain[model_id] = reason
    
    # Return results
    return {
        "models_to_retrain": models_to_retrain,
        "total_models": len(model_results),
        "retrain_count": len(models_to_retrain),
        "threshold_used": threshold,
        "min_improvement": min_improvement
    }


@task
def generate_comparison_report(
    comparison_results: Dict[str, Any],
    include_mlflow_links: bool = True,
    output_format: str = "markdown"
) -> str:
    """
    Generate a formatted report of model comparison results
    
    Args:
        comparison_results: Results from compare_model_results task
        include_mlflow_links: Whether to include MLflow links
        output_format: Output format ('markdown', 'html', or 'text')
        
    Returns:
        Formatted report string
    """
    if comparison_results.get("status") != "success":
        return f"Comparison status: {comparison_results.get('status', 'unknown')}"
    
    best_model = comparison_results.get("best_model")
    model_results = comparison_results.get("model_results", {})
    s3_locations = comparison_results.get("s3_locations", {})
    
    # Get MLflow tracking URI if available
    mlflow_uri = None
    try:
        mlflow_uri = mlflow.get_tracking_uri()
    except:
        pass
    
    # Generate report based on requested format
    if output_format == "markdown":
        report = "# Model Comparison Report\n\n"
        
        # Overall best model
        report += f"## Overall Best Model: {best_model}\n\n"
        
        # Performance metrics table
        report += "## Performance Metrics\n\n"
        report += "| Model | Status |"
        
        # Get all possible metrics from completed models
        all_metrics = set()
        for result in model_results.values():
            if result.get("status") == "completed" and "metrics" in result:
                all_metrics.update(result["metrics"].keys())
        
        # Add headers for metrics
        for metric in sorted(all_metrics):
            report += f" {metric} |"
        report += "\n"
        
        # Add separator row
        report += "| --- | --- |" + " --- |" * len(all_metrics) + "\n"
        
        # Add data rows
        for model_id, result in model_results.items():
            status = result.get("status", "unknown")
            report += f"| {model_id} | {status} |"
            
            if status == "completed" and "metrics" in result:
                metrics = result["metrics"]
                for metric in sorted(all_metrics):
                    value = metrics.get(metric, "N/A")
                    if isinstance(value, float):
                        report += f" {value:.4f} |"
                    else:
                        report += f" {value} |"
            else:
                report += " N/A |" * len(all_metrics)
            
            report += "\n"
        
        # MLflow links if available
        if include_mlflow_links and mlflow_uri:
            report += "\n## MLflow Experiment Tracking\n\n"
            for model_id in model_results.keys():
                # Create a direct link to MLflow results if possible
                if mlflow_uri.startswith("http"):
                    model_uri = f"{mlflow_uri}/#/experiments/0/s?searchFilter=tags.model_id%3D%22{model_id}%22"
                    report += f"- [{model_id} in MLflow]({model_uri})\n"
                else:
                    report += f"- {model_id}: Use MLflow UI to view results\n"
        
        # S3 links to plots and reports
        if s3_locations:
            report += "\n## Generated Reports and Plots\n\n"
            for name, location in s3_locations.items():
                if name in ["summary", "comparison_results"]:
                    report += f"- [{name.replace('_', ' ').title()}]({location})\n"
            
            report += "\n### Plots\n\n"
            for name, location in s3_locations.items():
                if name not in ["summary", "comparison_results"]:
                    report += f"- [{name.replace('_', ' ').title()}]({location})\n"
    
    elif output_format == "html":
        # Simplified HTML version
        report = "<h1>Model Comparison Report</h1>"
        report += f"<h2>Overall Best Model: {best_model}</h2>"
        
        # Performance metrics table
        report += "<h2>Performance Metrics</h2>"
        report += "<table border='1'><tr><th>Model</th><th>Status</th>"
        
        # Get all possible metrics
        all_metrics = set()
        for result in model_results.values():
            if result.get("status") == "completed" and "metrics" in result:
                all_metrics.update(result["metrics"].keys())
        
        # Add headers for metrics
        for metric in sorted(all_metrics):
            report += f"<th>{metric}</th>"
        report += "</tr>"
        
        # Add data rows
        for model_id, result in model_results.items():
            status = result.get("status", "unknown")
            report += f"<tr><td>{model_id}</td><td>{status}</td>"
            
            if status == "completed" and "metrics" in result:
                metrics = result["metrics"]
                for metric in sorted(all_metrics):
                    value = metrics.get(metric, "N/A")
                    if isinstance(value, float):
                        report += f"<td>{value:.4f}</td>"
                    else:
                        report += f"<td>{value}</td>"
            else:
                report += f"<td>N/A</td>" * len(all_metrics)
            
            report += "</tr>"
        
        report += "</table>"
        
        # S3 links
        if s3_locations:
            report += "<h2>Generated Reports and Plots</h2><ul>"
            for name, location in s3_locations.items():
                report += f"<li><a href='{location}'>{name.replace('_', ' ').title()}</a></li>"
            report += "</ul>"
    
    else:  # text format
        report = "MODEL COMPARISON REPORT\n"
        report += "=" * 23 + "\n\n"
        report += f"Overall Best Model: {best_model}\n\n"
        
        report += "Performance Metrics:\n"
        report += "-" * 20 + "\n"
        
        # Get all possible metrics
        all_metrics = set()
        for result in model_results.values():
            if result.get("status") == "completed" and "metrics" in result:
                all_metrics.update(result["metrics"].keys())
        
        # Model results
        for model_id, result in model_results.items():
            status = result.get("status", "unknown")
            report += f"{model_id} ({status}):\n"
            
            if status == "completed" and "metrics" in result:
                metrics = result["metrics"]
                for metric in sorted(all_metrics):
                    value = metrics.get(metric, "N/A")
                    if isinstance(value, float):
                        report += f"  {metric}: {value:.4f}\n"
                    else:
                        report += f"  {metric}: {value}\n"
            else:
                report += "  No metrics available\n"
            
            report += "\n"
        
        # S3 links
        if s3_locations:
            report += "Generated Reports and Plots:\n"
            report += "-" * 28 + "\n"
            for name, location in s3_locations.items():
                report += f"{name.replace('_', ' ').title()}: {location}\n"
    
    return report 


@task(multiple_outputs=True)
def compare_with_production_models(
    new_model_results: Dict[str, Dict[str, Any]],
    production_model_ids: List[str] = ["Model1", "Model4"],
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    notify_slack: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare new models with production models and raise a question if the new model performs better
    
    Args:
        new_model_results: Dictionary of new model results, keyed by model_id
        production_model_ids: List of production model IDs to compare against
        s3_bucket: S3 bucket to store comparison results (optional)
        s3_prefix: S3 prefix for comparison results (optional)
        notify_slack: Whether to send Slack notification with results
        
    Returns:
        Dictionary with comparison results and report paths
    """
    if not new_model_results:
        logger.warning("No new model results provided for comparison")
        raise AirflowSkipException("No new model results to compare")
    
    # Load production model results from MLflow
    production_model_results = {}
    for model_id in production_model_ids:
        try:
            runs = mlflow.search_runs(
                filter_string=f"tags.model_id = '{model_id}'",
                order_by=["start_time DESC"],
                max_results=1
            )
            if not runs.empty:
                run = runs.iloc[0]
                metrics = {col.replace("metrics.", ""): run[col] for col in run.index if col.startswith("metrics.")}
                production_model_results[model_id] = {
                    "model_id": model_id,
                    "run_id": run["run_id"],
                    "metrics": metrics,
                    "status": "completed"
                }
        except Exception as e:
            logger.warning(f"Failed to get production model results for {model_id}: {str(e)}")
    
    if not production_model_results:
        logger.warning("No production model results found")
        raise AirflowSkipException("No production model results to compare")
    
    # Combine new and production model results
    combined_results = {**new_model_results, **production_model_results}
    
    # Compare models
    logger.info(f"Comparing new models with production models")
    comparison_results = metrics.compare_models(
        model_results=combined_results,
        task_type="regression"
    )
    
    # Save comparison results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = kwargs.get("run_id", timestamp)
    output_dir = f"/tmp/model_comparison_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_json_path = os.path.join(output_dir, "comparison_results.json")
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Create comparison plots
    logger.info("Generating comparison plots")
    plot_paths = metrics.create_comparison_plots(
        model_results=combined_results,
        output_dir=output_dir,
        task_type="regression"
    )
    
    # Create summary report with key findings
    best_model = comparison_results.get("overall_best", {}).get("model_id")
    best_by_metric = comparison_results.get("best_by_metric", {})
    
    summary = {
        "timestamp": timestamp,
        "run_id": run_id,
        "num_models_compared": len(combined_results),
        "overall_best_model": best_model,
        "best_by_metric": best_by_metric,
        "generated_plots": list(plot_paths.keys()),
        "comparison_data_file": comparison_json_path,
    }
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Upload to S3 if configured
    s3_locations = {}
    if s3_bucket:
        bucket = s3_bucket or DEFAULT_RESULTS_BUCKET
        prefix = s3_prefix or f"{DEFAULT_REPORTS_PREFIX}/{run_id}"
        
        # Upload summary
        summary_s3_key = f"{prefix}/comparison_summary.json"
        s3_utils.upload_to_s3(summary_path, bucket, summary_s3_key)
        s3_locations["summary"] = f"s3://{bucket}/{summary_s3_key}"
        
        # Upload plots
        for plot_name, plot_path in plot_paths.items():
            plot_filename = os.path.basename(plot_path)
            plot_s3_key = f"{prefix}/{plot_filename}"
            s3_utils.upload_to_s3(plot_path, bucket, plot_s3_key)
            s3_locations[plot_name] = f"s3://{bucket}/{plot_s3_key}"
        
        # Upload comparison results
        comparison_s3_key = f"{prefix}/comparison_results.json"
        s3_utils.upload_to_s3(comparison_json_path, bucket, comparison_s3_key)
        s3_locations["comparison_results"] = f"s3://{bucket}/{comparison_s3_key}"
    
    # Send notification if configured
    if notify_slack and best_model:
        try:
            webhook_url = Variable.get("slack_webhook_url", default_var=None)
            if webhook_url:
                message = f"Model Comparison Results\n"
                message += f"Best overall model: {best_model}\n"
                message += f"Models compared: {len(combined_results)}\n"
                
                if s3_locations:
                    message += f"Full report: {s3_locations.get('summary')}"
                
                notifications.send_slack_notification(webhook_url, message)
                logger.info("Slack notification sent with comparison results")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")
    
    # Raise a question if the new model performs better than the production models
    if best_model in new_model_results:
        logger.info(f"New model {best_model} performs better than production models")
        # Here you can add logic to raise a question or notify a human for approval
    
    # Return results
    return {
        "status": "success",
        "comparison_performed": True,
        "best_model": best_model,
        "output_dir": output_dir,
        "summary_path": summary_path,
        "plot_paths": plot_paths,
        "s3_locations": s3_locations,
        "model_results": combined_results
    }
