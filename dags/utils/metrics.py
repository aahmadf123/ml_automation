#!/usr/bin/env python3
"""
metrics.py - Utilities for model metrics and comparison
------------------------------------------------------
This module provides functions for:
1. Computing standard ML metrics
2. Comparing multiple models performance
3. Generating visualization for model comparison
4. Determining if models should be retrained
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
import mlflow
import logging
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

# Define metric categories
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REGRESSION_METRICS = ["rmse", "mae", "r2"]
THRESHOLD_METRICS = ["accuracy", "precision", "recall", "f1"]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                      y_proba: Optional[np.ndarray] = None, 
                      task_type: str = "classification") -> Dict[str, float]:
    """
    Calculate metrics for model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (or values for regression)
        y_proba: Predicted probabilities (for classification)
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if task_type == "classification":
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="weighted")
        metrics["recall"] = recall_score(y_true, y_pred, average="weighted")
        metrics["f1"] = f1_score(y_true, y_pred, average="weighted")
        
        # ROC AUC if probabilities are provided
        if y_proba is not None:
            if y_proba.shape[1] == 2:  # Binary classification
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:  # Multi-class
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba, average="weighted", multi_class="ovr")
    
    elif task_type == "regression":
        # Regression metrics
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
    
    return metrics


def compare_models(model_results: Dict[str, Dict[str, Any]], 
                   task_type: str = "classification",
                   baseline_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare metrics across multiple models
    
    Args:
        model_results: Dictionary of model results, each containing metrics
        task_type: 'classification' or 'regression'
        baseline_model: Model ID to use as baseline for comparison
        
    Returns:
        Dictionary with comparison results
    """
    # Get relevant metrics based on task type
    metrics_list = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
    
    # Extract metrics from all models
    metrics_by_model = {}
    for model_id, result in model_results.items():
        if result.get("status") == "completed" and "metrics" in result:
            metrics_by_model[model_id] = result["metrics"]
    
    if not metrics_by_model:
        logger.warning("No completed models with metrics found")
        return {"status": "no_data"}
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(metrics_by_model).T
    
    # If baseline model is provided, calculate improvements
    improvements = {}
    if baseline_model and baseline_model in metrics_by_model:
        baseline_metrics = metrics_by_model[baseline_model]
        
        for model_id, metrics in metrics_by_model.items():
            if model_id != baseline_model:
                model_improvements = {}
                
                for metric in metrics_list:
                    if metric in metrics and metric in baseline_metrics:
                        # Higher is better for most metrics except error metrics
                        if metric in ["rmse", "mae"]:
                            # Lower is better for error metrics
                            pct_change = ((baseline_metrics[metric] - metrics[metric]) / 
                                         baseline_metrics[metric]) * 100
                        else:
                            # Higher is better
                            pct_change = ((metrics[metric] - baseline_metrics[metric]) / 
                                         baseline_metrics[metric]) * 100
                        
                        model_improvements[metric] = pct_change
                
                improvements[model_id] = model_improvements
    
    # Find best model for each metric
    best_models = {}
    for metric in metrics_list:
        if metric in comparison_df.columns:
            if metric in ["rmse", "mae"]:  # Lower is better
                best_idx = comparison_df[metric].idxmin()
                best_val = comparison_df[metric].min()
            else:  # Higher is better
                best_idx = comparison_df[metric].idxmax()
                best_val = comparison_df[metric].max()
            
            best_models[metric] = {"model_id": best_idx, "value": best_val}
    
    # Determine overall best model using a scoring system
    scores = {}
    for model_id in metrics_by_model.keys():
        score = 0
        for metric in metrics_list:
            if metric in comparison_df.columns:
                # Get rank for this metric (1 = best)
                if metric in ["rmse", "mae"]:  # Lower is better
                    ranks = comparison_df[metric].rank()
                else:  # Higher is better
                    ranks = comparison_df[metric].rank(ascending=False)
                
                # Add score based on rank (1st = n points, 2nd = n-1 points, etc.)
                score += len(metrics_by_model) + 1 - ranks[model_id]
        
        scores[model_id] = score
    
    # Find model with highest score
    if scores:
        best_overall = max(scores.items(), key=lambda x: x[1])
        overall_winner = {
            "model_id": best_overall[0],
            "score": best_overall[1],
            "total_possible": len(metrics_list) * len(metrics_by_model)
        }
    else:
        overall_winner = None
    
    return {
        "status": "success",
        "comparison_data": comparison_df.to_dict(),
        "best_by_metric": best_models,
        "overall_best": overall_winner,
        "improvements": improvements
    }


def create_comparison_plots(model_results: Dict[str, Dict[str, Any]], 
                           output_dir: str,
                           task_type: str = "classification") -> Dict[str, str]:
    """
    Generate comparison plots for multiple models
    
    Args:
        model_results: Dictionary of model results, each containing metrics
        output_dir: Directory to save plots
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with paths to generated plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metrics to compare based on task type
    metrics_list = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
    
    # Extract metrics from all models
    metrics_by_model = {}
    for model_id, result in model_results.items():
        if result.get("status") == "completed" and "metrics" in result:
            metrics_by_model[model_id] = result["metrics"]
    
    if not metrics_by_model:
        logger.warning("No completed models with metrics found")
        return {}
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(metrics_by_model).T
    
    # Dictionary to store plot paths
    plot_paths = {}
    
    # 1. Bar chart for each metric
    for metric in metrics_list:
        if metric in comparison_df.columns:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=comparison_df.index, y=comparison_df[metric])
            plt.title(f"Comparison of {metric.upper()} across models")
            plt.ylabel(metric.upper())
            plt.xlabel("Model")
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(comparison_df[metric]):
                ax.text(i, v, f"{v:.4f}", ha='center', va='bottom')
            
            # Save plot
            plot_path = os.path.join(output_dir, f"compare_{metric}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths[f"{metric}_comparison"] = plot_path
    
    # 2. Radar/Spider chart for comparing models across metrics
    # Normalize metrics to 0-1 scale for comparison
    normalized_df = comparison_df.copy()
    for col in normalized_df.columns:
        if col in ["rmse", "mae"]:  # Lower is better
            normalized_df[col] = (normalized_df[col].max() - normalized_df[col]) / (normalized_df[col].max() - normalized_df[col].min())
        else:  # Higher is better
            normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
    
    # Replace NaNs with 0
    normalized_df = normalized_df.fillna(0)
    
    # Create radar chart
    metrics_in_df = [m for m in metrics_list if m in normalized_df.columns]
    if len(metrics_in_df) >= 3:  # Need at least 3 metrics for a radar chart
        n_metrics = len(metrics_in_df)
        angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for model in normalized_df.index:
            values = normalized_df.loc[model, metrics_in_df].tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_in_df)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
        plt.title("Model Comparison Across Metrics")
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        radar_path = os.path.join(output_dir, "model_comparison_radar.png")
        plt.tight_layout()
        plt.savefig(radar_path)
        plt.close()
        
        plot_paths["radar_comparison"] = radar_path
    
    # 3. Heatmap of model performance
    plt.figure(figsize=(12, 8))
    sns.heatmap(comparison_df, annot=True, cmap="viridis", fmt=".4f", linewidths=.5)
    plt.title("Model Performance Heatmap")
    plt.ylabel("Model")
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, "model_comparison_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    
    plot_paths["heatmap_comparison"] = heatmap_path
    
    # Save comparison data as JSON for future reference
    json_path = os.path.join(output_dir, "model_comparison.json")
    with open(json_path, 'w') as f:
        json.dump({
            "comparison_data": comparison_df.to_dict(),
            "metrics_compared": metrics_list,
            "models_compared": list(metrics_by_model.keys()),
        }, f, indent=2)
    
    plot_paths["comparison_data"] = json_path
    
    return plot_paths


def should_retrain(model_id: str, 
                  current_metrics: Dict[str, float], 
                  historical_metrics: List[Dict[str, float]],
                  threshold: float = 0.05,
                  min_improvement: float = 0.02) -> Tuple[bool, str]:
    """
    Determine if a model should be retrained based on historical performance
    
    Args:
        model_id: Identifier for the model
        current_metrics: Current model metrics
        historical_metrics: List of historical metrics (latest first)
        threshold: Threshold for performance drop to trigger retraining
        min_improvement: Minimum improvement required to consider a retrain successful
        
    Returns:
        (should_retrain, reason)
    """
    if not historical_metrics:
        logger.info(f"No historical metrics for {model_id}, retraining recommended")
        return True, "No historical data available"
    
    # Get the latest historical metrics
    latest_historical = historical_metrics[0]
    
    # Determine the task type based on available metrics
    if "accuracy" in latest_historical:
        task_type = "classification"
        primary_metric = "f1" if "f1" in latest_historical else "accuracy"
        direction = "higher"  # Higher is better
    elif "rmse" in latest_historical:
        task_type = "regression"
        primary_metric = "r2" if "r2" in latest_historical else "rmse"
        direction = "higher" if primary_metric == "r2" else "lower"
    else:
        logger.warning(f"Cannot determine task type for {model_id}, using default metrics")
        primary_metric = list(latest_historical.keys())[0]
        direction = "higher"  # Assume higher is better by default
    
    # Check if primary metric exists in both current and historical
    if primary_metric not in current_metrics or primary_metric not in latest_historical:
        logger.warning(f"Primary metric {primary_metric} not found, using available metrics")
        # Use an available metric that's in both
        common_metrics = set(current_metrics.keys()).intersection(set(latest_historical.keys()))
        if not common_metrics:
            logger.error(f"No common metrics between current and historical data for {model_id}")
            return True, "No common metrics with historical data"
        
        primary_metric = list(common_metrics)[0]
        # Determine direction
        direction = "lower" if primary_metric in ["rmse", "mae"] else "higher"
    
    # Calculate the performance change
    current_value = current_metrics[primary_metric]
    historical_value = latest_historical[primary_metric]
    
    if direction == "higher":
        pct_change = (current_value - historical_value) / historical_value
        performance_dropped = pct_change < -threshold
    else:  # lower is better
        pct_change = (historical_value - current_value) / historical_value
        performance_dropped = pct_change < -threshold
    
    # Check for performance drop
    if performance_dropped:
        reason = f"{primary_metric} dropped by {abs(pct_change)*100:.2f}% (threshold: {threshold*100:.1f}%)"
        logger.info(f"Retraining recommended for {model_id}: {reason}")
        return True, reason
    
    # Check if we have enough historical data to analyze trends
    if len(historical_metrics) >= 3:
        # Look for consistent degradation over last 3 runs
        trend_values = [current_metrics[primary_metric]] + [h[primary_metric] for h in historical_metrics[:2]]
        
        if direction == "higher":
            is_degrading = all(trend_values[i] < trend_values[i+1] for i in range(2))
        else:  # lower is better
            is_degrading = all(trend_values[i] > trend_values[i+1] for i in range(2))
        
        if is_degrading:
            reason = f"Consistent degradation in {primary_metric} over last 3 runs"
            logger.info(f"Retraining recommended for {model_id}: {reason}")
            return True, reason
    
    # Default: no need to retrain
    logger.info(f"No retraining needed for {model_id}, performance is stable")
    return False, "Performance is stable" 