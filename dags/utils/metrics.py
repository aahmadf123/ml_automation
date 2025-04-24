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
    mean_squared_error, mean_absolute_error, r2_score
)
import mlflow
import logging
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

# Define metric categories
REGRESSION_METRICS = ["rmse", "mae", "r2"]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for model evaluation
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Regression metrics
    metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)
    
    return metrics


def compare_models(model_results: Dict[str, Dict[str, Any]],
                   baseline_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare metrics across multiple models
    
    Args:
        model_results: Dictionary of model results, each containing metrics
        baseline_model: Model ID to use as baseline for comparison
        
    Returns:
        Dictionary with comparison results
    """
    # Get relevant metrics
    metrics_list = REGRESSION_METRICS
    
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
                        # Lower is better for error metrics
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
                           output_dir: str) -> Dict[str, str]:
    """
    Generate comparison plots for multiple models
    
    Args:
        model_results: Dictionary of model results, each containing metrics
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with paths to generated plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metrics to compare
    metrics_list = REGRESSION_METRICS
    
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
    
    # Create bar plot for each metric
    for metric in metrics_list:
        if metric in comparison_df.columns:
            plt.figure(figsize=(12, 6))
            
            # Sort by metric value
            if metric in ["rmse", "mae"]:  # Lower is better
                sorted_df = comparison_df.sort_values(by=metric, ascending=True)
            else:  # Higher is better
                sorted_df = comparison_df.sort_values(by=metric, ascending=False)
            
            # Create bar plot
            ax = sns.barplot(x=sorted_df.index, y=sorted_df[metric])
            
            # Add value labels on top of each bar
            for i, v in enumerate(sorted_df[metric]):
                ax.text(i, v, f"{v:.4f}", ha='center', va='bottom')
            
            # Set title and labels
            plt.title(f"{metric.upper()} Comparison Across Models")
            plt.xlabel("Model")
            plt.ylabel(metric.upper())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(output_dir, f"comparison_{metric}.png")
            plt.savefig(output_path)
            plt.close()
            
            # Add to result dictionary
            plot_paths[metric] = output_path
    
    # Create a summary bar plot with all metrics
    try:
        plt.figure(figsize=(15, 8))
        
        # Normalize metric values for comparison (0-1 scale)
        normalized_df = comparison_df.copy()
        for metric in metrics_list:
            if metric in normalized_df.columns:
                if metric in ["rmse", "mae"]:  # Lower is better
                    # Invert so that higher is better for all metrics
                    max_val = normalized_df[metric].max()
                    min_val = normalized_df[metric].min()
                    if max_val > min_val:  # Avoid division by zero
                        normalized_df[metric] = 1 - ((normalized_df[metric] - min_val) / (max_val - min_val))
                    else:
                        normalized_df[metric] = 1.0
                else:  # Higher is better
                    max_val = normalized_df[metric].max()
                    min_val = normalized_df[metric].min()
                    if max_val > min_val:  # Avoid division by zero
                        normalized_df[metric] = (normalized_df[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_df[metric] = 1.0
        
        # Melt the dataframe for easier plotting
        melted_df = pd.melt(normalized_df.reset_index(), 
                           id_vars='index', 
                           value_vars=[m for m in metrics_list if m in normalized_df.columns],
                           var_name='Metric', 
                           value_name='Normalized Score')
        
        # Plot
        ax = sns.barplot(x='index', y='Normalized Score', hue='Metric', data=melted_df)
        
        # Set title and labels
        plt.title("Normalized Metric Comparison Across Models")
        plt.xlabel("Model")
        plt.ylabel("Normalized Score (higher is better)")
        plt.legend(title="Metric")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, "comparison_normalized.png")
        plt.savefig(output_path)
        plt.close()
        
        # Add to result dictionary
        plot_paths["normalized"] = output_path
    except Exception as e:
        logger.warning(f"Error creating normalized comparison plot: {str(e)}")
    
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
    
    # For regression models
    task_type = "regression"
    primary_metric = "r2" if "r2" in latest_historical else "rmse"
    direction = "higher" if primary_metric == "r2" else "lower"
    
    # Compare metrics
    if primary_metric not in current_metrics or primary_metric not in latest_historical:
        logger.warning(f"Primary metric {primary_metric} not available for comparison")
        return False, f"Metric {primary_metric} not available"
    
    current_value = current_metrics[primary_metric]
    historical_value = latest_historical[primary_metric]
    
    # Calculate performance change
    if direction == "higher":
        # Higher is better (e.g., R2)
        change = (current_value - historical_value) / max(abs(historical_value), 1e-10)
        significant_drop = change < -threshold
        significant_improvement = change > min_improvement
    else:
        # Lower is better (e.g., RMSE)
        change = (historical_value - current_value) / max(abs(historical_value), 1e-10)
        significant_drop = change < -threshold
        significant_improvement = change > min_improvement
    
    if significant_drop:
        reason = f"{primary_metric} dropped by {abs(change)*100:.2f}% ({historical_value:.4f} -> {current_value:.4f})"
        logger.info(f"Retraining recommended for {model_id}: {reason}")
        return True, reason
    elif significant_improvement:
        reason = f"{primary_metric} improved by {change*100:.2f}% ({historical_value:.4f} -> {current_value:.4f})"
        logger.info(f"Model {model_id} shows significant improvement: {reason}")
        return False, reason
    else:
        reason = f"{primary_metric} stable: {change*100:.2f}% change ({historical_value:.4f} -> {current_value:.4f})"
        logger.info(f"No retraining needed for {model_id}: {reason}")
        return False, reason 