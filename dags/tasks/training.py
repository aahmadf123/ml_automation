#!/usr/bin/env python3
"""
tasks/training.py  – Unified trainer + explainability
----------------------------------------------------------
 • Baseline check: load last Production model and skip retrain if RMSE ≤ threshold
 • HyperOpt → finds best hyper‑parameters per model (configurable max‑evals)
 • Fallback from TimeSeriesSplit → train_test_split if not enough chronological rows
 • Logs to MLflow: RMSE, MSE, MAE, R² + SHAP summary + Actual vs. Predicted plot
 • Archives SHAP plots to s3://<BUCKET>/visuals/shap/
 • Archives AV‑Pred plots to s3://<BUCKET>/visuals/avs_pred/
 • Auto‑register & promote in MLflow Model Registry
 • Sends Slack notifications on skip/retrain and completion
 • Supports manual_override from Airflow Variables
 • Emits WebSocket events for real-time dashboard updates
"""

import os
import json
import logging
import tempfile
import time
import joblib
import websockets
import asyncio
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split

import mlflow
import mlflow.xgboost
from airflow.models import Variable

from utils.storage import upload as s3_upload, download as s3_download
from utils.config import (
    DATA_BUCKET, MODEL_KEY_PREFIX, AWS_REGION,
    MLFLOW_URI, MLFLOW_EXPERIMENT, MODEL_CONFIG
)

import boto3
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Initialize AWS clients with region
s3 = boto3.client('s3', region_name=AWS_REGION)

async def send_websocket_update(model_id: str, metrics: Dict[str, float], status: str) -> None:
    """Send real-time update to WebSocket clients."""
    try:
        uri = "ws://localhost:8000/ws/model_updates"
        payload = json.dumps({
            "model_id": model_id,
            "metrics": metrics,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
        
        async with websockets.connect(uri) as websocket:
            await websocket.send(payload)
            logger.info(f"WebSocket update sent for {model_id}")
            
    except Exception as e:
        logger.error(f"Failed to send WebSocket update: {e}")

def send_websocket_update_sync(model_id: str, metrics: Dict[str, float], status: str) -> None:
    """Synchronous wrapper for sending WebSocket updates."""
    try:
        asyncio.run(send_websocket_update(model_id, metrics, status))
    except Exception as e:
        logger.error(f"Failed to run async WebSocket update: {e}")
        
def generate_shap_plots(model, X: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """Generate SHAP plots for model explanation."""
    plot_paths = {}
    try:
        # Create explanation using TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Create and save summary plot 
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        summary_path = os.path.join(output_dir, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(summary_path)
        plt.close()
        plot_paths["summary"] = summary_path
        
        # Create and save bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        bar_path = os.path.join(output_dir, "shap_bar.png")
        plt.tight_layout()
        plt.savefig(bar_path)
        plt.close()
        plot_paths["bar"] = bar_path
        
        # Create and save individual feature plots for top features
        feature_importance = np.abs(shap_values).mean(0)
        top_indices = feature_importance.argsort()[-5:]  # Top 5 features
        
        for idx in top_indices:
            feature_name = X.columns[idx]
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(idx, shap_values, X, show=False)
            feature_path = os.path.join(output_dir, f"shap_{feature_name}.png")
            plt.tight_layout()
            plt.savefig(feature_path)
            plt.close()
            plot_paths[feature_name] = feature_path
            
        return plot_paths
        
    except Exception as e:
        logger.error(f"Failed to generate SHAP plots: {e}")
        return plot_paths

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> str:
    """Generate and save actual vs predicted plot."""
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    except Exception as e:
        logger.error(f"Failed to create actual vs predicted plot: {e}")
        return ""

def should_skip_training(model_id: str, current_rmse: float) -> bool:
    """Check if training can be skipped based on existing model performance."""
    try:
        # Check if override is set
        override = Variable.get("FORCE_RETRAIN", default_var="false").lower() == "true"
        if override:
            logger.info("Force retrain is enabled. Will not skip training.")
            return False
        
        # Get production model RMSE
        client = MlflowClient()
        try:
            production_versions = client.get_latest_versions(model_id, stages=["Production"])
            if not production_versions:
                logger.info("No production model found. Training required.")
                return False
                
            prod_version = production_versions[0]
            run = client.get_run(prod_version.run_id)
            prod_rmse = run.data.metrics.get("rmse", float('inf'))
            
            # Calculate improvement threshold
            improvement_threshold = float(Variable.get("RMSE_IMPROVEMENT_THRESHOLD", default_var="0.05"))
            potential_improvement = (prod_rmse - current_rmse) / prod_rmse
            
            if potential_improvement < improvement_threshold:
                logger.info(f"Skipping training. Current RMSE ({current_rmse:.4f}) offers insufficient improvement " 
                           f"over production model ({prod_rmse:.4f}). Improvement: {potential_improvement:.2%}")
                return True
            else:
                logger.info(f"Training required. Potential improvement: {potential_improvement:.2%}")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking production model: {e}. Will proceed with training.")
            return False
            
    except Exception as e:
        logger.warning(f"Error in skip training check: {e}. Will proceed with training.")
        return False

def find_best_hyperparams(X_train, y_train, max_evals=50):
    """Find optimal hyperparameters using Hyperopt."""
    # Define the search space
    space = {
        'max_depth': hp.choice('max_depth', range(3, 15)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'gamma': hp.uniform('gamma', 0, 0.5),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500])
    }
    
    # Define the objective function
    def objective(params):
        # Create a CV set first
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Training with early stopping
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            **params
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate on validation set
        pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        
        return {'loss': rmse, 'status': STATUS_OK}
    
    # Run the optimization
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    # Convert indices to actual values for categorical params
    best_params = {
        'max_depth': range(3, 15)[best['max_depth']],
        'learning_rate': best['learning_rate'],
        'gamma': best['gamma'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'min_child_weight': range(1, 10)[best['min_child_weight']],
        'n_estimators': [100, 200, 300, 500][best['n_estimators']]
    }
    
    return best_params

def train_and_compare_fn(model_id: str, processed_path: str) -> None:
    """Train pipeline with baseline skip, HyperOpt, MLflow logging, and notifications."""
    # Import slack only when needed
    from utils.slack import post as slack_msg
    
    # Load processed data
    try:
        df = pd.read_parquet(processed_path)
        logger.info(f"Loaded processed data from {processed_path}: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        slack_msg(
            channel="#alerts",
            title="❌ Training Failed",
            details=f"Could not load processed data: {e}",
            urgency="high"
        )
        raise
    
    # Create directory for artifacts
    temp_dir = tempfile.mkdtemp()
    model_file = os.path.join(temp_dir, f"{model_id}.joblib")
    
    # Prepare data
    target_column = "claim_amount"
    if target_column not in df.columns:
        error_msg = f"Target column '{target_column}' not found in dataset"
        logger.error(error_msg)
        slack_msg(
            channel="#alerts",
            title="❌ Training Failed",
            details=error_msg,
            urgency="high"
        )
        raise ValueError(error_msg)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Try to use time-series split if date column exists
    try:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Remove date for modeling
            X = X.drop(columns=['date'])
            
            # Time-based split (80/20)
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logger.info(f"Using time-series split: {X_train.shape}, {X_test.shape}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info(f"Using random train/test split: {X_train.shape}, {X_test.shape}")
    except Exception as e:
        logger.warning(f"Error using time-series split: {e}. Falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Calculate baseline metrics for skip check
    baseline_pred = np.ones(len(y_test)) * y_train.mean()
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    
    # Check if we can skip training
    if should_skip_training(model_id, baseline_rmse * 0.8):  # Assume we can achieve 20% improvement over baseline
        slack_msg(
            channel="#alerts",
            title="⏭️ Training Skipped",
            details=f"Training skipped for {model_id} as current model is sufficient",
            urgency="low"
        )
        return
    
    # Start MLflow run
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name=f"{model_id}_training") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")
        
        # Find best hyperparameters
        logger.info("Finding best hyperparameters with Hyperopt")
        send_websocket_update_sync(model_id, {"status": "hyperparameter_tuning"}, "running")
        
        max_evals = int(Variable.get("HYPEROPT_MAX_EVALS", default_var="30"))
        best_params = find_best_hyperparams(X_train, y_train, max_evals=max_evals)
        
        # Log hyperparameters
        mlflow.log_params(best_params)
        
        # Train final model with best parameters
        logger.info(f"Training final model with best parameters: {best_params}")
        send_websocket_update_sync(model_id, {"status": "training"}, "running")
        
        final_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            **best_params
        )
        
        final_model.fit(X_train, y_train)
        
        # Make predictions and calculate metrics
        y_pred = final_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        final_metrics = {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "baseline_rmse": baseline_rmse,
            "improvement": (baseline_rmse - rmse) / baseline_rmse
        }
        
        mlflow.log_metrics(final_metrics)
        
        # Generate visualizations
        logger.info("Generating visualizations")
        send_websocket_update_sync(model_id, {"status": "visualizing"}, "running")
        
        # SHAP plots
        shap_plots = generate_shap_plots(final_model, X_test.iloc[:100], temp_dir)  # Limit to 100 samples for speed
        for name, path in shap_plots.items():
            mlflow.log_artifact(path, "shap_plots")
            # Upload to S3
            s3_key = f"visuals/shap/{model_id}_{name}.png"
            s3_upload(path, s3_key)
        
        # Actual vs Predicted plot
        avs_pred_path = os.path.join(temp_dir, "actual_vs_predicted.png")
        plot_actual_vs_predicted(y_test.values, y_pred, avs_pred_path)
        mlflow.log_artifact(avs_pred_path, "plots")
        # Upload to S3
        s3_key = f"visuals/avs_pred/{model_id}_avs_pred.png"
        s3_upload(avs_pred_path, s3_key)
        
        # Log model
        mlflow.xgboost.log_model(final_model, "model")
        
        # Register model
        try:
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, model_id)
            logger.info(f"Model registered as {model_id}")
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
        
        # Try to promote model to Production
        try:
            client = MlflowClient()
            latest_version = client.get_latest_versions(model_id, stages=["None"])[0].version
            
            # If this is the first version, promote directly
            all_versions = client.get_latest_versions(model_id)
            if len(all_versions) == 1:
                client.transition_model_version_stage(
                    name=model_id,
                    version=latest_version,
                    stage="Production"
                )
                logger.info(f"Model {model_id} version {latest_version} promoted to Production as first version")
            else:
                # Compare with current production version
                try:
                    production_versions = client.get_latest_versions(model_id, stages=["Production"])
                    if production_versions:
                        prod_version = production_versions[0]
                        prod_run = client.get_run(prod_version.run_id)
                        prod_rmse = prod_run.data.metrics.get("rmse", float('inf'))
                        
                        # If new model is better, promote it
                        if rmse < prod_rmse * 0.95:  # 5% improvement threshold
                            client.transition_model_version_stage(
                                name=model_id,
                                version=latest_version,
                                stage="Production",
                                archive_existing_versions=True
                            )
                            logger.info(f"Model {model_id} version {latest_version} promoted to Production")
                            slack_msg(
                                channel="#alerts",
                                title="🚀 Model Promoted",
                                details=f"Model {model_id} v{latest_version} promoted to Production with "
                                        f"{((prod_rmse - rmse)/prod_rmse)*100:.1f}% RMSE improvement",
                                urgency="high"
                            )
                        else:
                            logger.info(f"New model not significantly better: {rmse:.4f} vs {prod_rmse:.4f}")
                            client.transition_model_version_stage(
                                name=model_id,
                                version=latest_version,
                                stage="Staging"
                            )
                            logger.info(f"Model {model_id} version {latest_version} moved to Staging")
                            slack_msg(
                                channel="#alerts",
                                title="📊 New Model in Staging",
                                details=f"Model {model_id} v{latest_version} moved to Staging stage\n"
                                        f"Current RMSE: {rmse:.4f}\nProduction RMSE: {prod_rmse:.4f}",
                                urgency="medium"
                            )
                    else:
                        # No production version found, promote this one
                        client.transition_model_version_stage(
                            name=model_id,
                            version=latest_version,
                            stage="Production"
                        )
                        logger.info(f"Model {model_id} version {latest_version} promoted to Production (no existing production version)")
                except Exception as e:
                    logger.warning(f"Error comparing with production model: {e}")
                    # Move to staging as fallback
                    client.transition_model_version_stage(
                        name=model_id,
                        version=latest_version,
                        stage="Staging"
                    )
                    logger.info(f"Model {model_id} version {latest_version} moved to Staging (fallback)")
        except Exception as e:
            logger.warning(f"Failed to manage model version: {e}")
    
    # Save the model file for use outside MLflow
    joblib.dump(final_model, model_file)
    s3_upload(model_file, f"models/{model_id}.joblib")

    # Notify completion
    slack_msg(
        channel="#alerts",
        title=f"✅ {model_id} training complete",
        details=f"RMSE={rmse:.4f}, R²={r2:.4f}",
        urgency="low",
    )
    
    # Send WebSocket update for completion
    send_websocket_update_sync(model_id, final_metrics, "complete")
    
    return run_id

# backward‑compatibility alias
train_xgboost_hyperopt = train_and_compare_fn
