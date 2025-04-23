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
    """
    Train and compare ML models with optimized data loading using pyarrow.
    
    Args:
        model_id: Identifier for the model
        processed_path: Path to the processed data file
    """
    # Import slack only when needed
    from utils.slack import post as send_message
    
    run_id = None
    model = None
    start_time = time.time()
    metrics = {}
    
    try:
        # Set up MLflow tracking
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info(f"MLflow tracking setup complete. URI: {MLFLOW_URI}, Experiment: {MLFLOW_EXPERIMENT}")
        
        # Load the data using pyarrow optimizations
        logger.info(f"Loading data from {processed_path} with pyarrow optimizations")
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.dataset as ds
        import gc
        
        # Create dataset with multi-threading
        dataset = ds.dataset(processed_path, format="parquet")
        
        # Scan the dataset efficiently
        scanner = ds.Scanner.from_dataset(
            dataset,
            use_threads=True
        )
        
        # Load as table and convert to pandas
        table = scanner.to_table()
        df = table.to_pandas()
        
        # Free memory
        del table
        gc.collect()
        
        logger.info(f"Data loaded, shape={df.shape}")
        
        # Prepare features and target
        try:
            target_col = "claim_amount"
            X = df.drop(columns=[target_col], errors='ignore')
            y = df[target_col]
        except KeyError as e:
            logger.error(f"KeyError while preparing features: {e}")
            raise
            
        # Free memory by removing original dataframe
        del df
        gc.collect()
        
        # Split the data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info(f"Data split complete: train={X_train.shape}, test={X_test.shape}")
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
            
        # Calculate preliminary RMSE to check if we need to train
        baseline_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        baseline_model.fit(X_train, y_train)
        baseline_preds = baseline_model.predict(X_test)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
        
        logger.info(f"Baseline RMSE: {baseline_rmse:.4f}")
        
        # Check if we should skip training
        if should_skip_training(model_id, baseline_rmse):
            send_message(
                channel="#ml-updates",
                title="🔄 Training Skipped",
                details=f"Skipped training for {model_id}. Current RMSE: {baseline_rmse:.4f}",
                urgency="low"
            )
            metrics = {
                "rmse": float(baseline_rmse),
                "status": "skipped",
                "timestamp": datetime.now().isoformat()
            }
            send_websocket_update_sync(model_id, metrics, "skipped")
            return None
            
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            # Find optimal hyperparameters
            logger.info("Finding optimal hyperparameters...")
            max_evals = MODEL_CONFIG.get('max_evals', 30)
            best_params = find_best_hyperparams(X_train, y_train, max_evals=max_evals)
            mlflow.log_params(best_params)
            
            # Train the final model with best parameters
            logger.info(f"Training final model with params: {best_params}")
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                **best_params
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mse": float(mean_squared_error(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
                "baseline_rmse": float(baseline_rmse),
                "improvement": float((baseline_rmse - metrics["rmse"]) / baseline_rmse * 100)
            }
            
            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
                
            # Generate and log plots
            with tempfile.TemporaryDirectory() as tmp_dir:
                # SHAP plots
                shap_plots = generate_shap_plots(model, X_test, tmp_dir)
                for name, path in shap_plots.items():
                    mlflow.log_artifact(path)
                    
                # Actual vs Predicted plot
                avs_path = os.path.join(tmp_dir, "actual_vs_predicted.png")
                plot_actual_vs_predicted(y_test, y_pred, avs_path)
                mlflow.log_artifact(avs_path)
                
                # Upload plots to S3 for dashboard
                for name, path in shap_plots.items():
                    s3_key = f"{MODEL_KEY_PREFIX}/visuals/shap/{model_id}_{name}.png"
                    s3_upload(path, s3_key)
                    
                s3_key = f"{MODEL_KEY_PREFIX}/visuals/avs_pred/{model_id}_avs_pred.png"
                s3_upload(avs_path, s3_key)
                
            # Log the model
            mlflow.xgboost.log_model(model, "model")
            
            # Register the model
            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(model_uri, model_id)
            
            # Promote to production if it's better
            client = MlflowClient()
            try:
                production_versions = client.get_latest_versions(model_id, stages=["Production"])
                
                if not production_versions:
                    logger.info(f"No production model found, promoting {model_id} version {registered_model.version}")
                    client.transition_model_version_stage(
                        name=model_id,
                        version=registered_model.version,
                        stage="Production"
                    )
                else:
                    prod_version = production_versions[0]
                    prod_run = client.get_run(prod_version.run_id)
                    prod_rmse = prod_run.data.metrics.get("rmse", float('inf'))
                    
                    improvement_pct = (prod_rmse - metrics["rmse"]) / prod_rmse * 100
                    
                    if metrics["rmse"] < prod_rmse:
                        logger.info(f"New model performs better: {metrics['rmse']:.4f} vs {prod_rmse:.4f}")
                        client.transition_model_version_stage(
                            name=model_id,
                            version=registered_model.version,
                            stage="Production"
                        )
                        
                        # Archive the old version
                        client.transition_model_version_stage(
                            name=model_id,
                            version=prod_version.version,
                            stage="Archived"
                        )
                        
                        send_message(
                            channel="#ml-updates",
                            title="🚀 New Model Promoted",
                            details=f"New {model_id} model promoted to Production.\n"
                                    f"RMSE: {metrics['rmse']:.4f} (improved by {improvement_pct:.2f}%)",
                            urgency="high"
                        )
                    else:
                        logger.info(f"New model does not outperform current production model")
                        client.transition_model_version_stage(
                            name=model_id,
                            version=registered_model.version,
                            stage="Staging"
                        )
                        
                        send_message(
                            channel="#ml-updates",
                            title="⚠️ Model Not Promoted",
                            details=f"New {model_id} model (RMSE: {metrics['rmse']:.4f}) kept in Staging "
                                    f"as it does not outperform Production (RMSE: {prod_rmse:.4f})",
                            urgency="medium"
                        )
            except Exception as e:
                logger.error(f"Error promoting model: {e}")
                send_message(
                    channel="#alerts",
                    title="❌ Model Promotion Failed",
                    details=f"Error promoting {model_id} model: {str(e)}",
                    urgency="high"
                )
                
        # Record execution time
        elapsed = time.time() - start_time
        metrics["execution_time"] = elapsed
        metrics["status"] = "success"
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Send update via WebSocket
        send_websocket_update_sync(model_id, metrics, "success")
        
        return model
        
    except Exception as e:
        error_msg = f"Error in model training: {str(e)}"
        logger.error(error_msg)
        send_message(
            channel="#alerts",
            title="❌ Model Training Failed",
            details=f"Error training {model_id} model: {error_msg}",
            urgency="high"
        )
        
        # Send update via WebSocket
        metrics["status"] = "failed"
        metrics["error"] = str(e)
        metrics["timestamp"] = datetime.now().isoformat()
        send_websocket_update_sync(model_id, metrics, "failed")
        
        raise

# backward‑compatibility alias
train_xgboost_hyperopt = train_and_compare_fn
