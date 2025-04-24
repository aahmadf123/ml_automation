#!/usr/bin/env python3
"""
tasks/training.py  – Unified trainer + explainability
----------------------------------------------------------
 • Baseline check: load last Production model and skip retrain if RMSE ≤ threshold
 • HyperOpt → finds best hyper‑parameters per model (configurable max‑evals)
 • Fallback from TimeSeriesSplit → train_test_split if not enough chronological rows
 • Logs to MLflow: RMSE, MSE, MAE, R² + SHAP summary + Actual vs. Predicted plot
 • Logs to ClearML: Metrics, parameters, artifacts and model
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
from utils.clearml_config import init_clearml, log_model_to_clearml
from utils.cache import GLOBAL_CACHE, cache_result

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
        
@cache_result
def generate_shap_plots(model, X: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """Generate SHAP plots for model explanation."""
    plot_paths = {}
    try:
        # Check if SHAP values are already in cache
        model_hash = str(hash(model))
        df_hash = str(id(X))
        df_name = f"training_shap_{df_hash}"
        
        # Calculate or retrieve SHAP values
        cached_shap_values = GLOBAL_CACHE.get_column_result(
            df_name=df_name,
            column="all",
            operation=f"shap_values_{model_hash}"
        )
        
        if cached_shap_values is not None:
            logger.info("Using cached SHAP values for plot generation")
            shap_values = cached_shap_values
        else:
            # Create explanation using TreeExplainer
            logger.info("Calculating SHAP values")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Store in cache for future use
            GLOBAL_CACHE.store_column_result(
                df_name=df_name,
                column="all",
                operation=f"shap_values_{model_hash}",
                result=shap_values
            )
        
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

@cache_result
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

@cache_result
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

@cache_result
def find_best_hyperparams(X_train, y_train, max_evals=50):
    """Find optimal hyperparameters using Hyperopt."""
    # Check for cached results
    df_hash = str(id(X_train))
    y_hash = str(id(y_train))
    cache_key = f"hyperparams_{df_hash}_{y_hash}_{max_evals}"
    
    cached_result = GLOBAL_CACHE.get_column_result(
        df_name=f"training_df_{df_hash}",
        column="hyperparams",
        operation=f"hyperopt_{max_evals}"
    )
    
    if cached_result is not None:
        logger.info(f"Using cached hyperparameters from previous optimization")
        return cached_result
    
    # Define hyperparameter search space
    space = {
        'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8, 9, 10]),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'gamma': hp.uniform('gamma', 0, 1),
        'reg_alpha': hp.loguniform('reg_alpha', -5, 0),
        'reg_lambda': hp.loguniform('reg_lambda', -5, 0)
    }
    
    # Create validation set for early stopping
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Store the splits in the cache for reuse
    train_val_split = {
        "X_train": X_train_opt,
        "X_val": X_val,
        "y_train": y_train_opt,
        "y_val": y_val
    }
    
    # Cache the split data
    df_name = f"training_df_{df_hash}"
    GLOBAL_CACHE.store_column_result(
        df_name=df_name,
        column="data_split",
        operation="train_val_split",
        result=train_val_split
    )
    
    # Define the objective function for optimization
    def objective(params):
        # Create a CV set first
        train_data = xgb.DMatrix(X_train_opt, label=y_train_opt)
        val_data = xgb.DMatrix(X_val, label=y_val)
        
        # Train model with early stopping
        model = xgb.train(
            params,
            train_data,
            num_boost_round=1000,
            evals=[(val_data, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Get best validation score (rmse)
        val_pred = model.predict(val_data)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        # Store intermediate results in cache
        intermediate_results = GLOBAL_CACHE.get_column_result(
            df_name=df_name,
            column="hyperparams",
            operation="hyperopt_intermediate"
        ) or []
        
        # Append this result
        intermediate_results.append({
            'params': params,
            'rmse': float(rmse),
            'timestamp': datetime.now().isoformat()
        })
        
        # Store updated results
        GLOBAL_CACHE.store_column_result(
            df_name=df_name,
            column="hyperparams",
            operation="hyperopt_intermediate",
            result=intermediate_results
        )
        
        return {'loss': rmse, 'status': STATUS_OK}
    
    # Run the optimization
    logger.info(f"Starting hyperparameter optimization with max_evals={max_evals}")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.RandomState(42)
    )
    
    # Extract best parameters
    best_params = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10][best['max_depth']],
        'learning_rate': best['learning_rate'],
        'min_child_weight': best['min_child_weight'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'gamma': best['gamma'],
        'reg_alpha': np.exp(best['reg_alpha']),
        'reg_lambda': np.exp(best['reg_lambda']),
        'objective': 'reg:squarederror',
        'seed': 42
    }
    
    logger.info(f"Best hyperparameters found: {best_params}")
    
    # Cache the final result
    GLOBAL_CACHE.store_column_result(
        df_name=f"training_df_{df_hash}",
        column="hyperparams",
        operation=f"hyperopt_{max_evals}",
        result=best_params
    )
    
    return best_params

def train_and_compare_fn(model_id: str, processed_path: str) -> None:
    """
    Train a model, compare with baseline, and log everything to MLflow.
    
    Args:
        model_id: Unique identifier for the model
        processed_path: Path to processed data file
    """
    start_time = time.time()
    logger.info(f"Starting training for model {model_id}")
    
    # Initialize MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    # Start ClearML task
    clearml_task = None
    try:
        clearml_task = init_clearml(model_id)
    except Exception as e:
        logger.warning(f"Error initializing ClearML: {str(e)}")
    
    try:
        # Load the data
        logger.info(f"Loading data from {processed_path}")
        
        # Check cache first
        df_hash = os.path.basename(processed_path).replace('.parquet', '')
        df_name = f"training_df_{df_hash}"
        
        cached_df = GLOBAL_CACHE.get_transformed(df_name)
        if cached_df is not None:
            logger.info("Using cached DataFrame")
            df = cached_df
        else:
            df = pd.read_parquet(processed_path)
            GLOBAL_CACHE.store_transformed(df, df_name)
        
        # Get target column
        target_col = "claim_amount"
        
        if target_col not in df.columns:
            error_msg = f"Target column '{target_col}' not found in the data"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Calculate feature statistics and cache them for future use
        GLOBAL_CACHE.compute_statistics(X, df_name)
        
        # Train-test split
        try:
            # First try time-series split if date column exists
            if 'date' in df.columns or 'timestamp' in df.columns:
                date_col = 'date' if 'date' in df.columns else 'timestamp'
                logger.info(f"Using time-series split with date column: {date_col}")
                
                # Sort by date
                df = df.sort_values(by=date_col)
                
                # Use the last 20% as test set
                split_idx = int(len(df) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            else:
                # Fallback to random split
                logger.info("No date column found, using random train-test split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        except Exception as e:
            logger.warning(f"Error in time-series split, falling back to random split: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
        # Cache the train-test split
        train_test_data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
        
        GLOBAL_CACHE.store_column_result(
            df_name=df_name,
            column="data_split",
            operation="train_test_split",
            result=train_test_data
        )
        
        # Train a simple baseline model
        logger.info("Training baseline model")
        baseline_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        baseline_model.fit(X_train, y_train)
        
        # Get baseline predictions
        baseline_preds = baseline_model.predict(X_test)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
        logger.info(f"Baseline RMSE: {baseline_rmse:.4f}")
        
        # Check if we should skip training based on existing model performance
        skip_training = should_skip_training(model_id, baseline_rmse)
        
        if skip_training:
            logger.info(f"Skipping training for {model_id} as existing model performance is sufficient")
            
            # Send Slack notification
            from utils.slack import post as send_message
            send_message(
                channel="#ml-training",
                title="⏭️ Training Skipped",
                details=f"Training skipped for model '{model_id}' as existing performance is sufficient.\n" +
                        f"Baseline RMSE: {baseline_rmse:.4f}",
                urgency="low"
            )
            
            # Update dashboard via WebSocket
            send_websocket_update_sync(
                model_id=model_id,
                metrics={'rmse': float(baseline_rmse)},
                status='skipped'
            )
            
            return
            
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            # Find optimal hyperparameters
            logger.info("Finding optimal hyperparameters...")
            max_evals = MODEL_CONFIG.get('max_evals', 30)
            best_params = find_best_hyperparams(X_train, y_train, max_evals=max_evals)
            mlflow.log_params(best_params)
            
            # Log parameters to ClearML if available
            if clearml_task:
                for param_name, param_value in best_params.items():
                    clearml_task.set_parameter(param_name, param_value)
            
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
            
            # Log metrics to MLflow
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
                
            # Log metrics to ClearML if available
            if clearml_task:
                for metric_name, metric_value in metrics.items():
                    clearml_task.get_logger().report_scalar(
                        title="Metrics",
                        series=metric_name,
                        value=metric_value,
                        iteration=0
                    )
            
            # Save the feature importance plot
            plt.figure(figsize=(12, 6))
            xgb.plot_importance(model, max_num_features=20)
            plt.title('Feature Importance')
            feat_imp_path = f"/tmp/feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(feat_imp_path)
            plt.close()
            
            # Log feature importance plot to MLflow
            mlflow.log_artifact(feat_imp_path)
            
            # Log feature importance plot to ClearML if available
            if clearml_task:
                clearml_task.get_logger().report_image(
                    title="Feature Importance",
                    series="XGBoost",
                    image=feat_imp_path,
                    iteration=0
                )
            
            # Create actual vs predicted plot
            av_pred_path = f"/tmp/actual_vs_predicted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_actual_vs_predicted(y_test, y_pred, av_pred_path)
            
            # Log actual vs predicted plot to MLflow
            mlflow.log_artifact(av_pred_path)
            
            # Log to ClearML if available
            if clearml_task:
                clearml_task.get_logger().report_image(
                    title="Actual vs Predicted",
                    series="Regression",
                    image=av_pred_path,
                    iteration=0
                )
            
            # Generate and log SHAP plots
            shap_tmp_dir = tempfile.mkdtemp(prefix="shap_")
            try:
                shap_plots = generate_shap_plots(model, X_test.sample(min(200, len(X_test))), shap_tmp_dir)
                for plot_name, plot_path in shap_plots.items():
                    if os.path.exists(plot_path):
                        mlflow.log_artifact(plot_path, artifact_path="shap_plots")
                        
                        # Log to ClearML if available
                        if clearml_task:
                            clearml_task.get_logger().report_image(
                                title="SHAP",
                                series=plot_name,
                                image=plot_path,
                                iteration=0
                            )
                        
                        # Archive to S3
                        s3_key = f"{MODEL_KEY_PREFIX}/visuals/shap/{model_id}/{plot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        s3_upload(plot_path, s3_key)
            except Exception as e:
                logger.error(f"Error generating SHAP plots: {e}")
            
            # Log the model to MLflow
            mlflow.xgboost.log_model(model, "model")
            
            # Register the model in MLflow
            try:
                client = MlflowClient()
                
                # Try to register model version
                try:
                    registered_model = client.get_registered_model(model_id)
                    logger.info(f"Found existing registered model: {model_id}")
                except RestException:
                    # Create a new registered model if it doesn't exist
                    logger.info(f"Creating new registered model: {model_id}")
                    client.create_registered_model(model_id)
                    
                # Register new version
                model_version = client.create_model_version(
                    name=model_id,
                    source=f"runs:/{run_id}/model",
                    run_id=run_id
                )
                logger.info(f"Registered model version: {model_version.version}")
                
                # Check if this is a significant improvement over production
                try:
                    production_versions = client.get_latest_versions(model_id, stages=["Production"])
                    
                    # If no existing production model or clear improvement, promote to production
                    if not production_versions or metrics["improvement"] > 5.0:  # 5% improvement threshold
                        logger.info(f"Promoting to production: {model_id} version {model_version.version}")
                        client.transition_model_version_stage(
                            name=model_id,
                            version=model_version.version,
                            stage="Production",
                            archive_existing_versions=True
                        )
                    else:
                        logger.info(f"Not promoting to production due to insufficient improvement: {metrics['improvement']:.2f}%")
                        client.transition_model_version_stage(
                            name=model_id,
                            version=model_version.version,
                            stage="Staging"
                        )
                except Exception as e:
                    logger.error(f"Error evaluating production promotion: {e}")
                    # Default to Staging if there's an error
                    client.transition_model_version_stage(
                        name=model_id,
                        version=model_version.version,
                        stage="Staging"
                    )
                    
            except Exception as e:
                logger.error(f"Error registering model: {e}")
            
            # Log model to ClearML if available
            if clearml_task:
                try:
                    log_model_to_clearml(clearml_task, model, model_id)
                except Exception as e:
                    logger.error(f"Error logging model to ClearML: {e}")
                    
            # Send notification
            from utils.slack import post as send_message
            send_message(
                channel="#ml-training",
                title="✅ Training Complete",
                details=f"Model '{model_id}' trained successfully.\n" +
                        f"RMSE: {metrics['rmse']:.4f} (baseline: {metrics['baseline_rmse']:.4f})\n" +
                        f"Improvement: {metrics['improvement']:.2f}%\n" +
                        f"MLflow Run: {run_id}",
                urgency="high" if metrics["improvement"] > 10 else "normal"
            )
            
            # Update dashboard via WebSocket
            send_websocket_update_sync(
                model_id=model_id,
                metrics=metrics,
                status='completed'
            )
                
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
            
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        
        # Send notification
        from utils.slack import post as send_message
        send_message(
            channel="#alerts",
            title="❌ Training Failed",
            details=f"Error training model '{model_id}':\n{str(e)}",
            urgency="high"
        )
        
        # Update dashboard via WebSocket
        send_websocket_update_sync(
            model_id=model_id,
            metrics={},
            status='failed'
        )
        
        # Close ClearML task if still open
        if clearml_task:
            try:
                clearml_task.close()
            except:
                pass
                
        raise

# backward‑compatibility alias
train_xgboost_hyperopt = train_and_compare_fn
