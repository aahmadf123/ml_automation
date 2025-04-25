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
 • NEW: Efficient parallel training of all 5 models with shared data preparation
"""

import os
import json
import logging
import tempfile
import time
import joblib
import websockets
import asyncio
from typing import Dict, Optional, Tuple, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
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
    MLFLOW_URI, MLFLOW_EXPERIMENT, MLFLOW_ARTIFACT_URI, 
    MLFLOW_DB_URI, MODEL_REGISTRY_URI, MODEL_CONFIG,
    AUTO_APPROVE_MODEL
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

def find_best_hyperparams(X_train, y_train, max_evals=50, sample_weight=None):
    """Find the best hyperparameters for XGBoost model using Hyperopt.
    
    Args:
        X_train: Training features
        y_train: Training targets
        max_evals: Maximum number of evaluations
        sample_weight: Optional sample weights for training
        
    Returns:
        Dictionary of best hyperparameters
    """
    # Check for cached result
    cache_key = f"hyperopt_{hash(str(X_train.columns))}"
    cached = GLOBAL_CACHE.get(cache_key)
    if cached:
        logger.info("Using cached hyperparameters")
        return cached
    
    logger.info(f"Starting hyperparameter optimization with max_evals={max_evals}")
    
    # Define the search space - use more conservative ranges to avoid extreme values
    space = {
        'max_depth': hp.choice('max_depth', [3, 4, 5, 6]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'min_child_weight': hp.choice('min_child_weight', [1, 2, 3, 4]),
        'subsample': hp.uniform('subsample', 0.6, 0.95),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 0.95),
        'gamma': hp.uniform('gamma', 0, 1),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(1.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(1.0))
    }
    
    # Create validation set for early stopping
    try:
        X_train_opt, X_valid_opt, y_train_opt, y_valid_opt = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Handle sample weights for validation split if provided
        if sample_weight is not None:
            if isinstance(sample_weight, pd.Series):
                train_indices = X_train_opt.index
                valid_indices = X_valid_opt.index
                
                train_weights = sample_weight.loc[train_indices].values if all(idx in sample_weight.index for idx in train_indices) else None
                valid_weights = sample_weight.loc[valid_indices].values if all(idx in sample_weight.index for idx in valid_indices) else None
            else:
                # If weights are numpy array, use the indices from the train/valid split
                if len(sample_weight) == len(X_train):
                    train_mask = X_train.index.isin(X_train_opt.index)
                    valid_mask = X_train.index.isin(X_valid_opt.index)
                    
                    train_weights = sample_weight[train_mask] if sum(train_mask) > 0 else None
                    valid_weights = sample_weight[valid_mask] if sum(valid_mask) > 0 else None
                else:
                    logger.warning("Sample weight length doesn't match training data, ignoring weights for hyperopt")
                    train_weights = None
                    valid_weights = None
        else:
            train_weights = None
            valid_weights = None
            
        logger.info(f"Split data for hyperopt: train={len(X_train_opt)}, valid={len(X_valid_opt)}")
    except Exception as e:
        logger.warning(f"Error creating validation split for hyperopt: {str(e)}")
        # Fall back to using all training data
        X_train_opt = X_train
        y_train_opt = y_train
        X_valid_opt = None
        y_valid_opt = None
        train_weights = sample_weight
        valid_weights = None
        logger.info("Using all training data for hyperopt without validation split")
    
    start_time = time.time()
    trials = Trials()
    
    def objective(params):
        """Objective function for hyperopt - maximize validation performance."""
        # Convert from log-space to actual space for some parameters
        actual_params = params.copy()
        actual_params['objective'] = 'reg:squarederror'
        actual_params['seed'] = 42
        
        # Add tree_method='hist' for faster training
        actual_params['tree_method'] = 'hist'
        
        # Attempt training and evaluation
        try:
            # Create model
            model = xgb.XGBRegressor(**actual_params)
            
            # Check if we have validation data
            if X_valid_opt is not None and y_valid_opt is not None:
                # Train with early stopping
                eval_set = [(X_train_opt, y_train_opt), (X_valid_opt, y_valid_opt)]
                
                # Include sample weights in eval_set if available
                sample_weight_params = {}
                if train_weights is not None:
                    sample_weight_params['sample_weight'] = train_weights
                    
                eval_weights = None
                if valid_weights is not None:
                    eval_weights = [train_weights, valid_weights]
                    sample_weight_params['sample_weight_eval_set'] = eval_weights
                
                # Fit the model
                model.fit(
                    X_train_opt, y_train_opt,
                    eval_set=eval_set,
                    early_stopping_rounds=10,
                    verbose=False,
                    **sample_weight_params
                )
                
                # Get best score from early stopping
                val_score = model.best_score
                
                # Return validation RMSE for minimization
                return val_score
            else:
                # Train without validation (using cross-validation)
                cv_result = xgb.cv(
                    actual_params,
                    xgb.DMatrix(X_train_opt, label=y_train_opt, weight=train_weights),
                    num_boost_round=100,
                    nfold=3,
                    early_stopping_rounds=10,
                    metrics='rmse',
                    seed=42,
                    verbose_eval=False
                )
                
                # Get best mean test score
                best_score = cv_result['test-rmse-mean'].min()
                return best_score
        except Exception as e:
            # Log the error but don't fail the optimization
            logger.warning(f"Error in hyperopt iteration: {str(e)}")
            # Return a high loss value to penalize this parameter set
            return 9999.0
    
    # Run hyperopt optimization
    try:
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            show_progressbar=False
        )
        
        logger.info(f"Hyperparameter optimization completed in {time.time() - start_time:.2f} seconds")
        
        # Get best parameters
        best_params = space_eval(space, best)
        
        # Add required parameters
        best_params['objective'] = 'reg:squarederror'
        best_params['seed'] = 42
        best_params['tree_method'] = 'hist'  # Add hist method for faster training
        
        # Log results
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best score: {min([r['loss'] for r in trials.results if r['loss'] < 999])}")
        
        # Cache the result
        GLOBAL_CACHE[cache_key] = best_params
        
        return best_params
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}")
        # Return default parameters
        default_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0.001,
            'reg_lambda': 0.001,
            'objective': 'reg:squarederror',
            'seed': 42,
            'tree_method': 'hist'
        }
        
        logger.info(f"Using default parameters due to optimization error: {default_params}")
        return default_params

def train_and_compare_fn(model_id: str, processed_path: str) -> None:
    """
    Train a model, compare with baseline, and log everything to MLflow.
    
    Args:
        model_id: Unique identifier for the model
        processed_path: Path to processed data file
    """
    start_time = time.time()
    logger.info(f"Starting training for model {model_id}")
    
    # Initialize MLflow with EC2 URL
    mlflow.set_tracking_uri("http://3.146.46.179:5000")
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
        if target_column and target_column in df.columns:
            target_col = target_column
            logger.info(f"Using specified target column: {target_col}")
        else:
            # Try different common target column names in order of preference
            potential_targets = ['trgt', 'pure_premium', 'target', 'claim_amount']
            found = False
            
            for col in potential_targets:
                if col in df.columns:
                    target_col = col
                    logger.info(f"Using target column: {target_col}")
                    found = True
                    break
            
            if not found:
                # Look for columns that could be used to create a target
                logger.info("Target column not found, looking for components to create it")
                cols = df.columns.tolist()
                
                losses_cols = [col for col in cols if 'loss' in col.lower()]
                premium_cols = [col for col in cols if 'premium' in col.lower()]
                exposure_cols = [col for col in cols if any(word in col.lower() for word in ['exposure', 'eey', 'earned'])]
                
                logger.info(f"Potential loss columns: {losses_cols}")
                logger.info(f"Potential premium columns: {premium_cols}")
                logger.info(f"Potential exposure columns: {exposure_cols}")
                
                # Try to create target column from components
                if 'il_total' in df.columns and 'eey' in df.columns:
                    logger.info("Creating 'trgt' column from 'il_total' / 'eey'")
                    df['trgt'] = df['il_total'] / df['eey']
                    target_col = 'trgt'
                elif losses_cols and exposure_cols:
                    loss_col = losses_cols[0]
                    exposure_col = exposure_cols[0]
                    logger.info(f"Creating 'trgt' column from {loss_col} / {exposure_col}")
                    df['trgt'] = df[loss_col] / df[exposure_col]
                    target_col = 'trgt'
                else:
                    target_col = "claim_amount"  # default that likely won't be found
                    
        if target_col not in df.columns:
            error_msg = f"Target column '{target_col}' not found in the data and could not be created"
            logger.error(error_msg)
            logger.info(f"Available columns: {df.columns.tolist()}")
            raise ValueError(error_msg)
            
        # Validate target column - critical for insurance modeling
        target_values = df[target_col].dropna()
        if len(target_values) < 100:  # Minimum sample size for meaningful modeling
            error_msg = f"Insufficient non-null values in target column ({len(target_values)} found)"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if target_values.min() < 0:
            logger.warning(f"Negative values found in target column {target_col}. This may be problematic for insurance modeling.")
        
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
            
            # Log skip notification
            logger.info(
                f"⏭️ Training Skipped: Training skipped for model '{model_id}' as existing performance is sufficient.\n" +
                f"Baseline RMSE: {baseline_rmse:.4f}"
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
                    
            # Log success notification
            logger.info(
                f"Training Complete: Model '{model_id}' trained successfully.\n" +
                f"RMSE: {metrics['rmse']:.4f} (baseline: {metrics['baseline_rmse']:.4f})\n" +
                f"Improvement: {metrics['improvement']:.2f}%\n" +
                f"MLflow Run: {run_id}"
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
        
        # Log error notification
        logger.error(f"❌ Training Failed: Error training model '{model_id}':\n{str(e)}")
        
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

def train_single_model(
    model_id,
    processed_df,
    X_train,
    X_test,
    y_train,
    y_test,
    target_column,
    sample_weight=None,
    max_evals=None,
    evaluate_only=False
):
    """Train a single model with XGBoost and evaluate its performance.
    
    Args:
        model_id: Identifier for the model
        processed_df: Processed DataFrame
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        target_column: Name of the target column
        sample_weight: Optional sample weights
        max_evals: Maximum evaluations for hyperparameter tuning
        evaluate_only: If True, only evaluate existing model without training
        
    Returns:
        Dictionary with model results
    """
    start_time = time.time()
    
    # Validate inputs
    if processed_df is None or processed_df.empty:
        logger.error(f"Model {model_id}: Empty or None input data")
        return {"success": False, "error": "Empty input data", "model_id": model_id}
    
    if X_train is None or X_train.empty or y_train is None or len(y_train) == 0:
        logger.error(f"Model {model_id}: Empty or None training data")
        return {"success": False, "error": "Empty training data", "model_id": model_id}
    
    # Try to get MLflow connection
    try:
        client = MlflowClient()
    except Exception as e:
        logger.error(f"Model {model_id}: Failed to initialize MLflow client: {str(e)}")
        client = None
    
    # Set MLflow experiment
    experiment_id = None
    try:
        if client is not None:
            experiment_name = f"model_{model_id}"
            try:
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment:
                    experiment_id = experiment.experiment_id
                else:
                    experiment_id = client.create_experiment(experiment_name)
            except Exception as exp_err:
                logger.warning(f"Model {model_id}: Error setting MLflow experiment: {str(exp_err)}")
    except Exception as e:
        logger.warning(f"Model {model_id}: Error with MLflow experiment: {str(e)}")
    
    # Start MLflow run
    mlflow_run = None
    try:
        if experiment_id is not None:
            mlflow_run = client.create_run(experiment_id)
            run_id = mlflow_run.info.run_id
            logger.info(f"Model {model_id}: Started MLflow run {run_id}")
    except Exception as e:
        logger.warning(f"Model {model_id}: Failed to start MLflow run: {str(e)}")
        run_id = None
    
    # Get model features based on model_id
    try:
        model_X_train = select_model_features(X_train, model_id)
        model_X_test = select_model_features(X_test, model_id)
        if model_X_train.empty or model_X_test.empty:
            logger.error(f"Model {model_id}: Empty feature sets after selection")
            return {"success": False, "error": "Empty feature sets", "model_id": model_id}
        logger.info(f"Model {model_id}: Selected {len(model_X_train.columns)} features")
    except Exception as e:
        logger.error(f"Model {model_id}: Error in feature selection: {str(e)}")
        return {"success": False, "error": f"Feature selection error: {str(e)}", "model_id": model_id}
    
    # Log feature names to MLflow
    if run_id is not None:
        try:
            client.log_param(run_id, "features", list(model_X_train.columns))
            client.log_param(run_id, "feature_count", len(model_X_train.columns))
        except Exception as e:
            logger.warning(f"Model {model_id}: Failed to log features to MLflow: {str(e)}")
    
    # Skip training if evaluate_only is True
    if evaluate_only:
        logger.info(f"Model {model_id}: Skipping training, evaluate_only=True")
        # Implement evaluation-only logic if needed
        return {"success": False, "error": "Training skipped - evaluate_only", "model_id": model_id}
    
    # Find best hyperparameters or use default if optimization fails
    try:
        max_evals_actual = max_evals or 20  # Default to 20 if not specified
        best_params = find_best_hyperparams(model_X_train, y_train, max_evals=max_evals_actual, sample_weight=sample_weight)
        logger.info(f"Model {model_id}: Found best hyperparameters: {best_params}")
    except Exception as e:
        logger.warning(f"Model {model_id}: Error finding hyperparameters: {str(e)}, using defaults")
        # Use default hyperparameters as fallback
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0.001,
            'reg_lambda': 0.001,
            'objective': 'reg:squarederror',
            'seed': 42,
            'tree_method': 'hist'
        }
    
    # Log hyperparameters to MLflow
    if run_id is not None:
        try:
            for param_name, param_value in best_params.items():
                client.log_param(run_id, param_name, param_value)
        except Exception as e:
            logger.warning(f"Model {model_id}: Failed to log hyperparameters to MLflow: {str(e)}")
    
    # Train model with error handling
    model = None
    try:
        # Train with best parameters
        logger.info(f"Model {model_id}: Training model with best parameters")
        model = xgb.XGBRegressor(**best_params)
        
        # Use early_stopping for better performance
        eval_set = [(model_X_train, y_train), (model_X_test, y_test)]
        
        # Prepare sample weight parameters
        sample_weight_params = {}
        if sample_weight is not None:
            sample_weight_params['sample_weight'] = sample_weight
        
        # Fit model with early stopping
        model.fit(
            model_X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False,
            **sample_weight_params
        )
        
        logger.info(f"Model {model_id}: Training completed with {model.get_booster().num_boosted_rounds()} boosting rounds")
    except Exception as e:
        logger.warning(f"Model {model_id}: Error training model with best parameters: {str(e)}")
        logger.info(f"Model {model_id}: Trying with simplified parameters")
        
        # Fallback to simpler parameters
        try:
            simple_params = {
                'max_depth': 3,
                'learning_rate': 0.1,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'seed': 42
            }
            
            model = xgb.XGBRegressor(**simple_params)
            model.fit(model_X_train, y_train)
            logger.info(f"Model {model_id}: Training succeeded with simplified parameters")
            
            # Update best_params to reflect what was actually used
            best_params = simple_params
            
            # Log the fallback parameters to MLflow
            if run_id is not None:
                try:
                    client.log_param(run_id, "used_fallback_params", True)
                    for param_name, param_value in simple_params.items():
                        client.log_param(run_id, f"fallback_{param_name}", param_value)
                except Exception as mlflow_err:
                    logger.warning(f"Model {model_id}: Failed to log fallback params to MLflow: {str(mlflow_err)}")
                    
        except Exception as fallback_err:
            logger.error(f"Model {model_id}: Failed to train even with simplified parameters: {str(fallback_err)}")
            return {
                "success": False, 
                "error": f"Training failed: {str(fallback_err)}", 
                "model_id": model_id
            }
    
    # If we got here, model training succeeded
    if model is None:
        logger.error(f"Model {model_id}: Model is None after training")
        return {"success": False, "error": "Model is None after training", "model_id": model_id}
    
    # Evaluate model
    try:
        # Make predictions
        y_pred = model.predict(model_X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model {model_id}: Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Log metrics to MLflow
        if run_id is not None:
            try:
                client.log_metric(run_id, "rmse", rmse)
                client.log_metric(run_id, "mae", mae)
                client.log_metric(run_id, "r2", r2)
            except Exception as e:
                logger.warning(f"Model {model_id}: Failed to log metrics to MLflow: {str(e)}")
        
        # Create and log feature importance plot
        try:
            feature_importance_fig = plt.figure(figsize=(10, 6))
            xgb.plot_importance(model, ax=feature_importance_fig.gca(), importance_type='weight')
            plt.title(f'Feature Importance (Model {model_id})')
            plt.tight_layout()
            
            # Save feature importance plot
            feature_importance_path = f"/tmp/feature_importance_{model_id}.png"
            plt.savefig(feature_importance_path)
            plt.close()
            
            # Log feature importance to MLflow
            if run_id is not None:
                try:
                    client.log_artifact(run_id, feature_importance_path)
                except Exception as e:
                    logger.warning(f"Model {model_id}: Failed to log feature importance plot: {str(e)}")
        except Exception as plot_err:
            logger.warning(f"Model {model_id}: Error creating feature importance plot: {str(plot_err)}")
        
        # Create and log actual vs predicted plot
        try:
            actual_vs_pred_fig = plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Actual vs Predicted (Model {model_id})')
            plt.tight_layout()
            
            # Save actual vs predicted plot
            actual_vs_pred_path = f"/tmp/actual_vs_pred_{model_id}.png"
            plt.savefig(actual_vs_pred_path)
            plt.close()
            
            # Log actual vs predicted plot to MLflow
            if run_id is not None:
                try:
                    client.log_artifact(run_id, actual_vs_pred_path)
                except Exception as e:
                    logger.warning(f"Model {model_id}: Failed to log actual vs predicted plot: {str(e)}")
        except Exception as plot_err:
            logger.warning(f"Model {model_id}: Error creating actual vs predicted plot: {str(plot_err)}")
                
    except Exception as eval_err:
        logger.error(f"Model {model_id}: Error evaluating model: {str(eval_err)}")
        return {
            "success": False, 
            "error": f"Evaluation failed: {str(eval_err)}", 
            "model_id": model_id
        }
    
    # End MLflow run
    if run_id is not None:
        try:
            client.set_terminated(run_id)
        except Exception as e:
            logger.warning(f"Model {model_id}: Failed to terminate MLflow run: {str(e)}")
    
    # Log training time
    training_time = time.time() - start_time
    logger.info(f"Model {model_id}: Training completed in {training_time:.2f} seconds")
    
    # Return results
    return {
        "success": True,
        "model": model,
        "model_id": model_id,
        "best_params": best_params,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "training_time": training_time
    }

def load_pretrained_model(model_id: str, model_dir: str = None) -> Dict[str, Any]:
    """
    Load a pretrained model from S3
    
    Args:
        model_id: ID of the model to load (model1 or model4)
        model_dir: Local directory to save the model to
        
    Returns:
        Dict with model and related information
    """
    import os
    import tempfile
    import joblib
    from utils.storage import download
    from utils.config import DATA_BUCKET, MODEL_KEY_PREFIX, MODEL_CONFIG
    
    logger.info(f"Loading pretrained model {model_id} from S3")
    
    # Only allow model1 and model4
    if model_id not in ['model1', 'model4']:
        error_msg = f"Only model1 and model4 are supported, got {model_id}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg, "model_id": model_id}
    
    # Create a temporary directory if model_dir is not provided
    if model_dir is None:
        model_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory for model: {model_dir}")
    
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Get the correct case-sensitive filename from MODEL_CONFIG
    model_config = MODEL_CONFIG.get(model_id, {})
    s3_filename = model_config.get("file_name")
    
    # If filename not in config, use default pattern with capitalized first letter
    if not s3_filename:
        s3_filename = f"Model{model_id[5:]}.joblib" if model_id.startswith("model") else f"{model_id}.joblib"
        logger.warning(f"Model filename not found in MODEL_CONFIG, using default: {s3_filename}")
    
    model_key = f"{MODEL_KEY_PREFIX}/{s3_filename}"
    local_model_path = os.path.join(model_dir, f"{model_id}.joblib")  # Keep local path consistent
    
    # Try to download the model from S3
    try:
        logger.info(f"Downloading model from S3: s3://{DATA_BUCKET}/{model_key}")
        download(model_key, local_model_path, bucket=DATA_BUCKET)
        
        # Verify the file was downloaded
        if not os.path.exists(local_model_path):
            error_msg = f"Failed to download model file: {local_model_path}"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg, "model_id": model_id}
        
        file_size = os.path.getsize(local_model_path)
        if file_size == 0:
            error_msg = f"Downloaded model file is empty: {local_model_path}"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg, "model_id": model_id}
            
        logger.info(f"Successfully downloaded model file ({file_size} bytes)")
        
        # Load the model
        logger.info(f"Loading model from {local_model_path}")
        model = joblib.load(local_model_path)
        
        # Basic validation to check if it's a valid model
        if not hasattr(model, 'predict'):
            error_msg = f"Loaded object does not seem to be a valid model: {type(model)}"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg, "model_id": model_id}
        
        # Create a run ID in MLflow to track this loaded model
        run_id = None
        try:
            # Initialize MLflow with EC2 endpoint
            mlflow.set_tracking_uri("http://3.146.46.179:5000")
            logger.info("Set MLflow tracking URI to http://3.146.46.179:5000")
                
            client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
            experiment_name = f"model_{model_id}"
            
            # Get or create the experiment
            experiment_id = None
            try:
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Found existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
                else:
                    # Create a new experiment
                    experiment_id = client.create_experiment(experiment_name)
                    logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            except Exception as exp_err:
                logger.warning(f"Error getting/creating MLflow experiment: {str(exp_err)}")
                experiment_id = "0"  # Use default experiment as fallback
            
            # Create a new run in the experiment
            try:
                mlflow_run = client.create_run(experiment_id)
                run_id = mlflow_run.info.run_id
                logger.info(f"Created new MLflow run: {run_id}")
                
                # Log model info
                client.log_param(run_id, "model_source", "pretrained")
                client.log_param(run_id, "model_path", f"s3://{DATA_BUCKET}/{model_key}")
                client.log_param(run_id, "s3_filename", s3_filename)
                client.log_param(run_id, "model_id", model_id)
                client.log_param(run_id, "load_time", datetime.now().isoformat())
            
                # Get model metadata from MODEL_CONFIG for additional params
                model_config = MODEL_CONFIG.get(model_id, {})
                for param_key, param_value in model_config.get('hyperparameters', {}).items():
                    try:
                        client.log_param(run_id, param_key, param_value)
                    except Exception as param_error:
                        logger.warning(f"Error logging parameter {param_key}: {str(param_error)}")
                
                # Log model to MLflow with proper error handling
                try:
                    with mlflow.start_run(run_id=run_id):
                        # Use appropriate MLflow flavor based on model type
                        if 'xgboost' in str(type(model)).lower():
                            mlflow.xgboost.log_model(model, f"{model_id}_model")
                        else:
                            mlflow.sklearn.log_model(model, f"{model_id}_model")
                        logger.info(f"Successfully logged model to MLflow")
                except Exception as model_log_error:
                    logger.warning(f"Error logging model artifact to MLflow: {str(model_log_error)}")
            except Exception as run_err:
                logger.warning(f"Error logging run data to MLflow: {str(run_err)}")
                
        except Exception as e:
            logger.warning(f"Error setting up MLflow tracking for pretrained model: {str(e)}")
            if run_id:
                # Try to clean up the partially created run
                try:
                    client.set_terminated(run_id, "FAILED")
                    logger.info(f"Terminated incomplete MLflow run: {run_id}")
                except Exception as term_error:
                    logger.warning(f"Failed to terminate MLflow run: {str(term_error)}")
            run_id = None
        
        # Log successful model loading
        logger.info(f"Successfully loaded pretrained model {model_id}")
        if hasattr(model, 'n_estimators') and hasattr(model, 'max_depth'):
            logger.info(f"Model details: n_estimators={model.n_estimators}, max_depth={model.max_depth}")
            
        # Get additional model metadata from MODEL_CONFIG
        model_config = MODEL_CONFIG.get(model_id, {})
        
        # Return success result
        return {
            "status": "completed",
            "model": model,
            "model_id": model_id,
            "run_id": run_id,
            "name": model_config.get("name", model_id),
            "description": model_config.get("description", "Pretrained model loaded from S3"),
            "metrics": {
                # These will be populated when evaluating on test data
                "rmse": None,
                "mae": None,
                "r2": None
            }
        }
        
    except Exception as e:
        error_msg = f"Error loading pretrained model {model_id}: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full stack trace:")
        
        # Clean up any temporary files
        try:
            if model_dir is not None and os.path.exists(model_dir) and tempfile.gettempdir() in model_dir:
                import shutil
                shutil.rmtree(model_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {model_dir}")
        except Exception as cleanup_err:
            logger.warning(f"Error during cleanup: {str(cleanup_err)}")
            
        return {"status": "failed", "error": error_msg, "model_id": model_id}
def train_multiple_models(
    processed_path: str,
    parallel: bool = True,
    max_workers: int = 3,
    target_column: str = None,
    weight_column: str = None
) -> Dict[str, Dict]:
    """
    Load pretrained models (Model1 and Model4 only) from S3 and evaluate them
    
    Args:
        processed_path: Path to processed data
        parallel: Whether to load/evaluate models in parallel
        max_workers: Maximum number of parallel workers
        target_column: Name of target column
        weight_column: Name of weight column
        
    Returns:
        Dict with model results keyed by model_id
    """
    logger.info("Starting train_multiple_models (loading pretrained models)")
    logger.info(f"Using processed data from {processed_path}")
    
    try:
        # Verify the file exists
        if not os.path.exists(processed_path):
            error_msg = f"Processed data file not found: {processed_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Pre-load and prepare data once
        try:
            logger.info(f"Pre-loading and preparing data from {processed_path}")
            df = pd.read_parquet(processed_path)
            
            if df.empty:
                raise ValueError("Input DataFrame is empty.")
            
            available_cols = df.columns.tolist()
            actual_target_col = 'pure_premium' # Explicitly set target name

            # Check if target needs to be calculated
            if actual_target_col not in available_cols:
                logger.info(f"Target column '{actual_target_col}' not found, attempting to calculate from 'il_total' / 'eey'.")
                if 'il_total' in available_cols and 'eey' in available_cols:
                    # Avoid division by zero or near-zero, replace with NaN
                    original_rows = len(df)
                    df['eey'] = df['eey'].replace(0, np.nan)
                    df = df.dropna(subset=['eey'])
                    removed_rows = original_rows - len(df)
                    if removed_rows > 0:
                         logger.warning(f"Removed {removed_rows} rows with zero or NaN exposure ('eey') before calculating target.")
                    
                    if df.empty:
                         error_msg = "DataFrame became empty after removing rows with zero/NaN exposure ('eey'). Cannot calculate target."
                         logger.error(error_msg)
                         raise ValueError(error_msg)
                    
                    df[actual_target_col] = df['il_total'] / df['eey']
                    logger.info(f"Successfully created target column: '{actual_target_col}'")
                    available_cols = df.columns.tolist() # Update available columns
                else:
                    error_msg = f"Cannot create target column '{actual_target_col}': Missing required source columns 'il_total' or 'eey'. Available: {available_cols[:20]}..."
                    logger.error(error_msg)
                    # Return error status for all models if target cannot be created
                    results = {}
                    from utils.config import MODEL_CONFIG
                    model_ids = [key for key in MODEL_CONFIG.keys() if key in ['model1', 'model4']]
                    for model_id in model_ids:
                        results[model_id] = {"status": "error", "message": error_msg}
                    return results # Early exit
            else:
                 logger.info(f"Found existing target column: '{actual_target_col}'")

            # Verify target column exists after potential calculation
            if actual_target_col not in df.columns:
                 error_msg = f"Target column '{actual_target_col}' still not available after attempting calculation."
                 logger.error(error_msg)
                 results = {}
                 from utils.config import MODEL_CONFIG
                 model_ids = [key for key in MODEL_CONFIG.keys() if key in ['model1', 'model4']]
                 for model_id in model_ids:
                     results[model_id] = {"status": "error", "message": error_msg}
                 return results

            # Prepare features and target
            y = df[actual_target_col]
            # Drop the target and other potential (now unused) target candidates if they exist
            other_candidates = ['trgt', 'target', 'claim_amount']
            cols_to_drop = [actual_target_col] + [col for col in other_candidates if col in df.columns]
            X = df.drop(columns=cols_to_drop)
            
            # Handle sample weights
            sample_weight = None
            if weight_column and weight_column in df.columns:
                sample_weight = df[weight_column]
                logger.info(f"Using '{weight_column}' as sample weight")
            
            # Create train-test split with additional validation
            try:
                # Create train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Validate split data
                if len(X_train) < 30 or len(X_test) < 10:  # Reduced minimum for small datasets
                    error_msg = f"Train/test split resulted in too few samples (train: {len(X_train)}, test: {len(X_test)})"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
                logger.info(f"Created train-test split with {len(X_train)} training samples and {len(X_test)} test samples")
            except Exception as split_error:
                logger.error(f"Error creating train-test split: {str(split_error)}")
                raise
            
            # Get model IDs from MODEL_CONFIG - only use Model1 and Model4
            from utils.config import MODEL_CONFIG
            model_ids = [key for key in MODEL_CONFIG.keys() if key in ['model1', 'model4']]
            logger.info(f"Loading pretrained models: {model_ids}")
            
            if not model_ids:
                error_msg = "No valid model configurations found in MODEL_CONFIG"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Load and evaluate models
            results = {}
            
            # First attempt: Parallel loading/evaluation
            if parallel and len(model_ids) > 1:
                try:
                    logger.info(f"Attempting parallel loading/evaluation with {max_workers} workers")
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        for model_id in model_ids:
                            futures.append(
                                executor.submit(
                                    evaluate_pretrained_model,
                                    model_id=model_id,
                                    X_test=X_test,
                                    y_test=y_test
                                )
                            )
                            
                        # Collect results as they complete
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                if result and 'model_id' in result:
                                    results[result['model_id']] = result
                                    logger.info(f"Completed evaluation for model {result['model_id']} with status: {result.get('status', 'unknown')}")
                            except Exception as future_error:
                                logger.error(f"Error in parallel model evaluation: {str(future_error)}")
                
                    # Check if any models completed successfully
                    completed = sum(1 for r in results.values() if r.get('status') == 'completed')
                    logger.info(f"Parallel evaluation completed with {completed} successful models")
                    
                except Exception as parallel_error:
                    logger.error(f"Error in parallel evaluation setup: {str(parallel_error)}")
                    # Fall back to sequential evaluation
                    parallel = False
                    results = {}  # Clear any partial results
            
            # Second attempt: Sequential loading/evaluation (if parallel failed or wasn't requested)
            if not parallel or not results or not any(r.get('status') == 'completed' for r in results.values()):
                logger.info("Using sequential loading/evaluation approach")
                for model_id in model_ids:
                    try:
                        result = evaluate_pretrained_model(
                            model_id=model_id,
                            X_test=X_test,
                            y_test=y_test
                        )
                        if result:
                            results[model_id] = result
                            logger.info(f"Sequential evaluation completed for model {model_id} with status: {result.get('status', 'unknown')}")
                    except Exception as model_error:
                        logger.error(f"Error evaluating model {model_id}: {str(model_error)}")
                        results[model_id] = {
                            "status": "failed",
                            "error": str(model_error),
                            "model_id": model_id
                        }
                    
            # Check results
            completed = sum(1 for r in results.values() if r.get('status') == 'completed')
            if completed == 0:
                logger.error("No models were successfully loaded and evaluated")
                # Instead of returning a dict with status, message, results - just return results with error statuses
                if not results:
                    # Make sure we at least have entries for expected models
                    for model_id in model_ids:
                        results[model_id] = {
                            "status": "error", 
                            "message": "Model loading failed",
                            "model_id": model_id
                        }
                return results  # Return the results dictionary directly
            
            logger.info(f"Successfully loaded and evaluated {completed} models")
            return results
            
        except Exception as e:
            error_msg = f"Error in train_multiple_models: {str(e)}"
            logger.error(error_msg)
            logger.exception("Full exception details:")
            # Create a proper results dictionary with model entries instead of a status dictionary
            results = {}
            from utils.config import MODEL_CONFIG
            model_ids = [key for key in MODEL_CONFIG.keys() if key in ['model1', 'model4']]
            for model_id in model_ids:
                results[model_id] = {
                    "status": "error",
                    "message": error_msg,
                    "model_id": model_id
                }
            return results
        
    except Exception as e:
        error_msg = f"Error in train_multiple_models: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full exception details:")
        # Create a proper results dictionary with model entries instead of a status dictionary
        results = {}
        from utils.config import MODEL_CONFIG
        model_ids = [key for key in MODEL_CONFIG.keys() if key in ['model1', 'model4']]
        for model_id in model_ids:
            results[model_id] = {
                "status": "error",
                "message": error_msg,
                "model_id": model_id
            }
        return results

def evaluate_pretrained_model(model_id: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Load a pretrained model and evaluate it on test data
    
    Args:
        model_id: ID of the model to load (model1 or model4)
        X_test: Test features
        y_test: Test target variable
        
    Returns:
        Dict with model and evaluation results
    """
    try:
        # Load the pretrained model
        result = load_pretrained_model(model_id)
        
        if result.get('status') != 'completed':
            logger.error(f"Failed to load pretrained model {model_id}: {result.get('error')}")
            return result
            
        model = result.get('model')
        if model is None:
            error_msg = f"Loaded model {model_id} is None"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg, "model_id": model_id}
            
        # Select the relevant features for this model
        try:
            # Use the feature selection from preprocessing
            from utils.config import MODEL_CONFIG
            model_features = MODEL_CONFIG.get(model_id, {}).get('features', [])
            
            # Get all columns that start with any of the model features
            keep_cols = []
            for col in X_test.columns:
                if any(col.startswith(prefix) for prefix in model_features) or not any(col.startswith(prefix) for prefix in ['num_loss_', 'lhdwc_']):
                    keep_cols.append(col)
                    
            if not keep_cols:
                logger.warning(f"No matching features found for model {model_id}, using all features")
                X_test_model = X_test
            else:
                logger.info(f"Selected {len(keep_cols)} features for model {model_id}")
                X_test_model = X_test[keep_cols]
                
        except Exception as e:
            logger.warning(f"Error selecting features for model {model_id}: {str(e)}, using all features")
            X_test_model = X_test
            
        # Evaluate the model
        try:
            y_pred = model.predict(X_test_model)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model {model_id}: Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Update metrics in the result
            result['metrics'] = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            
            # Log metrics to MLflow
            run_id = result.get('run_id')
            if run_id is not None:
                try:
                    # Initialize MLflow client with proper error handling
                    # Always use the EC2 MLflow URI
                    logger.info("Using EC2 MLflow URI")
                    mlflow.set_tracking_uri("http://3.146.46.179:5000")
                    
                    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
                    
                    # Log all metrics
                    metrics_to_log = {
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                        "evaluation_time": time.time()
                    }
                    
                    for metric_name, metric_value in metrics_to_log.items():
                        try:
                            client.log_metric(run_id, metric_name, metric_value)
                        except Exception as metric_error:
                            logger.warning(f"Failed to log metric {metric_name}: {str(metric_error)}")
                    
                    # Create and log feature importance plot
                    try:
                        # Only attempt for XGBoost models
                        if hasattr(model, 'get_booster') or 'xgboost' in str(type(model)).lower():
                            feature_importance_path = f"/tmp/feature_importance_{model_id}.png"
                            plt.figure(figsize=(10, 6))
                            
                            if hasattr(model, 'feature_importances_'):
                                # For sklearn-like models
                                feature_indices = np.argsort(model.feature_importances_)[-20:]  # Top 20 features
                                plt.barh(range(len(feature_indices)), 
                                        model.feature_importances_[feature_indices])
                                plt.yticks(range(len(feature_indices)), 
                                         [X_test_model.columns[i] for i in feature_indices])
                            else:
                                # For XGBoost models
                                import xgboost as xgb
                                xgb.plot_importance(model, max_num_features=20, height=0.8, 
                                                 ax=plt.gca(), importance_type='weight')
                                
                            plt.title(f'Feature Importance (Model {model_id})')
                            plt.tight_layout()
                            plt.savefig(feature_importance_path)
                            plt.close()
                            
                            # Log the plot
                            client.log_artifact(run_id, feature_importance_path)
                            logger.info(f"Logged feature importance plot to MLflow")
                    except Exception as plot_error:
                        logger.warning(f"Failed to create or log feature importance plot: {str(plot_error)}")
                    
                    # Create and log actual vs predicted plot
                    try:
                        actual_vs_pred_path = f"/tmp/actual_vs_pred_{model_id}.png"
                        plt.figure(figsize=(8, 8))
                        plt.scatter(y_test, y_pred, alpha=0.5)
                        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                        plt.xlabel('Actual')
                        plt.ylabel('Predicted')
                        plt.title(f'Actual vs Predicted (Model {model_id})')
                        plt.tight_layout()
                        plt.savefig(actual_vs_pred_path)
                        plt.close()
                        
                        # Log the plot
                        client.log_artifact(run_id, actual_vs_pred_path)
                        logger.info(f"Logged actual vs predicted plot to MLflow")
                    except Exception as plot_error:
                        logger.warning(f"Failed to create or log actual vs predicted plot: {str(plot_error)}")
                    
                    # Set run status to completed
                    client.set_terminated(run_id, "FINISHED")
                    logger.info(f"MLflow run {run_id} updated with evaluation metrics")
                    
                except Exception as e:
                    logger.warning(f"Model {model_id}: Failed to log metrics to MLflow: {str(e)}")
                    
            return result
                
        except Exception as eval_error:
            error_msg = f"Error evaluating model {model_id}: {str(eval_error)}"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg, "model_id": model_id}
            
    except Exception as e:
        error_msg = f"Unexpected error in evaluate_pretrained_model for {model_id}: {str(e)}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg, "model_id": model_id}
