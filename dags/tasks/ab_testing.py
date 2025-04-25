#!/usr/bin/env python3
"""
tasks/ab_testing.py

Handles:
  - A/B testing pipeline for model comparison
  - Statistical significance testing
  - Performance metrics comparison
  - Automated model promotion based on test results
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, r2_score
from clearml import Task, Model
from utils.config import DATA_BUCKET, MODEL_KEY_PREFIX
import joblib
import boto3

logger = logging.getLogger(__name__)

class ABTestingPipeline:
    def __init__(self, model_id: str, test_duration_days: int = 7):
        self.model_id = model_id
        self.test_duration_days = test_duration_days
        self.task = Task.init(project_name="ABTesting", task_name=f"ABTest_{model_id}")
        
    def get_production_model(self) -> Tuple[object, float]:
        """Retrieve the current production model and its metrics."""
        try:
            model = Model(model_id=self.model_id)
            prod_model = model.get_local_copy()
            prod_metrics = model.get_model_metrics()
            return joblib.load(prod_model), prod_metrics.get("rmse", float('inf'))
        except Exception as e:
            logger.error(f"Error retrieving production model: {str(e)}")
            return None, float('inf')

    def load_model_from_s3(self, model_name: str) -> object:
        """Load a model from S3."""
        s3 = boto3.client('s3')
        model_path = f"/tmp/{model_name}.joblib"
        s3.download_file(DATA_BUCKET, f"{MODEL_KEY_PREFIX}/{model_name}.joblib", model_path)
        return joblib.load(model_path)

    def run_ab_test(self, new_model: object, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Run A/B test between new and production models."""
        # Import slack only when needed
        from utils.slack import post as slack_msg
        
        # Validate inputs
        if new_model is None:
            logger.error("New model is None, cannot perform A/B test")
            return {
                "status": "error",
                "message": "New model is None"
            }
            
        if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
            logger.error("Test data is empty or None")
            return {
                "status": "error",
                "message": "Test data is empty or None"
            }
        
        # Load production models
        try:
            model1 = self.load_model_from_s3("Model1")
            model4 = self.load_model_from_s3("Model4")
        except Exception as e:
            logger.error(f"Error loading models from S3: {str(e)}")
            return {
                "status": "error",
                "message": f"Error loading models from S3: {str(e)}"
            }

        try:
            # Make predictions with all models
            try:
                model1_preds = model1.predict(X_test)
                model4_preds = model4.predict(X_test)
                # Ensure predictions are in the right format
                if not isinstance(model1_preds, np.ndarray):
                    model1_preds = np.array(model1_preds)
                if not isinstance(model4_preds, np.ndarray):
                    model4_preds = np.array(model4_preds)
            except Exception as e:
                logger.error(f"Error making predictions with production models: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error with production models: {str(e)}"
                }
                
            try:
                new_preds = new_model.predict(X_test)
                # Ensure predictions are in the right format
                if not isinstance(new_preds, np.ndarray):
                    new_preds = np.array(new_preds)
            except Exception as e:
                logger.error(f"Error making predictions with new model: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error with new model: {str(e)}"
                }
            
            # Calculate metrics for all models
            model1_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, model1_preds)),
                "r2": r2_score(y_test, model1_preds)
            }
            
            model4_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, model4_preds)),
                "r2": r2_score(y_test, model4_preds)
            }
            
            new_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, new_preds)),
                "r2": r2_score(y_test, new_preds)
            }
            
            # Calculate improvements
            model1_rmse_improvement = (model1_metrics["rmse"] - new_metrics["rmse"]) / model1_metrics["rmse"] * 100
            model4_rmse_improvement = (model4_metrics["rmse"] - new_metrics["rmse"]) / model4_metrics["rmse"] * 100
            model1_r2_improvement = (new_metrics["r2"] - model1_metrics["r2"]) / max(0.01, abs(model1_metrics["r2"])) * 100
            model4_r2_improvement = (new_metrics["r2"] - model4_metrics["r2"]) / max(0.01, abs(model4_metrics["r2"])) * 100
            
            # Perform statistical significance test
            from scipy import stats
            t_stat1, p_value1 = stats.ttest_rel(
                np.abs(y_test - model1_preds),
                np.abs(y_test - new_preds)
            )
            t_stat4, p_value4 = stats.ttest_rel(
                np.abs(y_test - model4_preds),
                np.abs(y_test - new_preds)
            )
            
            # Results
            results = {
                "status": "success",
                "model1_metrics": model1_metrics,
                "model4_metrics": model4_metrics,
                "new_model_metrics": new_metrics,
                "model1_rmse_improvement": model1_rmse_improvement,
                "model4_rmse_improvement": model4_rmse_improvement,
                "model1_r2_improvement": model1_r2_improvement,
                "model4_r2_improvement": model4_r2_improvement,
                "model1_p_value": p_value1,
                "model4_p_value": p_value4,
                "model1_statistical_significance": p_value1 < 0.05,
                "model4_statistical_significance": p_value4 < 0.05,
                "timestamp": datetime.now().isoformat()
            }

            # Send notification
            if (model1_rmse_improvement > 0 and results['model1_statistical_significance']) or (model4_rmse_improvement > 0 and results['model4_statistical_significance']):
                slack_msg(
                    channel="#alerts",
                    title=f"🎯 A/B Test Results - {self.model_id}",
                    details=f"New model shows improvement over production models\n"
                            f"Model1 RMSE improvement: {model1_rmse_improvement:.2f}%\n"
                            f"Model4 RMSE improvement: {model4_rmse_improvement:.2f}%\n"
                            f"Model1 R2 improvement: {model1_r2_improvement:.2f}%\n"
                            f"Model4 R2 improvement: {model4_r2_improvement:.2f}%\n"
                            f"Statistically significant: Yes",
                    urgency="high"
                )
            else:
                slack_msg(
                    channel="#alerts",
                    title=f"⚠️ A/B Test Results - {self.model_id}",
                    details=f"New model does not show significant improvement\n"
                            f"Model1 RMSE change: {model1_rmse_improvement:.2f}%\n"
                            f"Model4 RMSE change: {model4_rmse_improvement:.2f}%\n"
                            f"Model1 R2 change: {model1_r2_improvement:.2f}%\n"
                            f"Model4 R2 change: {model4_r2_improvement:.2f}%",
                    urgency="medium"
                )

            return results
        except Exception as e:
            logger.error(f"Error during A/B testing: {str(e)}")
            return {
                "status": "error",
                "message": f"Error during A/B testing: {str(e)}"
            }

    def should_promote_new_model(self, results: Dict) -> bool:
        """Determine if new model should be promoted based on A/B test results."""
        if results["status"] != "success":
            return False

        # Criteria for promotion
        rmse_improvement_threshold = 5.0  # 5% improvement
        r2_improvement_threshold = 2.0    # 2% improvement
        
        if (results["model1_rmse_improvement"] > rmse_improvement_threshold and
            results["model1_r2_improvement"] > r2_improvement_threshold and
            results["model1_statistical_significance"]) or (
            results["model4_rmse_improvement"] > rmse_improvement_threshold and
            results["model4_r2_improvement"] > r2_improvement_threshold and
            results["model4_statistical_significance"]):
            # Raise a question to the human-in-the-loop
            question = f"New model shows significant improvement over production models. Should we make the new model the production model?"
            logger.info(question)
            
            # Send the question to Slack
            try:
                from utils.slack import post as slack_post
                slack_post(question, channel="#model-tuning")
            except Exception as e:
                logger.warning(f"Failed to send Slack notification: {str(e)}")
            
            return True
        
        return False
