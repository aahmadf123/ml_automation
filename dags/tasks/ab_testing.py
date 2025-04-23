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
import mlflow
from mlflow.tracking import MlflowClient
from utils.config import DATA_BUCKET, MODEL_KEY_PREFIX

logger = logging.getLogger(__name__)

class ABTestingPipeline:
    def __init__(self, model_id: str, test_duration_days: int = 7):
        self.model_id = model_id
        self.test_duration_days = test_duration_days
        self.client = MlflowClient()
        
    def get_production_model(self) -> Tuple[object, float]:
        """Retrieve the current production model and its metrics."""
        try:
            prod_version = self.client.get_latest_versions(
                name=self.model_id, 
                stages=["Production"]
            )[0]
            prod_model = mlflow.pyfunc.load_model(f"runs:/{prod_version.run_id}/model")
            prod_metrics = self.client.get_run(prod_version.run_id).data.metrics
            return prod_model, prod_metrics.get("rmse", float('inf'))
        except Exception as e:
            logger.error(f"Error retrieving production model: {str(e)}")
            return None, float('inf')

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
        
        # Get production model
        prod_model, prod_rmse = self.get_production_model()
        if prod_model is None:
            logger.warning("No production model found for comparison")
            results = {
                "status": "error",
                "message": "No production model found for comparison"
            }
            return results
        
        try:
            # Make predictions with both models
            try:
                prod_preds = prod_model.predict(X_test)
                # Ensure predictions are in the right format
                if not isinstance(prod_preds, np.ndarray):
                    prod_preds = np.array(prod_preds)
            except Exception as e:
                logger.error(f"Error making predictions with production model: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error with production model: {str(e)}"
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
            
            # Calculate metrics for both models
            prod_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, prod_preds)),
                "r2": r2_score(y_test, prod_preds)
            }
            
            new_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, new_preds)),
                "r2": r2_score(y_test, new_preds)
            }
            
            # Calculate improvements
            rmse_improvement = (prod_metrics["rmse"] - new_metrics["rmse"]) / prod_metrics["rmse"] * 100
            r2_improvement = (new_metrics["r2"] - prod_metrics["r2"]) / max(0.01, abs(prod_metrics["r2"])) * 100
            
            # Perform statistical significance test
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(
                np.abs(y_test - prod_preds),
                np.abs(y_test - new_preds)
            )
            
            # Results
            results = {
                "status": "success",
                "production_metrics": prod_metrics,
                "new_model_metrics": new_metrics,
                "rmse_improvement": rmse_improvement,
                "r2_improvement": r2_improvement,
                "p_value": p_value,
                "statistical_significance": p_value < 0.05,
                "timestamp": datetime.now().isoformat()
            }

            # Send notification
            if rmse_improvement > 0 and results['statistical_significance']:
                slack_msg(
                    channel="#alerts",
                    title=f"🎯 A/B Test Results - {self.model_id}",
                    details=f"New model shows {rmse_improvement:.2f}% RMSE improvement\n"
                            f"R2 improvement: {r2_improvement:.2f}%\n"
                            f"Statistically significant: Yes",
                    urgency="high"
                )
            else:
                slack_msg(
                    channel="#alerts",
                    title=f"⚠️ A/B Test Results - {self.model_id}",
                    details=f"New model does not show significant improvement\n"
                            f"RMSE change: {rmse_improvement:.2f}%\n"
                            f"R2 change: {r2_improvement:.2f}%",
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
        
        return (
            results["rmse_improvement"] > rmse_improvement_threshold and
            results["r2_improvement"] > r2_improvement_threshold and
            results["statistical_significance"]
        ) 