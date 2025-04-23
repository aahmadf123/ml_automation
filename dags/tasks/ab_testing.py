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
from plugins.utils.config import DATA_BUCKET, MODEL_KEY_PREFIX
from plugins.utils.slack import post as slack_msg

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
        prod_model, prod_rmse = self.get_production_model()
        if prod_model is None:
            return {"status": "error", "message": "Could not retrieve production model"}

        # Get predictions from both models
        new_preds = new_model.predict(X_test)
        prod_preds = prod_model.predict(X_test)

        # Calculate metrics
        new_rmse = np.sqrt(mean_squared_error(y_test, new_preds))
        new_r2 = r2_score(y_test, new_preds)
        prod_r2 = r2_score(y_test, prod_preds)

        # Calculate improvement percentages
        rmse_improvement = ((prod_rmse - new_rmse) / prod_rmse) * 100
        r2_improvement = ((new_r2 - prod_r2) / abs(prod_r2)) * 100

        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(new_preds, prod_preds)

        results = {
            "status": "success",
            "new_model_rmse": new_rmse,
            "prod_model_rmse": prod_rmse,
            "rmse_improvement": rmse_improvement,
            "r2_improvement": r2_improvement,
            "statistical_significance": p_value < 0.05,
            "p_value": p_value,
            "test_duration_days": self.test_duration_days
        }

        # Log results
        logger.info(f"A/B Test Results for {self.model_id}:")
        logger.info(f"RMSE Improvement: {rmse_improvement:.2f}%")
        logger.info(f"R2 Improvement: {r2_improvement:.2f}%")
        logger.info(f"Statistical Significance: {results['statistical_significance']}")

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