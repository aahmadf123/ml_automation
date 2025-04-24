import logging
import pandas as pd
from typing import Dict
import mlflow
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ModelEvaluation:
    def __init__(self):
        self.task = 'regression'
        self.metrics_history = []

    def evaluate_model(self, 
                       model: object, 
                       X_test: pd.DataFrame, 
                       y_test: pd.Series,
                       run_id: str = None) -> Dict:
        """
        Evaluate model performance and log metrics to MLflow.
        
        Args:
            model: Trained model object
            X_test: Test feature data
            y_test: Test target data
            run_id: MLflow run ID (optional)
            
        Returns:
            Dict of evaluation metrics
        """
        # Check for None values in inputs
        if model is None:
            logger.error("Cannot evaluate model: model is None")
            return {
                'status': 'error',
                'message': 'Model is None',
                'timestamp': datetime.now().isoformat()
            }
            
        if X_test is None or len(X_test) == 0:
            logger.error("Cannot evaluate model: test data is empty or None")
            return {
                'status': 'error',
                'message': 'Test data is empty or None',
                'timestamp': datetime.now().isoformat()
            }
            
        if y_test is None or len(y_test) == 0:
            logger.error("Cannot evaluate model: test labels are empty or None")
            return {
                'status': 'error',
                'message': 'Test labels are empty or None',
                'timestamp': datetime.now().isoformat()
            }
        
        # Get predictions
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error making predictions: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
            
        # Check prediction shape matches target shape
        if len(y_pred) != len(y_test):
            logger.error(f"Prediction shape {len(y_pred)} doesn't match target shape {len(y_test)}")
            return {
                'status': 'error',
                'message': f'Prediction shape mismatch: {len(y_pred)} vs {len(y_test)}',
                'timestamp': datetime.now().isoformat()
            }
            
        # Calculate metrics
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
            
        # Track metrics history
        timestamp = datetime.now().isoformat()
        self.metrics_history.append({
            'metrics': metrics,
            'timestamp': timestamp,
            'samples': len(X_test)
        })
        
        # Log to MLflow if run_id is provided
        if run_id:
            try:
                with mlflow.start_run(run_id=run_id):
                    # Log all metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Log additional metadata
                    mlflow.log_metric("test_samples", len(X_test))
                    
                logger.info(f"Logged evaluation metrics to MLflow run {run_id}")
            except Exception as e:
                logger.error(f"Failed to log metrics to MLflow: {str(e)}")
        
        return {
            'status': 'success',
            'metrics': metrics,
            'timestamp': timestamp
        } 