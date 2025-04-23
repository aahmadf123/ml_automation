import logging
import pandas as pd
from typing import Dict
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, roc_auc_score, confusion_matrix, classification_report
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ModelEvaluation:
    def __init__(self, task: str):
        self.task = task
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
        
        # Classification metrics if task is classification
        if self.task == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                # Get probability predictions if available
                y_prob = model.predict_proba(X_test)
                if hasattr(model, 'classes_'):
                    n_classes = len(model.classes_)
                    if n_classes == 2:  # Binary classification
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
                    else:  # Multi-class classification
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
            except (AttributeError, ValueError) as e:
                logger.warning(f"Could not calculate ROC AUC: {str(e)}")
                
        # Regression metrics if task is regression
        elif self.task == 'regression':
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
                    
                    # Log confusion matrix for classification
                    if self.task == 'classification':
                        conf_matrix = confusion_matrix(y_test, y_pred)
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title('Confusion Matrix')
                        
                        # Save confusion matrix and log as artifact
                        conf_matrix_path = f"confusion_matrix_{timestamp}.png"
                        plt.savefig(conf_matrix_path)
                        mlflow.log_artifact(conf_matrix_path)
                        os.remove(conf_matrix_path)  # Clean up
                        
                        # Create classification report and log as text
                        report = classification_report(y_test, y_pred)
                        report_path = f"classification_report_{timestamp}.txt"
                        with open(report_path, 'w') as f:
                            f.write(report)
                        mlflow.log_artifact(report_path)
                        os.remove(report_path)  # Clean up
                logger.info(f"Logged evaluation metrics to MLflow run {run_id}")
            except Exception as e:
                logger.error(f"Failed to log to MLflow: {str(e)}")
                # Continue execution despite MLflow error
        else:
            logger.info("No MLflow run_id provided, skipping MLflow logging")
                
        # Convert NumPy values to Python native types for JSON
        serializable_metrics = {k: float(v) for k, v in metrics.items()}
        
        # Return results
        results = {
            'status': 'success',
            'metrics': serializable_metrics,
            'timestamp': timestamp,
            'task': self.task,
            'samples': len(X_test)
        }
        
        return results 