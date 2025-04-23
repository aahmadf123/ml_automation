import logging
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple
import mlflow
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelExplainabilityTracker:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.feature_importance_history = []
        self.shap_values_history = []
        
    def calculate_feature_importance(self, model: object, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                # For models without built-in feature importance
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                importances = np.abs(shap_values).mean(0)
            
            feature_importance = dict(zip(X.columns, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}

    def calculate_shap_values(self, model: object, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Calculate SHAP values for the model."""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            return shap_values, X.columns.tolist()
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            return np.array([]), []
    
    def detect_feature_shift(self, current_importance: Dict[str, float], threshold: float = 0.1) -> Dict[str, float]:
        """Detect shifts in feature importance from historical data."""
        if not self.feature_importance_history:
            return {}
            
        last_importance = self.feature_importance_history[-1]
        shifts = {}
        
        for feature, importance in current_importance.items():
            if feature in last_importance:
                shift = abs(importance - last_importance[feature])
                if shift > threshold:
                    shifts[feature] = shift
                    
        if shifts:
            # Import slack only when needed
            from utils.slack import post as send_message
            shift_msg = "\n".join([f"{feature}: {shift:.4f}" for feature, shift in shifts.items()])
            send_message(
                channel="#alerts",
                title="🔄 Feature Importance Shift Detected",
                details=f"Shifts detected in feature importance:\n{shift_msg}",
                urgency="medium"
            )
                    
        return shifts
    
    def track_model_and_data(self, 
                           model: object, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           run_id: str = None) -> Dict:
        """
        Track model explainability metrics and log them to MLflow.
        
        Args:
            model: Trained model object
            X: Feature data
            y: Target data
            run_id: MLflow run ID (optional)
            
        Returns:
            Dict of explainability metrics
        """
        # Check for None values in inputs
        if model is None:
            logger.error("Cannot track explainability: model is None")
            return {
                'status': 'error',
                'message': 'Model is None',
                'timestamp': datetime.now().isoformat()
            }
            
        if X is None or len(X) == 0:
            logger.error("Cannot track explainability: feature data is empty or None")
            return {
                'status': 'error',
                'message': 'Feature data is empty or None',
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate and store feature importance
        feature_importance = self.calculate_feature_importance(model, X)
        self.feature_importance_history.append(feature_importance)
        
        # Calculate SHAP values
        shap_values, feature_names = self.calculate_shap_values(model, X)
        if len(shap_values) > 0:
            self.shap_values_history.append({
                'values': shap_values,
                'features': feature_names,
                'timestamp': datetime.now().isoformat()
            })
        
        # Detect feature importance shifts
        feature_shifts = self.detect_feature_shift(feature_importance)
        
        # Log to MLflow if run_id is provided
        if run_id:
            try:
                with mlflow.start_run(run_id=run_id):
                    # Log feature importance
                    for feature, importance in feature_importance.items():
                        mlflow.log_metric(f"importance_{feature}", importance)
                        
                    # Log feature shifts
                    for feature, shift in feature_shifts.items():
                        mlflow.log_metric(f"shift_{feature}", shift)
                    
                    # Log summary metrics
                    mlflow.log_metric("num_features", len(feature_importance))
                    mlflow.log_metric("num_feature_shifts", len(feature_shifts))
                    if feature_importance:
                        mlflow.log_metric("avg_importance", 
                                         sum(feature_importance.values()) / len(feature_importance))
                logger.info(f"Logged explainability metrics to MLflow run {run_id}")
            except Exception as e:
                logger.error(f"Failed to log to MLflow: {str(e)}")
                # Continue execution despite MLflow error
        else:
            logger.info("No MLflow run_id provided, skipping MLflow logging")
        
        # Convert NumPy arrays to lists for JSON serialization
        serializable_feature_importance = {k: float(v) for k, v in feature_importance.items()}
        serializable_feature_shifts = {k: float(v) for k, v in feature_shifts.items()}
        
        # Return results
        results = {
            'status': 'success',
            'feature_importance': serializable_feature_importance,
            'feature_shifts': serializable_feature_shifts,
            'has_shap_values': len(shap_values) > 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def generate_explainability_report(self, output_path: str) -> str:
        """Generate an explainability report with historical data."""
        if not self.feature_importance_history:
            return "No historical data available for reporting."
            
        # Import slack only when needed
        from utils.slack import post as send_message
        send_message(
            channel="#alerts",
            title="📊 Explainability Report",
            details=f"Model explainability report generated for {self.model_id}",
            urgency="low"
        )
        
        # Generate simple text report for now
        report = f"# Model Explainability Report - {self.model_id}\n\n"
        report += f"Generated on: {datetime.now().isoformat()}\n\n"
        
        # Add feature importance history
        report += "## Feature Importance History\n\n"
        for i, importance in enumerate(self.feature_importance_history):
            report += f"### Snapshot {i+1}\n"
            for feature, value in importance.items():
                report += f"- {feature}: {value:.4f}\n"
            report += "\n"
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(report)
            
        return output_path 