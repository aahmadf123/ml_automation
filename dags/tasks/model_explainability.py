import logging
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple
import mlflow
from datetime import datetime
from plugins.utils.slack import post as send_message

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

    def track_explainability(self, model: object, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Track model explainability metrics and detect significant changes."""
        current_time = datetime.now()
        
        # Calculate current metrics
        feature_importance = self.calculate_feature_importance(model, X)
        shap_values, feature_names = self.calculate_shap_values(model, X)
        
        # Store in history
        self.feature_importance_history.append({
            'timestamp': current_time,
            'importance': feature_importance
        })
        
        if len(shap_values) > 0:
            self.shap_values_history.append({
                'timestamp': current_time,
                'values': shap_values,
                'features': feature_names
            })
        
        # Detect significant changes if we have history
        changes = self.detect_significant_changes()
        
        # Log to MLflow
        mlflow.log_metrics({
            'top_feature_importance': list(feature_importance.values())[0] if feature_importance else 0,
            'feature_importance_entropy': self.calculate_importance_entropy(feature_importance)
        })
        
        # Store feature importance plot
        if feature_importance:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(feature_importance)), list(feature_importance.values()))
            plt.xticks(range(len(feature_importance)), list(feature_importance.keys()), rotation=45)
            plt.title('Feature Importance')
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "feature_importance.png")
            plt.close()
        
        return {
            'feature_importance': feature_importance,
            'significant_changes': changes,
            'timestamp': current_time
        }

    def detect_significant_changes(self) -> List[Dict]:
        """Detect significant changes in feature importance over time."""
        if len(self.feature_importance_history) < 2:
            return []
        
        changes = []
        current = self.feature_importance_history[-1]['importance']
        previous = self.feature_importance_history[-2]['importance']
        
        # Calculate relative changes
        for feature in current:
            if feature in previous:
                change = ((current[feature] - previous[feature]) / previous[feature]) * 100
                if abs(change) > 10:  # 10% change threshold
                    changes.append({
                        'feature': feature,
                        'change_percentage': change,
                        'current_value': current[feature],
                        'previous_value': previous[feature]
                    })
        
        # Alert on significant changes
        if changes:
            send_message(
                channel="#alerts",
                title=f"🔍 Significant Feature Importance Changes - {self.model_id}",
                details="\n".join([
                    f"{c['feature']}: {c['change_percentage']:.1f}% change"
                    for c in changes
                ]),
                urgency="medium"
            )
        
        return changes

    def calculate_importance_entropy(self, importance: Dict[str, float]) -> float:
        """Calculate entropy of feature importance distribution."""
        if not importance:
            return 0.0
        
        values = np.array(list(importance.values()))
        values = values / values.sum()  # Normalize
        return -np.sum(values * np.log2(values + 1e-10))

    def get_explainability_summary(self) -> Dict:
        """Get a summary of model explainability metrics."""
        if not self.feature_importance_history:
            return {}
        
        current = self.feature_importance_history[-1]['importance']
        top_features = list(current.items())[:5]
        
        return {
            'top_features': top_features,
            'importance_entropy': self.calculate_importance_entropy(current),
            'recent_changes': self.detect_significant_changes(),
            'history_length': len(self.feature_importance_history)
        } 