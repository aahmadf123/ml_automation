import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from dags.tasks.model_explainability import ModelExplainabilityTracker
from sklearn.ensemble import RandomForestRegressor

class TestModelExplainabilityTracker(unittest.TestCase):
    def setUp(self):
        """Set up test data and model."""
        # Create sample data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        self.y = 2 * self.X['feature1'] + 3 * self.X['feature2'] + np.random.normal(0, 0.1, 100)
        
        # Train a model
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)
        
        # Initialize tracker
        self.tracker = ModelExplainabilityTracker("test_model")

    def test_calculate_feature_importance(self):
        """Test feature importance calculation."""
        importance = self.tracker.calculate_feature_importance(self.model, self.X)
        
        # Check return type and structure
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 3)  # Three features
        
        # Check values are sorted in descending order
        values = list(importance.values())
        self.assertEqual(values, sorted(values, reverse=True))
        
        # Check all features are present
        self.assertIn('feature1', importance)
        self.assertIn('feature2', importance)
        self.assertIn('feature3', importance)

    def test_calculate_shap_values(self):
        """Test SHAP values calculation."""
        shap_values, features = self.tracker.calculate_shap_values(self.model, self.X)
        
        # Check return types
        self.assertIsInstance(shap_values, np.ndarray)
        self.assertIsInstance(features, list)
        
        # Check dimensions
        self.assertEqual(shap_values.shape[0], len(self.X))
        self.assertEqual(shap_values.shape[1], len(self.X.columns))
        self.assertEqual(len(features), len(self.X.columns))

    def test_track_explainability(self):
        """Test tracking explainability metrics."""
        metrics = self.tracker.track_explainability(self.model, self.X)
        
        # Check return type
        self.assertIsInstance(metrics, dict)
        
        # Check required metrics
        required_metrics = ['feature_importance', 'shap_values', 'importance_entropy']
        for metric in required_metrics:
            self.assertIn(metric, metrics)

    def test_detect_significant_changes(self):
        """Test detection of significant changes in feature importance."""
        # First tracking
        self.tracker.track_explainability(self.model, self.X)
        
        # Modify data to create significant changes
        self.X['feature1'] *= 2
        
        # Second tracking
        changes = self.tracker.detect_significant_changes(self.model, self.X)
        
        # Check return type
        self.assertIsInstance(changes, dict)
        self.assertIn('significant_changes', changes)
        self.assertIn('changed_features', changes)

    def test_calculate_importance_entropy(self):
        """Test entropy calculation for feature importance."""
        importance = {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.2}
        entropy = self.tracker.calculate_importance_entropy(importance)
        
        # Check return type and value range
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0)
        self.assertLessEqual(entropy, 1)

    def test_get_explainability_summary(self):
        """Test getting explainability summary."""
        # Track metrics first
        self.tracker.track_explainability(self.model, self.X)
        
        # Get summary
        summary = self.tracker.get_explainability_summary()
        
        # Check return type and structure
        self.assertIsInstance(summary, dict)
        self.assertIn('top_features', summary)
        self.assertIn('recent_changes', summary)
        self.assertIn('importance_trend', summary)

if __name__ == '__main__':
    unittest.main() 