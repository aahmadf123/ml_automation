import unittest
import pandas as pd
import numpy as np
from tasks.model_explainability import ModelExplainabilityTracker

class TestModelExplainability(unittest.TestCase):
    """Test the model explainability tracker."""
    
    def setUp(self):
        """Set up test data and tracker."""
        # Sample model predictions and actual values
        self.predictions = np.array([0.1, 0.8, 0.3, 0.9, 0.2, 0.7])
        self.actuals = np.array([0, 1, 0, 1, 0, 1])
        
        # Sample feature values
        self.features = pd.DataFrame({
            'feature1': [1.0, 2.0, 1.5, 3.0, 0.5, 2.5],
            'feature2': [0.5, 1.5, 1.0, 2.0, 0.2, 1.8],
            'feature3': [100, 200, 150, 250, 50, 220]
        })
        
        # Initialize tracker
        self.tracker = ModelExplainabilityTracker(model_id='test_model')
        
    def test_track_prediction_explanations(self):
        """Test prediction explanations tracking."""
        result = self.tracker.track_prediction_explanations(
            predictions=self.predictions,
            actuals=self.actuals,
            features=self.features
        )
        
        # Verify the result contains the expected keys
        self.assertIn('explanations_generated', result)
        self.assertIn('feature_importance', result)
        self.assertIn('global_explanations', result)
        self.assertIn('local_explanations', result)
        
        # Verify feature importance contains all features
        for feature in self.features.columns:
            self.assertIn(feature, result['feature_importance'])
            
        # Verify number of local explanations matches number of predictions
        self.assertEqual(len(result['local_explanations']), len(self.predictions))
        
    def test_compare_explanations(self):
        """Test comparison of explanations between models."""
        # Create another tracker for a different model
        tracker2 = ModelExplainabilityTracker(model_id='test_model2')
        
        # Generate explanations for both models
        result1 = self.tracker.track_prediction_explanations(
            predictions=self.predictions,
            actuals=self.actuals,
            features=self.features
        )
        
        # Slightly modify predictions for second model
        predictions2 = self.predictions + np.random.normal(0, 0.05, size=len(self.predictions))
        predictions2 = np.clip(predictions2, 0, 1)  # Keep in [0,1] range
        
        result2 = tracker2.track_prediction_explanations(
            predictions=predictions2,
            actuals=self.actuals,
            features=self.features
        )
        
        # Compare explanations
        comparison = self.tracker.compare_explanations(
            other_explanations=result2['global_explanations']
        )
        
        # Verify comparison result
        self.assertIn('models_compared', comparison)
        self.assertIn('feature_importance_diff', comparison)
        self.assertIn('similarity_score', comparison)
        
        # Verify models are correctly identified
        self.assertEqual(comparison['models_compared']['model1'], 'test_model')
        self.assertEqual(comparison['models_compared']['model2'], 'test_model2')
        
        # Verify similarity score is between 0 and 1
        self.assertGreaterEqual(comparison['similarity_score'], 0)
        self.assertLessEqual(comparison['similarity_score'], 1)
        
    def test_generate_explanation_report(self):
        """Test generation of explanation report."""
        # Track explanations first
        self.tracker.track_prediction_explanations(
            predictions=self.predictions,
            actuals=self.actuals,
            features=self.features
        )
        
        # Generate report
        report = self.tracker.generate_explanation_report()
        
        # Verify report contains expected sections
        self.assertIn('model_id', report)
        self.assertIn('global_importance', report)
        self.assertIn('top_features', report)
        self.assertIn('example_explanations', report)
        
        # Verify model ID is correct
        self.assertEqual(report['model_id'], 'test_model')
        
        # Verify top features are included
        self.assertGreaterEqual(len(report['top_features']), 1)
        
        # Verify example explanations are included
        self.assertGreaterEqual(len(report['example_explanations']), 1)

if __name__ == '__main__':
    unittest.main() 