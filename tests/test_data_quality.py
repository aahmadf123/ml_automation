import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from dags.tasks.data_quality import DataQualityMonitor

class TestDataQualityMonitor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create sample data
        np.random.seed(42)
        self.df = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.normal(0, 1, 100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'binary': np.random.choice([0, 1], 100),
            'missing': np.random.choice([1, 2, 3, np.nan], 100),
            'outlier': np.concatenate([np.random.normal(0, 1, 95), np.array([100, 200, 300, 400, 500])])
        })
        
        # Initialize monitor
        self.monitor = DataQualityMonitor()

    def test_calculate_basic_stats(self):
        """Test basic statistics calculation."""
        stats = self.monitor.calculate_basic_stats(self.df)
        
        # Check return type
        self.assertIsInstance(stats, dict)
        
        # Check numeric column stats
        numeric_stats = stats['numeric1']
        self.assertIn('mean', numeric_stats)
        self.assertIn('std', numeric_stats)
        self.assertIn('min', numeric_stats)
        self.assertIn('max', numeric_stats)
        self.assertIn('missing', numeric_stats)
        self.assertIn('unique_values', numeric_stats)
        
        # Check categorical column stats
        categorical_stats = stats['categorical']
        self.assertIn('missing', categorical_stats)
        self.assertIn('unique_values', categorical_stats)
        self.assertIn('most_common', categorical_stats)

    def test_detect_outliers(self):
        """Test outlier detection."""
        outliers = self.monitor.detect_outliers(self.df)
        
        # Check return type
        self.assertIsInstance(outliers, dict)
        
        # Check outlier detection for column with known outliers
        self.assertTrue(any(outliers['outlier']))
        
        # Check outlier detection for column without outliers
        self.assertFalse(any(outliers['numeric1']))

    def test_detect_drift(self):
        """Test drift detection."""
        # Set baseline
        self.monitor.baseline_stats = self.monitor.calculate_basic_stats(self.df)
        
        # Create drifted data
        drifted_df = self.df.copy()
        drifted_df['numeric1'] *= 2  # Significant drift
        
        # Detect drift
        drift = self.monitor.detect_drift(
            self.monitor.calculate_basic_stats(drifted_df),
            self.monitor.baseline_stats
        )
        
        # Check return type and content
        self.assertIsInstance(drift, dict)
        self.assertIn('numeric1', drift)
        self.assertGreater(drift['numeric1']['drift_score'], 0.1)

    def test_detect_correlations(self):
        """Test correlation detection."""
        # Create correlated data
        correlated_df = self.df.copy()
        correlated_df['correlated'] = correlated_df['numeric1'] * 0.8 + np.random.normal(0, 0.1, 100)
        
        correlations = self.monitor.detect_correlations(correlated_df)
        
        # Check return type
        self.assertIsInstance(correlations, list)
        
        # Check correlation detection
        found_correlation = False
        for corr in correlations:
            if 'numeric1' in corr['features'] and 'correlated' in corr['features']:
                found_correlation = True
                self.assertGreater(abs(corr['correlation']), 0.7)
        self.assertTrue(found_correlation)

    def test_monitor_data_quality(self):
        """Test data quality monitoring."""
        # Test baseline monitoring
        baseline_report = self.monitor.monitor_data_quality(self.df, is_baseline=True)
        
        # Check return type and structure
        self.assertIsInstance(baseline_report, dict)
        self.assertIn('timestamp', baseline_report)
        self.assertIn('basic_stats', baseline_report)
        self.assertIn('issues', baseline_report)
        
        # Test monitoring with issues
        df_with_issues = self.df.copy()
        df_with_issues['missing'] = np.nan  # Create missing values issue
        
        report = self.monitor.monitor_data_quality(df_with_issues)
        
        # Check issues detection
        self.assertTrue(any(issue['type'] == 'missing_values' for issue in report['issues']))

    def test_get_quality_summary(self):
        """Test quality summary generation."""
        # Generate some history
        self.monitor.monitor_data_quality(self.df, is_baseline=True)
        self.monitor.monitor_data_quality(self.df)
        
        summary = self.monitor.get_quality_summary()
        
        # Check return type and structure
        self.assertIsInstance(summary, dict)
        self.assertIn('total_issues', summary)
        self.assertIn('issues_by_type', summary)
        self.assertIn('recent_issues', summary)

if __name__ == '__main__':
    unittest.main() 