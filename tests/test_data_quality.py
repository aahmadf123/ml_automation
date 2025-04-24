import unittest
import pandas as pd
import numpy as np
from tasks.data_quality import DataQualityMonitor

class TestDataQualityMonitor(unittest.TestCase):
    """Test the data quality monitoring functionality."""
    
    def setUp(self):
        """Set up test data and monitor."""
        # Create sample test data
        self.test_data = pd.DataFrame({
            'numeric_col1': [1.0, 2.0, 3.0, 4.0, 5.0, None],
            'numeric_col2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'categorical_col': ['A', 'B', 'C', 'A', None, 'B'],
            'date_col': pd.date_range(start='2023-01-01', periods=6)
        })
        
        # Create quality monitor
        self.monitor = DataQualityMonitor()
        
    def test_check_missing_values(self):
        """Test missing values detection."""
        result = self.monitor.check_missing_values(self.test_data)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('missing_counts', result)
        self.assertIn('missing_percentages', result)
        self.assertIn('total_missing_percentage', result)
        
        # Verify correct missing value counts
        self.assertEqual(result['missing_counts']['numeric_col1'], 1)
        self.assertEqual(result['missing_counts']['numeric_col2'], 0)
        self.assertEqual(result['missing_counts']['categorical_col'], 1)
        self.assertEqual(result['missing_counts']['date_col'], 0)
        
        # Verify percentages are calculated correctly
        self.assertEqual(result['missing_percentages']['numeric_col1'], 1/6 * 100)
        
    def test_check_duplicate_rows(self):
        """Test duplicate row detection."""
        # Add a duplicate row
        duplicate_data = pd.concat([self.test_data, pd.DataFrame([self.test_data.iloc[0]])], ignore_index=True)
        
        result = self.monitor.check_duplicate_rows(duplicate_data)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('duplicate_count', result)
        self.assertIn('duplicate_percentage', result)
        self.assertIn('has_duplicates', result)
        
        # Verify correct duplicate count
        self.assertEqual(result['duplicate_count'], 1)
        self.assertEqual(result['duplicate_percentage'], 1/7 * 100)
        self.assertTrue(result['has_duplicates'])
        
    def test_check_outliers(self):
        """Test outlier detection."""
        # Add outliers
        outlier_data = self.test_data.copy()
        outlier_data.loc[6] = [1000.0, 2000.0, 'D', pd.Timestamp('2023-01-07')]
        
        result = self.monitor.check_outliers(outlier_data, method='zscore')
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('outlier_counts', result)
        self.assertIn('outlier_percentages', result)
        self.assertIn('outlier_rows', result)
        
        # Verify outliers were detected in the correct columns
        self.assertGreater(result['outlier_counts']['numeric_col1'], 0)
        self.assertGreater(result['outlier_counts']['numeric_col2'], 0)
        
    def test_check_data_types(self):
        """Test data type validation."""
        result = self.monitor.check_data_types(self.test_data)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('column_types', result)
        self.assertIn('numeric_columns', result)
        self.assertIn('categorical_columns', result)
        self.assertIn('datetime_columns', result)
        
        # Verify correct classification of columns
        self.assertIn('numeric_col1', result['numeric_columns'])
        self.assertIn('numeric_col2', result['numeric_columns'])
        self.assertIn('categorical_col', result['categorical_columns'])
        self.assertIn('date_col', result['datetime_columns'])
        
    def test_run_quality_checks(self):
        """Test running all quality checks."""
        result = self.monitor.run_quality_checks(self.test_data)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('missing_values', result)
        self.assertIn('duplicates', result)
        self.assertIn('outliers', result)
        self.assertIn('data_types', result)
        self.assertIn('summary_statistics', result)
        self.assertIn('quality_score', result)
        
        # Verify quality score is between 0 and 100
        self.assertGreaterEqual(result['quality_score'], 0)
        self.assertLessEqual(result['quality_score'], 100)
        
    def test_generate_quality_report(self):
        """Test quality report generation."""
        # Run quality checks first
        checks_result = self.monitor.run_quality_checks(self.test_data)
        
        # Generate report
        report = self.monitor.generate_quality_report(checks_result)
        
        # Verify report structure
        self.assertIsInstance(report, dict)
        self.assertIn('overview', report)
        self.assertIn('detailed_issues', report)
        self.assertIn('recommendations', report)
        
        # Verify overview contains quality score
        self.assertIn('quality_score', report['overview'])
        
        # Verify recommendations are provided
        self.assertGreater(len(report['recommendations']), 0)

if __name__ == '__main__':
    unittest.main() 