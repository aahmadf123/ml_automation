import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the preprocessing module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the preprocessing functions
from tasks.preprocessing import (
    load_data,
    clean_data,
    feature_engineering,
    handle_missing_values,
    detect_outliers,
    normalize_features,
    encode_categorical_variables,
    split_data,
    preprocess_homeowner_data,
    analyze_loss_history,
    encode_categorical_features
)

class TestPreprocessing(unittest.TestCase):
    """Test cases for the preprocessing module."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'claim_id': range(1, 11),
            'property_value': [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000],
            'claim_amount': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            'policy_age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'customer_age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'claim_frequency': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'deductible': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
            'coverage_type': ['basic', 'premium', 'basic', 'premium', 'basic', 'premium', 'basic', 'premium', 'basic', 'premium'],
            'region': ['north', 'south', 'east', 'west', 'north', 'south', 'east', 'west', 'north', 'south'],
            'vehicle_age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'credit_score': [600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050],
            'claim_date': [datetime.now() - timedelta(days=i) for i in range(10)]
        })
        
        # Create a DataFrame with missing values for testing
        self.data_with_missing = self.sample_data.copy()
        self.data_with_missing.loc[0, 'property_value'] = np.nan
        self.data_with_missing.loc[1, 'claim_amount'] = np.nan
        self.data_with_missing.loc[2, 'customer_age'] = np.nan
        
        # Create a DataFrame with outliers for testing
        self.data_with_outliers = self.sample_data.copy()
        self.data_with_outliers.loc[0, 'property_value'] = 10000000  # Extreme outlier
        self.data_with_outliers.loc[1, 'claim_amount'] = 100000  # Extreme outlier
        self.data_with_outliers.loc[2, 'customer_age'] = 150  # Extreme outlier
    
    def test_load_data(self):
        """Test the load_data function."""
        # This is a smoke test - we're just checking that the function doesn't raise an exception
        # In a real test, we would use a mock or a test file
        try:
            # For now, we'll just pass the sample data directly
            data = load_data(self.sample_data)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 10)
        except Exception as e:
            self.fail(f"load_data raised an exception: {e}")
    
    def test_clean_data(self):
        """Test the clean_data function."""
        # Test with sample data
        cleaned_data = clean_data(self.sample_data)
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        
        # Check that the function doesn't change the number of rows
        self.assertEqual(len(cleaned_data), len(self.sample_data))
        
        # Check that the function doesn't change the number of columns
        self.assertEqual(len(cleaned_data.columns), len(self.sample_data.columns))
    
    def test_handle_missing_values(self):
        """Test the handle_missing_values function."""
        # Test with data containing missing values
        data_without_missing = handle_missing_values(self.data_with_missing)
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(data_without_missing, pd.DataFrame)
        
        # Check that there are no missing values in the result
        self.assertFalse(data_without_missing.isnull().any().any())
        
        # Check that the function doesn't change the number of rows
        self.assertEqual(len(data_without_missing), len(self.data_with_missing))
    
    def test_detect_outliers(self):
        """Test the detect_outliers function."""
        # Test with data containing outliers
        outliers = detect_outliers(self.data_with_outliers)
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(outliers, pd.DataFrame)
        
        # Check that the function detects the outliers we added
        self.assertTrue(outliers.loc[0, 'property_value'])
        self.assertTrue(outliers.loc[1, 'claim_amount'])
        self.assertTrue(outliers.loc[2, 'customer_age'])
        
        # Check that the function doesn't detect outliers in normal data
        self.assertFalse(outliers.loc[3, 'property_value'])
        self.assertFalse(outliers.loc[4, 'claim_amount'])
        self.assertFalse(outliers.loc[5, 'customer_age'])
    
    def test_normalize_features(self):
        """Test the normalize_features function."""
        # Test with sample data
        normalized_data = normalize_features(self.sample_data)
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(normalized_data, pd.DataFrame)
        
        # Check that the function doesn't change the number of rows
        self.assertEqual(len(normalized_data), len(self.sample_data))
        
        # Check that numeric columns are normalized (mean close to 0, std close to 1)
        numeric_cols = ['property_value', 'claim_amount', 'policy_age', 'customer_age', 
                        'claim_frequency', 'deductible', 'vehicle_age', 'credit_score']
        
        for col in numeric_cols:
            if col in normalized_data.columns:
                self.assertAlmostEqual(normalized_data[col].mean(), 0, places=1)
                self.assertAlmostEqual(normalized_data[col].std(), 1, places=1)
    
    def test_encode_categorical_variables(self):
        """Test the encode_categorical_variables function."""
        # Test with sample data
        encoded_data = encode_categorical_variables(self.sample_data)
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(encoded_data, pd.DataFrame)
        
        # Check that categorical columns are encoded
        categorical_cols = ['coverage_type', 'region']
        
        for col in categorical_cols:
            # Check that the original column is not in the result
            self.assertNotIn(col, encoded_data.columns)
            
            # Check that one-hot encoded columns are created
            if col == 'coverage_type':
                self.assertIn('coverage_type_basic', encoded_data.columns)
                self.assertIn('coverage_type_premium', encoded_data.columns)
            elif col == 'region':
                self.assertIn('region_north', encoded_data.columns)
                self.assertIn('region_south', encoded_data.columns)
                self.assertIn('region_east', encoded_data.columns)
                self.assertIn('region_west', encoded_data.columns)
    
    def test_feature_engineering(self):
        """Test the feature_engineering function."""
        # Test with sample data
        engineered_data = feature_engineering(self.sample_data)
        
        # Check that the function returns a DataFrame
        self.assertIsInstance(engineered_data, pd.DataFrame)
        
        # Check that the function adds new features
        self.assertGreater(len(engineered_data.columns), len(self.sample_data.columns))
        
        # Check for specific engineered features
        expected_features = [
            'claim_to_property_ratio',
            'customer_age_group',
            'policy_age_group',
            'claim_frequency_group',
            'deductible_to_claim_ratio',
            'vehicle_age_group',
            'credit_score_group'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, engineered_data.columns)
    
    def test_split_data(self):
        """Test the split_data function."""
        # Test with sample data
        X_train, X_test, y_train, y_test = split_data(self.sample_data, target_col='claim_amount')
        
        # Check that the function returns DataFrames
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)
        
        # Check that the target column is not in X_train and X_test
        self.assertNotIn('claim_amount', X_train.columns)
        self.assertNotIn('claim_amount', X_test.columns)
        
        # Check that the target column is in y_train and y_test
        self.assertEqual(y_train.name, 'claim_amount')
        self.assertEqual(y_test.name, 'claim_amount')
        
        # Check that the split is roughly 80/20
        self.assertAlmostEqual(len(X_train) / len(self.sample_data), 0.8, places=1)
        self.assertAlmostEqual(len(X_test) / len(self.sample_data), 0.2, places=1)
    
    def test_end_to_end_preprocessing(self):
        """Test the entire preprocessing pipeline."""
        # Run the entire preprocessing pipeline
        try:
            # Load data
            data = load_data(self.sample_data)
            
            # Clean data
            cleaned_data = clean_data(data)
            
            # Handle missing values
            data_without_missing = handle_missing_values(cleaned_data)
            
            # Detect outliers
            outliers = detect_outliers(data_without_missing)
            
            # Feature engineering
            engineered_data = feature_engineering(data_without_missing)
            
            # Normalize features
            normalized_data = normalize_features(engineered_data)
            
            # Encode categorical variables
            encoded_data = encode_categorical_variables(normalized_data)
            
            # Split data
            X_train, X_test, y_train, y_test = split_data(encoded_data, target_col='claim_amount')
            
            # Check that the final datasets have the expected shape
            self.assertGreater(len(X_train.columns), 10)  # Should have more columns after encoding
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            
        except Exception as e:
            self.fail(f"End-to-end preprocessing raised an exception: {e}")

    def test_preprocess_homeowner_data(self):
        """Test the main preprocessing function."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            # Save test data
            self.sample_data.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
            
        try:
            # Call preprocessing function
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "processed.parquet")
                result = preprocess_homeowner_data(temp_file_path, output_path)
                
                # Verify output file was created
                self.assertTrue(os.path.exists(output_path))
                
                # Verify result contains expected keys
                self.assertIn('output_path', result)
                self.assertIn('num_rows', result)
                self.assertIn('num_features', result)
                
                # Load and verify processed data
                processed_df = pd.read_parquet(output_path)
                self.assertGreater(len(processed_df), 0)
                self.assertGreaterEqual(len(processed_df.columns), len(self.sample_data.columns))
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    def test_analyze_loss_history(self):
        """Test loss history analysis."""
        result = analyze_loss_history(self.sample_data)
        
        # Verify result is a dataframe
        self.assertIsInstance(result, pd.DataFrame)
        
        # Verify new features were created
        self.assertIn('total_loss_count', result.columns)
        self.assertIn('has_recent_loss', result.columns)
        
    def test_encode_categorical_features(self):
        """Test categorical feature encoding."""
        result = encode_categorical_features(self.sample_data)
        
        # Check that categorical column was encoded
        self.assertNotIn('coverage_type', result.columns)
        self.assertIn('coverage_type_basic', result.columns)
        self.assertIn('coverage_type_premium', result.columns)
        self.assertIn('region_north', result.columns)
        self.assertIn('region_south', result.columns)
        self.assertIn('region_east', result.columns)
        self.assertIn('region_west', result.columns)

if __name__ == '__main__':
    unittest.main() 