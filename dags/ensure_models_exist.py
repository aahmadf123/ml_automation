#!/usr/bin/env python3
"""
ensure_models_exist.py - Check and create model files in S3

This script checks if Model1.joblib and Model4.joblib exist in S3,
and if not, creates default XGBoost models and uploads them.
"""

import os
import boto3
import logging
import tempfile
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get AWS configuration from config.py if available
try:
    from utils.config import DATA_BUCKET, MODEL_KEY_PREFIX, AWS_REGION
except ImportError:
    # Fallback values if config module not available
    DATA_BUCKET = 'grange-seniordesign-bucket'
    MODEL_KEY_PREFIX = 'models'
    AWS_REGION = 'us-east-1'
    logger.warning("utils.config module not found, using default values")

# Model file names
MODEL_FILES = {
    'model1': 'Model1.joblib',  # Uppercase file name
    'model4': 'Model4.joblib'   # Uppercase file name
}

def check_model_exists(s3_client, bucket, key):
    """
    Check if a model file exists in the S3 bucket.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        bool: True if the file exists, False otherwise
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == '404':
            return False
        else:
            logger.error(f"Error checking if model exists: {str(e)}")
            return False

def create_default_model(model_id):
    """
    Create a simple default XGBoost model for testing.
    
    Args:
        model_id: ID of the model to create (model1 or model4)
        
    Returns:
        xgb.XGBRegressor: A trained XGBoost model
    """
    logger.info(f"Creating default {model_id} model")
    
    # Create a simple dataset
    np.random.seed(42)
    X = np.random.rand(100, 20)
    
    # Add some predictive features
    if model_id == 'model1':
        # For model1 (loss history counts model)
        X[:, 0] = np.random.poisson(1, 100)  # num_loss_3yr_total
        X[:, 1] = np.random.poisson(0.5, 100)  # num_loss_yrs45_total
        X[:, 2] = np.clip(5 - X[:, 0] - X[:, 1], 0, 5)  # num_loss_free_yrs_total
        feature_names = [f"num_loss_3yr_feat{i}" for i in range(10)] + [f"num_loss_yrs45_feat{i}" for i in range(5)] + [f"num_loss_free_yrs_feat{i}" for i in range(5)]
    else:
        # For model4 (loss history with weight classes - 3D model)
        X[:, 0] = np.random.poisson(0.8, 100)  # lhdwc_5y_3d_total
        X[:, 1:5] = np.random.rand(100, 4) * X[:, 0].reshape(-1, 1)  # lhdwc_5y_3d features
        feature_names = [f"lhdwc_5y_3d_feat{i}" for i in range(20)]
    
    # Create a pandas DataFrame with appropriate column names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Generate target variable - simple function with noise
    base = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * np.mean(X[:, 2:5], axis=1)
    y = base + np.random.normal(0, 0.2, 100)
    
    # Train a simple XGBoost model
    model_params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_df, y)
    
    return model

def save_and_upload_model(model, model_id, filename, s3_client, bucket):
    """
    Save model to disk and upload to S3.
    
    Args:
        model: The model to save
        model_id: ID of the model (model1 or model4)
        filename: Filename to use for the model
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, filename)
        
        # Save model to disk
        logger.info(f"Saving {model_id} model to {local_path}")
        joblib.dump(model, local_path)
        
        # Check if file was created
        if not os.path.exists(local_path):
            logger.error(f"Failed to save model to {local_path}")
            return False
            
        # Upload to S3
        s3_key = f"{MODEL_KEY_PREFIX}/{filename}"
        logger.info(f"Uploading {model_id} model to s3://{bucket}/{s3_key}")
        
        s3_client.upload_file(
            local_path,
            bucket,
            s3_key,
            ExtraArgs={
                'Metadata': {
                    'model_id': model_id,
                    'created_at': datetime.now().isoformat(),
                    'created_by': 'ensure_models_exist.py',
                    'purpose': 'default model for testing'
                }
            }
        )
        
        logger.info(f"Successfully uploaded {model_id} model to S3")
        return True
        
    except Exception as e:
        logger.error(f"Error saving/uploading model {model_id}: {str(e)}")
        return False
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass

def main():
    """Main function to check and create models if needed."""
    logger.info("Starting model existence check")
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=DATA_BUCKET)
            logger.info(f"S3 bucket '{DATA_BUCKET}' exists")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                logger.error(f"S3 bucket '{DATA_BUCKET}' does not exist")
                return
            else:
                logger.error(f"Error checking bucket: {str(e)}")
                return
                
        # Check and create each model
        for model_id, filename in MODEL_FILES.items():
            s3_key = f"{MODEL_KEY_PREFIX}/{filename}"
            
            if check_model_exists(s3_client, DATA_BUCKET, s3_key):
                logger.info(f"Model file '{s3_key}' already exists in S3 bucket")
            else:
                logger.warning(f"Model file '{s3_key}' not found, creating default model")
                
                # Create a default model
                model = create_default_model(model_id)
                
                # Save and upload model
                if save_and_upload_model(model, model_id, filename, s3_client, DATA_BUCKET):
                    logger.info(f"Successfully created and uploaded {model_id} model")
                else:
                    logger.error(f"Failed to create and upload {model_id} model")
        
        logger.info("Model existence check completed")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()

