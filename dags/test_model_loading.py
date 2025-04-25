#!/usr/bin/env python3
"""
test_model_loading.py - Test script to verify model loading

This script tests the loading of Model1.joblib and Model4.joblib from S3
with the updated configuration. This version bypasses ClearML dependencies.
"""

import os
import logging
import sys
import tempfile
import mlflow
import importlib.util
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"model_loading_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Set MLflow URI explicitly
os.environ['MLFLOW_TRACKING_URI'] = 'http://3.146.46.179:5000'
logger.info("Set MLflow tracking URI to http://3.146.46.179:5000")

try:
    # Create a direct boto3 client for S3 operations
    import boto3
    s3_client = boto3.client('s3')
    logger.info("Created S3 client for direct download")
    
    # Define a simple direct download function instead of importing complex module
    def download_from_s3(bucket, key, local_path):
        """Simple function to download a file from S3 without complex dependencies"""
        logger.info(f"Downloading from S3: s3://{bucket}/{key} to {local_path}")
        
        # Create local directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        
        # Download the file
        s3_client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=local_path
        )
        
        # Verify the file exists and has content
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            logger.info(f"Successfully downloaded {os.path.getsize(local_path)} bytes to {local_path}")
            return True
        else:
            raise FileNotFoundError(f"Downloaded file not found or empty: {local_path}")
    
    # Define a simplified version of load_pretrained_model to avoid complex imports
    def load_pretrained_model(model_id: str, model_dir: str = None):
        """
        Simplified version of load_pretrained_model that only loads models from S3
        
        Args:
            model_id: ID of the model to load (model1 or model4)
            model_dir: Local directory to save the model to
            
        Returns:
            Dict with model and status
        """
        import joblib
        
        logger.info(f"Loading pretrained model {model_id} from S3")
        
        # Only allow model1 and model4
        if model_id not in ['model1', 'model4']:
            error_msg = f"Only model1 and model4 are supported, got {model_id}"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg, "model_id": model_id}
        
        # Create a temporary directory if model_dir is not provided
        if model_dir is None:
            model_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory for model: {model_dir}")
        
        # Ensure the directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Use capitalized filenames
        s3_filename = f"Model{model_id[5:]}.joblib"
        model_key = f"models/{s3_filename}"
        local_model_path = os.path.join(model_dir, f"{model_id}.joblib")
        
        # Download the model
        try:
            logger.info(f"Downloading model from S3: s3://grange-seniordesign-bucket/{model_key}")
            
            # Use the simpler direct download function
            download_from_s3(
                bucket="grange-seniordesign-bucket",
                key=model_key, 
                local_path=local_model_path
            )
            
            # Load the model
            if os.path.exists(local_model_path) and os.path.getsize(local_model_path) > 0:
                logger.info(f"Loading model from {local_model_path}")
                model = joblib.load(local_model_path)
                
                return {
                    "status": "completed",
                    "model": model,
                    "model_id": model_id,
                    "s3_filename": s3_filename,
                    "local_path": local_model_path
                }
            else:
                error_msg = f"Downloaded model file not found or empty: {local_model_path}"
                logger.error(error_msg)
                return {"status": "failed", "error": error_msg, "model_id": model_id}
                
        except Exception as e:
            error_msg = f"Error loading pretrained model {model_id}: {str(e)}"
            logger.error(error_msg)
            return {"status": "failed", "error": error_msg, "model_id": model_id}
    
    logger.info("Successfully defined load_pretrained_model function")
    
except Exception as e:
    logger.critical(f"Unexpected error during module setup: {str(e)}")
    sys.exit(1)

def test_model_loading():
    """
    Test loading Model1.joblib and Model4.joblib from S3.
    Simplified version without MLflow logging, just basic S3 download.
    """
    logger.info("===== Starting Simplified Model Loading Test =====")
    
    # Test loading for each model
    models_to_test = ['model1', 'model4']
    results = {}
    
    for model_id in models_to_test:
        logger.info(f"----- Testing {model_id} loading -----")
        
        try:
            # Create a temporary directory for each model
            temp_dir = tempfile.mkdtemp(prefix=f"test_{model_id}_")
            logger.info(f"Created temporary directory {temp_dir} for {model_id}")
            
            # Load the model
            start_time = datetime.now()
            logger.info(f"Starting load of {model_id} at {start_time.strftime('%H:%M:%S')}")
            
            result = load_pretrained_model(model_id, model_dir=temp_dir)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Check the result
            if result.get('status') == 'completed':
                logger.info(f"Successfully loaded {model_id} in {duration:.2f} seconds")
                
                # Verify model object
                model = result.get('model')
                if model is not None and hasattr(model, 'predict'):
                    logger.info(f"Model {model_id} is valid and has predict method")
                    
                    # Log model info if available
                    if hasattr(model, 'n_estimators'):
                        logger.info(f"Model {model_id} has {model.n_estimators} estimators")
                    if hasattr(model, 'max_depth'):
                        logger.info(f"Model {model_id} max_depth is {model.max_depth}")
                        
                    # Try to log a basic prediction (optional)
                    try:
                        import numpy as np
                        # Create a small random dataset for testing
                        test_data = np.random.rand(5, 20)
                        predictions = model.predict(test_data)
                        logger.info(f"Model test predictions shape: {predictions.shape}")
                        logger.info(f"Model test predictions first value: {predictions[0]}")
                    except Exception as pred_err:
                        logger.warning(f"Couldn't test model predictions: {str(pred_err)}")
                        
                    results[model_id] = {
                        "success": True,
                        "duration": duration,
                        "local_path": result.get('local_path')
                    }
                else:
                    logger.error(f"Model {model_id} is not valid (does not have predict method)")
                    results[model_id] = {
                        "success": False,
                        "error": "Invalid model object",
                        "duration": duration
                    }
            else:
                logger.error(f"Failed to load {model_id}: {result.get('error')}")
                results[model_id] = {
                    "success": False,
                    "error": result.get('error'),
                    "duration": duration
                }
                
            # Clean up temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory {temp_dir}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up directory {temp_dir}: {str(cleanup_err)}")
                
        except Exception as e:
            logger.error(f"Error during {model_id} test: {str(e)}")
            logger.exception("Full exception details:")
            results[model_id] = {
                "success": False,
                "error": str(e)
            }
    
    # Print summary
    logger.info("===== Model Loading Test Results =====")
    for model_id, result in results.items():
        if result.get('success'):
            logger.info(f"{model_id}: SUCCESS (Duration: {result.get('duration', 'N/A'):.2f}s)")
        else:
            logger.info(f"{model_id}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Overall result
    all_success = all(result.get('success') for result in results.values())
    if all_success:
        logger.info("✅ ALL MODELS LOADED SUCCESSFULLY")
    else:
        logger.error("❌ SOME MODELS FAILED TO LOAD")
        
    return results

if __name__ == "__main__":
    try:
        test_model_loading()
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        logger.exception("Unhandled exception:")
        sys.exit(1)

