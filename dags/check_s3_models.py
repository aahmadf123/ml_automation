#!/usr/bin/env python3
"""
check_s3_models.py - Check S3 bucket for model files

Script to list contents of the S3 bucket models directory and check for 
model1.joblib and model4.joblib. Includes proper error handling and AWS 
region configuration.
"""

import boto3
import logging
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# S3 configuration
BUCKET_NAME = 'grange-seniordesign-bucket'
PREFIX = 'models/'
REGION_NAME = 'us-east-1'  # Change to your actual region
MODEL_FILES = ['model1.joblib', 'model4.joblib']


def check_s3_models():
    """
    Check S3 bucket for model files.
    
    Returns:
        dict: Dictionary with model file names as keys and existence status as values
    """
    try:
        # Create S3 client with region
        s3_client = boto3.client('s3', region_name=REGION_NAME)
        
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=BUCKET_NAME)
            logger.info(f"S3 bucket '{BUCKET_NAME}' exists")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                logger.error(f"S3 bucket '{BUCKET_NAME}' does not exist")
                return {model: False for model in MODEL_FILES}
            else:
                logger.error(f"Error checking bucket: {str(e)}")
                return {model: False for model in MODEL_FILES}
        
        # List objects in the models directory
        try:
            response = s3_client.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=PREFIX
            )
            
            logger.info(f"Listing objects in '{BUCKET_NAME}/{PREFIX}'")
            
            # Extract file names from response
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
                logger.info(f"Found {len(files)} files in prefix '{PREFIX}'")
                
                for file in files:
                    logger.info(f"  - {file}")
                
                # Check if specific model files exist
                model_status = {}
                for model in MODEL_FILES:
                    model_key = f"{PREFIX}{model}"
                    model_exists = model_key in files
                    model_status[model] = model_exists
                    if model_exists:
                        logger.info(f"Model file '{model}' found")
                    else:
                        logger.warning(f"Model file '{model}' NOT found")
                
                return model_status
            else:
                logger.warning(f"No files found in '{BUCKET_NAME}/{PREFIX}'")
                return {model: False for model in MODEL_FILES}
                
        except ClientError as e:
            logger.error(f"Error listing objects: {str(e)}")
            return {model: False for model in MODEL_FILES}
            
    except NoCredentialsError:
        logger.error("No AWS credentials found")
        return {model: False for model in MODEL_FILES}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {model: False for model in MODEL_FILES}


def main():
    """Main function to run the script."""
    logger.info("Checking S3 bucket for model files...")
    model_status = check_s3_models()
    
    # Summarize results
    logger.info("=== Model Files Status ===")
    all_found = True
    for model, exists in model_status.items():
        status = "EXISTS" if exists else "MISSING"
        logger.info(f"{model}: {status}")
        if not exists:
            all_found = False
    
    if all_found:
        logger.info("All required model files found in S3 bucket.")
    else:
        logger.warning("Some required model files are missing. They need to be created and uploaded.")
    
    return model_status


if __name__ == "__main__":
    main()

