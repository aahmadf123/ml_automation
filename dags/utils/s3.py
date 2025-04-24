#!/usr/bin/env python3
"""
s3.py - Utilities for interacting with AWS S3
--------------------------------------------
This module provides functions for uploading to and downloading from S3.
"""

import os
import logging
import boto3
from typing import Optional
from botocore.exceptions import ClientError

# Set up logging
logger = logging.getLogger(__name__)

def upload_to_s3(
    local_path: str, 
    bucket: str, 
    s3_key: str, 
    region_name: Optional[str] = None
) -> bool:
    """
    Upload a file to an S3 bucket
    
    Args:
        local_path: Path to local file to upload
        bucket: S3 bucket name
        s3_key: S3 key (path) to upload to
        region_name: AWS region name (optional)
        
    Returns:
        Boolean indicating success or failure
    """
    if not os.path.exists(local_path):
        logger.error(f"Local file not found: {local_path}")
        return False
    
    try:
        # Create S3 client
        s3_client = boto3.client('s3', region_name=region_name)
        
        # Upload file
        s3_client.upload_file(local_path, bucket, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error uploading to S3: {e}")
        return False


def download_from_s3(
    bucket: str, 
    s3_key: str, 
    local_path: str, 
    region_name: Optional[str] = None
) -> bool:
    """
    Download a file from an S3 bucket
    
    Args:
        bucket: S3 bucket name
        s3_key: S3 key (path) to download from
        local_path: Local path to save the file to
        region_name: AWS region name (optional)
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Create S3 client
        s3_client = boto3.client('s3', region_name=region_name)
        
        # Download file
        s3_client.download_file(bucket, s3_key, local_path)
        logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.error(f"File not found in S3: s3://{bucket}/{s3_key}")
        else:
            logger.error(f"Error downloading file from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading from S3: {e}")
        return False 