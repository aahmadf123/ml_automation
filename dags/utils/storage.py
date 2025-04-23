#!/usr/bin/env python3
"""
utils/storage.py

Retrying wrappers around common S3 operations.
"""

import logging
import os
from typing import Dict, Any, Optional, Union
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from utils.config import DATA_BUCKET

# Setup logging
log = logging.getLogger(__name__)

class StorageManager:
    """
    Manages storage operations for the ML Automation system.
    
    This class handles:
    - File uploads and downloads
    - File validation
    - Progress tracking
    - Error handling
    """
    
    def __init__(self) -> None:
        """
        Initialize the StorageManager.
        
        Sets up the S3 client and configuration.
        """
        self._s3_client = None
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """
        Initialize the S3 client.
        
        Raises:
            RuntimeError: If client initialization fails
        """
        try:
            from .security import SecurityManager
            security = SecurityManager()
            credentials = security.get_aws_credentials()
            self._s3_client = boto3.client('s3', **credentials)
        except Exception as e:
            log.error(f"Failed to initialize S3 client: {str(e)}")
            raise RuntimeError("S3 client initialization failed") from e
            
    def upload(
        self,
        file_path: str,
        bucket: str,
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to S3.
        
        Args:
            file_path: Path to the local file
            bucket: S3 bucket name
            key: S3 object key
            metadata: Optional metadata to attach to the object
            
        Returns:
            Dict[str, Any]: S3 upload response
            
        Raises:
            FileNotFoundError: If local file doesn't exist
            ClientError: If S3 upload fails
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            file_size = os.path.getsize(file_path)
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {key}") as pbar:
                def upload_progress(chunk):
                    pbar.update(chunk)
                    
                self._s3_client.upload_file(
                    file_path,
                    bucket,
                    key,
                    ExtraArgs={'Metadata': metadata or {}},
                    Callback=upload_progress
                )
                
            log.info(f"Successfully uploaded {key} to {bucket}")
            return {'status': 'success', 'bucket': bucket, 'key': key}
            
        except ClientError as e:
            log.error(f"S3 upload failed: {str(e)}")
            raise
        except Exception as e:
            log.error(f"Upload failed: {str(e)}")
            raise RuntimeError("Upload failed") from e
            
    def download(
        self,
        bucket: str,
        key: str,
        file_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Download a file from S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            file_path: Path to save the downloaded file
            metadata: Optional metadata to verify
            
        Returns:
            Dict[str, Any]: S3 download response
            
        Raises:
            ClientError: If S3 download fails
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Get object size for progress bar
            response = self._s3_client.head_object(Bucket=bucket, Key=key)
            file_size = response['ContentLength']
            
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {key}") as pbar:
                def download_progress(chunk):
                    pbar.update(chunk)
                    
                self._s3_client.download_file(
                    bucket,
                    key,
                    file_path,
                    ExtraArgs={'Metadata': metadata or {}},
                    Callback=download_progress
                )
                
            log.info(f"Successfully downloaded {key} from {bucket}")
            return {'status': 'success', 'bucket': bucket, 'key': key}
            
        except ClientError as e:
            log.error(f"S3 download failed: {str(e)}")
            raise
        except Exception as e:
            log.error(f"Download failed: {str(e)}")
            raise RuntimeError("Download failed") from e
            
    def validate_file(
        self,
        bucket: str,
        key: str,
        expected_size: Optional[int] = None,
        expected_metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Validate a file in S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            expected_size: Expected file size in bytes
            expected_metadata: Expected metadata
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            response = self._s3_client.head_object(Bucket=bucket, Key=key)
            
            if expected_size and response['ContentLength'] != expected_size:
                log.error(f"File size mismatch: expected {expected_size}, got {response['ContentLength']}")
                return False
                
            if expected_metadata:
                actual_metadata = response.get('Metadata', {})
                if actual_metadata != expected_metadata:
                    log.error(f"Metadata mismatch: expected {expected_metadata}, got {actual_metadata}")
                    return False
                    
            return True
            
        except Exception as e:
            log.error(f"File validation failed: {str(e)}")
            return False

def update_storage_process_with_ui_components():
    """
    Update the storage process with UI components.
    """
    pass

# Create standalone functions for backward compatibility
def upload(file_path: str, bucket: str, key: str, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Upload a file to S3.
    
    Args:
        file_path: Path to the file to upload
        bucket: S3 bucket name
        key: S3 key
        metadata: Optional metadata to attach to the file
        
    Returns:
        Dict containing upload result
    """
    manager = StorageManager()
    return manager.upload(file_path, bucket, key, metadata)

def download(bucket: str, key: str, file_path: str, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Download a file from S3.
    
    Args:
        bucket: S3 bucket name
        key: S3 key
        file_path: Path to save the downloaded file
        metadata: Optional metadata to attach to the file
        
    Returns:
        Dict containing download result
    """
    manager = StorageManager()
    return manager.download(bucket, key, file_path, metadata)
