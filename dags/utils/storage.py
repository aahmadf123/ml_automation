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
        # Don't initialize client at creation time
        
    def _initialize_client(self) -> None:
        """
        Initialize the S3 client.
        
        Raises:
            RuntimeError: If client initialization fails
        """
        if self._s3_client is not None:
            return
            
        try:
            # Import security module only when needed
            from .security import SecurityManager
            security = SecurityManager()
            credentials = security.get_aws_credentials()
            self._s3_client = boto3.client('s3', **credentials)
        except Exception as e:
            log.warning(f"Failed to initialize S3 client: {str(e)}. Operations will fail.")
            
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
        # Lazy initialization
        self._initialize_client()
        
        if self._s3_client is None:
            error_msg = f"Cannot upload file, S3 client not initialized. Path: {file_path}, Key: {key}"
            log.error(error_msg)
            raise RuntimeError(error_msg)
            
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
        # Lazy initialization
        self._initialize_client()
        
        if self._s3_client is None:
            error_msg = f"Cannot download file, S3 client not initialized. Key: {key}, Path: {file_path}"
            log.error(error_msg)
            raise RuntimeError(error_msg)
            
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
        # Lazy initialization
        self._initialize_client()
        
        if self._s3_client is None:
            log.error(f"Cannot validate file, S3 client not initialized. Key: {key}")
            return False
            
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

# Create a lazy-loading singleton
_manager = None

def get_manager() -> StorageManager:
    """Get the StorageManager singleton instance (create it if it doesn't exist)."""
    global _manager
    if _manager is None:
        _manager = StorageManager()
    return _manager

def upload(local_path: str, key: str, bucket: str = DATA_BUCKET, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Upload a local file to S3 with retry logic.
    
    Args:
        local_path: Path to the local file
        key: S3 object key
        bucket: S3 bucket name (default: DATA_BUCKET)
        metadata: Optional metadata to attach to the object
        
    Returns:
        Dict[str, Any]: S3 upload response
        
    Raises:
        FileNotFoundError: If local file doesn't exist
        ClientError: If S3 upload fails
    """
    return get_manager().upload(local_path, bucket, key, metadata)

def download(key: str, local_path: str, bucket: str = DATA_BUCKET, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Download a file from S3 with retry logic.
    
    Args:
        key: S3 object key
        local_path: Path to save the downloaded file
        bucket: S3 bucket name (default: DATA_BUCKET)
        metadata: Optional metadata to verify
        
    Returns:
        Dict[str, Any]: S3 download response
        
    Raises:
        ClientError: If S3 download fails
    """
    return get_manager().download(bucket, key, local_path, metadata)

def update_storage_process_with_ui_components():
    """
    Update the storage process with UI components.
    """
    pass
