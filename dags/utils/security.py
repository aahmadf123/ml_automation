"""
Security utilities for ML Automation.

This module provides security-related functionality:
- AWS credential management
- Encryption/decryption
- Access control
- Secure configuration handling
"""

import logging
import os
from typing import Dict, Any, Optional, Union
from base64 import b64encode, b64decode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import re
from functools import wraps
import jwt
from datetime import datetime, timedelta
from utils.config import Config

# Setup logging
log = logging.getLogger(__name__)

class SecurityUtils:
    """Security utilities for Airflow DAGs."""
    
    @staticmethod
    def validate_model_id(model_id: str) -> bool:
        """Validate model ID format."""
        return model_id in Config.MODEL_IDS
    
    @staticmethod
    def validate_metric_name(metric: str) -> bool:
        """Validate metric name."""
        return metric in Config.MODEL_METRICS
    
    @staticmethod
    def validate_timestamp(timestamp: str) -> bool:
        """Validate ISO timestamp format."""
        try:
            datetime.fromisoformat(timestamp)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
        """Validate numeric value is within range."""
        return min_val <= value <= max_val

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, window: int = 3600, max_requests: int = 1000):
        self.window = window
        self.max_requests = max_requests
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if a client is allowed to make a request."""
        now = datetime.now()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if (now - req_time).total_seconds() < self.window
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add new request
        self.requests[client_id].append(now)
        return True

def validate_input(schema: Dict[str, Any]) -> Callable:
    """Decorator for input validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                data = kwargs.get('data', {})
                for field, rules in schema.items():
                    if field not in data:
                        if rules.get('required', False):
                            raise ValueError(f"Missing required field: {field}")
                        continue
                    
                    value = data[field]
                    if 'type' in rules and not isinstance(value, rules['type']):
                        raise ValueError(f"Invalid type for field {field}")
                    
                    if 'pattern' in rules and not re.match(rules['pattern'], str(value)):
                        raise ValueError(f"Invalid format for field {field}")
                    
                    if 'min' in rules and value < rules['min']:
                        raise ValueError(f"Value too small for field {field}")
                    
                    if 'max' in rules and value > rules['max']:
                        raise ValueError(f"Value too large for field {field}")
                
                kwargs['validated_data'] = data
            except Exception as e:
                raise ValueError(str(e))
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class SecurityManager:
    """
    Manages security operations for the ML Automation system.
    
    This class handles:
    - Encryption/decryption of sensitive data
    - AWS credential management
    - Access control checks
    - Secure configuration handling
    """
    
    def __init__(self) -> None:
        """
        Initialize the SecurityManager.
        
        Sets up encryption keys and security configurations.
        """
        self._fernet = None
        self._initialize_encryption()
        
    def _initialize_encryption(self) -> None:
        """
        Initialize encryption components.
        
        Raises:
            RuntimeError: If encryption setup fails
        """
        try:
            # Get encryption key from environment
            key = os.getenv('ENCRYPTION_KEY')
            if not key:
                raise ValueError("ENCRYPTION_KEY environment variable not set")
                
            # Generate Fernet key from the encryption key
            salt = b'salt_'  # Should be stored securely
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = b64encode(kdf.derive(key.encode()))
            self._fernet = Fernet(key)
            
        except Exception as e:
            log.error(f"Failed to initialize encryption: {str(e)}")
            raise RuntimeError("Encryption initialization failed") from e
            
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            str: Encrypted data as base64 string
            
        Raises:
            RuntimeError: If encryption fails
        """
        try:
            if isinstance(data, str):
                data = data.encode()
            encrypted = self._fernet.encrypt(data)
            return b64encode(encrypted).decode()
        except Exception as e:
            log.error(f"Encryption failed: {str(e)}")
            raise RuntimeError("Encryption failed") from e
            
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            str: Decrypted data
            
        Raises:
            RuntimeError: If decryption fails
        """
        try:
            encrypted = b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            log.error(f"Decryption failed: {str(e)}")
            raise RuntimeError("Decryption failed") from e
            
    def get_aws_credentials(self) -> Dict[str, str]:
        """
        Get AWS credentials securely.
        
        Returns:
            Dict[str, str]: AWS credentials
            
        Raises:
            RuntimeError: If credential retrieval fails
        """
        try:
            return {
                'aws_access_key_id': self.decrypt(os.getenv('AWS_ACCESS_KEY_ID', '')),
                'aws_secret_access_key': self.decrypt(os.getenv('AWS_SECRET_ACCESS_KEY', '')),
                'region_name': os.getenv('AWS_REGION', 'us-east-1')
            }
        except Exception as e:
            log.error(f"Failed to get AWS credentials: {str(e)}")
            raise RuntimeError("AWS credential retrieval failed") from e
            
    def check_access(self, user: str, resource: str) -> bool:
        """
        Check if a user has access to a resource.
        
        Args:
            user: User identifier
            resource: Resource identifier
            
        Returns:
            bool: True if access is granted, False otherwise
        """
        try:
            # Implementation would check against an access control list
            return True
        except Exception as e:
            log.error(f"Access check failed: {str(e)}")
            return False
            
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate security-sensitive configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_keys = {'encryption_key', 'aws_region', 'access_control'}
            return all(key in config for key in required_keys)
        except Exception as e:
            log.error(f"Configuration validation failed: {str(e)}")
            return False 