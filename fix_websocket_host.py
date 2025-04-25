#!/usr/bin/env python3
"""
Fix WebSocket Host Configuration in AWS Secrets Manager

This script updates the WS_HOST value in AWS Secrets Manager to remove the invalid 'ss://' prefix
and ensures WebSocket connections use the correct protocol prefix.
"""

import boto3
import json
import logging
import sys
import time
import re
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS Secret Name
SECRET_NAME_DASHBOARD = "dashboard-secrets"
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

def get_secret(secret_name, retry_count=0):
    """
    Retrieve a secret from AWS Secrets Manager with retry logic.
    
    Args:
        secret_name (str): Name of the secret to retrieve
        retry_count (int): Current retry attempt
        
    Returns:
        dict: Secret values as dictionary or None if failed
    """
    client = boto3.client('secretsmanager')
    try:
        response = client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in response:
            return json.loads(response['SecretString'])
        else:
            logger.error("Secret not found or is binary")
            return None
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'ThrottlingException' and retry_count < MAX_RETRIES:
            wait_time = BACKOFF_FACTOR ** retry_count
            logger.warning(f"Rate limited. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            return get_secret(secret_name, retry_count + 1)
        elif error_code == 'ResourceNotFoundException':
            logger.error(f"Secret {secret_name} not found")
            return None
        else:
            logger.error(f"Error retrieving secret: {e}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error retrieving secret: {e}")
        return None

def update_secret(secret_name, secret_value, retry_count=0):
    """
    Update an existing secret in AWS Secrets Manager with retry logic.
    
    Args:
        secret_name (str): Name of the secret to update
        secret_value (dict): Secret values to store
        retry_count (int): Current retry attempt
        
    Returns:
        bool: True if update successful, False otherwise
    """
    client = boto3.client('secretsmanager')
    try:
        client.update_secret(
            SecretId=secret_name,
            SecretString=json.dumps(secret_value)
        )
        logger.info(f"Secret {secret_name} updated successfully")
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'ThrottlingException' and retry_count < MAX_RETRIES:
            wait_time = BACKOFF_FACTOR ** retry_count
            logger.warning(f"Rate limited. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            return update_secret(secret_name, secret_value, retry_count + 1)
        else:
            logger.error(f"Error updating secret: {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error updating secret: {e}")
        return False

def validate_ws_host(ws_host):
    """
    Validate and correct the WebSocket host string.
    
    Args:
        ws_host (str): Current WebSocket host value
        
    Returns:
        tuple: (is_valid, corrected_value)
    """
    # Remove any incorrect protocol prefixes
    if ws_host.startswith('ss://'):
        ws_host = ws_host[5:]
        
    # Check for other invalid protocol prefixes
    if ws_host.startswith('http://') or ws_host.startswith('https://'):
        ws_host = ws_host.split('://', 1)[1]
        
    # Ensure it's a valid hostname/URL structure
    hostname_pattern = r'^([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])(\.([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]{0,61}[a-zA-Z0-9]))*$'
    is_valid = bool(re.match(hostname_pattern, ws_host)) or 'amazonaws.com' in ws_host or 'localhost' in ws_host
    
    return is_valid, ws_host

def fix_websocket_host():
    """
    Fix the WS_HOST value by correcting any invalid protocol prefixes and validating the hostname.
    
    Returns:
        bool: True if fix successful or no fix needed, False otherwise
    """
    # Get current secrets
    secrets = get_secret(SECRET_NAME_DASHBOARD)
    if not secrets:
        logger.error("Could not retrieve dashboard secrets")
        return False
    
    # Check WS_HOST value
    current_ws_host = secrets.get('WS_HOST', '')
    if not current_ws_host:
        logger.warning("WS_HOST is not set in the secrets. No action taken.")
        return True
        
    logger.info(f"Current WS_HOST value: {current_ws_host}")
    
    # Validate and correct the WebSocket host
    is_valid, new_ws_host = validate_ws_host(current_ws_host)
    
    if new_ws_host != current_ws_host:
        # Update the secret with corrected value
        secrets['WS_HOST'] = new_ws_host
        logger.info(f"Updating WS_HOST to: {new_ws_host}")
        
        # Verify the new value is valid
        if not is_valid:
            logger.warning(f"Corrected WS_HOST value '{new_ws_host}' may not be a valid hostname.")
            
        # Update the secret
        return update_secret(SECRET_NAME_DASHBOARD, secrets)
    else:
        logger.info("WS_HOST is already in the correct format. No update needed.")
        return True

def verify_update():
    """
    Verify that the WebSocket host was updated correctly.
    
    Returns:
        bool: True if verification passed, False otherwise
    """
    # Get the updated secrets
    secrets = get_secret(SECRET_NAME_DASHBOARD)
    if not secrets:
        logger.error("Could not retrieve dashboard secrets for verification")
        return False
    
    current_ws_host = secrets.get('WS_HOST', '')
    is_valid, _ = validate_ws_host(current_ws_host)
    
    if is_valid and not current_ws_host.startswith('ss://'):
        logger.info(f"Verification successful. WS_HOST value is correct: {current_ws_host}")
        return True
    else:
        logger.error(f"Verification failed. WS_HOST value is incorrect: {current_ws_host}")
        return False

if __name__ == "__main__":
    logger.info("Starting WebSocket Host configuration fix...")
    
    if fix_websocket_host():
        # Verify the update was successful
        if verify_update():
            logger.info("WebSocket Host configuration fixed and verified successfully")
            sys.exit(0)
        else:
            logger.error("WebSocket Host configuration fixed but verification failed")
            sys.exit(1)
    else:
        logger.error("Failed to fix WebSocket Host configuration")
        sys.exit(1) 