#!/usr/bin/env python3
"""
Fix WebSocket Host Configuration in AWS Secrets Manager

This script updates the WS_HOST value in AWS Secrets Manager to remove the invalid 'ss://' prefix.
"""

import boto3
import json
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS Secret Name
SECRET_NAME_DASHBOARD = "dashboard-secrets"

def get_secret(secret_name):
    """Retrieve a secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    try:
        response = client.get_secret_value(SecretId=secret_name)
        if 'SecretString' in response:
            return json.loads(response['SecretString'])
        else:
            logger.error("Secret not found or is binary")
            return None
    except Exception as e:
        logger.error(f"Error retrieving secret: {e}")
        return None

def update_secret(secret_name, secret_value):
    """Update an existing secret in AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    try:
        client.update_secret(
            SecretId=secret_name,
            SecretString=json.dumps(secret_value)
        )
        logger.info(f"Secret {secret_name} updated successfully")
        return True
    except Exception as e:
        logger.error(f"Error updating secret: {e}")
        return False

def fix_websocket_host():
    """Fix the WS_HOST value by removing the invalid 'ss://' prefix."""
    # Get current secrets
    secrets = get_secret(SECRET_NAME_DASHBOARD)
    if not secrets:
        logger.error("Could not retrieve dashboard secrets")
        return False
    
    # Check WS_HOST value
    current_ws_host = secrets.get('WS_HOST', '')
    logger.info(f"Current WS_HOST value: {current_ws_host}")
    
    if current_ws_host.startswith('ss://'):
        # Remove the 'ss://' prefix
        new_ws_host = current_ws_host[5:]
        secrets['WS_HOST'] = new_ws_host
        logger.info(f"Updating WS_HOST to: {new_ws_host}")
        
        # Update the secret
        return update_secret(SECRET_NAME_DASHBOARD, secrets)
    else:
        logger.info("WS_HOST does not have 'ss://' prefix. No update needed.")
        return True

if __name__ == "__main__":
    logger.info("Starting WebSocket Host configuration fix...")
    
    if fix_websocket_host():
        logger.info("WebSocket Host configuration fixed successfully")
        sys.exit(0)
    else:
        logger.error("Failed to fix WebSocket Host configuration")
        sys.exit(1) 