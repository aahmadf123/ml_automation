import json
import os
import boto3
import logging
import time
from typing import Dict, Any
from botocore.exceptions import ClientError

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
logs = boto3.client('logs')

# Constants
LOG_GROUP = '/ml-automation/websocket-connections'
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def ensure_log_stream(log_group: str, log_stream: str) -> None:
    """
    Ensure a log stream exists, creating it if necessary.
    
    Args:
        log_group: Name of the log group
        log_stream: Name of the log stream
    """
    try:
        logs.create_log_stream(
            logGroupName=log_group,
            logStreamName=log_stream
        )
        logger.info(f"Created log stream {log_stream} in log group {log_group}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
            logger.debug(f"Log stream {log_stream} already exists in log group {log_group}")
        else:
            logger.error(f"Error creating log stream {log_stream}: {str(e)}")

def log_connection(connection_id: str) -> None:
    """
    Log connection event to CloudWatch Logs with error handling.
    
    Args:
        connection_id: ID of the connected client
    """
    try:
        # Ensure log group exists
        try:
            logs.create_log_group(logGroupName=LOG_GROUP)
            logger.info(f"Created log group {LOG_GROUP}")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                logger.error(f"Error creating log group {LOG_GROUP}: {str(e)}")
                return
        
        # Create daily log stream
        log_stream = time.strftime('%Y-%m-%d')
        ensure_log_stream(LOG_GROUP, log_stream)
        
        # Log the connection event
        logs.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=log_stream,
            logEvents=[
                {
                    'timestamp': int(time.time() * 1000),
                    'message': f"Connection established: {connection_id}"
                }
            ]
        )
        
        logger.info(f"Logged connection {connection_id} to {LOG_GROUP}/{log_stream}")
    except Exception as e:
        logger.error(f"Error logging connection {connection_id}: {str(e)}")

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle WebSocket connection events.
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        # Get connection ID from event
        if 'requestContext' not in event or 'connectionId' not in event['requestContext']:
            raise ValueError("Missing connection ID in event")
        
        connection_id = event['requestContext']['connectionId']
        
        # Log the connection
        log_connection(connection_id)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Connection established successfully'
            })
        }
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': str(e)
            })
        }
    except Exception as e:
        logger.error(f"Error handling connection: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error'
            })
        } 