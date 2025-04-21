import json
import os
import boto3
import logging
import time
from botocore.exceptions import ClientError

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize CloudWatch Logs client
logs = boto3.client('logs')

# Constants
LOG_GROUP = os.getenv('LOG_GROUP', '/ml-automation/websocket-connections')
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def ensure_log_stream():
    """Ensure the log group and stream exist."""
    try:
        # Create log group if it doesn't exist
        try:
            logs.create_log_group(logGroupName=LOG_GROUP)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass

        # Create log stream with timestamp
        stream_name = f"default-handler-{int(time.time())}"
        logs.create_log_stream(
            logGroupName=LOG_GROUP,
            logStreamName=stream_name
        )
        return stream_name
    except ClientError as e:
        logger.error(f"Error ensuring log stream: {str(e)}")
        raise

def log_event(stream_name: str, message: str):
    """Log an event to CloudWatch Logs."""
    try:
        logs.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=stream_name,
            logEvents=[{
                'timestamp': int(time.time() * 1000),
                'message': json.dumps(message)
            }]
        )
    except ClientError as e:
        logger.error(f"Error logging event: {str(e)}")
        raise

def handler(event, context):
    """
    Default WebSocket route handler.
    This handler processes any WebSocket messages that don't match other routes.
    """
    try:
        # Get connection ID from the event
        connection_id = event.get('requestContext', {}).get('connectionId')
        if not connection_id:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'message': 'Missing connection ID'
                })
            }

        # Ensure log stream exists
        stream_name = ensure_log_stream()

        # Log the event
        log_event(stream_name, {
            'type': 'default_route',
            'connection_id': connection_id,
            'event': event,
            'timestamp': int(time.time())
        })

        # Return success response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Default route handler executed successfully'
            })
        }

    except Exception as e:
        logger.error(f"Error in default handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Internal server error'
            })
        } 