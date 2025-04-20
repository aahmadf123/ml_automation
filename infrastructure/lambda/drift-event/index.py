import json
import os
import boto3
import logging
import time
import ssm
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
api_gateway = boto3.client('apigatewaymanagementapi')
cloudwatch = boto3.client('cloudwatch')
logs = boto3.client('logs')
ssm = boto3.client('ssm')

# Constants
LOG_GROUP = '/ml-automation/websocket-connections'
DRIFT_LOG_GROUP = '/ml-automation/drift-events'
CONNECTION_WINDOW = 3600000  # 1 hour in milliseconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def get_ssm_parameter(param_name: str, default_value: Optional[str] = None) -> str:
    """
    Get a parameter from SSM Parameter Store with retries.
    
    Args:
        param_name: Name of the parameter to retrieve
        default_value: Default value to return if parameter not found
        
    Returns:
        Parameter value from SSM or default value
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = ssm.get_parameter(
                Name=f'/ml-automation/{param_name}',
                WithDecryption=True
            )
            return response['Parameter']['Value']
        except ClientError as e:
            if e.response['Error']['Code'] == 'ParameterNotFound':
                logger.warning(f"Parameter {param_name} not found in SSM")
                return default_value
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Retrying SSM parameter retrieval for {param_name}: {str(e)}")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to retrieve SSM parameter {param_name}: {str(e)}")
                return default_value
        except Exception as e:
            logger.error(f"Unexpected error retrieving SSM parameter {param_name}: {str(e)}")
            return default_value

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

def get_active_connections() -> List[str]:
    """
    Get active connection IDs from CloudWatch Logs with pagination and error handling.
    
    Returns:
        List of active connection IDs
    """
    try:
        # Ensure log group exists
        try:
            logs.create_log_group(logGroupName=LOG_GROUP)
            logger.info(f"Created log group {LOG_GROUP}")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                logger.error(f"Error creating log group {LOG_GROUP}: {str(e)}")
                return []
        
        # Get paginator for log events
        paginator = logs.get_paginator('filter_log_events')
        connection_ids = set()
        
        # Query for recent connection events with pagination
        for page in paginator.paginate(
            logGroupName=LOG_GROUP,
            filterPattern='connectionId',
            startTime=int(time.time() * 1000) - CONNECTION_WINDOW
        ):
            for event in page['events']:
                log_message = event['message']
                if 'Connection established' in log_message:
                    # Extract connection ID from log message
                    connection_id = log_message.split('Connection established: ')[1].strip()
                    connection_ids.add(connection_id)
                elif 'Connection closed' in log_message:
                    # Remove closed connections
                    connection_id = log_message.split('Connection closed: ')[1].strip()
                    connection_ids.discard(connection_id)
        
        # Convert to list and log count
        active_connections = list(connection_ids)
        logger.info(f"Found {len(active_connections)} active connections")
        
        return active_connections
    except Exception as e:
        logger.error(f"Error getting active connections: {str(e)}")
        return []

def broadcast_message(message: Dict[str, Any], endpoint_url: str) -> None:
    """
    Broadcast a message to all connected clients with retries and error handling.
    
    Args:
        message: Message to broadcast
        endpoint_url: WebSocket API endpoint URL
    """
    try:
        # Get active connection IDs
        connection_ids = get_active_connections()
        if not connection_ids:
            logger.warning("No active connections found")
            return
        
        # Prepare message data
        message_data = json.dumps(message)
        api_id = endpoint_url.split('/')[-1]
        
        # Send message to each connection with retries
        for connection_id in connection_ids:
            for attempt in range(MAX_RETRIES):
                try:
                    api_gateway.post_to_connection(
                        ConnectionId=connection_id,
                        Data=message_data,
                        ApiId=api_id
                    )
                    logger.info(f"Message sent to connection {connection_id}")
                    break
                except ClientError as e:
                    if e.response['Error']['Code'] == 'GoneException':
                        # Connection is gone, log and remove
                        logger.warning(f"Connection {connection_id} is gone")
                        logs.put_log_events(
                            logGroupName=LOG_GROUP,
                            logStreamName='connection-errors',
                            logEvents=[{
                                'timestamp': int(time.time() * 1000),
                                'message': f"Connection gone: {connection_id}"
                            }]
                        )
                        break
                    elif attempt < MAX_RETRIES - 1:
                        logger.warning(f"Retrying message send to {connection_id}: {str(e)}")
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.error(f"Failed to send message to {connection_id}: {str(e)}")
                        logs.put_log_events(
                            logGroupName=LOG_GROUP,
                            logStreamName='connection-errors',
                            logEvents=[{
                                'timestamp': int(time.time() * 1000),
                                'message': f"Failed connection: {connection_id}, Error: {str(e)}"
                            }]
                        )
                except Exception as e:
                    logger.error(f"Unexpected error sending to {connection_id}: {str(e)}")
                    break
    except Exception as e:
        logger.error(f"Error broadcasting message: {str(e)}")

def log_drift_metrics(drift_event: Dict[str, Any]) -> None:
    """
    Log drift metrics to CloudWatch with error handling.
    
    Args:
        drift_event: Drift event data
    """
    try:
        # Log drift rate metric
        cloudwatch.put_metric_data(
            Namespace='MLAutomation',
            MetricData=[
                {
                    'MetricName': 'DriftRate',
                    'Value': drift_event.get('drift_rate', 0),
                    'Unit': 'Count',
                    'Timestamp': time.time()
                }
            ]
        )
        
        # Log feature drift metrics if available
        if 'feature_drift' in drift_event:
            for feature, drift_value in drift_event['feature_drift'].items():
                cloudwatch.put_metric_data(
                    Namespace='MLAutomation',
                    MetricData=[
                        {
                            'MetricName': f'FeatureDrift_{feature}',
                            'Value': drift_value,
                            'Unit': 'Count',
                            'Timestamp': time.time()
                        }
                    ]
                )
        
        # Log significant drift count
        significant_drift_count = sum(
            1 for value in drift_event.get('feature_drift', {}).values()
            if value > float(get_ssm_parameter('DRIFT_THRESHOLD', '0.1'))
        )
        cloudwatch.put_metric_data(
            Namespace='MLAutomation',
            MetricData=[
                {
                    'MetricName': 'SignificantDriftCount',
                    'Value': significant_drift_count,
                    'Unit': 'Count',
                    'Timestamp': time.time()
                }
            ]
        )
        
        logger.info(f"Logged drift metrics for {len(drift_event.get('feature_drift', {}))} features")
    except Exception as e:
        logger.error(f"Error logging drift metrics: {str(e)}")

def log_drift_event(drift_event: Dict[str, Any]) -> None:
    """
    Log drift event to CloudWatch Logs with error handling.
    
    Args:
        drift_event: Drift event data
    """
    try:
        # Ensure log group exists
        try:
            logs.create_log_group(logGroupName=DRIFT_LOG_GROUP)
            logger.info(f"Created log group {DRIFT_LOG_GROUP}")
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                logger.error(f"Error creating log group {DRIFT_LOG_GROUP}: {str(e)}")
                return
        
        # Create daily log stream
        log_stream = time.strftime('%Y-%m-%d')
        ensure_log_stream(DRIFT_LOG_GROUP, log_stream)
        
        # Log the drift event
        logs.put_log_events(
            logGroupName=DRIFT_LOG_GROUP,
            logStreamName=log_stream,
            logEvents=[
                {
                    'timestamp': int(time.time() * 1000),
                    'message': json.dumps(drift_event)
                }
            ]
        )
        
        logger.info(f"Logged drift event to {DRIFT_LOG_GROUP}/{log_stream}")
    except Exception as e:
        logger.error(f"Error logging drift event: {str(e)}")

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle drift events and broadcast them to connected clients.
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        # Parse the drift event
        if 'body' not in event:
            raise ValueError("Missing 'body' in event")
        
        drift_event = json.loads(event['body'])
        if not isinstance(drift_event, dict):
            raise ValueError("Invalid drift event format")
        
        # Log the drift event and metrics
        log_drift_event(drift_event)
        log_drift_metrics(drift_event)
        
        # Get WebSocket endpoint from SSM
        websocket_endpoint = get_ssm_parameter('websocket/endpoint')
        if not websocket_endpoint:
            logger.warning("WebSocket endpoint not found in SSM")
            websocket_endpoint = event['requestContext']['domainName']
        
        # Broadcast the drift event to all connected clients
        broadcast_message(
            {
                'type': 'driftEvent',
                'data': drift_event,
                'timestamp': int(time.time() * 1000)
            },
            websocket_endpoint
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Drift event processed successfully'
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
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'Invalid JSON in request body'
            })
        }
    except Exception as e:
        logger.error(f"Error processing drift event: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error'
            })
        } 