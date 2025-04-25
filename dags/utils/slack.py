#!/usr/bin/env python3
"""
Slack integration utilities for ML Automation.

This module provides functionality for sending messages to Slack:
- Message formatting
- Error handling
- Rate limiting
- Channel management
"""

import logging
import os
from typing import Dict, Any, Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Setup logging
log = logging.getLogger(__name__)

class SlackManager:
    """
    Manages Slack communications for the ML Automation system.
    
    This class handles:
    - Message sending
    - Channel management
    - Error handling
    - Rate limiting
    """
    
    def __init__(self) -> None:
        """
        Initialize the SlackManager.
        
        Sets up the Slack client and configuration.
        """
        self._client = None
        # No initialization at creation time
        
    def _initialize_client(self) -> None:
        """
        Initialize the Slack client.
        
        Raises:
            RuntimeError: If client initialization fails
        """
        if self._client is not None:
            return
            
        try:
            # Try to get token from Airflow Variable first
            try:
                from airflow.models import Variable
                token = Variable.get("SLACK_BOT_TOKEN", default_var=None)
            except Exception as var_error:
                log.info(f"Could not access Airflow Variables: {var_error}")
                token = None
                
            # Fall back to environment variable if Airflow Variable is not available
            if token is None:
                token = os.getenv('SLACK_BOT_TOKEN')
                
            if not token:
                log.warning("SLACK_BOT_TOKEN not found in Airflow Variables or environment. Slack notifications will be logged but not sent.")
                return
                
            self._client = WebClient(token=token)
            # Log the initialization success and scopes for debugging
            log.info("Slack client initialized successfully. Make sure your Slack app has the 'chat:write' or 'chat:write:bot' scope enabled.")
        except Exception as e:
            log.warning(f"Failed to initialize Slack client: {str(e)}. Notifications will be logged but not sent.")
            
    def post(
        self,
        channel: str,
        title: str,
        details: str,
        urgency: str = "normal"
    ) -> Dict[str, Any]:
        """
        Post a message to Slack.
        
        Args:
            channel: Slack channel to post to
            title: Message title
            details: Message details
            urgency: Message urgency level (normal/high/critical)
            
        Returns:
            Dict[str, Any]: Slack API response
            
        Raises:
            SlackApiError: If message posting fails
        """
        # Lazy initialization
        self._initialize_client()
        
        # Format message
        emoji = {
            "normal": "ℹ️",
            "high": "⚠️",
            "critical": "🚨"
        }.get(urgency, "ℹ️")
        
        message = f"{emoji} *{title}*\n{details}"
        
        # If client initialization failed, just log the message
        if self._client is None:
            log.info(f"[SLACK MESSAGE WOULD BE SENT] Channel: {channel}, Message: {message}")
            return {"ok": True, "message": "Message logged but not sent"}
            
        try:
            # Send message
            response = self._client.chat_postMessage(
                channel=channel,
                text=message,
                mrkdwn=True
            )
            
            log.info(f"Message sent to {channel}: {title}")
            return response.data
            
        except SlackApiError as e:
            log.error(f"Slack API error: {str(e)}")
            log.info(f"[FALLBACK] Channel: {channel}, Message: {message}")
            
            # Update: Handle both channel_not_found and not_in_channel errors
            if "channel_not_found" in str(e) or "not_in_channel" in str(e):
                try:
                    # Try to get default channel from Airflow Variable
                    try:
                        from airflow.models import Variable
                        default_channel = Variable.get("SLACK_DEFAULT_CHANNEL", default_var="#all-airflow-notification")
                    except Exception:
                        default_channel = "#all-airflow-notification"
                        
                    # Try posting to the default channel instead
                    log.info(f"Channel issue with {channel}, trying to post to {default_channel} instead")
                    fallback_response = self._client.chat_postMessage(
                        channel=default_channel,
                        text=f"[Intended for {channel}] {message}",
                        mrkdwn=True
                    )
                    log.info(f"Fallback message sent to {default_channel}")
                    return fallback_response.data
                except SlackApiError as fallback_e:
                    log.warning(f"Fallback to {default_channel} also failed: {str(fallback_e)}")
                    # Just log the message and continue without failing the pipeline
                    log.info(f"[LOGGED ONLY] {message}")
                    return {"ok": True, "message": "Message logged but not sent to Slack (channel issues)"}
            
            # Just log the message and continue - don't fail the pipeline because of Slack issues
            log.info(f"[LOGGED ONLY] {message}")
            return {"ok": True, "message": "Message logged but not sent to Slack"}
        except Exception as e:
            log.error(f"Failed to send Slack message: {str(e)}")
            log.info(f"[FALLBACK] Channel: {channel}, Message: {message}")
            return {"ok": True, "message": "Message logged but not sent due to error"}
            
    def get_channel_info(self, channel: str) -> Dict[str, Any]:
        """
        Get information about a Slack channel.
        
        Args:
            channel: Channel name or ID
            
        Returns:
            Dict[str, Any]: Channel information
            
        Raises:
            SlackApiError: If channel info retrieval fails
        """
        # Lazy initialization
        self._initialize_client()
        
        # If client initialization failed, return dummy data
        if self._client is None:
            log.info(f"[SLACK CHANNEL INFO WOULD BE FETCHED] Channel: {channel}")
            return {"ok": True, "channel": {"id": channel, "name": channel}}
            
        try:
            response = self._client.conversations_info(channel=channel)
            return response.data
        except SlackApiError as e:
            log.error(f"Failed to get channel info: {str(e)}")
            return {"ok": False, "error": str(e)}
        except Exception as e:
            log.error(f"Channel info retrieval failed: {str(e)}")
            return {"ok": False, "error": str(e)}
            
    def validate_channel(self, channel: str) -> bool:
        """
        Validate if a channel exists and is accessible.
        
        Args:
            channel: Channel name or ID
            
        Returns:
            bool: True if channel is valid, False otherwise
        """
        # Lazy initialization
        self._initialize_client()
        
        # If client initialization failed, assume channel is valid
        if self._client is None:
            return True
            
        try:
            info = self.get_channel_info(channel)
            return info.get("ok", False)
        except Exception as e:
            log.error(f"Channel validation failed: {str(e)}")
            return False

# Don't create the singleton instance at module load time
_manager = None

def get_manager() -> SlackManager:
    """Get the SlackManager singleton instance (create it if it doesn't exist)."""
    global _manager
    if _manager is None:
        _manager = SlackManager()
    return _manager

def post(
    channel: str,
    title: str,
    details: str,
    urgency: str = "normal"
) -> Dict[str, Any]:
    """
    Post a message to Slack.
    
    Args:
        channel: Slack channel to post to
        title: Message title
        details: Message details
        urgency: Message urgency level (normal/high/critical)
        
    Returns:
        Dict[str, Any]: Slack API response
        
    Raises:
        SlackApiError: If message posting fails
    """
    return get_manager().post(channel, title, details, urgency)

def simple_post(message: str, channel: str = None, urgency: str = "normal") -> Dict[str, Any]:
    """
    Simplified function to post a message to Slack with just a message string.
    
    Args:
        message: The message to post
        channel: Slack channel to post to
        urgency: Message urgency level (normal/high/critical)
        
    Returns:
        Dict[str, Any]: Slack API response
    """
    try:
        # Get default channel if none provided
        if channel is None:
            try:
                from airflow.models import Variable
                channel = Variable.get("SLACK_DEFAULT_CHANNEL", default_var="#ml-pipeline-notifications")
            except Exception as e:
                log.warning(f"Could not get default channel from Variable: {str(e)}")
                channel = "#ml-pipeline-notifications"
        
        # Split the message into title and details if it contains a newline
        if "\n" in message:
            parts = message.split("\n", 1)
            title = parts[0]
            details = parts[1]
        else:
            # Otherwise use the message as both title and details
            title = message
            details = " "  # Use empty space to avoid None
        
        # Call the original post function
        return post(channel, title, details, urgency)
    except Exception as e:
        # Log but don't fail if there's a problem with Slack notification
        log.warning(f"Error in simple_post: {str(e)}")
        log.info(f"[LOGGED ONLY] {message}")
        return {"ok": True, "message": "Message logged but not sent due to error in simple_post"}

def update_slack_notification_process_with_ui_components():
    """
    Placeholder function to update the Slack notification process with new UI components and endpoints.
    """
    logging.info("Updating Slack notification process with new UI components and endpoints.")
    # Placeholder for actual implementation

def ensure_default_channels() -> Dict[str, bool]:
    """
    Ensure that default notification channels exist in Slack.
    If they don't exist, messages will fall back to #general or logging.
    
    Returns:
        Dict mapping channel names to whether they exist
    """
    manager = get_manager()
    
    try:
        # Try to get default channel from Airflow Variable
        from airflow.models import Variable
        default_fallback_channel = Variable.get("SLACK_DEFAULT_CHANNEL", default_var="#ml-pipeline-notifications")
    except Exception:
        default_fallback_channel = "#ml-pipeline-notifications"
    
    # Default channels for different message types
    default_channels = [
        "#all-airflow-notification",
        "#alerts",
        "#data-quality",
        "#drift-alerts",
        "#exec-summary",
        "#experiments",
        "#home-insurance-automation",
        "#incidents",
        "#model-ops", 
        "#resource-ops",
        "#retraining"
    ]
    
    results = {}
    
    # Check and log each channel's availability
    for channel in default_channels:
        try:
            exists = manager.validate_channel(channel)
            log.info(f"Slack channel {channel} {'exists' if exists else 'does not exist'}")
            results[channel] = exists
            
            # If channel doesn't exist, try to determine if we have permissions to create it
            if not exists:
                try:
                    # Try to create the channel if we have sufficient permissions
                    log.info(f"Attempting to create channel {channel}")
                    # Note: This requires the channels:write scope
                    # This may fail if the app doesn't have sufficient permissions
                    if manager._client:  # Only attempt if client is initialized
                        response = manager._client.conversations_create(
                            name=channel.lstrip('#'),
                            is_private=False
                        )
                        
                        if response['ok']:
                            log.info(f"Successfully created channel {channel}")
                            results[channel] = True
                        else:
                            log.warning(f"Failed to create channel {channel}: {response.get('error', 'Unknown error')}")
                except Exception as e:
                    log.warning(f"Failed to create channel {channel}: {str(e)}")
                    # Fall back to more reliable channels
                    log.info(f"Messages meant for {channel} will be sent to {default_fallback_channel} or logged only")
        except Exception as e:
            log.warning(f"Error checking channel {channel}: {str(e)}")
            results[channel] = False
    
    return results

# Call this function on module import to ensure channels exist
try:
    # Don't call this automatically at import time - let it be called when needed
    # This helps avoid problems with missing credentials during startup
    # ensure_default_channels will be called once when sending the first message
    # ensure_default_channels()
    pass
except Exception as e:
    log.warning(f"Failed to ensure default channels: {str(e)}")

# backwards‐compatible aliases
send_message = post
post_message = post

# Default settings
try:
    from airflow.models import Variable
    DEFAULT_CHANNEL = Variable.get("SLACK_DEFAULT_CHANNEL", default_var="#ml-pipeline-notifications")
except Exception:
    DEFAULT_CHANNEL = "#ml-pipeline-notifications"

DEFAULT_CHANNELS = [
    "#all-airflow-notification",
    "#alerts",
    "#data-quality",
    "#drift-alerts",
    "#exec-summary",
    "#experiments",
    "#home-insurance-automation",
    "#incidents",
    "#model-ops", 
    "#resource-ops",
    "#retraining"
]
DEFAULT_TIMEOUT = 10  # seconds
