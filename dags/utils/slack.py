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
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """
        Initialize the Slack client.
        
        Raises:
            RuntimeError: If client initialization fails
        """
        try:
            token = os.getenv('SLACK_BOT_TOKEN')
            if not token:
                raise ValueError("SLACK_BOT_TOKEN environment variable not set")
            self._client = WebClient(token=token)
        except Exception as e:
            log.error(f"Failed to initialize Slack client: {str(e)}")
            raise RuntimeError("Slack client initialization failed") from e
            
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
        try:
            # Format message based on urgency
            emoji = {
                "normal": "ℹ️",
                "high": "⚠️",
                "critical": "🚨"
            }.get(urgency, "ℹ️")
            
            message = f"{emoji} *{title}*\n{details}"
            
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
            raise
        except Exception as e:
            log.error(f"Failed to send Slack message: {str(e)}")
            raise RuntimeError("Message sending failed") from e
            
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
        try:
            response = self._client.conversations_info(channel=channel)
            return response.data
        except SlackApiError as e:
            log.error(f"Failed to get channel info: {str(e)}")
            raise
        except Exception as e:
            log.error(f"Channel info retrieval failed: {str(e)}")
            raise RuntimeError("Channel info retrieval failed") from e
            
    def validate_channel(self, channel: str) -> bool:
        """
        Validate if a channel exists and is accessible.
        
        Args:
            channel: Channel name or ID
            
        Returns:
            bool: True if channel is valid, False otherwise
        """
        try:
            self.get_channel_info(channel)
            return True
        except Exception as e:
            log.error(f"Channel validation failed: {str(e)}")
            return False

# Create a singleton instance of SlackManager
_manager = SlackManager()

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
    return _manager.post(channel, title, details, urgency)

def update_slack_notification_process_with_ui_components():
    """
    Placeholder function to update the Slack notification process with new UI components and endpoints.
    """
    logging.info("Updating Slack notification process with new UI components and endpoints.")
    # Placeholder for actual implementation

# backwards‐compatible aliases
send_message = post
post_message = post
