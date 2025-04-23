# How to Update Slack App Permissions

To fix the Slack API error, follow these steps to update your Slack app permissions:

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps) and log in
2. Select your ML Automation app
3. In the left sidebar, click on "OAuth & Permissions"
4. Scroll down to "Bot Token Scopes" section
5. Click "Add an OAuth Scope"
6. Add the `chat:write` or `chat:write:bot` scope
7. Scroll up and click "Reinstall to Workspace"
8. Authorize the app when prompted
9. Copy the new Bot User OAuth Token
10. Update your environment or Airflow Variable with the new token:
    - As an Airflow Variable: `airflow variables set SLACK_BOT_TOKEN "xoxb-your-token"`
    - As an environment variable: Add to your environment or .env file

## Required Slack Channels
Make sure the following Slack channels exist in your workspace:

1. `#alerts` - For important data and processing alerts
2. `#general` - Used as a fallback channel
3. `#agent_logs` - For general logging information

To create these channels:
1. In Slack, click on the + next to "Channels" in the sidebar
2. Click "Create Channel"
3. Name the channel (without the #) and set privacy options
4. Click "Create"
5. Make sure to add your Slack bot to each channel by typing `/invite @YourBotName` in the channel

## Current Scopes
Current scopes that need to be updated:
- identify
- app_configurations:read
- app_configurations:write
+ chat:write (or chat:write:bot) 