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

These permissions will allow the app to post messages to channels.

## Current Scopes
Current scopes that need to be updated:
- identify
- app_configurations:read
- app_configurations:write
+ chat:write (or chat:write:bot) 