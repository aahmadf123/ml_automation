# ML Automation Configuration Guide

This document provides guidance on properly configuring the ML automation system.

## MLflow Configuration

The system uses MLflow for experiment tracking, model registry, and artifact storage. The following environment variables should be set:

### AWS Secrets Manager (airflow-secrets)

| Key | Description | Example Value |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | URI for the MLflow tracking server | `http://3.146.46.179:5000` |
| `MLFLOW_ARTIFACT_URI` | S3 path for MLflow artifacts | `s3://grange-seniordesign-bucket/mlflow-artifacts/` |
| `MLFLOW_DB_URI` | Database connection for MLflow | `postgresql://mlflow:password@your-db-host:5432/mlflow` |
| `MODEL_REGISTRY_URI` | URI for the Model Registry (usually same as tracking) | `http://3.146.46.179:5000` |

### Airflow Variables

| Key | Description | Example Value |
| --- | --- | --- |
| `MLFLOW_EXPERIMENT_NAME` | Name of the MLflow experiment | `Homeowner_Loss_Hist_Proj` |
| `AUTO_APPROVE_MODEL` | Whether to automatically approve models | `True` or `False` |
| `REQUIRE_MODEL_APPROVAL` | Whether model approval is required | Set to opposite of `AUTO_APPROVE_MODEL` |

## WebSocket Configuration

The WebSocket configuration is used for real-time updates to the dashboard.

### AWS Secrets Manager (dashboard-secrets)

| Key | Description | Example Value |
| --- | --- | --- |
| `NEXT_PUBLIC_WEBSOCKET_URL` | WebSocket URL for the client | `wss://g31s1e12m7.execute-api.us-east-2.amazonaws.com/prod` |
| `WS_HOST` | WebSocket host (without protocol) | `g31s1e12m7.execute-api.us-east-2.amazonaws.com/prod` |

**Note**: The `WS_HOST` should NOT include any protocol prefix like `ss://` or `wss://`. The protocol is added by the application code.

## S3 Configuration

### AWS Secrets Manager (airflow-secrets) or Airflow Variables

| Key | Description | Example Value |
| --- | --- | --- |
| `DATA_BUCKET` | Primary bucket for application data | `grange-seniordesign-bucket` |
| `S3_BUCKET` | Bucket for Airflow DAGs and files | `mlautomationstack-dagsbucket3bcf9ca5-uhw98w1yrjzq` |
| `S3_ARCHIVE_FOLDER` | Folder for archiving old artifacts | `archive` |

## Configuration Fixes

### 1. Added MLflow Artifact URI

The `MLFLOW_ARTIFACT_URI` variable was added to specify the S3 location for MLflow artifacts.

### 2. Added MLflow Database URI

The `MLFLOW_DB_URI` variable was added for MLflow database connection.

### 3. Fixed Model Approval Conflict

The system was configured to both auto-approve models and require approval, which is contradictory. We changed it so that `REQUIRE_MODEL_APPROVAL` is set to the opposite of `AUTO_APPROVE_MODEL`.

### 4. WebSocket Host Fix

A script (`fix_websocket_host.py`) was created to fix the `WS_HOST` value in AWS Secrets Manager by removing the invalid `ss://` prefix.

## Applying Changes

1. The configuration changes are applied automatically when you restart the Airflow services.

2. To fix the WebSocket host issue, run:
   ```bash
   python fix_websocket_host.py
   ```

3. To update AWS SSM parameters:
   ```bash
   aws ssm put-parameter --name "/ml-automation/MLFLOW_ARTIFACT_URI" --value "s3://grange-seniordesign-bucket/mlflow-artifacts/" --type "String" --overwrite
   
   aws ssm put-parameter --name "/ml-automation/MLFLOW_DB_URI" --value "sqlite:////tmp/mlflow.db" --type "String" --overwrite
   
   aws ssm put-parameter --name "/ml-automation/MODEL_REGISTRY_URI" --value "http://3.146.46.179:5000" --type "String" --overwrite
   ``` 