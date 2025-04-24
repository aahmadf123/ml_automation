# Setting Up AWS MWAA with ClearML, Streamlit, and MLflow on EC2

This guide walks through the steps to integrate Amazon Managed Workflows for Apache Airflow (MWAA) with ClearML for experiment tracking, Streamlit for dashboarding, and MLflow running on EC2.

## Prerequisites

- AWS account with appropriate permissions
- AWS CLI configured
- EC2 key pair for SSH access
- Knowledge of your VPC, subnet, and security group IDs

## 1. Setting up MLflow on EC2

### 1.1 Launch an EC2 instance

```bash
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxxxxxxxxxx \
  --subnet-id subnet-xxxxxxxxxxxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=MLflow-Server}]' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}'
```

### 1.2 Set up MLflow server

SSH into your EC2 instance:

```bash
ssh -i /path/to/your-key-pair.pem ec2-user@your-instance-ip
```

Install dependencies:

```bash
sudo yum update -y
sudo yum install -y python3 python3-pip python3-devel git
sudo pip3 install mlflow sqlalchemy boto3 psycopg2-binary
```

Create a PostgreSQL RDS instance for MLflow tracking:

```bash
aws rds create-db-instance \
  --db-instance-identifier mlflow-database \
  --db-instance-class db.t3.small \
  --engine postgres \
  --allocated-storage 20 \
  --master-username mlflow \
  --master-user-password your-secure-password \
  --vpc-security-group-ids sg-xxxxxxxxxxxxxxxxx \
  --db-subnet-group-name your-subnet-group
```

Create an S3 bucket for MLflow artifacts:

```bash
aws s3 mb s3://your-mlflow-artifacts-bucket --region us-east-2
```

Create a systemd service for MLflow:

```bash
sudo tee /etc/systemd/system/mlflow.service > /dev/null << 'EOF'
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ec2-user
Environment="MLFLOW_TRACKING_URI=postgresql://mlflow:your-secure-password@your-rds-endpoint:5432/mlflow"
Environment="MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://your-mlflow-artifacts-bucket"
Environment="AWS_REGION=us-east-2"
ExecStart=/usr/local/bin/mlflow server \
  --backend-store-uri postgresql://mlflow:your-secure-password@your-rds-endpoint:5432/mlflow \
  --default-artifact-root s3://your-mlflow-artifacts-bucket \
  --host 0.0.0.0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
```

## 2. Setting up ClearML Server (Optional)

You can use the hosted ClearML service at https://app.clear.ml or set up your own server:

```bash
docker run -d \
  --name clearml-server \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  -v clearml-data:/opt/clearml/data \
  allegroai/clearml:latest
```

## 3. Configuring AWS MWAA

### 3.1 Update your environment variables

Update your AWS SSM Parameter Store with the necessary configuration:

```bash
# MLflow Configuration
aws ssm put-parameter --name "MLFLOW_TRACKING_URI" --value "http://your-ec2-instance-ip:5000" --type "String" --overwrite
aws ssm put-parameter --name "MLFLOW_EXPERIMENT_NAME" --value "HomeownerLossHistoryProject" --type "String" --overwrite

# ClearML Configuration
aws ssm put-parameter --name "CLEARML_API_SERVER" --value "https://app.clear.ml" --type "String" --overwrite
aws ssm put-parameter --name "CLEARML_WEB_SERVER" --value "https://app.clear.ml" --type "String" --overwrite
aws ssm put-parameter --name "CLEARML_FILES_SERVER" --value "https://files.clear.ml" --type "String" --overwrite
aws ssm put-parameter --name "CLEARML_KEY" --value "your-clearml-key" --type "SecureString" --overwrite
aws ssm put-parameter --name "CLEARML_SECRET" --value "your-clearml-secret" --type "SecureString" --overwrite
aws ssm put-parameter --name "CLEARML_PROJECT" --value "HomeownerLossHistoryProject" --type "String" --overwrite
```

### 3.2 Update MWAA requirements

Update your MWAA requirements.txt file with ClearML and Streamlit dependencies:

```
clearml==1.14.2
streamlit==1.31.0
```

Upload the requirements.txt file to your MWAA S3 bucket:

```bash
aws s3 cp requirements.txt s3://your-mwaa-bucket/requirements.txt
```

### 3.3 Update MWAA environment

Update your MWAA environment to use the new requirements file:

```bash
aws mwaa update-environment \
  --name your-mwaa-environment \
  --requirements-s3-path requirements.txt
```

## 4. Setting up Streamlit Dashboard on EC2

### 4.1 Launch an EC2 instance for Streamlit

```bash
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxxxxxxxxxx \
  --subnet-id subnet-xxxxxxxxxxxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=Streamlit-Dashboard}]' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}'
```

### 4.2 Set up Streamlit

SSH into your EC2 instance:

```bash
ssh -i /path/to/your-key-pair.pem ec2-user@your-streamlit-instance-ip
```

Install dependencies:

```bash
sudo yum update -y
sudo yum install -y python3 python3-pip python3-devel git
git clone https://github.com/your-username/ml_automation.git
cd ml_automation
pip3 install -r requirements.txt
```

Create a systemd service for Streamlit:

```bash
sudo tee /etc/systemd/system/streamlit.service > /dev/null << 'EOF'
[Unit]
Description=Streamlit Dashboard
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/ml_automation
Environment="AWS_REGION=us-east-2"
Environment="MLFLOW_TRACKING_URI=http://your-mlflow-instance-ip:5000"
Environment="AIRFLOW_API_URL=https://your-mwaa-webserver-url/api/v1"
Environment="AIRFLOW_USERNAME=airflow-username"
Environment="AIRFLOW_PASSWORD=airflow-password"
Environment="CLEARML_API=https://app.clear.ml/api"
Environment="CLEARML_TOKEN=your-clearml-token"
ExecStart=/usr/local/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
```

## 5. Configure Security Groups and Access

### 5.1 MLflow Server Security Group

Ensure the MLflow server security group allows:
- Inbound TCP 5000 from the MWAA security group
- Inbound TCP 5000 from the Streamlit security group
- Inbound TCP 22 for SSH access

### 5.2 Streamlit Server Security Group

Ensure the Streamlit server security group allows:
- Inbound TCP 8501 from your required IP ranges
- Inbound TCP 22 for SSH access

### 5.3 MWAA Security Group

Ensure the MWAA security group allows:
- Outbound TCP to MLflow server (5000)
- Outbound TCP to ClearML server (80/443)

## 6. Testing the Integration

1. Connect to your Streamlit dashboard: http://your-streamlit-instance-ip:8501
2. Navigate to the MLflow experiments section
3. Verify that you can see experiments from MLflow
4. Navigate to the Airflow DAGs section
5. Verify that you can see and trigger DAGs
6. Run a DAG that includes model training
7. Verify the run appears in both MLflow and ClearML

## 7. Troubleshooting

### MLflow Connection Issues

If MWAA cannot connect to MLflow:
- Verify security group settings
- Check that MLflow service is running on EC2
- Ensure the MLflow tracking URI is correctly set in MWAA

### ClearML Connection Issues

If ClearML integration is not working:
- Verify API keys are correctly set
- Check network connectivity from MWAA to ClearML
- Look at ClearML logs for authentication issues

### Streamlit Dashboard Issues

If the Streamlit dashboard isn't loading:
- Check Streamlit service status on EC2
- Verify security group settings
- Examine Streamlit logs for connection errors

## 8. Resources

- [AWS MWAA Documentation](https://docs.aws.amazon.com/mwaa/latest/userguide/what-is-mwaa.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [ClearML Documentation](https://clear.ml/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/) 