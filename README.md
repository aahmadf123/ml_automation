# ML Automation Pipeline

A comprehensive machine learning automation pipeline for homeowner loss history prediction, featuring real-time monitoring, data quality checks, model explainability, and A/B testing.

## Overview

This project implements a production-ready ML pipeline with the following key features:

- **Data Ingestion & Preprocessing**: Automated data ingestion from S3, preprocessing, and schema validation
- **Data Quality Monitoring**: Continuous monitoring of data quality with anomaly detection
- **Model Explainability**: SHAP values and feature importance tracking
- **A/B Testing**: Robust A/B testing framework for model promotion
- **Real-time Monitoring**: WebSocket-based real-time dashboard updates
- **Automated Retraining**: Drift detection and automated model retraining
- **Human-in-the-Loop**: Manual override capabilities for critical decisions

## Architecture

### Components

1. **Airflow DAGs**: Orchestrate the entire ML pipeline
2. **Data Quality Monitor**: Tracks data quality metrics and detects anomalies
3. **Model Explainability Tracker**: Monitors model interpretability
4. **A/B Testing Pipeline**: Manages model comparison and promotion
5. **WebSocket Server**: Provides real-time updates to the dashboard
6. **MLflow Integration**: Tracks experiments and model metrics

### AWS Infrastructure

1. **Amazon MWAA**: Managed Apache Airflow service
   - DAGs stored in S3
   - Automatic high availability
   - IAM integration

2. **SageMaker Model Registry**: Model versioning and deployment
   - Automated model promotion
   - Version control
   - Secure deployment hooks

3. **AWS Amplify**: Dashboard hosting
   - Automatic builds on main branch
   - HTTPS and custom domain
   - Environment variable management

4. **API Gateway + Lambda**: Real-time updates
   - WebSocket API
   - Serverless architecture
   - Auto-scaling

5. **AWS Secrets Manager**: Secure configuration
   - Centralized secrets storage
   - IAM-based access control
   - Encrypted at rest

6. **CloudWatch + SNS**: Monitoring and alerts
   - Centralized logging
   - Custom metrics
   - Slack integration

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- AWS CLI configured
- GitHub account
- AWS account with appropriate permissions

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml_automation.git
   cd ml_automation
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

### AWS Deployment

1. **Infrastructure Setup**:
   ```bash
   # Install AWS CDK
   npm install -g aws-cdk

   # Deploy infrastructure
   cd infrastructure
   cdk deploy
   ```

2. **MWAA Environment**:
   - Create S3 bucket for DAGs
   - Configure MWAA environment
   - Set up IAM roles

3. **Model Registry**:
   - Configure SageMaker Model Registry
   - Set up model promotion workflow
   - Configure deployment hooks

4. **Dashboard Deployment**:
   - Connect GitHub repository to Amplify
   - Configure build settings
   - Set up environment variables

5. **WebSocket API**:
   - Deploy API Gateway WebSocket API
   - Configure Lambda functions
   - Set up connections

## Development

### Project Structure

```
ml_automation/
├── dags/                    # Airflow DAGs
│   ├── tasks/              # Task implementations
│   └── utils/              # Utility functions
├── loss-history-dashboard/ # React dashboard
├── infrastructure/         # AWS CDK code
├── tests/                  # Test files
├── requirements.txt        # Python dependencies
└── .env.template          # Environment template
```

### CI/CD Pipeline

1. **GitHub Actions Workflow**:
   - Run tests on PR
   - Deploy infrastructure
   - Update MWAA environment
   - Deploy dashboard

2. **Infrastructure as Code**:
   - AWS CDK for infrastructure
   - Automated deployments
   - Environment management

3. **Monitoring**:
   - CloudWatch metrics
   - SNS notifications
   - Slack integration

## Testing

Run the test suite:
```bash
# Backend tests
pytest tests/

# Frontend tests
cd loss-history-dashboard
npm test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.