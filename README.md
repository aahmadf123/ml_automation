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

The system consists of several interconnected components:

1. **Airflow DAGs**: Orchestrate the entire ML pipeline
2. **Data Quality Monitor**: Tracks data quality metrics and detects anomalies
3. **Model Explainability Tracker**: Monitors model interpretability
4. **A/B Testing Pipeline**: Manages model comparison and promotion
5. **WebSocket Server**: Provides real-time updates to the dashboard
6. **MLflow Integration**: Tracks experiments and model metrics

## Setup

### Prerequisites

- Python 3.12+
- Apache Airflow 2.7.1+
- PostgreSQL
- Redis
- MLflow

### Installation

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

5. Initialize Airflow:
   ```bash
   airflow db init
   airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
   ```

### Configuration

1. **S3 Configuration**:
   - Set `S3_BUCKET` in `.env`
   - Configure AWS credentials

2. **MLflow Configuration**:
   - Set `MLFLOW_TRACKING_URI` in `.env`
   - Initialize MLflow server

3. **Slack Integration**:
   - Set `SLACK_WEBHOOK_URL` in `.env`
   - Configure notification channels

## Usage

### Starting the Pipeline

1. Start Airflow:
   ```bash
   airflow webserver -p 8080
   airflow scheduler
   ```

2. Start the WebSocket server:
   ```bash
   python websocket_server.py
   ```

3. Access the dashboard at `http://localhost:3000`

### Monitoring

- **Data Quality**: View data quality metrics and alerts in the dashboard
- **Model Performance**: Track model metrics and drift detection
- **A/B Testing**: Monitor test results and model comparisons
- **System Health**: View system metrics and pipeline status

## Development

### Project Structure

```
ml_automation/
├── dags/                    # Airflow DAGs
│   ├── tasks/              # Task implementations
│   └── utils/              # Utility functions
├── loss-history-dashboard/ # React dashboard
├── tests/                  # Test files
├── requirements.txt        # Python dependencies
└── .env.template          # Environment template
```

### Adding New Features

1. Create new task modules in `dags/tasks/`
2. Update the DAG in `dags/homeowner_dag.py`
3. Add corresponding UI components in `loss-history-dashboard/`
4. Update tests in `tests/`

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.