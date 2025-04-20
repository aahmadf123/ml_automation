# ML Automation & Loss‑History Pipeline

A unified MLOps project that powers a daily homeowner loss‑history pipeline with:

- **Data ingestion & preprocessing** via Apache Airflow on EC2  
- **Drift monitoring & self‑healing** with versioned reference means  
- **Automated hyperparameter tuning & training** using HyperOpt + XGBoost  
- **Model tracking & promotion** in MLflow (with metrics, SHAP + Actual‑vs‑Pred plots)  
- **Real‑time dashboard** (Next.js + WebSockets) for system and model observability  
- **Notifications & alerts** via Slack integration  
- **Persistent storage** of data, artifacts, and models on S3

---

## 🚀 Features

1. **End‑to‑end pipeline**  
   - Ingest raw CSV from S3 → preprocess (missing data, outliers, encoding) → Parquet  
   - Validate schema with Pandera → version & snapshot on S3
2. **Drift detection & self‑healing**  
   - Compute & upload timestamped reference means  
   - Compare new data → branch to self‑heal or train
3. **Automated training & tuning**  
   - HyperOpt search for best XGBoost hyperparameters  
   - Fallback from TimeSeriesSplit → train_test_split  
   - Log RMSE, MSE, MAE, R² to MLflow  
   - Generate & upload SHAP summary & Actual vs Predicted plots  
   - Auto‑promote to "Production" in MLflow Registry
4. **Dashboard & API**  
   - Next.js frontend served on Vercel or EC2  
   - WebSocket updates for live metrics & drift alerts
5. **Alerts & notifications**  
   - Slack webhooks for drift, profiling, training results
6. **Version control & collaboration**  
   - All DAGs, scripts, and dashboard code in a single GitHub repo  
   - Environment variables isolated in `.env`

---

## 🛠️ Getting Started

### Prerequisites

- AWS account with S3 bucket & IAM credentials  
- EC2 instance (Ubuntu) running Airflow & MLflow  
- Node.js (for dashboard) or Vercel account

### Setup

1. **Clone repository**:
   ```bash
   git clone git@github.com:YourOrg/ml_automation.git
   cd ml_automation
   ```
2. **Configure environment**:
   - Copy `.env.example` to `.env` and fill in keys (Slack, AWS, Airflow, MLflow, WebSocket URL)
3. **Install dependencies**:
   ```bash
   # For Airflow DAGs and backend
   pip install -r requirements.txt

   # For Dashboard
   cd loss-history-dashboard && npm install
   ```
4. **Start services on EC2**:
   ```bash
   # Airflow
   export AIRFLOW_HOME=~/airflow
   airflow db init
   airflow webserver --port 8080 &
   airflow scheduler &

   # MLflow
   mlflow server --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root s3://$S3_BUCKET/mlruns \
     --host 0.0.0.0 --port 5000 &
   ```
5. **Run Dashboard**:
   - **Local**: `npm run dev` in `loss-history-dashboard`
   - **Vercel**: connect `main` branch & configure same env vars

---

## 📂 Repository Structure

```
ml_automation/
├── dags/                        # Airflow DAGs & task modules
│   ├── homeowner_dag.py
│   └── tasks/                   # Ingestion, preprocessing, drift, training, etc.
├── loss-history-dashboard/      # Next.js frontend & WebSocket client
├── mlflow-export-import/        # Optional MLflow registry scripts
├── tests/                       # Test files
│   ├── test_api_endpoints.js    # API endpoint tests
│   ├── test_websocket.js        # WebSocket functionality tests
│   └── test_preprocessing.py    # Preprocessing module tests
├── .env.example                 # Example environment variables
├── .gitignore
├── airflow.cfg                  # Airflow configuration on EC2
├── webserver_config.py          # Airflow webserver settings
├── requirements.txt             # Python dependencies
├── package.json                 # JavaScript/TypeScript dependencies
└── README.md                    # Project overview & instructions
```

---

## 📊 Dashboard Components

The dashboard consists of several key components:

1. **System Metrics**: Real-time monitoring of CPU, memory, disk, and network usage
2. **Model Performance**: Visualization of model metrics (RMSE, MSE, MAE, R²) over time
3. **Data Drift Alerts**: Detection and visualization of feature drift
4. **Pipeline Health**: Status of DAG execution and success rates
5. **Model Explainability**: SHAP values and actual vs. predicted comparisons
6. **Slack Notifications**: Configuration and history of alerts sent to Slack

## 🔄 WebSocket Communication

The dashboard uses WebSockets for real-time updates. The WebSocket server runs on port 8000 and provides the following message types:

- `connection`: Confirmation of successful connection
- `system_metrics`: CPU, memory, disk, and network usage
- `model_metrics`: Model performance metrics (RMSE, MSE, MAE, R²)
- `data_drift_alert`: Alerts for feature drift detection
- `pipeline_status`: DAG execution status and progress

## 🧪 Testing

### Backend Tests

```bash
pytest
```

### Frontend Tests

```bash
npm test
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.