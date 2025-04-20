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
   - Auto‑promote to “Production” in MLflow Registry
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
   # For Airflow DAGs
   cd dags && pip install -r requirements.txt

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
├── .env.example                 # Example environment variables
├── .gitignore
├── airflow.cfg                  # Airflow configuration on EC2
├── webserver_config.py          # Airflow webserver settings
└── README.md                    # Project overview & instructions
```

---

## 📦 Usage

1. **Trigger the pipeline**:
   - **Auto**: runs daily at 10 AM (Airflow schedule)
   - **Manual**: `airflow dags trigger homeowner_loss_history_full_pipeline`
2. **Monitor MLflow**: `http://<EC2_PUBLIC_IP>:5000`
3. **View Dashboard**: `http://<EC2_PUBLIC_IP>:3000` or your Vercel URL
4. **Check Alerts**: Slack channel `#alerts`

---

## 🤝 Contributing

1. Fork & checkout `main`  
2. Create feature branch & implement changes  
3. Update `.env.example` if env vars change  
4. Submit PR & pass CI checks

---

## ⚖️ License

=