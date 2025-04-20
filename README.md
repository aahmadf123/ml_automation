```markdown
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
   - Ingest raw CSV from S3 → preprocess (missing data, outliers, encoding) → parquet
   - Validate schema with Pandera → version & snapshot on S3
2. **Drift detection & self‑healing**
   - Compute & upload timestamped reference means
   - Compare new data → branch to self‑heal or train
3. **Automated training & tuning**
   - HyperOpt search for best XGBoost hyperparams
   - Fallback from TimeSeriesSplit → random split
   - Log RMSE, MSE, MAE, R² to MLflow
   - Generate + upload SHAP summary and Actual vs Predicted plots
   - Auto‑promote to “Production” stage in MLflow Registry
4. **Dashboard & API**
   - Next.js frontend served on Vercel (or EC2)
   - WebSocket updates for live metrics & drift alerts
5. **Alerts & notifications**
   - Slack webhooks for drift, profiling summaries, training results
6. **Version control & collaboration**
   - All DAGs, scripts, and dashboard code in a single GitHub repo
   - Environment variables isolated in `.env`

---

## 📐 Architecture

```

```
                            +------------+      +----------------+
```

CSV on S3 ───▶ Ingest Task  ──▶ Parquet ─▶ Preprocess Task ──▶ Processed Parquet
(Airflow)                       │
│
▼
Drift Check Task
──▶ Self‑heal ─▶ Slack
──▶ Train Branch ───► HyperOpt + XGBoost
│
▼
MLflow (Metrics + Models)
│
▼
Next.js Dashboard ← WebSocket API ← Lambda / AppSync
│
▼
Slack Notifications

````

---

## 🛠️ Getting Started

### Prerequisites

- **AWS account** with S3 bucket & IAM credentials
- **EC2 instance** (Ubuntu) with Docker (or Python 3.12 venv)
- **Apache Airflow** (2.10+) installed on EC2
- **MLflow** server running (can be on EC2 or separate)
- **Next.js** for dashboard (local or Vercel)

### Configuration

1. **Clone repository**
   ```bash
   git clone git@github.com:YourOrg/ml_automation.git
   cd ml_automation
````

2.  **Create & populate `.env`** (see [`.env.example`](https://www.google.com/search?q=./.env.example))
3.  **Install Python dependencies**
    ```bash
    cd dags
    pip install -r requirements.txt
    ```
4.  **Initialize Airflow DB & start services**
    ```bash
    export AIRFLOW_HOME=~/airflow
    airflow db init
    airflow webserver --port 8080 &
    airflow scheduler &
    ```
5.  **Start MLflow server**
    ```bash
    mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root s3://$S3_BUCKET/mlruns \
      --host 0.0.0.0 --port 5000
    ```
6.  **Deploy Dashboard**
      - Locally: `cd loss-history-dashboard && npm install && npm run dev`
      - Vercel: Connect to `main` branch & set same env vars

-----

## 📂 Repository Structure

```
ml_automation/
├── dags/                         # Airflow DAGs & task modules
│   ├── homeowner_dag.py
│   └── tasks/                    # Ingestion, preprocessing, drift, training, etc.
├── loss-history-dashboard/       # Next.js frontend & WebSocket client
├── mlflow-export-import/          # Optional scripts for MLflow model registry
├── .env.example                  # Example env vars
├── .gitignore
├── airflow.cfg                   # Airflow configuration
├── webserver_config.py           # Airflow webserver settings
└── README.md                     # Project overview & instructions
```

-----

## 📦 Usage

1.  **Trigger the pipeline**
      - Auto: runs daily at 10 AM via schedule
      - Manual:
        ```bash
        airflow dags trigger homeowner_loss_history_full_pipeline
        ```
2.  **Watch training & promotion**
      - Check MLflow UI: `http://<mlflow-host>:5000`
3.  **View Dashboard**
      - Visit `http://localhost:3000` or your Vercel URL
4.  **Inspect Alerts**
      - Slack channel: `#alerts`

-----

## 🤝 Contributing

1.  Fork & branch off `main`
2.  Add or update DAGs, tasks, or dashboard components
3.  Update `.env.example` if you introduce new env vars
4.  Submit a PR, and review CI checks

