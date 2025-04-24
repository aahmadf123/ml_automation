import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
import boto3
import mlflow
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ML Automation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Homeowner_Loss_Hist_Proj")
AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
CLEARML_API = os.getenv("CLEARML_API", "https://app.clear.ml/api")
CLEARML_TOKEN = os.getenv("CLEARML_TOKEN", "")

# Initialize clients
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Helper functions
@st.cache_data(ttl=300)
def get_mlflow_experiments():
    """Get MLflow experiments"""
    try:
        return mlflow.search_experiments()
    except Exception as e:
        st.error(f"Error connecting to MLflow: {e}")
        return []

@st.cache_data(ttl=300)
def get_mlflow_runs(experiment_id):
    """Get MLflow runs for a given experiment"""
    try:
        return mlflow.search_runs(experiment_ids=[experiment_id])
    except Exception as e:
        st.error(f"Error fetching MLflow runs: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_airflow_dags():
    """Get Airflow DAGs"""
    try:
        response = requests.get(
            f"{AIRFLOW_API_URL}/dags",
            auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
        )
        if response.status_code == 200:
            return response.json()["dags"]
        else:
            st.warning(f"Could not fetch Airflow DAGs: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to Airflow: {e}")
        return []

@st.cache_data(ttl=300)
def get_clearml_tasks():
    """Get ClearML tasks"""
    if not CLEARML_TOKEN:
        return []
    
    try:
        response = requests.get(
            f"{CLEARML_API}/projects",
            headers={"Authorization": f"Bearer {CLEARML_TOKEN}"}
        )
        if response.status_code == 200:
            return response.json()["data"]
        else:
            st.warning(f"Could not fetch ClearML tasks: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to ClearML: {e}")
        return []

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/machine-learning.png", width=100)
st.sidebar.title("ML Automation Dashboard")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "MLflow Experiments", "Airflow DAGs", "ClearML Tasks", "Model Comparison"]
)

# Overview Page
if page == "Overview":
    st.title("ML Automation Dashboard")
    st.markdown("## Overview of ML Systems")
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("MLflow")
        experiments = get_mlflow_experiments()
        st.metric(label="Experiments", value=len(experiments))
    
    with col2:
        st.info("Airflow")
        dags = get_airflow_dags()
        st.metric(label="DAGs", value=len(dags))
    
    with col3:
        st.info("ClearML")
        tasks = get_clearml_tasks()
        st.metric(label="Tasks", value=len(tasks))
    
    with col4:
        st.info("System")
        st.metric(label="Health", value="98%")
    
    # Recent runs
    st.markdown("## Recent Model Runs")
    
    if experiments:
        experiment_id = experiments[0].experiment_id
        runs = get_mlflow_runs(experiment_id)
        
        if not runs.empty:
            st.dataframe(
                runs[["run_id", "start_time", "metrics.rmse", "metrics.r2", "status"]].head(5),
                use_container_width=True
            )
            
            # Performance metrics chart
            st.markdown("## Model Performance Over Time")
            fig, ax = plt.subplots(figsize=(10, 6))
            runs["start_time"] = pd.to_datetime(runs["start_time"])
            runs = runs.sort_values("start_time")
            
            ax.plot(runs["start_time"], runs["metrics.rmse"], label="RMSE", marker="o")
            ax.set_xlabel("Run Time")
            ax.set_ylabel("RMSE")
            ax.set_title("Model Performance (RMSE) Over Time")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("No MLflow runs found.")
    else:
        st.info("No MLflow experiments found.")

# MLflow Experiments Page
elif page == "MLflow Experiments":
    st.title("MLflow Experiments")
    
    experiments = get_mlflow_experiments()
    if experiments:
        experiment_options = {exp.name: exp.experiment_id for exp in experiments}
        selected_experiment = st.selectbox(
            "Select Experiment", 
            options=list(experiment_options.keys())
        )
        
        experiment_id = experiment_options[selected_experiment]
        runs = get_mlflow_runs(experiment_id)
        
        if not runs.empty:
            st.dataframe(runs, use_container_width=True)
            
            # Filter to show specific run details
            run_ids = runs["run_id"].tolist()
            selected_run = st.selectbox("Select Run for Details", run_ids)
            
            if selected_run:
                run_data = runs[runs["run_id"] == selected_run].iloc[0]
                
                st.markdown("## Run Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Metrics")
                    metrics = {k.replace("metrics.", ""): v for k, v in run_data.items() 
                              if k.startswith("metrics.")}
                    st.json(metrics)
                
                with col2:
                    st.markdown("### Parameters")
                    params = {k.replace("params.", ""): v for k, v in run_data.items() 
                             if k.startswith("params.")}
                    st.json(params)
        else:
            st.info("No runs found for this experiment.")
    else:
        st.info("No MLflow experiments found.")

# Airflow DAGs Page
elif page == "Airflow DAGs":
    st.title("Airflow DAGs")
    
    dags = get_airflow_dags()
    if dags:
        dags_df = pd.DataFrame(dags)
        st.dataframe(dags_df, use_container_width=True)
        
        # Show DAG details
        dag_ids = [dag["dag_id"] for dag in dags]
        selected_dag = st.selectbox("Select DAG for Details", dag_ids)
        
        if selected_dag:
            dag_data = next((dag for dag in dags if dag["dag_id"] == selected_dag), None)
            if dag_data:
                st.markdown("## DAG Details")
                st.json(dag_data)
                
                # Trigger DAG run button
                if st.button("Trigger DAG Run"):
                    try:
                        response = requests.post(
                            f"{AIRFLOW_API_URL}/dags/{selected_dag}/dagRuns",
                            auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                            json={"conf": {}}
                        )
                        if response.status_code == 200:
                            st.success("DAG triggered successfully!")
                        else:
                            st.error(f"Failed to trigger DAG: {response.status_code}")
                    except Exception as e:
                        st.error(f"Error triggering DAG: {e}")
    else:
        st.info("No Airflow DAGs found.")

# ClearML Tasks Page
elif page == "ClearML Tasks":
    st.title("ClearML Tasks")
    
    if not CLEARML_TOKEN:
        st.warning("ClearML token not configured. Please set CLEARML_TOKEN environment variable.")
    else:
        tasks = get_clearml_tasks()
        if tasks:
            st.json(tasks)
        else:
            st.info("No ClearML tasks found.")

# Model Comparison Page
elif page == "Model Comparison":
    st.title("Model Comparison")
    
    experiments = get_mlflow_experiments()
    if experiments:
        experiment_options = {exp.name: exp.experiment_id for exp in experiments}
        selected_experiment = st.selectbox(
            "Select Experiment", 
            options=list(experiment_options.keys())
        )
        
        experiment_id = experiment_options[selected_experiment]
        runs = get_mlflow_runs(experiment_id)
        
        if not runs.empty:
            # Filter runs by tags or date
            st.markdown("## Filter Runs")
            
            # Date filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=30)
                )
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())
            
            # Convert dates to timestamp for filtering
            start_ts = pd.Timestamp(start_date).timestamp() * 1000
            end_ts = pd.Timestamp(end_date).timestamp() * 1000
            
            # Filter runs by date
            filtered_runs = runs[
                (runs["start_time"] >= start_ts) & 
                (runs["end_time"] <= end_ts)
            ]
            
            if not filtered_runs.empty:
                # Display metrics table
                st.markdown("## Performance Metrics")
                
                metrics_df = filtered_runs[[
                    "run_id", 
                    "metrics.rmse", 
                    "metrics.mse", 
                    "metrics.mae", 
                    "metrics.r2"
                ]].copy()
                
                metrics_df.columns = [
                    "Run ID", 
                    "RMSE", 
                    "MSE", 
                    "MAE", 
                    "R²"
                ]
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Display comparison chart
                st.markdown("## Metrics Comparison")
                
                # Get runs for the chart
                run_ids = metrics_df["Run ID"].tolist()
                selected_runs = st.multiselect(
                    "Select Runs to Compare",
                    options=run_ids,
                    default=run_ids[:min(3, len(run_ids))]
                )
                
                if selected_runs:
                    # Filter to selected runs
                    chart_data = metrics_df[metrics_df["Run ID"].isin(selected_runs)]
                    
                    # Prepare data for the chart
                    chart_data = chart_data.set_index("Run ID")
                    
                    # Create the chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot each metric as a group of bars
                    metrics = ["RMSE", "MSE", "MAE", "R²"]
                    x = np.arange(len(selected_runs))
                    width = 0.2
                    
                    for i, metric in enumerate(metrics):
                        values = chart_data[metric].values
                        ax.bar(x + i*width, values, width, label=metric)
                    
                    # Set chart properties
                    ax.set_xlabel("Runs")
                    ax.set_ylabel("Value")
                    ax.set_title("Metrics Comparison")
                    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
                    ax.set_xticklabels(selected_runs)
                    ax.legend()
                    
                    st.pyplot(fig)
            else:
                st.info("No runs found for the selected date range.")
        else:
            st.info("No runs found for this experiment.")
    else:
        st.info("No MLflow experiments found.") 