# Unified ML Pipeline

## Overview

The `unified_ml_pipeline.py` file consolidates three previously separate pipelines:
1. `integrated_ml_workflow.py`
2. `homeowner_dag.py`
3. `train_all_models_dag.py`

This consolidation eliminates cross-pipeline dependencies that were causing cascading errors and simplifies testing and maintenance.

## Pipeline Structure

The unified pipeline follows a logical ML workflow with these primary stages:

1. **Data Ingestion** - Downloads data from the S3 bucket (`grange-seniordesign-bucket`)
2. **Data Processing** - Applies preprocessing and feature engineering 
3. **Data Validation** - Performs schema validation and data quality checks in parallel
4. **Model Training** - Trains multiple models (can be run in parallel)
5. **Model Explainability** - Generates explanations for the trained models
6. **Artifact Storage** - Archives model artifacts to S3
7. **Cleanup** - Removes temporary files

## Key Benefits

- **Single dependency chain**: Eliminates cross-DAG dependencies that were causing cascading errors
- **Simplified testing**: The entire pipeline can be tested end-to-end
- **Improved error handling**: Consistent error handling across all tasks
- **Robust data management**: Multiple fallbacks for data access and storage
- **Resource optimization**: Parallel task execution where appropriate

## Configuration

The pipeline uses these Airflow variables:
- `DATA_BUCKET` - S3 bucket containing the data (default: "grange-seniordesign-bucket")
- `PARALLEL_TRAINING` - Whether to train models in parallel (default: "True")
- `MAX_PARALLEL_WORKERS` - Maximum number of parallel training workers (default: 3)

## Running the Pipeline

To run the unified pipeline:

```bash
airflow dags trigger unified_ml_pipeline
```

To disable catchup runs:

```bash
airflow dags backfill -s YYYY-MM-DD -e YYYY-MM-DD --reset_dagruns unified_ml_pipeline
```

## Temporary File Management

The pipeline stores intermediate files in these locations (in order of preference):
1. `/tmp/airflow_data/`
2. `/usr/local/airflow/tmp/`
3. `/tmp/`

## S3 Integration

The pipeline interacts with the S3 bucket (`grange-seniordesign-bucket`) for:
- Data ingestion - Reading raw data
- Model artifacts - Storing trained models and metadata

## Migration Guide

### For Previous Users of Multiple Pipelines

If you were previously using the separate pipelines:

1. Disable the previous DAGs:
   ```bash
   airflow dags pause integrated_ml_workflow
   airflow dags pause homeowner_loss_history_full_pipeline
   airflow dags pause train_all_models
   ```

2. Enable the new unified pipeline:
   ```bash
   airflow dags unpause unified_ml_pipeline
   ```

3. Update any scripts or external systems that were triggering the old pipelines to use the new unified pipeline.

### Notes

- The unified pipeline maintains the same data processing logic and model training as the previous pipelines
- No changes needed for S3 bucket configuration
- Error notifications (Slack) remain consistent with previous implementations 