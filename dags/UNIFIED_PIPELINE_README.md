# Unified ML Pipeline

## Overview

The `unified_ml_pipeline.py` file consolidates three previously separate pipelines:
1. `integrated_ml_workflow.py` (removed)
2. `homeowner_dag.py` (removed)
3. `train_all_models_dag.py` (removed)

This consolidation eliminates cross-pipeline dependencies that were causing cascading errors and simplifies testing and maintenance.

## Pipeline Structure

The unified pipeline follows a streamlined ML workflow with these primary stages:

1. **Data Ingestion** - Downloads data from the S3 bucket (`grange-seniordesign-bucket`)
2. **Data Processing** - Applies preprocessing and optional feature engineering 
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
- **Reduced redundancy**: Eliminated overlapping functionality between modules

## Task Modules Used

The pipeline uses these key task modules:
- `tasks.ingestion` - Handles data retrieval from S3
- `tasks.preprocessing` - Performs data cleaning and feature engineering 
- `tasks.data_quality` - Runs quality checks on processed data
- `tasks.schema_validation` - Validates data schema
- `tasks.drift` - Detects data drift
- `tasks.training` - Trains and evaluates models
- `tasks.model_explainability` - Generates model explanations

## Configuration

The pipeline uses these Airflow variables:
- `DATA_BUCKET` - S3 bucket containing the data (default: "grange-seniordesign-bucket")
- `PARALLEL_TRAINING` - Whether to train models in parallel (default: "True")
- `MAX_PARALLEL_WORKERS` - Maximum number of parallel training workers (default: 3)
- `APPLY_FEATURE_ENGINEERING` - Whether to apply feature engineering in preprocessing (default: "False")

### For Clean Datasets

If your dataset is already clean, set `APPLY_FEATURE_ENGINEERING` to "False" (default). This will:
- Skip intensive feature engineering steps
- Perform only minimal processing if input format is already parquet
- Preserve the original data structure as much as possible

For datasets requiring preprocessing, set `APPLY_FEATURE_ENGINEERING` to "True".

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

### Files Removed

The following files have been removed as they are no longer needed:
- `integrated_ml_workflow.py` - Consolidated into unified pipeline
- `homeowner_dag.py` - Consolidated into unified pipeline
- `train_all_models_dag.py` - Consolidated into unified pipeline
- `cross_dag_dependencies.py` - No longer needed as there are no cross-DAG dependencies

### Simplified Components

- Eliminated redundancy between `preprocessing.py` and `data_prep.py`
- Streamlined error handling and data flow
- Removed unused imports and modules 