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
4. **Human Validation** - Requests human review of data quality and drift detection results
5. **Model Training** - Trains multiple models (can be run in parallel)
6. **Model Explainability** - Generates explanations for the trained models
7. **Human Approval** - Requests human approval of trained models before deployment
8. **Artifact Storage** - Archives model artifacts to S3
9. **Cleanup** - Removes temporary files

## Key Benefits

- **Single dependency chain**: Eliminates cross-DAG dependencies that were causing cascading errors
- **Simplified testing**: The entire pipeline can be tested end-to-end
- **Improved error handling**: Consistent error handling across all tasks
- **Robust data management**: Multiple fallbacks for data access and storage
- **Resource optimization**: Parallel task execution where appropriate
- **Reduced redundancy**: Eliminated overlapping functionality between modules
- **Human Oversight**: Critical ML decisions now require human validation and approval

## Task Modules Used

The pipeline uses these key task modules:
- `tasks.ingestion` - Handles data retrieval from S3
- `tasks.preprocessing_simplified` - Performs data cleaning and feature engineering 
- `tasks.data_quality` - Runs quality checks on processed data
- `tasks.schema_validation` - Validates data schema
- `tasks.drift` - Detects data drift
- `tasks.training` - Trains and evaluates models
- `tasks.model_explainability` - Generates model explanations
- `tasks.hitl` - Provides Human-in-the-Loop functionality

## Human-in-the-Loop (HITL) Capabilities

The pipeline now includes critical human oversight at two key checkpoints:

1. **Data Validation Checkpoint**
   - Triggered after data quality checks, schema validation, and drift detection
   - Presents data quality metrics and issues to human reviewers
   - Enables manual override of data quality decisions
   - Can be configured to auto-approve if no issues are detected

2. **Model Approval Checkpoint**
   - Triggered after model training and explainability analysis
   - Presents model performance metrics to human reviewers
   - Enables manual approval or rejection of models before deployment
   - Can be configured to auto-approve with clean metrics

### HITL Configuration

The HITL functionality can be configured with these Airflow variables:
- `REQUIRE_DATA_VALIDATION` - Whether to require human validation of data (default: "True")
- `REQUIRE_MODEL_APPROVAL` - Whether to require human approval of models (default: "True")
- `AUTO_APPROVE_DATA` - Whether to automatically approve data without human input (default: "False")
- `AUTO_APPROVE_MODEL` - Whether to automatically approve models without human input (default: "False")
- `AUTO_APPROVE_QUALITY_THRESHOLD` - Maximum number of quality issues for auto-approval (default: 3)
- `AUTO_APPROVE_TIMEOUT_MINUTES` - Minutes to wait for data validation before timing out (default: 30)
- `MODEL_APPROVE_TIMEOUT_MINUTES` - Minutes to wait for model approval before timing out (default: 60)

The pipeline now supports three auto-approval mechanisms:
1. **Explicit Auto-Approval**: Set `AUTO_APPROVE_DATA` or `AUTO_APPROVE_MODEL` to "True" to bypass human validation
2. **Quality-Based Auto-Approval**: Data is auto-approved if quality issues are below threshold
3. **Timeout Auto-Approval**: After the specified timeout, the pipeline continues with auto-approval

This ensures the pipeline can run successfully in both interactive and automated environments, and prevents tasks from getting stuck waiting for human input indefinitely.

### Slack Integration

The HITL module sends notifications to Slack to alert humans when their input is required:
- Data validation requests go to the `#ml-approvals` channel
- Model approval requests go to the `#ml-approvals` channel
- Override confirmations go to relevant channels

## Data Target and Features

The pipeline uses the following approach for target variable calculation:
- Uses `trgt` as the primary target column (calculated as `il_total / eey`)
- Creates the target column automatically if not present in the dataset
- Also creates a weight column `wt` from the `eey` column for training

This approach ensures compatibility with the model training process while allowing flexibility in input data formats.

## Configuration

The pipeline uses these Airflow variables:
- `DATA_BUCKET` - S3 bucket containing the data (default: "grange-seniordesign-bucket")
- `PARALLEL_TRAINING` - Whether to train models in parallel (default: "True")
- `MAX_PARALLEL_WORKERS` - Maximum number of parallel training workers (default: 3)
- `APPLY_FEATURE_ENGINEERING` - Whether to apply feature engineering in preprocessing (default: "False")
- `DASHBOARD_URL` - URL to the dashboard UI for approvals (used in HITL notifications)

### For Clean Datasets

If your dataset is already clean, set `APPLY_FEATURE_ENGINEERING` to "False" (default). This will:
- Skip intensive feature engineering steps
- Perform only minimal processing if input format is already parquet
- Preserve the original data structure as much as possible

For datasets requiring preprocessing, set `APPLY_FEATURE_ENGINEERING` to "True".

## Data Processing Optimizations

The pipeline uses a simplified preprocessing module (`preprocessing_simplified.py`) that:

1. Skips outlier detection and handling since the dataset is already clean
2. Still handles all other preprocessing steps:
   - Missing value handling
   - Skewed feature transformation (if needed)
   - Categorical encoding
   - Model-specific feature selection
   - Target column creation (pure_premium/trgt)

This optimization improves performance and reduces processing time while ensuring all necessary data preparation is still performed.

## Robust Task Execution

The pipeline includes several features to ensure robustness in task execution:

1. **Never-Skip Task Execution** - Tasks are configured with `trigger_rule='all_done'` to ensure they execute even if upstream tasks fail or are skipped. This prevents pipeline failures due to non-critical issues in previous steps.

2. **Graceful HITL Error Handling** - The Human-in-the-Loop validation and approval steps have been enhanced to:
   - Catch and log all exceptions without stopping the pipeline
   - Provide detailed diagnostics when issues occur
   - Continue execution with reasonable defaults when human input is unavailable

3. **Fallback Data Source Selection** - Each task that needs data will:
   - First try to use data paths from XCom
   - Fall back to standardized locations if XCom values are missing
   - Search common directories for recent parquet files as a last resort
   - Provide clear logging about which data source was selected

4. **Comprehensive Logging** - Enhanced logging includes:
   - Detailed diagnostic information about data shapes and contents
   - XCom values received from upstream tasks
   - Configuration settings used for execution
   - Model statistics and performance metrics

These features ensure that the pipeline continues running even when individual components encounter errors, making it suitable for both attended and unattended execution.

## Running the Pipeline

To run the unified pipeline:

```bash
airflow dags trigger unified_ml_pipeline
```

To disable catchup runs:

```bash
airflow dags backfill -s YYYY-MM-DD -e YYYY-MM-DD --reset_dagruns unified_ml_pipeline
```

### Disabling HITL for Automated Runs

For fully automated runs without human intervention:

```bash
airflow variables set REQUIRE_DATA_VALIDATION False
airflow variables set REQUIRE_MODEL_APPROVAL False
airflow dags trigger unified_ml_pipeline
```

With this approach, the pipeline will automatically proceed through all steps without waiting for human input, while still performing all validation checks and logging any issues that arise.

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