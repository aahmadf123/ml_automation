#!/usr/bin/env python3
"""
dags/tasks/schema_validation.py

Validate the post‑preprocessed DataFrame against a Pandera schema,
then snapshot the column dtypes to S3 for versioning.
"""

import json
import time
import logging
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from utils.config import DATA_BUCKET     # from the new config.py
from utils.storage import upload       # your retrying S3 helper

# Setup logging
logger = logging.getLogger(__name__)

# Define the expected schema for your processed data
homeowner_schema = DataFrameSchema({
    "eey":           Column(float, Check.ge(0)),
    "il_total":      Column(float, Check.ge(0)),
    "pure_premium":  Column(float, Check.ge(0)),
    # … add all other columns you expect here …
})

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raises a SchemaError if the DataFrame doesn't match.
    Returns the validated DataFrame otherwise.
    """
    try:
        return homeowner_schema.validate(df)
    except pa.errors.SchemaError as e:
        # Log the error and add more context
        logger.error(f"Schema validation failed: {str(e)}")
        # Import slack if needed
        from utils.slack import post as send_message
        send_message(
            channel="#alerts",
            title="❌ Schema Validation Failed",
            details=f"Schema validation error: {str(e)}",
            urgency="high"
        )
        # Re-raise to ensure the task fails
        raise

def snapshot_schema(df: pd.DataFrame) -> None:
    """
    Take a snapshot of the DataFrame's dtypes, write to /tmp,
    and upload to S3 under 'schema_snapshots/' with a timestamp.
    """
    try:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        schema_map = df.dtypes.apply(lambda dt: dt.name).to_dict()
        local_path = f"/tmp/schema_{ts}.json"

        with open(local_path, "w") as f:
            json.dump(schema_map, f)

        s3_key = f"schema_snapshots/schema_{ts}.json"
        upload(local_path, s3_key)
        logger.info(f"Schema snapshot saved to S3: {s3_key}")
        
        return {"status": "success", "timestamp": ts}
    except Exception as e:
        logger.error(f"Failed to snapshot schema: {str(e)}")
        return {"status": "error", "message": str(e)}

def update_schema_validation_with_ui_components():
    """
    Placeholder function to update the schema validation process with new UI components and endpoints.
    """
    logger.info("Updating schema validation process with new UI components and endpoints.")
    # Placeholder for actual implementation
    return {"status": "success", "message": "Schema validation process updated with UI components"}
