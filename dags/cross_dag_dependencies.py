#!/usr/bin/env python3
"""
cross_dag_dependencies.py - Utility module for cross-DAG dependencies
--------------------------------------------------------------------
This module contains helper functions to create and manage cross-DAG dependencies
in the ML Automation pipeline.
"""

from airflow.sensors.external_task import ExternalTaskSensor
from datetime import timedelta, datetime
import logging

logger = logging.getLogger(__name__)

def wait_for_dag(
    dag, 
    upstream_dag_id, 
    upstream_task_id=None, 
    execution_delta=None,
    execution_date_fn=None,
    timeout=3600,
    mode="reschedule",
    poke_interval=60
):
    """
    Create an ExternalTaskSensor to wait for another DAG to complete.
    
    Args:
        dag: The current DAG instance
        upstream_dag_id: The DAG ID to wait for
        upstream_task_id: Specific task in the DAG to wait for (if None, waits for the entire DAG)
        execution_delta: Time difference with the previous execution date
        execution_date_fn: Function to calculate execution date
        timeout: Maximum time in seconds to wait
        mode: Sensor mode (reschedule or poke)
        poke_interval: How often to check for the task completion
        
    Returns:
        The ExternalTaskSensor operator
    """
    logger.info(f"Setting up dependency on {upstream_dag_id}")
    
    if not execution_date_fn and not execution_delta:
        execution_delta = timedelta(seconds=0)
    
    # Default execution date function just matches the same execution date
    if not execution_date_fn:
        execution_date_fn = lambda dt: dt - execution_delta
    
    task_id = f"wait_for_{upstream_dag_id}"
    if upstream_task_id:
        task_id += f"_{upstream_task_id}"
    
    # Create the sensor
    return ExternalTaskSensor(
        task_id=task_id,
        external_dag_id=upstream_dag_id,
        external_task_id=upstream_task_id,
        execution_date_fn=execution_date_fn,
        timeout=timeout,
        mode=mode,
        poke_interval=poke_interval,
        retries=3,
        retry_delay=timedelta(minutes=5),
        dag=dag
    ) 