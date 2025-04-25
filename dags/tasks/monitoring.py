#!/usr/bin/env python3
"""
tasks/monitoring.py

Monitoring utilities with Prometheus instrumentation:
  - Records DAG runtime
  - Records memory metrics
  - Exposes /metrics on port 8000
  - Provides WebSocket server for real-time dashboard updates
  - Updates the monitoring process to include the new UI components and endpoints
"""

import logging
import time
import json
import asyncio
import websockets
import psutil
import boto3
from prometheus_client import start_http_server, Gauge, REGISTRY
from typing import Dict, Set, Any
from tenacity import retry, stop_after_attempt, wait_fixed
from utils.config import AWS_REGION
from clearml import Task, Logger

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Initialize AWS clients with region
cloudwatch = boto3.client('cloudwatch', region_name=AWS_REGION)

# Singleton flag to ensure metrics are only registered once
_metrics_initialized = False
dag_runtime_gauge = None
memory_available_gauge = None
memory_used_gauge = None
memory_total_gauge = None
memory_percent_gauge = None

def initialize_metrics():
    """Initialize Prometheus metrics if not already done"""
    global _metrics_initialized, dag_runtime_gauge, memory_available_gauge, memory_used_gauge, memory_total_gauge, memory_percent_gauge
    
    if _metrics_initialized:
        return
    
    # Check if metrics already exist in registry and clear them if needed
    collectors_to_remove = []
    for collector in REGISTRY._collector_to_names:
        for name in REGISTRY._collector_to_names[collector]:
            if name in [
                "homeowner_dag_runtime_seconds",
                "homeowner_memory_available_mb",
                "homeowner_memory_used_mb",
                "homeowner_memory_total_mb",
                "homeowner_memory_usage_percent"
            ]:
                collectors_to_remove.append(collector)
                break
    
    for collector in collectors_to_remove:
        REGISTRY.unregister(collector)
    
    # Define Prometheus gauges
    dag_runtime_gauge = Gauge(
        "homeowner_dag_runtime_seconds",
        "Duration of the DAG run in seconds"
    )
    memory_available_gauge = Gauge(
        "homeowner_memory_available_mb",
        "Available system memory in megabytes"
    )
    memory_used_gauge = Gauge(
        "homeowner_memory_used_mb",
        "Used system memory in megabytes"
    )
    memory_total_gauge = Gauge(
        "homeowner_memory_total_mb",
        "Total system memory in megabytes"
    )
    memory_percent_gauge = Gauge(
        "homeowner_memory_usage_percent",
        "Percentage of system memory in use"
    )
    
    _metrics_initialized = True
    logger.info("Prometheus metrics initialized successfully")

# Attempt to start Prometheus metrics server with retry
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def start_metrics_server(port: int = 8000):
    try:
        # Initialize metrics before starting server
        initialize_metrics()
        start_http_server(port)
        logging.info(f"Prometheus metrics server started on port {port}")
    except OSError as e:
        logging.warning(f"Failed to start Prometheus metrics server on port {port}: {e}")
        raise

try:
    start_metrics_server()
except Exception as e:
    logging.error(f"Could not start Prometheus metrics server after retries: {e}")

# WebSocket server configuration
WEBSOCKET_PORT = 8765
connected_clients: Set[websockets.WebSocketServerProtocol] = set()
metrics_history: Dict[str, Dict[str, Any]] = {}

async def register(websocket: websockets.WebSocketServerProtocol):
    """Register a new WebSocket client."""
    connected_clients.add(websocket)
    logging.info(f"WebSocket client connected. Total clients: {len(connected_clients)}")
    
    # Send current metrics history to the new client
    if metrics_history:
        await websocket.send(json.dumps({
            "type": "metrics_history",
            "data": metrics_history
        }))

async def unregister(websocket: websockets.WebSocketServerProtocol):
    """Unregister a WebSocket client."""
    connected_clients.remove(websocket)
    logging.info(f"WebSocket client disconnected. Total clients: {len(connected_clients)}")

async def broadcast(message: str):
    """Broadcast a message to all connected WebSocket clients."""
    if connected_clients:
        await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True
        )

async def websocket_handler(websocket: websockets.WebSocketServerProtocol, path: str):
    """Handle WebSocket connections."""
    await register(websocket)
    try:
        async for message in websocket:
            # Handle incoming messages if needed
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                logging.warning(f"Received invalid JSON: {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await unregister(websocket)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def start_websocket_server():
    """Start the WebSocket server with retry logic."""
    try:
        server = await websockets.serve(websocket_handler, "0.0.0.0", WEBSOCKET_PORT)
        logging.info(f"WebSocket server started on port {WEBSOCKET_PORT}")
        return server
    except Exception as e:
        logging.error(f"Failed to start WebSocket server: {e}")
        raise

def record_system_metrics(runtime: float = None, memory_usage: str = None) -> None:
    """
    Record and export system metrics to Prometheus and log them.

    Args:
        runtime (float, optional): The runtime duration in seconds.
        memory_usage (str, optional): Ignored (string snapshots aren't Prometheus-friendly).
    """
    # Initialize metrics if needed
    initialize_metrics()
    
    # Record DAG runtime
    if runtime is not None:
        dag_runtime_gauge.set(runtime)
        logging.info(f"[Monitoring] DAG runtime set to {runtime:.2f}s")
    else:
        now = time.time()
        dag_runtime_gauge.set(now)
        logging.info(f"[Monitoring] DAG runtime timestamp set to {now:.2f}")

    # Record memory stats
    vm = psutil.virtual_memory()
    avail_mb = vm.available / (1024 * 1024)
    used_mb  = vm.used      / (1024 * 1024)
    total_mb = vm.total     / (1024 * 1024)

    memory_available_gauge.set(avail_mb)
    memory_used_gauge.set(used_mb)
    memory_total_gauge.set(total_mb)
    memory_percent_gauge.set(vm.percent)

    logging.info(
        "[Monitoring] Memory — "
        f"Avail: {avail_mb:.2f}MB, "
        f"Used: {used_mb:.2f}MB, "
        f"Total: {total_mb:.2f}MB, "
        f"Usage: {vm.percent:.1f}%"
    )
    
    # Broadcast system metrics to WebSocket clients
    system_metrics = {
        "type": "system_metrics",
        "data": {
            "runtime": runtime if runtime is not None else now,
            "memory": {
                "available_mb": avail_mb,
                "used_mb": used_mb,
                "total_mb": total_mb,
                "percent": vm.percent
            }
        }
    }
    asyncio.run(broadcast(json.dumps(system_metrics)))
    
    # Log metrics to ClearML
    try:
        task = Task.current_task()
        if task:
            logger = task.get_logger()
            logger.report_scalar("runtime", "DAG", value=runtime if runtime is not None else now)
            logger.report_scalar("memory_available_mb", "System", value=avail_mb)
            logger.report_scalar("memory_used_mb", "System", value=used_mb)
            logger.report_scalar("memory_total_mb", "System", value=total_mb)
            logger.report_scalar("memory_usage_percent", "System", value=vm.percent)
            logging.info("Logged system metrics to ClearML")
    except Exception as e:
        logging.error(f"Failed to log metrics to ClearML: {str(e)}")

def update_metrics_history(model_id: str, metrics: Dict[str, Any]):
    """
    Update the metrics history for a model and broadcast to WebSocket clients.
    
    Args:
        model_id (str): The ID of the model.
        metrics (Dict[str, Any]): The metrics data.
    """
    if model_id not in metrics_history:
        metrics_history[model_id] = {}
    
    metrics_history[model_id].update(metrics)
    
    # Broadcast the updated metrics
    message = {
        "type": "metrics_update",
        "model_id": model_id,
        "metrics": metrics
    }
    asyncio.run(broadcast(json.dumps(message)))
    
    # Log metrics to ClearML
    try:
        task = Task.current_task()
        if task:
            logger = task.get_logger()
            for metric_name, metric_value in metrics.items():
                logger.report_scalar(metric_name, model_id, value=metric_value)
            logging.info(f"Logged metrics for model {model_id} to ClearML")
    except Exception as e:
        logging.error(f"Failed to log metrics to ClearML: {str(e)}")

def update_monitoring_with_ui_components():
    """
    Update the monitoring process with new UI components and endpoints.
    """
    # Make sure metrics are initialized
    initialize_metrics()
    
    logging.info("Updating monitoring process with new UI components and endpoints.")
    
    # Start WebSocket server in a separate thread
    import threading
    def run_websocket_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server = loop.run_until_complete(start_websocket_server())
        loop.run_forever()
    
    websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
    websocket_thread.start()
    
    return {"status": "success", "message": "Monitoring UI components updated"}
