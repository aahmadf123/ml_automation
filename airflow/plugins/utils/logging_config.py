import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
from pythonjsonlogger import jsonlogger
from structlog import get_logger as structlog_get_logger, configure, processors, stdlib

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up structured logging with both JSON and console output.
    
    Args:
        log_level: The logging level to use (default: "INFO")
    """
    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create JSON handler
    json_handler = logging.StreamHandler(sys.stdout)
    json_handler.setFormatter(CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(module)s %(function)s %(line)s %(message)s'
    ))
    root_logger.addHandler(json_handler)
    
    # Create console handler with color
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(console_handler)
    
    # Configure structlog
    configure(
        processors=[
            processors.TimeStamper(fmt="iso"),
            processors.StackInfoRenderer(),
            processors.format_exc_info,
            processors.UnicodeDecoder(),
            processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=stdlib.LoggerFactory(),
        wrapper_class=stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> Any:
    """
    Get a structured logger instance.
    
    Args:
        name: The name of the logger
        
    Returns:
        A structured logger instance
    """
    return structlog_get_logger(name)

# Example usage:
# logger = get_logger(__name__)
# logger.info("message", extra={"key": "value"}) 