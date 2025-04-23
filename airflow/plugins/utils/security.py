import re
from typing import Dict, Any, Callable, Optional
from functools import wraps
import jwt
from datetime import datetime, timedelta
from plugins.utils.config import Config

class SecurityUtils:
    """Security utilities for Airflow DAGs."""
    
    @staticmethod
    def validate_model_id(model_id: str) -> bool:
        """Validate model ID format."""
        return model_id in Config.MODEL_IDS
    
    @staticmethod
    def validate_metric_name(metric: str) -> bool:
        """Validate metric name."""
        return metric in Config.MODEL_METRICS
    
    @staticmethod
    def validate_timestamp(timestamp: str) -> bool:
        """Validate ISO timestamp format."""
        try:
            datetime.fromisoformat(timestamp)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_numeric_range(value: float, min_val: float, max_val: float) -> bool:
        """Validate numeric value is within range."""
        return min_val <= value <= max_val

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, window: int = 3600, max_requests: int = 1000):
        self.window = window
        self.max_requests = max_requests
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if a client is allowed to make a request."""
        now = datetime.now()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if (now - req_time).total_seconds() < self.window
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add new request
        self.requests[client_id].append(now)
        return True

def validate_input(schema: Dict[str, Any]) -> Callable:
    """Decorator for input validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                data = kwargs.get('data', {})
                for field, rules in schema.items():
                    if field not in data:
                        if rules.get('required', False):
                            raise ValueError(f"Missing required field: {field}")
                        continue
                    
                    value = data[field]
                    if 'type' in rules and not isinstance(value, rules['type']):
                        raise ValueError(f"Invalid type for field {field}")
                    
                    if 'pattern' in rules and not re.match(rules['pattern'], str(value)):
                        raise ValueError(f"Invalid format for field {field}")
                    
                    if 'min' in rules and value < rules['min']:
                        raise ValueError(f"Value too small for field {field}")
                    
                    if 'max' in rules and value > rules['max']:
                        raise ValueError(f"Value too large for field {field}")
                
                kwargs['validated_data'] = data
            except Exception as e:
                raise ValueError(str(e))
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage:
# @validate_input({
#     "model_id": {"type": str, "required": True, "pattern": r"^model[1-5]$"},
#     "metric": {"type": str, "required": True},
#     "threshold": {"type": float, "required": True, "min": 0, "max": 1}
# })
# def update_model_metrics(**kwargs):
#     data = kwargs.get('validated_data', {})
#     # Process the validated data
#     pass 