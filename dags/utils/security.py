import re
from typing import Dict, Any, Callable, Optional
from functools import wraps
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import Config

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers and handling authentication."""
    
    def __init__(self, app):
        super().__init__(app)
        self.security = HTTPBearer()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and add security headers."""
        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response

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

class InputValidator:
    """Input validation utilities."""
    
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

def require_auth(func: Callable) -> Callable:
    """Decorator for requiring JWT authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request')
        if not request:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
        
        if not request:
            raise HTTPException(status_code=401, detail="No request object found")
        
        try:
            auth = await HTTPBearer()(request)
            token = auth.credentials
            payload = jwt.decode(token, Config.SECURITY_CONFIG["jwt_secret"], algorithms=["HS256"])
            request.state.user = payload
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))
        
        return await func(*args, **kwargs)
    return wrapper

def validate_input(schema: Dict[str, Any]) -> Callable:
    """Decorator for input validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                raise HTTPException(status_code=400, detail="No request object found")
            
            try:
                data = await request.json()
                for field, rules in schema.items():
                    if field not in data:
                        if rules.get('required', False):
                            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                        continue
                    
                    value = data[field]
                    if 'type' in rules and not isinstance(value, rules['type']):
                        raise HTTPException(status_code=400, detail=f"Invalid type for field {field}")
                    
                    if 'pattern' in rules and not re.match(rules['pattern'], str(value)):
                        raise HTTPException(status_code=400, detail=f"Invalid format for field {field}")
                    
                    if 'min' in rules and value < rules['min']:
                        raise HTTPException(status_code=400, detail=f"Value too small for field {field}")
                    
                    if 'max' in rules and value > rules['max']:
                        raise HTTPException(status_code=400, detail=f"Value too large for field {field}")
                
                request.state.validated_data = data
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage:
# @require_auth
# @validate_input({
#     "model_id": {"type": str, "required": True, "pattern": r"^model[1-5]$"},
#     "metric": {"type": str, "required": True},
#     "threshold": {"type": float, "required": True, "min": 0, "max": 1}
# })
# async def update_model_metrics(request: Request):
#     data = request.state.validated_data
#     # Process the validated data
#     pass 