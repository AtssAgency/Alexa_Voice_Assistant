#!/usr/bin/env python3
"""
Services Utilities - Shared functions to reduce code duplication

Provides:
- post_log(): Standardized logging to Logger service
- create_service_app(): Factory function for FastAPI app creation  
- compose_url(): Backward-compatible URL composition
- Other common utilities used across services

Benefits:
- Reduces boilerplate in service implementations
- Single point of change for common functionality
- Consistent implementation across services
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

from .shared import safe_post, get_http_client


def get_timestamp() -> str:
    """Get timestamp in dd-mm-yy hh:mm:ss format (consistent across all services)"""
    return datetime.now().strftime("%d-%m-%y %H:%M:%S")


def format_console_message(svc: str, level: str, message: str) -> str:
    """Format message for console output using the standard template"""
    return f"{svc:<10}{level.upper():<6}= {message}"


def compose_url(base: str, route: str) -> str:
    """
    Backward-compatible URL joiner:
    - If route is absolute (starts with http), return route.
    - If base already ends with the same route, return base.
    - If base already includes a path (e.g., .../speak) and route == '/speak', don't duplicate.
    - Else join cleanly.
    """
    if not base:
        return route or ""
    if route and route.startswith("http"):
        return route

    base_clean = base.rstrip("/")
    route_clean = (route or "").lstrip("/")

    # If base already contains a path and ends with route, don't append
    if route_clean and base_clean.split("://")[-1].rstrip("/").endswith(route_clean):
        return base_clean

    # If route empty, use base as-is
    if not route_clean:
        return base_clean

    return f"{base_clean}/{route_clean}"


async def post_log(
    service: str, 
    level: str, 
    message: str, 
    logger_url: str,
    event: Optional[str] = None,
    dialog_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout: float = 2.0
) -> bool:
    """
    Post log message to Logger service
    
    Args:
        service: Service name (e.g., "KWD", "STT", etc.)
        level: Log level ("info", "debug", "error", etc.)
        message: Log message
        logger_url: Logger service base URL
        event: Optional event name for filtering
        dialog_id: Optional dialog ID for correlation
        extra: Optional additional data
        timeout: Request timeout
    
    Returns:
        True if log was sent successfully, False otherwise
    """
    log_url = compose_url(logger_url, "/log")
    
    payload = {
        "svc": service.upper(),
        "level": level.lower(),
        "message": message,
        "ts_ms": int(time.time() * 1000)
    }
    
    if event:
        payload["event"] = event
    if dialog_id:
        payload["dialog_id"] = dialog_id
    if extra:
        payload["extra"] = extra
    
    fallback_msg = format_console_message(service, level, message)
    
    return await safe_post(
        url=log_url,
        json=payload,
        timeout=timeout,
        fallback_msg=fallback_msg
    )


def create_service_app(
    title: str,
    description: str,
    version: str,
    lifespan: Optional[asynccontextmanager] = None
) -> FastAPI:
    """
    Factory function to create FastAPI app with standard configuration
    
    Args:
        title: API title
        description: API description  
        version: API version
        lifespan: Optional lifespan context manager
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan
    )
    
    return app


def run_service(
    app: FastAPI,
    host: str = "127.0.0.1", 
    port: int = 5000,
    log_level: str = "error",
    access_log: bool = False
):
    """
    Standard uvicorn runner for services
    
    Args:
        app: FastAPI application
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to
        log_level: uvicorn log level
        access_log: Enable access logging
    """
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=access_log,
    )


class ServiceLogger:
    """
    Utility class for services that need logging functionality
    
    Usage:
        logger = ServiceLogger("KWD", "http://127.0.0.1:5000")
        await logger.info("Service started")
        await logger.debug("Debug message", extra={"key": "value"})
    """
    
    def __init__(self, service_name: str, logger_url: str):
        self.service_name = service_name.upper()
        self.logger_url = logger_url
    
    async def log(self, level: str, message: str, **kwargs) -> bool:
        """Send log message to Logger service"""
        return await post_log(
            service=self.service_name,
            level=level,
            message=message,
            logger_url=self.logger_url,
            **kwargs
        )
    
    async def debug(self, message: str, **kwargs) -> bool:
        """Send debug log message"""
        return await self.log("debug", message, **kwargs)
    
    async def info(self, message: str, **kwargs) -> bool:
        """Send info log message"""
        return await self.log("info", message, **kwargs)
    
    async def warning(self, message: str, **kwargs) -> bool:
        """Send warning log message"""
        return await self.log("warning", message, **kwargs)
    
    async def error(self, message: str, **kwargs) -> bool:
        """Send error log message"""
        return await self.log("error", message, **kwargs)
