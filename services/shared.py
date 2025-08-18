#!/usr/bin/env python3
"""
Shared Service Clients - Asynchronous HTTP client management

Provides shared httpx.AsyncClient instance for non-blocking service-to-service communication.
This replaces the blocking requests library to enable full asyncio performance benefits.
"""

import httpx
from contextlib import asynccontextmanager
from typing import Optional

class ServiceClients:
    """A holder for shared service clients."""
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None

# Global instance
service_clients = ServiceClients()

@asynccontextmanager
async def lifespan_with_httpx(service_lifespan=None):
    """
    Combined lifespan manager for FastAPI services with httpx client.
    
    Args:
        service_lifespan: Optional service-specific lifespan function
    """
    # Create the shared httpx client
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),  # Default timeout for all requests
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
    ) as client:
        service_clients.http_client = client
        
        # If there's service-specific lifespan logic, run it
        if service_lifespan:
            async with service_lifespan() as result:
                yield result
        else:
            yield
    
    # Clean up
    service_clients.http_client = None

def get_http_client() -> Optional[httpx.AsyncClient]:
    """Get the shared httpx client instance."""
    return service_clients.http_client

async def safe_post(url: str, json=None, timeout: float = 2.0, fallback_msg: str = None) -> bool:
    """
    Safe async POST request with fallback logging.
    
    Args:
        url: Target URL
        json: JSON payload
        timeout: Request timeout
        fallback_msg: Message to print if request fails
    
    Returns:
        True if request succeeded, False otherwise
    """
    client = get_http_client()
    if not client:
        if fallback_msg:
            print(fallback_msg + " (HTTP client not available)")
        return False
    
    try:
        response = await client.post(url, json=json, timeout=timeout)
        return response.status_code == 200
    except Exception:
        if fallback_msg:
            print(fallback_msg)
        return False

async def safe_get(url: str, timeout: float = 2.0) -> Optional[dict]:
    """
    Safe async GET request.
    
    Args:
        url: Target URL
        timeout: Request timeout
    
    Returns:
        Response JSON dict if successful, None otherwise
    """
    client = get_http_client()
    if not client:
        return None
    
    try:
        response = await client.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

# Alias for backwards compatibility
lifespan_clients = lifespan_with_httpx
