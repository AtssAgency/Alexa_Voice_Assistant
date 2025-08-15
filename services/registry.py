"""Service registration and discovery utilities for gRPC services"""

import grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from grpc_health.v1.health_pb2 import HealthCheckResponse
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import threading


class ServiceType(Enum):
    """Enumeration of available service types"""
    LOADER = "loader"
    LOGGER = "logger"
    KWD = "kwd"
    STT = "stt"
    LLM = "llm"
    TTS = "tts"


@dataclass
class ServiceInfo:
    """Information about a registered service"""
    name: str
    service_type: ServiceType
    host: str
    port: int
    healthy: bool = False
    last_health_check: Optional[float] = None
    
    @property
    def address(self) -> str:
        """Get the full address of the service"""
        return f"{self.host}:{self.port}"


class ServiceRegistry:
    """Registry for managing service discovery and health monitoring"""
    
    # Default service ports as defined in the design
    DEFAULT_PORTS = {
        ServiceType.LOGGER: 5001,
        ServiceType.LOADER: 5002,
        ServiceType.KWD: 5003,
        ServiceType.STT: 5004,
        ServiceType.LLM: 5005,
        ServiceType.TTS: 5006,
    }
    
    def __init__(self, host: str = "localhost"):
        self.host = host
        self.services: Dict[str, ServiceInfo] = {}
        self.logger = logging.getLogger("services.registry")
        self._lock = threading.Lock()
    
    def register_service(self, 
                        name: str, 
                        service_type: ServiceType, 
                        port: Optional[int] = None) -> ServiceInfo:
        """Register a service in the registry"""
        if port is None:
            port = self.DEFAULT_PORTS.get(service_type)
            if port is None:
                raise ValueError(f"No default port defined for service type {service_type}")
        
        with self._lock:
            service_info = ServiceInfo(
                name=name,
                service_type=service_type,
                host=self.host,
                port=port
            )
            self.services[name] = service_info
            self.logger.info(f"Registered service {name} ({service_type.value}) at {service_info.address}")
            return service_info
    
    def unregister_service(self, name: str) -> bool:
        """Unregister a service from the registry"""
        with self._lock:
            if name in self.services:
                del self.services[name]
                self.logger.info(f"Unregistered service {name}")
                return True
            return False
    
    def get_service(self, name: str) -> Optional[ServiceInfo]:
        """Get service information by name"""
        with self._lock:
            return self.services.get(name)
    
    def get_services_by_type(self, service_type: ServiceType) -> List[ServiceInfo]:
        """Get all services of a specific type"""
        with self._lock:
            return [info for info in self.services.values() 
                   if info.service_type == service_type]
    
    def list_services(self) -> List[ServiceInfo]:
        """Get all registered services"""
        with self._lock:
            return list(self.services.values())
    
    def check_service_health(self, name: str, timeout: float = 1.0) -> bool:
        """Check the health of a specific service"""
        service_info = self.get_service(name)
        if not service_info:
            return False
        
        try:
            channel = grpc.insecure_channel(service_info.address)
            stub = health_pb2_grpc.HealthStub(channel)
            
            request = health_pb2.HealthCheckRequest(service=name)
            response = stub.Check(request, timeout=timeout)
            
            is_healthy = response.status == HealthCheckResponse.SERVING
            
            # Update service info
            with self._lock:
                if name in self.services:
                    self.services[name].healthy = is_healthy
                    self.services[name].last_health_check = time.time()
            
            channel.close()
            return is_healthy
            
        except Exception as e:
            self.logger.debug(f"Health check failed for {name}: {e}")
            with self._lock:
                if name in self.services:
                    self.services[name].healthy = False
                    self.services[name].last_health_check = time.time()
            return False
    
    def check_all_health(self, timeout: float = 1.0) -> Dict[str, bool]:
        """Check health of all registered services"""
        results = {}
        for name in list(self.services.keys()):
            results[name] = self.check_service_health(name, timeout)
        return results
    
    def get_healthy_services(self) -> List[ServiceInfo]:
        """Get all services that are currently healthy"""
        with self._lock:
            return [info for info in self.services.values() if info.healthy]
    
    def wait_for_service(self, name: str, timeout: float = 30.0, check_interval: float = 1.0) -> bool:
        """Wait for a service to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.check_service_health(name):
                self.logger.info(f"Service {name} is now healthy")
                return True
            
            time.sleep(check_interval)
        
        self.logger.warning(f"Service {name} did not become healthy within {timeout}s")
        return False
    
    def wait_for_services(self, names: List[str], timeout: float = 30.0) -> bool:
        """Wait for multiple services to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_healthy = True
            for name in names:
                if not self.check_service_health(name):
                    all_healthy = False
                    break
            
            if all_healthy:
                self.logger.info(f"All services are healthy: {names}")
                return True
            
            time.sleep(1.0)
        
        self.logger.warning(f"Not all services became healthy within {timeout}s: {names}")
        return False


class ServiceClient:
    """Base client for connecting to gRPC services"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.logger = logging.getLogger("services.client")
        self._channels: Dict[str, grpc.Channel] = {}
    
    def get_channel(self, service_name: str) -> Optional[grpc.Channel]:
        """Get or create a gRPC channel to a service"""
        service_info = self.registry.get_service(service_name)
        if not service_info:
            self.logger.error(f"Service {service_name} not found in registry")
            return None
        
        if service_name not in self._channels:
            self._channels[service_name] = grpc.insecure_channel(service_info.address)
            self.logger.debug(f"Created channel to {service_name} at {service_info.address}")
        
        return self._channels[service_name]
    
    def close_channel(self, service_name: str):
        """Close a gRPC channel"""
        if service_name in self._channels:
            self._channels[service_name].close()
            del self._channels[service_name]
            self.logger.debug(f"Closed channel to {service_name}")
    
    def close_all_channels(self):
        """Close all gRPC channels"""
        for service_name in list(self._channels.keys()):
            self.close_channel(service_name)
    
    def __del__(self):
        """Cleanup channels on destruction"""
        self.close_all_channels()


class ServiceDiscovery:
    """Service discovery utilities with startup ordering"""
    
    # Service startup order as defined in requirements
    STARTUP_ORDER = [
        ServiceType.LOGGER,  # First - needed by all others
        ServiceType.TTS,     # TTS → LLM → STT → KWD
        ServiceType.LLM,
        ServiceType.STT,
        ServiceType.KWD,
        ServiceType.LOADER,  # Last - orchestrates others
    ]
    
    # Service shutdown order (reverse of startup)
    SHUTDOWN_ORDER = list(reversed(STARTUP_ORDER))
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.logger = logging.getLogger("services.discovery")
    
    def get_startup_order(self) -> List[ServiceType]:
        """Get the recommended service startup order"""
        return self.STARTUP_ORDER.copy()
    
    def get_shutdown_order(self) -> List[ServiceType]:
        """Get the recommended service shutdown order"""
        return self.SHUTDOWN_ORDER.copy()
    
    def validate_service_dependencies(self) -> Tuple[bool, List[str]]:
        """Validate that all required services are registered"""
        errors = []
        required_services = set(self.STARTUP_ORDER)
        registered_types = {info.service_type for info in self.registry.list_services()}
        
        missing_services = required_services - registered_types
        if missing_services:
            errors.extend([f"Missing required service: {svc.value}" for svc in missing_services])
        
        return len(errors) == 0, errors
    
    def get_services_in_startup_order(self) -> List[ServiceInfo]:
        """Get registered services in startup order"""
        services_by_type = {}
        for service in self.registry.list_services():
            services_by_type[service.service_type] = service
        
        ordered_services = []
        for service_type in self.STARTUP_ORDER:
            if service_type in services_by_type:
                ordered_services.append(services_by_type[service_type])
        
        return ordered_services
    
    def get_services_in_shutdown_order(self) -> List[ServiceInfo]:
        """Get registered services in shutdown order"""
        services_by_type = {}
        for service in self.registry.list_services():
            services_by_type[service.service_type] = service
        
        ordered_services = []
        for service_type in self.SHUTDOWN_ORDER:
            if service_type in services_by_type:
                ordered_services.append(services_by_type[service_type])
        
        return ordered_services


# Global registry instance
_global_registry = None

def get_global_registry() -> ServiceRegistry:
    """Get the global service registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry

def initialize_registry(host: str = "localhost") -> ServiceRegistry:
    """Initialize the global service registry"""
    global _global_registry
    _global_registry = ServiceRegistry(host)
    return _global_registry