"""Base gRPC service classes and health check implementation"""

import grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from grpc_health.v1.health_pb2 import HealthCheckResponse
import logging
from abc import ABC, abstractmethod
from concurrent import futures
import threading
import time


class BaseService(ABC):
    """Base class for all gRPC services with health check integration"""
    
    def __init__(self, service_name: str, port: int):
        self.service_name = service_name
        self.port = port
        self.server = None
        self.health_servicer = None
        self.logger = logging.getLogger(f"services.{service_name}")
        self._shutdown_event = threading.Event()
        
    @abstractmethod
    def register_service(self, server):
        """Register the specific service with the gRPC server"""
        pass
    
    @abstractmethod
    def configure(self, config):
        """Configure the service with provided configuration"""
        pass
    
    def start(self):
        """Start the gRPC server with health check"""
        try:
            # Create server
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            
            # Register the specific service
            self.register_service(self.server)
            
            # Add health check service
            self.health_servicer = HealthServicer()
            health_pb2_grpc.add_HealthServicer_to_server(self.health_servicer, self.server)
            
            # Add port and start
            listen_addr = f'localhost:{self.port}'
            self.server.add_insecure_port(listen_addr)
            self.server.start()
            
            # Mark service as serving
            self.health_servicer.set_status(
                self.service_name, 
                HealthCheckResponse.SERVING
            )
            
            self.logger.info(f"{self.service_name} service started on {listen_addr}")
            
        except Exception as e:
            self.logger.error(f"Failed to start {self.service_name} service: {e}")
            if self.health_servicer:
                self.health_servicer.set_status(
                    self.service_name, 
                    HealthCheckResponse.NOT_SERVING
                )
            raise
    
    def stop(self, grace_period=5):
        """Stop the gRPC server gracefully"""
        if self.server:
            self.logger.info(f"Stopping {self.service_name} service...")
            
            # Mark as not serving
            if self.health_servicer:
                self.health_servicer.set_status(
                    self.service_name, 
                    HealthCheckResponse.NOT_SERVING
                )
            
            # Stop server
            self.server.stop(grace_period)
            self.logger.info(f"{self.service_name} service stopped")
    
    def wait_for_termination(self):
        """Wait for the server to terminate"""
        if self.server:
            self.server.wait_for_termination()


class HealthServicer(health_pb2_grpc.HealthServicer):
    """gRPC Health Check Protocol implementation"""
    
    def __init__(self):
        self._status_map = {}
        self._lock = threading.Lock()
    
    def Check(self, request, context):
        """Synchronous health check"""
        with self._lock:
            status = self._status_map.get(
                request.service, 
                HealthCheckResponse.SERVICE_UNKNOWN
            )
            return HealthCheckResponse(status=status)
    
    def Watch(self, request, context):
        """Streaming health check (not implemented for simplicity)"""
        # For now, just return current status once
        with self._lock:
            status = self._status_map.get(
                request.service, 
                HealthCheckResponse.SERVICE_UNKNOWN
            )
            yield HealthCheckResponse(status=status)
    
    def set_status(self, service_name: str, status: HealthCheckResponse.ServingStatus):
        """Set the health status for a service"""
        with self._lock:
            self._status_map[service_name] = status


class ServiceManager:
    """Utility class for managing service lifecycle"""
    
    def __init__(self):
        self.services = {}
        self.logger = logging.getLogger("services.manager")
    
    def register_service(self, name: str, service: BaseService):
        """Register a service for management"""
        self.services[name] = service
        self.logger.info(f"Registered service: {name}")
    
    def start_service(self, name: str):
        """Start a specific service"""
        if name in self.services:
            self.services[name].start()
        else:
            raise ValueError(f"Service {name} not registered")
    
    def stop_service(self, name: str, grace_period=5):
        """Stop a specific service"""
        if name in self.services:
            self.services[name].stop(grace_period)
        else:
            raise ValueError(f"Service {name} not registered")
    
    def start_all(self):
        """Start all registered services"""
        for name, service in self.services.items():
            try:
                self.start_service(name)
            except Exception as e:
                self.logger.error(f"Failed to start service {name}: {e}")
                raise
    
    def stop_all(self, grace_period=5):
        """Stop all registered services"""
        for name in reversed(list(self.services.keys())):
            try:
                self.stop_service(name, grace_period)
            except Exception as e:
                self.logger.error(f"Failed to stop service {name}: {e}")
    
    def check_health(self, service_name: str) -> bool:
        """Check if a service is healthy via gRPC health check"""
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        try:
            # Create a channel to the service
            channel = grpc.insecure_channel(f'localhost:{service.port}')
            stub = health_pb2_grpc.HealthStub(channel)
            
            # Check health
            request = health_pb2.HealthCheckRequest(service=service_name)
            response = stub.Check(request, timeout=1.0)
            
            channel.close()
            return response.status == HealthCheckResponse.SERVING
            
        except Exception as e:
            self.logger.warning(f"Health check failed for {service_name}: {e}")
            return False