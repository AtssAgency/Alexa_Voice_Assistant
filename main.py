"""AI Voice Assistant - Main Orchestrator"""

import logging
import signal
import sys
import os
import time
import grpc
import subprocess
from typing import Optional, Dict, Any
from services.config import ConfigManager, ConfigValidationError, VRAMValidationError
from services.utils import check_vram_availability, setup_logging, wait_for_port
from services.registry import ServiceRegistry, ServiceType, ServiceDiscovery
from services.proto import loader_pb2
from services.proto import loader_pb2_grpc


class MainOrchestrator:
    """Main orchestrator for the AI Voice Assistant system"""
    
    def __init__(self):
        self.config_manager = None
        self.config = None
        self.logger = None
        self.registry = None
        self.discovery = None
        self.loader_process = None
        self.loader_channel = None
        self.loader_stub = None
        self._shutdown_requested = False
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the system with configuration loading and validation"""
        # Setup basic logging first
        self.logger = setup_logging("INFO", "main")
        self.logger.info("AI Voice Assistant starting...")
        
        # Load and validate configuration
        if not self._load_configuration():
            return False
        
        # Validate system requirements
        if not self._validate_system_requirements():
            return False
        
        # Initialize service registry
        self._initialize_service_registry()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self._initialized = True
        self.logger.info("System initialization complete")
        return True
    
    def _load_configuration(self) -> bool:
        """Load and validate configuration from file"""
        try:
            self.config_manager = ConfigManager()
            self.config = self.config_manager.load()
            self.config_manager.validate()
            
            # Update logging level from config
            logging.getLogger().setLevel(
                getattr(logging, self.config['main'].log_level.upper())
            )
            
            self.logger.info("Configuration loaded and validated successfully")
            return True
            
        except (FileNotFoundError, ConfigValidationError, VRAMValidationError) as e:
            if self.logger:
                self.logger.error(f"Configuration error: {e}")
            else:
                print(f"Configuration error: {e}", file=sys.stderr)
            return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error loading configuration: {e}")
            else:
                print(f"Unexpected error loading configuration: {e}", file=sys.stderr)
            return False
    
    def _validate_system_requirements(self) -> bool:
        """Validate system requirements including VRAM"""
        try:
            min_vram = self.config['system'].min_vram_mb
            if not check_vram_availability(min_vram):
                self.logger.fatal(f"Insufficient VRAM. Required: {min_vram}MB")
                return False
            
            self.logger.info(f"VRAM requirement satisfied: {min_vram}MB minimum")
            return True
            
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            return False
    
    def _initialize_service_registry(self):
        """Initialize service registry and discovery"""
        self.registry = ServiceRegistry()
        self.discovery = ServiceDiscovery(self.registry)
        
        # Register all services with their configured ports
        self.registry.register_service("logger", ServiceType.LOGGER, self.config['logging'].port)
        self.registry.register_service("loader", ServiceType.LOADER, self.config['loader'].port)
        self.registry.register_service("kwd", ServiceType.KWD, self.config['kwd'].port)
        self.registry.register_service("stt", ServiceType.STT, self.config['stt'].port)
        self.registry.register_service("llm", ServiceType.LLM, self.config['llm'].port)
        self.registry.register_service("tts", ServiceType.TTS, self.config['tts'].port)
        
        self.logger.info("Service registry initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.logger.info("Signal handlers configured")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        signal_names = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}
        signal_name = signal_names.get(signum, str(signum))
        self.logger.info(f"Received signal {signal_name}, initiating graceful shutdown...")
        self._shutdown_requested = True
    
    def start_loader_service(self) -> bool:
        """Start the loader service process"""
        try:
            # Start loader service as subprocess
            loader_cmd = [sys.executable, "-m", "services.loader"]
            self.loader_process = subprocess.Popen(
                loader_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.logger.info(f"Started loader service with PID {self.loader_process.pid}")
            
            # Wait for loader service to be ready
            loader_port = self.config['loader'].port
            if not wait_for_port("localhost", loader_port, timeout=10.0):
                self.logger.error("Loader service failed to start within timeout")
                return False
            
            # Create gRPC connection to loader
            self.loader_channel = grpc.insecure_channel(f'localhost:{loader_port}')
            self.loader_stub = loader_pb2_grpc.LoaderServiceStub(self.loader_channel)
            
            self.logger.info("Connected to loader service")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start loader service: {e}")
            return False
    
    def coordinate_startup_sequence(self) -> bool:
        """Coordinate the startup sequence with loader service"""
        try:
            # Get services in startup order
            startup_services = ["logger", "tts", "llm", "stt", "kwd"]
            
            for service_name in startup_services:
                self.logger.info(f"Starting service: {service_name}")
                
                request = loader_pb2.StartServiceRequest(service_name=service_name)
                response = self.loader_stub.StartService(request, timeout=30.0)
                
                if not response.success:
                    self.logger.error(f"Failed to start {service_name}: {response.message}")
                    return False
                
                self.logger.info(f"Service {service_name} started successfully (PID: {response.pid})")
            
            # Wait for all services to be healthy
            if not self.registry.wait_for_services(startup_services, timeout=60.0):
                self.logger.error("Not all services became healthy within timeout")
                return False
            
            self.logger.info("All services started and healthy")
            return True
            
        except grpc.RpcError as e:
            self.logger.error(f"gRPC error during startup coordination: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error during startup coordination: {e}")
            return False
    
    def run(self) -> int:
        """Main execution loop"""
        if not self.initialize():
            return 1
        
        # Start loader service
        if not self.start_loader_service():
            self.logger.error("Failed to start loader service")
            return 1
        
        # Coordinate startup sequence
        if not self.coordinate_startup_sequence():
            self.logger.error("Failed to start all services")
            self.shutdown()
            return 1
        
        self.logger.info("AI Voice Assistant ready - all services operational")
        
        # Main loop - monitor system and wait for shutdown
        try:
            while not self._shutdown_requested:
                # Perform periodic health checks
                self._perform_health_checks()
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        
        return self.shutdown()
    
    def _perform_health_checks(self):
        """Perform periodic health checks on critical services"""
        # Check loader service health
        if self.loader_process and self.loader_process.poll() is not None:
            self.logger.error("Loader service process has terminated unexpectedly")
            self._shutdown_requested = True
    
    def shutdown(self) -> int:
        """Graceful shutdown of all services"""
        self.logger.info("Shutting down AI Voice Assistant...")
        
        shutdown_timeout = self.config['main'].graceful_shutdown_ms / 1000.0 if self.config else 5.0
        
        try:
            # Stop services through loader if available
            if self.loader_stub:
                self._shutdown_services_via_loader()
            
            # Close gRPC connection
            if self.loader_channel:
                self.loader_channel.close()
                self.loader_channel = None
            
            # Terminate loader process
            if self.loader_process:
                self._terminate_loader_process(shutdown_timeout)
            
            self.logger.info("Shutdown complete")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return 1
    
    def _shutdown_services_via_loader(self):
        """Shutdown services through loader service"""
        try:
            # Get services in shutdown order (reverse of startup)
            shutdown_services = ["kwd", "stt", "llm", "tts", "logger"]
            
            for service_name in shutdown_services:
                try:
                    self.logger.info(f"Stopping service: {service_name}")
                    request = loader_pb2.StopServiceRequest(service_name=service_name)
                    response = self.loader_stub.StopService(request, timeout=10.0)
                    
                    if response.success:
                        self.logger.info(f"Service {service_name} stopped successfully")
                    else:
                        self.logger.warning(f"Failed to stop {service_name}: {response.message}")
                        
                except grpc.RpcError as e:
                    self.logger.warning(f"gRPC error stopping {service_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error shutting down services: {e}")
    
    def _terminate_loader_process(self, timeout: float):
        """Terminate the loader process gracefully"""
        try:
            # Try graceful termination first
            self.loader_process.terminate()
            
            try:
                self.loader_process.wait(timeout=timeout)
                self.logger.info("Loader process terminated gracefully")
            except subprocess.TimeoutExpired:
                self.logger.warning("Loader process did not terminate gracefully, forcing kill")
                self.loader_process.kill()
                self.loader_process.wait()
                
        except Exception as e:
            self.logger.error(f"Error terminating loader process: {e}")
        finally:
            self.loader_process = None


def main():
    """Main entry point"""
    orchestrator = MainOrchestrator()
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
