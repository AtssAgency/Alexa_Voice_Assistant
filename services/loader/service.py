"""Loader Service - Basic implementation for main orchestrator coordination"""

import logging
import sys
import os
from services.base import BaseService
from services.proto import loader_pb2
from services.proto import loader_pb2_grpc
from google.protobuf import empty_pb2


class LoaderService(BaseService, loader_pb2_grpc.LoaderServiceServicer):
    """Basic loader service for main orchestrator coordination"""
    
    def __init__(self, port: int = 5002):
        super().__init__("loader", port)
        self.service_pids = {}
    
    def register_service(self, server):
        """Register the loader service with gRPC server"""
        loader_pb2_grpc.add_LoaderServiceServicer_to_server(self, server)
    
    def configure(self, config):
        """Configure the loader service"""
        # Basic configuration - will be enhanced in task 6
        pass
    
    def StartService(self, request, context):
        """Start a service (placeholder implementation)"""
        service_name = request.service_name
        self.logger.info(f"Start service request received: {service_name}")
        
        # For now, just return success - actual implementation in task 6
        return loader_pb2.StartServiceResponse(
            success=True,
            message=f"Service {service_name} start requested",
            pid=0  # Placeholder PID
        )
    
    def StopService(self, request, context):
        """Stop a service (placeholder implementation)"""
        service_name = request.service_name
        self.logger.info(f"Stop service request received: {service_name}")
        
        # For now, just return success - actual implementation in task 6
        return loader_pb2.StopServiceResponse(
            success=True,
            message=f"Service {service_name} stop requested"
        )
    
    def GetPids(self, request, context):
        """Get service PIDs (placeholder implementation)"""
        return loader_pb2.GetPidsResponse(service_pids=self.service_pids)
    
    def HandleWakeWord(self, request, context):
        """Handle wake word event (placeholder implementation)"""
        self.logger.info(f"Wake word event: {request.type} (confidence: {request.confidence})")
        return empty_pb2.Empty()
    
    def HandleDialogComplete(self, request, context):
        """Handle dialog completion (placeholder implementation)"""
        self.logger.info(f"Dialog complete: {request.dialog_id} ({request.completion_reason})")
        return empty_pb2.Empty()


def main():
    """Main entry point for loader service"""
    from services.config import ConfigManager
    from services.utils import setup_logging
    
    # Setup logging
    logger = setup_logging("INFO", "loader")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load()
        
        # Create and start service
        service = LoaderService(config['loader'].port)
        service.configure(config['loader'])
        service.start()
        
        logger.info("Loader service started successfully")
        service.wait_for_termination()
        
    except KeyboardInterrupt:
        logger.info("Loader service interrupted")
    except Exception as e:
        logger.error(f"Loader service error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())