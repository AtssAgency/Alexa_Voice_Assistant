"""AI Voice Assistant - Main Orchestrator"""

import logging
import signal
import sys
import os
from services.config import ConfigManager
from services.utils import check_vram_availability, setup_logging


class MainOrchestrator:
    """Main orchestrator for the AI Voice Assistant system"""
    
    def __init__(self):
        self.config_manager = None
        self.config = None
        self.logger = None
        self._shutdown_requested = False
        
    def initialize(self):
        """Initialize the system"""
        # Setup basic logging first
        self.logger = setup_logging("INFO", "main")
        self.logger.info("AI Voice Assistant starting...")
        
        # Load configuration
        try:
            self.config_manager = ConfigManager()
            self.config = self.config_manager.load()
            self.config_manager.validate()
            
            # Update logging level from config
            logging.getLogger().setLevel(
                getattr(logging, self.config['main'].log_level.upper())
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
        
        # Validate VRAM requirements
        min_vram = self.config['system'].min_vram_mb
        if not check_vram_availability(min_vram):
            self.logger.fatal(f"Insufficient VRAM. Required: {min_vram}MB")
            return False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("System initialization complete")
        return True
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
    
    def run(self):
        """Main execution loop"""
        if not self.initialize():
            return 1
        
        self.logger.info("AI Voice Assistant ready")
        
        # Main loop - for now just wait for shutdown
        try:
            while not self._shutdown_requested:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        
        self.shutdown()
        return 0
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down AI Voice Assistant...")
        # TODO: Stop all services
        self.logger.info("Shutdown complete")


def main():
    """Main entry point"""
    orchestrator = MainOrchestrator()
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
