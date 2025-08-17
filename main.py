#!/usr/bin/env python3
"""
Main Bootstrap Orchestrator for Alexa (Local Voice Assistant)

Ultra-minimal entrypoint that:
1. Initializes environment and config
2. Starts Logger first (blocking until /log is accepted)
3. Spawns Loader (process manager) which handles the rest
4. Forwards signals and returns Loader's exit code

State flow: INIT → STARTING_LOADER → RUN → SHUTDOWN
"""

import sys
import os
import signal
import subprocess
import time
import json
import configparser
from pathlib import Path
import requests
from typing import Optional


class MainOrchestrator:
    def __init__(self):
        self.config: Optional[configparser.ConfigParser] = None
        self.logger_process: Optional[subprocess.Popen] = None
        self.loader_process: Optional[subprocess.Popen] = None
        self.logger_url = "http://127.0.0.1:5000"
        self.shutdown_requested = False
        
    def load_config(self) -> bool:
        """Load configuration from config/config.ini"""
        try:
            config_path = Path("config/config.ini")
            if not config_path.exists():
                print("ERROR: config/config.ini not found")
                return False
                
            self.config = configparser.ConfigParser()
            self.config.read(config_path)
            
            # Verify [main] section exists
            if 'main' not in self.config:
                print("ERROR: [main] section missing in config.ini")
                return False
                
            return True
        except Exception as e:
            print(f"ERROR: Failed to load config: {e}")
            return False
    
    def get_main_config(self, key: str, default: str = "") -> str:
        """Get value from [main] section of config"""
        return self.config.get('main', key, fallback=default)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.shutdown_requested = True
            self.shutdown()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_logger(self) -> bool:
        """Start logger service and wait for it to accept /log requests"""
        try:
            # Use current Python interpreter for logger
            wait_logger_ms = int(self.get_main_config('wait_logger_ms', '3000'))
            
            # Pre-logger bootstrap message to stdout
            print("MAIN: Starting logger service...")
            
            # Spawn logger process using current interpreter
            self.logger_process = subprocess.Popen(
                [sys.executable, "-m", "services.logger_service"],
               # stdout=subprocess.PIPE,
               # stderr=subprocess.PIPE
            )
            
            # Wait for logger to be ready by polling /log endpoint
            start_time = time.time()
            timeout_sec = wait_logger_ms / 1000.0
            
            while time.time() - start_time < timeout_sec:
                if self.logger_process.poll() is not None:
                    print("ERROR: Logger process exited prematurely")
                    return False
                    
                try:
                    # Try to send initial log message
                    response = requests.post(
                        f"{self.logger_url}/log",
                        json={
                            "svc": "MAIN",
                            "level": "info",
                            "message": "service_start",
                            "event": "service_start"
                        },
                        timeout=1.0
                    )
                    
                    if response.status_code == 200:
                        return True
                        
                except requests.RequestException:
                    # Logger not ready yet, continue waiting
                    time.sleep(0.1)
                    continue
            
            print("ERROR: Logger failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"ERROR: Failed to start logger: {e}")
            return False
    
    def log_message(self, level: str, message: str, event: Optional[str] = None):
        """Send log message to logger service"""
        try:
            payload = {
                "svc": "MAIN",
                "level": level,
                "message": message
            }
            if event:
                payload["event"] = event
                
            requests.post(
                f"{self.logger_url}/log",
                json=payload,
                timeout=2.0
            )
        except Exception:
            # Fallback to stdout if logger unavailable
            print(f"MAIN      {level.upper():<6}= {message}")
    
    def start_loader(self) -> bool:
        """Start loader service (process manager)"""
        try:
            self.log_message("info", "Starting Loader service")
            
            # Spawn loader process using current interpreter
            self.loader_process = subprocess.Popen(
                [sys.executable, "-m", "services.loader_service"],
               # stdout=subprocess.PIPE,
               # stderr=subprocess.PIPE
            )
            
            return True
            
        except Exception as e:
            self.log_message("error", f"Failed to start loader: {e}")
            return False
    
    def wait_for_loader(self) -> int:
        """Wait for loader process and forward signals"""
        if not self.loader_process:
            return 1
            
        try:
            # Wait for loader to complete
            while self.loader_process.poll() is None:
                if self.shutdown_requested:
                    break
                time.sleep(0.1)
                
            return self.loader_process.returncode or 0
            
        except Exception as e:
            self.log_message("error", f"Error waiting for loader: {e}")
            return 1
    
    def shutdown(self):
        """Graceful shutdown - terminate processes in reverse order"""
        try:
            # Signal loader to shutdown (it handles service shutdown)
            if self.loader_process and self.loader_process.poll() is None:
                self.log_message("info", "Terminating Loader")
                self.loader_process.terminate()
                
                # Give loader time to shutdown gracefully
                try:
                    self.loader_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.log_message("warning", "Loader shutdown timeout, killing")
                    self.loader_process.kill()
            
            # Shutdown logger last
            if self.logger_process and self.logger_process.poll() is None:
                # Send shutdown log message
                self.log_message("info", "System shutdown")
                time.sleep(0.1)  # Give logger time to process final message
                
                self.logger_process.terminate()
                try:
                    self.logger_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger_process.kill()
                    
        except Exception as e:
            print(f"ERROR during shutdown: {e}")
    
    def run(self) -> int:
        """Main execution flow"""
        try:
            # INIT: Load configuration
            if not self.load_config():
                return 1
            
            # Setup signal handlers
            self.setup_signal_handlers()
                    
            # STARTING_LOADER: Start loader (process manager)  
            if not self.start_loader():
                self.shutdown()
                return 1
            
            # RUN: Wait for loader and forward signals
            exit_code = self.wait_for_loader()
            
            # SHUTDOWN: Clean shutdown
            self.shutdown()
            
            return exit_code
            
        except KeyboardInterrupt:
            self.log_message("info", "Interrupted by user")
            self.shutdown()
            return 0
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            self.shutdown()
            return 1


def main():
    """Entry point"""
    orchestrator = MainOrchestrator()
    exit_code = orchestrator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
