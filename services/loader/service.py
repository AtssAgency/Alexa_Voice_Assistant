"""Loader Service - Process management and orchestration"""

import logging
import sys
import os
import subprocess
import signal
import time
import threading
from typing import Dict, List, Optional, Tuple
from services.base import BaseService
from services.proto import loader_pb2
from services.proto import loader_pb2_grpc
from services.utils import get_process_info, generate_dialog_id
from google.protobuf import empty_pb2
import psutil


class LoaderService(BaseService, loader_pb2_grpc.LoaderServiceServicer):
    """Service loader with process management and orchestration"""
    
    def __init__(self, port: int = 5002):
        super().__init__("loader", port)
        self.service_pids: Dict[str, int] = {}
        self.service_processes: Dict[str, subprocess.Popen] = {}
        self.restart_counts: Dict[str, List[float]] = {}
        self.config = None
        self._shutdown_event = threading.Event()
        
        # Service startup order (TTS → LLM → STT → KWD)
        self.startup_order = ["tts", "llm", "stt", "kwd"]
        self.shutdown_order = ["kwd", "stt", "llm", "tts"]
        
        # Service port mapping
        self.service_ports = {
            "logger": 5001,
            "tts": 5006,
            "llm": 5005,
            "stt": 5004,
            "kwd": 5003
        }
        
        # Dialog state management
        self.current_dialog_id: Optional[str] = None
        self.dialog_state: str = "idle"  # idle, listening, processing, speaking
        self.dialog_turn: int = 0
        self.dialog_timer: Optional[threading.Timer] = None
        self.dialog_lock = threading.Lock()
    
    def register_service(self, server):
        """Register the loader service with gRPC server"""
        loader_pb2_grpc.add_LoaderServiceServicer_to_server(self, server)
    
    def configure(self, config):
        """Configure the loader service"""
        self.config = config
        self.logger.info(f"Loader service configured with health interval: {config.health_interval_ms}ms")
        self.logger.info(f"Restart backoff: {config.restart_backoff_ms}ms")
        self.logger.info(f"Max restarts per minute: {config.max_restarts_per_minute}")
        
        # Update service ports from config if available
        if hasattr(config, 'service_ports'):
            self.service_ports.update(config.service_ports)
    
    def StartService(self, request, context):
        """Start a service with process management"""
        service_name = request.service_name
        self.logger.info(f"Starting service: {service_name}")
        
        try:
            # Check if service is already running
            if service_name in self.service_pids:
                pid = self.service_pids[service_name]
                if self._is_process_running(pid):
                    return loader_pb2.StartServiceResponse(
                        success=False,
                        message=f"Service {service_name} is already running (PID: {pid})",
                        pid=pid
                    )
                else:
                    # Clean up stale PID
                    self._cleanup_service(service_name)
            
            # Start the service process
            pid = self._start_service_process(service_name)
            if pid:
                self.service_pids[service_name] = pid
                self.logger.info(f"Service {service_name} started with PID: {pid}")
                return loader_pb2.StartServiceResponse(
                    success=True,
                    message=f"Service {service_name} started successfully",
                    pid=pid
                )
            else:
                return loader_pb2.StartServiceResponse(
                    success=False,
                    message=f"Failed to start service {service_name}",
                    pid=0
                )
                
        except Exception as e:
            self.logger.error(f"Error starting service {service_name}: {e}")
            return loader_pb2.StartServiceResponse(
                success=False,
                message=f"Error starting service {service_name}: {str(e)}",
                pid=0
            )
    
    def StopService(self, request, context):
        """Stop a service gracefully"""
        service_name = request.service_name
        self.logger.info(f"Stopping service: {service_name}")
        
        try:
            if service_name not in self.service_pids:
                return loader_pb2.StopServiceResponse(
                    success=False,
                    message=f"Service {service_name} is not running"
                )
            
            pid = self.service_pids[service_name]
            success = self._stop_service_process(service_name, pid)
            
            if success:
                self._cleanup_service(service_name)
                return loader_pb2.StopServiceResponse(
                    success=True,
                    message=f"Service {service_name} stopped successfully"
                )
            else:
                return loader_pb2.StopServiceResponse(
                    success=False,
                    message=f"Failed to stop service {service_name}"
                )
                
        except Exception as e:
            self.logger.error(f"Error stopping service {service_name}: {e}")
            return loader_pb2.StopServiceResponse(
                success=False,
                message=f"Error stopping service {service_name}: {str(e)}"
            )
    
    def GetPids(self, request, context):
        """Get service PIDs with status information"""
        # Clean up any stale PIDs
        active_pids = {}
        for service_name, pid in self.service_pids.items():
            if self._is_process_running(pid):
                active_pids[service_name] = pid
            else:
                self.logger.warning(f"Service {service_name} PID {pid} is no longer running")
        
        # Update internal state
        self.service_pids = active_pids
        
        return loader_pb2.GetPidsResponse(service_pids=self.service_pids)
    
    def HandleWakeWord(self, request, context):
        """Handle wake word event and initiate dialog"""
        self.logger.info(f"Wake word detected: {request.type} (confidence: {request.confidence})")
        
        with self.dialog_lock:
            if self.dialog_state != "idle":
                self.logger.warning(f"Wake word detected while in {self.dialog_state} state, ignoring")
                return empty_pb2.Empty()
            
            # Start new dialog session
            self.current_dialog_id = request.dialog_id or generate_dialog_id()
            self.dialog_state = "listening"
            self.dialog_turn = 1
            
            self.logger.info(f"Starting dialog session: {self.current_dialog_id}")
            
            # Cancel any existing dialog timer
            if self.dialog_timer:
                self.dialog_timer.cancel()
            
            # Play confirmation phrase and start STT
            self._play_confirmation_phrase()
            self._start_stt_for_dialog()
        
        return empty_pb2.Empty()
    
    def HandleDialogComplete(self, request, context):
        """Handle dialog completion"""
        self.logger.info(f"Dialog complete: {request.dialog_id} ({request.completion_reason})")
        
        with self.dialog_lock:
            if self.current_dialog_id == request.dialog_id:
                self._end_dialog_session(request.completion_reason)
            else:
                self.logger.warning(f"Dialog completion for unknown dialog: {request.dialog_id}")
        
        return empty_pb2.Empty()
    
    def _start_service_process(self, service_name: str) -> Optional[int]:
        """Start a service process and return its PID"""
        try:
            # Build command to start the service
            cmd = self._build_service_command(service_name)
            if not cmd:
                self.logger.error(f"Unknown service: {service_name}")
                return None
            
            self.logger.info(f"Starting {service_name} with command: {' '.join(cmd)}")
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Store process reference
            self.service_processes[service_name] = process
            
            # Give the process a moment to start
            time.sleep(0.5)
            
            # Check if process is still running
            if process.poll() is None:
                return process.pid
            else:
                # Process died immediately
                stdout, stderr = process.communicate()
                self.logger.error(f"Service {service_name} failed to start:")
                self.logger.error(f"STDOUT: {stdout.decode()}")
                self.logger.error(f"STDERR: {stderr.decode()}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to start service {service_name}: {e}")
            return None
    
    def _build_service_command(self, service_name: str) -> Optional[List[str]]:
        """Build the command to start a service"""
        service_commands = {
            "logger": ["python", "-m", "services.logger"],
            "tts": ["python", "-m", "services.tts"],
            "llm": ["python", "-m", "services.llm"],
            "stt": ["python", "-m", "services.stt"],
            "kwd": ["python", "-m", "services.kwd"]
        }
        
        return service_commands.get(service_name)
    
    def _stop_service_process(self, service_name: str, pid: int) -> bool:
        """Stop a service process gracefully"""
        try:
            # First try graceful shutdown with SIGTERM
            if service_name in self.service_processes:
                process = self.service_processes[service_name]
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5.0)
                    self.logger.info(f"Service {service_name} terminated gracefully")
                    return True
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown failed
                    self.logger.warning(f"Service {service_name} did not terminate gracefully, force killing")
                    process.kill()
                    process.wait()
                    return True
            else:
                # Try to kill by PID if we don't have process reference
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(2.0)
                    
                    # Check if still running
                    if self._is_process_running(pid):
                        os.kill(pid, signal.SIGKILL)
                        time.sleep(1.0)
                    
                    return not self._is_process_running(pid)
                except ProcessLookupError:
                    # Process already dead
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error stopping service {service_name}: {e}")
            return False
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is running"""
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def _cleanup_service(self, service_name: str):
        """Clean up service references"""
        if service_name in self.service_pids:
            del self.service_pids[service_name]
        if service_name in self.service_processes:
            del self.service_processes[service_name]
        if service_name in self.restart_counts:
            del self.restart_counts[service_name]
    
    def start_all_services(self) -> bool:
        """Start all services in the correct order"""
        self.logger.info("Starting all services in order: " + " → ".join(self.startup_order))
        
        for service_name in self.startup_order:
            try:
                pid = self._start_service_process(service_name)
                if pid:
                    self.service_pids[service_name] = pid
                    self.logger.info(f"✓ {service_name} started (PID: {pid})")
                    
                    # Wait a bit between service starts
                    time.sleep(1.0)
                else:
                    self.logger.error(f"✗ Failed to start {service_name}")
                    return False
            except Exception as e:
                self.logger.error(f"✗ Error starting {service_name}: {e}")
                return False
        
        self.logger.info("All services started successfully")
        return True
    
    def stop_all_services(self) -> bool:
        """Stop all services in reverse order"""
        self.logger.info("Stopping all services in order: " + " → ".join(self.shutdown_order))
        
        success = True
        for service_name in self.shutdown_order:
            if service_name in self.service_pids:
                try:
                    pid = self.service_pids[service_name]
                    if self._stop_service_process(service_name, pid):
                        self.logger.info(f"✓ {service_name} stopped")
                        self._cleanup_service(service_name)
                    else:
                        self.logger.error(f"✗ Failed to stop {service_name}")
                        success = False
                except Exception as e:
                    self.logger.error(f"✗ Error stopping {service_name}: {e}")
                    success = False
                    
                # Wait a bit between service stops
                time.sleep(0.5)
        
        return success
    
    def start_health_monitoring(self):
        """Start health monitoring thread"""
        if not self.config:
            self.logger.error("Cannot start health monitoring without configuration")
            return
        
        self.logger.info("Starting health monitoring thread")
        health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        health_thread.start()
    
    def _health_monitor_loop(self):
        """Main health monitoring loop"""
        interval_seconds = self.config.health_interval_ms / 1000.0
        
        while not self._shutdown_event.is_set():
            try:
                self._check_all_services_health()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(1.0)  # Brief pause on error
    
    def _check_all_services_health(self):
        """Check health of all running services"""
        for service_name in list(self.service_pids.keys()):
            try:
                self._check_service_health(service_name)
            except Exception as e:
                self.logger.error(f"Error checking health of {service_name}: {e}")
    
    def _check_service_health(self, service_name: str):
        """Check health of a specific service"""
        if service_name not in self.service_pids:
            return
        
        pid = self.service_pids[service_name]
        
        # First check if process is still running
        if not self._is_process_running(pid):
            self.logger.warning(f"Service {service_name} process (PID: {pid}) is not running")
            self._handle_service_failure(service_name, "process_dead")
            return
        
        # Check gRPC health endpoint
        if not self._check_grpc_health(service_name):
            self.logger.warning(f"Service {service_name} failed gRPC health check")
            self._handle_service_failure(service_name, "health_check_failed")
            return
        
        # Service is healthy
        self.logger.debug(f"Service {service_name} is healthy")
    
    def _check_grpc_health(self, service_name: str) -> bool:
        """Check gRPC health endpoint for a service"""
        try:
            import grpc
            from grpc_health.v1 import health_pb2, health_pb2_grpc
            
            port = self.service_ports.get(service_name)
            if not port:
                self.logger.warning(f"No port configured for service {service_name}")
                return False
            
            # Create channel with short timeout
            channel = grpc.insecure_channel(f'localhost:{port}')
            stub = health_pb2_grpc.HealthStub(channel)
            
            # Check health with timeout
            request = health_pb2.HealthCheckRequest(service=service_name)
            response = stub.Check(request, timeout=2.0)
            
            channel.close()
            return response.status == health_pb2.HealthCheckResponse.SERVING
            
        except Exception as e:
            self.logger.debug(f"gRPC health check failed for {service_name}: {e}")
            return False
    
    def _handle_service_failure(self, service_name: str, failure_reason: str):
        """Handle service failure with restart logic"""
        self.logger.error(f"Service {service_name} failed: {failure_reason}")
        
        # Clean up failed service
        self._cleanup_service(service_name)
        
        # Check restart policy
        if self._should_restart_service(service_name):
            self.logger.info(f"Attempting to restart service {service_name}")
            self._restart_service_with_backoff(service_name)
        else:
            self.logger.error(f"Service {service_name} exceeded restart limits, not restarting")
            self._handle_escalation(service_name)
    
    def _should_restart_service(self, service_name: str) -> bool:
        """Check if service should be restarted based on policy"""
        if not self.config:
            return False
        
        current_time = time.time()
        
        # Initialize restart tracking if needed
        if service_name not in self.restart_counts:
            self.restart_counts[service_name] = []
        
        # Clean old restart timestamps (older than 1 minute)
        self.restart_counts[service_name] = [
            timestamp for timestamp in self.restart_counts[service_name]
            if current_time - timestamp < 60.0
        ]
        
        # Check if we've exceeded the restart limit
        restart_count = len(self.restart_counts[service_name])
        max_restarts = self.config.max_restarts_per_minute
        
        if restart_count >= max_restarts:
            self.logger.error(
                f"Service {service_name} has failed {restart_count} times in the last minute "
                f"(limit: {max_restarts})"
            )
            return False
        
        return True
    
    def _restart_service_with_backoff(self, service_name: str):
        """Restart service with exponential backoff"""
        if not self.config:
            return
        
        # Record restart attempt
        current_time = time.time()
        if service_name not in self.restart_counts:
            self.restart_counts[service_name] = []
        self.restart_counts[service_name].append(current_time)
        
        # Calculate backoff delay
        attempt_count = len(self.restart_counts[service_name])
        backoff_index = min(attempt_count - 1, len(self.config.restart_backoff_ms) - 1)
        backoff_ms = self.config.restart_backoff_ms[backoff_index]
        backoff_seconds = backoff_ms / 1000.0
        
        self.logger.info(
            f"Restarting {service_name} in {backoff_seconds}s "
            f"(attempt {attempt_count}, backoff: {backoff_ms}ms)"
        )
        
        # Start restart in background thread
        restart_thread = threading.Thread(
            target=self._delayed_restart,
            args=(service_name, backoff_seconds),
            daemon=True
        )
        restart_thread.start()
    
    def _delayed_restart(self, service_name: str, delay_seconds: float):
        """Restart service after delay"""
        try:
            time.sleep(delay_seconds)
            
            if self._shutdown_event.is_set():
                return  # Don't restart if shutting down
            
            self.logger.info(f"Attempting restart of {service_name}")
            pid = self._start_service_process(service_name)
            
            if pid:
                self.service_pids[service_name] = pid
                self.logger.info(f"Service {service_name} restarted successfully (PID: {pid})")
            else:
                self.logger.error(f"Failed to restart service {service_name}")
                
        except Exception as e:
            self.logger.error(f"Error during delayed restart of {service_name}: {e}")
    
    def _handle_escalation(self, service_name: str):
        """Handle escalation when service cannot be restarted"""
        self.logger.error(f"ESCALATION: Service {service_name} has exceeded restart limits")
        
        # Log FATAL as required by requirements
        fatal_logger = logging.getLogger("FATAL")
        fatal_logger.critical(
            f"Service {service_name} failed repeatedly and cannot be restarted. "
            f"System may need manual intervention."
        )
        
        # For now, just log the escalation
        # In a full implementation, this might trigger:
        # - Graceful dialog termination
        # - System shutdown
        # - Alert notifications
        # - Fallback modes
    
    def shutdown(self):
        """Shutdown the loader service and all managed services"""
        self.logger.info("Shutting down loader service")
        self._shutdown_event.set()
        
        # Stop all managed services
        self.stop_all_services()
        
        # Stop our own service
        self.stop()
    
    def _play_confirmation_phrase(self):
        """Play random confirmation phrase via TTS"""
        try:
            if not self.config or not hasattr(self.config, 'yes_phrases'):
                # Use default phrases if config not available
                phrases = ["Yes?", "Yes, Master?", "Sup?", "Yo"]
            else:
                phrases = self.config.yes_phrases
            
            import random
            phrase = random.choice(phrases)
            
            self.logger.info(f"Playing confirmation phrase: '{phrase}'")
            
            # TODO: Send phrase to TTS service
            # This would be implemented when TTS service is available
            # For now, just log the action
            
        except Exception as e:
            self.logger.error(f"Error playing confirmation phrase: {e}")
    
    def _start_stt_for_dialog(self):
        """Start STT service for current dialog"""
        try:
            self.logger.info(f"Starting STT for dialog: {self.current_dialog_id}")
            
            # TODO: Send start command to STT service
            # This would be implemented when STT service is available
            # For now, just log the action
            
        except Exception as e:
            self.logger.error(f"Error starting STT for dialog: {e}")
    
    def handle_stt_result(self, dialog_id: str, text: str):
        """Handle STT transcription result"""
        with self.dialog_lock:
            if self.current_dialog_id != dialog_id:
                self.logger.warning(f"STT result for unknown dialog: {dialog_id}")
                return
            
            if self.dialog_state != "listening":
                self.logger.warning(f"STT result received in {self.dialog_state} state")
                return
            
            self.logger.info(f"User said: '{text}' (dialog: {dialog_id}, turn: {self.dialog_turn})")
            
            # Transition to processing state
            self.dialog_state = "processing"
            
            # Send to LLM for processing
            self._send_to_llm(text)
    
    def _send_to_llm(self, user_text: str):
        """Send user text to LLM for processing"""
        try:
            self.logger.info(f"Sending to LLM: '{user_text}'")
            
            # TODO: Send to LLM service
            # This would be implemented when LLM service is available
            # For now, simulate processing and move to speaking state
            
            # Simulate LLM response
            self._handle_llm_response("I understand your request.")
            
        except Exception as e:
            self.logger.error(f"Error sending to LLM: {e}")
            self._end_dialog_session("llm_error")
    
    def _handle_llm_response(self, response_text: str):
        """Handle LLM response and start TTS"""
        with self.dialog_lock:
            if self.dialog_state != "processing":
                self.logger.warning(f"LLM response received in {self.dialog_state} state")
                return
            
            self.logger.info(f"LLM response: '{response_text}'")
            
            # Transition to speaking state
            self.dialog_state = "speaking"
            
            # Send to TTS
            self._send_to_tts(response_text)
    
    def _send_to_tts(self, response_text: str):
        """Send response text to TTS for synthesis"""
        try:
            self.logger.info(f"Sending to TTS: '{response_text}'")
            
            # TODO: Send to TTS service
            # This would be implemented when TTS service is available
            # For now, simulate TTS completion
            
            # Simulate TTS completion
            self._handle_tts_complete()
            
        except Exception as e:
            self.logger.error(f"Error sending to TTS: {e}")
            self._end_dialog_session("tts_error")
    
    def _handle_tts_complete(self):
        """Handle TTS completion and start follow-up window"""
        with self.dialog_lock:
            if self.dialog_state != "speaking":
                self.logger.warning(f"TTS completion received in {self.dialog_state} state")
                return
            
            self.logger.info("TTS playback completed, opening follow-up window")
            
            # Transition back to listening state for follow-up
            self.dialog_state = "listening"
            self.dialog_turn += 1
            
            # Start 4-second follow-up timer
            self._start_followup_timer()
    
    def _start_followup_timer(self):
        """Start 4-second follow-up window timer"""
        if not self.config:
            timeout_ms = 4000  # Default 4 seconds
        else:
            timeout_ms = getattr(self.config, 'followup_window_ms', 4000)
        
        timeout_seconds = timeout_ms / 1000.0
        
        self.logger.info(f"Starting {timeout_seconds}s follow-up timer")
        
        # Cancel existing timer if any
        if self.dialog_timer:
            self.dialog_timer.cancel()
        
        # Start new timer
        self.dialog_timer = threading.Timer(timeout_seconds, self._handle_followup_timeout)
        self.dialog_timer.start()
    
    def _handle_followup_timeout(self):
        """Handle follow-up window timeout"""
        with self.dialog_lock:
            if self.dialog_state == "listening" and self.current_dialog_id:
                self.logger.info("Follow-up window timed out, ending dialog")
                self._end_dialog_session("timeout")
    
    def _end_dialog_session(self, reason: str):
        """End the current dialog session"""
        if not self.current_dialog_id:
            return
        
        self.logger.info(f"Ending dialog session: {self.current_dialog_id} (reason: {reason})")
        
        # Cancel any active timer
        if self.dialog_timer:
            self.dialog_timer.cancel()
            self.dialog_timer = None
        
        # Reset dialog state
        old_dialog_id = self.current_dialog_id
        self.current_dialog_id = None
        self.dialog_state = "idle"
        self.dialog_turn = 0
        
        # Re-enable wake word detection
        self._enable_wake_word_detection()
        
        # Log dialog completion
        self.logger.info(f"Dialog {old_dialog_id} ended, returning to idle state")
    
    def _enable_wake_word_detection(self):
        """Re-enable wake word detection"""
        try:
            self.logger.info("Re-enabling wake word detection")
            
            # TODO: Send enable command to KWD service
            # This would be implemented when KWD service is available
            # For now, just log the action
            
        except Exception as e:
            self.logger.error(f"Error enabling wake word detection: {e}")
    
    def get_dialog_state(self) -> dict:
        """Get current dialog state information"""
        with self.dialog_lock:
            return {
                "dialog_id": self.current_dialog_id,
                "state": self.dialog_state,
                "turn": self.dialog_turn,
                "has_timer": self.dialog_timer is not None
            }


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
        
        # Start health monitoring
        service.start_health_monitoring()
        
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