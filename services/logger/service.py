"""Centralized logging service implementation"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional
import threading
import uuid

import grpc
from google.protobuf import empty_pb2

from ..base import BaseService
from ..proto import logger_pb2, logger_pb2_grpc
from ..config import LoggingConfig


class LoggerService(BaseService, logger_pb2_grpc.LoggerServiceServicer):
    """Centralized logging service with gRPC interface"""
    
    def __init__(self, port: int = 5001):
        super().__init__("logger", port)
        self.config: Optional[LoggingConfig] = None
        self.app_logger: Optional[logging.Logger] = None
        self.logs_dir: str = "logs"
        self._lock = threading.Lock()
        self._dialog_files = {}  # dialog_id -> file handle
        
    def configure(self, config: LoggingConfig):
        """Configure the logging service"""
        self.config = config
        self.logs_dir = config.logs_dir
        
        # Ensure logs directory exists
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Setup application logger with rotation
        self._setup_app_logger()
        
        self.logger.info(f"Logger service configured with logs_dir: {self.logs_dir}")
    
    def register_service(self, server):
        """Register the LoggerService with the gRPC server"""
        logger_pb2_grpc.add_LoggerServiceServicer_to_server(self, server)
    
    def _setup_app_logger(self):
        """Setup the application logger with rotation"""
        app_log_path = os.path.join(self.logs_dir, "app.log")
        
        # Reset app log if configured to do so
        if self.config.app_reset_on_start and os.path.exists(app_log_path):
            try:
                os.remove(app_log_path)
                self.logger.info("Reset app.log on startup")
            except OSError as e:
                self.logger.warning(f"Failed to reset app.log: {e}")
        
        # Create rotating file handler
        max_bytes = self.config.rotate_size_mb * 1024 * 1024  # Convert MB to bytes
        handler = logging.handlers.RotatingFileHandler(
            app_log_path,
            maxBytes=max_bytes,
            backupCount=self.config.rotate_keep_files
        )
        
        # Set formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s|%(levelname)s|%(name)s|%(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Create dedicated app logger
        self.app_logger = logging.getLogger("app")
        self.app_logger.setLevel(logging.DEBUG)
        self.app_logger.handlers.clear()  # Remove any existing handlers
        self.app_logger.addHandler(handler)
        self.app_logger.propagate = False  # Don't propagate to root logger
        
        self.logger.info("Application logger configured with rotation")
    
    def WriteApp(self, request: logger_pb2.LogEntry, context) -> empty_pb2.Empty:
        """Write structured entry to app.log"""
        try:
            with self._lock:
                if not self.app_logger:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details("Logger not configured")
                    return empty_pb2.Empty()
                
                # Convert gRPC log level to Python logging level
                level_map = {
                    'DEBUG': logging.DEBUG,
                    'INFO': logging.INFO,
                    'WARNING': logging.WARNING,
                    'WARN': logging.WARNING,  # Accept both forms
                    'ERROR': logging.ERROR,
                    'FATAL': logging.CRITICAL,
                    'CRITICAL': logging.CRITICAL
                }
                
                log_level = level_map.get(request.level.upper(), logging.INFO)
                
                # Format message: service|event|details
                message = f"{request.service}|{request.event}|{request.details}"
                
                # Log with timestamp (formatter will add its own, but we preserve the original)
                self.app_logger.log(log_level, message)
                
            return empty_pb2.Empty()
            
        except Exception as e:
            self.logger.error(f"Failed to write app log: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to write log: {str(e)}")
            return empty_pb2.Empty()
    
    def WriteDialog(self, request: logger_pb2.DialogEntry, context) -> empty_pb2.Empty:
        """Write user/assistant entry to dialog log"""
        try:
            with self._lock:
                dialog_id = request.dialog_id
                
                # Get or create dialog file handle
                if dialog_id not in self._dialog_files:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details(f"Dialog {dialog_id} not found. Call NewDialog first.")
                    return empty_pb2.Empty()
                
                file_handle = self._dialog_files[dialog_id]
                
                # Format: TIMESTAMP|SPEAKER: TEXT
                timestamp = request.timestamp or datetime.now().isoformat()
                line = f"{timestamp}|{request.speaker}: {request.text}\n"
                
                file_handle.write(line)
                file_handle.flush()  # Ensure immediate write
                
            return empty_pb2.Empty()
            
        except Exception as e:
            self.logger.error(f"Failed to write dialog log: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to write dialog log: {str(e)}")
            return empty_pb2.Empty()
    
    def NewDialog(self, request: empty_pb2.Empty, context) -> logger_pb2.NewDialogResponse:
        """Create new dialog session and log file"""
        try:
            with self._lock:
                # Generate dialog ID with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dialog_id = f"dialog_{timestamp}_{uuid.uuid4().hex[:8]}"
                
                # Create dialog log file
                log_file_path = os.path.join(self.logs_dir, f"{dialog_id}.log")
                
                try:
                    file_handle = open(log_file_path, 'w', encoding='utf-8')
                    
                    # Write header
                    header = f"# Dialog Log: {dialog_id}\n"
                    header += f"# Started: {datetime.now().isoformat()}\n\n"
                    file_handle.write(header)
                    file_handle.flush()
                    
                    # Store file handle
                    self._dialog_files[dialog_id] = file_handle
                    
                    self.logger.info(f"Created new dialog: {dialog_id}")
                    
                    return logger_pb2.NewDialogResponse(
                        dialog_id=dialog_id,
                        log_file_path=log_file_path
                    )
                    
                except OSError as e:
                    self.logger.error(f"Failed to create dialog log file: {e}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(f"Failed to create dialog log: {str(e)}")
                    return logger_pb2.NewDialogResponse()
                    
        except Exception as e:
            self.logger.error(f"Failed to create new dialog: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to create dialog: {str(e)}")
            return logger_pb2.NewDialogResponse()
    
    def close_dialog(self, dialog_id: str):
        """Close a dialog session and its log file"""
        try:
            with self._lock:
                if dialog_id in self._dialog_files:
                    file_handle = self._dialog_files[dialog_id]
                    
                    # Write footer
                    footer = f"\n# Dialog ended: {datetime.now().isoformat()}\n"
                    file_handle.write(footer)
                    file_handle.close()
                    
                    del self._dialog_files[dialog_id]
                    self.logger.info(f"Closed dialog: {dialog_id}")
                    
        except Exception as e:
            self.logger.error(f"Failed to close dialog {dialog_id}: {e}")
    
    def stop(self, grace_period=5):
        """Stop the service and close all dialog files"""
        try:
            # Close all open dialog files
            with self._lock:
                for dialog_id, file_handle in list(self._dialog_files.items()):
                    try:
                        footer = f"\n# Dialog terminated: {datetime.now().isoformat()}\n"
                        file_handle.write(footer)
                        file_handle.close()
                    except Exception as e:
                        self.logger.error(f"Error closing dialog {dialog_id}: {e}")
                
                self._dialog_files.clear()
                
        except Exception as e:
            self.logger.error(f"Error during logger service shutdown: {e}")
        
        # Call parent stop
        super().stop(grace_period)
    
    def get_dialog_count(self) -> int:
        """Get the number of active dialog sessions"""
        with self._lock:
            return len(self._dialog_files)
    
    def get_active_dialogs(self) -> list:
        """Get list of active dialog IDs"""
        with self._lock:
            return list(self._dialog_files.keys())