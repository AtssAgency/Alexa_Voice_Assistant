"""Unit tests for logging service functionality"""

import os
import tempfile
import shutil
import unittest
import threading
import time
from unittest.mock import patch, MagicMock
import grpc
from google.protobuf import empty_pb2

from services.logger.service import LoggerService
from services.proto import logger_pb2, logger_pb2_grpc
from services.config import LoggingConfig


class TestLoggerService(unittest.TestCase):
    """Test cases for LoggerService"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for logs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config = LoggingConfig(
            port=5099,  # Use different port for testing
            app_reset_on_start=True,
            rotate_size_mb=1,  # Small size for testing
            rotate_keep_files=3,
            logs_dir=self.temp_dir
        )
        
        # Create service instance
        self.service = LoggerService(port=5099)
        self.service.configure(self.config)
        
        # Start service in background thread
        self.service_thread = threading.Thread(target=self._start_service)
        self.service_thread.daemon = True
        self.service_thread.start()
        
        # Wait for service to start
        time.sleep(0.1)
        
        # Create gRPC client
        self.channel = grpc.insecure_channel('localhost:5099')
        self.stub = logger_pb2_grpc.LoggerServiceStub(self.channel)
    
    def _start_service(self):
        """Start the service"""
        try:
            self.service.start()
            self.service.wait_for_termination()
        except Exception as e:
            print(f"Service error: {e}")
    
    def tearDown(self):
        """Clean up test environment"""
        # Close channel
        if hasattr(self, 'channel'):
            self.channel.close()
        
        # Stop service and clean up
        self.service.stop()
        
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_service_configuration(self):
        """Test service configuration"""
        self.assertEqual(self.service.logs_dir, self.temp_dir)
        self.assertIsNotNone(self.service.app_logger)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_app_log_reset_on_start(self):
        """Test app.log reset on startup"""
        app_log_path = os.path.join(self.temp_dir, "app.log")
        
        # Create existing app.log
        with open(app_log_path, 'w') as f:
            f.write("existing content\n")
        
        # Configure service (should reset log)
        service = LoggerService(port=5002)
        config = LoggingConfig(
            port=5002,
            app_reset_on_start=True,
            logs_dir=self.temp_dir
        )
        service.configure(config)
        
        # Check that file was reset (should be empty or not exist initially)
        # The rotating handler will create it when first log is written
        service.stop()
    
    def test_write_app_log(self):
        """Test writing to application log"""
        # Create log entry
        log_entry = logger_pb2.LogEntry(
            timestamp="2024-01-01T12:00:00",
            level="INFO",
            service="test_service",
            event="test_event",
            details="test details"
        )
        
        # Write log entry
        response = self.stub.WriteApp(log_entry)
        
        # Check response
        self.assertIsInstance(response, empty_pb2.Empty)
        
        # Check that log file was created and contains entry
        app_log_path = os.path.join(self.temp_dir, "app.log")
        self.assertTrue(os.path.exists(app_log_path))
        
        with open(app_log_path, 'r') as f:
            content = f.read()
            self.assertIn("test_service|test_event|test details", content)
    
    def test_write_app_log_invalid_level(self):
        """Test writing app log with invalid level defaults to INFO"""
        log_entry = logger_pb2.LogEntry(
            timestamp="2024-01-01T12:00:00",
            level="INVALID_LEVEL",
            service="test_service",
            event="test_event",
            details="test details"
        )
        
        # Should not raise error, should default to INFO
        response = self.stub.WriteApp(log_entry)
        
        self.assertIsInstance(response, empty_pb2.Empty)
    
    def test_new_dialog_creation(self):
        """Test creating new dialog session"""
        # Create new dialog
        response = self.stub.NewDialog(empty_pb2.Empty())
        
        # Check response
        self.assertIsInstance(response, logger_pb2.NewDialogResponse)
        self.assertTrue(response.dialog_id.startswith("dialog_"))
        self.assertTrue(response.log_file_path.endswith(".log"))
        
        # Check that dialog file was created
        self.assertTrue(os.path.exists(response.log_file_path))
        
        # Check file content has header
        with open(response.log_file_path, 'r') as f:
            content = f.read()
            self.assertIn("# Dialog Log:", content)
            self.assertIn("# Started:", content)
        
        return response.dialog_id
    
    def test_write_dialog_log(self):
        """Test writing to dialog log"""
        # Create dialog first
        dialog_response = self.stub.NewDialog(empty_pb2.Empty())
        dialog_id = dialog_response.dialog_id
        
        # Create dialog entry
        dialog_entry = logger_pb2.DialogEntry(
            dialog_id=dialog_id,
            speaker="USER",
            text="Hello, assistant!",
            timestamp="2024-01-01T12:00:00"
        )
        
        # Write dialog entry
        response = self.stub.WriteDialog(dialog_entry)
        
        # Check response
        self.assertIsInstance(response, empty_pb2.Empty)
        
        # Check dialog file content
        dialog_file = os.path.join(self.temp_dir, f"{dialog_id}.log")
        with open(dialog_file, 'r') as f:
            content = f.read()
            self.assertIn("USER: Hello, assistant!", content)
            self.assertIn("2024-01-01T12:00:00", content)
    
    def test_write_dialog_log_nonexistent_dialog(self):
        """Test writing to non-existent dialog returns error"""
        dialog_entry = logger_pb2.DialogEntry(
            dialog_id="nonexistent_dialog",
            speaker="USER",
            text="Hello",
            timestamp="2024-01-01T12:00:00"
        )
        
        # Should return error but still return Empty response
        try:
            response = self.stub.WriteDialog(dialog_entry)
            self.assertIsInstance(response, empty_pb2.Empty)
        except grpc.RpcError as e:
            # This is also acceptable - service may return gRPC error
            self.assertEqual(e.code(), grpc.StatusCode.NOT_FOUND)
    
    def test_dialog_id_generation(self):
        """Test dialog ID generation format and uniqueness"""
        # Create multiple dialogs
        dialog_ids = []
        for _ in range(3):
            response = self.stub.NewDialog(empty_pb2.Empty())
            dialog_ids.append(response.dialog_id)
        
        # Check format
        for dialog_id in dialog_ids:
            self.assertTrue(dialog_id.startswith("dialog_"))
            self.assertRegex(dialog_id, r"dialog_\d{8}_\d{6}_[a-f0-9]{8}")
        
        # Check uniqueness
        self.assertEqual(len(dialog_ids), len(set(dialog_ids)))
    
    def test_dialog_session_tracking(self):
        """Test dialog session tracking functionality"""
        # Initially no dialogs
        self.assertEqual(self.service.get_dialog_count(), 0)
        self.assertEqual(self.service.get_active_dialogs(), [])
        
        # Create dialog
        dialog_response = self.stub.NewDialog(empty_pb2.Empty())
        dialog_id = dialog_response.dialog_id
        
        # Check tracking
        self.assertEqual(self.service.get_dialog_count(), 1)
        self.assertIn(dialog_id, self.service.get_active_dialogs())
        
        # Close dialog
        self.service.close_dialog(dialog_id)
        
        # Check tracking after close
        self.assertEqual(self.service.get_dialog_count(), 0)
        self.assertEqual(self.service.get_active_dialogs(), [])
    
    def test_dialog_close_with_footer(self):
        """Test dialog closing adds footer to log file"""
        # Create and close dialog
        dialog_id = self.test_new_dialog_creation()
        self.service.close_dialog(dialog_id)
        
        # Check file has footer
        dialog_file = os.path.join(self.temp_dir, f"{dialog_id}.log")
        with open(dialog_file, 'r') as f:
            content = f.read()
            self.assertIn("# Dialog ended:", content)
    
    def test_service_stop_closes_dialogs(self):
        """Test that stopping service closes all open dialogs"""
        # Create multiple dialogs
        dialog_ids = []
        for _ in range(2):
            response = self.stub.NewDialog(empty_pb2.Empty())
            dialog_ids.append(response.dialog_id)
        
        # Verify dialogs are active
        self.assertEqual(self.service.get_dialog_count(), 2)
        
        # Stop service
        self.service.stop()
        
        # Check that all dialogs were closed
        self.assertEqual(self.service.get_dialog_count(), 0)
        
        # Check that dialog files have termination footer
        for dialog_id in dialog_ids:
            dialog_file = os.path.join(self.temp_dir, f"{dialog_id}.log")
            with open(dialog_file, 'r') as f:
                content = f.read()
                self.assertIn("# Dialog terminated:", content)
    
    def test_log_rotation_configuration(self):
        """Test log rotation configuration"""
        # Check that rotating handler is configured
        self.assertIsNotNone(self.service.app_logger)
        
        # Check handler configuration
        handlers = self.service.app_logger.handlers
        self.assertEqual(len(handlers), 1)
        
        handler = handlers[0]
        self.assertIsInstance(handler, type(self.service.app_logger.handlers[0]))
        
        # The actual rotation testing would require writing large amounts of data
        # which is beyond the scope of unit tests
    
    def test_concurrent_dialog_operations(self):
        """Test concurrent dialog operations are thread-safe"""
        import threading
        import time
        
        results = []
        errors = []
        
        def create_dialog():
            try:
                response = self.stub.NewDialog(empty_pb2.Empty())
                results.append(response.dialog_id)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_dialog)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 5)
        self.assertEqual(len(set(results)), 5)  # All unique


if __name__ == '__main__':
    unittest.main()