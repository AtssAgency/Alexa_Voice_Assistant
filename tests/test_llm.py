"""Unit tests for LLM service"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import grpc
import json
import requests
from io import StringIO

from services.llm.service import LlmService
from services.config import LlmConfig
from services.proto import llm_pb2


class TestLlmService(unittest.TestCase):
    """Test cases for LLM service"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = LlmService(port=5005)
        self.config = LlmConfig(
            port=5005,
            endpoint="localhost:11434",
            model="llama3.1:8b-instruct-q4",
            modelfile_path="config/Modelfile",
            request_timeout_ms=30000,
            stream_chunk_chars=80
        )
    
    def test_service_initialization(self):
        """Test service initialization"""
        self.assertEqual(self.service.service_name, "llm")
        self.assertEqual(self.service.port, 5005)
        self.assertFalse(self.service._configured)
    
    @patch('services.llm.service.requests.get')
    @patch('services.llm.service.os.path.exists')
    @patch('builtins.open', create=True)
    def test_configure_success(self, mock_open, mock_exists, mock_get):
        """Test successful service configuration"""
        # Mock Modelfile exists and content
        mock_exists.return_value = True
        mock_file = StringIO('SYSTEM "You are a helpful assistant."')
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock Ollama API responses
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'models': [{'name': 'llama3.1:8b-instruct-q4'}]
        }
        mock_get.return_value = mock_response
        
        # Mock logger connection
        with patch.object(self.service, '_setup_logger_connection'):
            self.service.configure(self.config)
        
        self.assertTrue(self.service._configured)
        self.assertEqual(self.service.model_name, "llama3.1:8b-instruct-q4")
        self.assertEqual(self.service.ollama_endpoint, "http://localhost:11434")
        self.assertEqual(self.service.system_prompt, "You are a helpful assistant.")
    
    @patch('services.llm.service.requests.get')
    def test_validate_ollama_connection_failure(self, mock_get):
        """Test Ollama connection validation failure"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with self.assertRaises(RuntimeError) as context:
            self.service._validate_ollama_connection()
        
        self.assertIn("Failed to connect to Ollama API", str(context.exception))
    
    @patch('services.llm.service.requests.get')
    def test_validate_model_availability_missing(self, mock_get):
        """Test model availability validation when model is missing"""
        # Mock API response without the required model
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'models': [{'name': 'other-model'}]
        }
        mock_get.return_value = mock_response
        
        # Mock model pull
        with patch.object(self.service, '_pull_model') as mock_pull:
            self.service.model_name = "llama3.1:8b-instruct-q4"
            self.service.ollama_endpoint = "http://localhost:11434"
            self.service._validate_model_availability()
            mock_pull.assert_called_once()
    
    def test_load_system_prompt_from_modelfile(self):
        """Test loading system prompt from Modelfile"""
        modelfile_content = '''FROM llama3.1:8b-instruct-q4
SYSTEM "You are a helpful AI assistant."
PARAMETER temperature 0.7'''
        
        with patch('services.llm.service.os.path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                mock_file = StringIO(modelfile_content)
                mock_open.return_value.__enter__.return_value = mock_file
                
                self.service._load_system_prompt("config/Modelfile")
                
                self.assertEqual(self.service.system_prompt, "You are a helpful AI assistant.")
    
    def test_load_system_prompt_file_not_found(self):
        """Test loading system prompt when file doesn't exist"""
        with patch('services.llm.service.os.path.exists', return_value=False):
            self.service._load_system_prompt("nonexistent/Modelfile")
            self.assertEqual(self.service.system_prompt, "You are a helpful AI assistant.")
    
    def test_grpc_configure_success(self):
        """Test gRPC Configure method success"""
        request = llm_pb2.LlmConfig(
            endpoint="localhost:11434",
            model="llama3.1:8b-instruct-q4",
            modelfile_path="config/Modelfile",
            request_timeout_ms=30000,
            stream_chunk_chars=80
        )
        
        with patch.object(self.service, 'configure') as mock_configure:
            response = self.service.Configure(request, None)
            
            mock_configure.assert_called_once()
            self.assertTrue(response.success)
            self.assertIn("configured", response.message)
    
    def test_grpc_configure_failure(self):
        """Test gRPC Configure method failure"""
        request = llm_pb2.LlmConfig(
            endpoint="localhost:11434",
            model="invalid-model",
            modelfile_path="config/Modelfile",
            request_timeout_ms=30000,
            stream_chunk_chars=80
        )
        
        with patch.object(self.service, 'configure', side_effect=Exception("Config failed")):
            response = self.service.Configure(request, None)
            
            self.assertFalse(response.success)
            self.assertIn("Configuration failed", response.message)
    
    @patch('services.llm.service.requests.post')
    def test_complete_streaming_response(self, mock_post):
        """Test Complete method with streaming response"""
        # Mock streaming response from Ollama
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            '{"response": "Hello", "done": false}',
            '{"response": " there!", "done": false}',
            '{"response": "", "done": true}'
        ]
        mock_post.return_value = mock_response
        
        # Configure service
        self.service._configured = True
        self.service.model_name = "test-model"
        self.service.ollama_endpoint = "http://localhost:11434"
        self.service.system_prompt = "Test prompt"
        self.service.stream_chunk_chars = 5
        
        # Mock logger
        with patch.object(self.service, '_log_app_event'), \
             patch.object(self.service, '_log_to_dialog'):
            
            request = llm_pb2.UserQuery(
                text="Hello",
                dialog_id="test-dialog",
                turn=1
            )
            
            # Collect streaming response
            chunks = list(self.service.Complete(request, Mock()))
            
            # Verify chunks
            self.assertGreater(len(chunks), 0)
            
            # Last chunk should have eot=True
            self.assertTrue(chunks[-1].eot)
    
    def test_complete_not_configured(self):
        """Test Complete method when service not configured"""
        mock_context = Mock()
        
        request = llm_pb2.UserQuery(
            text="Hello",
            dialog_id="test-dialog",
            turn=1
        )
        
        # Service not configured
        self.service._configured = False
        
        chunks = list(self.service.Complete(request, mock_context))
        
        # Should return empty and set error
        self.assertEqual(len(chunks), 0)
        mock_context.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)
    
    @patch('services.llm.service.requests.post')
    def test_complete_request_timeout(self, mock_post):
        """Test Complete method with request timeout"""
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")
        
        # Configure service
        self.service._configured = True
        self.service.model_name = "test-model"
        self.service.ollama_endpoint = "http://localhost:11434"
        
        mock_context = Mock()
        
        with patch.object(self.service, '_log_app_event'), \
             patch.object(self.service, '_log_to_dialog'):
            
            request = llm_pb2.UserQuery(
                text="Hello",
                dialog_id="test-dialog",
                turn=1
            )
            
            chunks = list(self.service.Complete(request, mock_context))
            
            # Should return empty and set timeout error
            self.assertEqual(len(chunks), 0)
            mock_context.set_code.assert_called_with(grpc.StatusCode.DEADLINE_EXCEEDED)


if __name__ == '__main__':
    unittest.main()