"""Large Language Model service implementation using Ollama"""

import grpc
import logging
import requests
import json
import time
import os
from typing import Iterator, Optional, Dict, Any
from concurrent import futures
import threading
from datetime import datetime

from ..base import BaseService
from ..proto import llm_pb2, llm_pb2_grpc, logger_pb2, logger_pb2_grpc
from ..config import LlmConfig


class LlmService(BaseService, llm_pb2_grpc.LlmServiceServicer):
    """Large Language Model service using Ollama API"""
    
    def __init__(self, port: int = 5005):
        super().__init__("llm", port)
        self.config: Optional[LlmConfig] = None
        self.system_prompt: str = ""
        self.ollama_endpoint: str = ""
        self.model_name: str = ""
        self.request_timeout: float = 30.0
        self.stream_chunk_chars: int = 80
        self._configured = False
        self._lock = threading.Lock()
        self.logger_stub: Optional[logger_pb2_grpc.LoggerStub] = None
        self._logger_channel: Optional[grpc.Channel] = None
        
    def register_service(self, server):
        """Register the LLM service with the gRPC server"""
        llm_pb2_grpc.add_LlmServiceServicer_to_server(self, server)
        
    def configure(self, config: LlmConfig):
        """Configure the LLM service"""
        with self._lock:
            self.config = config
            self.ollama_endpoint = f"http://{config.endpoint}"
            self.model_name = config.model
            self.request_timeout = config.request_timeout_ms / 1000.0
            self.stream_chunk_chars = config.stream_chunk_chars
            
            # Load system prompt from Modelfile
            self._load_system_prompt(config.modelfile_path)
            
            # Validate Ollama connection and model
            self._validate_ollama_connection()
            self._validate_model_availability()
            
            # Setup logger service connection
            self._setup_logger_connection()
            
            self._configured = True
            self.logger.info(f"LLM service configured with model: {self.model_name}")
    
    def _load_system_prompt(self, modelfile_path: str) -> None:
        """Load system prompt from Modelfile"""
        try:
            if os.path.exists(modelfile_path):
                with open(modelfile_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # Parse Modelfile format - look for SYSTEM instruction
                lines = content.split('\n')
                system_lines = []
                in_system = False
                
                for line in lines:
                    line = line.strip()
                    if line.upper().startswith('SYSTEM'):
                        # Extract system prompt from SYSTEM line
                        system_content = line[6:].strip()
                        if system_content.startswith('"') and system_content.endswith('"'):
                            system_content = system_content[1:-1]
                        system_lines.append(system_content)
                        in_system = True
                    elif in_system and line and not line.upper().startswith(('FROM', 'PARAMETER', 'TEMPLATE')):
                        # Continue multi-line system prompt
                        if line.startswith('"') and line.endswith('"'):
                            line = line[1:-1]
                        system_lines.append(line)
                    elif line.upper().startswith(('FROM', 'PARAMETER', 'TEMPLATE')):
                        in_system = False
                
                if system_lines:
                    self.system_prompt = '\n'.join(system_lines)
                    self.logger.info(f"Loaded system prompt from {modelfile_path}")
                else:
                    self.logger.warning(f"No SYSTEM instruction found in {modelfile_path}")
                    self.system_prompt = "You are a helpful AI assistant."
            else:
                self.logger.warning(f"Modelfile not found: {modelfile_path}, using default system prompt")
                self.system_prompt = "You are a helpful AI assistant."
                
        except Exception as e:
            self.logger.error(f"Failed to load system prompt from {modelfile_path}: {e}")
            self.system_prompt = "You are a helpful AI assistant."
    
    def _validate_ollama_connection(self) -> None:
        """Validate connection to Ollama API"""
        try:
            response = requests.get(
                f"{self.ollama_endpoint}/api/tags",
                timeout=5.0
            )
            response.raise_for_status()
            self.logger.info("Ollama API connection validated")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama API at {self.ollama_endpoint}: {e}")
    
    def _validate_model_availability(self) -> None:
        """Validate that the specified model is available"""
        try:
            response = requests.get(
                f"{self.ollama_endpoint}/api/tags",
                timeout=5.0
            )
            response.raise_for_status()
            
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if self.model_name not in available_models:
                self.logger.warning(
                    f"Model {self.model_name} not found in available models: {available_models}. "
                    "Attempting to pull model..."
                )
                self._pull_model()
            else:
                self.logger.info(f"Model {self.model_name} is available")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to validate model availability: {e}")
    
    def _pull_model(self) -> None:
        """Pull the specified model from Ollama"""
        try:
            self.logger.info(f"Pulling model {self.model_name}...")
            
            response = requests.post(
                f"{self.ollama_endpoint}/api/pull",
                json={"name": self.model_name},
                timeout=300.0  # 5 minutes for model download
            )
            response.raise_for_status()
            
            self.logger.info(f"Successfully pulled model {self.model_name}")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to pull model {self.model_name}: {e}")
    
    def _setup_logger_connection(self) -> None:
        """Setup connection to logger service"""
        try:
            self._logger_channel = grpc.insecure_channel('localhost:5001')
            self.logger_stub = logger_pb2_grpc.LoggerStub(self._logger_channel)
            self.logger.info("Connected to logger service")
        except Exception as e:
            self.logger.warning(f"Failed to connect to logger service: {e}")
            self.logger_stub = None
    
    def _log_to_dialog(self, dialog_id: str, speaker: str, text: str) -> None:
        """Log message to dialog file via logger service"""
        if not self.logger_stub:
            return
        
        try:
            entry = logger_pb2.DialogEntry(
                dialog_id=dialog_id,
                speaker=speaker,
                text=text,
                timestamp=datetime.now().isoformat()
            )
            self.logger_stub.WriteDialog(entry, timeout=1.0)
        except Exception as e:
            self.logger.warning(f"Failed to log to dialog {dialog_id}: {e}")
    
    def _log_app_event(self, level: str, event: str, details: str) -> None:
        """Log application event via logger service"""
        if not self.logger_stub:
            return
        
        try:
            entry = logger_pb2.LogEntry(
                timestamp=datetime.now().isoformat(),
                level=level,
                service="llm",
                event=event,
                details=details
            )
            self.logger_stub.WriteApp(entry, timeout=1.0)
        except Exception as e:
            self.logger.warning(f"Failed to log app event: {e}")
    
    def Configure(self, request: llm_pb2.LlmConfig, context) -> llm_pb2.ConfigureResponse:
        """Configure the LLM service via gRPC"""
        try:
            # Convert protobuf config to dataclass
            config = LlmConfig(
                port=self.port,
                endpoint=request.endpoint,
                model=request.model,
                modelfile_path=request.modelfile_path,
                request_timeout_ms=request.request_timeout_ms,
                stream_chunk_chars=request.stream_chunk_chars
            )
            
            self.configure(config)
            
            return llm_pb2.ConfigureResponse(
                success=True,
                message=f"LLM service configured with model {self.model_name}"
            )
            
        except Exception as e:
            self.logger.error(f"Configuration failed: {e}")
            return llm_pb2.ConfigureResponse(
                success=False,
                message=f"Configuration failed: {str(e)}"
            )
    
    def Complete(self, request: llm_pb2.UserQuery, context) -> Iterator[llm_pb2.LlmChunk]:
        """Generate streaming response for user query"""
        if not self._configured:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("LLM service not configured")
            return
        
        try:
            self.logger.info(f"Processing query for dialog {request.dialog_id}, turn {request.turn}")
            self._log_app_event("INFO", "query_start", f"dialog_id={request.dialog_id}, turn={request.turn}")
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": request.text,
                "system": self.system_prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            # Make streaming request to Ollama
            response = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json=payload,
                stream=True,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            
            # Process streaming response
            buffer = ""
            full_response = ""  # Collect full response for logging
            first_token_time = None
            
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                try:
                    chunk_data = json.loads(line)
                    
                    if 'response' in chunk_data:
                        token = chunk_data['response']
                        
                        # Track first token timing
                        if first_token_time is None:
                            first_token_time = time.time()
                            self.logger.debug("First token received")
                            self._log_app_event("DEBUG", "first_token", f"dialog_id={request.dialog_id}")
                        
                        buffer += token
                        full_response += token  # Collect for logging
                        
                        # Send chunks when buffer reaches configured size
                        while len(buffer) >= self.stream_chunk_chars:
                            chunk_text = buffer[:self.stream_chunk_chars]
                            buffer = buffer[self.stream_chunk_chars:]
                            
                            yield llm_pb2.LlmChunk(
                                text=chunk_text,
                                eot=False
                            )
                    
                    # Check if generation is complete
                    if chunk_data.get('done', False):
                        # Send any remaining buffer content
                        if buffer:
                            yield llm_pb2.LlmChunk(
                                text=buffer,
                                eot=False
                            )
                        
                        # Log full response to dialog file
                        if full_response.strip():
                            self._log_to_dialog(request.dialog_id, "ASSISTANT", full_response.strip())
                        
                        # Send end-of-turn marker
                        yield llm_pb2.LlmChunk(
                            text="",
                            eot=True
                        )
                        
                        self.logger.info(f"Completed response generation for dialog {request.dialog_id}")
                        self._log_app_event("INFO", "query_complete", 
                                          f"dialog_id={request.dialog_id}, response_length={len(full_response)}")
                        break
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse JSON chunk: {e}")
                    continue
                    
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout for dialog {request.dialog_id}"
            self.logger.error(error_msg)
            self._log_app_event("ERROR", "request_timeout", f"dialog_id={request.dialog_id}")
            self._log_to_dialog(request.dialog_id, "ASSISTANT", "Sorry, I had a problem.")
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("Request timeout")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed for dialog {request.dialog_id}: {e}"
            self.logger.error(error_msg)
            self._log_app_event("ERROR", "request_failed", f"dialog_id={request.dialog_id}, error={str(e)}")
            self._log_to_dialog(request.dialog_id, "ASSISTANT", "Sorry, I had a problem.")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Request failed: {str(e)}")
            
        except Exception as e:
            error_msg = f"Unexpected error for dialog {request.dialog_id}: {e}"
            self.logger.error(error_msg)
            self._log_app_event("ERROR", "unexpected_error", f"dialog_id={request.dialog_id}, error={str(e)}")
            self._log_to_dialog(request.dialog_id, "ASSISTANT", "Sorry, I had a problem.")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
    
    def stop(self, grace_period=5):
        """Stop the service and cleanup connections"""
        try:
            if self._logger_channel:
                self._logger_channel.close()
                self.logger.info("Closed logger service connection")
        except Exception as e:
            self.logger.error(f"Error closing logger connection: {e}")
        
        super().stop(grace_period)


def main():
    """Main entry point for running the LLM service standalone"""
    import sys
    import signal
    from ..config import ConfigManager
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    try:
        config_manager = ConfigManager()
        config = config_manager.load()
        llm_config = config['llm']
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create and configure service
    service = LlmService(llm_config.port)
    
    try:
        service.configure(llm_config)
        service.start()
        
        # Set up signal handling
        def signal_handler(signum, frame):
            logging.info("Received shutdown signal")
            service.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logging.info("LLM service is running. Press Ctrl+C to stop.")
        service.wait_for_termination()
        
    except Exception as e:
        logging.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()