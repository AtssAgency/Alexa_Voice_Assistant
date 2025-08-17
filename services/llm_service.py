#!/usr/bin/env python3
"""
LLM Service - Streaming Language Model Completions via Ollama

FastAPI service on port 5004 that:
- Provides streaming LLM completions using Ollama backend
- Loads model and system prompt from config/Modelfile (authoritative)
- Streams responses to STT service for real-time TTS forwarding
- Manages warmup and configuration hot-reload capabilities
- Maintains minimal state for fast, stateless operation

State: INIT → READY (with transient ERROR for retries)
"""

import os
import sys
import time
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator
import configparser
import re

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn


class LLMState:
    INIT = "INIT"
    READY = "READY"
    ERROR = "ERROR"


# Request/Response Models
class CompleteRequest(BaseModel):
    dialog_id: str
    text: str
    history: Optional[List[Dict[str, str]]] = None
    params: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    state: str


class StateResponse(BaseModel):
    state: str


class WarmupRequest(BaseModel):
    text: Optional[str] = None


class LLMService:
    def __init__(self):
        self.state = LLMState.INIT
        self.config: Optional[configparser.ConfigParser] = None
        
        # Service URLs
        self.logger_url = "http://127.0.0.1:5000"
        self.ollama_base_url = "http://127.0.0.1:11434"
        
        # Configuration
        self.port = 5004
        self.request_timeout_s = 120
        self.warmup_enabled = True
        self.warmup_text = "ok"
        
        # Default sampling parameters
        self.temperature = 0.6
        self.top_p = 0.9
        self.top_k = 40
        self.repeat_penalty = 1.05
        self.max_tokens = 512
        
        # Modelfile configuration (authoritative)
        self.model = "llama3.1:8b-instruct-q4_K_M"
        self.system_prompt = "You are a helpful AI voice assistant. You provide concise, friendly responses to user questions. Keep your answers brief and conversational, as they will be spoken aloud. Avoid using markdown formatting or special characters in your responses."
        
        # Metrics tracking
        self.active_streams = 0
    
    def load_config(self) -> bool:
        """Load configuration from config files"""
        try:
            # Load config.ini first for basic settings
            config_path = Path("config/config.ini")
            if config_path.exists():
                self.config = configparser.ConfigParser()
                self.config.read(config_path)
                
                # Load LLM section
                if 'llm' in self.config:
                    llm_section = self.config['llm']
                    self.port = llm_section.getint('port', self.port)
                    self.ollama_base_url = llm_section.get('ollama_base_url', self.ollama_base_url)
                    self.request_timeout_s = llm_section.getint('request_timeout_s', self.request_timeout_s)
                    self.warmup_enabled = llm_section.getboolean('warmup_enabled', self.warmup_enabled)
                    self.warmup_text = llm_section.get('warmup_text', self.warmup_text)
                    
                    # Load default sampling parameters
                    self.temperature = llm_section.getfloat('temperature', self.temperature)
                    self.top_p = llm_section.getfloat('top_p', self.top_p)
                    self.top_k = llm_section.getint('top_k', self.top_k)
                    self.repeat_penalty = llm_section.getfloat('repeat_penalty', self.repeat_penalty)
                    self.max_tokens = llm_section.getint('max_tokens', self.max_tokens)
            
            # Load Modelfile (authoritative for model and system prompt)
            if not self.load_modelfile():
                return False
                
            self.log("info", "Configuration loaded successfully", "config_loaded")
            return True
            
        except Exception as e:
            self.log("error", f"Failed to load config: {e}")
            return False
    
    def load_modelfile(self) -> bool:
        """Load model and system prompt from config/Modelfile"""
        try:
            modelfile_path = Path("config/Modelfile")
            if not modelfile_path.exists():
                self.log("warning", "config/Modelfile not found, using defaults")
                return True
            
            modelfile_content = modelfile_path.read_text().strip()
            if not modelfile_content:
                self.log("warning", "config/Modelfile is empty, using defaults")
                return True
            
            # Parse Modelfile format
            for line in modelfile_content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse FROM directive (model specification)
                if line.startswith('FROM '):
                    self.model = line[5:].strip()
                
                # Parse SYSTEM directive (system prompt)
                elif line.startswith('SYSTEM '):
                    # Handle quoted system prompts
                    system_part = line[7:].strip()
                    if system_part.startswith('"') and system_part.endswith('"'):
                        self.system_prompt = system_part[1:-1]
                    else:
                        self.system_prompt = system_part
                
                # Parse PARAMETER directives (sampling parameters)
                elif line.startswith('PARAMETER '):
                    param_part = line[10:].strip()
                    if ' ' in param_part:
                        param_name, param_value = param_part.split(' ', 1)
                        try:
                            if param_name == 'temperature':
                                self.temperature = float(param_value)
                            elif param_name == 'top_p':
                                self.top_p = float(param_value)
                            elif param_name == 'top_k':
                                self.top_k = int(param_value)
                            elif param_name == 'repeat_penalty':
                                self.repeat_penalty = float(param_value)
                            elif param_name == 'max_tokens':
                                self.max_tokens = int(param_value)
                        except ValueError:
                            self.log("warning", f"Invalid parameter value: {param_name}={param_value}")
            
            self.log("info", f"Modelfile parsed: model={self.model}", "modelfile_parsed")
            return True
            
        except Exception as e:
            self.log("error", f"Failed to parse Modelfile: {e}")
            return False
    
    def log(self, level: str, message: str, event: str = None, extra: Dict[str, Any] = None):
        """Send log message to Logger service"""
        try:
            # Add human-readable timestamp (dd-mm-yy HH:MM:SS format)
            from datetime import datetime
            timestamp = datetime.now().strftime("%d-%m-%y %H:%M:%S")
            
            payload = {
                "svc": "LLM",
                "level": level,
                "message": message,
                "ts_ms": timestamp
            }
            if event:
                payload["event"] = event
            if extra:
                payload["extra"] = extra
                
            requests.post(f"{self.logger_url}/log", json=payload, timeout=2.0)
        except Exception:
            # Fallback to stdout if logger unavailable
            print(f"LLM       {level.upper():<6}= {message}")
    
    def send_metric(self, metric: str, value: float, dialog_id: str = None):
        """Send metric to Logger service"""
        try:
            payload = {
                "svc": "LLM",
                "metric": metric,
                "value": value
            }
            if dialog_id:
                payload["dialog_id"] = dialog_id
                
            requests.post(f"{self.logger_url}/metrics", json=payload, timeout=2.0)
        except Exception:
            pass
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def ensure_model_available(self) -> bool:
        """Ensure the specified model is available in Ollama"""
        try:
            # Check if model exists
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10.0)
            if response.status_code != 200:
                return False
            
            available_models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in available_models]
            
            # Check if our model is available
            if self.model in model_names:
                return True
            
            # Try to pull the model if not available
            self.log("info", f"Model {self.model} not found, attempting to pull...")
            
            pull_response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json={"name": self.model},
                timeout=300.0  # Model pulling can take a long time
            )
            
            return pull_response.status_code == 200
            
        except Exception as e:
            self.log("error", f"Failed to ensure model availability: {e}")
            return False
    
    async def warmup_model(self) -> bool:
        """Perform model warmup to pre-allocate resources"""
        if not self.warmup_enabled:
            return True
            
        try:
            self.log("info", "Starting model warmup")
            
            # Simple warmup request
            warmup_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.warmup_text}
            ]
            
            payload = {
                "model": self.model,
                "messages": warmup_messages,
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "max_tokens": 5
                }
            }
            
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=30.0
            )
            
            if response.status_code == 200:
                # Consume the stream to complete warmup
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                warmup_time = (time.time() - start_time) * 1000
                self.log("info", f"Model warmup completed in {warmup_time:.1f}ms")
                self.send_metric("warmup_ms", warmup_time)
                return True
            else:
                self.log("error", f"Model warmup failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.log("error", f"Model warmup error: {e}")
            return False
    
    def build_chat_messages(self, user_text: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Build chat messages with system prompt and history"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history if provided
        if history:
            messages.extend(history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_text})
        
        return messages
    
    async def stream_completion(self, request: CompleteRequest) -> AsyncGenerator[str, None]:
        """Stream completion from Ollama"""
        start_time = time.time()
        token_count = 0
        first_token_time = None
        
        try:
            self.active_streams += 1
            self.log("info", f"Starting completion stream for dialog {request.dialog_id}", "stream_started")
            
            # Build messages
            messages = self.build_chat_messages(request.text, request.history)
            
            # Build request parameters
            options = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repeat_penalty": self.repeat_penalty,
                "num_predict": self.max_tokens
            }
            
            # Override with request parameters if provided
            if request.params:
                options.update(request.params)
            
            # Prepare Ollama request
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": options
            }
            
            # Start streaming request to Ollama
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.request_timeout_s
            )
            
            if response.status_code != 200:
                self.log("error", f"Ollama request failed: {response.status_code}")
                yield f"Error: Failed to get response from language model"
                return
            
            # Stream response chunks
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Extract content from response
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            
                            if content:
                                # Track first token time
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    ttft = (first_token_time - start_time) * 1000
                                    self.send_metric("ttft_ms", ttft, request.dialog_id)
                                
                                token_count += 1
                                yield content
                        
                        # Check if done
                        if data.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Calculate and log metrics
            total_time = time.time() - start_time
            if token_count > 0 and total_time > 0:
                tokens_per_sec = token_count / total_time
                self.send_metric("tokens_total", float(token_count), request.dialog_id)
                self.send_metric("tokens_sec", tokens_per_sec, request.dialog_id)
            
            self.log("info", f"Completion stream ended: {token_count} tokens in {total_time:.2f}s", "stream_ended")
            
        except Exception as e:
            self.log("error", f"Streaming error: {e}")
            yield f"Error: {str(e)}"
        finally:
            self.active_streams -= 1
    
    async def startup(self):
        """Service startup logic"""
        try:
            self.log("info", "LLM service starting", "service_start")
            
            # Load configuration
            if not self.load_config():
                self.state = LLMState.ERROR
                return False
            
            # Check Ollama connection
            if not self.check_ollama_connection():
                self.log("error", "Ollama not available")
                self.state = LLMState.ERROR
                return False
            
            self.log("info", "Connected to Ollama", "ollama_connected")
            
            # Ensure model is available
            if not self.ensure_model_available():
                self.log("error", f"Model {self.model} not available")
                self.state = LLMState.ERROR
                return False
            
            # Perform warmup if enabled
            if not await self.warmup_model():
                self.log("warning", "Model warmup failed, continuing anyway")
            
            self.state = LLMState.READY
            self.log("info", "LLM service ready", "service_ready")
            return True
            
        except Exception as e:
            self.log("error", f"LLM startup failed: {e}")
            self.state = LLMState.ERROR
            return False
    
    async def shutdown(self):
        """Service shutdown logic"""
        self.log("info", "LLM service shutting down")
        
        # Wait for active streams to complete
        if self.active_streams > 0:
            self.log("info", f"Waiting for {self.active_streams} active streams to complete")
            # Give streams time to complete gracefully
            await asyncio.sleep(2.0)


# Global LLM instance
llm_service = LLMService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await llm_service.startup()
    yield
    # Shutdown
    await llm_service.shutdown()


# FastAPI app
app = FastAPI(
    title="LLM Service",
    description="Streaming Language Model Completions via Ollama",
    version="1.0",
    lifespan=lifespan
)


@app.post("/complete")
async def complete(request: CompleteRequest):
    """Stream completion from language model"""
    if llm_service.state != LLMState.READY:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready, state: {llm_service.state}"
        )
    
    # Return streaming response
    return StreamingResponse(
        llm_service.stream_completion(request),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.post("/warmup")
async def warmup(request: WarmupRequest = WarmupRequest()):
    """Force model warmup"""
    if llm_service.state == LLMState.ERROR:
        raise HTTPException(status_code=503, detail="Service in error state")
    
    # Override warmup text if provided
    if request.text:
        llm_service.warmup_text = request.text
    
    if await llm_service.warmup_model():
        llm_service.state = LLMState.READY
        return {"status": "ok", "message": "Model warmup completed"}
    else:
        raise HTTPException(status_code=500, detail="Warmup failed")


@app.post("/reload")
async def reload_config():
    """Reload configuration from files"""
    if llm_service.load_config():
        return {
            "status": "ok",
            "config": {
                "model": llm_service.model,
                "temperature": llm_service.temperature,
                "top_p": llm_service.top_p,
                "top_k": llm_service.top_k,
                "max_tokens": llm_service.max_tokens
            }
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload configuration")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "ok" if llm_service.state == LLMState.READY else "error"
    return HealthResponse(status=status, state=llm_service.state)


@app.get("/state")
async def get_state():
    """Get current service state"""
    return StateResponse(state=llm_service.state)


def main():
    """Entry point when run as module"""
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=llm_service.port,
        log_level="error",  # Suppress uvicorn logs to keep console clean
        access_log=False
    )


if __name__ == "__main__":
    main()
