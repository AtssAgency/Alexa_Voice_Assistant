#!/usr/bin/env python3
"""
RMS Service - Dynamic Background Noise Monitoring

FastAPI service on port 5006 that:
- Continuously samples ambient audio for noise level monitoring
- Computes rolling RMS averages and noise floor metrics
- Serves /current-rms endpoint for KWD and STT adaptive thresholds
- Lightweight, passive monitoring with automatic recovery

State: INIT → READY → ERROR → READY (auto-retry)
"""

import os
import sys
import time
import threading
import queue
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import configparser
from collections import deque

import requests
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn


class RMSState:
    INIT = "INIT"
    READY = "READY"
    ERROR = "ERROR"


# Response Models
class CurrentRMSResponse(BaseModel):
    rms_dbfs: float
    noise_floor_db: float
    last_updated_ms: int


class HealthResponse(BaseModel):
    status: str
    state: str


class StateResponse(BaseModel):
    state: str


class RMSService:
    def __init__(self):
        self.state = RMSState.INIT
        self.config: Optional[configparser.ConfigParser] = None
        
        # Service URLs
        self.logger_url = "http://127.0.0.1:5000"
        
        # Configuration
        self.port = 5006
        self.sample_rate = 16000
        self.frame_ms = 50
        self.window_sec = 30
        self.ema_alpha = 0.2
        self.noise_floor_mode = "p05"  # "min", "p05", "p10"
        self.device = "default"
        self.shared_mode = True
        self.stats_interval_sec = 5
        
        # Audio sampling
        self.audio_stream = None
        self.audio_device = None
        self.frame_samples = 0
        
        # Data structures
        self.ring_buffer = deque()
        self.ring_buffer_max_size = 0
        self.ema_dbfs = -60.0  # Start with reasonable default
        self.noise_floor_db = -70.0
        self.last_updated_ms = 0
        
        # Threading
        self.sampler_thread: Optional[threading.Thread] = None
        self.stats_thread: Optional[threading.Thread] = None
        self.running = False
        self.data_lock = threading.Lock()
        
        # Error handling
        self.retry_count = 0
        self.max_retries = 5
        self.retry_backoff_base = 1.0
    
    def load_config(self) -> bool:
        """Load configuration from config files"""
        try:
            config_path = Path("config/config.ini")
            if config_path.exists():
                self.config = configparser.ConfigParser()
                self.config.read(config_path)
                
                # Load RMS section
                if 'rms' in self.config:
                    rms_section = self.config['rms']
                    self.port = rms_section.getint('port', self.port)
                    self.sample_rate = rms_section.getint('sample_rate', self.sample_rate)
                    self.frame_ms = rms_section.getint('frame_ms', self.frame_ms)
                    self.window_sec = rms_section.getint('window_sec', self.window_sec)
                    self.ema_alpha = rms_section.getfloat('ema_alpha', self.ema_alpha)
                    self.noise_floor_mode = rms_section.get('noise_floor_mode', self.noise_floor_mode)
                    self.device = rms_section.get('device', self.device)
                    self.shared_mode = rms_section.getboolean('shared_mode', self.shared_mode)
                    self.stats_interval_sec = rms_section.getint('stats_interval_sec', self.stats_interval_sec)
                
                # Load dependencies section
                if 'deps' in self.config:
                    deps_section = self.config['deps']
                    self.logger_url = deps_section.get('logger_url', self.logger_url)
            
            # Calculate derived values
            self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
            frames_per_window = int(self.window_sec * 1000 / self.frame_ms)
            self.ring_buffer_max_size = frames_per_window
            
            self.log("info", "Configuration loaded successfully", "config_loaded")
            return True
            
        except Exception as e:
            self.log("error", f"Failed to load config: {e}")
            return False
    
    def log(self, level: str, message: str, event: str = None, extra: Dict[str, Any] = None):
        """Send log message to Logger service"""
        try:
            payload = {
                "svc": "RMS",
                "level": level,
                "message": message
            }
            if event:
                payload["event"] = event
            if extra:
                payload["extra"] = extra
                
            requests.post(f"{self.logger_url}/log", json=payload, timeout=2.0)
        except Exception:
            # Fallback to stdout if logger unavailable
            print(f"RMS       {level.upper():<6}= {message}")
    
    def send_metric(self, metric: str, value: float):
        """Send metric to Logger service"""
        try:
            payload = {
                "svc": "RMS",
                "metric": metric,
                "value": value
            }
            requests.post(f"{self.logger_url}/metrics", json=payload, timeout=2.0)
        except Exception:
            pass
    
    def initialize_audio_device(self) -> bool:
        """Initialize audio input device"""
        try:
            # Query available devices
            devices = sd.query_devices()
            
            # Find input device
            if self.device == "default":
                device_id = sd.default.device[0]  # Input device
            else:
                # Try to find device by name or index
                device_id = None
                try:
                    device_id = int(self.device)  # Try as index
                except ValueError:
                    # Search by name
                    for i, device_info in enumerate(devices):
                        if device_info['max_input_channels'] > 0 and self.device.lower() in device_info['name'].lower():
                            device_id = i
                            break
                
                if device_id is None:
                    self.log("warning", f"Device '{self.device}' not found, using default")
                    device_id = sd.default.device[0]
            
            self.audio_device = device_id
            device_info = sd.query_devices(device_id)
            
            # Validate device supports required sample rate
            if device_info['max_input_channels'] == 0:
                self.log("error", "Selected device has no input channels")
                return False
            
            self.log("info", f"Audio device initialized: {device_info['name']}", "audio_initialized")
            return True
            
        except Exception as e:
            self.log("error", f"Failed to initialize audio device: {e}")
            return False
    
    def calculate_rms_dbfs(self, audio_data: np.ndarray) -> float:
        """Calculate RMS in dBFS"""
        try:
            # Ensure float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Calculate RMS
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Avoid log(0)
            if rms <= 0:
                return -100.0  # Very quiet
            
            # Convert to dBFS (full scale = 1.0)
            dbfs = 20.0 * math.log10(rms)
            
            # Clamp reasonable range
            return max(-100.0, min(0.0, dbfs))
            
        except Exception as e:
            self.log("error", f"RMS calculation error: {e}")
            return -100.0
    
    def calculate_noise_floor(self) -> float:
        """Calculate noise floor from ring buffer"""
        if not self.ring_buffer:
            return -100.0
        
        values = list(self.ring_buffer)
        
        try:
            if self.noise_floor_mode == "min":
                return min(values)
            elif self.noise_floor_mode == "p05":
                return np.percentile(values, 5)
            elif self.noise_floor_mode == "p10":
                return np.percentile(values, 10)
            else:
                return min(values)
        except Exception:
            return -100.0
    
    def update_metrics(self, dbfs: float):
        """Update EMA and ring buffer metrics"""
        current_time_ms = int(time.time() * 1000)
        
        with self.data_lock:
            # Update EMA
            self.ema_dbfs = self.ema_alpha * dbfs + (1 - self.ema_alpha) * self.ema_dbfs
            
            # Update ring buffer
            self.ring_buffer.append(dbfs)
            while len(self.ring_buffer) > self.ring_buffer_max_size:
                self.ring_buffer.popleft()
            
            # Update noise floor
            self.noise_floor_db = self.calculate_noise_floor()
            
            # Update timestamp
            self.last_updated_ms = current_time_ms
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            self.log("warning", f"Audio callback status: {status}")
        
        try:
            # Take first channel if multichannel
            if indata.ndim > 1:
                audio_data = indata[:, 0]
            else:
                audio_data = indata
            
            # Calculate RMS dBFS
            dbfs = self.calculate_rms_dbfs(audio_data)
            
            # Update metrics
            self.update_metrics(dbfs)
            
        except Exception as e:
            self.log("error", f"Audio callback error: {e}")
    
    def sampler_worker(self):
        """Background sampler thread"""
        while self.running:
            try:
                # Create audio stream
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32,
                    blocksize=self.frame_samples,
                    device=self.audio_device,
                    callback=self.audio_callback
                )
                
                with stream:
                    self.log("info", "Audio sampler started", "sampler_started")
                    self.retry_count = 0
                    
                    # Keep running while stream is active
                    while self.running and stream.active:
                        time.sleep(0.1)
                
                if self.running:
                    self.log("warning", "Audio stream ended unexpectedly")
                    
            except Exception as e:
                self.log("error", f"Sampler error: {e}", "sampler_error")
                
                if self.running:
                    # Set error state
                    self.state = RMSState.ERROR
                    
                    # Exponential backoff retry
                    self.retry_count += 1
                    if self.retry_count <= self.max_retries:
                        backoff_time = self.retry_backoff_base * (2 ** (self.retry_count - 1))
                        self.log("info", f"Retrying in {backoff_time}s (attempt {self.retry_count})")
                        
                        time.sleep(backoff_time)
                        
                        # Try to recover
                        if self.initialize_audio_device():
                            self.state = RMSState.READY
                    else:
                        self.log("error", "Max retries exceeded, staying in error state")
                        break
    
    def stats_worker(self):
        """Background stats reporting thread"""
        last_stats_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            if current_time - last_stats_time >= self.stats_interval_sec:
                try:
                    with self.data_lock:
                        rms_dbfs = self.ema_dbfs
                        noise_floor = self.noise_floor_db
                    
                    # Send metrics to logger
                    self.send_metric("rms_dbfs", rms_dbfs)
                    self.send_metric("noise_floor_db", noise_floor)
                    
                    # Log snapshot
                    self.log("debug", f"RMS snapshot: {rms_dbfs:.1f} dBFS, floor: {noise_floor:.1f} dB", "rms_snapshot")
                    
                    last_stats_time = current_time
                    
                except Exception as e:
                    self.log("error", f"Stats worker error: {e}")
            
            time.sleep(1.0)  # Check every second
    
    def start_sampler(self):
        """Start background sampling"""
        if self.running:
            return
        
        self.running = True
        
        # Start sampler thread
        self.sampler_thread = threading.Thread(target=self.sampler_worker, daemon=True)
        self.sampler_thread.start()
        
        # Start stats thread
        self.stats_thread = threading.Thread(target=self.stats_worker, daemon=True)
        self.stats_thread.start()
        
        self.log("info", "Background threads started")
    
    def stop_sampler(self):
        """Stop background sampling"""
        self.running = False
        
        # Wait for threads to finish
        if self.sampler_thread and self.sampler_thread.is_alive():
            self.sampler_thread.join(timeout=2.0)
        
        if self.stats_thread and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=2.0)
        
        self.log("info", "Background threads stopped")
    
    def get_current_rms(self) -> CurrentRMSResponse:
        """Get current RMS metrics"""
        with self.data_lock:
            return CurrentRMSResponse(
                rms_dbfs=round(self.ema_dbfs, 1),
                noise_floor_db=round(self.noise_floor_db, 1),
                last_updated_ms=self.last_updated_ms
            )
    
    async def startup(self):
        """Service startup logic"""
        try:
            self.log("info", "RMS service starting", "service_start")
            
            # Load configuration
            if not self.load_config():
                self.state = RMSState.ERROR
                return False
            
            # Initialize audio device
            if not self.initialize_audio_device():
                self.state = RMSState.ERROR
                return False
            
            # Start sampler
            self.start_sampler()
            
            # Wait a moment for initial samples
            time.sleep(0.5)
            
            self.state = RMSState.READY
            self.log("info", "RMS service ready", "service_ready")
            return True
            
        except Exception as e:
            self.log("error", f"RMS startup failed: {e}")
            self.state = RMSState.ERROR
            return False
    
    async def shutdown(self):
        """Service shutdown logic"""
        self.log("info", "RMS service shutting down")
        self.stop_sampler()


# Global RMS instance
rms_service = RMSService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await rms_service.startup()
    yield
    # Shutdown
    await rms_service.shutdown()


# FastAPI app
app = FastAPI(
    title="RMS Service",
    description="Dynamic Background Noise Monitoring",
    version="1.0",
    lifespan=lifespan
)


@app.get("/current-rms", response_model=CurrentRMSResponse)
async def get_current_rms():
    """Get current RMS metrics"""
    return rms_service.get_current_rms()


@app.post("/reload")
async def reload_config():
    """Reload configuration from files"""
    if rms_service.load_config():
        return {
            "status": "ok",
            "config": {
                "sample_rate": rms_service.sample_rate,
                "window_sec": rms_service.window_sec,
                "noise_floor_mode": rms_service.noise_floor_mode,
                "device": rms_service.device
            }
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload configuration")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Only report OK if sampler is running and recent data available
    current_time_ms = int(time.time() * 1000)
    data_fresh = (current_time_ms - rms_service.last_updated_ms) < 5000  # 5 second threshold
    
    if rms_service.state == RMSState.READY and rms_service.running and data_fresh:
        status = "ok"
    else:
        status = "error"
    
    return HealthResponse(status=status, state=rms_service.state)


@app.get("/state", response_model=StateResponse)
async def get_state():
    """Get current service state"""
    return StateResponse(state=rms_service.state)


def main():
    """Entry point when run as module"""
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=rms_service.port,
        log_level="error",  # Suppress uvicorn logs to keep console clean
        access_log=False
    )


if __name__ == "__main__":
    main()
