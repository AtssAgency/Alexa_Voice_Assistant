#!/usr/bin/env python3
"""
KWD Service - Wake Word Detection using openWakeWord

FastAPI service on port 5002 that:
- Continuously listens for "Alexa" wake word using openWakeWord
- Adapts sensitivity based on ambient noise from RMS service
- Triggers TTS confirmation and STT dialog on detection
- Manages state: INIT → READY → ACTIVE → TRIGGERED → SLEEP
- Handles loader greeting and STT re-arming

State flow: INIT → READY → ACTIVE → TRIGGERED → SLEEP → ACTIVE (cycle)
"""

import os
import sys
import time
import threading
import random
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import configparser

import numpy as np
import sounddevice as sd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

try:
    import openwakeword
    from openwakeword.model import Model
except ImportError:
    print("ERROR: openwakeword not installed. Run: pip install openwakeword")
    sys.exit(1)


class KWDState:
    INIT = "INIT"
    READY = "READY"
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    SLEEP = "SLEEP"


# Request/Response Models
class StartRequest(BaseModel):
    dialog_id: Optional[str] = None


class StopRequest(BaseModel):
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    state: str


class StateResponse(BaseModel):
    state: str


class TTSSpeakRequest(BaseModel):
    text: str
    dialog_id: Optional[str] = None


class STTStartRequest(BaseModel):
    dialog_id: str


class KWDService:
    def __init__(self):
        self.state = KWDState.INIT
        self.config: Optional[configparser.ConfigParser] = None
        
        # Service URLs
        self.logger_url = "http://127.0.0.1:5000"
        self.rms_url = "http://127.0.0.1:5006"
        self.tts_url = "http://127.0.0.1:5005"
        self.stt_url = "http://127.0.0.1:5003"
        
        # Configuration
        self.port = 5002
        self.model_path = "models/alexa_v0.1.onnx"
        self.base_rms_threshold = 0.050
        self.cooldown_ms = 1000
        self.yes_phrases = ["Yes?", "Yes Master?", "What's up?", "I'm listening", "Yo", "Here!"]
        self.greeting_text = "Hi Master. Ready to serve."
        
        # Adaptive thresholding
        self.quiet_dbfs = -60
        self.noisy_dbfs = -35
        self.quiet_factor = 0.80
        self.noisy_factor = 1.25
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1280  # 80ms at 16kHz
        
        # Wake word engine
        self.detector: Optional[Model] = None
        self.detection_thread: Optional[threading.Thread] = None
        self.audio_stream: Optional[sd.InputStream] = None
        self.stop_event = threading.Event()
        
        # State management
        self.current_threshold = self.base_rms_threshold
        self.last_detection_time = 0
        self.detection_lock = threading.Lock()
        
        # Audio buffer
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
    
    def load_config(self) -> bool:
        """Load configuration from config/config.ini"""
        try:
            config_path = Path("config/config.ini")
            if not config_path.exists():
                self.log("warning", "config/config.ini not found, using defaults")
                return True
                
            self.config = configparser.ConfigParser()
            self.config.read(config_path)
            
            # Load KWD section
            if 'kwd' in self.config:
                kwd_section = self.config['kwd']
                self.model_path = kwd_section.get('model_path', self.model_path)
                self.base_rms_threshold = kwd_section.getfloat('base_rms_threshold', self.base_rms_threshold)
                self.cooldown_ms = kwd_section.getint('cooldown_ms', self.cooldown_ms)
                
                yes_phrases_str = kwd_section.get('yes_phrases', ', '.join(self.yes_phrases))
                self.yes_phrases = [p.strip() for p in yes_phrases_str.split(',')]
                
                self.greeting_text = kwd_section.get('greeting_text', self.greeting_text)
                self.port = kwd_section.getint('port', self.port)
            
            # Load adaptive section
            if 'kwd.adaptive' in self.config:
                adaptive_section = self.config['kwd.adaptive']
                self.quiet_dbfs = adaptive_section.getfloat('quiet_dbfs', self.quiet_dbfs)
                self.noisy_dbfs = adaptive_section.getfloat('noisy_dbfs', self.noisy_dbfs)
                self.quiet_factor = adaptive_section.getfloat('quiet_factor', self.quiet_factor)
                self.noisy_factor = adaptive_section.getfloat('noisy_factor', self.noisy_factor)
            
            # Load dependency URLs
            if 'kwd.deps' in self.config:
                deps_section = self.config['kwd.deps']
                self.logger_url = deps_section.get('logger_url', self.logger_url)
                self.tts_url = deps_section.get('tts_url', self.tts_url)
                self.stt_url = deps_section.get('stt_url', self.stt_url)
                self.rms_url = deps_section.get('rms_url', self.rms_url)
            
            return True
        except Exception as e:
            self.log("error", f"Failed to load config: {e}")
            return False
    
    def log(self, level: str, message: str, event: str = None, extra: Dict[str, Any] = None):
        """Send log message to Logger service"""
        try:
            payload = {
                "svc": "KWD",
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
            print(f"KWD       {level.upper():<6}= {message}")
    
    def get_current_rms(self) -> Optional[float]:
        """Get current RMS noise level from RMS service"""
        try:
            # rms_url from config already includes the endpoint (/current-rms)
            response = requests.get(self.rms_url, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("rms_dbfs")
        except Exception as e:
            self.log("debug", f"Failed to get RMS: {e}")
        return None
    
    def calculate_adaptive_threshold(self, noise_dbfs: float) -> float:
        """Calculate adaptive threshold based on ambient noise"""
        if noise_dbfs <= self.quiet_dbfs:
            factor = self.quiet_factor
        elif noise_dbfs >= self.noisy_dbfs:
            factor = self.noisy_factor
        else:
            # Linear interpolation between quiet and noisy
            ratio = (noise_dbfs - self.quiet_dbfs) / (self.noisy_dbfs - self.quiet_dbfs)
            factor = self.quiet_factor + ratio * (self.noisy_factor - self.quiet_factor)
        
        return self.base_rms_threshold * factor
    
    def update_threshold_from_rms(self):
        """Update detection threshold based on current RMS"""
        rms_dbfs = self.get_current_rms()
        if rms_dbfs is not None:
            new_threshold = self.calculate_adaptive_threshold(rms_dbfs)
            if abs(new_threshold - self.current_threshold) > 0.005:  # Avoid noise
                self.current_threshold = new_threshold
                self.log("debug", f"Adaptive threshold updated: {self.current_threshold:.3f} (RMS: {rms_dbfs:.1f} dBFS)")
    
    def load_wake_word_model(self) -> bool:
        """Load the openWakeWord model"""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                self.log("error", f"Wake word model not found: {self.model_path}")
                return False
            
            self.log("info", f"Loading wake word model: {self.model_path}")
            # Initialize the detector with the single ONNX model
            self.detector = Model(wakeword_model_paths=[str(model_file)])
            
            self.log("info", "Wake word model loaded successfully", "model_loaded")
            return True
            
        except Exception as e:
            self.log("error", f"Failed to load wake word model: {e}")
            return False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            self.log("warning", f"Audio callback status: {status}")
            
        # Convert to float32 and add to buffer
        audio_data = indata[:, 0].astype(np.float32)
        
        with self.buffer_lock:
            self.audio_buffer.extend(audio_data)
            
            # Keep buffer size reasonable (2 seconds max)
            max_buffer_size = self.sample_rate * 2
            if len(self.audio_buffer) > max_buffer_size:
                self.audio_buffer = self.audio_buffer[-max_buffer_size:]
    
    def start_audio_stream(self) -> bool:
        """Start audio input stream"""
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            self.audio_stream.start()
            self.log("info", "Audio stream started")
            return True
        except Exception as e:
            self.log("error", f"Failed to start audio stream: {e}")
            return False
    
    def stop_audio_stream(self):
        """Stop audio input stream"""
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                self.log("info", "Audio stream stopped")
            except Exception as e:
                self.log("warning", f"Error stopping audio stream: {e}")
    
    def detection_worker(self):
        """Main detection thread worker"""
        self.log("info", "Detection worker started")
        
        while not self.stop_event.is_set():
            try:
                if self.state != KWDState.ACTIVE:
                    time.sleep(0.1)
                    continue
                
                # Get audio data from buffer
                with self.buffer_lock:
                    if len(self.audio_buffer) < self.chunk_size:
                        time.sleep(0.01)
                        continue
                    
                    # Get chunk and remove from buffer
                    audio_chunk = np.array(self.audio_buffer[:self.chunk_size])
                    self.audio_buffer = self.audio_buffer[self.chunk_size:]
                
                # Process with wake word detector
                if self.detector:
                    predictions = self.detector.predict(audio_chunk)
                    
                    # Check for wake word detection
                    for model_name, score in predictions.items():
                        if score >= self.current_threshold:
                            self.handle_wake_detection(score)
                            break
                
                # Update adaptive threshold periodically
                if time.time() % 5 < 0.1:  # Every ~5 seconds
                    self.update_threshold_from_rms()
                    
            except Exception as e:
                self.log("error", f"Detection worker error: {e}")
                time.sleep(0.1)
        
        self.log("info", "Detection worker stopped")
    
    def handle_wake_detection(self, confidence: float):
        """Handle wake word detection"""
        current_time = time.time()
        
        with self.detection_lock:
            # Check cooldown
            if (current_time - self.last_detection_time) * 1000 < self.cooldown_ms:
                return
            
            self.last_detection_time = current_time
        
        # State transition: ACTIVE → TRIGGERED → SLEEP
        self.state = KWDState.TRIGGERED
        self.log("info", f"Wake word detected (confidence {confidence:.3f})", "wake_detected", 
                {"confidence": confidence, "threshold": self.current_threshold})
        
        # Generate dialog ID
        dialog_id = f"dialog_{uuid.uuid4().hex[:8]}"
        
        # Immediately transition to SLEEP to prevent double-triggering
        self.state = KWDState.SLEEP
        
        # Start cooldown timer (will transition back to ACTIVE)
        threading.Thread(target=self.cooldown_timer, daemon=True).start()
        
        # Trigger side effects asynchronously
        self.trigger_tts_confirmation(dialog_id)
        self.trigger_stt_dialog(dialog_id)
    
    def cooldown_timer(self):
        """Cooldown timer to transition from SLEEP back to ACTIVE"""
        time.sleep(self.cooldown_ms / 1000.0)
        if self.state == KWDState.SLEEP:
            self.state = KWDState.ACTIVE
            self.log("debug", "Cooldown completed, re-armed to ACTIVE")
    
    def trigger_tts_confirmation(self, dialog_id: str):
        """Send confirmation phrase to TTS"""
        try:
            phrase = random.choice(self.yes_phrases)
            payload = TTSSpeakRequest(text=phrase, dialog_id=dialog_id)
            
            # tts_url from config already includes the endpoint (/speak)
            response = requests.post(self.tts_url, 
                                   json=payload.dict(), timeout=5.0)
            
            if response.status_code == 200:
                self.log("debug", f"TTS confirmation sent: '{phrase}'", "tts_called")
            else:
                self.log("warning", f"TTS confirmation failed: {response.status_code}")
                
        except Exception as e:
            self.log("warning", f"TTS confirmation error: {e}")
    
    def trigger_stt_dialog(self, dialog_id: str):
        """Start STT dialog session"""
        try:
            payload = STTStartRequest(dialog_id=dialog_id)
            
            # stt_url from config already includes the endpoint (/start)
            response = requests.post(self.stt_url, 
                                   json=payload.dict(), timeout=5.0)
            
            if response.status_code == 200:
                self.log("info", f"STT dialog started: {dialog_id}", "stt_started")
            else:
                self.log("warning", f"STT start failed: {response.status_code}")
                
        except Exception as e:
            self.log("warning", f"STT start error: {e}")
    
    def start_detection(self) -> bool:
        """Start wake word detection"""
        if self.state == KWDState.INIT:
            return False
            
        if self.detection_thread and self.detection_thread.is_alive():
            return True  # Already running
            
        try:
            # Start audio stream
            if not self.start_audio_stream():
                return False
            
            # Start detection thread
            self.stop_event.clear()
            self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
            self.detection_thread.start()
            
            self.state = KWDState.ACTIVE
            self.log("info", "Wake word detection started", "kwd_started")
            return True
            
        except Exception as e:
            self.log("error", f"Failed to start detection: {e}")
            return False
    
    def stop_detection(self):
        """Stop wake word detection"""
        self.state = KWDState.SLEEP
        self.stop_event.set()
        
        # Stop audio stream
        self.stop_audio_stream()
        
        # Wait for thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        self.log("info", "Wake word detection stopped", "kwd_stopped")
    
    def play_greeting_and_activate(self):
        """Play system ready greeting and transition to ACTIVE"""
        try:
            # Send greeting to TTS
            payload = TTSSpeakRequest(text=self.greeting_text)
            # tts_url from config already includes the endpoint (/speak)
            response = requests.post(self.tts_url, 
                                   json=payload.dict(), timeout=10.0)
            
            if response.status_code == 200:
                self.log("info", f"Greeting played: '{self.greeting_text}'", "greeting_played")
                
                # Wait a moment for greeting to start, then activate
                time.sleep(0.5)
                self.start_detection()
            else:
                self.log("warning", f"Greeting TTS failed: {response.status_code}")
                # Still activate even if greeting fails
                self.start_detection()
                
        except Exception as e:
            self.log("warning", f"Greeting error: {e}")
            # Still activate even if greeting fails
            self.start_detection()
    
    async def startup(self):
        """Service startup logic"""
        try:
            self.log("info", "KWD service starting", "service_start")
            
            # Load configuration
            if not self.load_config():
                self.log("error", "Failed to load configuration")
                return False
            
            # Load wake word model
            if not self.load_wake_word_model():
                self.log("error", "Failed to load wake word model")
                return False
            
            self.state = KWDState.READY
            self.log("info", "KWD service ready", "service_ready")
            return True
            
        except Exception as e:
            self.log("error", f"KWD startup failed: {e}")
            return False
    
    async def shutdown(self):
        """Service shutdown logic"""
        self.log("info", "KWD service shutting down")
        self.stop_detection()


# Global KWD instance
kwd_service = KWDService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await kwd_service.startup()
    yield
    # Shutdown
    await kwd_service.shutdown()


# FastAPI app
app = FastAPI(
    title="KWD Service",
    description="Wake Word Detection using openWakeWord",
    version="1.0",
    lifespan=lifespan
)


@app.post("/on-system-ready")
async def on_system_ready():
    """Called by loader after all services are ready"""
    if kwd_service.state not in [KWDState.READY, KWDState.SLEEP]:
        raise HTTPException(status_code=400, detail=f"Service not ready, state: {kwd_service.state}")
    
    kwd_service.log("info", "System ready signal received", "system_ready")
    
    # Play greeting and activate in background
    threading.Thread(target=kwd_service.play_greeting_and_activate, daemon=True).start()
    
    return {"status": "ok", "message": "Greeting and activation initiated"}


@app.post("/start")
async def start_detection(request: StartRequest = StartRequest()):
    """Start or re-arm wake word detection"""
    if kwd_service.state == KWDState.INIT:
        raise HTTPException(status_code=400, detail="Service not initialized")
    
    if kwd_service.start_detection():
        kwd_service.log("info", "Detection re-armed", "rearmed")
        return {"status": "ok", "state": kwd_service.state}
    else:
        raise HTTPException(status_code=500, detail="Failed to start detection")


@app.post("/stop")
async def stop_detection_endpoint(request: StopRequest = StopRequest()):
    """Stop wake word detection"""
    kwd_service.stop_detection()
    reason = request.reason or "manual stop"
    kwd_service.log("info", f"Detection stopped: {reason}", "kwd_stopped")
    return {"status": "ok", "state": kwd_service.state}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "ok" if kwd_service.state != KWDState.INIT else "error"
    return HealthResponse(status=status, state=kwd_service.state)


@app.get("/state")
async def get_state():
    """Get current service state"""
    return StateResponse(state=kwd_service.state)


def main():
    """Entry point when run as module"""
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=kwd_service.port,
        log_level="error",  # Suppress uvicorn logs to keep console clean
        access_log=False
    )


if __name__ == "__main__":
    main()
