#!/usr/bin/env python3
"""
STT Service - Speech-to-Text with Dialog Session Management

FastAPI service on port 5003 that:
- Owns and executes the STT-led dialog loop
- Captures user speech and transcribes with faster-whisper (small.en)
- Streams LLM responses and forwards to TTS via WebSocket
- Manages follow-up timing and re-arms KWD after dialog completion
- Adapts VAD thresholds based on RMS ambient noise readings

State: INIT → READY → ACTIVE (dialog loop) → READY
"""

import os
import sys
import time
import asyncio
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import configparser
import json

import numpy as np
import sounddevice as sd
import requests
import websockets
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("ERROR: faster-whisper not installed. Run: pip install faster-whisper")
    sys.exit(1)

try:
    import webrtcvad
except ImportError:
    print("WARNING: webrtcvad not installed. VAD will use basic energy detection")
    webrtcvad = None


class STTState:
    INIT = "INIT"
    READY = "READY"  
    ACTIVE = "ACTIVE"


# Request/Response Models
class StartRequest(BaseModel):
    dialog_id: str


class StopRequest(BaseModel):
    dialog_id: str
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    state: str


class StateResponse(BaseModel):
    state: str


class LLMCompleteRequest(BaseModel):
    text: str
    dialog_id: str
    history: Optional[List[Dict[str, str]]] = None


class STTService:
    def __init__(self):
        self.state = STTState.INIT
        self.config: Optional[configparser.ConfigParser] = None
        
        # Service URLs
        self.logger_url = "http://127.0.0.1:5000"
        self.rms_url = "http://127.0.0.1:5006"
        self.kwd_url = "http://127.0.0.1:5002"
        self.llm_url = "http://127.0.0.1:5004"
        self.tts_ws_speak = "ws://127.0.0.1:5005/speak-stream"
        self.tts_ws_events = "ws://127.0.0.1:5005/playback-events"
        
        # Configuration
        self.port = 5003
        self.model_size = "small.en"
        self.device = "cuda"
        self.compute_type = "int8_float16"
        self.beam_size = 1
        self.vad_finalize_sec = 2.0
        self.chunk_length_s = 15.0
        self.follow_up_timer_sec = 4.0
        self.word_timestamps = False
        
        # Adaptive VAD
        self.quiet_dbfs = -60
        self.noisy_dbfs = -35
        self.silence_margin_db = 8.0
        self.energy_delta_db = 6.0
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024  # 64ms at 16kHz
        
        # Whisper model
        self.model: Optional[WhisperModel] = None
        
        # Dialog session state
        self.active_dialog_id: Optional[str] = None
        self.dialog_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        
        # Audio capture
        self.audio_stream: Optional[sd.InputStream] = None
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # VAD
        self.vad = None
        if webrtcvad:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        
        # WebSocket connections
        self.tts_speak_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.tts_events_ws: Optional[websockets.WebSocketClientProtocol] = None
    
    def load_config(self) -> bool:
        """Load configuration from config/config.ini"""
        try:
            config_path = Path("config/config.ini")
            if not config_path.exists():
                self.log("warning", "config/config.ini not found, using defaults")
                return True
                
            self.config = configparser.ConfigParser()
            self.config.read(config_path)
            
            # Load STT section
            if 'stt' in self.config:
                stt_section = self.config['stt']
                self.port = stt_section.getint('port', self.port)
                self.model_size = stt_section.get('model_size', self.model_size)
                self.device = stt_section.get('device', self.device)
                self.compute_type = stt_section.get('compute_type', self.compute_type)
                self.beam_size = stt_section.getint('beam_size', self.beam_size)
                self.vad_finalize_sec = stt_section.getfloat('vad_finalize_sec', self.vad_finalize_sec)
                self.chunk_length_s = stt_section.getfloat('chunk_length_s', self.chunk_length_s)
                self.follow_up_timer_sec = stt_section.getfloat('follow_up_timer_sec', self.follow_up_timer_sec)
                self.word_timestamps = stt_section.getboolean('word_timestamps', self.word_timestamps)
            
            # Load adaptive section
            if 'stt.adaptive' in self.config:
                adaptive_section = self.config['stt.adaptive']
                self.quiet_dbfs = adaptive_section.getfloat('quiet_dbfs', self.quiet_dbfs)
                self.noisy_dbfs = adaptive_section.getfloat('noisy_dbfs', self.noisy_dbfs)
            
            # Load VAD section
            if 'stt.vad' in self.config:
                vad_section = self.config['stt.vad']
                self.silence_margin_db = vad_section.getfloat('silence_margin_db', self.silence_margin_db)
                self.energy_delta_db = vad_section.getfloat('energy_delta_db', self.energy_delta_db)
            
            # Load service dependencies
            if 'stt.deps' in self.config:
                deps_section = self.config['stt.deps']
                self.logger_url = deps_section.get('logger_url', self.logger_url)
                self.rms_url = deps_section.get('rms_url', self.rms_url)
                self.kwd_url = deps_section.get('kwd_url', self.kwd_url)
                self.llm_url = deps_section.get('llm_url', self.llm_url)
                self.tts_ws_speak = deps_section.get('tts_ws_speak', self.tts_ws_speak)
                self.tts_ws_events = deps_section.get('tts_ws_events', self.tts_ws_events)
            
            return True
        except Exception as e:
            self.log("error", f"Failed to load config: {e}")
            return False
    
    def log(self, level: str, message: str, event: str = None, extra: Dict[str, Any] = None):
        """Send log message to Logger service"""
        try:
            # Add human-readable timestamp (dd-mm-yy HH:MM:SS format)
            from datetime import datetime
            timestamp = datetime.now().strftime("%d-%m-%y %H:%M:%S")
            
            payload = {
                "svc": "STT",
                "level": level,
                "message": message,
            }
            if event:
                payload["event"] = event
            if extra:
                payload["extra"] = extra
                
            requests.post(f"{self.logger_url}/log", json=payload, timeout=2.0)
        except Exception:
            # Fallback to stdout if logger unavailable
            print(f"STT       {level.upper():<6}= {message}")
    
    def log_dialog(self, dialog_id: str, role: str, text: str):
        """Log dialog line to Logger service"""
        try:
            payload = {
                "dialog_id": dialog_id,
                "role": role,
                "text": text
            }
            requests.post(f"{self.logger_url}/dialog/line", json=payload, timeout=2.0)
        except Exception:
            pass
    
    def get_current_rms(self) -> Optional[float]:
        """Get current RMS noise level from RMS service"""
        try:
            response = requests.get(f"{self.rms_url}/current-rms", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("rms_dbfs")
        except Exception as e:
            self.log("debug", f"Failed to get RMS: {e}")
        return None
    
    def calculate_vad_threshold(self, noise_dbfs: Optional[float]) -> float:
        """Calculate VAD threshold based on ambient noise"""
        if noise_dbfs is None:
            return -40.0  # Default threshold
            
        # Adaptive threshold: quieter environments allow lower thresholds
        if noise_dbfs <= self.quiet_dbfs:
            return -50.0  # More sensitive in quiet
        elif noise_dbfs >= self.noisy_dbfs:
            return -30.0  # Less sensitive in noise
        else:
            # Linear interpolation
            ratio = (noise_dbfs - self.quiet_dbfs) / (self.noisy_dbfs - self.quiet_dbfs)
            return -50.0 + ratio * (-30.0 - (-50.0))
    
    def load_whisper_model(self) -> bool:
        """Load faster-whisper model"""
        try:
            self.log("info", f"Loading Whisper model: {self.model_size} on {self.device}")
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            self.log("info", "Whisper model loaded successfully", "model_loaded")
            return True
            
        except Exception as e:
            self.log("error", f"Failed to load Whisper model: {e}")
            return False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            self.log("warning", f"Audio callback status: {status}")
            
        # Convert to float32 and add to buffer
        audio_data = indata[:, 0].astype(np.float32)
        
        with self.buffer_lock:
            self.audio_buffer.extend(audio_data)
    
    def start_audio_capture(self) -> bool:
        """Start audio capture stream"""
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            self.audio_stream.start()
            self.log("info", "Audio capture started")
            return True
        except Exception as e:
            self.log("error", f"Failed to start audio capture: {e}")
            return False
    
    def stop_audio_capture(self):
        """Stop audio capture stream"""
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                self.log("info", "Audio capture stopped")
            except Exception as e:
                self.log("warning", f"Error stopping audio capture: {e}")
    
    def detect_speech_end(self, audio_chunk: np.ndarray, energy_threshold: float) -> bool:
        """Simple energy-based VAD for speech end detection"""
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        energy_db = 20 * np.log10(energy + 1e-8)
        
        # Use WebRTC VAD if available
        if self.vad and len(audio_chunk) == 480:  # 30ms at 16kHz
            # Convert to int16 for WebRTC VAD
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            return not self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
        
        # Fallback: simple energy threshold
        return energy_db < energy_threshold
    
    def capture_and_transcribe(self, dialog_id: str) -> Optional[str]:
        """Capture audio and transcribe to text"""
        try:
            # Get adaptive VAD threshold
            rms_dbfs = self.get_current_rms()
            energy_threshold = self.calculate_vad_threshold(rms_dbfs)
            
            self.log("debug", f"VAD threshold: {energy_threshold:.1f} dBFS (RMS: {rms_dbfs})")
            
            # Clear existing buffer
            with self.buffer_lock:
                self.audio_buffer.clear()
            
            # Start audio capture
            if not self.start_audio_capture():
                return None
            
            # Collect audio until silence
            audio_data = []
            silence_duration = 0
            chunk_duration = self.chunk_size / self.sample_rate
            
            self.log("info", "Listening for speech...")
            
            while not self.stop_flag.is_set():
                time.sleep(chunk_duration)
                
                # Get audio chunk
                with self.buffer_lock:
                    if len(self.audio_buffer) < self.chunk_size:
                        continue
                    
                    chunk = np.array(self.audio_buffer[:self.chunk_size])
                    self.audio_buffer = self.audio_buffer[self.chunk_size:]
                
                audio_data.extend(chunk)
                
                # Check for speech end
                if self.detect_speech_end(chunk, energy_threshold):
                    silence_duration += chunk_duration
                    if silence_duration >= self.vad_finalize_sec:
                        break
                else:
                    silence_duration = 0
                
                # Limit total recording time
                if len(audio_data) / self.sample_rate > self.chunk_length_s:
                    self.log("warning", "Maximum recording length reached")
                    break
            
            # Stop audio capture
            self.stop_audio_capture()
            
            if not audio_data:
                self.log("warning", "No audio captured")
                return None
            
            # Convert to numpy array
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Transcribe with Whisper
            self.log("info", "Transcribing speech...")
            segments, info = self.model.transcribe(
                audio_array,
                beam_size=self.beam_size,
                word_timestamps=self.word_timestamps,
                language="en"
            )
            
            # Combine segments into final text
            transcript = " ".join([segment.text.strip() for segment in segments])
            
            if transcript:
                self.log("info", f"USER: {transcript}", "stt_final_text")
                return transcript.strip()
            else:
                self.log("warning", "Empty transcription")
                return None
                
        except Exception as e:
            self.log("error", f"Transcription error: {e}")
            return None
        finally:
            self.stop_audio_capture()
    
    async def connect_tts_websockets(self, dialog_id: str) -> bool:
        """Connect to TTS WebSocket endpoints"""
        try:
            # Connect to speak stream
            self.tts_speak_ws = await websockets.connect(self.tts_ws_speak)
            
            # Connect to playback events
            events_url = f"{self.tts_ws_events}/{dialog_id}"
            self.tts_events_ws = await websockets.connect(events_url)
            
            self.log("debug", "Connected to TTS WebSockets")
            return True
            
        except Exception as e:
            self.log("error", f"Failed to connect to TTS WebSockets: {e}")
            return False
    
    async def disconnect_tts_websockets(self):
        """Disconnect from TTS WebSocket endpoints"""
        try:
            if self.tts_speak_ws:
                await self.tts_speak_ws.close()
                self.tts_speak_ws = None
            
            if self.tts_events_ws:
                await self.tts_events_ws.close()
                self.tts_events_ws = None
                
            self.log("debug", "Disconnected from TTS WebSockets")
            
        except Exception as e:
            self.log("warning", f"Error disconnecting TTS WebSockets: {e}")
    
    async def stream_llm_to_tts(self, dialog_id: str, user_text: str) -> bool:
        """Stream LLM response to TTS via WebSocket"""
        try:
            # Prepare LLM request
            llm_payload = LLMCompleteRequest(
                text=user_text,
                dialog_id=dialog_id
            )
            
            # Start LLM streaming request
            self.log("info", "Starting LLM completion stream", "llm_stream_started")
            
            response = requests.post(
                f"{self.llm_url}/complete",
                json=llm_payload.dict(),
                stream=True,
                timeout=120.0
            )
            
            if response.status_code != 200:
                self.log("error", f"LLM request failed: {response.status_code}")
                return False
            
            # Stream response chunks to TTS
            full_response = ""
            
            for line in response.iter_lines():
                if self.stop_flag.is_set():
                    break
                    
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        chunk_data = line_str[6:]  # Remove 'data: ' prefix
                        
                        if chunk_data.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_json = json.loads(chunk_data)
                            text_chunk = chunk_json.get('text', '')
                            
                            if text_chunk and self.tts_speak_ws:
                                # Send chunk to TTS
                                await self.tts_speak_ws.send(json.dumps({
                                    "text_chunk": text_chunk
                                }))
                                
                                full_response += text_chunk
                                
                        except json.JSONDecodeError:
                            continue
            
            # Log complete response
            if full_response:
                self.log("info", f"ALEXA: {full_response}", "llm_stream_end")
                self.log_dialog(dialog_id, "USER", user_text)
                self.log_dialog(dialog_id, "ALEXA", full_response)
            
            return True
            
        except Exception as e:
            self.log("error", f"LLM streaming error: {e}")
            return False
    
    async def wait_for_tts_completion(self, dialog_id: str) -> bool:
        """Wait for TTS playback completion via WebSocket events"""
        try:
            if not self.tts_events_ws:
                return False
            
            self.log("debug", "Waiting for TTS playback completion")
            
            while not self.stop_flag.is_set():
                try:
                    # Wait for event with timeout
                    message = await asyncio.wait_for(
                        self.tts_events_ws.recv(), 
                        timeout=30.0
                    )
                    
                    event_data = json.loads(message)
                    event_type = event_data.get("event")
                    
                    if event_type == "playback_finished":
                        self.log("info", "TTS playback completed", "tts_playback_finished")
                        return True
                    elif event_type == "playback_error":
                        self.log("warning", "TTS playback error")
                        return False
                        
                except asyncio.TimeoutError:
                    self.log("warning", "Timeout waiting for TTS completion")
                    return False
                except json.JSONDecodeError:
                    continue
            
            return False
            
        except Exception as e:
            self.log("error", f"TTS completion wait error: {e}")
            return False
    
    def rearm_kwd(self):
        """Re-arm KWD for next wake word detection"""
        try:
            response = requests.post(f"{self.kwd_url}/start", timeout=5.0)
            
            if response.status_code == 200:
                self.log("info", "KWD re-armed successfully")
            else:
                self.log("warning", f"KWD re-arm failed: {response.status_code}")
                
        except Exception as e:
            self.log("warning", f"KWD re-arm error: {e}")
    
    async def dialog_worker(self, dialog_id: str):
        """Main dialog processing worker"""
        try:
            self.log("info", f"Dialog session started: {dialog_id}", "dialog_turn_started")
            
            # Start dialog in logger
            requests.post(f"{self.logger_url}/dialog/start", 
                         json={"dialog_id": dialog_id}, timeout=2.0)
            
            # 1. Capture and transcribe user speech
            user_text = self.capture_and_transcribe(dialog_id)
            
            if not user_text:
                self.log("warning", "No user speech captured")
                return
            
            # 2. Connect to TTS WebSockets
            if not await self.connect_tts_websockets(dialog_id):
                return
            
            # 3. Stream LLM response to TTS
            if not await self.stream_llm_to_tts(dialog_id, user_text):
                return
            
            # 4. Wait for TTS playback completion
            if not await self.wait_for_tts_completion(dialog_id):
                return
            
            # 5. Follow-up timer for multi-turn conversation
            self.log("debug", f"Starting follow-up timer: {self.follow_up_timer_sec}s")
            
            # TODO: Implement follow-up logic if needed
            # For now, just wait and then end dialog
            await asyncio.sleep(self.follow_up_timer_sec)
            
        except Exception as e:
            self.log("error", f"Dialog worker error: {e}")
        finally:
            # Cleanup
            await self.disconnect_tts_websockets()
            
            # End dialog in logger
            try:
                requests.post(f"{self.logger_url}/dialog/end", 
                             json={"dialog_id": dialog_id}, timeout=2.0)
            except Exception:
                pass
            
            # Re-arm KWD
            self.rearm_kwd()
            
            # Reset state
            self.active_dialog_id = None
            self.state = STTState.READY
            self.log("info", f"Dialog session ended: {dialog_id}", "dialog_ended")
    
    def start_dialog(self, dialog_id: str) -> bool:
        """Start a new dialog session"""
        if self.state != STTState.READY:
            self.log("warning", f"Cannot start dialog in state: {self.state}")
            return False
        
        if self.active_dialog_id:
            self.log("warning", f"Dialog already active: {self.active_dialog_id}")
            return False
        
        try:
            self.active_dialog_id = dialog_id
            self.state = STTState.ACTIVE
            self.stop_flag.clear()
            
            # Start dialog worker in background
            self.dialog_thread = threading.Thread(
                target=lambda: asyncio.run(self.dialog_worker(dialog_id)),
                daemon=True
            )
            self.dialog_thread.start()
            
            return True
            
        except Exception as e:
            self.log("error", f"Failed to start dialog: {e}")
            self.active_dialog_id = None
            self.state = STTState.READY
            return False
    
    def stop_dialog(self, dialog_id: str, reason: str = "manual") -> bool:
        """Stop active dialog session"""
        if not self.active_dialog_id or self.active_dialog_id != dialog_id:
            return False
        
        try:
            self.log("info", f"Stopping dialog: {reason}")
            self.stop_flag.set()
            
            # Wait for dialog thread to finish
            if self.dialog_thread and self.dialog_thread.is_alive():
                self.dialog_thread.join(timeout=5.0)
            
            self.active_dialog_id = None
            self.state = STTState.READY
            
            return True
            
        except Exception as e:
            self.log("error", f"Error stopping dialog: {e}")
            return False
    
    async def startup(self):
        """Service startup logic"""
        try:
            self.log("info", "STT service starting", "service_start")
            
            # Load configuration
            if not self.load_config():
                self.log("error", "Failed to load configuration")
                return False
            
            # Load Whisper model
            if not self.load_whisper_model():
                self.log("error", "Failed to load Whisper model")
                return False
            
            self.state = STTState.READY
            self.log("info", "STT service ready", "service_ready")
            return True
            
        except Exception as e:
            self.log("error", f"STT startup failed: {e}")
            return False
    
    async def shutdown(self):
        """Service shutdown logic"""
        self.log("info", "STT service shutting down")
        
        # Stop any active dialog
        if self.active_dialog_id:
            self.stop_flag(self.active_dialog_id, "shutdown")
        
        # Stop audio capture
        self.stop_audio_capture()


# Global STT instance
stt_service = STTService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await stt_service.startup()
    yield
    # Shutdown
    await stt_service.shutdown()


# FastAPI app
app = FastAPI(
    title="STT Service",
    description="Speech-to-Text with Dialog Session Management",
    version="1.0",
    lifespan=lifespan
)


@app.post("/start")
async def start_dialog(request: StartRequest):
    """Start a new dialog session"""
    if stt_service.state != STTState.READY:
        raise HTTPException(
            status_code=400, 
            detail=f"Service not ready, state: {stt_service.state}"
        )
    
    if stt_service.start_dialog(request.dialog_id):
        return {"status": "ok", "dialog_id": request.dialog_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to start dialog")


@app.post("/stop")
async def stop_dialog(request: StopRequest):
    """Stop active dialog session"""
    reason = request.reason or "manual stop"
    
    if stt_service.stop_dialog(request.dialog_id, reason):
        return {"status": "ok", "dialog_id": request.dialog_id}
    else:
        raise HTTPException(status_code=404, detail="Dialog not found or already stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "ok" if stt_service.state == STTState.READY else "error"
    return HealthResponse(status=status, state=stt_service.state)


@app.get("/state")
async def get_state():
    """Get current service state"""
    return StateResponse(state=stt_service.state)


def main():
    """Entry point when run as module"""
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=stt_service.port,
        log_level="error",  # Suppress uvicorn logs to keep console clean
        access_log=False
    )


if __name__ == "__main__":
    main()
