#!/usr/bin/env python3
"""
TTS Service - Streaming Text-to-Speech using Kokoro

FastAPI service on port 5005 that:
- Provides low-latency streaming TTS using Kokoro/af_heart voice
- Consumes streamed text chunks (primary: from STT bridging LLM)
- GPU/CUDA inference with minimal VRAM via chunked synthesis
- Real-time audio playback with event emissions
- WebSocket interfaces for streaming text input and playback events

State: INIT → READY → PLAYING → READY/INTERRUPTED
"""

import os
import sys
import time
import json
import asyncio
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator
import re

import torch
import sounddevice as sd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from .config_loader import app_config
from .shared import lifespan_with_httpx, safe_post, safe_get
from .utils import post_log, create_service_app

# Import Kokoro TTS components
try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("Warning: Kokoro TTS not available. Install kokoro-onnx package.")


class TTSState:
    INIT = "INIT"
    READY = "READY"
    PLAYING = "PLAYING"
    INTERRUPTED = "INTERRUPTED"


# Request/Response Models
class SpeakRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    dialog_id: Optional[str] = None


class SpeakStreamMessage(BaseModel):
    text_chunk: str
    dialog_id: str
    voice: Optional[str] = None


class SpeakFromLLMRequest(BaseModel):
    dialog_id: str
    llm_url: str
    text: str
    history: Optional[List[Dict[str, str]]] = None
    params: Optional[Dict[str, Any]] = None


class StopRequest(BaseModel):
    dialog_id: str


class HealthResponse(BaseModel):
    status: str
    state: str


class StateResponse(BaseModel):
    state: str


class PlaybackEvent(BaseModel):
    event: str  # "playback_started", "playback_finished", "progress"
    dialog_id: Optional[str] = None
    at_ms: Optional[float] = None


class TTSService:
    def __init__(self):
        self.state = TTSState.INIT
        
        # Load configuration from config_loader
        tts_config = app_config.tts
        
        # Service URLs
        self.logger_url = tts_config.deps.logger_url
        
        # Configuration
        self.port = tts_config.port
        self.voice = tts_config.voice
        self.sample_rate = tts_config.sample_rate
        self.device = tts_config.device
        self.dtype = tts_config.dtype
        self.max_queue_text = tts_config.max_queue_text
        self.max_queue_audio = tts_config.max_queue_audio
        self.chunk_chars = tts_config.chunk_chars
        self.silence_pad_ms = tts_config.silence_pad_ms
        self.allow_llm_pull = tts_config.allow_llm_pull
        
        # ONNX assets
        self.model_path = tts_config.model_path
        self.voices_path = tts_config.voices_path
        self.quant_preference = tts_config.quant_preference
        
        # Kokoro TTS engine
        self.tts_engine: Optional[Kokoro] = None
        
        # Audio playback
        self.audio_stream = None
        self.audio_device = None
        
        # Queues for minimal VRAM operation
        self.text_queue = queue.Queue(maxsize=self.max_queue_text)
        self.audio_queue = queue.Queue(maxsize=self.max_queue_audio)
        
        # Current playback tracking
        self.current_dialog_id: Optional[str] = None
        self.playback_start_time: Optional[float] = None
        self.is_playing = False
        self.should_stop = False
        
        # WebSocket connections for events
        self.event_connections: Dict[str, List[WebSocket]] = {}
        
        # Processing threads
        self.synthesis_thread: Optional[threading.Thread] = None
        self.playback_thread: Optional[threading.Thread] = None
        self.threads_running = False
        
        # Thread locks
        self.state_lock = threading.Lock()
        self.queue_lock = threading.Lock()
    
    
    async def log(self, level: str, message: str, event: str = None, extra: Dict[str, Any] = None):
        """Send log message to Logger service"""
        # Delegate logging to the shared utility
        await post_log(
            service="TTS",
            level=level,
            message=message,
            logger_url=self.logger_url,
            event=event,
            extra=extra,
            timeout=2.0
        )
    
    async def send_metric(self, metric: str, value: float, dialog_id: str = None):
        """Send metric to Logger service"""
        payload = {
            "svc": "TTS",
            "metric": metric,
            "value": value
        }
        if dialog_id:
            payload["dialog_id"] = dialog_id
            
        await safe_post(f"{self.logger_url}/metrics", json=payload, timeout=2.0, fallback_msg="")
    
    def initialize_cuda(self) -> bool:
        """Initialize CUDA for TTS inference"""
        try:
            if not torch.cuda.is_available():
                print("TTS       ERROR = CUDA not available")
                return False
            
            # Set device
            torch.cuda.set_device(0)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            print("TTS       INFO  = CUDA initialized on device 0")
            return True
            
        except Exception as e:
            print(f"TTS       ERROR = Failed to initialize CUDA: {e}")
            return False
    
    def initialize_tts_engine(self) -> bool:
        """Initialize Kokoro TTS engine"""
        try:
            if not KOKORO_AVAILABLE:
                print("TTS       ERROR = Kokoro TTS not available")
                return False
            
            # Check if model and voices files exist
            model_path = Path(self.model_path)
            voices_path = Path(self.voices_path)
            
            if not model_path.exists():
                print(f"TTS       ERROR = Model file not found: {self.model_path}")
                return False
            
            if not voices_path.exists():
                print(f"TTS       ERROR = Voices file not found: {self.voices_path}")
                return False
            
            # Initialize Kokoro with model and voices paths
            device = self.device if torch.cuda.is_available() else "cpu"
            
            self.tts_engine = Kokoro(
                model_path=str(model_path),
                voices_path=str(voices_path),
                # device=cuda
                # if device == "cuda" else "cpu",
            )
            
            print(f"TTS       INFO  = TTS engine initialized with voice {self.voice}, model: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"TTS       ERROR = Failed to initialize TTS engine: {e}")
            return False
    
    def initialize_audio_device(self) -> bool:
        """Initialize audio playback device"""
        try:
            # Get default audio output device
            devices = sd.query_devices()
            default_device = sd.default.device[1]  # Output device
            
            device_info = sd.query_devices(default_device)
            self.audio_device = default_device
            
            print(f"TTS       INFO  = Audio device initialized: {device_info['name']}")
            return True
            
        except Exception as e:
            print(f"TTS       ERROR = Failed to initialize audio device: {e}")
            return False
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into synthesis chunks"""
        if len(text) <= self.chunk_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = re.split(r'([.!?]+)', text)
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]
            
            if len(current_chunk + sentence) <= self.chunk_chars:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def synthesize_chunk(self, text: str) -> Optional[np.ndarray]:
        """Synthesize a text chunk to audio"""
        try:
            if not self.tts_engine:
                return None
            
            start_time = time.time()
            
            # Generate audio using Kokoro - use create() method
            audio_data, sample_rate = self.tts_engine.create(text, voice=self.voice)
            
            # Update sample rate if different from config
            if sample_rate != self.sample_rate:
                print(f"TTS       INFO  = Using Kokoro sample rate: {sample_rate} Hz")
                self.sample_rate = sample_rate
            
            # Convert to numpy array if needed
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            
            # Ensure float32 for sounddevice
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            synth_time = (time.time() - start_time) * 1000
            # Use asyncio bridge for metrics from sync context
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self.send_metric("chunk_synth_ms", synth_time),
                    loop
                )
            except:
                pass
            
            return audio_data
            
        except Exception as e:
            print(f"TTS       ERROR = Synthesis failed: {e}")
            return None
    
    def synthesis_worker(self):
        """Background thread for text synthesis"""
        while self.threads_running:
            try:
                # Get text from queue with timeout
                try:
                    text_item = self.text_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if text_item is None:  # Shutdown signal
                    break
                
                text, dialog_id = text_item
                
                # Chunk the text
                chunks = self.chunk_text(text)
                
                for chunk in chunks:
                    if not self.threads_running or self.should_stop:
                        break
                    
                    # Synthesize chunk
                    audio_data = self.synthesize_chunk(chunk)
                    
                    if audio_data is not None:
                        # Add to audio queue (drop oldest if full)
                        try:
                            self.audio_queue.put((audio_data, dialog_id), block=False)
                        except queue.Full:
                            # Drop oldest audio chunk
                            try:
                                self.audio_queue.get(block=False)
                                self.audio_queue.put((audio_data, dialog_id), block=False)
                            except queue.Empty:
                                pass
                
                self.text_queue.task_done()
                
            except Exception as e:
                print(f"TTS       ERROR = Synthesis worker error: {e}")
    
    def playback_worker(self):
        """Background thread for audio playback"""
        while self.threads_running:
            try:
                # Get audio from queue with timeout
                try:
                    audio_item = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if audio_item is None:  # Shutdown signal
                    break
                
                audio_data, dialog_id = audio_item
                
                if self.should_stop:
                    self.audio_queue.task_done()
                    continue
                
                # Play audio chunk
                self.play_audio_chunk(audio_data, dialog_id)
                self.audio_queue.task_done()
                
            except Exception as e:
                print(f"TTS       ERROR = Playback worker error: {e}")
    
    def play_audio_chunk(self, audio_data: np.ndarray, dialog_id: str):
        """Play a single audio chunk"""
        try:
            if not self.is_playing:
                self.is_playing = True
                self.playback_start_time = time.time()
                # Skip event emission from background thread to avoid asyncio issues
                print(f"TTS       DEBUG = Playback started for dialog {dialog_id}")
            
            # Play audio using sounddevice
            sd.play(audio_data, samplerate=self.sample_rate, device=self.audio_device)
            sd.wait()  # Wait until audio is finished
            
            # Log progress instead of emitting events from background thread
            if self.playback_start_time:
                elapsed_ms = (time.time() - self.playback_start_time) * 1000
                print(f"TTS       DEBUG = Audio playback progress: {elapsed_ms:.0f}ms")
            
        except Exception as e:
            print(f"TTS       ERROR = Audio playback error: {e}")
    
    async def emit_playback_event(self, event_type: str, dialog_id: str, at_ms: Optional[float] = None):
        """Emit playback event to WebSocket subscribers"""
        event = PlaybackEvent(
            event=event_type,
            dialog_id=dialog_id,
            at_ms=at_ms
        )
        
        # Send to all connected clients for this dialog
        if dialog_id in self.event_connections:
            disconnected = []
            for ws in self.event_connections[dialog_id]:
                try:
                    await ws.send_text(event.model_dump_json())
                except Exception:
                    disconnected.append(ws)
            
            # Remove disconnected clients
            for ws in disconnected:
                self.event_connections[dialog_id].remove(ws)
    
    def start_processing_threads(self):
        """Start background processing threads"""
        if self.threads_running:
            return
        
        self.threads_running = True
        self.synthesis_thread = threading.Thread(target=self.synthesis_worker, daemon=True)
        self.playback_thread = threading.Thread(target=self.playback_worker, daemon=True)
        
        self.synthesis_thread.start()
        self.playback_thread.start()
        
        print("TTS       INFO  = Processing threads started")
    
    def stop_processing_threads(self):
        """Stop background processing threads"""
        self.threads_running = False
        
        # Send shutdown signals
        try:
            self.text_queue.put(None, block=False)
            self.audio_queue.put(None, block=False)
        except queue.Full:
            pass
        
        # Wait for threads to finish
        if self.synthesis_thread and self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=2.0)
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        
        print("TTS       INFO  = Processing threads stopped")
    
    def clear_queues(self):
        """Clear all queues"""
        with self.queue_lock:
            while not self.text_queue.empty():
                try:
                    self.text_queue.get(block=False)
                    self.text_queue.task_done()
                except queue.Empty:
                    break
            
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get(block=False)
                    self.audio_queue.task_done()
                except queue.Empty:
                    break
    
    async def speak_text(self, text: str, dialog_id: Optional[str] = None, voice: Optional[str] = None) -> bool:
        """Add text to synthesis queue"""
        try:
            # Use current voice if not specified
            if voice and voice != self.voice:
                await self.log("warning", f"Voice switching not implemented, using {self.voice}")
            
            # Update state
            with self.state_lock:
                if self.state == TTSState.READY:
                    self.state = TTSState.PLAYING
                    self.current_dialog_id = dialog_id
                    self.should_stop = False
            
            # Add to synthesis queue
            self.text_queue.put((text, dialog_id), block=False)
            
            await self.log("info", f"Text queued for synthesis: {len(text)} chars", "chunk_received")
            return True
            
        except queue.Full:
            await self.log("error", "Text queue full, dropping text")
            return False
        except Exception as e:
            await self.log("error", f"Failed to queue text: {e}")
            return False
    
    async def stop_playback(self, dialog_id: str):
        """Stop current playback"""
        try:
            with self.state_lock:
                if self.state == TTSState.PLAYING:
                    self.state = TTSState.INTERRUPTED
                    self.should_stop = True
            
            # Stop audio playback
            sd.stop()
            
            # Clear queues
            self.clear_queues()
            
            # Emit stop event
            await self.emit_playback_event("playback_finished", dialog_id)
            
            # Reset state
            with self.state_lock:
                self.state = TTSState.READY
                self.is_playing = False
                self.current_dialog_id = None
                self.should_stop = False
            
            await self.log("info", "Playback stopped", "stream_aborted")
            
        except Exception as e:
            await self.log("error", f"Failed to stop playback: {e}")
    
    async def speak_from_llm(self, request: SpeakFromLLMRequest) -> bool:
        """Pull text stream from LLM and synthesize"""
        if not self.allow_llm_pull:
            return False
        
        try:
            # Prepare LLM request
            llm_payload = {
                "dialog_id": request.dialog_id,
                "text": request.text,
                "history": request.history or [],
                "params": request.params or {}
            }
            
            # Start streaming request to LLM using httpx
            import httpx
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    request.llm_url,
                    json=llm_payload,
                    timeout=120.0
                ) as response:
                    
                    if response.status_code != 200:
                        await self.log("error", f"LLM request failed: {response.status_code}")
                        return False
                    
                    # Process LLM stream
                    accumulated_text = ""
                    
                    async for chunk in response.aiter_text(chunk_size=1024):
                        if chunk and not self.should_stop:
                            accumulated_text += chunk
                            
                            # Process in sentence chunks
                            if len(accumulated_text) >= self.chunk_chars or chunk.endswith(('.', '!', '?')):
                                await self.speak_text(accumulated_text, request.dialog_id)
                                accumulated_text = ""
                    
                    # Process remaining text
                    if accumulated_text and not self.should_stop:
                        await self.speak_text(accumulated_text, request.dialog_id)
            
            return True
            
        except Exception as e:
            await self.log("error", f"LLM pull failed: {e}")
            return False
    
    async def startup(self):
        """Service startup logic"""
        try:
            await self.log("info", "TTS service starting", "service_start")
            
            # Initialize CUDA
            if not self.initialize_cuda():
                self.state = TTSState.INIT
                return False
            
            # Initialize TTS engine
            if not self.initialize_tts_engine():
                self.state = TTSState.INIT
                return False
            
            # Initialize audio device
            if not self.initialize_audio_device():
                self.state = TTSState.INIT
                return False
            
            # Start processing threads
            self.start_processing_threads()
            
            self.state = TTSState.READY
            await self.log("info", "TTS service ready", "service_ready")
            return True
            
        except Exception as e:
            await self.log("error", f"TTS startup failed: {e}")
            self.state = TTSState.INIT
            return False
    
    async def shutdown(self):
        """Service shutdown logic"""
        await self.log("info", "TTS service shutting down")
        
        # Stop playback
        sd.stop()
        
        # Stop processing threads
        self.stop_processing_threads()
        
        # Clear queues
        self.clear_queues()
        
        # Close WebSocket connections
        for dialog_connections in self.event_connections.values():
            for ws in dialog_connections:
                try:
                    await ws.close()
                except Exception:
                    pass
        
        self.event_connections.clear()


# Global TTS instance
tts_service = TTSService()


@asynccontextmanager
async def service_lifespan():
    # Startup
    success = await tts_service.startup()
    if not success:
        raise RuntimeError("TTS service startup failed")
    yield
    # Shutdown
    await tts_service.shutdown()


# FastAPI app
app = create_service_app(
    title="TTS Service",
    description="Streaming Text-to-Speech using Kokoro",
    version="1.0",
    lifespan=lambda app: lifespan_with_httpx(service_lifespan)
)


@app.post("/speak")
async def speak(request: SpeakRequest):
    """Unary synthesis for short phrases"""
    if tts_service.state not in [TTSState.READY, TTSState.PLAYING]:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready, state: {tts_service.state}"
        )
    
    success = await tts_service.speak_text(
        request.text,
        request.dialog_id,
        request.voice
    )
    
    if success:
        return {"status": "ok", "message": "Text queued for synthesis"}
    else:
        raise HTTPException(status_code=500, detail="Failed to queue text")


@app.websocket("/speak-stream")
async def speak_stream(websocket: WebSocket):
    """Primary streaming interface for text chunks"""
    await websocket.accept()
    
    try:
        while True:
            # Receive text chunk
            data = await websocket.receive_text()
            message = SpeakStreamMessage.model_validate_json(data)
            
            # Queue text for synthesis
            success = await tts_service.speak_text(
                message.text_chunk,
                message.dialog_id,
                message.voice
            )
            
            # Send acknowledgment
            ack_message = {
                "ack": success,
                "queued": tts_service.text_queue.qsize()
            }
            await websocket.send_text(json.dumps(ack_message))
            
    except WebSocketDisconnect:
        # Use asyncio bridge for logging from WebSocket context
        try:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                tts_service.log("info", "Speak stream WebSocket disconnected"),
                loop
            )
        except:
            print("TTS       INFO  = Speak stream WebSocket disconnected")
    except Exception as e:
        try:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                tts_service.log("error", f"Speak stream error: {e}"),
                loop
            )
        except:
            print(f"TTS       ERROR = Speak stream error: {e}")


@app.websocket("/playback-events/{dialog_id}")
async def playback_events(websocket: WebSocket, dialog_id: str):
    """Subscribe to playback events for a dialog"""
    await websocket.accept()
    
    # Add to event connections
    if dialog_id not in tts_service.event_connections:
        tts_service.event_connections[dialog_id] = []
    tts_service.event_connections[dialog_id].append(websocket)
    
    try:
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        # Remove from connections
        if dialog_id in tts_service.event_connections:
            try:
                tts_service.event_connections[dialog_id].remove(websocket)
                if not tts_service.event_connections[dialog_id]:
                    del tts_service.event_connections[dialog_id]
            except ValueError:
                pass
        # Use asyncio bridge for logging from WebSocket context
        try:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                tts_service.log("info", f"Playback events WebSocket disconnected for dialog {dialog_id}"),
                loop
            )
        except:
            print(f"TTS       INFO  = Playback events WebSocket disconnected for dialog {dialog_id}")
    except Exception as e:
        try:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                tts_service.log("error", f"Playback events error: {e}"),
                loop
            )
        except:
            print(f"TTS       ERROR = Playback events error: {e}")


@app.post("/speak-from-llm")
async def speak_from_llm(request: SpeakFromLLMRequest):
    """Optional direct pull from LLM stream"""
    if not tts_service.allow_llm_pull:
        raise HTTPException(status_code=403, detail="LLM pull mode disabled")
    
    if tts_service.state not in [TTSState.READY, TTSState.PLAYING]:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready, state: {tts_service.state}"
        )
    
    success = await tts_service.speak_from_llm(request)
    
    if success:
        return {"status": "ok", "message": "LLM stream started"}
    else:
        raise HTTPException(status_code=502, detail="LLM pull failed")


@app.post("/stop")
async def stop_playback(request: StopRequest):
    """Abort current playback"""
    await tts_service.stop_playback(request.dialog_id)
    return {"status": "ok", "message": "Playback stopped"}



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "ok" if tts_service.state == TTSState.READY else "error"
    return HealthResponse(status=status, state=tts_service.state)


@app.get("/state")
async def get_state():
    """Get current service state"""
    return StateResponse(state=tts_service.state)


def main():
    """Entry point when run as module"""
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=app_config.tts.port,  # Get port from config
        log_level="error",  # Suppress uvicorn logs to keep console clean
        access_log=False
    )


if __name__ == "__main__":
    main()
