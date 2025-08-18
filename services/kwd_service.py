#!/usr/bin/env python3
"""
KWD Service - Wake Word Detection using openWakeWord

FastAPI service on port 5002 that:
- Continuously listens for wake words using openWakeWord (ONNX model path configurable)
- Adapts sensitivity based on ambient noise from RMS service (optional)
- Triggers TTS confirmation and STT dialog on detection
- Manages state: INIT → READY → ACTIVE → TRIGGERED → SLEEP → ACTIVE
- Provides endpoints for loader coordination and basic diagnostics

Notes:
- Audio is captured @16 kHz, mono, int16. Detector expects int16 PCM.
- Chunking uses 80 ms windows (1280 samples) as recommended for stable streaming features.
- Includes basic DC-block (one-pole high-pass) and RMS gating to avoid “dead mic” noise.
- Device selection is configurable via [audio] section.
"""

import os
import sys
import time
import threading
import random
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
import configparser

import numpy as np
import sounddevice as sd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# --- Third-party wake word model ---
try:
    from openwakeword.model import Model
except ImportError:
    print("ERROR: openwakeword not installed. Try: uv add openwakeword sounddevice")
    sys.exit(1)


# -----------------------------
# Constants / State
# -----------------------------
class KWDState:
    INIT = "INIT"
    READY = "READY"
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    SLEEP = "SLEEP"


# -----------------------------
# Models
# -----------------------------
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


# -----------------------------
# Service
# -----------------------------
class KWDService:
    def __init__(self):
        # State
        self.state = KWDState.INIT

        # Config
        self.config: Optional[configparser.ConfigParser] = None

        # Ports / URLs (bases + routes; backwards compatible with full URLs in base)
        self.port = 5002
        self.logger_base = "http://127.0.0.1:5000"
        self.rms_base = "http://127.0.0.1:5006"
        self.tts_base = "http://127.0.0.1:5005"  # may already include path in older configs
        self.stt_base = "http://127.0.0.1:5003"  # may already include path in older configs

        # Routes (can be empty; if base already has a path, we won't append)
        self.logger_route = "/log"
        self.rms_route = "/current-rms"
        self.tts_route = "/speak"
        self.stt_route = "/start"

        # Wake word
        self.model_path = "models/alexa_v0.1.onnx"

        # Thresholding (base sensitivity and adaptive multipliers)
        self.base_threshold = 0.50
        self.current_threshold = self.base_threshold
        self.quiet_dbfs = -60.0
        self.noisy_dbfs = -35.0
        self.quiet_factor = 0.80
        self.noisy_factor = 1.25

        # Cooldown
        self.cooldown_ms = 1000
        self.last_detection_time = 0.0
        self.detection_lock = threading.Lock()

        # Audio settings
        self.sample_rate = 16000
        self.frame_ms = 80  # window size for detector
        self.chunk_size = int(self.sample_rate * self.frame_ms / 1000)  # 1280
        self.audio_device: Optional[int] = None  # device index or None (default)

        # Audio stream + buffer (int16)
        self.audio_stream: Optional[sd.InputStream] = None
        self.audio_buffer: List[int] = []
        self.buffer_lock = threading.Lock()

        # DC-block (simple one-pole high-pass)
        self.hp_enabled = True
        self.hp_prev_x = 0.0
        self.hp_prev_y = 0.0
        self.hp_alpha = 0.995  # ~50–70 Hz corner @16k

        # RMS gating for “dead mic” frames
        self.min_frame_rms = 50  # int16 scale

        # Wake word engine
        self.detector: Optional[Model] = None
        self.detection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Phrases
        self.yes_phrases = ["Yes?", "Yes Master?", "What's up?", "I'm listening", "Yo", "Here!"]
        self.greeting_text = "Hi Master. Ready to serve."

        # Adaptive update timer
        self._last_thr_update = 0.0
        self._thr_update_interval_s = 5.0

    # -----------------------------
    # URL helpers (fix for 404 regression)
    # -----------------------------
    @staticmethod
    def _compose_url(base: str, route: str) -> str:
        """
        Backward-compatible URL joiner:
        - If route is absolute (starts with http), return route.
        - If base already ends with the same route, return base.
        - If base already includes a path (e.g., .../speak) and route == '/speak', don't duplicate.
        - Else join cleanly.
        """
        if not base:
            return route or ""
        if route and route.startswith("http"):
            return route

        base_clean = base.rstrip("/")
        route_clean = (route or "").lstrip("/")

        # If base already contains a path and ends with route, don't append
        if route_clean and base_clean.split("://")[-1].rstrip("/").endswith(route_clean):
            return base_clean

        # If route empty, use base as-is
        if not route_clean:
            return base_clean

        return f"{base_clean}/{route_clean}"

    def _url_logger(self) -> str:
        return self._compose_url(self.logger_base, self.logger_route)

    def _url_rms(self) -> str:
        return self._compose_url(self.rms_base, self.rms_route)

    def _url_tts(self) -> str:
        return self._compose_url(self.tts_base, self.tts_route)

    def _url_stt(self) -> str:
        return self._compose_url(self.stt_base, self.stt_route)

    # -----------------------------
    # Config / Logging
    # -----------------------------
    def load_config(self) -> bool:
        """Load configuration from config/config.ini (non-fatal: falls back to defaults)."""
        try:
            cfg_path = Path("config/config.ini")
            if not cfg_path.exists():
                self.log("warning", "config/config.ini not found, using defaults")
                return True

            cfg = configparser.ConfigParser()
            cfg.read(cfg_path)
            self.config = cfg

            # [kwd]
            if "kwd" in cfg:
                s = cfg["kwd"]
                self.model_path = s.get("model_path", self.model_path)
                self.base_threshold = s.getfloat("base_threshold", self.base_threshold)
                self.current_threshold = self.base_threshold
                self.cooldown_ms = s.getint("cooldown_ms", self.cooldown_ms)
                self.port = s.getint("port", self.port)
                yes_phrases_str = s.get("yes_phrases", ", ".join(self.yes_phrases))
                self.yes_phrases = [p.strip() for p in yes_phrases_str.split(",")]
                self.greeting_text = s.get("greeting_text", self.greeting_text)

            # [kwd.adaptive]
            if "kwd.adaptive" in cfg:
                s = cfg["kwd.adaptive"]
                self.quiet_dbfs = s.getfloat("quiet_dbfs", self.quiet_dbfs)
                self.noisy_dbfs = s.getfloat("noisy_dbfs", self.noisy_dbfs)
                self.quiet_factor = s.getfloat("quiet_factor", self.quiet_factor)
                self.noisy_factor = s.getfloat("noisy_factor", self.noisy_factor)

            # [kwd.deps]
            if "kwd.deps" in cfg:
                s = cfg["kwd.deps"]
                # Bases (may include full paths from older configs)
                self.logger_base = s.get("logger_url", self.logger_base)
                self.rms_base = s.get("rms_url", self.rms_base)
                self.tts_base = s.get("tts_url", self.tts_base)   # keep backward compatibility
                self.stt_base = s.get("stt_url", self.stt_base)   # keep backward compatibility
                # Optional routes (empty allowed)
                self.logger_route = s.get("logger_route", self.logger_route)
                self.rms_route = s.get("rms_route", self.rms_route)
                self.tts_route = s.get("tts_route", self.tts_route)
                self.stt_route = s.get("stt_route", self.stt_route)

            # [audio]
            if "audio" in cfg:
                s = cfg["audio"]
                if s.get("device_index", "").strip():
                    try:
                        self.audio_device = int(s.get("device_index"))
                    except ValueError:
                        self.audio_device = None
                else:
                    self.audio_device = None
                self.sample_rate = s.getint("sample_rate", self.sample_rate)
                self.frame_ms = s.getint("frame_ms", self.frame_ms)
                self.chunk_size = int(self.sample_rate * self.frame_ms / 1000)
                self.min_frame_rms = s.getint("min_frame_rms", self.min_frame_rms)
                self.hp_enabled = s.getboolean("dc_block", self.hp_enabled)

            return True
        except Exception as e:
            self.log("error", f"Failed to load config: {e}")
            return False

    def log(self, level: str, message: str, event: str = None, extra: Dict[str, Any] = None):
        """Send structured log to Logger or fallback to stdout."""
        try:
            payload = {"svc": "KWD", "level": level, "message": message}
            if event:
                payload["event"] = event
            if extra:
                payload["extra"] = extra
            requests.post(self._url_logger(), json=payload, timeout=1.5)
        except Exception:
            print(f"KWD       {level.upper():<6}= {message}")

    # -----------------------------
    # RMS → Adaptive threshold
    # -----------------------------
    def get_current_rms_dbfs(self) -> Optional[float]:
        """Query RMS service (optional)."""
        try:
            r = requests.get(self._url_rms(), timeout=1.0)
            if r.status_code == 200:
                data = r.json()
                return data.get("rms_dbfs")
        except Exception as e:
            self.log("debug", f"RMS query failed: {e}")
        return None

    def _interp_factor(self, noise_dbfs: float) -> float:
        if noise_dbfs <= self.quiet_dbfs:
            return self.quiet_factor
        if noise_dbfs >= self.noisy_dbfs:
            return self.noisy_factor
        ratio = (noise_dbfs - self.quiet_dbfs) / (self.noisy_dbfs - self.quiet_dbfs)
        return self.quiet_factor + ratio * (self.noisy_factor - self.quiet_factor)

    def update_threshold_from_rms(self):
        """Recalculate detector threshold from ambient noise."""
        rms_dbfs = self.get_current_rms_dbfs()
        if rms_dbfs is None:
            return
        new_thr = self.base_threshold * self._interp_factor(rms_dbfs)
        if abs(new_thr - self.current_threshold) >= 0.01:
            self.current_threshold = float(np.clip(new_thr, 0.05, 0.99))
            self.log(
                "debug",
                f"Adaptive threshold → {self.current_threshold:.3f} (RMS {rms_dbfs:.1f} dBFS)",
            )

    # -----------------------------
    # Model / Audio
    # -----------------------------
    def load_wake_word_model(self) -> bool:
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                self.log("error", f"Wake word model not found: {self.model_path}")
                return False

            self.log("info", f"Loading wake word model: {self.model_path}")
            self.detector = Model(wakeword_model_paths=[str(model_file)])
            self.log("info", "Wake word model loaded", "model_loaded")
            return True
        except Exception as e:
            self.log("error", f"Model load failed: {e}")
            return False

    def _dc_block(self, x: np.ndarray) -> np.ndarray:
        if not self.hp_enabled:
            return x
        # y[n] = x[n] - x[n-1] + a * y[n-1]
        y = np.empty_like(x, dtype=np.float32)
        prev_x = self.hp_prev_x
        prev_y = self.hp_prev_y
        a = self.hp_alpha
        xf = x.astype(np.float32)
        for i in range(len(xf)):
            y_i = xf[i] - prev_x + a * prev_y
            prev_x = xf[i]
            prev_y = y_i
            y[i] = y_i
        self.hp_prev_x = float(prev_x)
        self.hp_prev_y = float(prev_y)
        y = np.clip(y, -32768.0, 32767.0)
        return y.astype(np.int16)

    def audio_callback(self, indata, frames, time_info, status):
        """Audio capture callback → accumulate int16 samples."""
        if status:
            self.log("warning", f"Audio callback status: {status}")

        pcm = indata
        if pcm.ndim == 2:
            pcm = pcm[:, 0]
        if pcm.dtype != np.int16:
            pf = pcm.astype(np.float32)
            pf = np.clip(pf, -1.0, 1.0)
            pcm16 = (pf * 32767.0).astype(np.int16)
        else:
            pcm16 = pcm

        pcm16 = self._dc_block(pcm16)

        with self.buffer_lock:
            self.audio_buffer.extend(pcm16.tolist())
            max_len = self.sample_rate * 2  # keep ~2 seconds max
            if len(self.audio_buffer) > max_len:
                self.audio_buffer = self.audio_buffer[-max_len:]

    def start_audio_stream(self) -> bool:
        """Open sounddevice.InputStream with desired device/index."""
        try:
            self.audio_stream = sd.InputStream(
                device=self.audio_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=self.audio_callback,
                blocksize=self.chunk_size,  # 80 ms chunks
            )
            self.audio_stream.start()
            self.log("info", f"Audio stream started (device={self.audio_device}, fs={self.sample_rate})")
            return True
        except Exception as e:
            self.log("error", f"Failed to start audio stream: {e}")
            return False

    def stop_audio_stream(self):
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                self.log("warning", f"Error stopping audio stream: {e}")
            finally:
                self.audio_stream = None
                self.log("info", "Audio stream stopped")

    # -----------------------------
    # Detection Loop
    # -----------------------------
    def detection_worker(self):
        self.log("info", "Detection worker started")
        while not self.stop_event.is_set():
            try:
                if self.state != KWDState.ACTIVE:
                    time.sleep(0.1)
                    continue

                # Pull a frame
                with self.buffer_lock:
                    if len(self.audio_buffer) < self.chunk_size:
                        frame = None
                    else:
                        frame = np.array(self.audio_buffer[:self.chunk_size], dtype=np.int16)
                        self.audio_buffer = self.audio_buffer[self.chunk_size:]

                if frame is None:
                    time.sleep(0.01)
                    continue

                # Frame gating by RMS (dead mic / wrong source)
                rms = int(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                if rms < self.min_frame_rms:
                    continue

                # Run prediction
                if self.detector:
                    predictions = self.detector.predict(frame)
                    for model_name, score in predictions.items():
                        if score >= self.current_threshold:
                            self.handle_wake_detection(float(score))
                            break

                # Adaptive threshold every ~5s
                now = time.time()
                if now - self._last_thr_update > self._thr_update_interval_s:
                    self.update_threshold_from_rms()
                    self._last_thr_update = now

            except Exception as e:
                self.log("error", f"Detection worker error: {e}")
                time.sleep(0.05)

        self.log("info", "Detection worker stopped")

    def handle_wake_detection(self, confidence: float):
        now = time.time()
        with self.detection_lock:
            if (now - self.last_detection_time) * 1000 < self.cooldown_ms:
                return
            self.last_detection_time = now

        self.state = KWDState.TRIGGERED
        self.log(
            "info",
            f"Wake word detected (score={confidence:.3f}, thr={self.current_threshold:.3f})",
            "wake_detected",
            {"score": confidence, "threshold": self.current_threshold},
        )

        dialog_id = f"dialog_{uuid.uuid4().hex[:8]}"

        # Move to SLEEP so we don't double-trigger while TTS/STT spin up
        self.state = KWDState.SLEEP
        threading.Thread(target=self._cooldown_reattach, daemon=True).start()

        # Fire-and-forget side effects
        threading.Thread(target=self.trigger_tts_confirmation, args=(dialog_id,), daemon=True).start()
        threading.Thread(target=self.trigger_stt_dialog, args=(dialog_id,), daemon=True).start()

    def _cooldown_reattach(self):
        time.sleep(self.cooldown_ms / 1000.0)
        if self.state == KWDState.SLEEP:
            self.state = KWDState.ACTIVE
            self.log("debug", "Cooldown done → ACTIVE")

    # -----------------------------
    # Side Effects: TTS / STT
    # -----------------------------
    def trigger_tts_confirmation(self, dialog_id: str):
        phrase = random.choice(self.yes_phrases)
        url = self._url_tts()
        try:
            payload = TTSSpeakRequest(text=phrase, dialog_id=dialog_id).dict()
            r = requests.post(url, json=payload, timeout=8.0)
            if r.status_code == 200:
                self.log("debug", f"TTS confirmation sent: {phrase}", "tts_called", {"url": url})
            else:
                self.log("warning", f"TTS confirmation failed: {r.status_code}", extra={"url": url, "body": r.text})
        except Exception as e:
            self.log("warning", f"TTS confirmation error: {e}", extra={"url": url})

    def trigger_stt_dialog(self, dialog_id: str):
        url = self._url_stt()
        try:
            payload = STTStartRequest(dialog_id=dialog_id).dict()
            r = requests.post(url, json=payload, timeout=8.0)
            if r.status_code == 200:
                self.log("info", f"STT dialog started: {dialog_id}", "stt_started", {"url": url})
            else:
                self.log("warning", f"STT start failed: {r.status_code}", extra={"url": url, "body": r.text})
        except Exception as e:
            self.log("warning", f"STT start error: {e}", extra={"url": url})

    # -----------------------------
    # Start/Stop
    # -----------------------------
    def start_detection(self) -> bool:
        if self.state == KWDState.INIT:
            return False
        if self.detection_thread and self.detection_thread.is_alive():
            return True

        if not self.start_audio_stream():
            return False

        self.stop_event.clear()
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        self.state = KWDState.ACTIVE
        self.log("info", "Wake word detection started", "kwd_started")
        return True

    def stop_detection(self):
        self.state = KWDState.SLEEP
        self.stop_event.set()
        self.stop_audio_stream()
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.5)
        self.log("info", "Wake word detection stopped", "kwd_stopped")

    def play_greeting_and_activate(self):
        """Say greeting, then arm detection (even if TTS fails)."""
        url = self._url_tts()
        try:
            payload = TTSSpeakRequest(text=self.greeting_text).dict()
            r = requests.post(url, json=payload, timeout=8.0)
            if r.status_code == 200:
                self.log("info", f"Greeting played: '{self.greeting_text}'", "greeting_played", {"url": url})
                time.sleep(0.5)
            else:
                self.log("warning", f"Greeting TTS failed: {r.status_code}", extra={"url": url, "body": r.text})
        except Exception as e:
            self.log("warning", f"Greeting error: {e}", extra={"url": url})
        finally:
            self.start_detection()

    # -----------------------------
    # Lifecycle
    # -----------------------------
    async def startup(self) -> bool:
        try:
            self.log("info", "KWD service starting", "service_start")
            if not self.load_config():
                self.log("error", "Configuration load failed")
                return False
            if not self.load_wake_word_model():
                self.log("error", "Wake word model load failed")
                return False
            self.state = KWDState.READY
            self.log("info", "KWD service ready", "service_ready")
            return True
        except Exception as e:
            self.log("error", f"KWD startup failed: {e}")
            return False

    async def shutdown(self):
        self.log("info", "KWD service shutting down")
        self.stop_detection()

    # -----------------------------
    # Diagnostics
    # -----------------------------
    def list_input_devices(self) -> List[Dict[str, Any]]:
        """Return a compact list of input-capable devices for debugging selection."""
        out = []
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if int(dev.get("max_input_channels", 0)) > 0:
                    out.append(
                        {
                            "index": idx,
                            "name": dev.get("name"),
                            "hostapi": dev.get("hostapi"),
                            "default_samplerate": dev.get("default_samplerate"),
                            "max_input_channels": dev.get("max_input_channels"),
                        }
                    )
        except Exception as e:
            self.log("warning", f"Device query failed: {e}")
        return out


# -----------------------------
# Global Instance
# -----------------------------
kwd_service = KWDService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await kwd_service.startup()
    yield
    await kwd_service.shutdown()


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="KWD Service",
    description="Wake Word Detection using openWakeWord",
    version="1.2",
    lifespan=lifespan,
)


@app.post("/on-system-ready")
async def on_system_ready():
    """Called by loader after all services are READY."""
    if kwd_service.state not in (KWDState.READY, KWDState.SLEEP):
        raise HTTPException(status_code=400, detail=f"Service not ready (state={kwd_service.state})")
    kwd_service.log("info", "System ready received", "system_ready")
    threading.Thread(target=kwd_service.play_greeting_and_activate, daemon=True).start()
    return {"status": "ok", "message": "Greeting + activation started"}

@app.post("/start")
async def start_detection(_req: StartRequest = StartRequest()):
    """Arm or re-arm detection."""
    if kwd_service.state == KWDState.INIT:
        raise HTTPException(status_code=400, detail="Service not initialized")
    if kwd_service.start_detection():
        kwd_service.log("info", "Detection armed", "rearmed")
        return {"status": "ok", "state": kwd_service.state}
    raise HTTPException(status_code=500, detail="Failed to start detection")

@app.post("/stop")
async def stop_detection_endpoint(req: StopRequest = StopRequest()):
    """Stop detection (sleep)."""
    kwd_service.stop_detection()
    kwd_service.log("info", f"Detection stopped: {req.reason or 'manual'}", "kwd_stopped")
    return {"status": "ok", "state": kwd_service.state}

@app.get("/health")
async def health_check():
    """Health: ok when not INIT."""
    status = "ok" if kwd_service.state != KWDState.INIT else "error"
    return HealthResponse(status=status, state=kwd_service.state)

@app.get("/state")
async def get_state():
    return StateResponse(state=kwd_service.state)

@app.get("/devices")
async def list_devices():
    """List input-capable devices to help select the correct mic."""
    return {"devices": kwd_service.list_input_devices()}


def main():
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=kwd_service.port,
        log_level="error",
        access_log=False,
    )


if __name__ == "__main__":
    main()
