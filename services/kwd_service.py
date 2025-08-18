#!/usr/bin/env python3
"""
KWD Service - Wake Word Detection using openWakeWord

FastAPI service on port 5002 that:
- Continuously listens for wake words using openWakeWord (ONNX model path configurable)
- Adapts sensitivity based on ambient noise from RMS service (optional)
- Triggers TTS confirmation and STT dialog on detection
- Manages state: INIT → READY → ACTIVE → TRIGGERED → SLEEP → ACTIVE
- Provides endpoints for loader coordination and basic diagnostics

Echo/loop prevention (this version):
- Longer default cooldown (2000 ms)
- Post-fire debounce window (e.g., 2000 ms) that ignores all detections
- Re-arm requires a period of silence (e.g., 800–1200 ms) before returning to ACTIVE
- Optional consecutive-frame requirement before firing (default 2 frames)

Notes:
- Audio is captured @16 kHz, mono, int16. Detector expects int16 PCM.
- Chunking uses 80 ms windows (1280 samples) as recommended for stable streaming features.
- Includes basic DC-block (one-pole high-pass) and RMS gating to avoid “dead mic” noise.
- Device selection is configurable via [audio] section.
"""

import sys
import time
import threading
import random
import uuid
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from .config_loader import app_config
from .shared import lifespan_with_httpx, safe_post, safe_get

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

        # Load configuration from config_loader
        kwd_config = app_config.kwd
        kwd_adaptive_config = app_config.kwd_adaptive
        kwd_deps_config = app_config.kwd_deps
        audio_config = app_config.audio

        # Ports / URLs (bases + routes; backwards compatible with full URLs in base)
        self.port = kwd_config.port
        self.logger_base = kwd_deps_config.logger_url
        self.rms_base = kwd_deps_config.rms_url
        self.tts_base = kwd_deps_config.tts_url
        self.stt_base = kwd_deps_config.stt_url

        # Routes (can be empty; if base already has a path, we won't append)
        self.logger_route = kwd_deps_config.logger_route
        self.rms_route = kwd_deps_config.rms_route
        self.tts_route = kwd_deps_config.tts_route
        self.stt_route = kwd_deps_config.stt_route

        # Wake word
        self.model_path = kwd_config.model_path

        # Thresholding (base sensitivity and adaptive multipliers)
        self.base_threshold = kwd_config.base_threshold
        self.current_threshold = self.base_threshold
        self.quiet_dbfs = kwd_adaptive_config.quiet_dbfs
        self.noisy_dbfs = kwd_adaptive_config.noisy_dbfs
        self.quiet_factor = kwd_adaptive_config.quiet_factor
        self.noisy_factor = kwd_adaptive_config.noisy_factor

        # Echo / re-trigger guards
        self.cooldown_ms = kwd_config.cooldown_ms
        self.debounce_after_tts_ms = kwd_config.debounce_after_tts_ms
        self.silence_required_ms = kwd_config.silence_required_ms
        self.rearm_timeout_ms = kwd_config.rearm_timeout_ms
        self.fire_cons_frames = kwd_config.fire_cons_frames

        self.debounce_until = 0.0               # epoch seconds
        self.last_detection_time = 0.0
        self.detection_lock = threading.Lock()
        self.in_progress = False                # one-shot guard to prevent parallel side-effects

        # Audio settings
        self.sample_rate = audio_config.sample_rate
        self.frame_ms = kwd_config.frame_ms
        self.chunk_size = int(self.sample_rate * self.frame_ms / 1000)  # 1280
        self.audio_device = audio_config.device_index if audio_config.device_index is not None else None

        # Audio stream + buffer (int16)
        self.audio_stream: Optional[sd.InputStream] = None
        self.audio_buffer: List[int] = []
        self.buffer_lock = threading.Lock()

        # DC-block (simple one-pole high-pass)
        self.hp_enabled = audio_config.dc_block
        self.hp_prev_x = 0.0
        self.hp_prev_y = 0.0
        self.hp_alpha = 0.995  # ~50–70 Hz corner @16k

        # RMS gating for "dead mic" frames
        self.min_frame_rms = audio_config.min_frame_rms

        # Wake word engine
        self.detector: Optional[Model] = None
        self.detection_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Phrases
        self.yes_phrases = kwd_config.yes_phrases
        self.greeting_text = kwd_config.greeting_text

        # Adaptive update timer
        self._last_thr_update = 0.0
        self._thr_update_interval_s = 5.0

        # Detector hit counter (for consecutive-frame requirement)
        self._hit_count = 0

    # -----------------------------
    # URL helpers
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
    # Logging
    # -----------------------------

    async def log(self, level: str, message: str, event: str = None, extra: Dict[str, Any] = None):
        """Send structured log to Logger or fallback to stdout."""
        payload = {"svc": "KWD", "level": level, "message": message}
        if event:
            payload["event"] = event
        if extra:
            payload["extra"] = extra
        
        fallback_msg = f"KWD       {level.upper():<6}= {message}"
        await safe_post(self._url_logger(), json=payload, timeout=1.5, fallback_msg=fallback_msg)

    # -----------------------------
    # RMS → Adaptive threshold
    # -----------------------------
    async def get_current_rms_dbfs(self) -> Optional[float]:
        """Query RMS service (optional)."""
        try:
            data = await safe_get(self._url_rms(), timeout=1.0)
            if data:
                return data.get("rms_dbfs")
        except Exception as e:
            await self.log("debug", f"RMS query failed: {e}")
        return None

    def _interp_factor(self, noise_dbfs: float) -> float:
        if noise_dbfs <= self.quiet_dbfs:
            return self.quiet_factor
        if noise_dbfs >= self.noisy_dbfs:
            return self.noisy_factor
        ratio = (noise_dbfs - self.quiet_dbfs) / (self.noisy_dbfs - self.quiet_dbfs)
        return self.quiet_factor + ratio * (self.noisy_factor - self.quiet_factor)

    async def update_threshold_from_rms(self):
        """Recalculate detector threshold from ambient noise."""
        rms_dbfs = await self.get_current_rms_dbfs()
        if rms_dbfs is None:
            return
        new_thr = self.base_threshold * self._interp_factor(rms_dbfs)
        if abs(new_thr - self.current_threshold) >= 0.01:
            self.current_threshold = float(np.clip(new_thr, 0.05, 0.99))
            await self.log(
                "debug",
                f"Adaptive threshold → {self.current_threshold:.3f} (RMS {rms_dbfs:.1f} dBFS)",
            )

    # -----------------------------
    # Model / Audio
    # -----------------------------
    async def load_wake_word_model(self) -> bool:
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                await self.log("error", f"Wake word model not found: {self.model_path}")
                return False

            await self.log("info", f"Loading wake word model: {self.model_path}")
            self.detector = Model(wakeword_model_paths=[str(model_file)])
            await self.log("info", "Wake word model loaded", "model_loaded")
            return True
        except Exception as e:
            await self.log("error", f"Model load failed: {e}")
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
            # Can't await in audio callback, use sync fallback
            print(f"KWD       WARNING= Audio callback status: {status}")

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

    async def start_audio_stream(self) -> bool:
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
            await self.log("info", f"Audio stream started (device={self.audio_device}, fs={self.sample_rate})")
            return True
        except Exception as e:
            await self.log("error", f"Failed to start audio stream: {e}")
            return False

    async def stop_audio_stream(self):
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                await self.log("warning", f"Error stopping audio stream: {e}")
            finally:
                self.audio_stream = None
                await self.log("info", "Audio stream stopped")

    # -----------------------------
    # Silence wait for re-arm
    # -----------------------------
    def _rms_of_tail_ms(self, tail_ms: int) -> int:
        """Compute RMS of last 'tail_ms' milliseconds in buffer."""
        with self.buffer_lock:
            n = int(self.sample_rate * tail_ms / 1000)
            if n <= 0 or len(self.audio_buffer) < n:
                return 0
            arr = np.array(self.audio_buffer[-n:], dtype=np.int16).astype(np.float32)
        return int(np.sqrt(np.mean(arr ** 2)))

    def _wait_for_silence(self, required_ms: int, timeout_ms: int) -> bool:
        """
        Wait until we observe 'required_ms' of continuous silence (RMS < min_frame_rms).
        Break early if timeout_ms exceeded. Uses short polling windows to remain responsive.
        """
        start = time.time()
        continuous_ms = 0
        poll_ms = min(80, required_ms // 4 or 20)  # 20–80 ms polls
        while (time.time() - start) * 1000.0 < timeout_ms:
            rms = self._rms_of_tail_ms(poll_ms)
            if rms < self.min_frame_rms:
                continuous_ms += poll_ms
                if continuous_ms >= required_ms:
                    return True
            else:
                continuous_ms = 0
            time.sleep(poll_ms / 1000.0 * 0.6)  # small sleep, keep responsive
        return False

    # -----------------------------
    # Detection Loop
    # -----------------------------
    def detection_worker(self):
        # Use sync fallback logging in worker thread
        print("KWD       INFO  = Detection worker started")
        while not self.stop_event.is_set():
            try:
                if self.state != KWDState.ACTIVE:
                    time.sleep(0.05)
                    continue

                # Global debounce window (ignore everything while active)
                now = time.time()
                if now < self.debounce_until:
                    time.sleep(0.01)
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
                frms = int(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                if frms < self.min_frame_rms:
                    self._hit_count = 0
                    continue

                # Run prediction
                if self.detector:
                    predictions = self.detector.predict(frame)
                    fired = False
                    for _, score in predictions.items():
                        if score >= self.current_threshold:
                            self._hit_count += 1
                            if self._hit_count >= self.fire_cons_frames:
                                fired = True
                                self._hit_count = 0
                                break
                        else:
                            # reset if any head falls below on this frame
                            self._hit_count = 0

                    if fired:
                        self.handle_wake_detection(float(score))
                        # state changes to SLEEP inside handle_wake_detection
                        continue

                # Adaptive threshold every ~5s
                if now - self._last_thr_update > self._thr_update_interval_s:
                    # Run async method from thread using asyncio bridge
                    try:
                        loop = asyncio.get_event_loop()
                        asyncio.run_coroutine_threadsafe(
                            self.update_threshold_from_rms(), loop
                        )
                    except:
                        pass  # Skip if no event loop available
                    self._last_thr_update = now

            except Exception as e:
                print(f"KWD       ERROR = Detection worker error: {e}")
                time.sleep(0.05)

        print("KWD       INFO  = Detection worker stopped")

    def handle_wake_detection(self, confidence: float):
        # one-shot guard to prevent parallel side-effects
        with self.detection_lock:
            now = time.time()

            # basic cooldown
            if (now - self.last_detection_time) * 1000 < self.cooldown_ms:
                return

            # hard debounce window (covers speaker playback echo)
            if now < self.debounce_until:
                return

            if self.in_progress:
                return
            self.in_progress = True
            self.last_detection_time = now
            # set a post-fire debounce window
            self.debounce_until = now + (self.debounce_after_tts_ms / 1000.0)

        self.state = KWDState.TRIGGERED
        # Use asyncio bridge for logging from this thread
        try:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                self.log(
                    "info",
                    f"Wake word detected (score={confidence:.3f}, thr={self.current_threshold:.3f})",
                    "wake_detected",
                    {"score": confidence, "threshold": self.current_threshold},
                ), loop
            )
        except:
            print(f"KWD       INFO  = Wake word detected (score={confidence:.3f}, thr={self.current_threshold:.3f})")

        dialog_id = f"dialog_{uuid.uuid4().hex[:8]}"

        # Move to SLEEP so we don't double-trigger while TTS/STT spin up
        self.state = KWDState.SLEEP

        # Fire-and-forget side effects
        threading.Thread(target=self.trigger_tts_confirmation, args=(dialog_id,), daemon=True).start()
        threading.Thread(target=self.trigger_stt_dialog, args=(dialog_id,), daemon=True).start()

        # Begin re-arm workflow
        threading.Thread(target=self._cooldown_and_rearm, daemon=True).start()

    def _cooldown_and_rearm(self):
        """
        Wait cooldown, then require a window of silence before re-arming.
        If silence doesn't arrive before timeout, re-arm anyway (failsafe).
        """
        try:
            time.sleep(self.cooldown_ms / 1000.0)

            # Require continuous silence to avoid echo re-triggers
            got_silence = self._wait_for_silence(
                required_ms=self.silence_required_ms,
                timeout_ms=self.rearm_timeout_ms,
            )

            if not got_silence:
                print("KWD       DEBUG = Re-arm without silence (timeout reached)")

            # Re-arm
            self.state = KWDState.ACTIVE
            self.in_progress = False
            print("KWD       DEBUG = Cooldown + silence satisfied → ACTIVE")
        except Exception as e:
            self.in_progress = False
            self.state = KWDState.ACTIVE
            print(f"KWD       WARNING= Re-arm workflow error: {e}")

    # -----------------------------
    # Side Effects: TTS / STT
    # -----------------------------
    def trigger_tts_confirmation(self, dialog_id: str):
        phrase = random.choice(self.yes_phrases)
        url = self._url_tts()
        try:
            payload = TTSSpeakRequest(text=phrase, dialog_id=dialog_id).dict()
            # Use asyncio bridge for HTTP call from thread
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    safe_post(url, json=payload, timeout=8.0, 
                             fallback_msg=f"KWD       DEBUG = TTS confirmation sent: {phrase}"), 
                    loop
                )
            except:
                print(f"KWD       DEBUG = TTS confirmation sent: {phrase}")
        except Exception as e:
            print(f"KWD       WARNING= TTS confirmation error: {e}")

    def trigger_stt_dialog(self, dialog_id: str):
        url = self._url_stt()
        try:
            payload = STTStartRequest(dialog_id=dialog_id).dict()
            # Use asyncio bridge for HTTP call from thread
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    safe_post(url, json=payload, timeout=8.0,
                             fallback_msg=f"KWD       INFO  = STT dialog started: {dialog_id}"),
                    loop
                )
            except:
                print(f"KWD       INFO  = STT dialog started: {dialog_id}")
        except Exception as e:
            print(f"KWD       WARNING= STT start error: {e}")

    # -----------------------------
    # Start/Stop
    # -----------------------------
    async def start_detection(self) -> bool:
        if self.state == KWDState.INIT:
            return False
        if self.detection_thread and self.detection_thread.is_alive():
            return True

        if not await self.start_audio_stream():
            return False

        self.stop_event.clear()
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        self.state = KWDState.ACTIVE
        await self.log("info", "Wake word detection started", "kwd_started")
        return True

    async def stop_detection(self):
        self.state = KWDState.SLEEP
        self.stop_event.set()
        await self.stop_audio_stream()
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.5)
        self.in_progress = False
        await self.log("info", "Wake word detection stopped", "kwd_stopped")

    def play_greeting_and_activate(self):
        """Say greeting, then arm detection (even if TTS fails)."""
        url = self._url_tts()
        try:
            payload = TTSSpeakRequest(text=self.greeting_text).dict()
            # Use asyncio bridge for HTTP call from thread
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    safe_post(url, json=payload, timeout=8.0,
                             fallback_msg=f"KWD       INFO  = Greeting played: '{self.greeting_text}'"),
                    loop
                )
                time.sleep(0.5)
            except:
                print(f"KWD       INFO  = Greeting played: '{self.greeting_text}'")
        except Exception as e:
            print(f"KWD       WARNING= Greeting error: {e}")
        finally:
            # Start detection using asyncio bridge
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self.start_detection(), loop
                )
            except:
                pass  # Skip if no event loop available

    # -----------------------------
    # Lifecycle
    # -----------------------------
    async def startup(self) -> bool:
        try:
            await self.log("info", "KWD service starting", "service_start")
            if not await self.load_wake_word_model():
                await self.log("error", "Wake word model load failed")
                return False
            self.state = KWDState.READY
            await self.log("info", "KWD service ready", "service_ready")
            return True
        except Exception as e:
            await self.log("error", f"KWD startup failed: {e}")
            return False

    async def shutdown(self):
        await self.log("info", "KWD service shutting down")
        await self.stop_detection()

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
            # Use sync fallback in device query
            print(f"KWD       WARNING= Device query failed: {e}")
        return out


# -----------------------------
# Global Instance
# -----------------------------
kwd_service = KWDService()


@asynccontextmanager
async def service_lifespan():
    success = await kwd_service.startup()
    if not success:
        raise RuntimeError("KWD service startup failed")
    yield
    await kwd_service.shutdown()


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="KWD Service",
    description="Wake Word Detection using openWakeWord",
    version="1.3",
    lifespan=lambda app: lifespan_with_httpx(service_lifespan),
)


@app.post("/on-system-ready")
async def on_system_ready():
    """Called by loader after all services are READY."""
    if kwd_service.state not in (KWDState.READY, KWDState.SLEEP):
        raise HTTPException(status_code=400, detail=f"Service not ready (state={kwd_service.state})")
    await kwd_service.log("info", "System ready received", "system_ready")
    threading.Thread(target=kwd_service.play_greeting_and_activate, daemon=True).start()
    return {"status": "ok", "message": "Greeting + activation started"}

@app.post("/start")
async def start_detection(_req: StartRequest = StartRequest()):
    """Arm or re-arm detection."""
    if kwd_service.state == KWDState.INIT:
        raise HTTPException(status_code=400, detail="Service not initialized")
    if await kwd_service.start_detection():
        await kwd_service.log("info", "Detection armed", "rearmed")
        return {"status": "ok", "state": kwd_service.state}
    raise HTTPException(status_code=500, detail="Failed to start detection")

@app.post("/stop")
async def stop_detection_endpoint(req: StopRequest = StopRequest()):
    """Stop detection (sleep)."""
    await kwd_service.stop_detection()
    await kwd_service.log("info", f"Detection stopped: {req.reason or 'manual'}", "kwd_stopped")
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
