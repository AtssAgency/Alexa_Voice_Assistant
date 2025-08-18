#!/usr/bin/env python3
"""
STT Service - Speech-to-Text with Silero VAD & Dialog Session Management

FastAPI service on port [stt.port] that:
- Waits for wake word (KWD), then captures speech with Silero VAD start/stop gating
- Transcribes via faster-whisper (configurable), streams LLM response to TTS
- Re-arms KWD after dialog, exposes /devices, robust /health, graceful shutdown

State: INIT → READY → ACTIVE → READY
"""

import os
import time
import json
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Deque
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import sounddevice as sd
import websockets
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from .config_loader import app_config
from .shared import lifespan_with_httpx, safe_post, safe_get
from .utils import post_log, create_service_app

# ============
# Runtime env (stability for CTranslate2 / faster-whisper)
# ============
os.environ.setdefault("CT2_USE_MKL", "0")
os.environ.setdefault("CT2_FORCE_CPU_ISA", "AVX2")  # use "DEFAULT" if AVX2 missing

# =========================
# Dependencies (runtime)
# =========================
try:
    import torch
except Exception:
    print("ERROR: PyTorch is required for Silero VAD. Install it in your venv.")
    raise

try:
    # faster-whisper for transcription (keeps VRAM low per config)
    from faster_whisper import WhisperModel
except Exception:
    print("ERROR: faster-whisper is required. Install it in your venv.")
    raise


# -------------------------
# Models / DTOs
# -------------------------
class STTState:
    INIT = "INIT"
    READY = "READY"
    ACTIVE = "ACTIVE"


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


# =========================
# Silero VAD wrapper
# =========================
class SileroVAD:
    """
    Minimal Silero VAD wrapper.
    - Accepts int16 mono @ 16kHz frames.
    - Returns speech probability per frame (0..1).
    """
    def __init__(self, sample_rate: int = 16000, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        # Load Silero from torch.hub (snakers4)
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False
        )
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils
        self.model.to(self.device)
        self.model.eval()
        self._dtype = torch.float32

    @torch.no_grad()
    def prob(self, audio_int16: np.ndarray) -> float:
        """
        Returns model speech prob for a single frame buffer (int16).
        Internally converts to float32 [-1,1] torch tensor.
        """
        if audio_int16.ndim != 1:
            audio_int16 = audio_int16.reshape(-1)
        a = torch.tensor(audio_int16.astype(np.float32) / 32768.0, dtype=self._dtype, device=self.device)
        logits = self.model(a, self.sample_rate).squeeze(0)
        if logits.numel() == 0:
            return 0.0
        p = torch.sigmoid(logits).mean().item()
        return float(p)


# =========================
# STT Service
# =========================
class STTService:
    def __init__(self):
        self.state = STTState.INIT
        
        # --- Config is now clean and type-safe ---
        self.cfg = app_config.stt
        
        # Main service settings
        self.bind_host = self.cfg.bind_host
        self.port = self.cfg.port

        # Whisper
        self.model_size = self.cfg.model_size
        self.whisper_device = self.cfg.device
        self.compute_type = self.cfg.compute_type
        self.beam_size = self.cfg.beam_size
        self.word_timestamps = self.cfg.word_timestamps
        self.chunk_length_s = self.cfg.chunk_length_s

        # Follow-up
        self.follow_up_timer_sec = self.cfg.follow_up_timer_sec

        # Audio configuration
        self.audio_device_index = self.cfg.audio.device_index
        self.audio_device_name = ""  # preferred device by name (exact/substring)
        self.sample_rate = self.cfg.audio.sample_rate
        self.frame_ms = self.cfg.audio.frame_ms
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)
        self.dtype = self.cfg.audio.dtype
        self.dc_block = self.cfg.audio.dc_block
        self.min_frame_rms = self.cfg.audio.min_frame_rms

        # VAD / gating (Silero)
        self.vad_start_threshold = self.cfg.vad.start_threshold
        self.vad_end_threshold = self.cfg.vad.end_threshold
        self.vad_min_start_frames = self.cfg.vad.min_start_frames
        self.vad_max_silence_ms = self.cfg.vad.max_silence_ms
        self.vad_pre_roll_ms = self.cfg.vad.pre_roll_ms
        self.vad_speech_pad_ms = self.cfg.vad.speech_pad_ms

        # Adaptive (from RMS)
        self.quiet_dbfs = self.cfg.adaptive.quiet_dbfs
        self.noisy_dbfs = self.cfg.adaptive.noisy_dbfs

        # Dependencies
        self.logger_url = self.cfg.deps.logger_url
        self.llm_complete_url = self.cfg.deps.llm_url
        self.tts_ws_speak = self.cfg.deps.tts_ws_speak
        self.tts_ws_events = self.cfg.deps.tts_ws_events
        self.kwd_start_url = self.cfg.deps.kwd_url
        self.kwd_state_url = self.cfg.deps.kwd_url + "/state"  # Construct state URL
        self.rms_current_url = self.cfg.deps.rms_url

        # Runtime holders
        self.model: Optional[WhisperModel] = None
        self.vad: Optional[SileroVAD] = None

        self.active_dialog_id: Optional[str] = None
        self.stop_flag = threading.Event()
        self.dialog_thread: Optional[threading.Thread] = None

        self.audio_stream: Optional[sd.InputStream] = None
        self._raw_queue: Deque[np.ndarray] = deque(maxlen=2048)  # store int16 frames
        self._dc_prev = 0.0  # for DC-block filter

        # WS
        self.tts_speak_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.tts_events_ws: Optional[websockets.WebSocketClientProtocol] = None

    # ---------------
    # Logging
    # ---------------
    async def log(self, level: str, message: str, event: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        # Delegate logging to the shared utility
        await post_log(
            service="STT",
            level=level,
            message=message,
            logger_url=self.logger_url,
            event=event,
            extra=extra,
            timeout=1.5
        )

    async def log_dialog(self, dialog_id: str, role: str, text: str):
        await safe_post(
            f"{self.logger_url}/dialog/line",
            json={"dialog_id": dialog_id, "role": role, "text": text},
            timeout=1.5,
            fallback_msg=""  # No fallback for dialog logging
        )


    # ---------------
    # Audio utils
    # ---------------
    def _apply_dc_block(self, x: np.ndarray, alpha: float = 0.995) -> np.ndarray:
        # One-pole high-pass (DC-blocker): y[n] = x[n] - x[n-1] + alpha*y[n-1]
        y = np.empty_like(x, dtype=np.float32)
        prev_x = self._dc_prev
        prev_y = 0.0
        for i, xi in enumerate(x.astype(np.float32)):
            yi = xi - prev_x + alpha * prev_y
            y[i] = yi
            prev_x = xi
            prev_y = yi
        self._dc_prev = prev_x
        return y

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        x = x.astype(np.float32)
        return float(np.sqrt(np.mean(x * x)) + 1e-12)

    # ---------------
    # Device resolution
    # ---------------
    def _resolve_input_device(self) -> int:
        """
        Resolve an input-capable device index using:
        1) device_name (exact → substring),
        2) device_index from config (>=0),
        3) system's default input,
        4) first input-capable device.
        """
        try:
            devs = sd.query_devices()

            # 1) by name (exact → substring)
            name = (self.audio_device_name or "").strip().lower()
            if name:
                for i, d in enumerate(devs):
                    if d.get("max_input_channels", 0) > 0 and d.get("name", "").lower() == name:
                        return i
                for i, d in enumerate(devs):
                    if d.get("max_input_channels", 0) > 0 and name in d.get("name", "").lower():
                        return i

            # 2) by explicit index
            if self.audio_device_index is not None and int(self.audio_device_index) >= 0:
                i = int(self.audio_device_index)
                if 0 <= i < len(devs) and devs[i].get("max_input_channels", 0) > 0:
                    return i

            # 3) default input
            defaults = sd.default.device  # (input, output)
            if isinstance(defaults, (list, tuple)) and defaults and defaults[0] is not None:
                i = int(defaults[0])
                if 0 <= i < len(devs) and devs[i].get("max_input_channels", 0) > 0:
                    return i

            # 4) first input-capable
            for i, d in enumerate(devs):
                if d.get("max_input_channels", 0) > 0:
                    return i
        except Exception as e:
            # Use sync fallback in device resolution
            print(f"STT       WARNING= Device resolution error: {e}")

        return -1

    def _log_resolved_device(self, when: str = "startup"):
        idx = self._resolve_input_device()
        try:
            info = sd.query_devices(idx) if idx >= 0 else {}
        except Exception:
            info = {}
        # Use sync fallback for device logging during startup
        print(f"STT       INFO  = [{when}] Resolved input device → index={idx}, "
              f"name={info.get('name')}, rate={info.get('default_samplerate')}, "
              f"max_in={info.get('max_input_channels')}")

    # ---------------
    # Audio I/O
    # ---------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # Use sync fallback in audio callback
            print(f"STT       WARNING= Audio callback status: {status}")
        frame = indata[:, 0].copy()  # int16 mono
        self._raw_queue.append(frame)

    def _start_audio(self) -> bool:
        """
        Open an input stream using the resolved device.
        If no frames arrive within 500ms, retry once on system default (device=None).
        """
        try:
            self._raw_queue.clear()
            self._dc_prev = 0.0

            device_idx = self._resolve_input_device()
            self.audio_device_index = device_idx  # remember last chosen

            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=self.dtype,
                blocksize=self.frame_samples,
                device=(device_idx if device_idx >= 0 else None),
                callback=self._audio_callback,
            )
            self.audio_stream.start()
            # Use sync fallback for audio startup logging
            print(f"STT       INFO  = Audio started (device={device_idx}, {self.frame_ms}ms frames)")

            # Warm-up probe: ensure frames arrive (dead-mic guard)
            t0 = time.time()
            while not self._raw_queue and (time.time() - t0) < 0.5:
                time.sleep(0.01)

            if not self._raw_queue:
                print("STT       WARNING= No audio frames after 500ms; retrying on system default device=None")
                try:
                    if self.audio_stream:
                        self.audio_stream.stop()
                        self.audio_stream.close()
                except Exception:
                    pass
                self.audio_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=self.dtype,
                    blocksize=self.frame_samples,
                    device=None,  # force default (matches test_stt.py behavior)
                    callback=self._audio_callback,
                )
                self.audio_stream.start()
                print("STT       INFO  = Audio restarted on system default device")
            return True

        except Exception as e:
            print(f"STT       ERROR = Failed to start audio: {e}")
            return False

    def _stop_audio(self):
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                print(f"STT       WARNING= Error stopping audio: {e}")
            finally:
                self.audio_stream = None
                print("STT       INFO  = Audio stopped")

    # ---------------
    # VAD capture
    # ---------------
    def _capture_utterance_silero(self) -> Optional[np.ndarray]:
        """
        State machine:
        - IDLE/ARM: wait for min_start_frames with p>=start_threshold.
        - RECORD: accumulate frames, end when tail silence accumulated (p<end_th) for max_silence_ms.
        - Pre-roll frames preserved; speech_pad added at both ends.
        """
        if not self._start_audio():
            return None

        try:
            # Init VAD
            if self.vad is None:
                self.vad = SileroVAD(sample_rate=self.sample_rate, device="cpu")

            # Derived counters
            pre_roll_frames = max(1, int(self.vad_pre_roll_ms / self.frame_ms))
            tail_silence_frames = max(1, int(self.vad_max_silence_ms / self.frame_ms))
            speech_pad_frames = max(0, int(self.vad_speech_pad_ms / self.frame_ms))

            # Buffers
            preroll: Deque[np.ndarray] = deque(maxlen=pre_roll_frames)
            utter: List[np.ndarray] = []
            tail_silence = 0
            start_consec = 0
            started = False

            print("STT       INFO  = Listening for speech...")

            while not self.stop_flag.is_set():
                if not self._raw_queue:
                    time.sleep(self.frame_ms / 1000.0 * 0.5)
                    continue

                frame_i16 = self._raw_queue.popleft()

                # Optional DC-block + min RMS gate
                frame_f32 = frame_i16.astype(np.float32)
                if self.dc_block:
                    frame_f32 = self._apply_dc_block(frame_f32)

                if self._rms(frame_f32) < self.min_frame_rms:
                    # Very flat "dead mic" frame; treat as silence for arming.
                    preroll.append(frame_i16)
                    if started:
                        tail_silence += 1
                    continue

                # Silero prob on original int16 (per frame)
                p = self.vad.prob(frame_i16)

                if not started:
                    # Keep preroll fresh
                    preroll.append(frame_i16)

                    if p >= self.vad_start_threshold:
                        start_consec += 1
                    else:
                        start_consec = 0

                    if start_consec >= self.vad_min_start_frames:
                        started = True
                        # promote preroll + current to utter
                        utter.extend(list(preroll))
                        utter.append(frame_i16)
                        tail_silence = 0
                        preroll.clear()
                else:
                    # already recording
                    utter.append(frame_i16)
                    if p < self.vad_end_threshold:
                        tail_silence += 1
                        if tail_silence >= tail_silence_frames:
                            break
                    else:
                        tail_silence = 0

                # Safety cap
                max_frames = int(self.chunk_length_s * 1000 / self.frame_ms)
                if len(utter) > max_frames:
                    print("STT       WARNING= Max recording length reached")
                    break

            if not utter:
                print("STT       WARNING= No speech captured")
                return None

            # Add speech pads at both ends if available
            pad = np.zeros((self.frame_samples,), dtype=np.int16)
            for _ in range(speech_pad_frames):
                utter.insert(0, pad.copy())
                utter.append(pad.copy())

            # Concatenate and convert to float32 [-1,1] for ASR
            audio_i16 = np.concatenate(utter).astype(np.int16)
            audio_f32 = (audio_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
            return audio_f32

        except Exception as e:
            print(f"STT       ERROR = capture error: {e}")
            return None
        finally:
            self._stop_audio()

    # ---------------
    # Transcription
    # ---------------
    def _transcribe(self, audio_f32: np.ndarray) -> Optional[str]:
        try:
            segments, _info = self.model.transcribe(
                audio_f32,
                beam_size=self.beam_size,
                word_timestamps=self.word_timestamps,
                language="en"
            )
            text = " ".join([seg.text.strip() for seg in segments]).strip()
            if not text:
                print("STT       WARNING= Empty transcription")
                return None
            
            # Use asyncio bridge for logging transcription result
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self.log("info", f"USER: {text}", "stt_final_text"),
                    loop
                )
            except:
                print(f"STT       INFO  = USER: {text}")
            return text
        except Exception as e:
            print(f"STT       ERROR = Transcription error: {e}")
            return None

    # ---------------
    # TTS / LLM bridge
    # ---------------
    async def _connect_tts_ws(self, dialog_id: str) -> bool:
        try:
            self.tts_speak_ws = await websockets.connect(self.tts_ws_speak)
            self.tts_events_ws = await websockets.connect(f"{self.tts_ws_events}/{dialog_id}")
            await self.log("debug", "Connected to TTS WebSockets")
            return True
        except Exception as e:
            await self.log("error", f"TTS WS connect failed: {e}")
            return False

    async def _disconnect_tts_ws(self):
        try:
            if self.tts_speak_ws:
                await self.tts_speak_ws.close()
            if self.tts_events_ws:
                await self.tts_events_ws.close()
        except Exception as e:
            await self.log("warning", f"TTS WS disconnect error: {e}")
        finally:
            self.tts_speak_ws = None
            self.tts_events_ws = None

    async def _stream_llm_to_tts(self, dialog_id: str, user_text: str) -> bool:
        try:
            await self.log("info", "Starting LLM completion stream", "llm_stream_started")
            
            # Use httpx for streaming HTTP request
            import httpx
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", self.llm_complete_url, json={
                    "text": user_text,
                    "dialog_id": dialog_id
                }, timeout=120.0) as r:

                    if r.status_code != 200:
                        await self.log("error", f"LLM request failed: {r.status_code}")
                        return False

                    full = ""
                    async for line in r.aiter_lines():
                        if self.stop_flag.is_set():
                            break
                        if not line:
                            continue
                        s = line
                        if s.startswith("data: "):
                            payload = s[6:].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                j = json.loads(payload)
                                chunk = j.get("text", "")
                                if chunk and self.tts_speak_ws:
                                    await self.tts_speak_ws.send(json.dumps({"text_chunk": chunk}))
                                    full += chunk
                            except json.JSONDecodeError:
                                continue

                    if full:
                        await self.log("info", f"ALEXA: {full}", "llm_stream_end")
                        await self.log_dialog(dialog_id, "USER", user_text)
                        await self.log_dialog(dialog_id, "ALEXA", full)
                    return True
        except Exception as e:
            await self.log("error", f"LLM streaming error: {e}")
            return False

    async def _wait_tts_done(self, dialog_id: str) -> bool:
        try:
            if not self.tts_events_ws:
                return False
            await self.log("debug", "Waiting for TTS playback completion")
            while not self.stop_flag.is_set():
                try:
                    msg = await asyncio.wait_for(self.tts_events_ws.recv(), timeout=30.0)
                    ev = json.loads(msg)
                    et = ev.get("event")
                    if et == "playback_finished":
                        await self.log("info", "TTS playback completed", "tts_playback_finished")
                        return True
                    if et == "playback_error":
                        await self.log("warning", "TTS playback error")
                        return False
                except asyncio.TimeoutError:
                    await self.log("warning", "Timeout waiting for TTS completion")
                    return False
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            await self.log("error", f"TTS completion wait error: {e}")
            return False

    # ---------------
    # KWD re-arm (state-aware)
    # ---------------
    async def _rearm_kwd(self):
        try:
            data = await safe_get(self.kwd_state_url, timeout=1.5)
            if not data:
                await self.log("warning", "KWD state probe failed")
                return
            st = data.get("state", "").upper()
            if st not in ("READY", "SLEEP"):
                await self.log("info", f"KWD is {st}; skip re-arm")
                return
            
            result = await safe_post(self.kwd_start_url, timeout=2.0, fallback_msg="")
            if result is not None:
                await self.log("info", "KWD re-armed successfully")
            else:
                await self.log("warning", "KWD re-arm failed")
        except Exception as e:
            await self.log("warning", f"KWD re-arm error: {e}")

    # ---------------
    # Dialog worker
    # ---------------
    async def _dialog_worker(self, dialog_id: str):
        try:
            await self.log("info", f"Dialog started: {dialog_id}", "dialog_turn_started")
            # Start dialog in logger
            await safe_post(f"{self.logger_url}/dialog/start", json={"dialog_id": dialog_id}, timeout=1.5, fallback_msg="")

            # 1) Capture
            audio = self._capture_utterance_silero()
            if audio is None or self.stop_flag.is_set():
                await self.log("warning", "No user speech captured")
                return

            # 2) Transcribe
            text = self._transcribe(audio)
            if not text:
                return

            # 3) Connect TTS WS
            if not await self._connect_tts_ws(dialog_id):
                return

            # 4) Stream LLM → TTS
            if not await self._stream_llm_to_tts(dialog_id, text):
                return

            # 5) Wait TTS done
            await self._wait_tts_done(dialog_id)

            # 6) Optional follow-up delay
            await asyncio.sleep(self.follow_up_timer_sec)

        except Exception as e:
            await self.log("error", f"Dialog worker error: {e}")
        finally:
            await self._disconnect_tts_ws()
            # End dialog
            await safe_post(f"{self.logger_url}/dialog/end", json={"dialog_id": dialog_id}, timeout=1.5, fallback_msg="")
            
            # Rearm wake word
            await self._rearm_kwd()
            # Reset state
            self.active_dialog_id = None
            self.state = STTState.READY
            await self.log("info", f"Dialog ended: {dialog_id}", "dialog_ended")

    async def start_dialog(self, dialog_id: str) -> bool:
        if self.state != STTState.READY:
            await self.log("warning", f"Cannot start dialog in state {self.state}")
            return False
        if self.model is None:
            await self.log("warning", "Start rejected: Whisper not loaded yet")
            return False
        if self.active_dialog_id:
            await self.log("warning", f"Dialog already active: {self.active_dialog_id}")
            return False

        try:
            self.active_dialog_id = dialog_id
            self.state = STTState.ACTIVE
            self.stop_flag.clear()
            self.dialog_thread = threading.Thread(
                target=lambda: asyncio.run(self._dialog_worker(dialog_id)),
                daemon=True
            )
            self.dialog_thread.start()
            return True
        except Exception as e:
            await self.log("error", f"Failed to start dialog: {e}")
            self.active_dialog_id = None
            self.state = STTState.READY
            return False

    async def stop_dialog(self, dialog_id: str, reason: str = "manual") -> bool:
        if not self.active_dialog_id or self.active_dialog_id != dialog_id:
            return False
        try:
            await self.log("info", f"Stopping dialog: {reason}")
            self.stop_flag.set()
            if self.dialog_thread and self.dialog_thread.is_alive():
                self.dialog_thread.join(timeout=5.0)
            self.active_dialog_id = None
            self.state = STTState.READY
            return True
        except Exception as e:
            await self.log("error", f"Error stopping dialog: {e}")
            return False

    # ---------------
    # Lifecycle
    # ---------------
    def _load_asr_model(self) -> bool:
        try:
            # Use sync fallback for model loading in background thread
            print(f"STT       INFO  = Loading Whisper model: {self.model_size} on {self.whisper_device}")
            self.model = WhisperModel(
                self.model_size,
                device=self.whisper_device,
                compute_type=self.compute_type
            )
            print("STT       INFO  = Whisper model loaded")
            return True
        except Exception as e:
            print(f"STT       ERROR = Failed to load Whisper model: {e}")
            self.model = None
            return False

    async def startup(self):
        await self.log("info", "STT service starting", "service_start")
        
        # Log resolved audio device at startup (observability)
        self._log_resolved_device("startup")

        # Lazy-load Whisper in background so /health is reachable
        def _bg_load():
            ok = self._load_asr_model()
            if not ok:
                print("STT       ERROR = Whisper load failed; /start will be rejected until fixed")

        threading.Thread(target=_bg_load, daemon=True).start()

        # VAD is lazy-loaded on first capture
        self.state = STTState.READY
        await self.log("info", "STT service ready", "service_ready")
        return True

    async def shutdown(self):
        await self.log("info", "STT service shutting down")
        # Stop any active dialog
        if self.active_dialog_id:
            self.stop_flag.set()
            try:
                if self.dialog_thread and self.dialog_thread.is_alive():
                    self.dialog_thread.join(timeout=5.0)
            except Exception:
                pass
        self._stop_audio()


# =========================
# FastAPI app
# =========================
stt_service = STTService()

@asynccontextmanager
async def service_lifespan():
    success = await stt_service.startup()
    if not success:
        raise RuntimeError("STT service startup failed")
    yield
    await stt_service.shutdown()

app = create_service_app(
    title="STT Service",
    description="Silero VAD + faster-whisper dialog manager",
    version="1.3",
    lifespan=lambda app: lifespan_with_httpx(service_lifespan)
)


@app.post("/start")
async def start_dialog(request: StartRequest):
    if stt_service.state != STTState.READY:
        raise HTTPException(status_code=400, detail=f"Service not ready (state={stt_service.state})")
    if stt_service.model is None:
        raise HTTPException(status_code=503, detail="ASR model is still loading")
    if await stt_service.start_dialog(request.dialog_id):
        return {"status": "ok", "dialog_id": request.dialog_id}
    raise HTTPException(status_code=500, detail="Failed to start dialog")


@app.post("/stop")
async def stop_dialog(request: StopRequest):
    reason = request.reason or "manual stop"
    if await stt_service.stop_dialog(request.dialog_id, reason):
        return {"status": "ok", "dialog_id": request.dialog_id}
    raise HTTPException(status_code=404, detail="Dialog not found or already stopped")


@app.get("/health")
async def health_check():
    """
    READY and ACTIVE are both healthy. Include state for observability.
    """
    status = "ok" if stt_service.state in (STTState.READY, STTState.ACTIVE) else "error"
    return HealthResponse(status=status, state=stt_service.state)


@app.get("/state")
async def get_state():
    return StateResponse(state=str(stt_service.state))


@app.get("/devices")
async def list_devices():
    """
    List input-capable devices (index, name, sample rate, channels).
    """
    try:
        devs = sd.query_devices()
        items = []
        for idx, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                items.append({
                    "index": idx,
                    "name": d.get("name"),
                    "default_samplerate": d.get("default_samplerate"),
                    "max_input_channels": d.get("max_input_channels")
                })
        return {"status": "ok", "devices": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/device")
async def get_chosen_device():
    """
    Returns what device STT will actually use if started now (after config/env).
    """
    idx = stt_service._resolve_input_device()
    try:
        info = sd.query_devices(idx) if idx >= 0 else {}
    except Exception:
        info = {}
    return {"index": idx, "info": info}


def main():
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=app_config.stt.port,  # Get port from config
        log_level="error",
        access_log=False
    )


if __name__ == "__main__":
    main()
