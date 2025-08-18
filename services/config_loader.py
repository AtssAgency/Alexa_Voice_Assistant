# services/config_loader.py

import configparser
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

# --- Pydantic Models for each service's config ---

class LoggerConfig(BaseModel):
    port: int = 5000
    log_level: str = "info"
    dialog_prefix: str = "dialog_"
    append_timestamps: bool = True

class RMSDeps(BaseModel):
    logger_url: str

class RMSConfig(BaseModel):
    bind_host: str = "127.0.0.1"
    port: int = 5006
    sample_rate: int = 16000
    frame_ms: int = 50
    window_sec: int = 30
    ema_alpha: float = 0.2
    noise_floor_mode: str = "p05"
    device: str = "default"
    shared_mode: bool = True
    stats_interval_sec: int = 5
    deps: RMSDeps

class KWDAdaptive(BaseModel):
    quiet_dbfs: float = -60.0
    noisy_dbfs: float = -35.0
    quiet_factor: float = 0.80
    noisy_factor: float = 1.25

class KWDeps(BaseModel):
    logger_url: str
    tts_url: str
    stt_url: str
    rms_url: str

class KWDConfig(BaseModel):
    bind_host: str = "127.0.0.1"
    port: int = 5002
    model_path: str
    base_rms_threshold: float
    cooldown_ms: int
    yes_phrases: str
    greeting_text: str
    adaptive: KWDAdaptive
    deps: KWDeps

class STTAudio(BaseModel):
    device_index: int = -1
    sample_rate: int = 16000
    frame_ms: int = 32
    dtype: str = "int16"
    dc_block: bool = True
    min_frame_rms: float = 0.003

class STTAdaptive(BaseModel):
    quiet_dbfs: float = -60.0
    noisy_dbfs: float = -35.0

class STTVAD(BaseModel):
    silence_margin_db: float = 8.0
    energy_delta_db: float = 6.0
    engine: str = "silero"
    start_threshold: float = 0.6
    end_threshold: float = 0.45
    min_start_frames: int = 5
    max_silence_ms: int = 800
    pre_roll_ms: int = 400
    speech_pad_ms: int = 300

class STTDeps(BaseModel):
    logger_url: str
    llm_url: str
    tts_ws_speak: str
    tts_ws_events: str
    kwd_url: str
    rms_url: str
    
class STTConfig(BaseModel):
    bind_host: str = "127.0.0.1"
    port: int = 5003
    model_size: str = "small.en"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 1
    vad_finalize_sec: float = 2.0
    chunk_length_s: float = 15.0
    follow_up_timer_sec: float = 4.0
    word_timestamps: bool = False
    audio: STTAudio
    adaptive: STTAdaptive
    vad: STTVAD
    deps: STTDeps

class LLMDeps(BaseModel):
    logger_url: str

class LLMConfig(BaseModel):
    bind_host: str = "127.0.0.1"
    port: int = 5004
    ollama_base_url: str = "http://127.0.0.1:11434"
    request_timeout_s: int = 120
    warmup_enabled: bool = True
    warmup_text: str = "ok"
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.05
    max_tokens: int = 512
    deps: LLMDeps

class TTSDeps(BaseModel):
    logger_url: str
    stt_ws_events_allow: bool = True

class TTSConfig(BaseModel):
    bind_host: str = "127.0.0.1"
    port: int = 5005
    voice: str = "af_heart"
    sample_rate: int = 24000
    device: str = "cuda"
    dtype: str = "float16"
    max_queue_text: int = 64
    max_queue_audio: int = 8
    chunk_chars: int = 120
    silence_pad_ms: int = 60
    allow_llm_pull: bool = False
    model_path: str = "models/kokoro-v1.0.fp16.onnx"
    voices_path: str = "models/voices-v1.0.bin"
    quant_preference: str = "fp16"
    deps: TTSDeps

class LoaderPaths(BaseModel):
    python: str = "python3"

class LoaderConfig(BaseModel):
    startup_order: str = "logger,rms,kwd,stt,ollama,llm,tts"
    health_timeout_s: int = 30
    health_interval_ms: int = 200
    restart_backoff_ms: str = "250,500,1000,2000"
    crash_loop_threshold: int = 3
    port_hygiene: bool = True
    vram_probe_cmd: str = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
    warmup_llm: bool = True
    greeting_on_ready: bool = True
    paths: LoaderPaths

class AppConfig(BaseModel):
    """Holds all service configurations."""
    logger: LoggerConfig
    rms: RMSConfig
    kwd: KWDConfig
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig
    loader: LoaderConfig

# --- Loader Function ---

def load_app_config() -> AppConfig:
    """Parses config.ini and populates Pydantic models."""
    cfg_path = Path("config/config.ini")
    if not cfg_path.exists():
        raise FileNotFoundError("config/config.ini not found")

    parser = configparser.ConfigParser()
    parser.read(cfg_path)

    # Helper to convert section to dict
    def section_to_dict(section_name: str):
        if parser.has_section(section_name):
            return {key: val for key, val in parser.items(section_name)}
        return {}

    # Populate the models
    config_data = {
        "logger": section_to_dict("logger"),
        "rms": {
            **section_to_dict("rms"),
            "deps": section_to_dict("rms.deps")
        },
        "kwd": {
            **section_to_dict("kwd"),
            "adaptive": section_to_dict("kwd.adaptive"),
            "deps": section_to_dict("kwd.deps")
        },
        "stt": {
            **section_to_dict("stt"),
            "audio": section_to_dict("stt.audio"),
            "adaptive": section_to_dict("stt.adaptive"),
            "vad": section_to_dict("stt.vad"),
            "deps": section_to_dict("stt.deps")
        },
        "llm": {
            **section_to_dict("llm"),
            "deps": section_to_dict("llm.deps")
        },
        "tts": {
            **section_to_dict("tts"),
            "deps": section_to_dict("tts.deps")
        },
        "loader": {
            **section_to_dict("loader"),
            "paths": section_to_dict("paths")
        }
    }

    return AppConfig.model_validate(config_data)

# --- Global Config Instance ---
# Load once on import
app_config = load_app_config()
