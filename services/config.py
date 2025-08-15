"""Configuration management for AI Voice Assistant"""

import configparser
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SystemConfig:
    """System-level configuration"""
    min_vram_mb: int = 8000


@dataclass
class MainConfig:
    """Main orchestrator configuration"""
    log_level: str = "INFO"
    graceful_shutdown_ms: int = 5000


@dataclass
class LoaderConfig:
    """Service loader configuration"""
    port: int = 5002
    health_interval_ms: int = 2000
    restart_backoff_ms: list = None
    max_restarts_per_minute: int = 3
    
    def __post_init__(self):
        if self.restart_backoff_ms is None:
            self.restart_backoff_ms = [1000, 3000, 5000]


@dataclass
class ServiceConfig:
    """Base service configuration"""
    port: int
    
    
@dataclass
class KwdConfig(ServiceConfig):
    """Wake Word Detection service configuration"""
    port: int = 5003
    model_path: str = "models/alexa_v0.1.onnx"
    confidence_threshold: float = 0.6
    cooldown_ms: int = 1000
    yes_phrases: list = None
    
    def __post_init__(self):
        if self.yes_phrases is None:
            self.yes_phrases = ["Yes?", "Yes, Master?", "Sup?", "Yo"]


@dataclass
class SttConfig(ServiceConfig):
    """Speech-to-Text service configuration"""
    port: int = 5004
    device: str = "cuda"
    model: str = "small.en"
    vad_end_silence_ms: int = 2000
    max_recording_sec: int = 60
    aec_enabled: bool = True
    aec_backend: str = "webrtc_aec3"


@dataclass
class LlmConfig(ServiceConfig):
    """Large Language Model service configuration"""
    port: int = 5005
    endpoint: str = "localhost:11434"
    model: str = "llama3.1:8b-instruct-q4"
    modelfile_path: str = "config/Modelfile"
    request_timeout_ms: int = 30000
    stream_chunk_chars: int = 80


@dataclass
class TtsConfig(ServiceConfig):
    """Text-to-Speech service configuration"""
    port: int = 5006
    voice: str = "af_heart"
    device: str = "cuda"
    queue_max_chunks: int = 200
    chunk_max_ms: int = 400


@dataclass
class DialogConfig:
    """Dialog management configuration"""
    followup_window_ms: int = 4000
    warmup_greeting: str = "Hi, Master!"


@dataclass
class LoggingConfig:
    """Logging service configuration"""
    port: int = 5001
    app_reset_on_start: bool = True
    rotate_size_mb: int = 10
    rotate_keep_files: int = 5
    logs_dir: str = "logs"


class ConfigManager:
    """Configuration manager for the AI Voice Assistant"""
    
    def __init__(self, config_path: str = "config/config.txt"):
        self.config_path = config_path
        self.logger = logging.getLogger("config")
        self._config = None
        
    def load(self) -> Dict[str, Any]:
        """Load and validate configuration from file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        
        # Parse each section
        config = {}
        
        try:
            # System configuration
            if parser.has_section('system'):
                config['system'] = SystemConfig(
                    min_vram_mb=parser.getint('system', 'min_vram_mb', fallback=8000)
                )
            else:
                config['system'] = SystemConfig()
            
            # Main configuration
            if parser.has_section('main'):
                config['main'] = MainConfig(
                    log_level=parser.get('main', 'log_level', fallback='INFO'),
                    graceful_shutdown_ms=parser.getint('main', 'graceful_shutdown_ms', fallback=5000)
                )
            else:
                config['main'] = MainConfig()
            
            # Loader configuration
            if parser.has_section('loader'):
                restart_backoff = parser.get('loader', 'restart_backoff_ms', fallback='1000,3000,5000')
                backoff_list = [int(x.strip()) for x in restart_backoff.split(',')]
                
                config['loader'] = LoaderConfig(
                    port=parser.getint('loader', 'port', fallback=5002),
                    health_interval_ms=parser.getint('loader', 'health_interval_ms', fallback=2000),
                    restart_backoff_ms=backoff_list,
                    max_restarts_per_minute=parser.getint('loader', 'max_restarts_per_minute', fallback=3)
                )
            else:
                config['loader'] = LoaderConfig()
            
            # Service configurations
            config['kwd'] = self._parse_kwd_config(parser)
            config['stt'] = self._parse_stt_config(parser)
            config['llm'] = self._parse_llm_config(parser)
            config['tts'] = self._parse_tts_config(parser)
            
            # Dialog configuration
            if parser.has_section('dialog'):
                config['dialog'] = DialogConfig(
                    followup_window_ms=parser.getint('dialog', 'followup_window_ms', fallback=4000),
                    warmup_greeting=parser.get('dialog', 'warmup_greeting', fallback='Hi, Master!')
                )
            else:
                config['dialog'] = DialogConfig()
            
            # Logging configuration
            if parser.has_section('logging'):
                config['logging'] = LoggingConfig(
                    port=parser.getint('logging', 'port', fallback=5001),
                    app_reset_on_start=parser.getboolean('logging', 'app_reset_on_start', fallback=True),
                    rotate_size_mb=parser.getint('logging', 'rotate_size_mb', fallback=10),
                    rotate_keep_files=parser.getint('logging', 'rotate_keep_files', fallback=5),
                    logs_dir=parser.get('logging', 'logs_dir', fallback='logs')
                )
            else:
                config['logging'] = LoggingConfig()
            
            self._config = config
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to parse configuration: {e}")
            raise
    
    def _parse_kwd_config(self, parser: configparser.ConfigParser) -> KwdConfig:
        """Parse KWD service configuration"""
        if not parser.has_section('kwd'):
            return KwdConfig()
        
        yes_phrases_str = parser.get('kwd', 'yes_phrases', fallback='Yes?;Yes, Master?;Sup?;Yo')
        yes_phrases = [phrase.strip() for phrase in yes_phrases_str.split(';')]
        
        return KwdConfig(
            port=parser.getint('kwd', 'port', fallback=5003),
            model_path=parser.get('kwd', 'model_path', fallback='models/alexa_v0.1.onnx'),
            confidence_threshold=parser.getfloat('kwd', 'confidence_threshold', fallback=0.6),
            cooldown_ms=parser.getint('kwd', 'cooldown_ms', fallback=1000),
            yes_phrases=yes_phrases
        )
    
    def _parse_stt_config(self, parser: configparser.ConfigParser) -> SttConfig:
        """Parse STT service configuration"""
        if not parser.has_section('stt'):
            return SttConfig()
        
        return SttConfig(
            port=parser.getint('stt', 'port', fallback=5004),
            device=parser.get('stt', 'device', fallback='cuda'),
            model=parser.get('stt', 'model', fallback='small.en'),
            vad_end_silence_ms=parser.getint('stt', 'vad_end_silence_ms', fallback=2000),
            max_recording_sec=parser.getint('stt', 'max_recording_sec', fallback=60),
            aec_enabled=parser.getboolean('stt', 'aec_enabled', fallback=True),
            aec_backend=parser.get('stt', 'aec_backend', fallback='webrtc_aec3')
        )
    
    def _parse_llm_config(self, parser: configparser.ConfigParser) -> LlmConfig:
        """Parse LLM service configuration"""
        if not parser.has_section('llm'):
            return LlmConfig()
        
        return LlmConfig(
            port=parser.getint('llm', 'port', fallback=5005),
            endpoint=parser.get('llm', 'endpoint', fallback='localhost:11434'),
            model=parser.get('llm', 'model', fallback='llama3.1:8b-instruct-q4'),
            modelfile_path=parser.get('llm', 'modelfile_path', fallback='config/Modelfile'),
            request_timeout_ms=parser.getint('llm', 'request_timeout_ms', fallback=30000),
            stream_chunk_chars=parser.getint('llm', 'stream_chunk_chars', fallback=80)
        )
    
    def _parse_tts_config(self, parser: configparser.ConfigParser) -> TtsConfig:
        """Parse TTS service configuration"""
        if not parser.has_section('tts'):
            return TtsConfig()
        
        return TtsConfig(
            port=parser.getint('tts', 'port', fallback=5006),
            voice=parser.get('tts', 'voice', fallback='af_heart'),
            device=parser.get('tts', 'device', fallback='cuda'),
            queue_max_chunks=parser.getint('tts', 'queue_max_chunks', fallback=200),
            chunk_max_ms=parser.getint('tts', 'chunk_max_ms', fallback=400)
        )
    
    def validate(self) -> bool:
        """Validate the loaded configuration"""
        if not self._config:
            raise ValueError("Configuration not loaded")
        
        # Validate required directories exist
        logs_dir = self._config['logging'].logs_dir
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
            self.logger.info(f"Created logs directory: {logs_dir}")
        
        # Validate model files exist (will be checked by individual services)
        model_path = self._config['kwd'].model_path
        if not os.path.exists(model_path):
            self.logger.warning(f"KWD model file not found: {model_path}")
        
        # Validate Modelfile exists
        modelfile_path = self._config['llm'].modelfile_path
        if not os.path.exists(modelfile_path):
            self.logger.warning(f"LLM Modelfile not found: {modelfile_path}")
        
        return True
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """Get the loaded configuration"""
        return self._config