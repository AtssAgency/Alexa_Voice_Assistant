"""Configuration management for AI Voice Assistant"""

import configparser
import os
import logging
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


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


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class VRAMValidationError(ConfigValidationError):
    """Raised when VRAM requirements are not met"""
    pass


class ConfigManager:
    """Configuration manager for the AI Voice Assistant"""
    
    def __init__(self, config_path: str = "config/config.txt"):
        self.config_path = config_path
        self.logger = logging.getLogger("config")
        self._config = None
        self._required_sections = ['system', 'main', 'loader', 'kwd', 'stt', 'llm', 'tts', 'dialog', 'logging']
        
    def load(self) -> Dict[str, Any]:
        """Load and validate configuration from file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        parser = configparser.ConfigParser()
        try:
            parser.read(self.config_path)
        except configparser.Error as e:
            raise ConfigValidationError(f"Failed to parse INI file: {e}")
        
        # Validate required sections exist
        self._validate_required_sections(parser)
        
        # Parse each section
        config = {}
        
        try:
            # System configuration
            config['system'] = self._parse_system_config(parser)
            
            # Main configuration
            config['main'] = self._parse_main_config(parser)
            
            # Loader configuration
            config['loader'] = self._parse_loader_config(parser)
            
            # Service configurations
            config['kwd'] = self._parse_kwd_config(parser)
            config['stt'] = self._parse_stt_config(parser)
            config['llm'] = self._parse_llm_config(parser)
            config['tts'] = self._parse_tts_config(parser)
            
            # Dialog configuration
            config['dialog'] = self._parse_dialog_config(parser)
            
            # Logging configuration
            config['logging'] = self._parse_logging_config(parser)
            
            self._config = config
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to parse configuration: {e}")
            raise ConfigValidationError(f"Configuration parsing failed: {e}")
    
    def _validate_required_sections(self, parser: configparser.ConfigParser) -> None:
        """Validate that all required sections are present"""
        missing_sections = []
        for section in self._required_sections:
            if not parser.has_section(section):
                missing_sections.append(section)
        
        if missing_sections:
            raise ConfigValidationError(f"Missing required sections: {missing_sections}")
    
    def _parse_system_config(self, parser: configparser.ConfigParser) -> SystemConfig:
        """Parse system configuration with validation"""
        try:
            min_vram_mb = parser.getint('system', 'min_vram_mb', fallback=8000)
            if min_vram_mb < 1000:
                raise ConfigValidationError("min_vram_mb must be at least 1000MB")
            return SystemConfig(min_vram_mb=min_vram_mb)
        except ValueError as e:
            raise ConfigValidationError(f"Invalid system configuration: {e}")
    
    def _parse_main_config(self, parser: configparser.ConfigParser) -> MainConfig:
        """Parse main configuration with validation"""
        try:
            log_level = parser.get('main', 'log_level', fallback='INFO')
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if log_level not in valid_levels:
                raise ConfigValidationError(f"log_level must be one of: {valid_levels}")
            
            graceful_shutdown_ms = parser.getint('main', 'graceful_shutdown_ms', fallback=5000)
            if graceful_shutdown_ms < 1000:
                raise ConfigValidationError("graceful_shutdown_ms must be at least 1000ms")
            
            return MainConfig(
                log_level=log_level,
                graceful_shutdown_ms=graceful_shutdown_ms
            )
        except ValueError as e:
            raise ConfigValidationError(f"Invalid main configuration: {e}")
    
    def _parse_loader_config(self, parser: configparser.ConfigParser) -> LoaderConfig:
        """Parse loader configuration with validation"""
        try:
            port = parser.getint('loader', 'port', fallback=5002)
            self._validate_port(port, 'loader')
            
            health_interval_ms = parser.getint('loader', 'health_interval_ms', fallback=2000)
            if health_interval_ms < 500:
                raise ConfigValidationError("health_interval_ms must be at least 500ms")
            
            restart_backoff = parser.get('loader', 'restart_backoff_ms', fallback='1000,3000,5000')
            try:
                backoff_list = [int(x.strip()) for x in restart_backoff.split(',')]
                if not backoff_list or any(x < 100 for x in backoff_list):
                    raise ConfigValidationError("restart_backoff_ms values must be at least 100ms")
            except ValueError:
                raise ConfigValidationError("restart_backoff_ms must be comma-separated integers")
            
            max_restarts = parser.getint('loader', 'max_restarts_per_minute', fallback=3)
            if max_restarts < 1:
                raise ConfigValidationError("max_restarts_per_minute must be at least 1")
            
            return LoaderConfig(
                port=port,
                health_interval_ms=health_interval_ms,
                restart_backoff_ms=backoff_list,
                max_restarts_per_minute=max_restarts
            )
        except ValueError as e:
            raise ConfigValidationError(f"Invalid loader configuration: {e}")
    
    def _parse_dialog_config(self, parser: configparser.ConfigParser) -> DialogConfig:
        """Parse dialog configuration with validation"""
        try:
            followup_window_ms = parser.getint('dialog', 'followup_window_ms', fallback=4000)
            if followup_window_ms < 1000:
                raise ConfigValidationError("followup_window_ms must be at least 1000ms")
            
            warmup_greeting = parser.get('dialog', 'warmup_greeting', fallback='Hi, Master!')
            if not warmup_greeting.strip():
                raise ConfigValidationError("warmup_greeting cannot be empty")
            
            return DialogConfig(
                followup_window_ms=followup_window_ms,
                warmup_greeting=warmup_greeting
            )
        except ValueError as e:
            raise ConfigValidationError(f"Invalid dialog configuration: {e}")
    
    def _parse_logging_config(self, parser: configparser.ConfigParser) -> LoggingConfig:
        """Parse logging configuration with validation"""
        try:
            port = parser.getint('logging', 'port', fallback=5001)
            self._validate_port(port, 'logging')
            
            rotate_size_mb = parser.getint('logging', 'rotate_size_mb', fallback=10)
            if rotate_size_mb < 1:
                raise ConfigValidationError("rotate_size_mb must be at least 1MB")
            
            rotate_keep_files = parser.getint('logging', 'rotate_keep_files', fallback=5)
            if rotate_keep_files < 1:
                raise ConfigValidationError("rotate_keep_files must be at least 1")
            
            logs_dir = parser.get('logging', 'logs_dir', fallback='logs')
            if not logs_dir.strip():
                raise ConfigValidationError("logs_dir cannot be empty")
            
            return LoggingConfig(
                port=port,
                app_reset_on_start=parser.getboolean('logging', 'app_reset_on_start', fallback=True),
                rotate_size_mb=rotate_size_mb,
                rotate_keep_files=rotate_keep_files,
                logs_dir=logs_dir
            )
        except ValueError as e:
            raise ConfigValidationError(f"Invalid logging configuration: {e}")
    
    def _validate_port(self, port: int, service_name: str) -> None:
        """Validate port number is in valid range"""
        if not (1024 <= port <= 65535):
            raise ConfigValidationError(f"{service_name} port must be between 1024 and 65535")
    
    def _parse_kwd_config(self, parser: configparser.ConfigParser) -> KwdConfig:
        """Parse KWD service configuration with validation"""
        try:
            port = parser.getint('kwd', 'port', fallback=5003)
            self._validate_port(port, 'kwd')
            
            model_path = parser.get('kwd', 'model_path', fallback='models/alexa_v0.1.onnx')
            if not model_path.strip():
                raise ConfigValidationError("kwd model_path cannot be empty")
            
            confidence_threshold = parser.getfloat('kwd', 'confidence_threshold', fallback=0.6)
            if not (0.0 <= confidence_threshold <= 1.0):
                raise ConfigValidationError("kwd confidence_threshold must be between 0.0 and 1.0")
            
            cooldown_ms = parser.getint('kwd', 'cooldown_ms', fallback=1000)
            if cooldown_ms < 100:
                raise ConfigValidationError("kwd cooldown_ms must be at least 100ms")
            
            yes_phrases_str = parser.get('kwd', 'yes_phrases', fallback='Yes?;Yes, Master?;Sup?;Yo')
            yes_phrases = [phrase.strip() for phrase in yes_phrases_str.split(';') if phrase.strip()]
            if not yes_phrases:
                raise ConfigValidationError("kwd yes_phrases cannot be empty")
            
            return KwdConfig(
                port=port,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                cooldown_ms=cooldown_ms,
                yes_phrases=yes_phrases
            )
        except ValueError as e:
            raise ConfigValidationError(f"Invalid kwd configuration: {e}")
    
    def _parse_stt_config(self, parser: configparser.ConfigParser) -> SttConfig:
        """Parse STT service configuration with validation"""
        try:
            port = parser.getint('stt', 'port', fallback=5004)
            self._validate_port(port, 'stt')
            
            device = parser.get('stt', 'device', fallback='cuda')
            valid_devices = ['cuda', 'cpu']
            if device not in valid_devices:
                raise ConfigValidationError(f"stt device must be one of: {valid_devices}")
            
            model = parser.get('stt', 'model', fallback='small.en')
            if not model.strip():
                raise ConfigValidationError("stt model cannot be empty")
            
            vad_end_silence_ms = parser.getint('stt', 'vad_end_silence_ms', fallback=2000)
            if vad_end_silence_ms < 500:
                raise ConfigValidationError("stt vad_end_silence_ms must be at least 500ms")
            
            max_recording_sec = parser.getint('stt', 'max_recording_sec', fallback=60)
            if max_recording_sec < 5:
                raise ConfigValidationError("stt max_recording_sec must be at least 5 seconds")
            
            aec_backend = parser.get('stt', 'aec_backend', fallback='webrtc_aec3')
            valid_backends = ['webrtc_aec3', 'none']
            if aec_backend not in valid_backends:
                raise ConfigValidationError(f"stt aec_backend must be one of: {valid_backends}")
            
            return SttConfig(
                port=port,
                device=device,
                model=model,
                vad_end_silence_ms=vad_end_silence_ms,
                max_recording_sec=max_recording_sec,
                aec_enabled=parser.getboolean('stt', 'aec_enabled', fallback=True),
                aec_backend=aec_backend
            )
        except ValueError as e:
            raise ConfigValidationError(f"Invalid stt configuration: {e}")
    
    def _parse_llm_config(self, parser: configparser.ConfigParser) -> LlmConfig:
        """Parse LLM service configuration with validation"""
        try:
            port = parser.getint('llm', 'port', fallback=5005)
            self._validate_port(port, 'llm')
            
            endpoint = parser.get('llm', 'endpoint', fallback='localhost:11434')
            if not endpoint.strip():
                raise ConfigValidationError("llm endpoint cannot be empty")
            
            # Validate endpoint format (basic check)
            if ':' not in endpoint:
                raise ConfigValidationError("llm endpoint must include port (host:port)")
            
            model = parser.get('llm', 'model', fallback='llama3.1:8b-instruct-q4')
            if not model.strip():
                raise ConfigValidationError("llm model cannot be empty")
            
            modelfile_path = parser.get('llm', 'modelfile_path', fallback='config/Modelfile')
            if not modelfile_path.strip():
                raise ConfigValidationError("llm modelfile_path cannot be empty")
            
            request_timeout_ms = parser.getint('llm', 'request_timeout_ms', fallback=30000)
            if request_timeout_ms < 1000:
                raise ConfigValidationError("llm request_timeout_ms must be at least 1000ms")
            
            stream_chunk_chars = parser.getint('llm', 'stream_chunk_chars', fallback=80)
            if stream_chunk_chars < 1:
                raise ConfigValidationError("llm stream_chunk_chars must be at least 1")
            
            return LlmConfig(
                port=port,
                endpoint=endpoint,
                model=model,
                modelfile_path=modelfile_path,
                request_timeout_ms=request_timeout_ms,
                stream_chunk_chars=stream_chunk_chars
            )
        except ValueError as e:
            raise ConfigValidationError(f"Invalid llm configuration: {e}")
    
    def _parse_tts_config(self, parser: configparser.ConfigParser) -> TtsConfig:
        """Parse TTS service configuration with validation"""
        try:
            port = parser.getint('tts', 'port', fallback=5006)
            self._validate_port(port, 'tts')
            
            voice = parser.get('tts', 'voice', fallback='af_heart')
            if not voice.strip():
                raise ConfigValidationError("tts voice cannot be empty")
            
            device = parser.get('tts', 'device', fallback='cuda')
            valid_devices = ['cuda', 'cpu']
            if device not in valid_devices:
                raise ConfigValidationError(f"tts device must be one of: {valid_devices}")
            
            queue_max_chunks = parser.getint('tts', 'queue_max_chunks', fallback=200)
            if queue_max_chunks < 10:
                raise ConfigValidationError("tts queue_max_chunks must be at least 10")
            
            chunk_max_ms = parser.getint('tts', 'chunk_max_ms', fallback=400)
            if chunk_max_ms < 50:
                raise ConfigValidationError("tts chunk_max_ms must be at least 50ms")
            
            return TtsConfig(
                port=port,
                voice=voice,
                device=device,
                queue_max_chunks=queue_max_chunks,
                chunk_max_ms=chunk_max_ms
            )
        except ValueError as e:
            raise ConfigValidationError(f"Invalid tts configuration: {e}")
    
    def validate(self) -> bool:
        """Validate the loaded configuration and system requirements"""
        if not self._config:
            raise ValueError("Configuration not loaded")
        
        # Validate VRAM requirements
        self._validate_vram()
        
        # Validate port conflicts
        self._validate_port_conflicts()
        
        # Validate and create required directories
        self._validate_directories()
        
        # Validate file paths
        self._validate_file_paths()
        
        self.logger.info("Configuration validation completed successfully")
        return True
    
    def _validate_vram(self) -> None:
        """Validate VRAM requirements are met"""
        try:
            # Get GPU memory info using psutil (if available) or fallback methods
            min_vram_mb = self._config['system'].min_vram_mb
            
            # Try to get GPU memory info
            available_vram_mb = self._get_available_vram()
            
            if available_vram_mb is not None and available_vram_mb < min_vram_mb:
                raise VRAMValidationError(
                    f"Insufficient VRAM: {available_vram_mb}MB available, "
                    f"{min_vram_mb}MB required"
                )
            elif available_vram_mb is not None:
                self.logger.info(f"VRAM check passed: {available_vram_mb}MB available")
            else:
                self.logger.warning("Could not verify VRAM requirements - proceeding with caution")
                
        except Exception as e:
            if isinstance(e, VRAMValidationError):
                raise
            self.logger.warning(f"VRAM validation failed: {e}")
    
    def _get_available_vram(self) -> Optional[int]:
        """Get available VRAM in MB"""
        try:
            # Try to import torch to check CUDA memory
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                return int(total_memory / (1024 * 1024))  # Convert to MB
        except (ImportError, Exception):
            pass
        
        try:
            # Try using nvidia-ml-py if available
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.total / (1024 * 1024))  # Convert to MB
        except (ImportError, Exception):
            pass
        
        # Could not determine VRAM
        return None
    
    def _validate_port_conflicts(self) -> None:
        """Validate that no two services use the same port"""
        ports = []
        service_ports = {
            'logging': self._config['logging'].port,
            'loader': self._config['loader'].port,
            'kwd': self._config['kwd'].port,
            'stt': self._config['stt'].port,
            'llm': self._config['llm'].port,
            'tts': self._config['tts'].port,
        }
        
        for service, port in service_ports.items():
            if port in ports:
                raise ConfigValidationError(f"Port conflict: {port} is used by multiple services")
            ports.append(port)
    
    def _validate_directories(self) -> None:
        """Validate and create required directories"""
        logs_dir = self._config['logging'].logs_dir
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(logs_dir):
            try:
                os.makedirs(logs_dir, exist_ok=True)
                self.logger.info(f"Created logs directory: {logs_dir}")
            except OSError as e:
                raise ConfigValidationError(f"Failed to create logs directory {logs_dir}: {e}")
        
        # Validate logs directory is writable
        if not os.access(logs_dir, os.W_OK):
            raise ConfigValidationError(f"Logs directory is not writable: {logs_dir}")
        
        # Create models directory if referenced model doesn't exist
        model_path = self._config['kwd'].model_path
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
                self.logger.info(f"Created models directory: {model_dir}")
            except OSError as e:
                raise ConfigValidationError(f"Failed to create models directory {model_dir}: {e}")
    
    def _validate_file_paths(self) -> None:
        """Validate critical file paths"""
        # Check KWD model file (warning only, as it may be downloaded later)
        model_path = self._config['kwd'].model_path
        if not os.path.exists(model_path):
            self.logger.warning(f"KWD model file not found: {model_path}")
        
        # Check LLM Modelfile (warning only, as it may be created later)
        modelfile_path = self._config['llm'].modelfile_path
        if not os.path.exists(modelfile_path):
            self.logger.warning(f"LLM Modelfile not found: {modelfile_path}")
        
        # Validate config file directory is readable
        config_dir = os.path.dirname(self.config_path)
        if config_dir and not os.access(config_dir, os.R_OK):
            raise ConfigValidationError(f"Config directory is not readable: {config_dir}")
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """Get the loaded configuration"""
        return self._config