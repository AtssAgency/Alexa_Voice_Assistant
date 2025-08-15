"""Unit tests for configuration management system"""

import pytest
import tempfile
import os
import configparser
from unittest.mock import patch, MagicMock

from services.config import (
    ConfigManager, ConfigValidationError, VRAMValidationError,
    SystemConfig, MainConfig, LoaderConfig, KwdConfig, SttConfig,
    LlmConfig, TtsConfig, DialogConfig, LoggingConfig
)


class TestConfigManager:
    """Test cases for ConfigManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.txt")
        
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_valid_config(self) -> str:
        """Create a valid configuration file for testing"""
        config_content = """
[system]
min_vram_mb=8000

[main]
log_level=INFO
graceful_shutdown_ms=5000

[loader]
port=5002
health_interval_ms=2000
restart_backoff_ms=1000,3000,5000
max_restarts_per_minute=3

[kwd]
port=5003
model_path=models/alexa_v0.1.onnx
confidence_threshold=0.6
cooldown_ms=1000
yes_phrases=Yes?;Yes, Master?;Sup?;Yo

[stt]
port=5004
device=cuda
model=small.en
vad_end_silence_ms=2000
max_recording_sec=60
aec_enabled=true
aec_backend=webrtc_aec3

[llm]
port=5005
endpoint=localhost:11434
model=llama3.1:8b-instruct-q4
modelfile_path=config/Modelfile
request_timeout_ms=30000
stream_chunk_chars=80

[tts]
port=5006
voice=af_heart
device=cuda
queue_max_chunks=200
chunk_max_ms=400

[dialog]
followup_window_ms=4000
warmup_greeting=Hi, Master!

[logging]
port=5001
app_reset_on_start=true
rotate_size_mb=10
rotate_keep_files=5
logs_dir=logs
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content.strip())
        return self.config_path
    
    def test_load_valid_config(self):
        """Test loading a valid configuration"""
        self.create_valid_config()
        manager = ConfigManager(self.config_path)
        config = manager.load()
        
        # Verify all sections are loaded
        assert 'system' in config
        assert 'main' in config
        assert 'loader' in config
        assert 'kwd' in config
        assert 'stt' in config
        assert 'llm' in config
        assert 'tts' in config
        assert 'dialog' in config
        assert 'logging' in config
        
        # Verify specific values
        assert config['system'].min_vram_mb == 8000
        assert config['main'].log_level == 'INFO'
        assert config['kwd'].confidence_threshold == 0.6
        assert config['stt'].device == 'cuda'
        assert config['llm'].endpoint == 'localhost:11434'
        assert config['tts'].voice == 'af_heart'
    
    def test_load_missing_file(self):
        """Test loading non-existent configuration file"""
        manager = ConfigManager("nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            manager.load()
    
    def test_load_invalid_ini_format(self):
        """Test loading malformed INI file"""
        with open(self.config_path, 'w') as f:
            f.write("invalid ini content [[[")
        
        manager = ConfigManager(self.config_path)
        with pytest.raises(ConfigValidationError):
            manager.load()
    
    def test_missing_required_sections(self):
        """Test configuration with missing required sections"""
        with open(self.config_path, 'w') as f:
            f.write("[system]\nmin_vram_mb=8000\n")
        
        manager = ConfigManager(self.config_path)
        with pytest.raises(ConfigValidationError, match="Missing required sections"):
            manager.load()
    
    def test_invalid_system_config(self):
        """Test invalid system configuration values"""
        config_content = """
[system]
min_vram_mb=500

[main]
log_level=INFO
graceful_shutdown_ms=5000

[loader]
port=5002
health_interval_ms=2000
restart_backoff_ms=1000,3000,5000
max_restarts_per_minute=3

[kwd]
port=5003
model_path=models/alexa_v0.1.onnx
confidence_threshold=0.6
cooldown_ms=1000
yes_phrases=Yes?

[stt]
port=5004
device=cuda
model=small.en
vad_end_silence_ms=2000
max_recording_sec=60
aec_enabled=true
aec_backend=webrtc_aec3

[llm]
port=5005
endpoint=localhost:11434
model=llama3.1:8b-instruct-q4
modelfile_path=config/Modelfile
request_timeout_ms=30000
stream_chunk_chars=80

[tts]
port=5006
voice=af_heart
device=cuda
queue_max_chunks=200
chunk_max_ms=400

[dialog]
followup_window_ms=4000
warmup_greeting=Hi, Master!

[logging]
port=5001
app_reset_on_start=true
rotate_size_mb=10
rotate_keep_files=5
logs_dir=logs
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
        
        manager = ConfigManager(self.config_path)
        with pytest.raises(ConfigValidationError, match="min_vram_mb must be at least 1000MB"):
            manager.load()
    
    def test_invalid_main_config(self):
        """Test invalid main configuration values"""
        self.create_valid_config()
        
        # Test invalid log level
        with open(self.config_path, 'r') as f:
            content = f.read()
        content = content.replace("log_level=INFO", "log_level=INVALID")
        with open(self.config_path, 'w') as f:
            f.write(content)
        
        manager = ConfigManager(self.config_path)
        with pytest.raises(ConfigValidationError, match="log_level must be one of"):
            manager.load()
    
    def test_invalid_port_values(self):
        """Test invalid port configurations"""
        self.create_valid_config()
        
        # Test port out of range
        with open(self.config_path, 'r') as f:
            content = f.read()
        content = content.replace("port=5001", "port=100")
        with open(self.config_path, 'w') as f:
            f.write(content)
        
        manager = ConfigManager(self.config_path)
        with pytest.raises(ConfigValidationError, match="port must be between 1024 and 65535"):
            manager.load()
    
    def test_port_conflicts(self):
        """Test port conflict detection"""
        self.create_valid_config()
        
        # Create port conflict
        with open(self.config_path, 'r') as f:
            content = f.read()
        content = content.replace("port=5002", "port=5001")  # Conflict with logging port
        with open(self.config_path, 'w') as f:
            f.write(content)
        
        manager = ConfigManager(self.config_path)
        config = manager.load()
        
        # Mock VRAM validation to pass so we can test port conflicts
        with patch.object(manager, '_get_available_vram', return_value=12000):
            with pytest.raises(ConfigValidationError, match="Port conflict"):
                manager.validate()
    
    def test_invalid_kwd_config(self):
        """Test invalid KWD configuration values"""
        self.create_valid_config()
        
        # Test invalid confidence threshold
        with open(self.config_path, 'r') as f:
            content = f.read()
        content = content.replace("confidence_threshold=0.6", "confidence_threshold=1.5")
        with open(self.config_path, 'w') as f:
            f.write(content)
        
        manager = ConfigManager(self.config_path)
        with pytest.raises(ConfigValidationError, match="confidence_threshold must be between 0.0 and 1.0"):
            manager.load()
    
    def test_invalid_stt_config(self):
        """Test invalid STT configuration values"""
        self.create_valid_config()
        
        # Test invalid device
        with open(self.config_path, 'r') as f:
            content = f.read()
        content = content.replace("device=cuda", "device=invalid")
        with open(self.config_path, 'w') as f:
            f.write(content)
        
        manager = ConfigManager(self.config_path)
        with pytest.raises(ConfigValidationError, match="stt device must be one of"):
            manager.load()
    
    def test_invalid_llm_config(self):
        """Test invalid LLM configuration values"""
        self.create_valid_config()
        
        # Test invalid endpoint format
        with open(self.config_path, 'r') as f:
            content = f.read()
        content = content.replace("endpoint=localhost:11434", "endpoint=localhost")
        with open(self.config_path, 'w') as f:
            f.write(content)
        
        manager = ConfigManager(self.config_path)
        with pytest.raises(ConfigValidationError, match="endpoint must include port"):
            manager.load()
    
    def test_invalid_tts_config(self):
        """Test invalid TTS configuration values"""
        self.create_valid_config()
        
        # Test invalid queue size
        with open(self.config_path, 'r') as f:
            content = f.read()
        content = content.replace("queue_max_chunks=200", "queue_max_chunks=5")
        with open(self.config_path, 'w') as f:
            f.write(content)
        
        manager = ConfigManager(self.config_path)
        with pytest.raises(ConfigValidationError, match="queue_max_chunks must be at least 10"):
            manager.load()
    
    def test_vram_validation_sufficient(self):
        """Test VRAM validation with sufficient memory"""
        self.create_valid_config()
        manager = ConfigManager(self.config_path)
        config = manager.load()
        
        # Mock the _get_available_vram method to return sufficient VRAM
        with patch.object(manager, '_get_available_vram', return_value=12000):
            # Should not raise exception
            assert manager.validate() is True
    
    def test_vram_validation_insufficient(self):
        """Test VRAM validation with insufficient memory"""
        self.create_valid_config()
        manager = ConfigManager(self.config_path)
        config = manager.load()
        
        # Mock the _get_available_vram method to return insufficient VRAM
        with patch.object(manager, '_get_available_vram', return_value=4000):
            with pytest.raises(VRAMValidationError, match="Insufficient VRAM"):
                manager.validate()
    
    def test_vram_validation_no_gpu_libs(self):
        """Test VRAM validation when GPU libraries are not available"""
        self.create_valid_config()
        manager = ConfigManager(self.config_path)
        config = manager.load()
        
        # Mock the _get_available_vram method to return None (no GPU detection)
        with patch.object(manager, '_get_available_vram', return_value=None):
            # Should not raise exception, just log warning
            assert manager.validate() is True
    
    def test_directory_creation(self):
        """Test automatic directory creation"""
        self.create_valid_config()
        
        # Ensure logs directory doesn't exist
        logs_dir = os.path.join(self.temp_dir, "logs")
        assert not os.path.exists(logs_dir)
        
        manager = ConfigManager(self.config_path)
        config = manager.load()
        
        # Update config to use temp directory
        config['logging'].logs_dir = logs_dir
        manager._config = config
        
        # Mock VRAM validation to pass so we can test directory creation
        with patch.object(manager, '_get_available_vram', return_value=12000):
            manager.validate()
        
        # Directory should be created
        assert os.path.exists(logs_dir)
    
    def test_get_config_before_load(self):
        """Test getting config before loading"""
        manager = ConfigManager(self.config_path)
        assert manager.get_config() is None
    
    def test_get_config_after_load(self):
        """Test getting config after loading"""
        self.create_valid_config()
        manager = ConfigManager(self.config_path)
        config = manager.load()
        
        retrieved_config = manager.get_config()
        assert retrieved_config is config
        assert retrieved_config is not None


class TestConfigDataClasses:
    """Test configuration data classes"""
    
    def test_system_config_defaults(self):
        """Test SystemConfig default values"""
        config = SystemConfig()
        assert config.min_vram_mb == 8000
    
    def test_main_config_defaults(self):
        """Test MainConfig default values"""
        config = MainConfig()
        assert config.log_level == "INFO"
        assert config.graceful_shutdown_ms == 5000
    
    def test_loader_config_defaults(self):
        """Test LoaderConfig default values and post_init"""
        config = LoaderConfig()
        assert config.port == 5002
        assert config.health_interval_ms == 2000
        assert config.restart_backoff_ms == [1000, 3000, 5000]
        assert config.max_restarts_per_minute == 3
    
    def test_kwd_config_defaults(self):
        """Test KwdConfig default values and post_init"""
        config = KwdConfig()
        assert config.port == 5003
        assert config.model_path == "models/alexa_v0.1.onnx"
        assert config.confidence_threshold == 0.6
        assert config.cooldown_ms == 1000
        assert config.yes_phrases == ["Yes?", "Yes, Master?", "Sup?", "Yo"]
    
    def test_stt_config_defaults(self):
        """Test SttConfig default values"""
        config = SttConfig()
        assert config.port == 5004
        assert config.device == "cuda"
        assert config.model == "small.en"
        assert config.vad_end_silence_ms == 2000
        assert config.max_recording_sec == 60
        assert config.aec_enabled is True
        assert config.aec_backend == "webrtc_aec3"
    
    def test_llm_config_defaults(self):
        """Test LlmConfig default values"""
        config = LlmConfig()
        assert config.port == 5005
        assert config.endpoint == "localhost:11434"
        assert config.model == "llama3.1:8b-instruct-q4"
        assert config.modelfile_path == "config/Modelfile"
        assert config.request_timeout_ms == 30000
        assert config.stream_chunk_chars == 80
    
    def test_tts_config_defaults(self):
        """Test TtsConfig default values"""
        config = TtsConfig()
        assert config.port == 5006
        assert config.voice == "af_heart"
        assert config.device == "cuda"
        assert config.queue_max_chunks == 200
        assert config.chunk_max_ms == 400
    
    def test_dialog_config_defaults(self):
        """Test DialogConfig default values"""
        config = DialogConfig()
        assert config.followup_window_ms == 4000
        assert config.warmup_greeting == "Hi, Master!"
    
    def test_logging_config_defaults(self):
        """Test LoggingConfig default values"""
        config = LoggingConfig()
        assert config.port == 5001
        assert config.app_reset_on_start is True
        assert config.rotate_size_mb == 10
        assert config.rotate_keep_files == 5
        assert config.logs_dir == "logs"


class TestConfigValidationErrors:
    """Test configuration validation error classes"""
    
    def test_config_validation_error(self):
        """Test ConfigValidationError"""
        error = ConfigValidationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_vram_validation_error(self):
        """Test VRAMValidationError"""
        error = VRAMValidationError("VRAM error")
        assert str(error) == "VRAM error"
        assert isinstance(error, ConfigValidationError)
        assert isinstance(error, Exception)