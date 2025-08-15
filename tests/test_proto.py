"""Tests for protobuf definitions and gRPC service stubs"""

import pytest
from services.proto import (
    DialogRef, KwdConfig, SttConfig, LlmConfig, TtsConfig,
    LogEntry, DialogEntry, UserQuery, LlmChunk, KwdEvent,
    LoaderServiceStub, KwdServiceStub, SttServiceStub,
    LlmServiceStub, TtsServiceStub, LoggerServiceStub
)


class TestProtobufMessages:
    """Test protobuf message creation and serialization"""
    
    def test_dialog_ref_creation(self):
        """Test DialogRef message creation"""
        dialog_ref = DialogRef(dialog_id="test_dialog_123", turn=1)
        assert dialog_ref.dialog_id == "test_dialog_123"
        assert dialog_ref.turn == 1
    
    def test_kwd_config_creation(self):
        """Test KwdConfig message creation"""
        config = KwdConfig(
            model_path="models/alexa_v0.1.onnx",
            confidence_threshold=0.6,
            cooldown_ms=1000,
            yes_phrases=["Yes?", "Yes, Master?", "Sup?", "Yo"]
        )
        assert config.model_path == "models/alexa_v0.1.onnx"
        assert abs(config.confidence_threshold - 0.6) < 0.001
        assert config.cooldown_ms == 1000
        assert len(config.yes_phrases) == 4
    
    def test_stt_config_creation(self):
        """Test SttConfig message creation"""
        config = SttConfig(
            device="cuda",
            model="small.en",
            vad_end_silence_ms=2000,
            max_recording_sec=60,
            aec_enabled=True,
            aec_backend="webrtc_aec3"
        )
        assert config.device == "cuda"
        assert config.model == "small.en"
        assert config.vad_end_silence_ms == 2000
        assert config.aec_enabled is True
    
    def test_llm_config_creation(self):
        """Test LlmConfig message creation"""
        config = LlmConfig(
            endpoint="localhost:11434",
            model="llama3.1:8b-instruct-q4",
            modelfile_path="config/Modelfile",
            request_timeout_ms=30000,
            stream_chunk_chars=80
        )
        assert config.endpoint == "localhost:11434"
        assert config.model == "llama3.1:8b-instruct-q4"
        assert config.request_timeout_ms == 30000
    
    def test_tts_config_creation(self):
        """Test TtsConfig message creation"""
        config = TtsConfig(
            voice="af_heart",
            device="cuda",
            queue_max_chunks=200,
            chunk_max_ms=400
        )
        assert config.voice == "af_heart"
        assert config.device == "cuda"
        assert config.queue_max_chunks == 200
    
    def test_log_entry_creation(self):
        """Test LogEntry message creation"""
        entry = LogEntry(
            timestamp="2024-01-01T12:00:00Z",
            level="INFO",
            service="test_service",
            event="test_event",
            details="Test details"
        )
        assert entry.timestamp == "2024-01-01T12:00:00Z"
        assert entry.level == "INFO"
        assert entry.service == "test_service"
    
    def test_user_query_creation(self):
        """Test UserQuery message creation"""
        query = UserQuery(
            text="Hello, how are you?",
            dialog_id="dialog_123",
            turn=1
        )
        assert query.text == "Hello, how are you?"
        assert query.dialog_id == "dialog_123"
        assert query.turn == 1
    
    def test_llm_chunk_creation(self):
        """Test LlmChunk message creation"""
        chunk = LlmChunk(text="Hello", eot=False)
        assert chunk.text == "Hello"
        assert chunk.eot is False
        
        final_chunk = LlmChunk(text="", eot=True)
        assert final_chunk.eot is True
    
    def test_kwd_event_creation(self):
        """Test KwdEvent message creation"""
        event = KwdEvent(
            type="WAKE_DETECTED",
            confidence=0.85,
            timestamp="2024-01-01T12:00:00Z",
            dialog_id="dialog_123"
        )
        assert event.type == "WAKE_DETECTED"
        assert abs(event.confidence - 0.85) < 0.001
        assert event.dialog_id == "dialog_123"


class TestServiceStubs:
    """Test that service stubs can be imported and instantiated"""
    
    def test_service_stubs_importable(self):
        """Test that all service stubs can be imported"""
        # This test passes if imports work without errors
        assert LoaderServiceStub is not None
        assert KwdServiceStub is not None
        assert SttServiceStub is not None
        assert LlmServiceStub is not None
        assert TtsServiceStub is not None
        assert LoggerServiceStub is not None


class TestMessageSerialization:
    """Test protobuf message serialization and deserialization"""
    
    def test_dialog_ref_serialization(self):
        """Test DialogRef serialization roundtrip"""
        original = DialogRef(dialog_id="test_123", turn=5)
        
        # Serialize to bytes
        serialized = original.SerializeToString()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Deserialize back
        deserialized = DialogRef()
        deserialized.ParseFromString(serialized)
        
        # Verify data integrity
        assert deserialized.dialog_id == original.dialog_id
        assert deserialized.turn == original.turn
    
    def test_kwd_config_serialization(self):
        """Test KwdConfig serialization roundtrip"""
        original = KwdConfig(
            model_path="test/path",
            confidence_threshold=0.7,
            cooldown_ms=1500,
            yes_phrases=["Yes", "OK"]
        )
        
        # Serialize and deserialize
        serialized = original.SerializeToString()
        deserialized = KwdConfig()
        deserialized.ParseFromString(serialized)
        
        # Verify data integrity
        assert deserialized.model_path == original.model_path
        assert deserialized.confidence_threshold == original.confidence_threshold
        assert deserialized.cooldown_ms == original.cooldown_ms
        assert list(deserialized.yes_phrases) == list(original.yes_phrases)