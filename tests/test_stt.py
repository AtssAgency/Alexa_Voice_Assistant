"""Unit tests for Speech-to-Text service"""

import unittest
import time
import threading
import queue
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from services.stt.service import SttService, AudioProcessor
from services.proto import stt_pb2
from services.proto import common_pb2


class TestSttService(unittest.TestCase):
    """Test cases for SttService"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = SttService(port=5998)  # Use different port for testing
        self.test_config = stt_pb2.SttConfig(
            device="cpu",
            model="small.en",
            vad_end_silence_ms=2000,
            max_recording_sec=60,
            aec_enabled=True,
            aec_backend="webrtc_aec3"
        )
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.service, 'active_sessions'):
            # Stop any active sessions
            for dialog_id in list(self.service.active_sessions.keys()):
                try:
                    self.service.active_sessions[dialog_id].stop_recording()
                    del self.service.active_sessions[dialog_id]
                except:
                    pass
        
        if hasattr(self.service, 'audio_interface') and self.service.audio_interface:
            try:
                self.service.audio_interface.terminate()
            except:
                pass
    
    def test_initialization(self):
        """Test service initialization"""
        self.assertEqual(self.service.service_name, "stt")
        self.assertEqual(self.service.port, 5998)
        self.assertIsNone(self.service.config)
        self.assertIsNone(self.service.model)
        self.assertEqual(self.service.device, "cpu")
        self.assertEqual(len(self.service.active_sessions), 0)
    
    @patch('whisper.load_model')
    @patch('pyaudio.PyAudio')
    @patch('webrtcvad.Vad')
    def test_configure_success(self, mock_vad, mock_pyaudio, mock_whisper):
        """Test successful configuration"""
        # Mock dependencies
        mock_model = Mock()
        mock_whisper.return_value = mock_model
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_vad_instance = Mock()
        mock_vad.return_value = mock_vad_instance
        
        # Mock device info
        mock_pa_instance.get_default_input_device_info.return_value = {
            'index': 0,
            'name': 'Test Microphone'
        }
        mock_pa_instance.get_device_info_by_index.return_value = {
            'maxInputChannels': 2
        }
        mock_pa_instance.is_format_supported.return_value = True
        
        # Configure service
        self.service.config = self.test_config
        success = self.service._load_model_and_setup()
        
        # Verify model loading
        mock_whisper.assert_called_once_with("small.en", device="cpu")
        self.assertEqual(self.service.model, mock_model)
        
        # Verify audio setup
        mock_pyaudio.assert_called_once()
        self.assertEqual(self.service.audio_interface, mock_pa_instance)
        
        # Verify VAD setup
        mock_vad.assert_called_once_with(2)
        self.assertEqual(self.service.vad, mock_vad_instance)
    
    @patch('torch.cuda.is_available')
    @patch('whisper.load_model')
    @patch('pyaudio.PyAudio')
    def test_configure_cuda_fallback(self, mock_pyaudio, mock_whisper, mock_cuda):
        """Test CUDA fallback to CPU when CUDA unavailable"""
        mock_cuda.return_value = False
        mock_whisper.return_value = Mock()
        mock_pa_instance = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        
        # Mock device info
        mock_pa_instance.get_default_input_device_info.return_value = {
            'index': 0,
            'name': 'Test Microphone'
        }
        mock_pa_instance.get_device_info_by_index.return_value = {
            'maxInputChannels': 2
        }
        
        # Configure with CUDA requested
        cuda_config = stt_pb2.SttConfig(
            device="cuda",
            model="small.en",
            vad_end_silence_ms=2000,
            max_recording_sec=60,
            aec_enabled=False
        )
        self.service.config = cuda_config
        
        success = self.service._load_model_and_setup()
        
        # Should fallback to CPU
        self.assertEqual(self.service.device, "cpu")
        mock_whisper.assert_called_once_with("small.en", device="cpu")
    
    def test_grpc_configure_success(self):
        """Test gRPC Configure method success"""
        with patch.object(self.service, '_load_model_and_setup', return_value=True):
            response = self.service.Configure(self.test_config, None)
            
            self.assertTrue(response.success)
            self.assertEqual(response.message, "STT service configured successfully")
            self.assertEqual(self.service.config, self.test_config)
    
    def test_grpc_configure_failure(self):
        """Test gRPC Configure method failure"""
        with patch.object(self.service, '_load_model_and_setup', return_value=False):
            response = self.service.Configure(self.test_config, None)
            
            self.assertFalse(response.success)
            self.assertEqual(response.message, "Failed to configure STT service")
    
    def test_grpc_configure_exception(self):
        """Test gRPC Configure method with exception"""
        with patch.object(self.service, '_load_model_and_setup', 
                         side_effect=Exception("Test error")):
            response = self.service.Configure(self.test_config, None)
            
            self.assertFalse(response.success)
            self.assertIn("Test error", response.message)
    
    @patch('services.stt.service.RecordingSession')
    def test_start_recording_session(self, mock_session_class):
        """Test starting a recording session"""
        # Setup service
        self.service.config = self.test_config
        self.service.model = Mock()
        self.service.audio_interface = Mock()
        self.service.vad = Mock()
        
        # Mock session
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        # Start recording
        dialog_ref = common_pb2.DialogRef(dialog_id="test_dialog", turn=1)
        response = self.service.Start(dialog_ref, None)
        
        # Verify session creation and start
        mock_session_class.assert_called_once()
        mock_session.start_recording.assert_called_once()
        self.assertIn("test_dialog", self.service.active_sessions)
        self.assertEqual(self.service.active_sessions["test_dialog"], mock_session)
    
    def test_start_recording_duplicate_session(self):
        """Test starting recording for existing session"""
        # Add existing session
        mock_session = Mock()
        self.service.active_sessions["test_dialog"] = mock_session
        
        # Try to start again
        dialog_ref = common_pb2.DialogRef(dialog_id="test_dialog", turn=1)
        response = self.service.Start(dialog_ref, None)
        
        # Should not create new session
        self.assertEqual(len(self.service.active_sessions), 1)
    
    def test_stop_recording_session(self):
        """Test stopping a recording session"""
        # Add active session
        mock_session = Mock()
        self.service.active_sessions["test_dialog"] = mock_session
        
        # Stop recording
        dialog_ref = common_pb2.DialogRef(dialog_id="test_dialog", turn=1)
        response = self.service.Stop(dialog_ref, None)
        
        # Verify session stopped and removed
        mock_session.stop_recording.assert_called_once()
        self.assertNotIn("test_dialog", self.service.active_sessions)
    
    def test_stop_recording_nonexistent_session(self):
        """Test stopping recording for non-existent session"""
        dialog_ref = common_pb2.DialogRef(dialog_id="nonexistent", turn=1)
        response = self.service.Stop(dialog_ref, None)
        
        # Should handle gracefully
        self.assertEqual(len(self.service.active_sessions), 0)
    
    def test_results_streaming(self):
        """Test streaming results from session"""
        # Create mock session with results
        mock_session = Mock()
        test_results = [
            stt_pb2.SttResult(final=True, text="Hello world", timestamp="2023-01-01T00:00:00"),
            stt_pb2.SttResult(final=True, text="How are you", timestamp="2023-01-01T00:00:01")
        ]
        mock_session.get_results.return_value = iter(test_results)
        
        self.service.active_sessions["test_dialog"] = mock_session
        
        # Get results
        dialog_ref = common_pb2.DialogRef(dialog_id="test_dialog", turn=1)
        mock_context = Mock()
        mock_context.is_active.return_value = True
        
        results = list(self.service.Results(dialog_ref, mock_context))
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "Hello world")
        self.assertEqual(results[1].text, "How are you")
    
    def test_results_no_session(self):
        """Test streaming results for non-existent session"""
        dialog_ref = common_pb2.DialogRef(dialog_id="nonexistent", turn=1)
        mock_context = Mock()
        
        # Should set error status
        list(self.service.Results(dialog_ref, mock_context))
        
        mock_context.set_code.assert_called_once()
        mock_context.set_details.assert_called_once()
    
    def test_logger_connection_setup(self):
        """Test logger service connection setup"""
        with patch('grpc.insecure_channel') as mock_channel:
            with patch('services.proto.logger_pb2_grpc.LoggerServiceStub') as mock_stub:
                mock_channel_instance = Mock()
                mock_stub_instance = Mock()
                mock_channel.return_value = mock_channel_instance
                mock_stub.return_value = mock_stub_instance
                
                self.service._setup_logger_connection()
                
                mock_channel.assert_called_once_with('localhost:5001')
                mock_stub.assert_called_once_with(mock_channel_instance)
                self.assertEqual(self.service.logger_channel, mock_channel_instance)
                self.assertEqual(self.service.logger_stub, mock_stub_instance)
    
    def test_dialog_logging(self):
        """Test logging to dialog service"""
        # Setup mock logger stub
        mock_stub = Mock()
        self.service.logger_stub = mock_stub
        
        # Log text
        self.service._log_to_dialog("test_dialog", "Hello world")
        
        # Verify call
        mock_stub.WriteDialog.assert_called_once()
        call_args = mock_stub.WriteDialog.call_args[0][0]
        self.assertEqual(call_args.dialog_id, "test_dialog")
        self.assertEqual(call_args.speaker, "USER")
        self.assertEqual(call_args.text, "Hello world")
        self.assertIsNotNone(call_args.timestamp)
    
    def test_dialog_logging_no_stub(self):
        """Test dialog logging when no logger stub available"""
        self.service.logger_stub = None
        
        # Should not raise exception
        self.service._log_to_dialog("test_dialog", "Hello world")


class TestAudioProcessor(unittest.TestCase):
    """Test cases for AudioProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = AudioProcessor(
            sample_rate=16000,
            aec_backend="webrtc_aec3",
            logger=Mock()
        )
    
    def test_initialization(self):
        """Test audio processor initialization"""
        self.assertEqual(self.processor.sample_rate, 16000)
        self.assertEqual(self.processor.aec_backend, "webrtc_aec3")
        self.assertTrue(self.processor.aec_enabled)
        self.assertIsNotNone(self.processor.highpass_filter)
    
    def test_audio_processing(self):
        """Test basic audio processing"""
        # Create test audio data
        test_audio = np.random.randn(1024).astype(np.float32)
        
        # Process audio
        processed = self.processor.process_audio(test_audio)
        
        # Should return processed audio
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.dtype, np.float32)
        self.assertEqual(len(processed), len(test_audio))
    
    def test_empty_audio_processing(self):
        """Test processing empty audio"""
        empty_audio = np.array([])
        
        processed = self.processor.process_audio(empty_audio)
        
        # Should return empty array unchanged
        self.assertEqual(len(processed), 0)
    
    def test_aec_disabled(self):
        """Test audio processing with AEC disabled"""
        processor = AudioProcessor(
            sample_rate=16000,
            aec_backend="none",
            logger=Mock()
        )
        
        self.assertFalse(processor.aec_enabled)
        
        test_audio = np.random.randn(1024).astype(np.float32)
        processed = processor.process_audio(test_audio)
        
        # Should still process (filter) but not apply AEC
        self.assertIsInstance(processed, np.ndarray)
    
    def test_highpass_filter_creation(self):
        """Test high-pass filter creation"""
        filter_coeffs = self.processor._create_highpass_filter()
        
        # Should return tuple of (b, a) coefficients
        self.assertIsInstance(filter_coeffs, tuple)
        self.assertEqual(len(filter_coeffs), 2)
        self.assertIsInstance(filter_coeffs[0], np.ndarray)  # b coefficients
        self.assertIsInstance(filter_coeffs[1], np.ndarray)  # a coefficients


if __name__ == '__main__':
    unittest.main()