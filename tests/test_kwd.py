"""Unit tests for Wake Word Detection service"""

import unittest
import time
import threading
import queue
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from services.kwd.service import KwdService
from services.config import KwdConfig
from services.proto import kwd_pb2


class TestKwdService(unittest.TestCase):
    """Test cases for KwdService"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = KwdService(port=5999)  # Use different port for testing
        self.test_config = KwdConfig(
            port=5999,
            model_path="test_model.onnx",
            confidence_threshold=0.6,
            cooldown_ms=1000,
            yes_phrases=["Yes?", "Test phrase", "Hello"]
        )
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self.service, '_is_running') and self.service._is_running:
            self.service._stop_detection()
    
    def test_initialization(self):
        """Test service initialization"""
        self.assertEqual(self.service.service_name, "kwd")
        self.assertEqual(self.service.port, 5999)
        self.assertIsNone(self.service._config)
        self.assertFalse(self.service._is_running)
        self.assertEqual(self.service._total_detections, 0)
        self.assertEqual(self.service._cooldown_rejections, 0)
    
    @patch('os.path.exists')
    @patch('services.kwd.service.Model')
    def test_configure_success(self, mock_model, mock_exists):
        """Test successful configuration"""
        mock_exists.return_value = True
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        self.service.configure(self.test_config)
        
        self.assertEqual(self.service._config, self.test_config)
        mock_model.assert_called_once_with(
            wakeword_model_paths=["test_model.onnx"]
        )
        self.assertEqual(self.service._model, mock_model_instance)
    
    @patch('os.path.exists')
    def test_configure_missing_model(self, mock_exists):
        """Test configuration with missing model file"""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            self.service.configure(self.test_config)
    
    @patch('os.path.exists')
    @patch('services.kwd.service.Model')
    def test_configure_model_load_error(self, mock_model, mock_exists):
        """Test configuration with model loading error"""
        mock_exists.return_value = True
        mock_model.side_effect = Exception("Model load failed")
        
        with self.assertRaises(Exception):
            self.service.configure(self.test_config)
    
    def test_grpc_configure_success(self):
        """Test gRPC Configure method success"""
        with patch.object(self.service, 'configure') as mock_configure:
            request = kwd_pb2.KwdConfig(
                model_path="test.onnx",
                confidence_threshold=0.7,
                cooldown_ms=500,
                yes_phrases=["Yes", "Hello"]
            )
            
            response = self.service.Configure(request, None)
            
            self.assertTrue(response.success)
            self.assertEqual(response.message, "KWD service configured successfully")
            mock_configure.assert_called_once()
    
    def test_grpc_configure_failure(self):
        """Test gRPC Configure method failure"""
        with patch.object(self.service, 'configure', side_effect=Exception("Config error")):
            request = kwd_pb2.KwdConfig(
                model_path="test.onnx",
                confidence_threshold=0.7,
                cooldown_ms=500,
                yes_phrases=["Yes"]
            )
            
            response = self.service.Configure(request, None)
            
            self.assertFalse(response.success)
            self.assertIn("Config error", response.message)
    
    def test_random_confirmation_phrase(self):
        """Test random confirmation phrase selection"""
        self.service._config = self.test_config
        
        # Test multiple calls to ensure randomness works
        phrases = set()
        for _ in range(20):
            phrase = self.service.get_random_confirmation_phrase()
            phrases.add(phrase)
            self.assertIn(phrase, self.test_config.yes_phrases)
        
        # Should get at least 2 different phrases in 20 tries (very likely)
        self.assertGreaterEqual(len(phrases), 1)
    
    def test_random_confirmation_phrase_no_config(self):
        """Test confirmation phrase with no configuration"""
        phrase = self.service.get_random_confirmation_phrase()
        self.assertEqual(phrase, "Yes?")
    
    def test_random_confirmation_phrase_empty_list(self):
        """Test confirmation phrase with empty phrase list"""
        config = KwdConfig(
            port=5999,
            model_path="test.onnx",
            confidence_threshold=0.6,
            cooldown_ms=1000,
            yes_phrases=[]
        )
        self.service._config = config
        
        phrase = self.service.get_random_confirmation_phrase()
        self.assertEqual(phrase, "Yes?")
    
    def test_dialog_id_generation(self):
        """Test dialog ID generation"""
        dialog_id1 = self.service._generate_dialog_id()
        time.sleep(0.001)  # Ensure different timestamp
        dialog_id2 = self.service._generate_dialog_id()
        
        self.assertNotEqual(dialog_id1, dialog_id2)
        self.assertTrue(dialog_id1.startswith("dialog_"))
        self.assertTrue(dialog_id2.startswith("dialog_"))
    
    def test_cooldown_logic(self):
        """Test wake word detection cooldown logic"""
        self.service._config = self.test_config
        
        # Mock prediction with high confidence
        prediction = {"test_model": 0.8}
        
        # First detection should succeed
        with patch.object(self.service, '_generate_dialog_id', return_value="test_dialog_1"):
            self.service._process_prediction(prediction)
        
        self.assertEqual(self.service._total_detections, 1)
        self.assertEqual(self.service._cooldown_rejections, 0)
        
        # Immediate second detection should be rejected (cooldown)
        with patch.object(self.service, '_generate_dialog_id', return_value="test_dialog_2"):
            self.service._process_prediction(prediction)
        
        self.assertEqual(self.service._total_detections, 1)  # Still 1
        self.assertEqual(self.service._cooldown_rejections, 1)
        
        # Wait for cooldown to expire
        time.sleep(1.1)  # Cooldown is 1000ms
        
        # Third detection should succeed
        with patch.object(self.service, '_generate_dialog_id', return_value="test_dialog_3"):
            self.service._process_prediction(prediction)
        
        self.assertEqual(self.service._total_detections, 2)
        self.assertEqual(self.service._cooldown_rejections, 1)
    
    def test_confidence_threshold_filtering(self):
        """Test confidence threshold filtering"""
        self.service._config = self.test_config
        
        # Low confidence should be ignored
        low_confidence_prediction = {"test_model": 0.3}
        self.service._process_prediction(low_confidence_prediction)
        self.assertEqual(self.service._total_detections, 0)
        
        # High confidence should be detected
        high_confidence_prediction = {"test_model": 0.8}
        with patch.object(self.service, '_generate_dialog_id', return_value="test_dialog"):
            self.service._process_prediction(high_confidence_prediction)
        self.assertEqual(self.service._total_detections, 1)
        
        # Exactly at threshold should be detected
        threshold_prediction = {"test_model": 0.6}
        time.sleep(1.1)  # Wait for cooldown
        with patch.object(self.service, '_generate_dialog_id', return_value="test_dialog_2"):
            self.service._process_prediction(threshold_prediction)
        self.assertEqual(self.service._total_detections, 2)
    
    def test_event_queue_management(self):
        """Test event queue management"""
        self.service._config = self.test_config
        
        # Process detection to generate event
        prediction = {"test_model": 0.8}
        with patch.object(self.service, '_generate_dialog_id', return_value="test_dialog"):
            self.service._process_prediction(prediction)
        
        # Check event was queued
        self.assertEqual(self.service._event_queue.qsize(), 1)
        
        # Get event from queue
        event = self.service._event_queue.get_nowait()
        self.assertEqual(event.type, "WAKE_DETECTED")
        self.assertAlmostEqual(event.confidence, 0.8, places=6)
        self.assertEqual(event.dialog_id, "test_dialog")
        self.assertIsNotNone(event.timestamp)
    
    def test_event_queue_full_handling(self):
        """Test handling of full event queue"""
        self.service._config = self.test_config
        
        # Fill up the queue (maxsize=100)
        prediction = {"test_model": 0.8}
        
        # Fill queue to capacity
        for i in range(100):
            time.sleep(0.01)  # Small delay to avoid cooldown
            self.service._last_detection_time = 0  # Reset cooldown for test
            with patch.object(self.service, '_generate_dialog_id', return_value=f"dialog_{i}"):
                self.service._process_prediction(prediction)
        
        self.assertEqual(self.service._event_queue.qsize(), 100)
        
        # Next detection should be dropped (queue full)
        self.service._last_detection_time = 0  # Reset cooldown
        with patch.object(self.service, '_generate_dialog_id', return_value="dropped_dialog"):
            with patch.object(self.service.logger, 'warning') as mock_warning:
                self.service._process_prediction(prediction)
                mock_warning.assert_called_with("Event queue full, dropping wake word event")
        
        # Queue should still be at capacity
        self.assertEqual(self.service._event_queue.qsize(), 100)
    
    def test_detection_statistics(self):
        """Test detection statistics tracking"""
        self.service._config = self.test_config
        self.service._start_time = time.time()
        
        # Initial stats
        stats = self.service.get_detection_stats()
        self.assertEqual(stats['total_detections'], 0)
        self.assertEqual(stats['cooldown_rejections'], 0)
        self.assertFalse(stats['is_running'])
        self.assertGreaterEqual(stats['uptime_seconds'], 0)
        
        # Process some detections
        prediction = {"test_model": 0.8}
        
        # Successful detection
        with patch.object(self.service, '_generate_dialog_id', return_value="dialog_1"):
            self.service._process_prediction(prediction)
        
        # Cooldown rejection
        with patch.object(self.service, '_generate_dialog_id', return_value="dialog_2"):
            self.service._process_prediction(prediction)
        
        # Check updated stats
        stats = self.service.get_detection_stats()
        self.assertEqual(stats['total_detections'], 1)
        self.assertEqual(stats['cooldown_rejections'], 1)
        self.assertEqual(stats['event_queue_size'], 1)
    
    @patch('pyaudio.PyAudio')
    def test_start_stop_detection(self, mock_pyaudio):
        """Test starting and stopping detection"""
        # Mock PyAudio and stream
        mock_pa_instance = Mock()
        mock_stream = Mock()
        mock_pyaudio.return_value = mock_pa_instance
        mock_pa_instance.open.return_value = mock_stream
        
        # Configure service
        self.service._config = self.test_config
        self.service._model = Mock()
        
        # Start detection
        self.service._start_detection()
        
        self.assertTrue(self.service._is_running)
        self.assertIsNotNone(self.service._detection_thread)
        mock_pa_instance.open.assert_called_once()
        
        # Stop detection
        self.service._stop_detection()
        
        self.assertFalse(self.service._is_running)
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
        mock_pa_instance.terminate.assert_called_once()
    
    def test_multiple_model_predictions(self):
        """Test handling multiple model predictions"""
        self.service._config = self.test_config
        
        # Multiple models with different confidences
        prediction = {
            "model1": 0.3,  # Below threshold
            "model2": 0.8,  # Above threshold
            "model3": 0.9   # Above threshold
        }
        
        with patch.object(self.service, '_generate_dialog_id', return_value="test_dialog"):
            self.service._process_prediction(prediction)
        
        # Should only process first model above threshold (model2)
        self.assertEqual(self.service._total_detections, 1)
        
        # Check event details
        event = self.service._event_queue.get_nowait()
        self.assertAlmostEqual(event.confidence, 0.8, places=6)  # Should be model2's confidence


if __name__ == '__main__':
    unittest.main()