"""Wake Word Detection service implementation using openWakeWord"""

import os
import time
import threading
import queue
import logging
import random
from datetime import datetime
from typing import Optional, List, Iterator
import numpy as np

import grpc
from google.protobuf import empty_pb2

# Import openWakeWord
try:
    import openwakeword
    from openwakeword import Model
except ImportError as e:
    raise ImportError(f"openWakeWord not installed: {e}. Run 'uv add openwakeword'")

# Import audio processing
try:
    import pyaudio
except ImportError as e:
    raise ImportError(f"PyAudio not installed: {e}. Run 'uv add pyaudio'")

# Import protobuf definitions
from services.proto import kwd_pb2, kwd_pb2_grpc
from services.base import BaseService
from services.config import KwdConfig


class KwdService(BaseService, kwd_pb2_grpc.KwdServiceServicer):
    """Wake Word Detection service using openWakeWord"""
    
    def __init__(self, port: int = 5003):
        super().__init__("kwd", port)
        
        # Configuration
        self._config: Optional[KwdConfig] = None
        
        # Wake word detection
        self._model: Optional[Model] = None
        self._audio_stream: Optional[pyaudio.Stream] = None
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        
        # State management
        self._is_running = False
        self._detection_thread: Optional[threading.Thread] = None
        self._event_queue = queue.Queue(maxsize=100)  # Limit queue size
        self._last_detection_time = 0.0
        
        # Statistics
        self._total_detections = 0
        self._cooldown_rejections = 0
        self._start_time = None
        
        # Audio parameters (openWakeWord requirements)
        self._sample_rate = 16000
        self._chunk_size = 1280  # 80ms at 16kHz
        self._channels = 1
        self._format = pyaudio.paInt16
        
        self.logger.info("KWD service initialized")
    
    def register_service(self, server):
        """Register the KWD service with gRPC server"""
        kwd_pb2_grpc.add_KwdServiceServicer_to_server(self, server)
        self.logger.info("KWD service registered with gRPC server")
    
    def configure(self, config: KwdConfig):
        """Configure the service with provided configuration"""
        self._config = config
        self.logger.info(f"KWD service configured with model: {config.model_path}")
        
        # Validate model file exists
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Wake word model not found: {config.model_path}")
        
        # Initialize openWakeWord model
        try:
            # Load the custom model
            self._model = Model(
                wakeword_model_paths=[config.model_path]
            )
            self.logger.info(f"Loaded wake word model: {config.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load wake word model: {e}")
            raise
    
    def Configure(self, request: kwd_pb2.KwdConfig, context) -> kwd_pb2.ConfigureResponse:
        """gRPC Configure method"""
        try:
            # Convert protobuf config to dataclass
            config = KwdConfig(
                port=self.port,
                model_path=request.model_path,
                confidence_threshold=request.confidence_threshold,
                cooldown_ms=request.cooldown_ms,
                yes_phrases=list(request.yes_phrases)
            )
            
            self.configure(config)
            
            return kwd_pb2.ConfigureResponse(
                success=True,
                message="KWD service configured successfully"
            )
            
        except Exception as e:
            self.logger.error(f"Configuration failed: {e}")
            return kwd_pb2.ConfigureResponse(
                success=False,
                message=f"Configuration failed: {str(e)}"
            )
    
    def Start(self, request: empty_pb2.Empty, context) -> empty_pb2.Empty:
        """gRPC Start method"""
        try:
            self._start_detection()
            self.logger.info("Wake word detection started")
            return empty_pb2.Empty()
            
        except Exception as e:
            self.logger.error(f"Failed to start wake word detection: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to start: {str(e)}")
            return empty_pb2.Empty()
    
    def Stop(self, request: empty_pb2.Empty, context) -> empty_pb2.Empty:
        """gRPC Stop method"""
        try:
            self._stop_detection()
            self.logger.info("Wake word detection stopped")
            return empty_pb2.Empty()
            
        except Exception as e:
            self.logger.error(f"Failed to stop wake word detection: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to stop: {str(e)}")
            return empty_pb2.Empty()
    
    def Events(self, request: empty_pb2.Empty, context) -> Iterator[kwd_pb2.KwdEvent]:
        """gRPC server-streaming Events method"""
        self.logger.info("Client connected to wake word events stream")
        
        try:
            while not context.is_active() or not self._shutdown_event.is_set():
                try:
                    # Get event from queue with timeout
                    event = self._event_queue.get(timeout=1.0)
                    yield event
                    
                except queue.Empty:
                    # Timeout - continue loop to check context
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in Events stream: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Stream error: {str(e)}")
        
        finally:
            self.logger.info("Client disconnected from wake word events stream")
    
    def _start_detection(self):
        """Start wake word detection in background thread"""
        if self._is_running:
            self.logger.warning("Wake word detection already running")
            return
        
        if not self._config or not self._model:
            raise RuntimeError("Service not configured")
        
        # Initialize PyAudio
        self._pyaudio = pyaudio.PyAudio()
        
        # Open audio stream
        try:
            self._audio_stream = self._pyaudio.open(
                format=self._format,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=self._chunk_size
            )
            
            self.logger.info(f"Audio stream opened: {self._sample_rate}Hz, {self._channels} channel(s)")
            
        except Exception as e:
            self.logger.error(f"Failed to open audio stream: {e}")
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None
            raise
        
        # Start detection thread
        self._is_running = True
        self._start_time = time.time()
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._detection_thread.start()
        
        self.logger.info("Wake word detection thread started")
    
    def _stop_detection(self):
        """Stop wake word detection"""
        if not self._is_running:
            return
        
        # Stop detection loop
        self._is_running = False
        
        # Wait for thread to finish
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=2.0)
        
        # Close audio stream
        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
            self._audio_stream = None
        
        # Terminate PyAudio
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        
        self.logger.info("Wake word detection stopped")
    
    def _detection_loop(self):
        """Main detection loop running in background thread"""
        self.logger.info("Wake word detection loop started")
        
        try:
            while self._is_running:
                try:
                    # Read audio chunk
                    audio_data = self._audio_stream.read(
                        self._chunk_size, 
                        exception_on_overflow=False
                    )
                    
                    # Convert to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    
                    # Normalize to [-1, 1] range
                    audio_array = audio_array / 32768.0
                    
                    # Run wake word detection
                    prediction = self._model.predict(audio_array)
                    
                    # Check for wake word detection
                    self._process_prediction(prediction)
                    
                except Exception as e:
                    if self._is_running:  # Only log if we're supposed to be running
                        self.logger.error(f"Error in detection loop: {e}")
                        time.sleep(0.1)  # Brief pause before retrying
                    
        except Exception as e:
            self.logger.error(f"Fatal error in detection loop: {e}")
        
        finally:
            self.logger.info("Wake word detection loop ended")
    
    def _process_prediction(self, prediction: dict):
        """Process wake word prediction and emit events if detected"""
        if not self._config:
            return
        
        # Check each model's prediction
        for model_name, confidence in prediction.items():
            if confidence >= self._config.confidence_threshold:
                current_time = time.time()
                
                # Check cooldown period
                if current_time - self._last_detection_time < (self._config.cooldown_ms / 1000.0):
                    self._cooldown_rejections += 1
                    self.logger.debug(
                        f"Wake word detected but in cooldown period "
                        f"(confidence: {confidence:.3f}, model: {model_name})"
                    )
                    continue
                
                # Update last detection time and statistics
                self._last_detection_time = current_time
                self._total_detections += 1
                
                # Generate dialog ID
                dialog_id = self._generate_dialog_id()
                
                # Get random confirmation phrase
                confirmation_phrase = self.get_random_confirmation_phrase()
                
                # Log detection with confirmation phrase
                self.logger.info(
                    f"Wake word detected: {model_name} "
                    f"(confidence: {confidence:.3f}, dialog_id: {dialog_id}, "
                    f"confirmation: '{confirmation_phrase}')"
                )
                
                # Create and emit event
                event = kwd_pb2.KwdEvent(
                    type="WAKE_DETECTED",
                    confidence=confidence,
                    timestamp=datetime.now().isoformat(),
                    dialog_id=dialog_id
                )
                
                # Add to event queue
                try:
                    self._event_queue.put_nowait(event)
                    self.logger.debug(f"Wake word event queued for dialog {dialog_id}")
                except queue.Full:
                    self.logger.warning("Event queue full, dropping wake word event")
                
                # Only process first detection in this chunk
                break
    
    def _generate_dialog_id(self) -> str:
        """Generate a unique dialog ID with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        return f"dialog_{timestamp}"
    
    def get_random_confirmation_phrase(self) -> str:
        """Get a random confirmation phrase from configuration"""
        if not self._config or not self._config.yes_phrases:
            return "Yes?"
        
        return random.choice(self._config.yes_phrases)
    
    def get_detection_stats(self) -> dict:
        """Get wake word detection statistics"""
        uptime = time.time() - self._start_time if self._start_time else 0
        
        return {
            'total_detections': self._total_detections,
            'cooldown_rejections': self._cooldown_rejections,
            'uptime_seconds': uptime,
            'is_running': self._is_running,
            'last_detection_time': self._last_detection_time,
            'event_queue_size': self._event_queue.qsize()
        }
    
    def stop(self, grace_period=5):
        """Override base stop to ensure detection is stopped"""
        self._stop_detection()
        super().stop(grace_period)


def main():
    """Main entry point for standalone KWD service"""
    import sys
    import signal
    from services.config import ConfigManager
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("kwd.main")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load()
        kwd_config = config['kwd']
        
        # Create and configure service
        service = KwdService(kwd_config.port)
        service.configure(kwd_config)
        
        # Set up signal handling
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            service.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start service
        service.start()
        logger.info(f"KWD service running on port {kwd_config.port}")
        
        # Wait for termination
        service.wait_for_termination()
        
    except Exception as e:
        logger.error(f"Failed to start KWD service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()