"""Speech-to-Text service implementation with Whisper integration"""

import grpc
import whisper
import torch
import threading
import time
import queue
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import pyaudio
import numpy as np
import webrtcvad
from concurrent import futures
import scipy.signal
from collections import deque

from services.base import BaseService
from services.proto import stt_pb2
from services.proto import stt_pb2_grpc
from services.proto import common_pb2
from services.proto import logger_pb2
from services.proto import logger_pb2_grpc
from google.protobuf import empty_pb2


class SttService(BaseService, stt_pb2_grpc.SttServiceServicer):
    """Speech-to-Text service using OpenAI Whisper with VAD"""
    
    def __init__(self, port: int = 5004):
        super().__init__("stt", port)
        
        # Configuration
        self.config: Optional[stt_pb2.SttConfig] = None
        self.model: Optional[whisper.Whisper] = None
        self.device: str = "cpu"
        
        # Audio processing
        self.audio_format = pyaudio.paFloat32
        self.channels = 1
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_size = 1024
        self.vad = None
        
        # Audio device management
        self.input_device_index = None
        self.audio_processor = None
        
        # Recording state
        self.active_sessions: Dict[str, 'RecordingSession'] = {}
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        
        # Logger service client
        self.logger_channel = None
        self.logger_stub = None
        
        self.logger = logging.getLogger("services.stt")
    
    def register_service(self, server):
        """Register STT service with gRPC server"""
        stt_pb2_grpc.add_SttServiceServicer_to_server(self, server)
    
    def configure(self, config):
        """Configure the STT service"""
        self.config = config
        return self._load_model_and_setup()
    
    def Configure(self, request, context):
        """gRPC Configure method"""
        try:
            self.config = request
            success = self._load_model_and_setup()
            
            if success:
                return stt_pb2.ConfigureResponse(
                    success=True,
                    message="STT service configured successfully"
                )
            else:
                return stt_pb2.ConfigureResponse(
                    success=False,
                    message="Failed to configure STT service"
                )
                
        except Exception as e:
            self.logger.error(f"Configuration failed: {e}")
            return stt_pb2.ConfigureResponse(
                success=False,
                message=f"Configuration error: {str(e)}"
            )
    
    def Start(self, request, context):
        """Start recording for a dialog session"""
        dialog_id = request.dialog_id
        
        try:
            if dialog_id in self.active_sessions:
                self.logger.warning(f"Session {dialog_id} already active")
                return empty_pb2.Empty()
            
            # Create new recording session
            session = RecordingSession(
                dialog_id=dialog_id,
                turn=request.turn,
                config=self.config,
                model=self.model,
                device=self.device,
                audio_interface=self.audio_interface,
                vad=self.vad,
                logger=self.logger,
                input_device_index=self.input_device_index,
                audio_processor=self.audio_processor,
                log_callback=self._log_to_dialog
            )
            
            self.active_sessions[dialog_id] = session
            session.start_recording()
            
            self.logger.info(f"Started recording for dialog {dialog_id}, turn {request.turn}")
            return empty_pb2.Empty()
            
        except Exception as e:
            self.logger.error(f"Failed to start recording for {dialog_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to start recording: {str(e)}")
            return empty_pb2.Empty()
    
    def Stop(self, request, context):
        """Stop recording for a dialog session"""
        dialog_id = request.dialog_id
        
        try:
            if dialog_id not in self.active_sessions:
                self.logger.warning(f"No active session for dialog {dialog_id}")
                return empty_pb2.Empty()
            
            session = self.active_sessions[dialog_id]
            session.stop_recording()
            del self.active_sessions[dialog_id]
            
            self.logger.info(f"Stopped recording for dialog {dialog_id}")
            return empty_pb2.Empty()
            
        except Exception as e:
            self.logger.error(f"Failed to stop recording for {dialog_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to stop recording: {str(e)}")
            return empty_pb2.Empty()
    
    def Results(self, request, context):
        """Stream transcription results for a dialog session"""
        dialog_id = request.dialog_id
        
        try:
            if dialog_id not in self.active_sessions:
                self.logger.error(f"No active session for dialog {dialog_id}")
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"No active session for dialog {dialog_id}")
                return
            
            session = self.active_sessions[dialog_id]
            
            # Stream results from the session
            for result in session.get_results():
                if context.is_active():
                    yield result
                else:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error streaming results for {dialog_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error streaming results: {str(e)}")
    
    def _setup_logger_connection(self):
        """Setup connection to logger service"""
        try:
            self.logger_channel = grpc.insecure_channel('localhost:5001')
            self.logger_stub = logger_pb2_grpc.LoggerServiceStub(self.logger_channel)
            self.logger.info("Connected to logger service")
        except Exception as e:
            self.logger.warning(f"Failed to connect to logger service: {e}")
            self.logger_stub = None
    
    def _log_to_dialog(self, dialog_id: str, text: str):
        """Log transcription result to dialog log"""
        if not self.logger_stub:
            return
        
        try:
            dialog_entry = logger_pb2.DialogEntry(
                dialog_id=dialog_id,
                speaker="USER",
                text=text,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger_stub.WriteDialog(dialog_entry)
            self.logger.debug(f"Logged user text to dialog {dialog_id}: '{text}'")
            
        except Exception as e:
            self.logger.warning(f"Failed to log to dialog {dialog_id}: {e}")
    
    def _configure_audio_device(self):
        """Configure the default audio input device"""
        try:
            # Get default input device info
            default_device = self.audio_interface.get_default_input_device_info()
            self.input_device_index = default_device['index']
            
            self.logger.info(f"Using audio input device: {default_device['name']} "
                           f"(index: {self.input_device_index})")
            
            # Validate device supports our requirements
            device_info = self.audio_interface.get_device_info_by_index(self.input_device_index)
            max_channels = int(device_info['maxInputChannels'])
            
            if max_channels < self.channels:
                raise ValueError(f"Device only supports {max_channels} input channels, "
                               f"need {self.channels}")
            
            # Test if device supports our sample rate
            try:
                supported = self.audio_interface.is_format_supported(
                    rate=self.sample_rate,
                    input_device=self.input_device_index,
                    input_channels=self.channels,
                    input_format=self.audio_format
                )
                if not supported:
                    self.logger.warning(f"Device may not support {self.sample_rate}Hz, "
                                      f"will attempt anyway")
            except Exception as e:
                self.logger.warning(f"Could not verify format support: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to configure audio device: {e}")
            self.input_device_index = None
    
    def _load_model_and_setup(self) -> bool:
        """Load Whisper model and setup audio processing"""
        try:
            if not self.config:
                self.logger.error("No configuration provided")
                return False
            
            # Set device
            self.device = self.config.device if self.config.device else "cpu"
            if self.device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Load Whisper model
            model_name = self.config.model if self.config.model else "small.en"
            self.logger.info(f"Loading Whisper model: {model_name} on {self.device}")
            
            self.model = whisper.load_model(model_name, device=self.device)
            self.logger.info("Whisper model loaded successfully")
            
            # Setup audio interface
            self.audio_interface = pyaudio.PyAudio()
            
            # Find and configure default input device
            self._configure_audio_device()
            
            # Setup audio processor with AEC if enabled
            if self.config.aec_enabled:
                self.audio_processor = AudioProcessor(
                    sample_rate=self.sample_rate,
                    aec_backend=self.config.aec_backend,
                    logger=self.logger
                )
                self.logger.info(f"Audio processor initialized with {self.config.aec_backend}")
            
            # Setup VAD
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
            self.logger.info("WebRTC VAD initialized")
            
            # Setup logger service connection
            self._setup_logger_connection()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model and setup: {e}")
            return False
    
    def stop(self, grace_period=5):
        """Stop the service and cleanup resources"""
        # Stop all active sessions
        for dialog_id in list(self.active_sessions.keys()):
            try:
                session = self.active_sessions[dialog_id]
                session.stop_recording()
                del self.active_sessions[dialog_id]
            except Exception as e:
                self.logger.error(f"Error stopping session {dialog_id}: {e}")
        
        # Cleanup audio interface
        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None
        
        # Cleanup logger connection
        if self.logger_channel:
            self.logger_channel.close()
            self.logger_channel = None
            self.logger_stub = None
        
        # Call parent stop
        super().stop(grace_period)


class AudioProcessor:
    """Audio processing with echo cancellation and noise reduction"""
    
    def __init__(self, sample_rate: int, aec_backend: str, logger: logging.Logger):
        self.sample_rate = sample_rate
        self.aec_backend = aec_backend
        self.logger = logger
        
        # Audio processing buffers
        self.frame_size = int(sample_rate * 0.02)  # 20ms frames
        self.buffer = deque(maxlen=self.frame_size * 10)  # 200ms buffer
        
        # High-pass filter for noise reduction (removes low-frequency noise)
        self.highpass_filter = self._create_highpass_filter()
        
        # AEC state (simplified - real AEC3 would require more complex implementation)
        self.aec_enabled = aec_backend == "webrtc_aec3"
        if self.aec_enabled:
            self.logger.info("WebRTC AEC3 simulation enabled")
    
    def _create_highpass_filter(self):
        """Create a high-pass filter to remove low-frequency noise"""
        # High-pass filter at 80Hz to remove low-frequency noise
        nyquist = self.sample_rate / 2
        cutoff = 80.0 / nyquist
        b, a = scipy.signal.butter(4, cutoff, btype='high')
        return (b, a)
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio data with filtering and AEC"""
        try:
            # Apply high-pass filter to remove low-frequency noise
            if len(audio_data) > 0:
                filtered_audio = scipy.signal.filtfilt(
                    self.highpass_filter[0], 
                    self.highpass_filter[1], 
                    audio_data
                )
                
                # Apply AEC if enabled (simplified implementation)
                if self.aec_enabled:
                    filtered_audio = self._apply_aec(filtered_audio)
                
                # Normalize audio to prevent clipping
                max_val = np.max(np.abs(filtered_audio))
                if max_val > 0:
                    filtered_audio = filtered_audio / max_val * 0.8
                
                return filtered_audio.astype(np.float32)
            
            return audio_data
            
        except Exception as e:
            self.logger.warning(f"Audio processing failed: {e}")
            return audio_data
    
    def _apply_aec(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply echo cancellation (simplified implementation)"""
        # This is a simplified AEC implementation
        # Real WebRTC AEC3 would require reference signal from speakers
        # For now, we apply adaptive filtering to reduce echo-like artifacts
        
        # Add to buffer
        self.buffer.extend(audio_data)
        
        # Simple adaptive filter to reduce repetitive patterns (echo-like)
        if len(self.buffer) >= self.frame_size * 2:
            # Get recent history
            history = np.array(list(self.buffer)[-self.frame_size * 2:])
            current = history[-len(audio_data):]
            reference = history[:-len(audio_data)]
            
            # Calculate correlation and reduce correlated components
            if len(reference) == len(current):
                correlation = np.corrcoef(current, reference)[0, 1]
                if not np.isnan(correlation) and correlation > 0.3:
                    # Reduce echo by subtracting correlated component
                    echo_estimate = reference * correlation * 0.3
                    current = current - echo_estimate
                    return current
        
        return audio_data


class RecordingSession:
    """Manages a single recording session with VAD and transcription"""
    
    def __init__(self, dialog_id: str, turn: int, config: stt_pb2.SttConfig,
                 model: whisper.Whisper, device: str, audio_interface: pyaudio.PyAudio,
                 vad: Optional[webrtcvad.Vad], logger: logging.Logger,
                 input_device_index: Optional[int] = None,
                 audio_processor: Optional[AudioProcessor] = None,
                 log_callback=None):
        self.dialog_id = dialog_id
        self.turn = turn
        self.config = config
        self.model = model
        self.device = device
        self.audio_interface = audio_interface
        self.vad = vad
        self.logger = logger
        self.input_device_index = input_device_index
        self.audio_processor = audio_processor
        self.log_callback = log_callback
        
        # Recording state
        self.is_recording = False
        self.audio_stream = None
        self.audio_buffer = []
        self.result_queue = queue.Queue()
        self.recording_thread = None
        self.processing_thread = None
        
        # VAD parameters
        self.vad_end_silence_ms = config.vad_end_silence_ms if config.vad_end_silence_ms > 0 else 2000
        self.max_recording_sec = config.max_recording_sec if config.max_recording_sec > 0 else 60
        
        # Timing
        self.last_speech_time = None
        self.recording_start_time = None
        
        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
    
    def start_recording(self):
        """Start audio recording and processing"""
        if self.is_recording:
            return
        
        try:
            # Open audio stream with device configuration
            stream_kwargs = {
                'format': pyaudio.paFloat32,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size
            }
            
            if self.input_device_index is not None:
                stream_kwargs['input_device_index'] = self.input_device_index
            
            self.audio_stream = self.audio_interface.open(**stream_kwargs)
            
            self.is_recording = True
            self.recording_start_time = time.time()
            self.last_speech_time = time.time()
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info(f"Recording started for dialog {self.dialog_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Close audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        # Wait for threads to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        self.logger.info(f"Recording stopped for dialog {self.dialog_id}")
    
    def get_results(self):
        """Generator that yields transcription results"""
        while self.is_recording or not self.result_queue.empty():
            try:
                # Wait for result with timeout
                result = self.result_queue.get(timeout=0.5)
                yield result
            except queue.Empty:
                if not self.is_recording:
                    break
                continue
    
    def _recording_loop(self):
        """Main recording loop that captures audio"""
        while self.is_recording:
            try:
                # Check maximum recording time
                if time.time() - self.recording_start_time > self.max_recording_sec:
                    self.logger.info(f"Maximum recording time reached for dialog {self.dialog_id}")
                    self._finalize_transcription()
                    break
                
                # Read audio data
                if self.audio_stream:
                    data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    # Process audio with AEC and filtering if available
                    if self.audio_processor:
                        audio_data = self.audio_processor.process_audio(audio_data)
                    
                    # Add to buffer
                    self.audio_buffer.extend(audio_data)
                    
                    # Check for speech activity using VAD
                    if self.vad and len(audio_data) > 0:
                        # Convert to 16-bit PCM for VAD
                        pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                        
                        # VAD requires specific frame sizes (10, 20, or 30ms)
                        frame_duration = 30  # ms
                        frame_size = int(self.sample_rate * frame_duration / 1000)
                        
                        if len(pcm_data) >= frame_size * 2:  # 2 bytes per sample
                            frame = pcm_data[:frame_size * 2]
                            is_speech = self.vad.is_speech(frame, self.sample_rate)
                            
                            if is_speech:
                                self.last_speech_time = time.time()
                    else:
                        # Without VAD, assume continuous speech
                        self.last_speech_time = time.time()
                    
                    # Check for silence timeout
                    silence_duration = time.time() - self.last_speech_time
                    if silence_duration * 1000 >= self.vad_end_silence_ms:
                        self.logger.info(f"Silence detected, finalizing transcription for dialog {self.dialog_id}")
                        self._finalize_transcription()
                        break
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in recording loop: {e}")
                break
    
    def _processing_loop(self):
        """Processing loop for handling transcription"""
        # This could be used for real-time processing if needed
        # For now, we only do final transcription when recording stops
        pass
    
    def _finalize_transcription(self):
        """Finalize and transcribe the recorded audio"""
        if not self.audio_buffer:
            self.logger.warning(f"No audio data to transcribe for dialog {self.dialog_id}")
            return
        
        try:
            # Convert audio buffer to numpy array
            audio_array = np.array(self.audio_buffer, dtype=np.float32)
            
            # Ensure audio is the right length and format for Whisper
            if len(audio_array) < self.sample_rate * 0.1:  # Less than 100ms
                self.logger.warning(f"Audio too short for transcription: {len(audio_array)} samples")
                return
            
            # Transcribe with Whisper
            self.logger.info(f"Transcribing audio for dialog {self.dialog_id}")
            result = self.model.transcribe(
                audio_array,
                language="en",
                task="transcribe",
                fp16=self.device == "cuda"
            )
            
            text = result["text"].strip()
            if text:
                # Create result message
                timestamp = datetime.now().isoformat()
                stt_result = stt_pb2.SttResult(
                    final=True,
                    text=text,
                    timestamp=timestamp
                )
                
                # Log to dialog if callback provided
                if self.log_callback:
                    self.log_callback(self.dialog_id, text)
                
                # Add to result queue
                self.result_queue.put(stt_result)
                self.logger.info(f"Transcription complete for dialog {self.dialog_id}: '{text}'")
            else:
                self.logger.info(f"No speech detected in audio for dialog {self.dialog_id}")
                
        except Exception as e:
            self.logger.error(f"Transcription failed for dialog {self.dialog_id}: {e}")
        finally:
            # Stop recording after transcription
            self.is_recording = False


def main():
    """Main entry point for STT service"""
    import sys
    import signal
    from services.config import ConfigManager
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_manager = ConfigManager()
    if not config_manager.load_config("config/config.txt"):
        print("Failed to load configuration")
        sys.exit(1)
    
    stt_config = config_manager.get_stt_config()
    
    # Create and configure service
    service = SttService()
    
    # Convert config to protobuf message
    config_proto = stt_pb2.SttConfig(
        device=stt_config.device,
        model=stt_config.model,
        vad_end_silence_ms=stt_config.vad_end_silence_ms,
        max_recording_sec=stt_config.max_recording_sec,
        aec_enabled=stt_config.aec_enabled,
        aec_backend=stt_config.aec_backend
    )
    
    if not service.configure(config_proto):
        print("Failed to configure STT service")
        sys.exit(1)
    
    # Setup signal handling
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start service
    try:
        service.start()
        print(f"STT service running on port {service.port}")
        service.wait_for_termination()
    except Exception as e:
        print(f"Failed to start STT service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()