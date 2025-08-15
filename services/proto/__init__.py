"""
gRPC protobuf definitions and generated stubs for AI Voice Assistant services.

This module provides easy access to all protobuf message types and service stubs.
"""

# Import all generated protobuf modules
from . import common_pb2
from . import common_pb2_grpc

from . import loader_pb2
from . import loader_pb2_grpc

from . import logger_pb2
from . import logger_pb2_grpc

from . import kwd_pb2
from . import kwd_pb2_grpc

from . import stt_pb2
from . import stt_pb2_grpc

from . import llm_pb2
from . import llm_pb2_grpc

from . import tts_pb2
from . import tts_pb2_grpc

# Re-export commonly used message types for convenience
from .common_pb2 import DialogRef, Timestamp, ServiceStatus

from .loader_pb2 import (
    StartServiceRequest, StartServiceResponse,
    StopServiceRequest, StopServiceResponse,
    GetPidsResponse, WakeWordEvent, DialogCompleteEvent
)

from .logger_pb2 import LogEntry, DialogEntry, NewDialogResponse

from .kwd_pb2 import KwdConfig, ConfigureResponse as KwdConfigureResponse, KwdEvent

from .stt_pb2 import (
    SttConfig, ConfigureResponse as SttConfigureResponse, SttResult
)

from .llm_pb2 import (
    LlmConfig, ConfigureResponse as LlmConfigureResponse, 
    UserQuery, LlmChunk
)

from .tts_pb2 import (
    TtsConfig, ConfigureResponse as TtsConfigureResponse,
    TextInput, TtsPlayAck, TtsEvent
)

# Re-export service stubs for convenience
from .loader_pb2_grpc import LoaderServiceStub, LoaderServiceServicer
from .logger_pb2_grpc import LoggerServiceStub, LoggerServiceServicer
from .kwd_pb2_grpc import KwdServiceStub, KwdServiceServicer
from .stt_pb2_grpc import SttServiceStub, SttServiceServicer
from .llm_pb2_grpc import LlmServiceStub, LlmServiceServicer
from .tts_pb2_grpc import TtsServiceStub, TtsServiceServicer

__all__ = [
    # Common types
    'DialogRef', 'Timestamp', 'ServiceStatus',
    
    # Loader service
    'StartServiceRequest', 'StartServiceResponse',
    'StopServiceRequest', 'StopServiceResponse',
    'GetPidsResponse', 'WakeWordEvent', 'DialogCompleteEvent',
    'LoaderServiceStub', 'LoaderServiceServicer',
    
    # Logger service
    'LogEntry', 'DialogEntry', 'NewDialogResponse',
    'LoggerServiceStub', 'LoggerServiceServicer',
    
    # KWD service
    'KwdConfig', 'KwdConfigureResponse', 'KwdEvent',
    'KwdServiceStub', 'KwdServiceServicer',
    
    # STT service
    'SttConfig', 'SttConfigureResponse', 'SttResult',
    'SttServiceStub', 'SttServiceServicer',
    
    # LLM service
    'LlmConfig', 'LlmConfigureResponse', 'UserQuery', 'LlmChunk',
    'LlmServiceStub', 'LlmServiceServicer',
    
    # TTS service
    'TtsConfig', 'TtsConfigureResponse', 'TextInput', 'TtsPlayAck', 'TtsEvent',
    'TtsServiceStub', 'TtsServiceServicer',
    
    # Raw protobuf modules (for advanced usage)
    'common_pb2', 'common_pb2_grpc',
    'loader_pb2', 'loader_pb2_grpc',
    'logger_pb2', 'logger_pb2_grpc',
    'kwd_pb2', 'kwd_pb2_grpc',
    'stt_pb2', 'stt_pb2_grpc',
    'llm_pb2', 'llm_pb2_grpc',
    'tts_pb2', 'tts_pb2_grpc',
]