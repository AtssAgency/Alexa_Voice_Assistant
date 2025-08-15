import common_pb2 as _common_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SttConfig(_message.Message):
    __slots__ = ("device", "model", "vad_end_silence_ms", "max_recording_sec", "aec_enabled", "aec_backend")
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    VAD_END_SILENCE_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_RECORDING_SEC_FIELD_NUMBER: _ClassVar[int]
    AEC_ENABLED_FIELD_NUMBER: _ClassVar[int]
    AEC_BACKEND_FIELD_NUMBER: _ClassVar[int]
    device: str
    model: str
    vad_end_silence_ms: int
    max_recording_sec: int
    aec_enabled: bool
    aec_backend: str
    def __init__(self, device: _Optional[str] = ..., model: _Optional[str] = ..., vad_end_silence_ms: _Optional[int] = ..., max_recording_sec: _Optional[int] = ..., aec_enabled: bool = ..., aec_backend: _Optional[str] = ...) -> None: ...

class ConfigureResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class SttResult(_message.Message):
    __slots__ = ("final", "text", "timestamp")
    FINAL_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    final: bool
    text: str
    timestamp: str
    def __init__(self, final: bool = ..., text: _Optional[str] = ..., timestamp: _Optional[str] = ...) -> None: ...
