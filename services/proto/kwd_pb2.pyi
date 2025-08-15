import common_pb2 as _common_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class KwdConfig(_message.Message):
    __slots__ = ("model_path", "confidence_threshold", "cooldown_ms", "yes_phrases")
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    COOLDOWN_MS_FIELD_NUMBER: _ClassVar[int]
    YES_PHRASES_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    confidence_threshold: float
    cooldown_ms: int
    yes_phrases: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model_path: _Optional[str] = ..., confidence_threshold: _Optional[float] = ..., cooldown_ms: _Optional[int] = ..., yes_phrases: _Optional[_Iterable[str]] = ...) -> None: ...

class ConfigureResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class KwdEvent(_message.Message):
    __slots__ = ("type", "confidence", "timestamp", "dialog_id")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DIALOG_ID_FIELD_NUMBER: _ClassVar[int]
    type: str
    confidence: float
    timestamp: str
    dialog_id: str
    def __init__(self, type: _Optional[str] = ..., confidence: _Optional[float] = ..., timestamp: _Optional[str] = ..., dialog_id: _Optional[str] = ...) -> None: ...
