import common_pb2 as _common_pb2
import llm_pb2 as _llm_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TtsConfig(_message.Message):
    __slots__ = ("voice", "device", "queue_max_chunks", "chunk_max_ms")
    VOICE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    QUEUE_MAX_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_MAX_MS_FIELD_NUMBER: _ClassVar[int]
    voice: str
    device: str
    queue_max_chunks: int
    chunk_max_ms: int
    def __init__(self, voice: _Optional[str] = ..., device: _Optional[str] = ..., queue_max_chunks: _Optional[int] = ..., chunk_max_ms: _Optional[int] = ...) -> None: ...

class ConfigureResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class TextInput(_message.Message):
    __slots__ = ("dialog_id", "turn", "text")
    DIALOG_ID_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    dialog_id: str
    turn: int
    text: str
    def __init__(self, dialog_id: _Optional[str] = ..., turn: _Optional[int] = ..., text: _Optional[str] = ...) -> None: ...

class TtsPlayAck(_message.Message):
    __slots__ = ("started_ts",)
    STARTED_TS_FIELD_NUMBER: _ClassVar[int]
    started_ts: str
    def __init__(self, started_ts: _Optional[str] = ...) -> None: ...

class TtsEvent(_message.Message):
    __slots__ = ("type", "timestamp", "details")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    type: str
    timestamp: str
    details: str
    def __init__(self, type: _Optional[str] = ..., timestamp: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...
