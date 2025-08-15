import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class LlmConfig(_message.Message):
    __slots__ = ("endpoint", "model", "modelfile_path", "request_timeout_ms", "stream_chunk_chars")
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    MODELFILE_PATH_FIELD_NUMBER: _ClassVar[int]
    REQUEST_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    STREAM_CHUNK_CHARS_FIELD_NUMBER: _ClassVar[int]
    endpoint: str
    model: str
    modelfile_path: str
    request_timeout_ms: int
    stream_chunk_chars: int
    def __init__(self, endpoint: _Optional[str] = ..., model: _Optional[str] = ..., modelfile_path: _Optional[str] = ..., request_timeout_ms: _Optional[int] = ..., stream_chunk_chars: _Optional[int] = ...) -> None: ...

class ConfigureResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class UserQuery(_message.Message):
    __slots__ = ("text", "dialog_id", "turn")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    DIALOG_ID_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    text: str
    dialog_id: str
    turn: int
    def __init__(self, text: _Optional[str] = ..., dialog_id: _Optional[str] = ..., turn: _Optional[int] = ...) -> None: ...

class LlmChunk(_message.Message):
    __slots__ = ("text", "eot")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    EOT_FIELD_NUMBER: _ClassVar[int]
    text: str
    eot: bool
    def __init__(self, text: _Optional[str] = ..., eot: bool = ...) -> None: ...
