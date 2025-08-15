from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class LogEntry(_message.Message):
    __slots__ = ("timestamp", "level", "service", "event", "details")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    timestamp: str
    level: str
    service: str
    event: str
    details: str
    def __init__(self, timestamp: _Optional[str] = ..., level: _Optional[str] = ..., service: _Optional[str] = ..., event: _Optional[str] = ..., details: _Optional[str] = ...) -> None: ...

class DialogEntry(_message.Message):
    __slots__ = ("dialog_id", "speaker", "text", "timestamp")
    DIALOG_ID_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    dialog_id: str
    speaker: str
    text: str
    timestamp: str
    def __init__(self, dialog_id: _Optional[str] = ..., speaker: _Optional[str] = ..., text: _Optional[str] = ..., timestamp: _Optional[str] = ...) -> None: ...

class NewDialogResponse(_message.Message):
    __slots__ = ("dialog_id", "log_file_path")
    DIALOG_ID_FIELD_NUMBER: _ClassVar[int]
    LOG_FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    dialog_id: str
    log_file_path: str
    def __init__(self, dialog_id: _Optional[str] = ..., log_file_path: _Optional[str] = ...) -> None: ...
