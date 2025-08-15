from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DialogRef(_message.Message):
    __slots__ = ("dialog_id", "turn")
    DIALOG_ID_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    dialog_id: str
    turn: int
    def __init__(self, dialog_id: _Optional[str] = ..., turn: _Optional[int] = ...) -> None: ...

class Timestamp(_message.Message):
    __slots__ = ("iso8601",)
    ISO8601_FIELD_NUMBER: _ClassVar[int]
    iso8601: str
    def __init__(self, iso8601: _Optional[str] = ...) -> None: ...

class ServiceStatus(_message.Message):
    __slots__ = ("service_name", "healthy", "status_message", "last_check")
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LAST_CHECK_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    healthy: bool
    status_message: str
    last_check: Timestamp
    def __init__(self, service_name: _Optional[str] = ..., healthy: bool = ..., status_message: _Optional[str] = ..., last_check: _Optional[_Union[Timestamp, _Mapping]] = ...) -> None: ...
