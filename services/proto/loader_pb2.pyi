import common_pb2 as _common_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class StartServiceRequest(_message.Message):
    __slots__ = ("service_name",)
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    def __init__(self, service_name: _Optional[str] = ...) -> None: ...

class StartServiceResponse(_message.Message):
    __slots__ = ("success", "message", "pid")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    pid: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., pid: _Optional[int] = ...) -> None: ...

class StopServiceRequest(_message.Message):
    __slots__ = ("service_name",)
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    def __init__(self, service_name: _Optional[str] = ...) -> None: ...

class StopServiceResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class GetPidsResponse(_message.Message):
    __slots__ = ("service_pids",)
    class ServicePidsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    SERVICE_PIDS_FIELD_NUMBER: _ClassVar[int]
    service_pids: _containers.ScalarMap[str, int]
    def __init__(self, service_pids: _Optional[_Mapping[str, int]] = ...) -> None: ...

class WakeWordEvent(_message.Message):
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

class DialogCompleteEvent(_message.Message):
    __slots__ = ("dialog_id", "completion_reason")
    DIALOG_ID_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_REASON_FIELD_NUMBER: _ClassVar[int]
    dialog_id: str
    completion_reason: str
    def __init__(self, dialog_id: _Optional[str] = ..., completion_reason: _Optional[str] = ...) -> None: ...
