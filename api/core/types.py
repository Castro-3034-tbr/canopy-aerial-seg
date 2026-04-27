"""Tipos propios de la API y compatibilidad con los tipos compartidos."""

from __future__ import annotations

from multiprocessing import Process
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypeAlias, runtime_checkable

import ultralytics
from pydantic import Field, field_validator
from typing_extensions import TypedDict

from common.types.base import StrictModel
from common.types.media import FramePackage, Imagen
from common.types.model import InferenceDetection


TcpPortValue: TypeAlias = Annotated[int, Field(ge=1, le=65535)]
UploadKind: TypeAlias = Literal["image", "video"]
StreamState: TypeAlias = Literal["running", "stopped", "stopping"]


class ApiConfig(StrictModel):
    """Configuración validable del servidor HTTP."""

    IP: str = Field(min_length=1)
    PORT: int = Field(ge=1, le=65535)

    @field_validator("IP")
    @classmethod
    def validate_ip(cls, value: str) -> str:
        """Evita valores vacíos para la IP de la API."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("La IP de la API no puede estar vacía.")
        if normalized.lower() != "localhost" and normalized.count(".") != 3:
            raise ValueError(
                "La IP de la API debe ser 'localhost' o una dirección IPv4."
            )
        return normalized


class ApiSavePathConfig(StrictModel):
    """Rutas de salida configurables del proyecto para la API."""

    Logs: Path
    Inference: Path


class ModelConfig(StrictModel):
    """Configuración validable del modelo YOLO para inferencia online."""

    Name: str = Field(min_length=1)
    Path: Path
    Device: str = Field(min_length=1)

    @field_validator("Name", "Device")
    @classmethod
    def validate_model_field(cls, value: str) -> str:
        """Evita campos vacíos en la configuración del modelo."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Los campos del modelo no pueden estar vacíos.")
        return normalized

    @field_validator("Path")
    @classmethod
    def validate_model_path(cls, value: Path) -> Path:
        """Asegura que la ruta del modelo exista y no sea vacía."""
        if not str(value).strip():
            raise ValueError("La ruta del modelo no puede estar vacía.")
        return value


class RtspURL(StrictModel):
    """Representación validable de una URL RTSP."""

    url: str

    @field_validator("url")
    @classmethod
    def validate_rtsp_url(cls, value: str) -> str:
        """Valida que la URL tenga formato RTSP."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("La URL RTSP no puede estar vacía.")
        if not normalized.lower().startswith("rtsp://"):
            raise ValueError("La URL debe comenzar con 'rtsp://'.")
        if normalized.count(":") < 2:
            raise ValueError("La URL RTSP debe contener IP y puerto.")
        return normalized


class AppConfig(StrictModel):
    """Configuración raíz validable de la API."""

    API: ApiConfig
    SavePath: ApiSavePathConfig
    Model: ModelConfig


class MQTTConfig(TypedDict):
    """Configuración MQTT asociada a una sesión."""

    broker: str
    port: int
    topic: str


class StreamSession(TypedDict):
    """Estado interno asociado a un stream activo."""

    stream_id: str
    state: StreamState
    rtsp_url: str
    shared_data: "SharedData"
    project_data: "ProjectData"
    reader_process: Process
    processor_process: Process
    mqtt: MQTTConfig


class StreamStartedResponse(TypedDict):
    """Respuesta del endpoint de arranque de stream."""

    msg: str
    stream_id: str
    state: StreamState
    rtsp_url: str
    mqtt: MQTTConfig


class StreamStoppedResponse(TypedDict):
    """Respuesta del endpoint de parada de un stream."""

    msg: str
    stream_id: str
    state: StreamState


class StopAllStreamsResponse(TypedDict):
    """Respuesta del endpoint de parada de todos los streams."""

    msg: str
    stopped: list[StreamStoppedResponse]


class StreamHealth(TypedDict):
    """Resumen del estado de un stream."""

    stream_id: str
    state: StreamState
    rtsp_url: str
    reader_alive: bool
    processor_alive: bool


class StreamsHealthResponse(TypedDict):
    """Estado agregado del gestor de streams."""

    active_streams: int
    streams: list[StreamHealth]


class HealthResponse(TypedDict):
    """Respuesta del healthcheck de la API."""

    msg: str
    stream: StreamsHealthResponse


@runtime_checkable
class EventLike(Protocol):
    """Interfaz mínima de un evento compartido."""

    def set(self) -> None: ...

    def clear(self) -> None: ...

    def is_set(self) -> bool: ...


@runtime_checkable
class FrameQueueLike(Protocol):
    """Interfaz mínima de la cola compartida de frames."""

    def put(self, obj: FramePackage, timeout: float | None = None) -> None: ...

    def put_nowait(self, obj: FramePackage) -> None: ...

    def get(self, timeout: float | None = None) -> FramePackage: ...

    def get_nowait(self) -> FramePackage: ...


@runtime_checkable
class SharedData(Protocol):
    """Namespace compartido entre procesos lector y procesador."""

    frame_queue: FrameQueueLike


@runtime_checkable
class ProjectData(Protocol):
    """Namespace compartido con configuración y control de un stream."""

    reader_process_running: EventLike
    processor_process_running: EventLike
    save_path_logs: str
    save_path_inference: str
    yolo_path: str
    yolo_device: str
    stream_id: str


@runtime_checkable
class RuntimeState(Protocol):
    """Estado global compartido entre la API y el gestor de streams."""

    running: bool
    active_streams: int


@runtime_checkable
class GlobalManager(Protocol):
    """Métodos del manager usados por la aplicación."""

    def Namespace(self) -> Any: ...

    def Queue(self, maxsize: int = 0) -> FrameQueueLike: ...

    def Event(self) -> EventLike: ...

    def shutdown(self) -> None: ...


class AppRuntime(TypedDict):
    """Objetos compartidos durante la vida de la aplicación."""

    config: AppConfig
    manager: "GlobalManager"
    runtime_state: "RuntimeState"
    yolo_model: "ultralytics.YOLO"
    stream_manager: Any


@runtime_checkable
class PahoMQTTClient(Protocol):
    """Superficie mínima usada del cliente `paho-mqtt`."""

    on_connect: Any
    on_disconnect: Any
    broker: str
    port: int
    topic: str
    keepalive: int

    def connect_async(self, host: str, port: int, keepalive: int = 60) -> Any: ...

    def loop_start(self) -> Any: ...

    def publish(self, topic: str, payload: str) -> Any: ...

    def loop_stop(self) -> Any: ...

    def disconnect(self) -> Any: ...
