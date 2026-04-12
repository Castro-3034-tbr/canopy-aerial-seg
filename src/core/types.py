"""Tipos compartidos del proyecto."""

from __future__ import annotations

from multiprocessing import Process
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TypeAlias, runtime_checkable

import ultralytics
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator
from typing_extensions import TypedDict
from ultralytics.engine.model import Model as UltralyticsModelBase


class StrictModel(BaseModel):
    """Modelo base de Pydantic que rechaza campos no declarados."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class Coordinates(StrictModel):
    """Coordenadas normalizadas de un punto en el rango [0, 1]."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


Polygon: TypeAlias = list[Coordinates]
Mask: TypeAlias = list[Coordinates]
FrameArray: TypeAlias = NDArray[np.uint8]
FrameMask: TypeAlias = NDArray[np.floating[Any] | np.uint8 | np.bool_]
UltralyticsModel: TypeAlias = UltralyticsModelBase
OutputPathResult: TypeAlias = tuple[Path, str]

ConfidenceThreshold: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
TcpPortValue: TypeAlias = Annotated[int, Field(ge=1, le=65535)]
BoundingBoxList: TypeAlias = list["BoundingBox"]
CentroidList: TypeAlias = list[Coordinates]
UploadKind: TypeAlias = Literal["image", "video"]
StreamState: TypeAlias = Literal["running", "stopped", "stopping"]


class BoundingBox(StrictModel):
    """Caja envolvente derivada de un polígono en coordenadas normalizadas."""

    p1: Coordinates
    p2: Coordinates
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)

    @field_validator("width", "height")
    @classmethod
    def validar_dimension(cls, value: float) -> float:
        """Evita dimensiones negativas por redondeo."""
        return max(0.0, value)


class Detection(StrictModel):
    """Estructura de una detección serializable."""

    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    mask: Mask
    frame_mask: FrameMask
    centroid: Coordinates


class MaskMetric(StrictModel):
    """Métricas resumidas de una máscara."""

    mask_index: int
    mask_area: int
    frame_area: int
    area_ratio: float


class FramePackage(StrictModel):
    """Frame compartido entre el lector y el procesador."""

    img: FrameArray
    frame_id: int
    pts: int | None
    width: int
    height: int


class OutputFile(StrictModel):
    """Metadatos de un archivo de salida generado por la API."""

    path: FilePath
    media_type: str


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


class SavePathConfigModel(StrictModel):
    """Rutas de salida configurables del proyecto."""

    Logs: Path
    Inference: Path


class ModelConfig(StrictModel):
    """Configuración validable del modelo YOLO."""

    Name: str = Field(min_length=1)
    Path: FilePath
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


ModelConfigModel: TypeAlias = ModelConfig


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


class AppConfigModel(StrictModel):
    """Configuración raíz validable del proyecto."""

    API: ApiConfig
    SavePath: SavePathConfigModel
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


@runtime_checkable
class YoloModel(Protocol):
    """Interfaz requerida del wrapper de inferencia."""

    def predict(
        self,
        frame: FrameArray,
        confidence_threshold: float = 0.0,
        debug: bool = False,
    ) -> list[Detection]: ...

    def extract_detections(self, results: Any) -> list[Detection]: ...

    def draw_results(self, frame: FrameArray, results: list[Detection]) -> FrameArray: ...


class AppRuntime(TypedDict):
    """Objetos compartidos durante la vida de la aplicación."""

    config: AppConfigModel
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
