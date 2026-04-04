"""Tipos compartidos del proyecto."""

from __future__ import annotations

from multiprocessing import Process
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator
from typing_extensions import TypedDict


class StrictModel(BaseModel):
    """Modelo base de Pydantic que rechaza campos no declarados."""

    model_config = ConfigDict(extra="forbid")


class Coordinates(StrictModel):
    """Coordenadas enteras de un punto 2D."""

    x: int
    y: int


class BoundingBox(StrictModel):
    """Coordenadas de una caja delimitadora."""

    x1: float
    y1: float
    x2: float
    y2: float


class OutputFile(StrictModel):
    """Metadatos de un archivo de salida generado por la API."""

    path: FilePath
    media_type: str


class ModelConfigModel(StrictModel):
    """Configuracion validable del modelo YOLO."""

    Name: str = Field(min_length=1)
    Path: FilePath
    Device: str = Field(min_length=1)

    @field_validator("Name", "Device")
    @classmethod
    def validate_model_field(cls, value: str) -> str:
        """Evita campos vacios en la configuracion del modelo."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Los campos del modelo no pueden estar vacios.")
        return normalized

    @field_validator("Path")
    @classmethod
    def validate_model_path(cls, value: Path) -> Path:
        """Asegura que la ruta del modelo exista y no sea vacia."""
        if not str(value).strip():
            raise ValueError("La ruta del modelo no puede estar vacia.")
        return value


class RtspURLModel(StrictModel):
    """Representacion validable de una URL RTSP."""

    url: str

    @field_validator("url")
    @classmethod
    def validate_rtsp_url(cls, value: str) -> str:
        """Valida que la URL tenga formato RTSP."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("La URL RTSP no puede estar vacia.")
        if not normalized.lower().startswith("rtsp://"):
            raise ValueError("La URL debe comenzar con 'rtsp://'.")
        if normalized.count(":") < 2:
            raise ValueError("La URL RTSP debe contener IP y puerto.")
        return normalized


class ApiConfigModel(StrictModel):
    """Configuracion validable del servidor HTTP."""

    IP: str = Field(min_length=1)
    PORT: int = Field(ge=1, le=65535)

    @field_validator("IP")
    @classmethod
    def validate_ip(cls, value: str) -> str:
        """Evita valores vacios para la IP de la API."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("La IP de la API no puede estar vacia.")
        if normalized.lower() != "localhost" and not normalized.count(".") == 3:
            raise ValueError("La IP de la API debe ser 'localhost' o una direccion IPv4.")
        return normalized


class SavePathConfigModel(StrictModel):
    """Configuracion validable de rutas de salida."""

    Logs: Path
    Inference: Path

    @field_validator("Logs", "Inference")
    @classmethod
    def validate_output_path(cls, value: Path) -> Path:
        """Evita rutas vacias en la configuracion."""
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Las rutas de salida no pueden estar vacias.")
        return value


class AppConfigModel(StrictModel):
    """Configuracion raiz validable del proyecto."""

    API: ApiConfigModel
    SavePath: SavePathConfigModel
    Model: ModelConfigModel


FrameArray: TypeAlias = NDArray[np.uint8]
MaskArray: TypeAlias = NDArray[np.floating[Any] | np.uint8 | np.bool_]
VerticesList: TypeAlias = list[Coordinates]
BoundingBoxList: TypeAlias = list[BoundingBox]
CentroidList: TypeAlias = list[Coordinates]
UploadKind: TypeAlias = Literal["image", "video"]
StreamState: TypeAlias = Literal["running", "stopped", "stopping"]


class FramePackage(TypedDict):
    """Frame compartido entre el lector y el procesador."""

    img: FrameArray
    frame_id: int
    pts: int | None
    width: int
    height: int


class Detection(TypedDict):
    """Deteccion serializable publicada y persistida por el sistema."""

    class_id: int
    confidence: float
    bbox: BoundingBox
    mask: list[Any]
    vertices: VerticesList
    centroid: CentroidList


class MaskMetric(TypedDict):
    """Metricas resumidas de una mascara."""

    mask_index: int
    mask_area: int
    frame_area: int
    area_ratio: float


DetectionBatch: TypeAlias = list[Detection]


class MQTTConfig(TypedDict):
    """Configuracion MQTT asociada a una sesion."""

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


class AppRuntime(TypedDict):
    """Objetos compartidos durante la vida de la aplicacion."""

    config: AppConfigModel
    manager: "GlobalManager"
    runtime_state: "RuntimeState"
    yolo_model: "YoloModel"
    stream_manager: Any


@runtime_checkable
class EventLike(Protocol):
    """Interfaz minima de un evento compartido."""

    def set(self) -> None: ...

    def clear(self) -> None: ...

    def is_set(self) -> bool: ...


@runtime_checkable
class FrameQueueLike(Protocol):
    """Interfaz minima de la cola compartida de frames."""

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
    """Namespace compartido con configuracion y control de un stream."""

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
    """Metodos del manager usados por la aplicacion."""

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
    ) -> Any: ...

    def extract_detections(self, results: Any) -> DetectionBatch: ...

    def draw_results(self, frame: FrameArray, results: Any) -> FrameArray: ...


OutputPathResult: TypeAlias = tuple[Path, str]
