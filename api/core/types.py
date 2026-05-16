"""
Tipos y contratos de la API.

Estructura:
- Tipos base y aliases
- Configuración (Pydantic)
- Estado de ejecución
- Protocolos (IPC / multiprocessing)
- Runtime global
"""

from __future__ import annotations

from multiprocessing import Process
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import Field, field_validator
from typing_extensions import TypedDict

from common.types.base import StrictModel
from common.types.media import FramePackage
from common.types.model import YoloModel


# ==========================================================
# TYPE ALIASES
# ==========================================================

StreamState: TypeAlias = Literal["running", "stopped", "stopping"]
IPv4: TypeAlias = str


# ==========================================================
# CONFIGURACIÓN (Pydantic)
# ==========================================================

class ApiConfig(StrictModel):
    """
    Configuración del servidor HTTP.

    Attributes:
        ip: Dirección IP o 'localhost'
        port: Puerto TCP válido
    """

    ip: IPv4 = Field(min_length=1)
    port: int = Field(ge=1, le=65535)

    @field_validator("ip")
    @classmethod
    def validate_ip(cls, value: str) -> str:
        """
        Valida formato básico de IP o localhost.

        Nota:
            No realiza validación completa RFC.
        """
        v = value.strip()

        if not v:
            raise ValueError("La IP no puede estar vacía.")

        if v.lower() != "localhost" and v.count(".") != 3:
            raise ValueError("Debe ser 'localhost' o una IPv4 válida.")

        return v


class SavePathConfig(StrictModel):
    """
    Rutas de persistencia del sistema.
    """

    logs: Path
    inference: Path


class ModelConfig(StrictModel):
    """
    Configuración del modelo de inferencia.

    Attributes:
        name: Identificador lógico del modelo
        path: Ruta al modelo
        device: Dispositivo (cpu, cuda, etc.)
    """

    name: str = Field(min_length=1)
    path: Path
    device: str = Field(min_length=1)

    @field_validator("name", "device")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("Campo vacío no permitido.")
        return v

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        if not str(value).strip():
            raise ValueError("Ruta inválida.")
        return value


class MQTTConfig(StrictModel):
    """
    Configuración de conexión MQTT.
    """

    client_id: str = Field(min_length=1)
    host: str
    port: int = Field(ge=1, le=65535)
    topic: str = Field(min_length=1)
    keepalive: int = Field(default=60, ge=1, le=3600)


class AppConfig(StrictModel):
    """
    Configuración global de la aplicación.
    """

    api: ApiConfig
    save_path: SavePathConfig
    model: ModelConfig


# ==========================================================
# ESTADO DE EJECUCIÓN
# ==========================================================

class StreamSession(StrictModel):
    """
    Estado asociado a un stream activo.

    Nota:
        Contiene referencias a procesos y datos compartidos.
    """

    state: StreamState
    rtsp_url: str

    shared_data: "SharedData"
    project_data: "ProjectData"

    reader_process: Process
    processor_process: Process

    mqtt: MQTTConfig


# ==========================================================
# PROTOCOLOS (IPC / MULTIPROCESSING)
# ==========================================================

@runtime_checkable
class EventLike(Protocol):
    """Interfaz mínima compatible con multiprocessing.Event."""

    def set(self) -> None: ...
    def clear(self) -> None: ...
    def is_set(self) -> bool: ...


@runtime_checkable
class FrameQueueLike(Protocol):
    """Cola tipada para transporte de frames."""

    def put(self, obj: FramePackage, timeout: float | None = None) -> None: ...
    def put_nowait(self, obj: FramePackage) -> None: ...
    def get(self, timeout: float | None = None) -> FramePackage: ...
    def get_nowait(self) -> FramePackage: ...


@runtime_checkable
class SharedData(Protocol):
    """
    Datos compartidos entre procesos.

    Diseño mínimo para evitar acoplamiento.
    """

    frame_queue: FrameQueueLike


@runtime_checkable
class ProjectData(Protocol):
    """
    Estado compartido de un stream.

    Incluye:
    - Flags de control
    - Configuración efectiva
    """

    reader_running: EventLike
    processor_running: EventLike

    save_path_logs: str
    save_path_inference: str

    yolo_path: str
    yolo_device: str

    session_id: str


@runtime_checkable
class RuntimeState(Protocol):
    """
    Estado global mutable de la aplicación.
    """

    running: bool
    active_streams: int


@runtime_checkable
class GlobalManager(Protocol):
    """
    Abstracción del multiprocessing.Manager.
    """

    def Namespace(self) -> Any: ...
    def Queue(self, maxsize: int = 0) -> FrameQueueLike: ...
    def Event(self) -> EventLike: ...
    def shutdown(self) -> None: ...


@runtime_checkable
class PahoMQTTClient(Protocol):
    """
    Interfaz mínima usada del cliente paho-mqtt.
    """

    def connect_async(self, host: str, port: int, keepalive: int = 60) -> Any: ...
    def loop_start(self) -> Any: ...
    def publish(self, topic: str, payload: str) -> Any: ...
    def loop_stop(self) -> Any: ...
    def disconnect(self) -> Any: ...


# ==========================================================
# RUNTIME GLOBAL
# ==========================================================

class AppRuntime(TypedDict):
    """
    Contenedor global de dependencias en ejecución.

    Uso:
        Inyección de dependencias en endpoints/servicios.
    """

    config: AppConfig
    manager: GlobalManager
    runtime_state: RuntimeState
    yolo_model: YoloModel
    stream_manager: Any