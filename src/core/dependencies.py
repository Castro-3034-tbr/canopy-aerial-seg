"""Construccion del runtime y la aplicacion FastAPI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from multiprocessing import Manager

from fastapi import FastAPI

from src.api.routes import router
from src.core.config import load_config
from src.core.constants import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from src.core.data_init import init_runtime_state
from src.perception.yolo_inference import YoloInference
from src.processes.stream_manager import StreamManager
from core.logger import configure_logging


def build_runtime() -> dict:
    """Construye el estado compartido y los componentes principales."""
    # Configura el logging antes de inicializar el resto del runtime.
    configure_logging()
    # Carga la configuracion persistida del proyecto.
    config = load_config()
    # Crea el gestor de objetos compartidos para los procesos hijos.
    manager = Manager()
    runtime_state = init_runtime_state(manager)

    # Extrae los bloques de configuracion que usan el modelo y las salidas.
    save_path_config = config.get("SavePath", {})
    model_config = config.get("Model", {})

    # Construye el modelo una sola vez para reutilizarlo en la aplicacion.
    yolo_model = YoloInference(
        model_config.get("Path"),
        model_config.get("Device", "cpu"),
    )
    # Crea el gestor responsable de arrancar y detener streams.
    stream_manager = StreamManager(
        manager=manager,
        model_config=model_config,
        save_path_config=save_path_config,
        runtime_state=runtime_state,
        yolo_model=yolo_model,
    )

    return {
        "config": config,
        "manager": manager,
        "runtime_state": runtime_state,
        "yolo_model": yolo_model,
        "stream_manager": stream_manager,
    }


def shutdown_runtime(app: FastAPI) -> None:
    """Libera los recursos construidos durante el startup."""
    stream_manager = getattr(app.state, "stream_manager", None)
    if stream_manager is not None:
        stream_manager.stop()

    manager = getattr(app.state, "manager", None)
    if manager is not None:
        manager.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa y libera el runtime pesado durante el ciclo de vida."""
    runtime = build_runtime()

    # Expone el runtime en app.state para reutilizarlo desde las rutas.
    for key, value in runtime.items():
        setattr(app.state, key, value)

    try:
        yield
    finally:
        shutdown_runtime(app)


def create_app() -> FastAPI:
    """Crea la aplicacion FastAPI y registra las dependencias."""
    # Inicializa la aplicacion con los metadatos visibles en OpenAPI.
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        lifespan=lifespan,
    )

    # Registra el router principal de la API.
    app.include_router(router)
    return app
