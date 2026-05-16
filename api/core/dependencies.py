"""Construcción del runtime y la aplicación FastAPI.

Este módulo crea el estado compartido (manager, runtime_state), carga la
configuración, inicializa el modelo YOLO y construye el `StreamManager`.
También expone el contexto de vida (`lifespan`) y helpers para arrancar y
parar el proceso `mediamtx` usado para retransmisión.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from multiprocessing import Manager
import subprocess
from pathlib import Path
from typing import AsyncIterator, cast

from fastapi import FastAPI

from api.core.config import load_api_config
from api.core.constants import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from api.core.data_init import init_runtime_state
from api.core.types import AppRuntime, GlobalManager
from api.perception.yolo_inference import initialize_model
from api.processes.stream_manager import StreamManager
from api.routes.routes import router
from common.logger import configure_logging
from common.constants import CONFIG_DIR


logger = logging.getLogger(__name__)

def _start_mediamtx() -> subprocess.Popen[str]:
    """Lanza Mediamtx como proceso hijo de la API.

    Returns:
        subprocess.Popen[str]: Proceso lanzado.

    Raises:
        FileNotFoundError: si faltan binario o fichero de configuración.
    """
    mediamtx_dir = Path(CONFIG_DIR).parent / "mediamtx"
    mediamtx_bin = mediamtx_dir / "mediamtx"
    mediamtx_config = mediamtx_dir / "mediamtx.yml"

    if not mediamtx_bin.exists():
        raise FileNotFoundError(f"No se encontró el binario de Mediamtx: {mediamtx_bin}")
    if not mediamtx_config.exists():
        raise FileNotFoundError(f"No se encontró la configuración de Mediamtx: {mediamtx_config}")

    logger.info("Iniciando Mediamtx desde %s", mediamtx_bin)
    return subprocess.Popen(
        [str(mediamtx_bin), str(mediamtx_config)],
        cwd=mediamtx_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _stop_mediamtx(mediamtx_process: subprocess.Popen[str] | None) -> None:
    """Detiene Mediamtx si está en ejecución.

    Args:
        mediamtx_process (subprocess.Popen[str] | None): proceso retornado por `_start_mediamtx`.
    """
    if mediamtx_process is None:
        return

    if mediamtx_process.poll() is not None:
        return

    logger.info("Deteniendo Mediamtx")
    mediamtx_process.terminate()
    try:
        mediamtx_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("Mediamtx no cerró a tiempo; forzando apagado")
        mediamtx_process.kill()
        mediamtx_process.wait(timeout=5)


def build_runtime() -> AppRuntime:
    """Construye y devuelve el runtime compartido usado por la aplicación.

    El runtime incluye la configuración, el manager de multiprocessing, el
    estado de ejecución, el modelo YOLO cargado y el `StreamManager`.

    Returns:
        AppRuntime: Diccionario con claves `config`, `manager`, `runtime_state`,
            `yolo_model` y `stream_manager`.
    """
    configure_logging()
    config = load_api_config(Path(CONFIG_DIR) / "config_api.json")

    manager = cast(GlobalManager, Manager())
    runtime_state = init_runtime_state(manager=manager)

    save_path_config = config.SavePath
    model_config = config.Model

    yolo_model = initialize_model(model_config.Path, model_config.Device)

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
    """Libera los recursos creados en `build_runtime` y durante el lifespan.

    Args:
        app (FastAPI): Instancia de la aplicación que contiene `app.state`.
    """
    mediamtx_process = getattr(app.state, "mediamtx_process", None)
    _stop_mediamtx(mediamtx_process)

    stream_manager = getattr(app.state, "stream_manager", None)
    if stream_manager is not None:
        stream_manager.stop()

    manager = getattr(app.state, "manager", None)
    if manager is not None:
        manager.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Context manager que inicializa y libera el runtime en el startup/shutdown.

    Durante el startup construye el runtime y arranca `mediamtx`. En shutdown
    detiene `mediamtx` y libera recursos.
    """
    runtime = build_runtime()
    mediamtx_process = _start_mediamtx()

    for key, value in runtime.items():
        setattr(app.state, key, value)
    app.state.mediamtx_process = mediamtx_process

    try:
        yield
    finally:
        shutdown_runtime(app)


def create_app() -> FastAPI:
    """Crea la aplicación FastAPI y registra las dependencias principales.

    Returns:
        FastAPI: Aplicación configurada y lista para arrancar.
    """
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        lifespan=lifespan,
    )

    app.include_router(router)
    return app
