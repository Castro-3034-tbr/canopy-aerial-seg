from __future__ import annotations

from multiprocessing import Manager

from fastapi import FastAPI

from src.api.routes import router
from src.core.config import load_config
from src.core.constants import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from src.core.data_init import init_runtime_state
from src.perception.yolo_inference import YoloInference
from src.processes.stream_manager import StreamManager
from src.utils.logger import configure_logging


def build_runtime() -> dict:
    """Construye el estado compartido y los componentes necesarios para la aplicacion, 
    incluyendo la configuración, el modelo de inferencia y el gestor de streams."""
    
    #Configuracion del sistema de logging
    configure_logging()
    
    #Carga de la configuración de la aplicación desde un archivo JSON
    config = load_config()
    
    #Inicializacion del Manager de multriprocesos y del estado compartido para la aplicación
    manager = Manager()
    
    #Inicializacion del estado compartido para la aplicación
    runtime_state = init_runtime_state(manager)
    
    #Obtencion de las configuraciones específicas
    save_path = config.get("SavePath", {})
    model_config = config.get("Model", {})
    
    #Inicializacion del modelo de inferencia YOLO
    yolo_model = YoloInference(
        model_config.get("Path"),
        model_config.get("Device", "cpu"),
    )
    
    #Inicializacion del gestor de streams, encargado de gestionar los flujos de video y su procesamiento
    stream_manager = StreamManager(
        manager=manager,
        model_config=model_config,
        save_path_config=save_path,
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


def create_application() -> FastAPI:
    """Crea la instancia de la aplicación FastAPI, construyendo las dependencias necesarias y registrando las rutas del API."""
    
    #Creacion de la intancia de la aplcicaion FastAPI
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
    )
    #Construccion de las dependencias
    runtime = build_runtime()
    
    #Registro del estado compartido y los componentes necesarios en el estado de la aplicación para su uso en los endpoints
    for key, value in runtime.items():
        setattr(app.state, key, value)

    #Registro de las rutas del API en la aplicación
    app.include_router(router)
    return app
