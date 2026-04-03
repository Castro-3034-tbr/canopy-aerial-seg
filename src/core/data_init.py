"""Inicializacion de estados y estructuras compartidas."""

from __future__ import annotations

from src.core.constants import DEFAULT_QUEUE_SIZE


def init_shared_data(manager, max_queue_size: int = DEFAULT_QUEUE_SIZE):
    """Inicializa los datos compartidos entre procesos.

    Args:
        manager (Manager): Gestor de multiprocessing.
        max_queue_size (int, optional): Tamano maximo de la cola de frames.

    Returns:
        Namespace: Espacio compartido con la cola de frames.
    """
    # Agrupa los recursos compartidos entre lector y procesador.
    shared_data = manager.Namespace()

    # Limita la cola para evitar un crecimiento descontrolado en memoria.
    shared_data.frame_queue = manager.Queue(maxsize=max_queue_size)
    return shared_data


def init_project_data(
    manager,
    yolo_path: str,
    yolo_device: str,
    save_path_logs: str,
    save_path_inference: str,
):
    """Inicializa los datos compartidos especificos del proyecto.

    Args:
        manager (Manager): Gestor de multiprocessing.
        yolo_path (str): Ruta del modelo YOLO.
        yolo_device (str): Dispositivo usado para inferencia.
        save_path_logs (str): Ruta de salida para logs.
        save_path_inference (str): Ruta de salida para inferencias.

    Returns:
        Namespace: Espacio compartido con la configuracion del proyecto.
    """
    # Crea el Namespace para los datos de configuracion y estado del proyecto.
    project_data = manager.Namespace()

    # Inicializa las señales de control para los procesos de lectura y
    # procesamiento mediante eventos compartidos.
    project_data.reader_process_running = manager.Event()
    project_data.processor_process_running = manager.Event()
    project_data.save_path_logs = save_path_logs
    project_data.save_path_inference = save_path_inference
    project_data.yolo_path = yolo_path
    project_data.yolo_device = yolo_device
    return project_data


def init_runtime_state(manager):
    """Inicializa el estado compartido a nivel de API."""
    # Crea un Namespace para el estado global de la aplicacion.
    runtime_state = manager.Namespace()

    # Estado de ejecucion global, usado para controlar el ciclo de vida de la aplicacion.
    runtime_state.running = True
    runtime_state.active_streams = 0
    return runtime_state
