from __future__ import annotations

from src.core.constants import DEFAULT_QUEUE_SIZE


def init_shared_data(manager, max_queue_size: int = DEFAULT_QUEUE_SIZE):
    """Inicializa los datos compartidos entre procesos, incluyendo la cola de frames.

    Args:
        manager (Manager): Instancia del Manager de multiprocessing para crear objetos compartidos entre procesos.
        max_queue_size (int, optional): Tamaño máximo de la cola de frames. Defaults to DEFAULT_QUEUE_SIZE.

    Returns:
        Namespace: Instancia del objeto compartido que contiene la cola de frames.
    """
    # Creación de un Namespace para los datos compartidos y una cola de frames con el tamaño máximo especificado
    shared_data = manager.Namespace()

    # Creación de una cola de frames compartida entre procesos con un tamaño máximo para evitar un consumo excesivo de memoria
    shared_data.frame_queue = manager.Queue(maxsize=max_queue_size)
    return shared_data


def init_project_data(
    manager,
    yolo_path: str,
    yolo_device: str,
    save_path_logs: str,
    save_path_inference: str,
):
    """Inicializa los datos especificos del proyecto

    Args:
        manager (Manager): Instancia del Manager de multiprocessing para crear objetos compartidos entre procesos.
        yolo_path (str): Ruta al modelo YOLO.
        yolo_device (str): Dispositivo para la inferencia (cpu o cuda).
        save_path_logs (str): Ruta para guardar los archivos de log.
        save_path_inference (str): Ruta para guardar los resultados de la inferencia.

    Returns:
        Namespace: Instancia del objeto compartido que contiene los datos del proyecto.
    """
    project_data = manager.Namespace()
    project_data.reader_process_running = False
    project_data.processor_process_running = False
    project_data.save_path_logs = save_path_logs
    project_data.save_path_inference = save_path_inference
    project_data.yolo_path = yolo_path
    project_data.yolo_device = yolo_device
    return project_data

def init_runtime_state(manager):
    """Inicializa el estado de tiempo de ejecución de la aplicación compartido a nivel de API."""
    runtime_state = manager.Namespace()
    runtime_state.running = True
    runtime_state.active_streams = 0
    return runtime_state