from types import SimpleNamespace

def initSharedData(manager, max_queueSize: int = 30) -> SimpleNamespace:
    """Inicializa la clase SharedData para compartir datos entre los hilos de lectura y procesamiento.
    Args:
        manager: Instancia del Manager para crear un Namespace compartido entre procesos.
        max_queueSize (int): Tamaño máximo de la cola de frames. Defaults to 30.
    Returns:
        SharedData: Instancia de SharedData para compartir datos entre hilos.
    """
    #Creacion del namespace para almacenar los datos compartidos entre hilos
    sharedData = manager.Namespace()

    #Inicializacion de una cola de frames compartida entre procesos
    sharedData.frame_queue = manager.Queue(maxsize=max_queueSize)

    return sharedData