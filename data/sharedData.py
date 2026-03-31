import queue
from typing import Any


class SharedData:
    """
    Clase que encapsula los datos compartidos entre los hilos de lectura y procesamiento del stream RTSP,
    para evitar problemas de concurrencia y facilitar la comunicación entre ellos.
    """

    def __init__(self, max_queueSize: int = 30):
        """Inicializa la instancia de SharedData con una cola de frames y un lock para sincronización.

        Args:
            max_queueSize (int): Tamaño máximo de la cola de frames. Defaults to 30.
        """
        self._frame_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queueSize)

    def putFrame(self, package: dict[str, Any]) -> None:
        """Almacena un frame en la cola.
        Si la cola está llena, elimina el frame más antiguo antes de agregar el nuevo para evitar bloqueos.
        Args:
            package (dict): Diccionario que contiene el frame y sus metadatos.
        """
        try:
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put_nowait(package)
        except Exception as e:
            raise Exception(f"Error al almacenar frame en sharedData: {e}")

    def getFrame(self, timeout: int = 1) -> dict[str, Any] | None:
        """Obtiene un frame de la cola.
        Args:
            timeout (int): Tiempo máximo en segundos para esperar un frame antes de devolver None. Defaults to 1.
        Returns:
        dict | None: Diccionario con el frame y sus metadatos, o None si no se pudo obtener un frame en el tiempo especificado.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def isQueueEmpty(self) -> bool:
        """Comprueba si la cola está vacía.
        Returns:
            bool: True si la cola está vacía, False en caso contrario.
        """
        return self._frame_queue.empty()

    def queueSize(self) -> int:
        """Obtiene el tamaño actual de la cola.
        Returns:
            int: Tamaño de la cola.
        """
        return self._frame_queue.qsize()
