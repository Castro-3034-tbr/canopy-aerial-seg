import queue
import threading


class SharedData:
    """
    Clase que encapsula los datos compartidos entre los hilos de lectura y procesamiento del stream RTSP,
    para evitar problemas de concurrencia y facilitar la comunicación entre ellos.
    """

    def __init__(self, max_queue_size=30):
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.queue_lock = threading.Lock()

    def put_frame(self, package):
        """Almacena un frame en la cola."""
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(package)
        except Exception as e:
            raise Exception(f"Error al almacenar frame en sharedData: {e}")

    def get_frame(self, timeout=1):
        """Obtiene un frame de la cola."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_queue_empty(self):
        """Comprueba si la cola está vacía."""
        return self.frame_queue.empty()

    def queue_size(self):
        """Obtiene el tamaño actual de la cola."""
        return self.frame_queue.qsize()
