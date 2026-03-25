import threading


class ProjectData:
    """
    Clase que encapsula la información del proyecto, como el nombre del proyecto, la descripción, el autor, etc.
    """

    def __init__(self):

        self.readerThreadRunning = (
            False  # Variable para controlar el estado del hilo de lectura
        )
        self.processorThreadRunning = (
            False  # Variable para controlar el estado del hilo de procesamiento
        )

        self.threadStateLock = threading.Lock()

    def setReaderThreadRunning(self, running: bool):
        """Establece el estado del hilo de lectura."""
        with self.threadStateLock:
            self.readerThreadRunning = running

    def setProcessorThreadRunning(self, running: bool):
        """Establece el estado del hilo de procesamiento."""
        with self.threadStateLock:
            self.processorThreadRunning = running

    def getReaderThreadRunning(self) -> bool:
        """Obtiene el estado del hilo de lectura."""
        with self.threadStateLock:
            return self.readerThreadRunning

    def getProcessorThreadRunning(self) -> bool:
        """Obtiene el estado del hilo de procesamiento."""
        with self.threadStateLock:
            return self.processorThreadRunning
