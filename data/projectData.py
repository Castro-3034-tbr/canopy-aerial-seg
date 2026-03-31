import threading


class ProjectData:
    """Estado compartido del proyecto y de ejecución de los hilos."""

    def __init__(self, savePathLogs: str, savePathInference: str):
        self._reader_thread_running = False
        self._processor_thread_running = False
        self._save_path_logs = savePathLogs
        self._save_path_inference = savePathInference


    def setReaderThreadRunning(self, running: bool):
        """Establece el estado del hilo de lectura."""
        self._reader_thread_running = running

    def setProcessorThreadRunning(self, running: bool):
        """Establece el estado del hilo de procesamiento."""
        self._processor_thread_running = running

    def getReaderThreadRunning(self) -> bool:
        """Obtiene el estado del hilo de lectura."""
        return self._reader_thread_running

    def getProcessorThreadRunning(self) -> bool:
        """Obtiene el estado del hilo de procesamiento."""
        return self._processor_thread_running

    def getSavePathLogs(self) -> str:
        """Obtiene la ruta de guardado de logs."""
        return self._save_path_logs

    def getSavePathInference(self) -> str:
        """Obtiene la ruta de guardado de inferencias."""
        return self._save_path_inference
