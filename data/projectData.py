
from types import SimpleNamespace

def initProjectData(manager, yoloPath: str, yoloDevice: str, savePathLogs: str, savePathInference: str) -> SimpleNamespace:
    """Inicializa la clase ProjectData con las rutas de guardado de logs e inferencias.
    Args:
        manager: Instancia del Manager para crear un Namespace compartido entre procesos.
        yoloPath (str): Ruta del modelo YOLO.
        yoloDevice (str): Dispositivo para ejecutar el modelo YOLO.
        savePathLogs (str): Ruta de guardado de logs.
        savePathInference (str): Ruta de guardado de inferencias.
    Returns:
        Manager.Namespace: Instancia del namespace compartido entre procesos.
    """
    #Creacion del namespace para almacenar los datos del proyecto
    projectData = manager.Namespace()

    #Inicializacion de las variables del namespace
    projectData.readerProcessRunning = False              # Flag para controlar la ejecución de los hilos de lectura y procesamiento
    projectData.processorProcessRunning = False              # Flag para controlar la ejecución del hilo de procesamiento

    projectData.savePathLogs = savePathLogs                 # Ruta de guardado de logs
    projectData.savePathInference = savePathInference       # Ruta de guardado de inferencias

    projectData.yoloPath = yoloPath                         # Ruta del modelo YOLO
    projectData.yoloDevice = yoloDevice                     # Dispositivo para ejecutar el modelo YOLO

    return projectData