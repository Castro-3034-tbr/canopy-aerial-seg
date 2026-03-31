import logging
import threading
from pathlib import Path
import json


import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from starlette.background import BackgroundTask

from core.procesorThread import processorThread
from core.readerThread import readerThread
from data.projectData import ProjectData
from data.sharedData import SharedData

from utils.utils_mqtt import MQTTClient
from utils.utils_yolo import ClassYOLO

# Importar funciones auxiliares
from utils.utils_aux import processImage, processVideo


# Configuracion del logging para la API guardando en la carpeta logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "api.log"), logging.StreamHandler()],
)


# Carga del archivo de configuración JSON y validación de su existencia y formato
PATH_CONFIG = Path("config/config.json")
if not PATH_CONFIG.is_file():
    logging.error(f"Archivo de configuración no encontrado: {PATH_CONFIG}")
    raise FileNotFoundError(f"Archivo de configuración no encontrado: {PATH_CONFIG}")

try:
    with open(PATH_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)
    logging.info(f"Archivo de configuración cargado correctamente: {PATH_CONFIG}")
except json.JSONDecodeError as e:
    logging.error(f"Error al parsear el archivo de configuración: {e}")


# Inicializacion de la aplicación FastAPI
app = FastAPI(
    title="TFM API", description="API for YOLOv8 Object Detection", version="1.0.0"
)


# Creacion de clases para almacenar datos compartidos entre hilos y datos del proyecto
sharedData = SharedData()

SaveData = config.get("SaveData", {})
projectData = ProjectData(SaveData.get("Logs", "logs/"), SaveData.get("Inference", "inference/"))

#Creacion de la clase YOLO
yoloConfig = config.get("Model", {})
yoloModel = ClassYOLO(yoloConfig.get("Path"), yoloConfig.get("Device", "cpu"))


@app.post("/stream/start")
def start_stream(
    rtspUrl: str = Query(..., description="RTSP URL of the video stream"),
    saveLog: bool = Query(False),
    saveInference: bool = Query(False),
    confidenceClass: float = 0.60,
    mqttBroker: str = Query(..., description="MQTT broker IP address"),
    mqttPort: int = Query(..., description="MQTT broker port"),
    mqttTopic: str = Query(
        "detecciones", description="MQTT topic for publishing detections"
    ),
):
    """Inicia el procesamiento de un stream RTSP con las siguientes opciones:

    Args:
        rtspUrl (str): URL del stream RTSP a procesar
        saveLog (bool): Indica si se deben guardar los logs de detección en un archivo de texto. Default: False
        saveInference (bool): Indica si se deben guardar las inferencias (clases y coordenadas) en un archivo JSON. Default: False
        confidenceClass (float): Umbral de confianza para la clase. Default: 0.60
        mqttBroker (str): Dirección IP del broker MQTT al que se publicarán las detecciones
        mqttPort (int): Puerto del broker MQTT al que se publicarán las detecciones
        mqttTopic (str): Topic MQTT en el que se publicarán las detecciones. Default: "detecciones"

    Returns:
            dict: Un diccionario con un mensaje de inicio y la configuración utilizada para el stream
    """

    #Creacion de la instancia del cliente MQTT para publicar las detecciones
    mqttClient = MQTTClient(
        clientID="TFM_API_Client",
        broker=mqttBroker,
        port=mqttPort,
        topic=mqttTopic
    )

    # Inicializacion de los dos hilos para lectura y procesamiento del stream RTSP (a implementar en la función real)

    hiloReader = threading.Thread(
        target=readerThread,
        args=(sharedData, projectData, rtspUrl),
        name="HiloReader",
        daemon=True,
    )
    projectData.setReaderThreadRunning(True)
    hiloReader.start()

    hiloProcessor = threading.Thread(
        target=processorThread,
        args=(
            sharedData,
            projectData,
            saveLog,
            saveInference,
            confidenceClass,
            mqttClient,
            yoloModel
        ),
        name="HiloProcessor",
        daemon=True,
    )
    projectData.setProcessorThreadRunning(True)
    hiloProcessor.start()

    return {
        "msg": "Inicio del stream {}",
        "rtspUrl": rtspUrl,
        "mqtt": {"broker": mqttBroker, "port": mqttPort, "topic": mqttTopic},
    }


@app.post("/stream/stop")
def stop_stream():
    """Detiene el procesamiento del stream RTSP."""

    # Lógica para detener los hilos de lectura y procesamiento del stream RTSP (a implementar en la función real)
    projectData.setReaderThreadRunning(False)
    projectData.setProcessorThreadRunning(False)

    return {"msg": "Detención del stream RTSP realizada correctamente"}


@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    confidenceClass0: float = 0.6,
):
    """

    Args:
        file (UploadFile): Archivo de imagen a procesar
        confidenceClass0 (float): Umbral de confianza para la clase 0. Defaults to 0.6.

    Returns:
            FileResponse: Descarga directa del archivo anotado, tanto para imagen como para video.
    """

    # Leer el contenido del archivo subido
    contents = await file.read()

    if file.content_type.startswith("image/"):
        output_path, media_type = processImage(yoloModel, contents, confidenceClass0)
        download_name = f"annotated_{Path(file.filename or 'image').stem}.jpg"
    elif file.content_type.startswith("video/"):
        output_path, media_type = processVideo(yoloModel, contents, confidenceClass0)
        download_name = f"annotated_{Path(file.filename or 'video').stem}.mp4"
    else:
        raise HTTPException(
            status_code=400,
            detail="Tipo de archivo no soportado. Se esperaba una imagen o un video válido.",
        )

    await file.close()

    return FileResponse(
        path=output_path,
        media_type=media_type,
        filename=download_name,
        background=BackgroundTask(output_path.unlink, missing_ok=True),
    )


@app.get("/health")
def health():
    """
        Endpoint de salud para verificar que la API está funcionando correctamente.

    Returns:
        dict: Un diccionario con el estado de la API.
    """
    return {"msg": "API esta funcionando correctamente"}


@app.get("/")
async def root():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    #Validacion de la IP y el puerto del broker MQTT
    APIconfig = config.get("API", {})
    APIIp = APIconfig.get("IP")
    APIPort = APIconfig.get("PORT")
    if not APIIp or not APIPort:
        logging.error("IP o puerto del broker MQTT no especificados en la configuración")
        raise ValueError("IP o puerto del broker MQTT no especificados en la configuración")

    uvicorn.run(app, host=APIIp, port=APIPort)
