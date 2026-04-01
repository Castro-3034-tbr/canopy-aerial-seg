import json
import logging
from multiprocessing import Manager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from starlette.background import BackgroundTask

from services.streamManager import StreamManager

# Importar funciones auxiliares
from utils.utils_aux import processImage, processVideo
from utils.utils_yolo import ClassYOLO

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
    config = {}
    with open(PATH_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)
        if config is None or not isinstance(config, dict):
            logging.error("Formato de configuración inválido: se esperaba un objeto JSON")
            raise ValueError("Formato de configuración inválido: se esperaba un objeto JSON")
    logging.info(f"Archivo de configuración cargado correctamente: {PATH_CONFIG}")
except json.JSONDecodeError as e:
    logging.error(f"Error al parsear el archivo de configuración: {e}")
    raise ValueError(f"Error al parsear el archivo de configuración: {e}") from e


# Inicializacion de la aplicación FastAPI
app = FastAPI(
    title="TFM API", description="API for YOLOv8 Object Detection", version="1.0.0"
)

# Creación de manager para datos compartidos entre procesos
manager = Manager()

savePath = config.get("SavePath", {})
if len(savePath) == 0:
    logging.warning("Sección 'SavePath' no encontrada en la configuración")
yoloConfig = config.get("Model", {})
if len(yoloConfig) == 0:
    logging.warning("Sección 'Model' no encontrada en la configuración")

# Creacion de la clase YOLO
yoloModel = ClassYOLO(yoloConfig.get("Path"), yoloConfig.get("Device", "cpu"))
stream_manager = StreamManager(manager=manager, model_config=yoloConfig, save_path_config=savePath)

@app.post("/stream/start")
def start_stream(
    streamId: str | None = Query(
        None,
        description="Identificador único del stream. Si no se envía, se genera automáticamente.",
    ),
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

    return stream_manager.start(
        stream_id=streamId,
        rtsp_url=rtspUrl,
        save_log=saveLog,
        save_inference=saveInference,
        confidence_class=confidenceClass,
        mqtt_broker=mqttBroker,
        mqtt_port=mqttPort,
        mqtt_topic=mqttTopic,
    )


@app.post("/stream/stop")
def stop_stream(
    streamId: str | None = Query(
        None,
        description="Si se indica, detiene solo ese stream. Si no, detiene todos.",
    )
):
    """Detiene uno o todos los streams RTSP activos."""
    return stream_manager.stop(stream_id=streamId)


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
    return {"msg": "API esta funcionando correctamente", "stream": stream_manager.health()}


@app.get("/")
async def root():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    # Validacion de la IP y el puerto del broker MQTT
    APIconfig = config.get("API", {})
    APIIp = APIconfig.get("IP")
    APIPort = APIconfig.get("PORT")
    if not APIIp or not APIPort:
        logging.error(
            "IP o puerto del broker MQTT no especificados en la configuración"
        )
        raise ValueError(
            "IP o puerto del broker MQTT no especificados en la configuración"
        )

    uvicorn.run(app, host=APIIp, port=APIPort)
