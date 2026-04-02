from __future__ import annotations

from pathlib import Path


#---- Configuracion de la aplicación ----#
APP_TITLE = "TFM API"                                                                   # Título de la aplicación FastAPI
APP_DESCRIPTION = "API for YOLOv8 Object Detection"                                     # Descripción de la aplicación FastAPI
APP_VERSION = "1.0.0"                                                                   # Versión de la aplicación FastAPI
DOCS_PATH = "/docs"                                                                     # Ruta para la documentación automática de la API

#---- Configuracion de logging ----#
DEFAULT_CONFIG_PATH = Path("config/config.json")                                        # Ruta por defecto del archivo de configuración
DEFAULT_LOG_DIR = Path("logs")                                                          # Directorio por defecto para los archivos de log
DEFAULT_LOG_FILENAME = "api.log"                                                        # Nombre por defecto del archivo de log
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"                        # Formato por defecto para los mensajes de log
DEFAULT_LOG_LEVEL = "INFO"                                                              # Nivel de log por defecto (DEBUG, INFO, WARNING, ERROR, CRITICAL)

#---- Configuracion de la API y procesamiento ----#
DEFAULT_API_HOST_KEY = "IP"                                                             # Clave para obtener la IP de la API desde la configuración
DEFAULT_API_PORT_KEY = "PORT"                                                           # Clave para obtener el puerto de la API desde la configuración
DEFAULT_QUEUE_SIZE = 30                                                                 # Tamaño máximo de la cola para los frames de video
PROCESS_JOIN_TIMEOUT = 5.0                                                              # Tiempo máximo de espera para que los procesos terminen al detener un stream
FRAME_QUEUE_TIMEOUT_SECONDS = 1                                                         # Tiempo máximo de espera para obtener un frame de la cola antes de continuar con el siguiente ciclo
RECONNECT_DELAY_SECONDS = 2                                                             # Tiempo de espera antes de intentar reconectar a la fuente RTSP en caso de fallo de conexión

#---- Configuracion del modelo de inferencia ----#
DEFAULT_MODEL_PATH = "yolov26n-seg.pt"                                                  # Ruta por defecto del modelo de inferencia YOLO
DEFAULT_MODEL_DEVICE = "cpu"                                                            # Dispositivo por defecto para la inferencia (cpu o cuda)
DEFAULT_CONFIDENCE_THRESHOLD = 0.60                                                     # Umbral de confianza por defecto para filtrar las detecciones del modelo
DEFAULT_MASK_THRESHOLD = 0.5                                                            # Umbral de máscara por defecto para filtrar las detecciones del modelo

#---- Configuracion MQTT ----#
DEFAULT_MQTT_TOPIC = "detecciones"                                                      # Tema por defecto para publicar las detecciones en el broker MQTT
DEFAULT_MQTT_KEEPALIVE = 60                                                             # Tiempo de keepalive por defecto para el cliente MQTT (en segundos)
PROCESSOR_MQTT_CLIENT_ID = "processor_process"                                          # ID del cliente MQTT para el proceso de procesamiento

#---- Configuracion de guardado de resultados ----#
DEFAULT_IMAGE_DOWNLOAD_STEM = "image"                                                   # Prefijo por defecto para los archivos de imagen guardados
DEFAULT_VIDEO_DOWNLOAD_STEM = "video"                                                   # Prefijo por defecto para los archivos de video guardados
ANNOTATED_FILENAME_PREFIX = "annotated_"                                                # Prefijo para los archivos de imagen anotados guardados
TEMP_IMAGE_SUFFIX = ".jpg"                                                              # Sufijo para los archivos de imagen temporales
TEMP_VIDEO_SUFFIX = ".mp4"                                                              # Sufijo para los archivos de video temporales
IMAGE_MEDIA_TYPE = "image/jpeg"                                                         # Tipo de medio (MIME type) para las imágenes JPEG
VIDEO_MEDIA_TYPE = "video/mp4"                                                          # Tipo de medio (MIME type) para los videos MP4
DEFAULT_VIDEO_FPS = 25.0                                                                # FPS por defecto para los videos anotados guardados
DEFAULT_VIDEO_CODEC = "mp4v"                                                            # Codec por defecto para los videos anotados guardados

#---- Configuracion de anotacion visual ----#
FRAME_FILENAME_PREFIX = "frame_"                                                        # Prefijo para los archivos de imagen de frames guardados
FRAME_FILENAME_SUFFIX = ".jpg"                                                          # Sufijo para los archivos de imagen de frames guardados
DETECTIONS_LOG_PREFIX = "detections_log_"                                               # Prefijo para los archivos de log de detecciones guardados
DETECTIONS_LOG_SUFFIX = ".csv"                                                          # Sufijo para los archivos de log de detecciones guardados
DETECTION_LOG_COLUMNS = [                                                               # Columnas para el log de detecciones guardado en formato CSV
    "timestamp",
    "frame_id",
    "class",
    "confidence",
    "bbox",
    "mask",
]

#---- Configuracion de conexión RTSP ----#
RTSP_OPTIONS = {                                                                        # Opciones para la conexión RTSP utilizando PyAV
    "rtsp_transport": "tcp",
    "fflags": "nobuffer",
    "stimeout": "5000000",
}

#---- Configuracion de anotacion visual ----#
MASK_COLOR = (0, 255, 0)                                                                # Color para las máscaras de detección (verde en formato BGR)
CENTROID_COLOR = (0, 0, 255)                                                            # Color para los centroides de detección (rojo en formato BGR)
CENTROID_RADIUS = 5                                                                     # Radio para los centroides de detección
MASK_OVERLAY_ALPHA = 0.5                                                                # Valor de alpha para la superposición de las máscaras de detección sobre la imagen original
MASK_OVERLAY_BETA = 1.0                                                                 # Valor de beta para la superposición de las máscaras de detección sobre la imagen original
MASK_OVERLAY_GAMMA = 0                                                                  # Valor de gamma para la superposición de las máscaras de detección sobre la imagen original

#---- Mensajes de respuesta ----#
HEALTH_OK_MESSAGE = "API esta funcionando correctamente"                                # Mensaje de respuesta para la ruta de salud (health check) cuando la API está funcionando correctamente
STREAM_START_SUCCESS_MESSAGE = "Inicio del stream realizado correctamente"              # Mensaje de respuesta para la ruta de inicio de stream cuando el stream se inicia correctamente
STREAM_STOP_SUCCESS_MESSAGE = "Detención del stream RTSP realizada correctamente"       # Mensaje de respuesta para la ruta de detención de stream cuando el stream RTSP se detiene correctamente
STREAMS_STOP_SUCCESS_MESSAGE = "Detención de streams realizada correctamente"           # Mensaje de respuesta para la ruta de detención de todos los streams cuando todos los streams se detienen correctamente
