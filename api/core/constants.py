# trunk-ignore-all(isort)
# trunk-ignore-all(black)
"""Constantes compartidas usadas por la aplicacion."""

from __future__ import annotations

# Configuracion general de la aplicacion.
APP_TITLE = "TFM API"                              # Titulo visible de la aplicacion FastAPI.
APP_DESCRIPTION = "API para detección de objetos con YOLOv8"  # Descripcion general de la API.
APP_VERSION = "1.0.0"                             # Version publicada de la aplicacion.
DOCS_PATH = "/docs"                               # Ruta de la documentacion interactiva.


# Configuracion de API, colas y control de procesos.
DEFAULT_QUEUE_SIZE = 30                           # Tamano maximo de la cola compartida de frames.
PROCESS_JOIN_TIMEOUT = 5.0                        # Tiempo maximo de espera al detener procesos.
FRAME_QUEUE_TIMEOUT_SECONDS = 1                   # Tiempo maximo de espera al leer un frame de la cola.
FRAME_QUEUE_PUT_TIMEOUT_SECONDS = 1               # Tiempo maximo de espera al insertar un frame en la cola.
RECONNECT_DELAY_SECONDS = 2                       # Pausa antes de reintentar la conexion RTSP.


# Configuracion del modelo de inferencia.
DEFAULT_CONFIDENCE_THRESHOLD = 0.60               # Umbral de confianza por defecto para filtrar detecciones.
DEFAULT_MASK_THRESHOLD = 0.5                      # Umbral usado para interpretar las mascaras.

# Configuracion MQTT.
DEFAULT_MQTT_TOPIC = "detecciones"                # Topic por defecto para publicar resultados.
DEFAULT_MQTT_KEEPALIVE = 60                       # Intervalo de keepalive del cliente MQTT.
PROCESSOR_MQTT_CLIENT_ID = "processor_process"    # Identificador MQTT del proceso de inferencia

# Configuracion de archivos temporales y descargas.
DEFAULT_IMAGE_DOWNLOAD_STEM = "image"             # Nombre base por defecto para descargas de imagen.
DEFAULT_VIDEO_DOWNLOAD_STEM = "video"             # Nombre base por defecto para descargas de video.
ANNOTATED_FILENAME_PREFIX = "annotated_"          # Prefijo aplicado a salidas anotadas.
TEMP_IMAGE_SUFFIX = ".jpg"                        # Extension usada para imagenes temporales.
TEMP_VIDEO_SUFFIX = ".mp4"                        # Extension usada para videos temporales.
IMAGE_MEDIA_TYPE = "image/jpeg"                   # Tipo MIME de las respuestas de imagen.
VIDEO_MEDIA_TYPE = "video/mp4"                    # Tipo MIME de las respuestas de video.
DEFAULT_VIDEO_FPS = 25.0                          # FPS por defecto al reconstruir videos.
DEFAULT_VIDEO_CODEC = "mp4v"                      # Codec por defecto del video de salida.


# Configuracion de persistencia de resultados.
FRAME_FILENAME_PREFIX = "frame_"                  # Prefijo de los frames anotados guardados.
FRAME_FILENAME_SUFFIX = ".jpg"                    # Extension de los frames anotados guardados.
DETECTIONS_LOG_PREFIX = "detections_log_"         # Prefijo de los ficheros CSV de detecciones.
DETECTIONS_LOG_SUFFIX = ".csv"                    # Extension de los logs tabulares de detecciones.
DETECTION_LOG_COLUMNS = [
    "timestamp",
    "frame_id",
    "class",
    "confidence",
    "bbox",
    "mask",
]                                               # Columnas del CSV de detecciones.


# Configuracion de conexion RTSP.
RTSP_OPTIONS = {
    "rtsp_transport": "tcp",
    "fflags": "nobuffer",
    "stimeout": "5000000",
}                                               # Opciones de PyAV para mejorar estabilidad y latencia del stream.


# Configuracion de anotacion visual.
MASK_COLOR = (0, 255, 0)                        # Color BGR de las mascaras dibujadas.
CENTROID_COLOR = (0, 0, 255)                    # Color BGR del centroide.
CENTROID_RADIUS = 5                             # Radio del punto dibujado para el centroide.
MASK_OVERLAY_ALPHA = 0.5                        # Peso visual aplicado a la mascara superpuesta.
MASK_OVERLAY_BETA = 1.0                         # Peso visual aplicado al frame original.
MASK_OVERLAY_GAMMA = 0                          # Ajuste gamma usado en la fusion de imagenes.


# Mensajes de respuesta HTTP.
HEALTH_OK_MESSAGE = "API esta funcionando correctamente"  # Mensaje devuelto por el healthcheck.
STREAM_START_SUCCESS_MESSAGE = "Inicio del stream realizado correctamente"  # Mensaje de arranque correcto.
STREAM_STOP_SUCCESS_MESSAGE = "Detención del stream RTSP realizada correctamente"  # Mensaje de parada de un stream.
STREAMS_STOP_SUCCESS_MESSAGE = "Detención de streams realizada correctamente"  # Mensaje de parada global.
