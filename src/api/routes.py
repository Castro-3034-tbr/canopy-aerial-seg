"""Rutas HTTP para gestion de streams y prediccion de archivos."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from starlette.background import BackgroundTask

from src.core.constants import (
    ANNOTATED_FILENAME_PREFIX,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_DOWNLOAD_STEM,
    DEFAULT_MQTT_TOPIC,
    DEFAULT_VIDEO_DOWNLOAD_STEM,
    DOCS_PATH,
    HEALTH_OK_MESSAGE
)

from src.api.request_validation import (
    _require_runtime,
    _validate_confidence_threshold,
    _validate_mqtt_port,
    _validate_rtsp_url,
    _validate_upload_contents,
    _detect_upload_kind
)

from src.utils.file_utils import process_image, process_video

# Define las rutas para streaming y prediccion de archivos.
router = APIRouter()


@router.post("/stream/start")
def start_stream(
    request: Request,
    stream_id: str | None = Query(
        default=None,
        alias="streamId",
        description=(
            "Identificador unico del stream. "
            "Si no se envia, se genera automaticamente."
        ),
    ),
    rtsp_url: str = Query(
        ...,
        alias="rtspUrl",
        description="URL RTSP del flujo de video.",
    ),
    save_log: bool = Query(default=False, alias="saveLog"),
    save_inference: bool = Query(default=False, alias="saveInference"),
    confidence_threshold: float = Query(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        alias="confidenceThreshold",
        description="Umbral de confianza para filtrar detecciones.",
    ),
    mqtt_broker: str = Query(
        ...,
        alias="mqttBroker",
        description="Direccion IP o nombre del broker MQTT.",
    ),
    mqtt_port: int = Query(
        ...,
        alias="mqttPort",
        description="Puerto del broker MQTT.",
    ),
    mqtt_topic: str = Query(
        default=DEFAULT_MQTT_TOPIC,
        alias="mqttTopic",
        description="Topic MQTT para publicar detecciones.",
    ),
) -> dict:
    """Inicia un nuevo stream RTSP para procesamiento en tiempo real."""
    _require_runtime(request)
    _validate_confidence_threshold(confidence_threshold)
    _validate_rtsp_url(rtsp_url)
    _validate_mqtt_port(mqtt_port)

    # Delega el alta del stream en el gestor central de procesos.
    return request.app.state.stream_manager.start(
        stream_id=stream_id,
        rtsp_url=rtsp_url,
        save_log=save_log,
        save_inference=save_inference,
        confidence_threshold=confidence_threshold,
        mqtt_broker=mqtt_broker,
        mqtt_port=mqtt_port,
        mqtt_topic=mqtt_topic,
    )


@router.post("/stream/stop")
def stop_stream(
    request: Request,
    stream_id: str | None = Query(
        default=None,
        alias="streamId",
        description=(
            "Si se indica, detiene solo ese stream. "
            "Si no, detiene todos los activos."
        ),
    ),
) -> dict:
    """Detiene un stream RTSP en ejecucion."""
    _require_runtime(request)

    # Si no llega identificador, el gestor detendra todos los streams.
    return request.app.state.stream_manager.stop(stream_id=stream_id)


@router.post("/predict/file")
async def predict_file(
    request: Request,
    file: UploadFile = File(
        ...,
        alias="file",
        description="Archivo de imagen o video a procesar.",
    ),
    confidence_threshold: float = Query(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        alias="confidenceThreshold",
        description="Umbral de confianza para las detecciones.",
    ),
) -> FileResponse:
    """Procesa una imagen o un video y devuelve el resultado anotado."""
    _require_runtime(request)
    _validate_confidence_threshold(confidence_threshold)

    # Lee el contenido completo para derivarlo al pipeline adecuado.
    contents = await file.read()
    await file.close()
    upload_kind = _detect_upload_kind(file.content_type)
    _validate_upload_contents(contents)

    #Obtencion del modelo YOLO
    yolo_model = request.app.state.yolo_model

    if upload_kind == "image":
        # Procesa imagenes y genera una salida JPEG anotada.
        output_path, media_type = process_image(
            yolo_model,
            contents,
            confidence_threshold,
        )
        download_name = (
            f"{ANNOTATED_FILENAME_PREFIX}"
            f"{Path(file.filename or DEFAULT_IMAGE_DOWNLOAD_STEM).stem}.jpg"
        )
    else:
        # Procesa videos y conserva una salida MP4 anotada.
        output_path, media_type = process_video(
            yolo_model,
            contents,
            confidence_threshold,
        )
        download_name = (
            f"{ANNOTATED_FILENAME_PREFIX}"
            f"{Path(file.filename or DEFAULT_VIDEO_DOWNLOAD_STEM).stem}.mp4"
        )

    return FileResponse(
        path=output_path,
        media_type=media_type,
        filename=download_name,
        background=BackgroundTask(output_path.unlink, missing_ok=True),
    )


@router.get("/health")
def health(request: Request) -> dict:
    """Verifica el estado de la API y del gestor de streams."""
    _require_runtime(request)
    # Expone tanto el estado general como el detalle de los streams activos.
    return {
        "msg": HEALTH_OK_MESSAGE,
        "stream": request.app.state.stream_manager.health(),
    }


@router.get("/")
async def root() -> RedirectResponse:
    """Redirige a la documentacion de la API."""
    # Centraliza la entrada a la documentacion interactiva.
    return RedirectResponse(DOCS_PATH)
