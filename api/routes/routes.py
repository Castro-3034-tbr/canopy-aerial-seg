"""Rutas HTTP para gestion de streams y prediccion de archivos."""

from __future__ import annotations

from pathlib import Path

from typing import Annotated, cast
from fastapi import APIRouter, Query, Request, UploadFile, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from starlette.background import BackgroundTask


from api.core.constants import (
    ANNOTATED_FILENAME_PREFIX,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_DOWNLOAD_STEM,
    DEFAULT_MQTT_TOPIC,
    DEFAULT_VIDEO_DOWNLOAD_STEM,
    DOCS_PATH,
    HEALTH_OK_MESSAGE,
)
from api.core.types import (
    RtspURL,
    HealthResponse,
    StopAllStreamsResponse,
    StreamStartedResponse,
    StreamStoppedResponse,
)
from api.utils.file_utils import process_image, process_video

from common.types.model import ConfidenceThreshold

# Define las rutas para streaming y prediccion de archivos.
router = APIRouter()


def _require_runtime(request: Request) -> None:
    """Comprueba que el runtime pesado de la aplicacion este disponible."""
    if (
        not hasattr(request.app.state, "stream_manager")
        or not hasattr(request.app.state, "yolo_model")
        or request.app.state.yolo_model is None
    ):
        raise HTTPException(
            status_code=503,
            detail="La aplicacion aun no ha inicializado su runtime.",
        )


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
    rtsp_url: str = cast(str, Query(
        ...,
        alias="rtspUrl",
        description="URL RTSP del flujo de video.",
    )),
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
    mqtt_port: str = Query(
        ...,
        alias="mqttPort",
        description="Puerto del broker MQTT.",
    ),
    mqtt_topic: str = Query(
        default=DEFAULT_MQTT_TOPIC,
        alias="mqttTopic",
        description="Topic MQTT para publicar detecciones.",
    ),
    
) -> StreamStartedResponse:
    """Inicia un nuevo stream RTSP para procesamiento en tiempo real."""
    _require_runtime(request=request)
    validated_rtsp_url = RtspURL(url=rtsp_url)

    # TODO: Añidir parametos de vuelo

    # Delega el alta del stream en el gestor central de procesos.
    return request.app.state.stream_manager.start(
        stream_id=stream_id,
        rtsp_url=validated_rtsp_url.url,
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
) -> StreamStoppedResponse | StopAllStreamsResponse:
    """Detiene un stream RTSP en ejecucion."""
    _require_runtime(request=request)

    # Si no llega identificador, el gestor detendra todos los streams.
    return request.app.state.stream_manager.stop(stream_id=stream_id)

@router.post("/analysis/folder")
def analyze_folder(
    request: Request,
    folder_path: str = cast(str, Query(
        ...,
        alias="folderPath",
        description="Ruta del directorio a analizar.",
    )),
    confidence_threshold: float = Query(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        alias="confidenceThreshold",
        description="Umbral de confianza para filtrar detecciones.",
    ),
) -> dict[str, str]:
    """Analiza un directorio completo de imagenes o videos y devuelve un resumen."""
    _require_runtime(request=request)

@router.post("/predict/file")
async def predict_file(
    request: Request,
    file: UploadFile,
    confidence_threshold: Annotated[
        ConfidenceThreshold,
        Query(
            alias="confidenceThreshold",
            description="Umbral de confianza para las detecciones.",
        ),
    ] = DEFAULT_CONFIDENCE_THRESHOLD,
) -> FileResponse:
    """Procesa una imagen o un video y devuelve el resultado anotado."""
    _require_runtime(request=request)

    # Lee el contenido completo para derivarlo al pipeline adecuado.
    contents = await file.read()
    content_type = file.content_type
    await file.close()

    # Obtencion del modelo YOLO
    yolo_model = request.app.state.yolo_model

    if not content_type or not content_type.startswith(("image/", "video/")):
        raise HTTPException(
            status_code=415,
            detail=(
                "Tipo de archivo no soportado. "
                "Solo se aceptan contenidos image/* o video/*."
            ),
        )
    elif content_type.startswith("image/"):
        # Procesa imagenes y genera una salida JPEG anotada.
        output_path, media_type = process_image(
            yolo_model=yolo_model,
            contents=contents,
            confidence_threshold=confidence_threshold,
        )
        download_name = (
            f"{ANNOTATED_FILENAME_PREFIX}"
            f"{Path(file.filename or DEFAULT_IMAGE_DOWNLOAD_STEM).stem}.jpg"
        )
    else:
        # Procesa videos y conserva una salida MP4 anotada.
        output_path, media_type = process_video(
            yolo_model=yolo_model,
            contents=contents,
            confidence_threshold=confidence_threshold,
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
def health(request: Request) -> HealthResponse:
    """Verifica el estado de la API y del gestor de streams."""
    _require_runtime(request=request)
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
