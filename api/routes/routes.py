"""Rutas HTTP para gestion de streams y prediccion de archivos."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from typing import Annotated, cast, Any
import cv2
import numpy as np
from fastapi import APIRouter, Query, Request, UploadFile, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from starlette.background import BackgroundTask


from api.core.constants import (
    ANNOTATED_FILENAME_PREFIX,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_DOWNLOAD_STEM,
    DEFAULT_MQTT_TOPIC,
    TEMP_IMAGE_SUFFIX,
    IMAGE_MEDIA_TYPE,
    DOCS_PATH,
    HEALTH_OK_MESSAGE,
)
from api.core.types import (
    MQTTConfig,
)
from api.utils.file_utils import process_image, process_zip

from common.types.model import ConfidenceThreshold

# Define las rutas para streaming y prediccion de archivos.
router = APIRouter()


def _require_runtime(request: Request) -> None:
    """Verifica que los componentes de runtime necesarios estén inicializados.

    Args:
        request (Request): petición entrante de FastAPI.

    Raises:
        HTTPException: con código 503 si `stream_manager` o `yolo_model`
            no están disponibles en `request.app.state`.
    """
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
    session_id: str | None = Query(
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
    mqtt_host: str = Query(
        ...,
        alias="mqttHost",
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
    
    overlap: Annotated[tuple[float, float], Query(
        alias="overlap",
        description="Fracción de la máscara a recortar en cada dimensión para eliminar el solape entre detecciones.",
    )] = (0.1, 0.1),

    gsd: float = Query(
        default=0.1,
        alias="gsd",
        description=(
            "Ground Sample Distance (GSD) del video en metros por pixel. "
            "Se utiliza para ajustar el tamaño de los objetos detectados "
            "y mejorar la precision de las predicciones."
        ),
    ),

) -> dict[str, Any]:
    """Inicia un stream RTSP para procesamiento en tiempo real.

    Args:
        request (Request): petición entrante de FastAPI.
        session_id (str | None): identificador del stream; si no se proporciona,
            se genera uno automáticamente.
        rtsp_url (str): URL RTSP del flujo de video.
        save_log (bool): si True, se guardan logs de detecciones.
        save_inference (bool): si True, se guardan imágenes con las inferencias.
        confidence_threshold (float): umbral para filtrar detecciones por confianza.
        mqtt_host (str): dirección IP o nombre del broker MQTT.
        mqtt_port (str): puerto del broker MQTT.
        mqtt_topic (str): topic MQTT para publicar detecciones.
        overlap (tuple[float, float]): tupla (x, y) para recortar solapes en máscaras.
        gsd (float): Ground Sample Distance en metros/píxel para cálculo de áreas.

    Returns:
        dict[str, Any]: detalles del stream iniciado.

    Raises:
        HTTPException: con código 503 si el runtime no está inicializado.
    """

    # Verifica que el runtime de la aplicacion este disponible antes de iniciar el stream.
    _require_runtime(request=request)

    # Valida la URL RTSP.
    if not rtsp_url.startswith("rtsp://"):
        raise HTTPException(
            status_code=400,
            detail="La URL RTSP debe comenzar con 'rtsp://'."
        )

    if session_id is None:
        # Genera un identificador unico para el stream si no se proporciono uno.
        session_id = f"session_{len(request.app.state.stream_manager.sessions) + 1}"

    # Creacion de los tipos de datos
    mqtt_config = MQTTConfig(
        client_id=f"{session_id}-mqtt-client",
        host=mqtt_host,
        port=int(mqtt_port),
        topic=mqtt_topic,
        keepalive=60,
    )

    # Delega el alta del stream en el gestor central de procesos.
    return request.app.state.stream_manager.start(
        session_id=session_id,
        rtsp_url=rtsp_url,
        save_log=save_log,
        save_inference=save_inference,
        confidence_threshold=confidence_threshold,
        mqtt_config=mqtt_config,
        overlap = overlap,
        gsd=gsd,
    )


@router.post("/stream/stop")
def stop_stream(
    request: Request,
    session_id: str | None = Query(
        default=None,
        alias="sessionId",
        description=(
            "Si se indica, detiene solo esa session. "
            "Si no, detiene todas las activas."
        ),
    ),
) -> dict[str, Any]:
    """Detiene un stream activo o todos si no se indica `session_id`.

    Args:
        request (Request): petición entrante de FastAPI.
        session_id (str | None): si se proporciona, detiene solo ese stream;
            si no, detiene todas las sesiones activas.

    Returns:
        dict[str, Any]: respuesta del gestor.

    Raises:
        HTTPException: con código 503 si el runtime no está inicializado.
    """

    # Verifica que el runtime de la aplicacion este disponible antes de detener el stream.
    _require_runtime(request=request)

    return request.app.state.stream_manager.stop(session_id=session_id)



@router.post("/analysis/file")
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
    overlap: Annotated[tuple[float, float], Query(
        alias="overlap",
        description="Fracción de la máscara a recortar en cada dimensión para eliminar el solape entre detecciones.",
    )] = (0.1, 0.1),
    gsd: Annotated[float, Query(
        alias="gsd",
        description="Ground Sample Distance en metros por pixel para calcular el area real.",
    )] = 0.1,
) -> object:
    """Procesa un archivo subido (imagen o ZIP) y devuelve el resultado del análisis.

    Args:
            request (Request): petición entrante de FastAPI.
            file (UploadFile): archivo subido por el usuario (imagen o ZIP).
            confidence_threshold (float): umbral de confianza para las detecciones.
            overlap (tuple[float, float]): recorte de solape en máscaras.
            gsd (float): Ground Sample Distance en metros/píxel.

    Returns:
            object: resultado del análisis (estructura JSON definida por el pipeline).

    Raises:
            HTTPException: con código 415 si el tipo de archivo no es soportado.
    """

    # Verifica que el runtime de la aplicacion este disponible antes de procesar el archivo.
    _require_runtime(request=request)

    # Lee el contenido completo para derivarlo al pipeline adecuado.
    contents = await file.read()
    content_type = file.content_type
    await file.close()

    # Obtencion del modelo YOLO
    yolo_model = request.app.state.yolo_model

    if not content_type or not content_type.startswith(("image/", "application/zip")):
        raise HTTPException(
            status_code=415,
            detail=(
                "Tipo de archivo no soportado. "
                "Solo se aceptan contenidos image/*, o application/zip."
            ),
        )
    elif content_type.startswith("image/"):
        # Procesa imagenes y genera una salida JPEG anotada.
        anlysis_result = process_image(
            yolo_model=yolo_model,
            contents=contents,
            confidence_threshold=confidence_threshold,
        )

    elif content_type.startswith("application/zip"):
        # Procesa archivos ZIP y genera una salida anotada.
        anlysis_result = process_zip(
            yolo_model=yolo_model,
            contents=contents,
            confidence_threshold=confidence_threshold,
        )

    else:
        # Esta rama no deberia ser alcanzable por la validacion previa, pero se deja por seguridad.
        raise HTTPException(
            status_code=415,
            detail="Tipo de archivo no soportado.",
        )

    # Devolvemos el resultado del analisis como JSON.
    return anlysis_result



@router.post("/test/detection")
async def test_detection(
    request: Request,
    file: UploadFile,
    confidence_threshold: Annotated[
        ConfidenceThreshold,
        Query(
            alias="confidenceThreshold",
            description="Umbral de confianza para las detecciones.",
        ),
    ] = DEFAULT_CONFIDENCE_THRESHOLD,
    overlap: Annotated[tuple[float, float], Query(
        alias="overlap",
        description="Fracción de la máscara a recortar en cada dimensión para eliminar el solape entre detecciones.",
    )] = (0.1, 0.1),
) -> FileResponse:
    """Procesa una imagen en modo prueba y devuelve un archivo JPEG anotado.

    Args:
        request (Request): petición entrante de FastAPI.
        file (UploadFile): imagen subida por el usuario.
        confidence_threshold (float): umbral de confianza para las detecciones.
        overlap (tuple[float, float]): recorte de solape en máscaras.

    Returns:
        FileResponse: respuesta con la imagen JPEG anotada.

    Raises:
        HTTPException: con código 415 si el tipo de archivo no es soportado.
        HTTPException: con código 500 si no se puede generar o escribir la imagen.
    """

    # Verifica que el runtime de la aplicacion este disponible antes de procesar el archivo.
    _require_runtime(request=request)

    # Lee el contenido completo para derivarlo al pipeline adecuado.
    contents = await file.read()
    content_type = file.content_type
    await file.close()

    # Obtencion del modelo YOLO
    yolo_model = request.app.state.yolo_model

    if content_type and content_type.startswith(("image/")):
        # Procesa imagenes y genera una salida JPEG anotada.
        annotated_frame= process_image(
            yolo_model=yolo_model,
            contents=contents,
            confidence_threshold=confidence_threshold,
            overlap=overlap,
            gsd=0.1,
            test_mode=True,  # Activamos el modo de prueba para devolver la imagen anotada sin calcular métricas
        )

        # Creación de un archivo temporal para guardar la imagen anotada
        with NamedTemporaryFile(delete=False, suffix=TEMP_IMAGE_SUFFIX) as output_file:
            output_path = Path(output_file.name)

        # Verificamos que la función devolvió realmente una imagen antes de escribirla
        if not isinstance(annotated_frame, np.ndarray):
            raise HTTPException(
                status_code=500,
                detail="El procesamiento no devolvió una imagen anotada.",
            )

        # Codificamos la imagen a JPEG y la escribimos en el archivo temporal
        # Aseguramos dtype uint8 para compatibilidad con OpenCV
        annotated_frame = annotated_frame.astype(np.uint8)
        success, encoded = cv2.imencode('.jpg', annotated_frame)
        if not success:
            output_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="No se pudo generar la imagen resultante.",
            )

        with open(output_path, 'wb') as f:
            f.write(encoded.tobytes())

        download_name = (
            f"{ANNOTATED_FILENAME_PREFIX}"
            f"{Path(file.filename or DEFAULT_IMAGE_DOWNLOAD_STEM).stem}.jpg"
        )

    else:
        # Esta rama no deberia ser alcanzable por la validacion previa, pero se deja por seguridad.
        raise HTTPException(
            status_code=415,
            detail="Tipo de archivo no soportado.",
        )

    return FileResponse(
        path=output_path,
        media_type=IMAGE_MEDIA_TYPE,
        filename=download_name,
        background=BackgroundTask(output_path.unlink, missing_ok=True),
    )




@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Comprueba salud de la API y devuelve estado de los streams activos.

    Args:
        request (Request): petición entrante de FastAPI.

    Returns:
        dict[str, Any]: diccionario con claves `msg` y `streams`.

    Raises:
        HTTPException: con código 503 si el runtime no está inicializado.
    """
    _require_runtime(request=request)
    # Expone tanto el estado general como el detalle de los streams activos.
    return {
        "msg": HEALTH_OK_MESSAGE,
        "streams": request.app.state.stream_manager.health(),
    }


@router.get("/")
async def root() -> RedirectResponse:
    """Redirige a la ruta de documentación interactiva de la API.

    Returns:
        RedirectResponse: redirección a la documentación (Swagger/Redoc).
    """
    # Centraliza la entrada a la documentacion interactiva.
    return RedirectResponse(DOCS_PATH)
