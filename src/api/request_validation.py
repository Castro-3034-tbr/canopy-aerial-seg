"""Validaciones compartidas para las rutas HTTP."""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import HTTPException, Request

from src.core.constants import (
    MAX_CONFIDENCE_THRESHOLD,
    MAX_TCP_PORT,
    MAX_UPLOAD_SIZE_BYTES,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_TCP_PORT,
)
from src.core.types import UploadKind


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


def _validate_confidence_threshold(confidence_threshold: float) -> None:
    """Valida el rango admitido para el umbral de confianza.
    Args:
        confidence_threshold (float): Valor del umbral de confianza a validar.
    Raises:
        HTTPException: Si el umbral de confianza esta fuera del rango permitido.
    """
    if not MIN_CONFIDENCE_THRESHOLD <= confidence_threshold <= MAX_CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=(
                "El umbral de confianza debe estar entre "
                f"{MIN_CONFIDENCE_THRESHOLD:.1f} y {MAX_CONFIDENCE_THRESHOLD:.1f}."
            ),
        )


def _validate_mqtt_port(mqtt_port: int) -> None:
    """Valida que el puerto MQTT este dentro del rango TCP valido.
    Args:
        mqtt_port (int): Numero de puerto a validar.
    Raises:
        HTTPException: Si el puerto MQTT no es un numero entero entre 1 y 655
    """
    if not MIN_TCP_PORT <= mqtt_port <= MAX_TCP_PORT:
        raise HTTPException(
            status_code=400,
            detail=(
                "El puerto MQTT debe ser un numero entero entre "
                f"{MIN_TCP_PORT} y {MAX_TCP_PORT}."
            ),
        )


def _validate_rtsp_url(rtsp_url: str) -> None:
    """Valida de forma basica el formato de la URL RTSP.
    Args:
        rtsp_url (str): URL RTSP a validar.
    Raises:
        HTTPException: Si la URL no tiene un formato valido de RTSP.
    """
    parsed_url = urlparse(rtsp_url)
    if parsed_url.scheme.lower() != "rtsp" or not parsed_url.netloc:
        raise HTTPException(
            status_code=400,
            detail=(
                "La URL RTSP no es valida. Se esperaba un valor con formato "
                "'rtsp://host/recurso'."
            ),
        )


def _detect_upload_kind(content_type: str | None) -> UploadKind:
    """Clasifica el tipo de archivo subido a partir del content type.
    Args:
        content_type (str | None): Valor del content type del archivo subido.
    Returns:
        str: "image" si es una imagen, "video" si es un video.
    Raises:
        HTTPException: Si el content type no es valido o no corresponde a una imagen o video.
    """
    if not content_type:
        raise HTTPException(
            status_code=400,
            detail="El archivo debe incluir un content type valido.",
        )

    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"

    raise HTTPException(
        status_code=400,
        detail="Tipo de archivo no soportado. Se esperaba una imagen o un video valido.",
    )


def _validate_upload_contents(contents: bytes) -> None:
    """Valida tamano y contenido minimo del archivo subido."""
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="El archivo enviado esta vacio.",
        )

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "El archivo supera el tamano maximo permitido de "
                f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
            ),
        )
