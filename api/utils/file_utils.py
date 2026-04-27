"""Procesamiento temporal de imagenes y videos subidos a la API."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from fastapi import HTTPException

from api.core.constants import (
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_FPS,
    IMAGE_MEDIA_TYPE,
    TEMP_IMAGE_SUFFIX,
    TEMP_VIDEO_SUFFIX,
    VIDEO_MEDIA_TYPE,
)
from common.types.media import OutputPathResult
from common.types.model import YoloModel

from api.perception.yolo_inference import draw_results, predict


def process_image(
    yolo_model: YoloModel,
    contents: bytes,
    confidence_threshold: float,
) -> OutputPathResult:
    """Procesa una imagen utilizando el modelo YOLO y devuelve la ruta del archivo temporal con la imagen anotada.

    Args:
        yolo_model (ultralytics.yolo.engine.model.YOLO): El modelo YOLO cargado para realizar las predicciones.
        contents (bytes): Los bytes de la imagen a procesar.
        confidence_threshold (float): El umbral de confianza para filtrar las detecciones del modelo.

    Raises:
        HTTPException: Si no se pudo leer la imagen enviada.
        HTTPException: Si no se pudo generar la imagen resultante.

    Returns:
        tuple[Path, str]: Una tupla que contiene la ruta del archivo temporal con la imagen anotada y el tipo de medio (MIME type) correspondiente a la imagen.
    """
    # Lectura de la imagen desde los bytes recibidos
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer la imagen enviada.",
        )

    # Conversión explícita a uint8 para que sea compatible con el modelo YOLO
    frame = np.asarray(frame, dtype=np.uint8)

    # Realización de las predicciones utilizando el modelo YOLO y anotación de la imagen con los resultados
    results = predict(model=yolo_model, frame=frame, confidence_threshold=confidence_threshold)
    annotated_frame = draw_results(frame=frame, results=results)

    # Creación de un archivo temporal para guardar la imagen anotada
    with NamedTemporaryFile(delete=False, suffix=TEMP_IMAGE_SUFFIX) as output_file:
        output_path = Path(output_file.name)

    # Comprobación de que se pudo guardar la imagen anotada en el archivo temporal
    if not cv2.imwrite(str(output_path), annotated_frame):
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="No se pudo generar la imagen resultante.",
        )

    return output_path, IMAGE_MEDIA_TYPE


def process_video(
    yolo_model: YoloModel,
    contents: bytes,
    confidence_threshold: float,
) -> OutputPathResult:
    """Procesa un video utilizando el modelo YOLO y devuelve la ruta del archivo temporal con el video anotado.

    Args:
        yolo_model (ultralytics.yolo.engine.model.YOLO): El modelo YOLO cargado para realizar las predicciones.
        contents (bytes): Los bytes del video a procesar.
        confidence_threshold (float): El umbral de confianza para filtrar las detecciones del modelo.

    Raises:
        HTTPException: Si no se pudo abrir el video enviado.
        HTTPException: Si no se pudo generar el video resultante.
        HTTPException: Si no se pudieron obtener las dimensiones del video.

    Returns:
        tuple[Path, str]: Una tupla que contiene la ruta del archivo temporal con el video anotado y el tipo de medio (MIME type) correspondiente al video.
    """

    # Creación de archivos temporales para el video de entrada y el video de salida
    with NamedTemporaryFile(delete=False, suffix=TEMP_VIDEO_SUFFIX) as input_file:
        input_file.write(contents)
        input_path = Path(input_file.name)

    # Creación de un archivo temporal para guardar el video anotado
    with NamedTemporaryFile(delete=False, suffix=TEMP_VIDEO_SUFFIX) as output_file:
        output_path = Path(output_file.name)

    # Apertura del video de entrada utilizando OpenCV y comprobación de que se pudo abrir correctamente
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        # Liberación del recurso de captura y eliminación de los archivos temporales antes de lanzar la excepción
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="No se pudo abrir el video enviado.",
        )

    # Obtención de los FPS, ancho y alto del video para configurar el VideoWriter
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = DEFAULT_VIDEO_FPS
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        # Liberación del recurso de captura y eliminación de los archivos temporales antes de lanzar la excepción
        capture.release()
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="No se pudieron obtener las dimensiones del video.",
        )

    # Creación del VideoWriter para escribir el video anotado
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="No se pudo generar el video resultante.",
        )

    try:
        # Bucle de procesamiento de cada frame del video
        while True:
            # Lectura de un frame del video
            success, frame = capture.read()
            if not success:
                break

            # Conversión explícita a uint8 para que sea compatible con el modelo YOLO
            frame = np.asarray(frame, dtype=np.uint8)

            # Realización de las predicciones utilizando el modelo YOLO y anotación del frame con los resultados
            results = predict(model=yolo_model, frame=frame, confidence_threshold=confidence_threshold)
            annotated_frame = draw_results(frame=frame, results=results)

            # Escritura del frame anotado en el video de salida
            writer.write(annotated_frame)

    finally:
        # Liberación de los recursos de captura y escritura, y eliminación del archivo temporal de entrada
        capture.release()
        writer.release()
        input_path.unlink(missing_ok=True)

    return output_path, VIDEO_MEDIA_TYPE
