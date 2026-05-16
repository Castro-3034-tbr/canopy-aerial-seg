"""Procesamiento temporal de imagenes y videos subidos a la API."""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import cv2
import numpy as np
from fastapi import HTTPException

from common.types.media import Imagen
from common.types.model import YoloModel

from api.perception.yolo_inference import draw_results, predict

from api.perception.analisis import analyze_results


def process_image(
    yolo_model: YoloModel,
    contents: bytes,
    confidence_threshold: float,
    overlap: tuple[float, float] = (0.1, 0.1),
    gsd: float = 0.1,
    test_mode: bool = False,
) -> dict | Imagen:
    """Procesa una imagen y devuelve métricas de área o la imagen anotada.

    Args:
        yolo_model (YoloModel): Modelo YOLO cargado para realizar las predicciones.
        contents (bytes): Bytes de la imagen a procesar.
        confidence_threshold (float): Umbral de confianza para filtrar las detecciones.
        overlap (tuple[float, float], optional): Fracción de la máscara a recortar
            en cada dimensión para eliminar solapes. Por defecto (0.1, 0.1).
        gsd (float, optional): Ground Sample Distance en metros por píxel para
            calcular el área real. Por defecto 0.1.
        test_mode (bool, optional): Si True, devuelve la imagen anotada en
            lugar de las métricas (útil para pruebas). Por defecto False.

    Returns:
        dict | Imagen: Si `test_mode` es False, devuelve un diccionario con el
            resumen de áreas (clave `total_area_m2`) y la lista `metrics`.
            Si `test_mode` es True, devuelve la imagen anotada (`Imagen`,
            típicamente un `np.ndarray`).

    Raises:
        HTTPException: con código 400 si no se puede decodificar la imagen.
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
    results = predict(model=yolo_model, frame=frame, confidence_threshold=confidence_threshold, overlap=overlap)

    # Anotacion de la imagen con los resultados de la inferencia
    annotated_frame = draw_results(frame=frame, results=results)

    # Si estamos en modo de prueba, devolvemos las detecciones sin procesar en lugar de la imagen anotada.
    if test_mode:
        return annotated_frame

    #Calculo del area real de las mascaras
    mask_metrics = analyze_results(result=results, gsd=gsd)
    return mask_metrics


def process_zip(
    yolo_model: YoloModel,
    contents: bytes,
    confidence_threshold: float,
    gsd: float = 0.1,
) -> dict:
    """Procesa un archivo ZIP con imágenes y agrega el área total por archivo.

    Args:
        yolo_model (YoloModel): Modelo YOLO cargado para realizar las predicciones.
        contents (bytes): Bytes del archivo ZIP a procesar.
        confidence_threshold (float): Umbral de confianza para filtrar las detecciones.
        gsd (float, optional): Ground Sample Distance en metros por píxel para
            calcular el área real. Por defecto 0.1.

    Returns:
        dict: Diccionario con `total_area_m2` (float) y `metrics` (lista) donde
            cada entrada contiene `file` y `total_area_m2` y, opcionalmente,
            `metrics` con el desglose por etiquetas.

    Raises:
        HTTPException: con código 400 si el contenido no es un ZIP válido.
    """

    analyze_results_total = {
        "total_area_m2": 0.0,
        "metrics": []
    }

    try:
        # Desempaquetamos el ZIP en memoria
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            for member_name in archive.namelist():
                # Comprobacion de que el miembro es un archivo de imagen valido por su extension
                if Path(member_name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue

                # Lectura de la imagen desde el ZIP
                with archive.open(member_name) as member_file:
                    file_bytes = member_file.read()

                # Procesamiento de la imagen utilizando el mismo flujo que en process_image
                frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Procesamos la imagen
                image_results = process_image(
                    yolo_model=yolo_model,
                    contents=file_bytes,
                    confidence_threshold=confidence_threshold,
                    gsd=gsd,
                )

                # Validamos el resultado: esperamos un diccionario con 'total_area_m2'
                if not isinstance(image_results, dict) or "total_area_m2" not in image_results:
                    # Si no se obtiene métricas, saltamos este miembro pero lo registramos
                    analyze_results_total["metrics"].append({
                        "file": member_name,
                        "error": "no_metrics_generated",
                    })
                    continue

                # Actualizamos los resultados totales y guardamos el detalle por archivo
                area = float(image_results.get("total_area_m2", 0.0))
                analyze_results_total["total_area_m2"] += area
                metrics_entry = {
                    "file": member_name,
                    "total_area_m2": area,
                }

                # Si el análisis por imagen incluye un desglose, lo agregamos
                if "metrics" in image_results:
                    metrics_entry["metrics"] = image_results["metrics"]

                analyze_results_total["metrics"].append(metrics_entry)

        # Devolvemos el resumen total de las áreas calculadas para todas las imágenes procesadas.
        return analyze_results_total

    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer el archivo ZIP enviado.",
        ) from exc
