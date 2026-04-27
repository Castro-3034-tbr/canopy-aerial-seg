"""Funciones de postproceso para mascaras y detecciones."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from common.types.model import InferenceDetection
from common.types.geometry import Coordinates, Polygon, Mask
from common.types.media import FrameMask,MaskMetric


def calculate_centroid(polygon: Polygon) -> Coordinates:
    """Calcula el centroide de un polígono.

    Si el área es casi cero, devuelve la media de los puntos para evitar
    divisiones por cero.
    
    Args:
        polygon: Lista de puntos que definen el polígono, con coordenadas normalizadas
    Returns:
        Coordenadas del centroide con valores normalizados entre 0 y 1.
    
    """

    # Si no hay puntos, no se puede calcular un centroide válido.
    if not polygon:
        return Coordinates(x=-1.0, y=-1.0)

    # Extraemos las coordenadas x e y de los puntos del polígono.
    x = np.array([point.x for point in polygon], dtype=np.float64)
    y = np.array([point.y for point in polygon], dtype=np.float64)

    # Calculamos el área del polígono usando la fórmula del área de un polígono
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = np.sum(cross)

    if abs(area2) < 1e-12:
        return Coordinates(x=float(np.mean(x)), y=float(np.mean(y)))

    # Calculamos el centroide usando la fórmula del centroide de un polígono
    cx = np.sum((x + np.roll(x, -1)) * cross) / (3 * area2)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (3 * area2)

    return Coordinates(
        x=float(np.clip(cx, 0.0, 1.0)),
        y=float(np.clip(cy, 0.0, 1.0)),
    )


def extract_vertices(mask: FrameMask) -> Mask:
    """Extrae los vertices del contorno principal de una mascara.
    Args:
        mask (FrameMask): Mascara binaria del objeto detectado.
    Returns:
        Mask: Lista de coordenadas normalizadas de los vertices del contorno.
    """
    # Si no hay mascara, no hay contorno que analizar.
    if len(mask) == 0:
        return []

    # Recupera solo contornos externos para representar la silueta principal.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []

    # Selecciona el contorno mas grande asumiendo que es el objeto principal.
    contour = max(contours, key=cv2.contourArea)
    mask_height, mask_width = mask.shape[:2]
    pts = contour.reshape(-1, 2)
    vertices = [
        Coordinates(
            x=float(float(x) / mask_width),
            y=float(float(y) / mask_height),
        )
        for x, y in pts
    ]
    return vertices


def convert_detections_to_json(detections: list[InferenceDetection], frame_id: int | None) -> dict[str, Any]:
    """Convierte las detecciones a un formato JSON serializable.
    Args:
        detections (list[InferenceDetection]): Lista de detecciones con coordenadas normalizadas.
        frame_id (int | None): Identificador del frame asociado a las detecciones, si
            está disponible.
    Returns:
        dict[str, Any]: Diccionario con la información de las detecciones listo para ser convertido a JSON.
    """
    
    if frame_id is not None:
        json_detections: dict[str, Any] = {"frame_id": frame_id}
    else:
        json_detections = {}
    for i , det in enumerate(detections):
        json_detections[f"detection_{i}"] = {
            "class_id": det.class_id,
            "confidence": det.confidence,
            "bbox": {
                "p1": {"x": det.bbox.p1.x, "y": det.bbox.p1.y},
                "p2": {"x": det.bbox.p2.x, "y": det.bbox.p2.y},
                "width": det.bbox.width,
                "height": det.bbox.height,
            },
            "mask": [{"x": point.x, "y": point.y} for point in det.mask],
            "centroid": {"x": det.centroid.x, "y": det.centroid.y},
        }

    return json_detections
