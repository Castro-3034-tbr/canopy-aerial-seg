"""Funciones de postproceso para mascaras y detecciones."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.constants import DEFAULT_MASK_THRESHOLD
from src.core.types import Coordinates, MaskArray, MaskMetric, VerticesList


def calculate_centroid(mask: MaskArray) -> Coordinates:
    """Calcula el centroide de una mascara de segmentacion."""
    # Sin mascara no existe un centroide calculable.
    if len(mask) == 0:
        return Coordinates(x=-1, y=-1)

    # Usa los momentos geometricos para obtener el centro de masa.
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return Coordinates(x=-1, y=-1)

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return Coordinates(x=center_x, y=center_y)


def extract_vertices(mask: MaskArray) -> VerticesList:
    """Extrae los vertices del contorno principal de una mascara."""
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

    contour = max(contours, key=cv2.contourArea)
    vertices = [
        Coordinates(x=int(point[0][0]), y=int(point[0][1])) for point in contour
    ]
    return vertices


def process_mask(masks: np.ndarray) -> list[MaskMetric]:
    """Calcula metricas basicas para un conjunto de mascaras."""
    # Devuelve una lista vacia cuando el modelo no produce mascaras.
    if len(masks) == 0:
        return []

    # Calcula el area del frame para normalizar las metricas.
    frame_height, frame_width = masks.shape[1:3]
    frame_area = frame_width * frame_height
    mask_metrics = []

    # Resume cada mascara con area absoluta y proporcion relativa.
    for index, mask in enumerate(masks):
        mask_area = int(np.sum(mask > DEFAULT_MASK_THRESHOLD))
        area_ratio = float(mask_area / frame_area) if frame_area else 0.0
        mask_metrics.append(
            {
                "mask_index": index,
                "mask_area": mask_area,
                "frame_area": frame_area,
                "area_ratio": area_ratio,
            }
        )

    return mask_metrics
