"""Funciones de postproceso para mascaras y detecciones."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.core.constants import DEFAULT_MASK_THRESHOLD
from src.core.types import Coordinates, Polygon, FrameMask, MaskMetric, Detection, Mask


def calculate_centroid(polygon: Polygon) -> Coordinates:
    """Calcula el centroide de un polígono.

    Si el área es casi cero, devuelve la media de los puntos para evitar
    divisiones por cero.
    """

    x = np.array([point.x for point in polygon], dtype=np.float64)
    y = np.array([point.y for point in polygon], dtype=np.float64)

    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = np.sum(cross)

    if abs(area2) < 1e-12:
        return Coordinates(x=float(np.mean(x)), y=float(np.mean(y)))

    cx = np.sum((x + np.roll(x, -1)) * cross) / (3 * area2)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (3 * area2)

    return Coordinates(
        x=float(np.clip(cx, 0.0, 1.0)),
        y=float(np.clip(cy, 0.0, 1.0)),
    )


def extract_vertices(mask: FrameMask) -> Mask:
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

    # Selecciona el contorno mas grande asumiendo que es el objeto principal.
    contour = max(contours, key=cv2.contourArea)
    mask_height, mask_width = mask.shape[:2]
    vertices = [
        Coordinates(
            x=float(point[0][0] / mask_width),
            y=float(point[0][1] / mask_height),
        )
        for point in contour
    ]
    return vertices

def crop_mask(
    masks : list[Mask],
    image_shape : tuple[int,int],
    overlap: tuple[float,float]
)-> list[FrameMask]:
    """Recorta máscaras eliminando solo la parte en solape izquierdo y superior.

    Args:
        masks (list[FrameMask]): _description_
        image (tuple[int,int]): _description_
        overlap (tuple[float,float]): _description_

    Returns:
        FrameMasks: _description_
    """

    # Obtenemos las dimensiones de la imagen para calcular los limites de recorte
    h, w = image_shape

    # Calculamos los limites de recorte en pixeles a partir del porcentaje de solape
    left_limit = int(overlap[0] * w)
    top_limit = int(overlap[1] * h)

    # Crear ROI válida
    valid_region = np.ones((h, w), dtype=np.uint8)
    valid_region[:, :left_limit] = 0     # eliminar izquierda
    valid_region[:top_limit, :] = 0      # eliminar arriba

    cropped_masks = []

    for mask in masks:
        cropped = mask * valid_region  # AND lógico

        # Opcional: descartar si queda demasiado pequeña
        if np.sum(cropped) > 0:
            cropped_masks.append(cropped)

    return cropped_masks


def process_mask(masks: FrameMask, gsd: float) -> list[MaskMetric]:
    """Calcula metricas basicas para un conjunto de mascaras."""

    #TODO: Refactorizar para:
    # - Calcular el area de la mascara mediante los pixeles 
    # - Calcular el area real usando el GSD (Ground Sample Distance) 

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


def convert_detections_to_json(detections: list[Detection], frame_id: int | None) -> dict[str, Any]:
    """Convierte las detecciones a un formato JSON serializable."""
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