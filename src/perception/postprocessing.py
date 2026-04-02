"""Funciones de postproceso para mascaras y detecciones."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.constants import DEFAULT_MASK_THRESHOLD


def calculate_centroid(mask: np.ndarray | None) -> tuple[int, int] | None:
    """Calcula el centroide de una mascara de segmentacion."""
    # Sin mascara no existe un centroide calculable.
    if mask is None:
        return None

    # Usa los momentos geometricos para obtener el centro de masa.
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return center_x, center_y


def extract_vertices(mask: np.ndarray | None) -> np.ndarray:
    """Extrae los vertices del contorno principal de una mascara."""
    # Si no hay mascara, no hay contorno que analizar.
    if mask is None:
        return np.array([])

    # Recupera solo contornos externos para representar la silueta principal.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return np.array([])

    contour = max(contours, key=cv2.contourArea)
    return contour.squeeze()


def process_mask(masks) -> list[dict[str, float | int]]:
    """Calcula metricas basicas para un conjunto de mascaras."""
    # Devuelve una lista vacia cuando el modelo no produce mascaras.
    if masks is None:
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
