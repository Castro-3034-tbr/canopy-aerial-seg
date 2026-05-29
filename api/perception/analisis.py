"""Módulo de análisis de máscaras para la API de percepción.

Proporciona utilidades para eliminar solapes en máscaras, calcular áreas
reales y agregar métricas por detección.
"""

from __future__ import annotations

import numpy as np

from typing import Any
from common.types.media import Imagen
from common.types.model import InferenceDetection

# from api.core.constants import DEFAULT_MASK_THRESHOLD

DEFAULT_MASK_THRESHOLD = (
    0.5  # Valor de umbral para considerar un píxel como parte de la máscara
)


def remove_overlap(mask: Imagen, overlap: tuple[float, float]) -> Imagen:
    """Recorta una máscara eliminando la región en solape izquierdo y superior.

    Args:
        mask (Imagen): Máscara binaria (H x W) del objeto detectado.
        overlap (tuple[float,float]): Fracción a recortar en (x, y) en [0.0, 1.0].

    Returns:
        Imagen: Máscara recortada conservando el mismo tipo y forma.
    """


    
    # Calculamos los píxeles a recortar en cada dimensión (x -> ancho, y -> alto).
    h, w = mask.shape[:2]
    overlap_x = int(max(0, min(1.0, overlap[0])) * w)
    overlap_y = int(max(0, min(1.0, overlap[1])) * h)

    # Evita recortes fuera de rango.
    overlap_x = min(overlap_x, w)
    overlap_y = min(overlap_y, h)

    # Ponemos en cero (blanco-negro) los píxeles en solape izquierdo y superior.
    if overlap_y > 0:
        mask[:overlap_y, :] = 0
    if overlap_x > 0:
        mask[:, :overlap_x] = 0

    return mask


def calculate_real_area(mask: Imagen, gsd: float) -> float:
    """Calcula el área real de una máscara usando el GSD.

    Args:
        mask (Imagen): Máscara binaria o probabilística del objeto.
        gsd (float): Ground Sample Distance en metros/píxel.

    Returns:
        float: Área estimada en metros cuadrados.
    """

    pixel_area_m2 = gsd * gsd
    mask_area = int(np.sum(mask > DEFAULT_MASK_THRESHOLD))
    area_real = mask_area * pixel_area_m2
    return area_real


def analyze_results(result: list[InferenceDetection], gsd: float) -> dict[str, Any]:
    """Agrega métricas de área real para una lista de detecciones.

    Args:
        result (list[InferenceDetection]): Lista de detecciones con `frame_mask`.
        gsd (float): Ground Sample Distance en metros/píxel.

    Returns:
        dict[str, Any]: Diccionario con `total_area_m2` (float) y `metrics` (lista)
            donde cada entrada contiene `index` y `area`.
    """

    areas = {
        "total_area_m2": 0.0,
        "metrics": [],
    }

    for index, detection in enumerate(result):
        metric = calculate_real_area(mask=detection.frame_mask, gsd=gsd)
        areas["total_area_m2"] += metric
        areas["metrics"].append({"index": index, "area": metric})

    return areas
