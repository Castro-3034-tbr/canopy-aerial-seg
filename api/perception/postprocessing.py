"""Funciones de postproceso para máscaras y detecciones.

Contiene utilidades para extraer vértices de máscaras y calcular centroides
normalizados sobre polígonos definidos por coordenadas.
"""

from __future__ import annotations

import cv2
import numpy as np

from common.types.geometry import Coordinates, Polygon, Mask
from common.types.media import Imagen


def calculate_centroid(polygon: Polygon) -> Coordinates:
    """Calcula el centroide de un polígono en coordenadas normalizadas.

    Si el área es prácticamente cero se devuelve la media de los puntos
    para evitar divisiones por cero.

    Args:
        polygon (Polygon): Lista de `Coordinates` que definen el polígono
            con valores normalizados en el rango [0.0, 1.0].

    Returns:
        Coordinates: Coordenadas del centroide normalizadas entre 0 y 1.
    """

    # Si no hay puntos, devuelve el origen (0,0) como centroide por defecto.
    if not polygon:
        return Coordinates(x=0.0, y=0.0)

    # Extraemos las coordenadas x e y de los puntos del polígono.
    x = np.array([point.x for point in polygon], dtype=np.float64)
    y = np.array([point.y for point in polygon], dtype=np.float64)

    # Calculamos el área del polígono usando la fórmula del área de un polígono
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = np.sum(cross)

    if abs(area2) < 1e-12:
        return Coordinates(x=float(np.clip(np.mean(x), 0.0, 1.0)), y=float(np.clip(np.mean(y), 0.0, 1.0)))

    # Calculamos el centroide usando la fórmula del centroide de un polígono
    cx = np.sum((x + np.roll(x, -1)) * cross) / (3 * area2)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (3 * area2)

    return Coordinates(
        x=float(np.clip(cx, 0.0, 1.0)),
        y=float(np.clip(cy, 0.0, 1.0)),
    )


def extract_vertices(mask: Imagen) -> Mask:
    """Extrae los vértices del contorno principal de una máscara.

    Args:
        mask (Imagen): Máscara binaria del objeto detectado (H x W) con
            valores 0/1 o 0/255.

    Returns:
        Mask: Lista de `Coordinates` normalizadas del contorno principal. Si
            no hay contornos, se devuelve una lista vacía.
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
