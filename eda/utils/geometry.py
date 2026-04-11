from __future__ import annotations

import numpy as np

from eda.core.types import Coordinates, Polygon, BoundingBox


def calculate_area_polygon(polygon: Polygon) -> float:
    """Calcula el área normalizada de un polígono usando Shoelace."""
    x = np.array([point.x for point in polygon], dtype=np.float64)
    y = np.array([point.y for point in polygon], dtype=np.float64)

    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def calculate_centroid_polygon(polygon: Polygon) -> Coordinates:
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

def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calcula el Intersection over Union entre dos bounding boxes."""
    x_min_inter = max(bbox1.x_min, bbox2.x_min)
    y_min_inter = max(bbox1.y_min, bbox2.y_min)
    x_max_inter = min(bbox1.x_max, bbox2.x_max)
    y_max_inter = min(bbox1.y_max, bbox2.y_max)

    inter_width = max(0.0, x_max_inter - x_min_inter)
    inter_height = max(0.0, y_max_inter - y_min_inter)
    area_inter = inter_width * inter_height

    area_bbox1 = bbox1.width * bbox1.height
    area_bbox2 = bbox2.width * bbox2.height
    area_union = area_bbox1 + area_bbox2 - area_inter

    return area_inter / area_union if area_union > 0 else 0.0