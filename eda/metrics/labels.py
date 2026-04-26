"""Métricas EDA para etiquetas."""

#TODO: Revisar que esten correctos

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Sequence

from eda.utils.geometry import(
    calculate_area_polygon,
    calculate_centroid_polygon,
    calculate_iou,
)
from eda.core.types import (
    LabelData,
    ImageData,
    LabelsCenters,
    LabelsPerImage,
    LabelsSizes,
    MetricValues,
    QuadrantCounts,
)


def compute_label_areas(labels: Sequence[LabelData]) -> LabelsSizes:
    """Calcula el área relativa de cada máscara en porcentaje."""
    label_areas: LabelsSizes = []

    for label in labels:
        for _, polygon in label.masks:
            area_percent = calculate_area_polygon(polygon) * 100.0
            label_areas.append((area_percent))

    return label_areas

def count_label_aspect_ratios(labels: Sequence[LabelData]) -> dict[str, int]:
    """Cuenta etiquetas por relación de aspecto usando su bounding box."""
    aspect_ratios: list[str] = []

    for label in labels:
        for _, bbox in label.bboxes:
            if bbox.width <= 0 or bbox.height <= 0:
                continue

            width_int = int(round(bbox.width))
            height_int = int(round(bbox.height))
            if width_int <= 0 or height_int <= 0:
                continue
            fraction = Fraction(width_int, height_int)
            aspect_ratios.append(f"{fraction.numerator}/{fraction.denominator}")

    return dict(Counter(aspect_ratios))


def compute_label_centers(labels: Sequence[LabelData]) -> LabelsCenters:
    """Calcula el centroide de cada máscara y lo asocia a su archivo."""
    centers: LabelsCenters = []

    for label in labels:
        for _, polygon in label.masks:
            centers.append((calculate_centroid_polygon(polygon)))

    return centers


def count_label_quadrants_x(labels_centers: LabelsCenters) -> QuadrantCounts:
    """Cuenta centros de etiquetas por zona horizontal."""
    quadrants: QuadrantCounts = {"Izquierda": 0, "Centro": 0, "Derecha": 0}

    for center in labels_centers:
        if center.x < 0.33:
            quadrants["Izquierda"] += 1
        elif center.x < 0.66:
            quadrants["Centro"] += 1
        else:
            quadrants["Derecha"] += 1

    return quadrants


def count_label_quadrants_y(labels_centers: LabelsCenters) -> QuadrantCounts:
    """Cuenta centros de etiquetas por zona vertical."""
    quadrants: QuadrantCounts = {"Arriba": 0, "Centro": 0, "Abajo": 0}

    for center in labels_centers:
        if center.y < 0.33:
            quadrants["Arriba"] += 1
        elif center.y < 0.66:
            quadrants["Centro"] += 1
        else:
            quadrants["Abajo"] += 1

    return quadrants


def count_labels_per_image(labels: Sequence[LabelData]) -> LabelsPerImage:
    """Cuenta el número de etiquetas por archivo."""
    return [len(label.masks) for label in labels]


def compute_labels_iou(labels: Sequence[LabelData]) -> MetricValues:
    """Calcula IoU entre todas las parejas de etiquetas por archivo."""
    iou_values: MetricValues = []

    for label in labels:
        bboxes = [bbox for _, bbox in label.bboxes]
        if len(bboxes) < 2:
            continue

        for i, bbox1 in enumerate(bboxes):
            for bbox2 in bboxes[i + 1 :]:
                iou_values.append(calculate_iou(bbox1, bbox2))

    return iou_values

def compute_area_label_image_ratio(images: Sequence[ImageData], labels: Sequence[LabelData]) -> MetricValues:
    """Calculo de la porporcion de area cubierta por cada etiqueta en cada imagen."""
    # Implementar otras métricas relevantes para las etiquetas

    ratios = []

    for image, label in zip(images, labels, strict=True):
        for _, polygon in label.masks:
            area_polygon = calculate_area_polygon(polygon)
            area_image = image.width * image.height
            if area_image > 0:
                ratios.append(area_polygon / area_image)

    return ratios

def compute_area_labels_image_ratio(images: Sequence[ImageData], labels: Sequence[LabelData]) -> MetricValues:
    """Calculo de la porporcion de area cubierta por cada etiqueta en cada imagen."""
    # Implementar otras métricas relevantes para las etiquetas

    ratios = []

    for image, label in zip(images, labels, strict=True):
        image_area = image.width * image.height
        labels_total_area = sum(calculate_area_polygon(polygon) for _, polygon in label.masks)
        ratios.append(labels_total_area / image_area if image_area > 0 else 0)

    return ratios