"""Tipos compartidos del proyecto."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Modelo base que fuerza validación estricta de campos."""

    model_config = ConfigDict(extra="forbid")


class Coordinates(StrictModel):
    """Coordenadas normalizadas de un punto en el rango [0, 1]."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class BoundingBox(StrictModel):
    """Caja envolvente derivada de un polígono en coordenadas normalizadas."""

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)

    @field_validator("width", "height")
    @classmethod
    def validar_dimension(cls, value: float) -> float:
        """Evita dimensiones negativas por redondeo."""
        return max(0.0, value)


Polygon: TypeAlias = list[Coordinates]
MaskLabel: TypeAlias = tuple[int, Polygon]
BboxLabel: TypeAlias = tuple[int, BoundingBox]

ImageSize: TypeAlias = tuple[int, int]
AspectRatioCounts: TypeAlias = dict[str, int]
QuadrantCounts: TypeAlias = dict[str, int]
LabelsPerImage: TypeAlias = list[int]
MetricValues: TypeAlias = list[float]
LabelsSizes: TypeAlias = list[float]
LabelsCenters: TypeAlias = list[Coordinates]


class ImageData(StrictModel):
    """Imagen con su ruta y array de píxeles."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    path: Path
    data: np.ndarray

    @field_validator("data")
    @classmethod
    def validar_array(cls, value: np.ndarray) -> np.ndarray:
        """Verifica que el array represente una imagen válida."""
        if value is None or value.size == 0:
            raise ValueError("El array de imagen está vacío o es None.")
        if value.ndim not in (2, 3):
            raise ValueError(
                f"Se esperaba un array 2D o 3D, recibido: {value.ndim}D."
            )
        return value


class LabelData(StrictModel):
    """Etiquetas válidas extraídas de un archivo de texto."""

    path: Path
    masks: list[MaskLabel]
    bboxes: list[BboxLabel]


class ImagesLoaderResult(StrictModel):
    """Resultado de la carga de imágenes."""

    images: list[ImageData]
    incorrect_images: list[Path]


class LabelsLoaderResult(StrictModel):
    """Resultado de la carga de etiquetas."""

    labels: list[LabelData]
    incorrect_labels: list[Path]


def _default_horizontal_quadrants() -> QuadrantCounts:
    return {"Izquierda": 0, "Centro": 0, "Derecha": 0}


def _default_vertical_quadrants() -> QuadrantCounts:
    return {"Arriba": 0, "Centro": 0, "Abajo": 0}


class AnalysisResult(StrictModel):
    """Resultados agregados del análisis EDA."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    images: list[ImageData] = Field(default_factory=list)
    labels: list[LabelData] = Field(default_factory=list)
    incorrect_images: list[Path] = Field(default_factory=list)
    incorrect_labels: list[Path] = Field(default_factory=list)

    image_types: AspectRatioCounts = Field(default_factory=dict)
    image_sizes: dict[ImageSize, int] = Field(default_factory=dict)
    image_aspect_ratios: AspectRatioCounts = Field(default_factory=dict)
    images_brightness: MetricValues = Field(default_factory=list)
    images_contrast: MetricValues = Field(default_factory=list)
    images_blur: MetricValues = Field(default_factory=list)

    num_labels_per_image: LabelsPerImage = Field(default_factory=list)
    label_areas: LabelsSizes = Field(default_factory=list)
    label_aspect_ratios: AspectRatioCounts = Field(default_factory=dict)
    label_area_ratios: MetricValues = Field(default_factory=list)
    label_area_ratios_per_image: MetricValues = Field(default_factory=list)
    labels_centers: LabelsCenters = Field(default_factory=list)
    label_quadrants_x: QuadrantCounts = Field(default_factory=_default_horizontal_quadrants)
    label_quadrants_y: QuadrantCounts = Field(default_factory=_default_vertical_quadrants)
    labels_iou: MetricValues = Field(default_factory=list)