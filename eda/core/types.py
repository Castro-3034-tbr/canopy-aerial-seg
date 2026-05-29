"""Tipos compartidos del dominio EDA."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from common.types.base import StrictModel
from common.types.geometry import (
    Coordinates,
    ImageSize,
)
from common.types.model import BboxLabel, MaskLabel


class ImageData(StrictModel):
    """Imagen con su ruta y metadatos básicos."""

    path: Path
    width: int = Field(..., ge=1, description="Ancho de la imagen en píxeles")
    height: int = Field(..., ge=1, description="Alto de la imagen en píxeles")
    channels: int = Field(..., ge=1, description="Número de canales de color")


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


class AnalysisResult(StrictModel):
    """Resultados agregados del análisis EDA."""

    images: list[ImageData] = Field(default_factory=list)
    labels: list[LabelData] = Field(default_factory=list)
    incorrect_images: list[Path] = Field(default_factory=list)
    incorrect_labels: list[Path] = Field(default_factory=list)

    image_types: dict[str, int] = Field(default_factory=dict)
    image_sizes: dict[ImageSize, int] = Field(default_factory=dict)
    image_aspect_ratios: dict[str, int] = Field(default_factory=dict)
    images_brightness: list[float] = Field(default_factory=list)
    images_contrast: list[float] = Field(default_factory=list)
    images_blur: list[float] = Field(default_factory=list)

    num_labels_per_image: list[int] = Field(default_factory=list)
    label_areas: list[float] = Field(default_factory=list)
    labels_areas: list[float] = Field(default_factory=list)
    label_aspect_ratios: dict[str, int] = Field(default_factory=dict)
    labels_centers: list[Coordinates] = Field(default_factory=list)
    label_quadrants_x: dict[str, int] = Field(
        default_factory=lambda: {"Izquierda": 0, "Centro": 0, "Derecha": 0}
    )
    label_quadrants_y: dict[str, int] = Field(
        default_factory=lambda: {"Arriba": 0, "Centro": 0, "Abajo": 0}
    )
    labels_iou: list[float] = Field(default_factory=list)
