"""Tipos compartidos para frames, inferencia y salidas multimedia."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, FilePath

from common.types.base import StrictModel
from common.types.geometry import Coordinates, BoundingBox, Mask


Imagen: TypeAlias = NDArray[np.uint8]
FrameMask: TypeAlias = NDArray[np.floating[Any] | np.uint8 | np.bool_]
OutputPathResult: TypeAlias = tuple[Path, str]

class InferenceDetection(StrictModel):
    """Detección serializable generada por inferencia."""

    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    mask: Mask
    frame_mask: FrameMask
    centroid: Coordinates


class MaskMetric(StrictModel):
    """Métricas resumidas de una máscara."""

    mask_index: int
    mask_area: int
    frame_area: int
    area_ratio: float


class FramePackage(StrictModel):
    """Frame compartido entre procesos lector y procesador."""

    img: Imagen
    frame_id: int
    pts: int | None
    width: int
    height: int


class OutputFile(StrictModel):
    """Metadatos de un archivo de salida generado por inferencia."""

    path: FilePath
    media_type: str
