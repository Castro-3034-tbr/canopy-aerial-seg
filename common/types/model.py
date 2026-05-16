from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.metrics import DetMetrics

from common.types.base import StrictModel
from common.types.geometry import BoundingBox, Coordinates, Mask
from common.types.media import Imagen

YoloModel: TypeAlias = YOLO
YoloResult: TypeAlias = DetMetrics | list[Results] | None


class InferenceDetection(StrictModel):
    """Detección serializable generada por inferencia."""

    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    mask: Mask
    frame_mask: Imagen
    centroid: Coordinates


ConfidenceThreshold: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]

MaskLabel: TypeAlias = tuple[int, Mask]
BboxLabel: TypeAlias = tuple[int, BoundingBox]
