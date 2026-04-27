from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field

from common.types.base import StrictModel
from common.types.media import FrameMask
from common.types.geometry import Mask, BoundingBox, Coordinates

class InferenceDetection(StrictModel):
    """Detección serializable generada por inferencia."""

    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    mask: Mask
    frame_mask: FrameMask
    centroid: Coordinates


ConfidenceThreshold: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]

MaskLabel: TypeAlias = tuple[int, Mask]
BboxLabel: TypeAlias = tuple[int, BoundingBox]
