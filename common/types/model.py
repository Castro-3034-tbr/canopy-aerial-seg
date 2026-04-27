from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field

from common.types.geometry import Mask, BoundingBox

ConfidenceThreshold: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]

MaskLabel: TypeAlias = tuple[int, Mask]
BboxLabel: TypeAlias = tuple[int, BoundingBox]
