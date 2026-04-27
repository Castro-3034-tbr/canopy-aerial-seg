from __future__ import annotations

from typing import TypeAlias

from pydantic import Field, field_validator, model_validator

from common.types.base import StrictModel

class Coordinates(StrictModel):
    """Coordenadas normalizadas de un punto en el rango [0, 1]."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class BoundingBox(StrictModel):
    """Caja envolvente derivada de un polígono en coordenadas normalizadas."""

    p1: Coordinates
    p2: Coordinates
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)

    @field_validator("width", "height")
    @classmethod
    def validar_dimension(cls, value: float) -> float:
        """Evita dimensiones negativas por redondeo."""
        return max(0.0, value)

    @model_validator(mode="before")
    @classmethod
    def normalize_bbox_input(cls, data: object) -> object:
        """Acepta bboxes definidas por puntos o por extremos min/max."""
        if not isinstance(data, dict):
            return data

        if {"x_min", "y_min", "x_max", "y_max"}.issubset(data):
            x_min = float(data["x_min"])
            y_min = float(data["y_min"])
            x_max = float(data["x_max"])
            y_max = float(data["y_max"])
            return {
                "p1": {"x": x_min, "y": y_min},
                "p2": {"x": x_max, "y": y_max},
                "width": data.get("width", max(0.0, x_max - x_min)),
                "height": data.get("height", max(0.0, y_max - y_min)),
            }

        return data

    @property
    def x_min(self) -> float:
        return self.p1.x

    @property
    def y_min(self) -> float:
        return self.p1.y

    @property
    def x_max(self) -> float:
        return self.p2.x

    @property
    def y_max(self) -> float:
        return self.p2.y


Polygon: TypeAlias = list[Coordinates]
Mask: TypeAlias = list[Coordinates]
ImageSize: TypeAlias = tuple[int, int]
