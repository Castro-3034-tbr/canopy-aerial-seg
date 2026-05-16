"""Tipos compartidos para frames, inferencia y salidas multimedia."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import FilePath

from common.types.base import StrictModel

Imagen: TypeAlias = NDArray[np.floating[Any] | np.uint8 | np.bool_]
OutputPathResult: TypeAlias = tuple[Path, str]


class MaskMetric(StrictModel):
    """Métricas resumidas de una máscara."""

    index: int
    area: float


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
