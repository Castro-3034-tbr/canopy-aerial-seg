"""Tipos compartidos del pipeline de entrenamiento e inferencia."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from ultralytics import YOLO


YoloTaskResult = object
YoloModel: TypeAlias = YOLO


class StrictModel(BaseModel):
    """Modelo base con validación estricta de campos."""

    model_config = ConfigDict(extra="forbid")


class TaskConfig(StrictModel):
    """Flags para activar las etapas del pipeline."""

    train: bool = False
    val: bool = False
    predict: bool = False


class ModelConfig(StrictModel):
    """Configuración del modelo YOLO."""

    path: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Evita rutas vacías o con solo espacios."""
        value = value.strip()
        if not value:
            raise ValueError("La ruta del modelo no puede estar vacia.")
        return value


class TrainConfig(StrictModel):
    """Configuración admitida para entrenamiento."""

    augmentation: bool = True
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=16, gt=0)
    img_size: int = Field(default=640, gt=0)
    device: str = "cpu"
    save_period: int = Field(default=-1, ge=-1)
    seed: int = Field(default=42, ge=0)
    patience: int = Field(default=10, ge=0)
    workers: int = Field(default=4, ge=0)
    mosaic: bool = False

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Evita valores vacíos para el dispositivo."""
        value = value.strip()
        if not value:
            raise ValueError(
                "El dispositivo de entrenamiento no puede estar vacio."
            )
        return value


class ValidationConfig(StrictModel):
    """Configuración admitida para validación."""

    batch_size: int = Field(default=16, gt=0)
    img_size: int = Field(default=640, gt=0)
    device: str = "cpu"
    workers: int = Field(default=4, ge=0)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Evita valores vacíos para el dispositivo."""
        value = value.strip()
        if not value:
            raise ValueError(
                "El dispositivo de validacion no puede estar vacio."
            )
        return value


class PredictConfig(StrictModel):
    """Configuración admitida para predicción."""

    source: str | None = None
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    img_size: int = Field(default=640, gt=0)
    device: str = "cpu"
    save_results: bool = False
    save_txt: bool = False
    save_conf: bool = False
    exist_ok: bool = True

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str | None) -> str | None:
        """Evita rutas de entrada vacías cuando se informan."""
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError(
                "La ruta de entrada para prediccion no puede estar vacia."
            )
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Evita valores vacíos para el dispositivo."""
        value = value.strip()
        if not value:
            raise ValueError(
                "El dispositivo de prediccion no puede estar vacio."
            )
        return value


class PipelineConfig(StrictModel):
    """Configuración global del pipeline."""

    pathData: str
    pathResult: str
    model: ModelConfig
    task: TaskConfig = Field(default_factory=TaskConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    val: ValidationConfig = Field(default_factory=ValidationConfig)
    predict: PredictConfig = Field(default_factory=PredictConfig)

    @field_validator("pathData", "pathResult")
    @classmethod
    def validate_non_empty_path(cls, value: str) -> str:
        """Evita rutas vacías en la configuración principal."""
        value = value.strip()
        if not value:
            raise ValueError("Las rutas principales no pueden estar vacias.")
        return value

    @model_validator(mode="after")
    def validate_task_dependencies(self) -> "PipelineConfig":
        """Comprueba coherencia mínima entre tareas y configuración."""
        if not any((self.task.train, self.task.val, self.task.predict)):
            raise ValueError(
                "Debe haber al menos una tarea activada en 'task'."
            )

        return self


class PipelineResults(TypedDict, total=False):
    """Resultados devueltos por las etapas del pipeline."""

    train: YoloTaskResult
    val: YoloTaskResult
    predict: YoloTaskResult
