"""Tipos compartidos del pipeline de entrenamiento e inferencia."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeAlias, TypedDict

from pydantic import Field, field_validator, model_validator
from ultralytics import YOLO

from common.types.base import StrictModel
from train.utils.filesystem import resolve_path

YoloTaskResult = object
YoloModel: TypeAlias = YOLO


def _validate_workers_count(value: int, field_name: str) -> int:
    """Valida que el numero de workers sea razonable para la CPU local."""
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return value

    if value > cpu_count:
        raise ValueError(
            f"El campo '{field_name}' no puede ser mayor que las CPUs "
            f"disponibles ({cpu_count})."
        )

    return value


class TaskConfig(StrictModel):
    """Flags para activar las etapas del pipeline."""

    train: bool = False
    val: bool = False
    predict: bool = False


class TrainingModelConfig(StrictModel):
    """Configuración del modelo YOLO para entrenamiento e inferencia offline."""

    path: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Valida que la ruta del modelo exista y sea un archivo."""
        value = value.strip()
        if not value:
            raise ValueError("La ruta del modelo no puede estar vacia.")

        model_path = resolve_path(value=value)
        if not model_path.exists():
            raise ValueError(f"La ruta del modelo no existe: {model_path}")
        if not model_path.is_file():
            raise ValueError(f"La ruta del modelo debe ser un archivo: {model_path}")

        return str(model_path)


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
            raise ValueError("El dispositivo de entrenamiento no puede estar vacio.")
        return value

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, value: int) -> int:
        """Valida que los workers no excedan la CPU disponible."""
        return _validate_workers_count(value, "train.workers")


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
            raise ValueError("El dispositivo de validacion no puede estar vacio.")
        return value

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, value: int) -> int:
        """Valida que los workers no excedan la CPU disponible."""
        return _validate_workers_count(value, "val.workers")


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
        """Valida la ruta de entrada para predicción si se informa."""
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("La ruta de entrada para prediccion no puede estar vacia.")

        source_path = resolve_path(value=value)
        if not source_path.exists():
            raise ValueError(
                f"La ruta de entrada para prediccion no existe: {source_path}"
            )

        return str(source_path)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Evita valores vacíos para el dispositivo."""
        value = value.strip()
        if not value:
            raise ValueError("El dispositivo de prediccion no puede estar vacio.")
        return value


class TrainPipelineConfig(StrictModel):
    """Configuración global del pipeline offline."""

    pathData: str
    pathResult: str
    model: TrainingModelConfig
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

    @field_validator("pathData")
    @classmethod
    def validate_data_path_exists(cls, value: str) -> str:
        """Valida que el dataset de entrada exista y sea un archivo."""
        data_path = resolve_path(value=value)
        if not data_path.exists():
            raise ValueError(f"La ruta de datos no existe: {data_path}")
        if not data_path.is_file():
            raise ValueError(f"La ruta de datos debe ser un archivo: {data_path}")
        return str(data_path)

    @field_validator("pathResult")
    @classmethod
    def validate_or_create_output_path(cls, value: str) -> str:
        """Crea el directorio de salida si no existe y valida su tipo."""
        output_path = resolve_path(value=value)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(
                "La ruta de salida debe ser un directorio: " f"{output_path}"
            )

        output_path.mkdir(parents=True, exist_ok=True)

        return str(output_path)

    @model_validator(mode="after")
    def validate_task_dependencies(self) -> "TrainPipelineConfig":
        """Comprueba coherencia mínima entre tareas y configuración."""
        if not any((self.task.train, self.task.val, self.task.predict)):
            raise ValueError("Debe haber al menos una tarea activada en 'task'.")

        if self.task.predict and self.predict.source is None:
            raise ValueError(
                "Si 'task.predict' es true, debe informarse "
                "'predict.source' con una ruta valida."
            )

        if self.task.predict and self.predict.source is not None:
            predict_source = Path(self.predict.source)
            if not predict_source.exists():
                raise ValueError(
                    "La fuente de prediccion configurada no existe: "
                    f"{predict_source}"
                )

        return self


class PipelineResults(TypedDict, total=False):
    """Resultados devueltos por las etapas del pipeline."""

    train: YoloTaskResult
    val: YoloTaskResult
    predict: YoloTaskResult


AppConfig = TrainPipelineConfig
