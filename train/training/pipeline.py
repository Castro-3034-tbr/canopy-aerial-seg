from __future__ import annotations

import logging
import os
from pathlib import Path

from train.core.types import (
    PipelineConfig,
    PipelineResults,
    PredictConfig,
    TaskConfig,
    TrainConfig,
    ValidationConfig,
    YoloModel,
)
from train.inference.predictor import yolo_predict
from train.training.trainer import yolo_train
from train.training.validator import yolo_validate
from train.utils.filesystem import clean_cache

logger = logging.getLogger(__name__)


class YoloPipeline:
    """Orquesta la ejecución de entrenamiento, validación y predicción.

    Esta clase coordina la ejecución de distintas etapas del flujo
    sin implementar lógica específica de cada tarea.
    """

    def __init__(
        self,
        model: YoloModel,
        data_path: str,
        output_path: str,
    ) -> None:
        """Inicializa el pipeline.

        Args:
            model: Modelo YOLO cargado con Ultralytics.
            data_path (str): Ruta al dataset (data.yaml).
            output_path (str): Ruta base de salida.
        """
        self.model = model
        self.data_path = data_path
        self.output_path = output_path

    def run(self, config: PipelineConfig) -> PipelineResults:
        """Ejecuta las etapas activadas en la configuración.

        Args:
            config (PipelineConfig): Configuración global validada
                del pipeline.

        Returns:
            PipelineResults: Resultados de las etapas ejecutadas.
        """

        results: PipelineResults = {}
        task_config: TaskConfig = config.task
        train_config: TrainConfig = config.train
        validation_config: ValidationConfig = config.val
        predict_config: PredictConfig = config.predict

        logger.info("Iniciando ejecución del pipeline YOLO.")

        data_directory = os.path.dirname(Path(self.data_path))
        clean_cache(directory=data_directory)

        # Entreno del modelo
        if task_config.train:
            results["train"] = yolo_train(
                model=self.model,
                data_path=self.data_path,
                output_path=self.output_path,
                config=train_config,
            )

        # Validación del modelo
        if task_config.val:
            results["val"] = yolo_validate(
                model=self.model,
                data_path=self.data_path,
                output_path=self.output_path,
                config=validation_config,
            )

        # Predicción con el modelo
        if task_config.predict:
            if predict_config.source is None:
                raise RuntimeError(
                    "Configuración inválida: 'predict.source' es obligatorio "
                    "cuando 'task.predict' es true."
                )

            results["predict"] = yolo_predict(
                model=self.model,
                source=predict_config.source,
                output_path=self.output_path,
                config=predict_config,
            )

        # Muestra de resultados
        logger.info("Resultados del pipeline:")
        for stage, result in results.items():
            logger.info(f" - {stage}: {result}")

        return results
