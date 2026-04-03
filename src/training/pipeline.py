from __future__ import annotations

import logging
import os

from src.core.types import (
    PipelineConfig,
    PipelineResults,
    PredictConfig,
    TaskConfig,
    TrainConfig,
    ValidationConfig,
    YoloModel,
)
from src.inference.predictor import yolo_predict
from src.training.trainer import yolo_train
from src.training.validator import yolo_validate
from src.utils.filesystem import clean_cache

logger = logging.getLogger(__name__)


class YoloPipeline:
    """
    Orquestador de tareas YOLO (train, validate, predict).

    Esta clase coordina la ejecución de distintas etapas del flujo
    sin implementar lógica específica de cada tarea.
    """

    def __init__(
        self,
        model: YoloModel,
        data_path: str,
        output_path: str,
    ) -> None:
        """
        Args:
            model: Modelo YOLO inicializado.
            data_path (str): Ruta al dataset (data.yaml).
            output_path (str): Ruta base de salida.
        """
        self.model = model
        self.data_path = data_path
        self.output_path = output_path

    def run(self, config: PipelineConfig) -> PipelineResults:
        """
        Ejecuta el pipeline según la configuración.

        Args:
            config (PipelineConfig): Configuración global del pipeline.

        Returns:
            PipelineResults: Resultados de cada etapa ejecutada.
        """

        results: PipelineResults = {}
        task_config: TaskConfig = config.task
        train_config: TrainConfig = config.train
        validation_config: ValidationConfig = config.val
        predict_config: PredictConfig = config.predict

        logger.info("Iniciando ejecución del pipeline YOLO.")

        data_directory = os.path.dirname(self.data_path)
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
            source = predict_config.source or self.data_path

            results["predict"] = yolo_predict(
                model=self.model,
                source=source,
                output_path=self.output_path,
                config=predict_config,
            )

        # Muestra de resultados
        logger.info("Resultados del pipeline:")
        for stage, result in results.items():
            logger.info(f" - {stage}: {result}")

        return results
