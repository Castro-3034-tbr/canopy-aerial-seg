from __future__ import annotations

from typing import Dict, Any, Callable

from src.utils.filesystem import clean_cache
from src.training.trainer import yolo_train
from src.training.validator import yolo_validate
from src.inference.predictor import yolo_predict


class YoloPipeline:
    """
    Orquestador de tareas YOLO (train, validate, predict).

    Esta clase coordina la ejecución de distintas etapas del flujo
    sin implementar lógica específica de cada tarea.
    """

    def __init__(
        self,
        model,
        data_path: str,
        output_path: str,
        tasks: Dict[str, Callable] | None = None,
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
        self.tasks = tasks or {}

    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el pipeline según la configuración.

        Args:
            config (Dict[str, Any]): Configuración global del pipeline.

        Returns:
            Dict[str, Any]: Resultados de cada etapa ejecutada.
        """

        results: Dict[str, Any] = {}
        
        # Limpieza de cache
        clean_cache(directory=self.data_path)
        
        print("Tareas que se van a ejecutar:")
        if config.get("task", {}).get("train", False):
            print(" - Entrenamiento")
        if config.get("task", {}).get("val", False):
            print(" - Validación")
        if config.get("task", {}).get("predict", False):
            print(" - Predicción")

        # Entreno del modelo
        if config.get("task", {}).get("train", False):
            results["train"] = yolo_train(
                model=self.model,
                data_path=self.data_path,
                output_path=self.output_path,
                config=config["train"],
            )

        # Validación del modelo
        if config.get("task", {}).get("val", False):
            results["val"] = yolo_validate(
                model=self.model,
                data_path=self.data_path,
                output_path=self.output_path,
                config=config["val"],
            )

        # Predicción con el modelo
        if config.get("task", {}).get("predict", False):
            source = config["predict"].get("source", self.data_path)

            results["predict"] = yolo_predict(
                model=self.model,
                source=source,
                output_path=self.output_path,
                config=config["predict"],
            )

        return results
