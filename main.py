"""Punto de entrada para ejecutar el pipeline de entrenamiento YOLO."""

from __future__ import annotations

import logging

from ultralytics import YOLO

from src.core.config import load_config
from src.core.constants import PROJECT_ROOT
from src.core.logger import configure_logging
from src.training.pipeline import YoloPipeline
from src.utils.filesystem import resolve_path


logger = logging.getLogger(__name__)


def main() -> int:
    """CLI minima para lanzar el pipeline desde terminal."""

    #Configuracion de los elementos necesarios
    configure_logging()
    config = load_config("config/config.json")

    #Resolucion de rutas
    data_path = resolve_path(config.get("pathData"))
    output_path = resolve_path(
        config.get("pathResult"),
        default=PROJECT_ROOT / "results",
    )
    model_path = resolve_path(config.get("model", {}).get("path"))

    logger.info(f"Rutas cargadas - Data: {data_path}, Output: {output_path}, Model: {model_path}")

    #Carga del modelo
    try:
        model = YOLO(model_path)
        logger.info(f"Modelo YOLO cargado exitosamente desde {model_path}")
    except Exception as exc:
        logger.exception(f"Error al cargar el modelo YOLO desde {model_path}: {exc}")
        return 1

    #Creacion de la instancia del pipeline para ejecutarla completa
    pipeline = YoloPipeline(
        model=model,
        data_path=str(data_path),
        output_path=str(output_path),
    )

    try:
        results = pipeline.run(config=config)
    except Exception:
        logger.exception("Fallo durante la ejecucion del pipeline.")
        return 1

    logger.info("Pipeline ejecutado exitosamente.")
    logger.info(f"Etapas ejecutadas: {', '.join(results.keys()) or 'ninguna'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
