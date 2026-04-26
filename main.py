"""Punto de entrada para ejecutar el pipeline de entrenamiento YOLO."""

from __future__ import annotations

import logging

from ultralytics import YOLO

from src.core.config import load_config
from src.core.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from src.core.logger import configure_logging
from src.training.pipeline import YoloPipeline
from src.utils.filesystem import resolve_path

logger = logging.getLogger(__name__)


def main() -> int:
    """CLI minima para lanzar el pipeline desde terminal."""

    # Configuracion de los elementos necesarios
    configure_logging()
    config = load_config(path=DEFAULT_CONFIG_PATH)

    # Resolucion de rutas
    data_path = resolve_path(value=config.pathData)
    output_path = resolve_path(
        config.pathResult,
        default=PROJECT_ROOT / "results",
    )
    model_path = resolve_path(value=config.model.path)

    logger.info(
        "Rutas cargadas - Data: %s, Output: %s, Model: %s",
        data_path,
        output_path,
        model_path,
    )

    # Carga del modelo
    try:
        model = YOLO(model=model_path)
        logger.info(f"Modelo YOLO cargado exitosamente desde {model_path}")
    except Exception as exc:
        logger.exception(
            "Error al cargar el modelo YOLO desde %s: %s",
            model_path,
            exc,
        )
        return 1

    # Creacion de la instancia del pipeline para ejecutarla completa
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
