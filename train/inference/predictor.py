from __future__ import annotations

import logging

import torch

from common.types.model import YoloModel
from train.core.types import PredictConfig, YoloResult

logger = logging.getLogger(__name__)


def yolo_predict(
    model: YoloModel,
    source: str,
    output_path: str,
    config: PredictConfig,
) -> YoloResult:
    """Ejecuta inferencia con un modelo YOLO.

    Args:
        model: Instancia del modelo YOLO (ultralytics).
        source (str): Ruta de entrada (imagen, vídeo o directorio).
        output_path (str): Directorio base de salida.
        config (PredictConfig): Configuración validada de inferencia.

    Returns:
        YoloResult: Resultado devuelto por Ultralytics.

    Raises:
        RuntimeError: Si falla la inferencia.
    """

    def _predict(device: str) -> YoloResult:
        return model.predict(
            source=source,
            conf=config.conf_threshold,
            imgsz=config.img_size,
            device=device,
            save=config.save_results,
            save_txt=config.save_txt,
            save_conf=config.save_conf,
            project=output_path,
            name="detect",
            exist_ok=config.exist_ok,
        )

    # Ejecución de la inferencia
    try:
        return _predict(device=config.device)
    except torch.cuda.OutOfMemoryError as exc:
        if config.device.lower() == "cpu":
            raise RuntimeError(f"Error durante la inferencia YOLO: {exc}") from exc

        logger.warning(
            "OOM en CUDA durante la inferencia con guardado de resultados. "
            "Se reintentara en CPU."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            return _predict(device="cpu")
        except Exception as retry_exc:
            raise RuntimeError(
                "Error durante la inferencia YOLO tras reintentar en CPU: "
                f"{retry_exc}"
            ) from retry_exc

    except Exception as exc:
        raise RuntimeError(f"Error durante la inferencia YOLO: {exc}") from exc
