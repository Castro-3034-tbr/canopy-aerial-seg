from __future__ import annotations

from src.core.types import PredictConfig, YoloModel, YoloTaskResult

from pathlib import Path


def yolo_predict(
    model: YoloModel,
    source: str,
    output_path: str,
    config: PredictConfig,
) -> YoloTaskResult:
    """Ejecuta inferencia con un modelo YOLO.

    Args:
        model: Instancia del modelo YOLO (ultralytics).
        source (str): Ruta de entrada (imagen, vídeo o directorio).
        output_path (str): Directorio base de salida.
        config (PredictConfig): Configuración validada de inferencia.

    Returns:
        YoloTaskResult: Resultado devuelto por Ultralytics.

    Raises:
        RuntimeError: Si falla la inferencia.
    """

    # Creación de argumentos para la inferencia
    predict_args: dict[str, str | float | int | bool] = {
        "source": source,
        "conf": config.conf_threshold,
        "imgsz": config.img_size,
        "device": config.device,
        "save": config.save_results,
        "save_txt": config.save_txt,
        "save_conf": config.save_conf,
        "name": str(Path(output_path) / "detect"),
        "exist_ok": config.exist_ok,
    }
    # Cast to proper types to satisfy type checker
    _: dict[str, str | float | int | bool] = {
        k: v for k, v in predict_args.items()
    }

    # Ejecución de la inferencia
    try:
        results = model.predict(**predict_args)
        return results

    except Exception as exc:
        raise RuntimeError(f"Error durante la inferencia YOLO: {exc}") from exc
