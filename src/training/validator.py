from __future__ import annotations

from pathlib import Path

from src.core.types import ValidationConfig, YoloModel, YoloTaskResult


def yolo_validate(
    model: YoloModel,
    data_path: str,
    output_path: str,
    config: ValidationConfig,
) -> YoloTaskResult:
    """Ejecuta la validación de un modelo YOLO.

    Args:
        model: Instancia del modelo YOLO (ultralytics).
        data_path (str): Ruta al archivo data.yaml.
        output_path (str): Directorio base de salida.
        config (ValidationConfig): Configuración validada de validación.

    Returns:
        YoloTaskResult: Resultado devuelto por Ultralytics.

    Raises:
        RuntimeError: Si falla la validación.
    """

    # Creación de argumentos para la validación
    val_args: dict[str, object] = {
        "data": data_path,
        "batch": config.batch_size,
        "imgsz": config.img_size,
        "device": config.device,
        "workers": config.workers,
        "name": str(Path(output_path) / "validation"),
    }

    # Ejecución de la validación
    try:
        results = model.val(**val_args)
        return results

    except Exception as exc:
        raise RuntimeError(f"Error durante la validación YOLO: {exc}") from exc
