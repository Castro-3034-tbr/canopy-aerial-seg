from __future__ import annotations

from typing import Dict, Any


def yolo_validate(
    model,
    data_path: str,
    output_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ejecuta la validación de un modelo YOLO.

    Args:
        model: Instancia del modelo YOLO (ultralytics).
        data_path (str): Ruta al archivo data.yaml.
        output_path (str): Directorio base de salida.
        config (Dict[str, Any]): Configuración de validación.

    Returns:
        Dict[str, Any]: Resultados de la validación.

    Raises:
        RuntimeError: Si falla la validación.
    """

    # Creación de argumentos para la validación
    val_args: Dict[str, Any] = {
        "data": data_path,
        "batch": config.get("batch_size", config.get("batch", 16)),
        "imgsz": config.get("img_size", config.get("imgsz", 640)),
        "device": config.get("device", "cpu"),
        "workers": config.get("workers", 4),
        "name": f"{output_path}/validation",
    }

    # Ejecución de la validación
    try:
        results = model.val(**val_args)
        return results

    except Exception as exc:
        raise RuntimeError(f"Error durante la validación YOLO: {exc}") from exc