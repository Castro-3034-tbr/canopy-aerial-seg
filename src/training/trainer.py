from __future__ import annotations

from typing import Dict, Any


def yolo_train(
    model,
    data_path: str,
    output_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ejecuta el entrenamiento de un modelo YOLO.

    Args:
        model: Instancia del modelo YOLO (ultralytics).
        data_path (str): Ruta al archivo data.yaml.
        output_path (str): Directorio base de salida.
        config (Dict[str, Any]): Configuración de entrenamiento.

    Returns:
        Dict[str, Any]: Resultados del entrenamiento.
    
    Raises:
        ValueError: Si faltan parámetros obligatorios.
        RuntimeError: Si falla el entrenamiento.
    """

    # Validación de parámetros obligatorios
    if "epochs" not in config:
        raise ValueError("El parámetro 'epochs' es obligatorio en config.")

    # Creación de argumentos para el entrenamiento
    train_args: Dict[str, Any] = {
        "data": data_path,
        "epochs": config["epochs"],
        "batch": config.get("batch_size", config.get("batch", 16)),
        "imgsz": config.get("img_size", config.get("imgsz", 640)),
        "device": config.get("device", "cpu"),
        "save_period": config.get("save_period", -1),
        "seed": config.get("seed", 42),
        "patience": config.get("patience", 10),
        "workers": config.get("workers", 4),
        "name": f"{output_path}/train",
        "mosaic": config.get("mosaic", False),
    }
    
    if config.get("augmentation", False):
        # Desactivar augmentations explícitamente
        train_args.update({
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
        })

    # Ejecución del entrenamiento
    try:
        results = model.train(**train_args)
        return results

    except Exception as exc:
        raise RuntimeError(f"Error durante el entrenamiento YOLO: {exc}") from exc