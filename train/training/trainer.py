from __future__ import annotations

from pathlib import Path

from train.core.types import TrainConfig, YoloModel, YoloTaskResult


def yolo_train(
    model: YoloModel,
    data_path: str,
    output_path: str,
    config: TrainConfig,
) -> YoloTaskResult:
    """Ejecuta el entrenamiento de un modelo YOLO.

    Args:
        model: Instancia del modelo YOLO (ultralytics).
        data_path (str): Ruta al archivo data.yaml.
        output_path (str): Directorio base de salida.
        config (TrainConfig): Configuración validada de entrenamiento.

    Returns:
        YoloTaskResult: Resultado devuelto por Ultralytics.

    Raises:
        RuntimeError: Si falla el entrenamiento.
    """

    # Creación de argumentos para el entrenamiento
    train_args: dict[str, object] = {
        "data": data_path,
        "epochs": config.epochs,
        "batch": config.batch_size,
        "imgsz": config.img_size,
        "device": config.device,
        "save_period": config.save_period,
        "seed": config.seed,
        "patience": config.patience,
        "workers": config.workers,
        "name": str(Path(output_path) / "train"),
        "mosaic": config.mosaic,
    }

    if not config.augmentation:
        # Desactiva augmentations explícitamente cuando lo pide la config.
        train_args.update(
            {
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
            }
        )

    # Ejecución del entrenamiento
    try:
        results = model.train(**train_args)
        return results

    except Exception as exc:
        raise RuntimeError(
            f"Error durante el entrenamiento YOLO: {exc}"
        ) from exc
