from __future__ import annotations

from typing import Dict, Any


def yolo_predict(
    model,
    source: str,
    output_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ejecuta inferencia (predicción) con un modelo YOLO.

    Args:
        model: Instancia del modelo YOLO (ultralytics).
        source (str): Ruta de entrada (imagen, vídeo o directorio).
        output_path (str): Directorio base de salida.
        config (Dict[str, Any]): Configuración de inferencia.

    Returns:
        Dict[str, Any]: Resultados de la predicción.

    Raises:
        RuntimeError: Si falla la inferencia.
    """

    # Creación de argumentos para la inferencia
    predict_args: Dict[str, Any] = {
        "source": source,
        "conf": config.get("conf_threshold", config.get("conf", 0.25)),
        "imgsz": config.get("img_size", config.get("imgsz", 640)),
        "device": config.get("device", "cpu"),
        "save": config.get("save_results", True),
        "save_txt": config.get("save_txt", config.get("save_results", True)),
        "save_conf": config.get("save_conf", True),
        "name": f"{output_path}/detect",
        "exist_ok": config.get("exist_ok", True),
    }

    # Ejecución de la inferencia
    try:
        results = model.predict(**predict_args)
        return results

    except Exception as exc:
        raise RuntimeError(f"Error durante la inferencia YOLO: {exc}") from exc