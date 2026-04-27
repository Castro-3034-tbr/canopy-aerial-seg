"""
config.py (training)
"""

from pathlib import Path
from pydantic import ValidationError

from common.loader_config import load_json
from train.core.types import AppConfig


def load_training_config(path: str | Path) -> AppConfig:
    """
    Carga y valida la configuración de entrenamiento.

    :param path: Ruta del archivo de configuración.
    :return: Configuración tipada.
    """
    data = load_json(path)

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Error de validacion en training config: {exc}"
        ) from exc