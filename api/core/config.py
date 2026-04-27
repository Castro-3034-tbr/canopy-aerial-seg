"""
Carga la configuración de la aplicación desde un archivo JSON, validando su estructura y tipos.
"""

from pathlib import Path
from pydantic import ValidationError

from common.loader_config import load_json
from api.core.types import AppConfig


def load_api_config(path: str | Path) -> AppConfig:
    """
    Carga y valida la configuración de la API.

    :param path: Ruta del archivo de configuración.
    :return: Configuración tipada.
    """
    data = load_json(path)

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Error de validacion en API config: {exc}"
        ) from exc