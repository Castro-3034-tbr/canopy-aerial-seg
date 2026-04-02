from __future__ import annotations

import json
from pathlib import Path

from src.core.constants import DEFAULT_CONFIG_PATH


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Carga la configuracion de la aplicacion desde un archivo JSON, validando su existencia y formato.

    Args:
        path (str | Path, optional): Ruta del archivo de configuración. Defaults to DEFAULT_CONFIG_PATH.

    Raises:
        FileNotFoundError: Si el archivo de configuración no se encuentra.
        ValueError: Si el archivo de configuración tiene un formato inválido.

    Returns:
        dict: El diccionario con la configuración cargada.
    """

    #Validación de la existencia del archivo de configuración
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Archivo de configuración no encontrado: {config_path}"
        )

    try:
        #Lectura y parseo del archivo de configuración JSON, con manejo de errores para formato inválido
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        #Error al parsear el archivo de configuracion
        raise ValueError(
            f"Error al parsear el archivo de configuración: {exc}"
        ) from exc

    if not isinstance(config, dict):
        #
        raise ValueError(
            "Formato de configuración inválido: se esperaba un objeto JSON."
        )

    return config
