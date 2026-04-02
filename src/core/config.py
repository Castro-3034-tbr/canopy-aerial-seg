from __future__ import annotations

import json
from pathlib import Path

from src.core.constants import DEFAULT_CONFIG_PATH


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Carga la configuracion de la aplicacion desde un archivo JSON.

    Args:
        path (str | Path, optional): Ruta del archivo de configuracion.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el JSON no es valido o no contiene un objeto.

    Returns:
        dict: Configuracion cargada desde disco.
    """
    # Normaliza la ruta para trabajar siempre con objetos Path.
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Archivo de configuracion no encontrado: {config_path}"
        )

    try:
        # Lee y parsea el JSON principal de configuracion.
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        # Convierte el error de parseo a una excepcion mas expresiva.
        raise ValueError(
            f"Error al parsear el archivo de configuracion: {exc}"
        ) from exc

    if not isinstance(config, dict):
        # La aplicacion espera un objeto JSON en la raiz del archivo.
        raise ValueError(
            "Formato de configuracion invalido: se esperaba un objeto JSON."
        )

    return config
