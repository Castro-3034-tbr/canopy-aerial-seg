"""
loader.py

Carga genérica de archivos JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Carga un archivo JSON y devuelve un diccionario.

    :param path: Ruta al archivo JSON.
    :raises FileNotFoundError: Si el archivo no existe.
    :raises ValueError: Si el JSON es inválido o no es un objeto.
    :return: Diccionario con los datos cargados.
    """
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Archivo de configuracion no encontrado: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Error al parsear el archivo JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Formato invalido: se esperaba un objeto JSON."
        )

    return data