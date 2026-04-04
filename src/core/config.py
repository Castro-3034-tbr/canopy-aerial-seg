"""Carga y validacion de la configuracion de la aplicacion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from src.core.constants import DEFAULT_CONFIG_PATH
from src.core.types import AppConfig, AppConfigModel


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
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

    # Valida estructura, tipos y restricciones de la configuracion.
    try:
        validated_config = AppConfigModel.model_validate(config)
        return cast(AppConfig, validated_config.model_dump())
    except ValidationError as exc:
        raise ValueError(
            f"Error de validacion en la configuracion: {exc}"
        ) from exc
