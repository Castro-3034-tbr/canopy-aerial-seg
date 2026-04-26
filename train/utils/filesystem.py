"""Utilidades para gestion de sistema de ficheros."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from train.core.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


def resolve_path(value: str | None, *, default: Path | None = None) -> Path:
    """
    Resuelve una ruta absoluta o relativa respecto a la raiz del proyecto.

    Args:
        value (str | None): Ruta recibida desde configuracion.
        default (Path | None): Ruta por defecto si ``value`` no viene
            informada.

    Returns:
        Path: Ruta normalizada.
    """
    if value is None:
        if default is None:
            raise ValueError(
                "Se esperaba una ruta valida en la configuracion."
            )
        return default

    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_cache_files(directory: str) -> List[str]:
    """
    Busca recursivamente archivos .cache en un directorio.

    Args:
        directory (str): Directorio raíz donde buscar.

    Returns:
        List[str]: Lista de rutas completas a archivos .cache.
    """
    cache_paths: List[str] = []

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".cache"):
                full_path = os.path.join(dirpath, filename)
                cache_paths.append(full_path)

    return cache_paths


def clean_cache(directory: str) -> None:
    """
    Elimina todos los archivos .cache de un directorio recursivamente.

    Args:
        directory (str): Directorio raíz donde limpiar.
    """
    logger.info(f"Buscando archivos de cache en: {directory}")

    # Busqueda de archivos de cache
    cache_paths = find_cache_files(directory=directory)

    if not cache_paths:
        logger.info("No se encontraron archivos de cache.")
        return

    removed_count = 0

    # Eliminación de archivos de cache encontrados
    for path in cache_paths:
        try:
            os.remove(path)
            removed_count += 1
            logger.debug(f"Cache eliminado: {path}")
        except OSError as exc:
            logger.error(f"Error eliminando {path}: {exc}")

    logger.info(f"Archivos de cache eliminados: {removed_count}")

    # Verificación final
    remaining_cache = find_cache_files(directory=directory)

    if remaining_cache:
        logger.warning(
            f"Quedan {len(remaining_cache)} archivos de cache sin eliminar."
        )
    else:
        logger.info("Limpieza de cache completada correctamente.")
