"""Utilidades compartidas para gestion de rutas y ficheros."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from common.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


def resolve_path(value: str | Path | None, *, default: Path | None = None) -> Path:
    """Resuelve una ruta absoluta o relativa respecto a la raiz del proyecto."""
    if value is None:
        if default is None:
            raise ValueError("Se esperaba una ruta valida en la configuracion.")
        return default

    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_output_dir(
    value: str | Path | None,
    *,
    default: Path | None = None,
) -> Path:
    """Resuelve y crea el directorio de salida si no existe."""
    output_path = resolve_path(value=value, default=default)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def find_cache_files(directory: str | Path) -> list[str]:
    """Busca recursivamente archivos ``.cache`` dentro de ``directory``."""
    cache_paths: list[str] = []

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".cache"):
                cache_paths.append(os.path.join(dirpath, filename))

    return cache_paths


def clean_cache(directory: str | Path) -> None:
    """Elimina todos los archivos ``.cache`` de un directorio recursivamente."""
    logger.info("Buscando archivos de cache en: %s", directory)
    cache_paths = find_cache_files(directory=directory)

    if not cache_paths:
        logger.info("No se encontraron archivos de cache.")
        return

    removed_count = 0
    for path in cache_paths:
        try:
            os.remove(path)
            removed_count += 1
            logger.debug("Cache eliminado: %s", path)
        except OSError as exc:
            logger.error("Error eliminando %s: %s", path, exc)

    logger.info("Archivos de cache eliminados: %s", removed_count)

    remaining_cache = find_cache_files(directory=directory)
    if remaining_cache:
        logger.warning(
            "Quedan %s archivos de cache sin eliminar.",
            len(remaining_cache),
        )
    else:
        logger.info("Limpieza de cache completada correctamente.")
