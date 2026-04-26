"""Configuracion comun del sistema de logging."""

from __future__ import annotations

import logging

from eda.core.constants import (
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
)


def configure_logging(log_filename: str = DEFAULT_LOG_FILENAME) -> None:
    """Configura el sistema de logging de la aplicacion.

    Args:
        log_filename (str, optional): Nombre del archivo de log.
    """
    # Garantiza que el directorio de logs exista antes de crear handlers.
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Evita registrar handlers duplicados si el logger raiz ya esta listo.
    if logging.getLogger().handlers:
        return

    # Envia los mensajes tanto a fichero como a consola.
    logging.basicConfig(
        level=DEFAULT_LOG_LEVEL,
        format=DEFAULT_LOG_FORMAT,
        handlers=[
            logging.FileHandler(DEFAULT_LOG_DIR / log_filename),
            logging.StreamHandler(),
        ],
    )