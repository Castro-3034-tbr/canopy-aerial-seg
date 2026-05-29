"""Configuracion comun del sistema de logging."""

from __future__ import annotations

import logging

from common.constants import (
    LOG_DIR,
    LOG_FILENAME,
    LOG_FORMAT,
    LOG_LEVEL,
)


def configure_logging(log_filename: str = LOG_FILENAME) -> None:
    """Configura el sistema de logging de la aplicacion.

    Args:
        log_filename (str, optional): Nombre del archivo de log.
    """
    # Garantiza que el directorio de logs exista antes de crear handlers.
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Evita registrar handlers duplicados si el logger raiz ya esta listo.
    if logging.getLogger().handlers:
        return

    # Envia los mensajes tanto a fichero como a consola.
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_DIR / log_filename),
            logging.StreamHandler(),
        ],
    )
