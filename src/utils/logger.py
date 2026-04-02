from __future__ import annotations

import logging
from src.core.constants import (
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
)


def configure_logging(log_filename: str = DEFAULT_LOG_FILENAME) -> None:
    """Configura el sistema de logging para la aplicación, 
    estableciendo el nivel de log, el formato y los handlers para escribir en un archivo y en la consola.

    Args:
        log_filename (str, optional):Ruta del archivo de log. Por defecto es DEFAULT_LOG_FILENAME.
    """
    
    #Creacion del directorio de logs si no existe
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    #Comprobación de si ya hay handlers configurados para evitar configurar el logging más de una vez
    if logging.getLogger().handlers:
        return

    #Configuración del logging con el nivel, formato y handlers para escribir en un archivo y en la consola
    logging.basicConfig(
        level=DEFAULT_LOG_LEVEL,
        format=DEFAULT_LOG_FORMAT,
        handlers=[
            logging.FileHandler(DEFAULT_LOG_DIR / log_filename),
            logging.StreamHandler(),
        ],
    )
