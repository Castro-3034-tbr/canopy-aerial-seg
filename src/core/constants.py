"""Constantes compartidas usadas por la aplicacion."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Ruta raiz del proyecto.


# Configuracion general de la aplicacion.
APP_TITLE = "TFM API"                              # Titulo visible de la aplicacion FastAPI.
APP_DESCRIPTION = "API para detección de objetos con YOLOv8"  # Descripcion general de la API.
APP_VERSION = "1.0.0"                             # Version publicada de la aplicacion.
DOCS_PATH = "/docs"                               # Ruta de la documentacion interactiva.


# Configuracion base de archivos y logging.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/config.json"  # Ruta por defecto del archivo de configuracion.
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"                    # Directorio donde se guardan los logs.
DEFAULT_LOG_FILENAME = "train.log"                  # Nombre del archivo principal de log.
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # Formato comun de las trazas.
DEFAULT_LOG_LEVEL = "INFO"                        # Nivel de log por defecto.
