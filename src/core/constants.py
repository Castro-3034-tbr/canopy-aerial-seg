"""Constantes compartidas usadas por la aplicacion."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Configuracion base de archivos y logging.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/config.json"           # Ruta por defecto del archivo de configuracion.
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"                             # Directorio donde se guardan los logs.
DEFAULT_LOG_FILENAME = "train.log"                                  # Nombre del archivo principal de log.
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"    # Formato comun de las trazas.
DEFAULT_LOG_LEVEL = "INFO"                                          # Nivel de log por defecto.