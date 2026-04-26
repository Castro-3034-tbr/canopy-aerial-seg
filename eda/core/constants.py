# trunk-ignore-all(isort)
# trunk-ignore-all(black)
"""Constantes compartidas usadas por la aplicacion."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Ruta raiz del proyecto.

# Configuracion base de archivos y logging.
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"                             # Directorio donde se guardan los logs.
DEFAULT_LOG_FILENAME = "EDA.log"                                    # Nombre del archivo principal de log.
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"    # Formato comun de las trazas.
DEFAULT_LOG_LEVEL = "INFO"                                          # Nivel de log por defecto.