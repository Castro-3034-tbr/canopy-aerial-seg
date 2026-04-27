"""Constantes compartidas usadas por la aplicacion."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ================================
# Configuracion de rutas y logging
# ================================

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILENAME = "tfm.log"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = "INFO"
