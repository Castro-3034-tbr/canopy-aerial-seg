"""Constantes compartidas usadas por la aplicacion."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/config.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILENAME = "train.log"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = "INFO"
