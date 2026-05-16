"""Punto de entrada de la aplicacion."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from api.core.config import load_api_config
from api.core.dependencies import create_app
from api.core.types import ApiConfig
from common.constants import CONFIG_DIR
from common.logger import configure_logging

# Expone una app ligera; el runtime pesado se construye en startup.
app = create_app()


def _is_reload_enabled(api_config: ApiConfig) -> bool:
    """Determina si Uvicorn debe reiniciarse al detectar cambios."""
    reload_from_env = os.getenv("TFM_API_RELOAD")
    if reload_from_env is not None:
        return reload_from_env.strip().lower() in {"1", "true", "yes", "on"}

    return bool(getattr(api_config, "RELOAD", False))


def main() -> None:
    """Inicia Uvicorn con la configuracion declarada en disco."""

    # Configuramos el logger
    configure_logging()

    # Obtiene la configuracion de la API desde el archivo principal.
    api_config = load_api_config(Path(CONFIG_DIR) / "config_api.json").API
    api_host = api_config.IP
    api_port = api_config.PORT

    if not api_host or not api_port:
        raise ValueError("IP o puerto de la API no especificados en la configuracion.")

    api_reload = _is_reload_enabled(api_config)

    # Crea la aplicacion en modo factory para evitar side effects al importar.
    uvicorn.run(
        "run_api:create_app",
        factory=True,
        host=api_host,
        port=api_port,
        reload=api_reload,
    )


if __name__ == "__main__":
    main()
