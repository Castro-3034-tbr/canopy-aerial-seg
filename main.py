"""Punto de entrada de la aplicacion."""

from __future__ import annotations

import uvicorn

from src.core.config import load_config
from src.core.constants import DEFAULT_API_HOST_KEY, DEFAULT_API_PORT_KEY
from src.core.dependencies import create_application

# Expone una app ligera; el runtime pesado se construye en startup.
app = create_application()


def main() -> None:
    """Inicia Uvicorn con la configuracion declarada en disco."""
    # Obtiene la configuracion de la API desde el archivo principal.
    api_config = load_config().get("API", {})
    api_host = api_config.get(DEFAULT_API_HOST_KEY)
    api_port = api_config.get(DEFAULT_API_PORT_KEY)

    if not api_host or not api_port:
        raise ValueError(
            "IP o puerto de la API no especificados en la configuracion."
        )

    # Crea la aplicacion en modo factory para evitar side effects al importar.
    uvicorn.run("main:create_application", factory=True, host=api_host, port=api_port)


if __name__ == "__main__":
    main()
