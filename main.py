"""Punto de entrada de la aplicacion."""

import uvicorn

from src.core.config import load_config
from src.core.constants import DEFAULT_API_HOST_KEY, DEFAULT_API_PORT_KEY
from src.core.dependencies import create_application

# Crea la aplicacion FastAPI con todas sus dependencias registradas.
app = create_application()


if __name__ == "__main__":
    # Obtiene la configuracion de la API desde el archivo principal.
    api_config = load_config().get("API", {})
    api_host = api_config.get(DEFAULT_API_HOST_KEY)
    api_port = api_config.get(DEFAULT_API_PORT_KEY)

    if not api_host or not api_port:
        raise ValueError(
            "IP o puerto de la API no especificados en la configuracion."
        )

    # Inicia el servidor Uvicorn con la IP y el puerto configurados.
    uvicorn.run(app, host=api_host, port=api_port)
