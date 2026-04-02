import uvicorn
from src.core.config import load_config
from src.core.constants import DEFAULT_API_HOST_KEY, DEFAULT_API_PORT_KEY
from src.core.dependencies import create_application

# Creacion de la aplicación FastAPI utilizando las dependencias definidas en el módulo de dependencias
app = create_application()

if __name__ == "__main__":

    #Obtencion de la configuracion de la API
    api_config = load_config().get("API", {})
    api_ip = api_config.get(DEFAULT_API_HOST_KEY)
    api_port = api_config.get(DEFAULT_API_PORT_KEY)
    if not api_ip or not api_port:
        raise ValueError("IP o puerto de la API no especificados en la configuración")

    #Inicio del servidor API utilizando Uvicorn, escuchando en la IP y puerto especificados en la configuración
    uvicorn.run(app, host=api_ip, port=api_port)
