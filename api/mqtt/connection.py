"""Creación y cierre del cliente MQTT del proyecto.

Proporciona funciones para inicializar el cliente Paho MQTT y detenerlo de
forma segura al terminar el proceso.
"""

from __future__ import annotations

import logging

import paho.mqtt.client as mqtt

from api.core.types import MQTTConfig, PahoMQTTClient
from typing import cast

logger = logging.getLogger(__name__)


def connect_mqtt(
    config: MQTTConfig,
) -> PahoMQTTClient | None:
    """Inicializa y arranca un cliente MQTT basado en Paho.

    Args:
        config (MQTTConfig): Configuración del broker MQTT (host, port, topic, client_id, keepalive).

    Returns:
        PahoMQTTClient | None: Cliente MQTT inicializado y con el loop de red
            arrancado, o `None` si la inicialización falló.

    Notes:
        En caso de error la función registra la excepción y devuelve `None`.
    """
    try:
        logger.info(
            "Inicializando cliente MQTT con broker=%s puerto=%s topic=%s client_id=%s",
            config.host,
            config.port,
            config.topic,
            config.client_id,
        )
        
        # Configuramos el cliente MQTT y sus callbacks.
        client = mqtt.Client(client_id=config.client_id)
        
        # Conectamos de forma asíncrona para no bloquear el proceso principal. El loop de red se iniciará en el callback de conexión.
        client.connect_async(
            host=config.host,
            port=config.port,
            keepalive=config.keepalive,
        )
        # Bind de callbacks con el topic incluido en el closure para que puedan usarlo.
        client.loop_start()
        return cast(PahoMQTTClient, client)
    except Exception:
        logger.exception(
            "No se pudo inicializar MQTT con broker=%s puerto=%s topic=%s. "
            "El procesado continuara sin MQTT.",
            config.host,
            config.port,
            config.topic,
        )
        return None

def disconnect_mqtt(
    client: PahoMQTTClient | None,
) -> None:
    """Detiene el loop de red y desconecta el cliente MQTT de forma segura.

    Args:
        client (PahoMQTTClient | None): Cliente retornado por `connect_mqtt`.

    Returns:
        None
    """
    if client is None:
        logger.debug("El cliente MQTT no estaba inicializado.")
        return

    client.loop_stop()
    client.disconnect()
