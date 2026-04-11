"""Creacion y cierre del cliente MQTT del proyecto."""

from __future__ import annotations

import logging

import paho.mqtt.client as mqtt

from src.core.constants import DEFAULT_MQTT_KEEPALIVE
from src.core.types import PahoMQTTClient
from typing import cast

logger = logging.getLogger(__name__)


def connect_mqtt(
    client_id: str,
    broker: str,
    port: int,
    topic: str,
    keepalive: int = DEFAULT_MQTT_KEEPALIVE,
) -> PahoMQTTClient:
    """Inicializa un cliente MQTT y devuelve cliente, config y estado."""
    try:
        logger.info(
            "Inicializando cliente MQTT con broker=%s puerto=%s topic=%s client_id=%s",
            broker,
            port,
            topic,
            client_id,
        )
        
        # Configuramos el cliente MQTT y sus callbacks.
        client = mqtt.Client(client_id=client_id)

        # Guardamos la configuracion en el objeto para que el publicador y los logs
        # puedan acceder a ella sin depender de variables externas.
        client.broker = broker
        client.port = port
        client.topic = topic
        client.keepalive = keepalive
        
        client.connect_async(
            broker,
            port,
            keepalive=keepalive,
        )
        # Bind de callbacks con el topic incluido en el closure para que puedan usarlo.
        client.loop_start()
        return cast(PahoMQTTClient, client)
    except Exception:
        logger.exception(
            "No se pudo inicializar MQTT con broker=%s puerto=%s topic=%s. "
            "El procesado continuara sin MQTT.",
            broker,
            port,
            topic,
        )


def disconnect_mqtt(
    client: PahoMQTTClient | None,
) -> None:
    """Detiene el loop de red y desconecta el cliente MQTT."""
    if client is None:
        logger.debug("El cliente MQTT no estaba inicializado.")
        return

    client.loop_stop()
    client.disconnect()
