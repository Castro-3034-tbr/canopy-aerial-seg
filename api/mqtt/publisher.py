"""Serialización y publicación de payloads MQTT."""

from __future__ import annotations

import json
import logging
from typing import Any

import paho.mqtt.client as mqtt

from api.core.types import PahoMQTTClient

logger = logging.getLogger(__name__)


def publish_message(
    client: PahoMQTTClient,
    payload: dict[str, Any],
    frame_id: int | None = None,
) -> None:
    """Publica un lote de detecciones serializado en el topic configurado."""

    #TODO: Ajustarlo
    topic = getattr(client, "topic", None)
    broker = getattr(client, "broker", "desconocido")
    port = getattr(client, "port", "desconocido")

    if not topic:
        logger.error(
            "No se puede publicar MQTT porque el cliente no tiene topic. frame_id=%s",
            frame_id,
        )
        return

    try:
        result = client.publish(
            topic,
            json.dumps(payload),
        )
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning(
                "No se pudo publicar el mensaje MQTT en broker=%s puerto=%s "
                "topic=%s frame_id=%s error=%s",
                broker,
                port,
                topic,
                frame_id,
                mqtt.error_string(result.rc),
            )
            return

        logger.debug(
            "Mensaje MQTT publicado en topic=%s frame_id=%s",
            topic,
            frame_id,
        )
    except Exception:
        logger.exception(
            "Error al publicar MQTT en broker=%s puerto=%s topic=%s frame_id=%s",
            broker,
            port,
            topic,
            frame_id,
        )
