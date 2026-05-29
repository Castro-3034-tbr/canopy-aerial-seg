"""Serialización y publicación de payloads MQTT.

Contiene una función auxiliar para serializar mensajes a JSON y publicarlos
usando un cliente Paho MQTT opcional.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, cast

import paho.mqtt.client as mqtt

from api.core.types import PahoMQTTClient, MQTTConfig

logger = logging.getLogger(__name__)


def publish_message(
    client: Optional[PahoMQTTClient],
    mqtt_config: MQTTConfig,
    payload: dict[str, Any],
    qos: int = 0,
    retain: bool = False,
) -> bool:
    """Publica un mensaje serializado en el topic configurado.

    Args:
        client (Optional[PahoMQTTClient]): Cliente Paho MQTT, puede ser `None`.
        mqtt_config (MQTTConfig): Configuración del broker y topic.
        payload (dict[str, Any]): Datos serializables a publicar como JSON.
        qos (int, optional): Quality of Service de la publicación.
        retain (bool, optional): Flag de retención del mensaje en el broker.

    Returns:
        bool: True si la publicación fue aceptada por el cliente, False en caso contrario.

    Notes:
        - Si `client` es `None` la función registra y devuelve False.
        - Se registran advertencias si `publish()` devuelve un código de error.
    """

    if client is None:
        logger.debug("Cliente MQTT no inicializado; omitiendo publicación del mensaje.")
        return False

    try:
        # Serializamos de forma compacta y permitimos unicode en la carga.
        payload_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        result = cast(Any, client).publish(mqtt_config.topic, payload_str, qos, retain)

        # MQTTMessageInfo suele exponer `rc`; si no, asumimos fallo si el atributo no existe.
        rc = getattr(result, "rc", None)
        if rc is not None and rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning(
                "No se pudo publicar el mensaje MQTT en broker=%s puerto=%s topic=%s error=%s",
                mqtt_config.host,
                mqtt_config.port,
                mqtt_config.topic,
                mqtt.error_string(rc),
            )
            return False

        return True
    except Exception:
        logger.exception(
            "Error al publicar MQTT en broker=%s puerto=%s topic=%s",
            mqtt_config.host,
            mqtt_config.port,
            mqtt_config.topic,
        )
        return False
