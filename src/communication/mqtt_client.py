"""Cliente MQTT y utilidades de publicacion para el proyecto."""

import json
import logging
from typing import Any

import paho.mqtt.client as mqtt

from src.core.constants import DEFAULT_MQTT_KEEPALIVE

logger = logging.getLogger(__name__)


class MQTTClient:

    def __init__(
        self,
        client_id: str,
        broker: str,
        port: int,
        topic: str,
    ) -> None:
        """Inicializa el cliente MQTT y arranca el loop de red.

        Args:
            client_id (str): Identificador unico del cliente MQTT.
            broker (str): Host o IP del broker MQTT.
            port (int): Puerto TCP del broker MQTT.
            topic (str): Topic por defecto donde publicar mensajes.
        """
        self.mqtt_enabled = False
        self.mqtt_topic = topic
        self._broker = broker
        self._port = port
        self.mqtt_client: mqtt.Client | None = None

        try:
            # Inicializa el cliente y prepara los callbacks de estado.
            logger.info(
                "Inicializando cliente MQTT con broker %s:%s y topic %s",
                broker,
                port,
                topic,
            )
            self.mqtt_client = mqtt.Client(client_id=client_id)
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_disconnect = self._on_disconnect
            # La conexion se realiza en segundo plano para no bloquear.
            self.mqtt_client.connect_async(
                broker,
                port,
                keepalive=DEFAULT_MQTT_KEEPALIVE,
            )
            self.mqtt_client.loop_start()
        except Exception as exc:
            logger.warning(
                "No se pudo inicializar MQTT: %s. " "El procesado continuara sin MQTT.",
                exc,
            )
            self.mqtt_enabled = False

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: dict,
        rc: int,
        properties: Any = None,
    ) -> None:
        """Gestiona la conexion del cliente MQTT."""
        del client, userdata, flags, properties

        if rc == 0:
            # Activa la publicacion solo si la conexion fue correcta.
            self.mqtt_enabled = True
            logger.info(
                "MQTT conectado a %s:%s con topic %s",
                self._broker,
                self._port,
                self.mqtt_topic,
            )
            return

        logger.warning("Fallo de conexion MQTT con codigo %s", rc)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        rc: int,
        properties: Any = None,
    ) -> None:
        """Gestiona la desconexion del cliente MQTT."""
        del client, userdata, properties

        # Cualquier desconexion invalida temporalmente la publicacion.
        self.mqtt_enabled = False
        if rc != 0:
            logger.warning("MQTT se ha desconectado de forma inesperada")

    def publish(self, payload: dict) -> None:
        """Publica un mensaje JSON en el topic configurado."""
        # Evita intentar publicar cuando el cliente aun no esta operativo.
        if not self.mqtt_enabled or self.mqtt_client is None:
            logger.warning(
                "El cliente MQTT no esta disponible. " "No se publicara el mensaje."
            )
            return

        try:
            # Serializa el payload a JSON antes de enviarlo al broker.
            payload_json = json.dumps(payload)
            result = self.mqtt_client.publish(self.mqtt_topic, payload_json)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(
                    "No se pudo publicar el mensaje MQTT: %s",
                    mqtt.error_string(result.rc),
                )
                return

            logger.info(
                "Mensaje MQTT publicado correctamente en %s",
                self.mqtt_topic,
            )
        except Exception as exc:
            logger.warning("Error al publicar el mensaje MQTT: %s", exc)

    def disconnect(self) -> None:
        """Detiene el loop de red y desconecta el cliente MQTT."""
        if self.mqtt_client is None:
            logger.warning("El cliente MQTT no estaba inicializado.")
            return

        # Detiene el loop en segundo plano y libera la conexion activa.
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        self.mqtt_enabled = False
