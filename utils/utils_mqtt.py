"""
Utilidades para la comunicación MQTT
"""
import logging
import json
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTClient:

    def __init__(self, clientID, broker, port, topic):
        """Inicializa el cliente MQTT y arranca el loop de red en segundo plano.

        La conexión se realiza de forma asíncrona mediante `connect_async`, por lo que
        el estado final de conexión se notifica en los callbacks `_on_connect` y
        `_on_disconnect`.

        Args:
            clientID (str): Identificador único del cliente MQTT.
            broker (str): Host o IP del broker MQTT.
            port (int): Puerto TCP del broker MQTT.
            topic (str): Topic por defecto donde se publicarán mensajes.

        Returns:
            None
        """
        self.mqtt_enabled = False
        self.mqttTopic = topic
        self._broker = broker
        self._port = port
        self.mqttClient = None

        try:
            logger.info(f"Inicializando MQTT client con broker: {broker}:{port}, topic: {topic}")

            self.mqttClient = mqtt.Client(client_id=clientID)

            # Callbacks de estado de conexión
            self.mqttClient.on_connect = self._on_connect
            self.mqttClient.on_disconnect = self._on_disconnect

            # Non-blocking connect
            self.mqttClient.connect_async(broker, port, keepalive=60)
            self.mqttClient.loop_start()

        except Exception as e:
            logger.warning(f"Failed to setup MQTT client: {e}. Video processing will continue without MQTT.")
            self.mqtt_enabled = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Gestiona el evento de conexión del cliente MQTT.

        Args:
            client (mqtt.Client): Instancia del cliente que disparó el callback.
            userdata (Any): Datos de usuario asociados al cliente (si existen).
            flags (dict): Banderas de respuesta de conexión enviadas por el broker.
            rc (int): Código de resultado de conexión. `0` indica éxito.
            properties (Any, optional): Propiedades MQTT v5 cuando están disponibles.

        Returns:
            None
        """
        if rc == 0:
            self.mqtt_enabled = True
            logger.info(f"MQTT connected to {self._broker}:{self._port}, topic: {self.mqttTopic}")
        else:
            logger.warning(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Gestiona el evento de desconexión del cliente MQTT.

        Desactiva la publicación marcando `mqtt_enabled` como `False`.

        Args:
            client (mqtt.Client): Instancia del cliente que disparó el callback.
            userdata (Any): Datos de usuario asociados al cliente (si existen).
            rc (int): Código de desconexión. Distinto de `0` suele indicar corte inesperado.
            properties (Any, optional): Propiedades MQTT v5 cuando están disponibles.

        Returns:
            None
        """
        self.mqtt_enabled = False
        if rc != 0:
            logger.warning("MQTT disconnected unexpectedly, will attempt reconnect")
    
    def publish(self, payload: dict):
        """Publica un mensaje JSON en el topic configurado.

        Si el cliente no está habilitado o no fue inicializado, no envía el mensaje.

        Args:
            payload (dict): Estructura de datos a serializar y publicar en MQTT.

        Returns:
            None
        """

        # Verificamos si el cliente MQTT está habilitado antes de intentar publicar
        if not self.mqtt_enabled or self.mqttClient is None:
            logger.warning("MQTT client is not enabled. Message will not be published.")
            return

        try:
            # Convertimos el payload a JSON antes de publicarlo
            payloadJson = json.dumps(payload)

            # Publicamos el mensaje en el topic configurado
            result = self.mqttClient.publish(self.mqttTopic, payloadJson)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(f"Failed to publish MQTT message: {mqtt.error_string(result.rc)}")
            else:
                logger.info(f"MQTT message published successfully to topic {self.mqttTopic}")
        except Exception as e:
            logger.warning(f"Error while publishing MQTT message: {e}")

    def disconnect(self):
        """Detiene el loop de red y desconecta el cliente MQTT de forma segura.

        Si el cliente no fue inicializado, no realiza ninguna acción.

        Returns:
            None
        """

        if self.mqttClient is None:
            logger.warning("MQTT client was not initialized. No need to disconnect.")
            return

        # Detenemos el loop y desconectamos el cliente MQTT de forma segura
        self.mqttClient.loop_stop()
        self.mqttClient.disconnect()
        self.mqtt_enabled = False