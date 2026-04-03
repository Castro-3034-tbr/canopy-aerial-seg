"""Gestion del ciclo de vida de los procesos asociados a streams."""

from __future__ import annotations

import logging
import multiprocessing
from uuid import uuid4

from fastapi import HTTPException

from src.core.constants import (
    DEFAULT_MODEL_DEVICE,
    DEFAULT_MODEL_PATH,
    PROCESS_JOIN_TIMEOUT,
    STREAM_START_SUCCESS_MESSAGE,
    STREAM_STOP_SUCCESS_MESSAGE,
    STREAMS_STOP_SUCCESS_MESSAGE,
)
from src.core.data_init import init_project_data, init_shared_data
from src.processes.processor_process import processor_process
from src.processes.reader_process import reader_process

logger = logging.getLogger(__name__)


class StreamManager:
    """Manager del ciclo de vida de los procesos de lectura e inferencia de cada stream,
    incluyendo la creacion de los datos compartidos, arranque y parada de los procesos,
    y el mantenimiento del estado de cada stream activo."""

    def __init__(
        self,
        manager,
        model_config: dict,
        save_path_config: dict,
        runtime_state,
        yolo_model: object | None = None,
    ) -> None:
        """Inicializa el StreamManager con la configuración del modelo, las rutas de guardado y el estado de ejecución compartido.

        Args:
            manager (Manager): Instancia del Manager para la creación de datos compartidos entre procesos.
            model_config (dict): Configuración del modelo, incluyendo la ruta al modelo YOLO y el dispositivo de inferencia.
            save_path_config (dict): Configuración de las rutas de guardado para logs e inferencias visuales.
            runtime_state (Manager.Namespace): Estado de ejecución compartido a nivel de aplicación, incluyendo el número de streams activos.
            yolo_model (object | None, optional): Instancia del modelo de inferencia YOLO para ser reutilizada en los procesos. Defaults to None.
        """

        self.manager = manager
        self.model_config = model_config
        self.save_path_config = save_path_config
        self.runtime_state = runtime_state
        self.sessions: dict[str, dict] = {}
        self.yolo_model = yolo_model

    def start(
        self,
        stream_id: str | None,
        rtsp_url: str,
        save_log: bool,
        save_inference: bool,
        confidence_threshold: float,
        mqtt_broker: str,
        mqtt_port: int,
        mqtt_topic: str,
    ) -> dict:
        """Inicio de los procesos de lectura e inferencia para un nuevo stream

        Args:
            stream_id (str | None): ID opcional para la sesión del stream; si no se proporciona, se genera uno automáticamente.
            rtsp_url (str): URL de la fuente de video RTSP.
            save_log (bool): Indicador para guardar logs.
            save_inference (bool): Indicador para guardar inferencias visuales.
            confidence_threshold (float): Umbral de confianza para filtrar detecciones.
            mqtt_broker (str): Dirección del broker MQTT.
            mqtt_port (int): Puerto del broker MQTT.
            mqtt_topic (str): Tema del broker MQTT.

        Raises:
            HTTPException: Si ya existe una sesión con el stream_id proporcionado.
            HTTPException: Si no se pudieron iniciar los procesos del stream.
            HTTPException: Si los procesos del stream no quedaron activos tras el arranque.

        Returns:
            dict: Información sobre la sesión del stream iniciada, incluyendo el stream_id, estado, URL de RTSP y configuración MQTT.
        """
        # Si no se proporciona un stream_id, se genera uno automaticamente
        session_id = stream_id or f"stream-{uuid4().hex[:8]}"
        if session_id in self.sessions:
            raise HTTPException(
                status_code=409,
                detail=f"Ya existe una sesión con stream_id '{session_id}'.",
            )

        # Inicialización de los datos compartidos y de los procesos de lectura e inferencia para el nuevo stream
        shared_data = init_shared_data(self.manager)
        project_data = init_project_data(
            self.manager,
            self.model_config.get("Path", DEFAULT_MODEL_PATH),
            self.model_config.get("Device", DEFAULT_MODEL_DEVICE),
            self.save_path_config.get("Logs", "logs/"),
            self.save_path_config.get("Inference", "inference/"),
        )
        project_data.reader_process_running.set()
        project_data.processor_process_running.set()
        project_data.stream_id = session_id

        # Construccion del proceso de lectura
        reader = multiprocessing.Process(
            target=reader_process,
            args=(shared_data, project_data, rtsp_url),
            name=f"ReaderProcess-{session_id}",
            daemon=True,
        )

        # Construccion del proceso de inferencia
        processor = multiprocessing.Process(
            target=processor_process,
            args=(
                shared_data,
                project_data,
                save_log,
                save_inference,
                confidence_threshold,
                mqtt_broker,
                mqtt_port,
                mqtt_topic,
                self.yolo_model,
            ),
            name=f"ProcessorProcess-{session_id}",
            daemon=True,
        )

        try:
            # Arranque de los procesos de lectura e inferencia
            reader.start()
            processor.start()
        except Exception as exc:
            # Si ocurre cualquier excepción durante el arranque de los procesos, se asegura que ambos procesos se detengan y se lanza una HTTPException con el error.
            project_data.reader_process_running.clear()
            project_data.processor_process_running.clear()
            logger.exception(
                "No se pudieron iniciar los procesos del stream_id=%s rtsp_url=%s",
                session_id,
                rtsp_url,
            )
            raise HTTPException(
                status_code=500,
                detail="No se pudieron iniciar los procesos del stream.",
            ) from exc

        if not reader.is_alive() or not processor.is_alive():
            # Si alguno de los procesos no quedó activo tras el arranque, se asegura que ambos procesos se detengan y se lanza una HTTPException indicando el error.
            project_data.reader_process_running.clear()
            project_data.processor_process_running.clear()
            raise HTTPException(
                status_code=500,
                detail="Los procesos del stream no quedaron activos tras el arranque.",
            )

        # Guardado de la informacion de la sesion del stream
        self.sessions[session_id] = {
            "stream_id": session_id,
            "state": "running",
            "rtsp_url": rtsp_url,
            "shared_data": shared_data,
            "project_data": project_data,
            "reader_process": reader,
            "processor_process": processor,
            "mqtt": {
                "broker": mqtt_broker,
                "port": mqtt_port,
                "topic": mqtt_topic,
            },
        }
        self.runtime_state.active_streams = len(self.sessions)

        return {
            "msg": STREAM_START_SUCCESS_MESSAGE,
            "stream_id": session_id,
            "state": "running",
            "rtsp_url": rtsp_url,
            "mqtt": self.sessions[session_id]["mqtt"],
        }

    def stop(
        self,
        stream_id: str | None = None,
        timeout: float = PROCESS_JOIN_TIMEOUT,
    ) -> dict:
        """Parada de los procesos de lectura e inferencia para un stream específico o para todos los streams activos.

        Args:
            stream_id (str | None, optional): ID del stream a detener. Defaults to None.
            timeout (float, optional): Tiempo máximo de espera para la finalización de los procesos. Defaults to PROCESS_JOIN_TIMEOUT.

        Raises:
            HTTPException: Si no se puede detener el stream.

        Returns:
            dict: Información sobre el stream detenido.
        """

        # Si no se proporciona un stream_id, se detienen todos los streams activos
        if stream_id is None:

            # Detención de todos los streams activos utilizando el método _stop_one para cada sesión, y construcción de la respuesta con la información de los streams detenidos.
            stopped_streams = [
                self._stop_one(session_id, timeout)
                for session_id in list(self.sessions)
            ]
            return {
                "msg": STREAMS_STOP_SUCCESS_MESSAGE,
                "stopped": stopped_streams,
            }

        # Si no se encuentra una sesión con el stream_id proporcionado, se lanza una HTTPException indicando que no existe la sesión.
        if stream_id not in self.sessions:
            raise HTTPException(
                status_code=404,
                detail=f"No existe ninguna sesión con stream_id '{stream_id}'.",
            )

        # Parada del stream específico utilizando el método _stop_one.
        stopped = self._stop_one(stream_id, timeout)
        return {
            "msg": STREAM_STOP_SUCCESS_MESSAGE,
            "stream_id": stopped["stream_id"],
            "state": stopped["state"],
        }

    def health(self) -> dict:
        """Proporciona información sobre el estado de los streams activos,
        incluyendo el número de streams activos, el estado de cada stream,
        la URL de RTSP y el estado de los procesos de lectura e inferencia.

        Returns:
            dict: Información sobre el estado de los streams activos, incluyendo el número de streams activos, el estado de cada stream, la URL de RTSP y el estado de los procesos de lectura e inferencia.
        """
        return {
            "active_streams": len(self.sessions),
            "streams": [
                {
                    "stream_id": session_id,
                    "state": session["state"],
                    "rtsp_url": session["rtsp_url"],
                    "reader_alive": bool(session["reader_process"].is_alive()),
                    "processor_alive": bool(session["processor_process"].is_alive()),
                }
                for session_id, session in self.sessions.items()
            ],
        }

    def _stop_one(self, stream_id: str, timeout: float) -> dict:
        """Parada de los procesos de lectura e inferencia para un stream especifico,
        incluyendo la señalización a los procesos para que se detengan, la espera de su finalización con un timeout,
        y la eliminación de la sesión del stream.

        Args:
            stream_id (str): ID del stream a detener
            timeout (float): Tiempo máximo de espera para la finalización de los procesos

        Returns:
            dict: Información sobre el stream detenido
        """

        # Obtencion de la sesion del stream
        session = self.sessions[stream_id]
        session["state"] = "stopping"
        session["project_data"].reader_process_running.clear()
        session["project_data"].processor_process_running.clear()

        # Espera para la finalizacion de los procesos
        for process in (
            session["reader_process"],
            session["processor_process"],
        ):
            process.join(timeout=timeout)

            # Si el proceso no termina a tiempo, se fuerza su terminacion
            if process.is_alive():
                logging.warning(
                    "El proceso %s del stream_id=%s no terminó a tiempo; se fuerza terminate().",
                    process.name,
                    stream_id,
                )
                process.terminate()
                process.join(timeout=timeout)

        # Eliminacion de la sesion del stream
        del self.sessions[stream_id]
        self.runtime_state.active_streams = len(self.sessions)

        return {"stream_id": stream_id, "state": "stopped"}
