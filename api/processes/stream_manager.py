"""Gestión del ciclo de vida de los procesos asociados a streams.

Contiene la implementación de `StreamManager` que arranca y detiene los
procesos de lectura e inferencia para cada sesión RTSP, mantiene el estado
de las sesiones y expone métodos auxiliares para salud y parada.
"""

from __future__ import annotations

import logging
import multiprocessing

from fastapi import HTTPException

from api.core.constants import (
    PROCESS_JOIN_TIMEOUT,
    STREAM_START_SUCCESS_MESSAGE,
    STREAM_STOP_SUCCESS_MESSAGE,
    STREAMS_STOP_SUCCESS_MESSAGE,
)

from api.core.data_init import init_project_data, init_shared_data
from api.processes.processor_process import processor_process
from api.processes.reader_process import reader_process

from typing import Any

from api.core.types import (
    GlobalManager,
    ModelConfig,
    MQTTConfig,
    RuntimeState,
    SavePathConfig,
    StreamSession,
)
from common.types.model import YoloModel

logger = logging.getLogger(__name__)


class StreamManager:
    """Manager del ciclo de vida de los procesos de lectura e inferencia.

    Crea los recursos compartidos por proceso, arranca y para los procesos
    `reader_process` y `processor_process`, y mantiene un registro de las
    sesiones activas en `self.sessions`.
    """

    def __init__(
        self,
        manager: GlobalManager,
        model_config: ModelConfig,
        save_path_config: SavePathConfig,
        runtime_state: RuntimeState,
        yolo_model: YoloModel,
    ) -> None:
        """Inicializa el `StreamManager`.

        Args:
            manager (GlobalManager): Instancia para crear estructuras compartidas.
            model_config (ModelConfig): Configuración del modelo (ruta, dispositivo, etc.).
            save_path_config (SavePathConfig): Rutas para guardar logs e inferencias.
            runtime_state (RuntimeState): Estado de ejecución compartido de la aplicación.
            yolo_model (YoloModel): Instancia del modelo YOLO para pasar a los procesos.
        """

        self.manager = manager
        self.model_config = model_config
        self.save_path_config = save_path_config
        self.runtime_state = runtime_state
        self.sessions: dict[str, StreamSession] = {}
        self.yolo_model = yolo_model

    def start(
        self,
        session_id: str,
        rtsp_url: str,
        save_log: bool,
        save_inference: bool,
        confidence_threshold: float,
        mqtt_config: MQTTConfig,
        overlap: tuple[float, float],
        gsd: float,
    ) -> dict[str, Any]:
        """Arranca los procesos para una nueva sesión de stream.

        Args:
            session_id (str): Identificador de la sesión.
            rtsp_url (str): URL de la fuente RTSP.
            save_log (bool): Si True, se guardan logs.
            save_inference (bool): Si True, se guardan inferencias visuales.
            confidence_threshold (float): Umbral para filtrar detecciones.
            mqtt_config (MQTTConfig): Configuración MQTT para la sesión.
            overlap (tuple[float, float]): Recorte de solapes en máscaras.
            gsd (float): Ground Sample Distance en metros/píxel.

        Returns:
            dict[str, Any]: Detalles de la sesión arrancada.

        Raises:
            HTTPException: con código 500 si falla el arranque de procesos o si
                alguno no queda activo tras el start().
        """

        # Inicialización de los datos compartidos y de los procesos de lectura e inferencia para el nuevo stream
        shared_data = init_shared_data(manager=self.manager)
        project_data = init_project_data(
            manager=self.manager,
            yolo_path=str(self.model_config.path),
            yolo_device=self.model_config.device,
            save_path_logs=str(self.save_path_config.logs),
            save_path_inference=str(self.save_path_config.inference),
        )

        # Inicialización de los flags de ejecución para los procesos de lectura e inferencia, y asignación del session_id a la sesión.
        project_data.reader_running.set()
        project_data.processor_running.set()
        project_data.session_id = session_id

        # Construccion del proceso de lectura
        reader = multiprocessing.Process(
            target=reader_process,
            args=(shared_data, project_data, session_id, rtsp_url),
            name=f"ReaderProcess-{session_id}",
            daemon=True,
        )

        # Construccion del proceso de inferencia
        processor = multiprocessing.Process(
            target=processor_process,
            args=(
                shared_data,
                project_data,
                session_id,
                save_log,
                save_inference,
                confidence_threshold,
                self.yolo_model,
                mqtt_config,
                overlap,
                gsd,
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
            project_data.reader_running.clear()
            project_data.processor_running.clear()
            logger.exception(
                "No se pudieron iniciar los procesos del session_id=%s rtsp_url=%s",
                session_id,
                rtsp_url,
            )
            raise HTTPException(
                status_code=500,
                detail="No se pudieron iniciar los procesos del stream.",
            ) from exc

        if not reader.is_alive() or not processor.is_alive():
            # Si alguno de los procesos no quedó activo tras el arranque, se asegura que ambos procesos se detengan y se lanza una HTTPException indicando el error.
            project_data.reader_running.clear()
            project_data.processor_running.clear()
            raise HTTPException(
                status_code=500,
                detail="Los procesos del stream no quedaron activos tras el arranque.",
            )

        # Registro de la sesión del stream en el manager con su información.
        self.sessions[session_id] = StreamSession(
            state="running",
            rtsp_url = rtsp_url,
            shared_data = shared_data,
            project_data = project_data,
            reader_process = reader,
            processor_process = processor,
            mqtt = mqtt_config,
        )

        self.runtime_state.active_streams = len(self.sessions)

        return {
            "msg": STREAM_START_SUCCESS_MESSAGE,
            "session_id": session_id,
            "state": "running",
            "rtsp_url": rtsp_url,
            "mqtt": mqtt_config,
        }

    def stop(
        self,
        session_id: str | None = None,
        timeout: float = PROCESS_JOIN_TIMEOUT,
    ) -> dict[str, Any]:
        """Parada de los procesos de lectura e inferencia para un stream específico o para todos los streams activos.

        Args:
            session_id (str | None, optional): ID del stream a detener. Defaults to None.
            timeout (float, optional): Tiempo máximo de espera para la finalización de los procesos. Defaults to PROCESS_JOIN_TIMEOUT.

        Raises:
            HTTPException: Si no se puede detener el stream.

        Returns:
            dict[str, Any]: Información sobre el stream detenido.
        """

        # Si no se proporciona un session_id, se detienen todos los streams activos
        if session_id is None:

            # Detención de todos los streams activos utilizando el método _stop_one para cada sesión, y construcción de la respuesta con la información de los streams detenidos.
            stopped_streams = [
                self._stop_one(session_id=session_id, timeout=timeout)
                for session_id in list(self.sessions)
            ]
            return {
                "msg": STREAMS_STOP_SUCCESS_MESSAGE,
                "stopped": stopped_streams,
            }

        # Si no se encuentra una sesión con el session_id proporcionado, se lanza una HTTPException indicando que no existe la sesión.
        if session_id not in self.sessions:
            raise HTTPException(
                status_code=404,
                detail=f"No existe ninguna sesión con session_id '{session_id}'.",
            )

        # Parada del stream específico utilizando el método _stop_one.
        stopped = self._stop_one(session_id=session_id, timeout=timeout)
        return {
            "msg": STREAM_STOP_SUCCESS_MESSAGE,
            "session_id": stopped["session_id"],
            "state": stopped["state"],
        }

    def health(self) -> dict[str, Any]:
        """Proporciona información sobre el estado de los streams activos,
        incluyendo el número de streams activos, el estado de cada stream,
        la URL de RTSP y el estado de los procesos de lectura e inferencia.

        Returns:
            dict: Información sobre el estado de los streams activos, incluyendo el número de streams activos, el estado de cada stream, la URL de RTSP y el estado de los procesos de lectura e inferencia.
        """
        info_streams = {
            "active_streams": len(self.sessions),
            "streams": []
        }

        for session_id, session in self.sessions.items():
            stream_info = {
                "session_id": session_id,
                "state": session.state,
                "rtsp_url": session.rtsp_url,
                "reader_alive": bool(session.reader_process.is_alive()),
                "processor_alive": bool(session.processor_process.is_alive()),
                "mqtt": session.mqtt,
            }
            info_streams["streams"].append(stream_info)

        return info_streams

    def _stop_one(
        self,
        session_id: str,
        timeout: float,
    ) -> dict[str, Any]:
        """Parada de los procesos de lectura e inferencia para un stream especifico,
        incluyendo la señalización a los procesos para que se detengan, la espera de su finalización con un timeout,
        y la eliminación de la sesión del stream.

        Args:
            session_id (str): ID del stream a detener
            timeout (float): Tiempo máximo de espera para la finalización de los procesos

        Returns:
            dict[str, Any]: Información sobre el stream detenido
        """

        # Obtencion de la session del stream
        session = self.sessions[session_id]
        session.state = "stopping"
        session.project_data.reader_running.clear()
        session.project_data.processor_running.clear()

        # Espera para la finalizacion de los procesos
        for process in (
            session.reader_process,
            session.processor_process,
        ):
            process.join(timeout=timeout)

            # Si el proceso no termina a tiempo, se fuerza su terminación
            if process.is_alive():
                logger.warning(
                    "El proceso %s del session_id=%s no terminó a tiempo; se fuerza terminate().",
                    process.name,
                    session_id,
                )
                process.terminate()
                process.join(timeout=timeout)

        # Eliminacion de la session del stream
        del self.sessions[session_id]
        self.runtime_state.active_streams = len(self.sessions)

        return {
            "msg": STREAM_STOP_SUCCESS_MESSAGE,
            "session_id": session_id,
            "state": "stopped",
        }
