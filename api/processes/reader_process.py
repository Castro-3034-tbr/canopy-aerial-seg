"""Proceso lector para flujos de video RTSP.

Este módulo implementa la función `reader_process` que se ejecuta en un
proceso separado, se conecta a una fuente RTSP usando PyAV, decodifica
frames y los coloca en una cola compartida para que el procesador los
consuma.
"""

from __future__ import annotations

import logging
import queue
import time

import av

from api.core.constants import (
    FRAME_QUEUE_PUT_TIMEOUT_SECONDS,
    RECONNECT_DELAY_SECONDS,
    RTSP_OPTIONS,
)
from api.core.types import FramePackage, ProjectData, SharedData

logger = logging.getLogger(__name__)


def reader_process(
    shared_data: SharedData,
    project_data: ProjectData,
    session_id: str,
    rtsp_url: str,
) -> None:
    """Proceso lector que conecta a RTSP, decodifica frames y los encola.

    Args:
        shared_data (SharedData): Estructuras compartidas entre procesos (incluye `frame_queue`).
        project_data (ProjectData): Datos y flags específicos de la sesión (ej. `reader_running`).
        session_id (str): Identificador de la sesión.
        rtsp_url (str): URL de la fuente RTSP.

    Returns:
        None: Esta función se ejecuta en un proceso separado y finaliza cuando
            `project_data.reader_running` se limpia.

    Notes:
        - Los errores de conexión o lectura se registran y el proceso reintenta
            la conexión en la conexión tras `RECONNECT_DELAY_SECONDS`.
        - Si la cola de frames está llena, se descarta el frame más antiguo
            para mantener la latencia y se intenta insertar el nuevo.
    """

    # Contador de frames para asignar un ID único a cada frame leído
    frame_counter = 0

    # Bucle principal de lectura de frames
    while project_data.reader_running.is_set():
        container = None
        try:
            # Creacion del contenedor de video utilizando PyAV para conectarse a la fuente RTSP
            container = av.open(
                rtsp_url,
                options=RTSP_OPTIONS,
            )
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            timebase = float(stream.time_base) if stream.time_base is not None else 0.0
            fps = float(stream.average_rate) if stream.average_rate else None
            logger.info(
                "Stream RTSP conectado session_id=%s rtsp_url=%s timebase=%s fps=%s",
                session_id,
                rtsp_url,
                timebase,
                fps,
            )

            # Lectura de frames del video y colocación en la cola compartida
            for frame in container.decode(stream):
                # Verificación de la señal de parada del proceso de lectura antes de colocar el frame en la cola
                if not project_data.reader_running.is_set():
                    break

                # Guardado del frame y su informacion en la cola
                package: FramePackage = FramePackage(
                    img=frame.to_ndarray(format="bgr24").astype("uint8"),
                    frame_id=frame_counter,
                    pts=frame.pts,
                    width=frame.width,
                    height=frame.height,
                )
                try:
                    shared_data.frame_queue.put(
                        package,
                        timeout=FRAME_QUEUE_PUT_TIMEOUT_SECONDS,
                    )
                except queue.Full:
                    # Mantiene el stream vivo priorizando los frames mas recientes.
                    try:
                        dropped_package = shared_data.frame_queue.get_nowait()
                        dropped_frame_id = dropped_package.frame_id
                    except queue.Empty:
                        dropped_frame_id = None

                    logger.warning(
                        "Cola de frames llena; se descarta el frame mas antiguo "
                        "session_id=%s dropped_frame_id=%s new_frame_id=%s",
                        session_id,
                        dropped_frame_id,
                        frame_counter,
                    )
                    try:
                        shared_data.frame_queue.put_nowait(package)
                    except queue.Full:
                        logger.warning(
                            "No se pudo insertar el nuevo frame tras descartar uno "
                            "session_id=%s frame_id=%s",
                            session_id,
                            frame_counter,
                        )
                        continue
                frame_counter += 1

            # Si el bucle de lectura termina de forma natural, se espera un tiempo y se intenta reconectar si el proceso de lectura sigue activo
            if project_data.reader_running.is_set():
                logger.info(
                    "Stream RTSP finalizado; se reintentara la conexion "
                    "session_id=%s rtsp_url=%s",
                    session_id,
                    rtsp_url,
                )
                time.sleep(RECONNECT_DELAY_SECONDS)
        except Exception:
            logger.exception(
                "Lectura RTSP fallida; se reintentara la conexion "
                "session_id=%s rtsp_url=%s",
                session_id,
                rtsp_url,
            )
            time.sleep(RECONNECT_DELAY_SECONDS)
        finally:
            # Asegurar el cierre del contenedor de video para liberar recursos
            if container is not None:
                container.close()
