"""Proceso lector para flujos de video RTSP."""

from __future__ import annotations

import logging
import queue
import time

import av

from src.core.constants import (
    FRAME_QUEUE_PUT_TIMEOUT_SECONDS,
    RECONNECT_DELAY_SECONDS,
    RTSP_OPTIONS,
)
from src.core.types import FramePackage, ProjectData, SharedData

logger = logging.getLogger(__name__)


def reader_process(
    shared_data: SharedData,
    project_data: ProjectData,
    rtsp_url: str,
) -> None:
    """Funcion principal del proceso de lectura de los frame de video,
    Se encarga de:
        - Conectarse a la fuente de video RTSP utilizando PyAV.
        - Leer los frames del video y extraer la información relevante (ID de frame, PTS, dimensiones).
        - Colocar los frames en una cola compartida para que el proceso de procesamiento pueda acceder a ellos.

    Args:
        shared_data (Manager.Namespace): Datos compartidos entre procesos, incluyendo la cola de frames.
        project_data (Manager.Namespace): Configuración y datos específicos del proyecto, como rutas de guardado y configuración del modelo.
        rtsp_url (str): URL de la fuente de video RTSP.
    """

    stream_id = getattr(project_data, "stream_id", "unknown")

    # Contador de frames para asignar un ID único a cada frame leído
    frame_counter = 0

    # Bucle principal de lectura de frames
    while project_data.reader_process_running.is_set():
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
                "Stream RTSP conectado stream_id=%s rtsp_url=%s timebase=%s fps=%s",
                stream_id,
                rtsp_url,
                timebase,
                fps,
            )

            # Lectura de frames del video y colocación en la cola compartida
            for frame in container.decode(stream):
                # Verificación de la señal de parada del proceso de lectura antes de colocar el frame en la cola
                if not project_data.reader_process_running.is_set():
                    break

                # Guardado del frame y su informacion en la cola
                package: FramePackage = {
                    "img": frame.to_ndarray(format="bgr24").astype("uint8"),
                    "frame_id": frame_counter,
                    "pts": frame.pts,
                    "width": frame.width,
                    "height": frame.height,
                }
                try:
                    shared_data.frame_queue.put(
                        package,
                        timeout=FRAME_QUEUE_PUT_TIMEOUT_SECONDS,
                    )
                except queue.Full:
                    # Mantiene el stream vivo priorizando los frames mas recientes.
                    try:
                        dropped_package = shared_data.frame_queue.get_nowait()
                        dropped_frame_id = dropped_package.get("frame_id")
                    except queue.Empty:
                        dropped_frame_id = None

                    logger.warning(
                        "Cola de frames llena; se descarta el frame mas antiguo "
                        "stream_id=%s dropped_frame_id=%s new_frame_id=%s",
                        stream_id,
                        dropped_frame_id,
                        frame_counter,
                    )
                    try:
                        shared_data.frame_queue.put_nowait(package)
                    except queue.Full:
                        logger.warning(
                            "No se pudo insertar el nuevo frame tras descartar uno "
                            "stream_id=%s frame_id=%s",
                            stream_id,
                            frame_counter,
                        )
                        continue
                frame_counter += 1

            # Si el bucle de lectura termina de forma natural, se espera un tiempo y se intenta reconectar si el proceso de lectura sigue activo
            if project_data.reader_process_running.is_set():
                logger.info(
                    "Stream RTSP finalizado; se reintentara la conexion "
                    "stream_id=%s rtsp_url=%s",
                    stream_id,
                    rtsp_url,
                )
                time.sleep(RECONNECT_DELAY_SECONDS)
        except Exception:
            logger.exception(
                "Lectura RTSP fallida; se reintentara la conexion "
                "stream_id=%s rtsp_url=%s",
                stream_id,
                rtsp_url,
            )
            time.sleep(RECONNECT_DELAY_SECONDS)
        finally:
            # Asegurar el cierre del contenedor de video para liberar recursos
            if container is not None:
                container.close()
