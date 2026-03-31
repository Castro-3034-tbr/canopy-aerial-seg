"""
Archivo que encapsula la logica de lectura y procesamiento del stream RTSP,
para que se pueda ejecutar en tiempo real de forma asincrona y no bloquee la API.
"""

import av
import logging
import time

logger = logging.getLogger(__name__)


def readerThread(sharedData, projectData, rtspUrl):
    """ Hilo encargado de leer el stream RTSP usando PyAV, procesar los frames y almacenarlos en sharedData para que el hilo de inferencia los consuma.

    Args:
        sharedData (SharedData): Instancia de SharedData para almacenar los frames leidos del stream RTSP
        projectData (ProjectData): Instancia de ProjectData para controlar el estado del hilo y la vida del proceso
        rtspUrl (str): URL del stream RTSP a procesar
    """

    # frame_counter global para mantener continuidad entre reconexiones.
    frame_counter = 0

    # Bucle externo de vida del hilo: solo reconecta cuando la conexión falla o termina.
    while projectData.getReaderThreadRunning():
        container = None
        try:
            # Conexión al stream RTSP usando PyAV con opciones para minimizar la latencia.
            container = av.open(
                rtspUrl,
                options={
                    "rtsp_transport": "tcp",
                    "fflags": "nobuffer",
                    "stimeout": "5000000",
                },
            )
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            # Obtenemos el timebase y fps del stream para usarlo en la sincronización de frames.
            timebase = float(stream.time_base) if stream.time_base is not None else 0.0
            fps = float(stream.average_rate) if stream.average_rate else None

            logger.info("RTSP stream connected (PyAV) - Timebase: %s, FPS: %s", timebase, fps)

            # Bucle interno de lectura: procesa frames mientras la conexión siga viva.
            for frame in container.decode(stream):  # type: ignore[attr-defined]
                if not projectData.getReaderThreadRunning():
                    break

                # Capturamos el frame junto con sus metadatos de sincronizacion nativos.
                package = {
                    "img": frame.to_ndarray(format="bgr24"),
                    "frame_id": frame_counter,
                    "pts": frame.pts,
                    "width": frame.width,
                    "height": frame.height,
                }
                frame_counter += 1

                # Almacenamos el frame en sharedData.
                sharedData.putFrame(package)

            # Si el decode termina sin excepción y el hilo sigue activo, forzamos reconexión.
            if projectData.getReaderThreadRunning():
                logger.warning("RTSP stream ended, reconnecting...")
                time.sleep(2)

        except Exception as e:
            logger.warning("RTSP read failed, reconnecting... (%s)", e)
            time.sleep(2)
        finally:
            if container is not None:
                container.close()
