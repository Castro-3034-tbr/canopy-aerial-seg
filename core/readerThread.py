"""
Archivo que encapsula la logica de lectura y procesamiento del stream RTSP,
para que se pueda ejecutar en tiempo real de forma asincrona y no bloquee la API.
"""

import av
import logging
import queue
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def readerThread(sharedData, projectData, rtspUrl):
    """Función que se ejecuta en un hilo separado para leer y procesar el stream RTSP."""

    # Lógica para leer los frames del stream RTSP y almacenarlos en sharedData para que el hilo de procesamiento los pueda usar
    while projectData.getReaderThreadRunning():
        try:
            # Conexión al stream RTSP usando PyAV con opciones para minimizar la latencia
            container = av.open(rtspUrl, options={"rtsp_transport": "tcp", "fflags": "nobuffer", "stimeout": "5000000"})
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            # Obtenemos el timebase y fps del stream para usarlo en la sincronización de frames
            timebase = float(stream.time_base)
            fps = float(stream.average_rate) if stream.average_rate else None

            logging.info(f"RTSP stream connected (PyAV) - Timebase: {timebase}, FPS: {fps}")

            frame_counter = 0
            for frame in container.decode(stream):
                if not projectData.getReaderThreadRunning():
                    break

                # Capturamos el frame junto con sus metadatos de sincronizacion nativos
                package = {
                    "img": frame.to_ndarray(format="bgr24"),
                    "frame_id": frame_counter,
                    "pts": frame.pts,  # Suficiente para sincronizacion (timestamp = pts * timebase)
                    "width": frame.width,
                    "height": frame.height
                }
                frame_counter += 1

                # Almacenamos el frame en sharedData
                sharedData.put_frame(package)

            container.close()
        except Exception as e:
            logging.warning(f"RTSP read failed, reconnecting... ({e})")
            time.sleep(2)
