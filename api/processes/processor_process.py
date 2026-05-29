"""Proceso de inferencia, publicación y persistencia de resultados.

Este módulo ejecuta el bucle principal del proceso de inferencia: extrae
frames de la cola compartida, ejecuta el modelo YOLO, calcula métricas,
publica por MQTT y persiste logs e imágenes anotadas según la configuración.
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from common.types.model import YoloModel

from api.mqtt.connection import (
    connect_mqtt,
    disconnect_mqtt,
)
from api.mqtt.publisher import publish_message

from api.core.constants import (
    DETECTION_LOG_COLUMNS,
    DETECTIONS_LOG_PREFIX,
    DETECTIONS_LOG_SUFFIX,
    FRAME_FILENAME_PREFIX,
    FRAME_FILENAME_SUFFIX,
    FRAME_QUEUE_TIMEOUT_SECONDS,
)
from common.constants import PROJECT_ROOT

from api.perception.analisis import (
    analyze_results,
)

from api.core.types import (
    FramePackage,
    ProjectData,
    SharedData,
    MQTTConfig,
)
from api.perception.yolo_inference import draw_results, predict

logger = logging.getLogger(__name__)


def processor_process(
    shared_data: SharedData,
    project_data: ProjectData,
    session_id: str,
    save_log: bool,
    save_inference: bool,
    confidence_threshold: float,
    yolo_model: YoloModel,
    mqtt_config: MQTTConfig,
    overlap: tuple[float, float],
    gsd: float,
) -> None:
    """Bucle principal del proceso de inferencia y publicación.

    Args:
        shared_data (SharedData): Estructuras compartidas entre procesos (incluye `frame_queue`).
        project_data (ProjectData): Datos y flags de la sesión (ej. rutas de guardado).
        session_id (str): Identificador de la sesión.
        save_log (bool): Si True, se persisten logs en CSV.
        save_inference (bool): Si True, se guardan imágenes anotadas.
        confidence_threshold (float): Umbral de confianza para filtrar detecciones.
        yolo_model (YoloModel): Modelo YOLO ya cargado para inferencia.
        mqtt_config (MQTTConfig): Configuración MQTT para publicar mensajes.
        overlap (tuple[float, float]): Recorte de solapes en máscaras.
        gsd (float): Ground Sample Distance en metros/píxel.

    Returns:
        None: Esta función se ejecuta en un proceso separado.

    Raises:
        Exception: Las excepciones internas se registran; en la clausura se
            desconecta el cliente MQTT.
    """

    # Inicializacion del cliente MQTT
    mqtt_client = connect_mqtt(
        config = mqtt_config,
    )

    # Preparacion de rutas de guardado para logs e inferencias visuales.
    project_root = PROJECT_ROOT
    logs_dir = project_root / project_data.save_path_logs
    inference_dir = project_root / project_data.save_path_inference

    log_csv: Path | None = None
    if save_log:
        # Crea un CSV por ejecucion para persistir las detecciones del stream.
        log_csv = logs_dir / (
            f"{DETECTIONS_LOG_PREFIX}{int(time.time())}" f"{DETECTIONS_LOG_SUFFIX}"
        )
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        if not log_csv.exists():
            pd.DataFrame(columns=DETECTION_LOG_COLUMNS).to_csv(
                log_csv,
                index=False,
            )

    if save_inference:
        # Prepara el directorio de salida de frames anotados.
        inference_dir.mkdir(parents=True, exist_ok=True)

    try:
        while (
            project_data.processor_running.is_set()
            or not shared_data.frame_queue.empty()
        ):
            try:
                # Espera un frame sin bloquear indefinidamente.
                package: FramePackage = shared_data.frame_queue.get(
                    timeout=FRAME_QUEUE_TIMEOUT_SECONDS
                )

                frame = package.img
                frame_id = package.frame_id

            except queue.Empty:
                # Si ya no se aceptan mas frames y la cola quedo vacia,
                # finaliza el proceso de consumo.
                if not project_data.processor_running.is_set():
                    break
                logger.debug(
                    "No se recibio ningun frame en el tiempo de espera "
                    "session_id=%s timeout_s=%s",
                    session_id,
                    FRAME_QUEUE_TIMEOUT_SECONDS,
                )
                continue

            try:
                # Ejecuta inferencia
                yolo_results = predict(
                    model=yolo_model,
                    frame=frame,
                    confidence_threshold=confidence_threshold,
                    overlap=overlap,
                )

                logger.debug(
                    "Frame procesado session_id=%s frame_id=%s detections=%s ",
                    session_id,
                    frame_id,
                    len(yolo_results) if yolo_results else 0,
                )

                # Calculo de areas de las mascaras y otras metricas basicas.
                areas = analyze_results(result=yolo_results, gsd=gsd)

                # Publica el lote actual de detecciones en MQTT.
                mensaje = {
                    "frame_id": frame_id,
                    "area_m2": areas["total_area_m2"],
                }

                publish_message(
                    client=mqtt_client,
                    payload=mensaje,
                    mqtt_config=mqtt_config,
                )

                if save_log and log_csv is not None:
                    # Añade una fila por deteccion al log tabular del stream.
                    timestamp = time.time()
                    row=[{
                            "timestamp": timestamp,
                            "frame_id": frame_id,
                            "area_m2": areas["total_area_m2"],
                        }]
                    if row:
                        pd.DataFrame(row, columns=DETECTION_LOG_COLUMNS).to_csv(
                            log_csv,
                            mode="a",
                            header=False,
                            index=False,
                        )
                if save_inference:

                    # Dibuja las detecciones sobre el frame original y lo guarda como imagen.
                    frame_annotated = draw_results(frame=frame, results=yolo_results)

                    inference_path = inference_dir / (
                        f"{FRAME_FILENAME_PREFIX}{frame_id}" f"{FRAME_FILENAME_SUFFIX}"
                    )
                    cv2.imwrite(str(inference_path), np.asarray(frame_annotated, dtype=np.uint8))

            except Exception:
                logger.exception(
                    "Error durante el procesamiento del frame session_id=%s frame_id=%s",
                    session_id,
                    frame_id,
                )
    finally:
        # Cierra la conexion MQTT aunque el proceso termine por error.
        disconnect_mqtt(client=mqtt_client)
