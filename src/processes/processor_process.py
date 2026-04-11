"""Proceso de inferencia, publicacion y persistencia de resultados."""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import cv2
import pandas as pd
import ultralytics

from src.communication.mqtt_client import (
    connect_mqtt,
    disconnect_mqtt,
    publish_message,
)
from src.core.constants import (
    DETECTION_LOG_COLUMNS,
    DETECTIONS_LOG_PREFIX,
    DETECTIONS_LOG_SUFFIX,
    FRAME_FILENAME_PREFIX,
    FRAME_FILENAME_SUFFIX,
    FRAME_QUEUE_TIMEOUT_SECONDS,
    PROCESSOR_MQTT_CLIENT_ID,
)
from src.perception.postprocessing import convert_detections_to_json

from src.core.types import (
    FramePackage,
    ProjectData,
    SharedData,
)
from src.perception.yolo_inference import predict, draw_results

logger = logging.getLogger(__name__)


def processor_process(
    shared_data: SharedData,
    project_data: ProjectData,
    save_log: bool,
    save_inference: bool,
    confidence_threshold: float,
    mqtt_broker: str,
    mqtt_port: int,
    mqtt_topic: str,
    yolo_model: ultralytics.YOLO,
) -> None:
    """Funcion principal del proceso de procesamiento de los frame de video,
    Se encarga:
        - Obtener el frame desde la cola compartida.
        - Realizar la inferencia utilizando el modelo YOLO.
        - Adaptar los resultados de la inferencia al formato requerido por el cliente MQTT.
        - Publicar los resultados en el broker MQTT.
        - Guardar los resultados en un log CSV si se ha habilitado la opción de guardar logs.
        - Guardar las inferencias visuales en formato de imagen si se ha habilitado la opción de guardar inferencias.

    Args:
        shared_data (SharedData): Datos compartidos entre procesos, incluyendo la cola de frames.
        project_data (ProjectData): Configuración y datos específicos del proyecto, como rutas de guardado y configuración del modelo.
        save_log (bool): Indicador para guardar logs.
        save_inference (bool): Indicador para guardar inferencias visuales.
        confidence_threshold (float): Umbral de confianza para filtrar detecciones.
        mqtt_broker (str): Dirección del broker MQTT.
        mqtt_port (int): Puerto del broker MQTT.
        mqtt_topic (str): Tema del broker MQTT.
        yolo_model (ultralytics.YOLO): Instancia del modelo de inferencia YOLO cargado.
    """

    # Inicializacion del cliente MQTT
    mqtt_client = connect_mqtt(
        client_id=PROCESSOR_MQTT_CLIENT_ID,
        broker=mqtt_broker,
        port=mqtt_port,
        topic=mqtt_topic,
    )

    # Preparacion de rutas de guardado para logs e inferencias visuales.
    project_root = Path.cwd()
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
        while project_data.processor_process_running.is_set():
            try:
                # Espera un frame sin bloquear indefinidamente.
                package: FramePackage = shared_data.frame_queue.get(
                    timeout=FRAME_QUEUE_TIMEOUT_SECONDS
                )
            except queue.Empty:
                logger.debug(
                    "No se recibio ningun frame en el tiempo de espera "
                    "stream_id=%s timeout_s=%s",
                    project_data.stream_id,
                    FRAME_QUEUE_TIMEOUT_SECONDS,
                )
                continue

            try:
                # Ejecuta inferencia y adapta el resultado a un formato simple.
                frame = package["img"]
                frame_id = package["frame_id"]

                yolo_results = predict(
                    yolo_model,
                    frame,
                    confidence_threshold=confidence_threshold,
                )

                logger.debug(
                    "Frame procesado stream_id=%s frame_id=%s detections=%s latency_ms=%.2f",
                    project_data.stream_id,
                    frame_id,
                    len(yolo_results.boxes) if yolo_results else 0,
                )

                # Calculo de areas de las mascaras y otras metricas basicas.
                # TODO: 
                

                # Publica el lote actual de detecciones en MQTT.
                result_json = convert_detections_to_json(yolo_results, frame_id)
                publish_message(
                    mqtt_client,
                    result_json,
                    frame_id=frame_id,
                )

                if save_log and log_csv is not None:
                    # Añade una fila por deteccion al log tabular del stream.
                    timestamp = time.time()
                    rows = []
                    json_detections = {"frame_id": frame_id, "timestamp": timestamp, "detections": []}
                    for i , detection in yolo_results:
                        json_detections["detections"].append({
                            "index": i,
                            "class": str(detection.class_id),
                            "confidence": str(detection.confidence),
                            "bbox": str(detection.bbox),
                            "mask": "present" if detection.mask else "None",
                            "centroid": str(detection.centroid),
                        })
                    if rows:
                        pd.DataFrame(rows).to_csv(
                            log_csv,
                            mode="a",
                            header=False,
                            index=False,
                        )

                if save_inference:
                    # Guarda una version anotada del frame procesado.
                    annotated_frame = draw_results(frame=frame, results=yolo_results)
                    image_path = inference_dir / (
                        f"{FRAME_FILENAME_PREFIX}{frame_id}" f"{FRAME_FILENAME_SUFFIX}"
                    )
                    if not cv2.imwrite(str(image_path), annotated_frame):
                        logger.warning(
                            "No se pudo guardar la inferencia stream_id=%s frame_id=%s path=%s",
                            project_data.stream_id,
                            frame_id,
                            image_path,
                        )
            except Exception:
                logger.exception(
                    "Error durante el procesamiento del frame stream_id=%s frame_id=%s",
                    project_data.stream_id,
                    package.get("frame_id"),
                )
    finally:
        # Cierra la conexion MQTT aunque el proceso termine por error.
        disconnect_mqtt(mqtt_client)
