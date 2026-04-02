from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

import cv2
import pandas as pd

from src.communication.mqtt_client import MQTTClient
from src.core.constants import (
    DETECTION_LOG_COLUMNS,
    DETECTIONS_LOG_PREFIX,
    DETECTIONS_LOG_SUFFIX,
    FRAME_FILENAME_PREFIX,
    FRAME_FILENAME_SUFFIX,
    FRAME_QUEUE_TIMEOUT_SECONDS,
    PROCESSOR_MQTT_CLIENT_ID,
)

logger = logging.getLogger(__name__)


def processor_process(
    shared_data,
    project_data,
    save_log,
    save_inference,
    confidence_class,
    mqtt_broker,
    mqtt_port,
    mqtt_topic,
    yolo_model
):
    """Funcion principal del proceso de procesamiento de los frame de video,
    Se encarga:
        - Obtener el frame desde la cola compartida.
        - Realizar la inferencia utilizando el modelo YOLO.
        - Adaptar los resultados de la inferencia al formato requerido por el cliente MQTT.
        - Publicar los resultados en el broker MQTT.
        - Guardar los resultados en un log CSV si se ha habilitado la opción de guardar logs.
        - Guardar las inferencias visuales en formato de imagen si se ha habilitado la opción de guardar inferencias.

    Args:
        shared_data (Manager.Namespace): Datos compartidos entre procesos, incluyendo la cola de frames.
        project_data (Manager.Namespace): Configuración y datos específicos del proyecto, como rutas de guardado y configuración del modelo.
        save_log (bool): Indicador para guardar logs.
        save_inference (bool): Indicador para guardar inferencias visuales.
        confidence_class (float): Umbral de confianza para filtrar detecciones.
        mqtt_broker (str): Dirección del broker MQTT.
        mqtt_port (int): Puerto del broker MQTT.
        mqtt_topic (str): Tema del broker MQTT.
        yolo_model (YoloInference): Instancia del modelo de inferencia YOLO cargado.
    """
    
    # Inicialización del cliente MQTT
    mqtt_client = MQTTClient(
        client_id=PROCESSOR_MQTT_CLIENT_ID,
        broker=mqtt_broker,
        port=mqtt_port,
        topic=mqtt_topic,
    )

    project_root = Path.cwd()
    logs_dir = project_root / project_data.save_path_logs
    inference_dir = project_root / project_data.save_path_inference

    log_csv = None
    if save_log:
        # Crea un CSV por ejecucion para persistir las detecciones del stream.
        log_csv = logs_dir / (
            f"{DETECTIONS_LOG_PREFIX}{int(time.time())}"
            f"{DETECTIONS_LOG_SUFFIX}"
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
        while project_data.processor_process_running:
            try:
                # Espera un frame sin bloquear indefinidamente.
                package = shared_data.frame_queue.get(
                    timeout=FRAME_QUEUE_TIMEOUT_SECONDS
                )
            except queue.Empty:
                logger.info(
                    "No se recibio ningun frame en el ultimo segundo."
                )
                continue

            # Ejecuta inferencia y adapta el resultado a un formato simple.
            frame = package["img"]
            frame_id = package["frame_id"]
            yolo_results = yolo_model.predict(
                frame,
                confidence_threshold=confidence_class,
            )
            detections = yolo_model.extract_detections(yolo_results)

            # Publica el lote actual de detecciones en MQTT.
            mqtt_client.publish(detections)

            if save_log and log_csv is not None:
                # Añade una fila por deteccion al log tabular del stream.
                timestamp = time.time()
                rows = [
                    {
                        "timestamp": timestamp,
                        "frame_id": frame_id,
                        "class": str(detection["class_id"]),
                        "confidence": str(detection["confidence"]),
                        "bbox": str(detection["bbox"]),
                        "mask": "present" if detection["mask"] else "None",
                    }
                    for detection in detections
                ]
                if rows:
                    pd.DataFrame(rows).to_csv(
                        log_csv,
                        mode="a",
                        header=False,
                        index=False,
                    )

            if save_inference:
                # Guarda una version anotada del frame procesado.
                annotated_frame = yolo_model.draw_results(frame, yolo_results)
                image_path = inference_dir / (
                    f"{FRAME_FILENAME_PREFIX}{frame_id}"
                    f"{FRAME_FILENAME_SUFFIX}"
                )
                if not cv2.imwrite(str(image_path), annotated_frame):
                    logger.warning(
                        "No se pudo guardar la inferencia en %s",
                        image_path,
                    )
    finally:
        # Cierra la conexion MQTT aunque el proceso termine por error.
        mqtt_client.disconnect()
