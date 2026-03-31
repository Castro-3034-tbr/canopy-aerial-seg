"""
Archivo que encapsula la logica de procesamiento del stream RTSP y publicación de resultados en MQTT,
para que se pueda ejecutar en tiempo real de forma asincrona y no bloquee la API.
"""

import logging
import time
from pathlib import Path

import cv2
import pandas as pd

from utils.utils_mqtt import MQTTClient
from utils.utils_yolo import ClassYOLO

logger = logging.getLogger(__name__)


def processorThread(
    sharedData,
    projectData,
    saveLog,
    saveInference,
    confidenceClass,
    mqttBroker,
    mqttPort,
    mqttTopic,
):
    """Función que se ejecuta en un hilo separado para procesar el stream RTSP y publicar los resultados en MQTT."""

    # Creacion de la clase YOLO
    yoloModel = ClassYOLO(projectData.yoloPath, projectData.yoloDevice)
    # Creacion de la clase MQTT
    mqttClient = MQTTClient(
        clientID="processorThread",
        broker=mqttBroker,
        port=mqttPort,
        topic=mqttTopic,
    )

    # Obtenemos el directorio del proyecto para guardar los logs y las inferencias.
    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / projectData.savePathLogs
    inference_dir = project_root / projectData.savePathInference

    log_csv = None
    if saveLog:
        log_csv = logs_dir / f"detections_log_{int(time.time())}.csv"
        if not log_csv.exists():
            log_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                columns=["timestamp", "frame_id", "class", "confidence", "bbox", "mask"]
            ).to_csv(log_csv, index=False)

    if saveInference:
        inference_dir.mkdir(parents=True, exist_ok=True)

    # Bucle de procesamiento.
    while projectData.processorProcessRunning:
        # Intentamos obtener un frame del sharedData con un timeout para evitar bloqueos indefinidos.
        package = sharedData.frame_queue.get(timeout=1)
        if package is None:
            continue

        frame = package["img"]
        frame_id = package["frame_id"]

        # Realizamos la deteccion con el modelo YOLO y obtenemos los resultados.
        yolo_results = yoloModel.predict(frame)

        # Obtenemos los resultados en formato serializable.
        detections = yoloModel.extractDetections(
            yolo_results,
            confidence_threshold=confidenceClass,
        )

        # Publicamos los resultados en MQTT.
        mqttClient.publish(detections)

        # Guardamos los logs de detección en el CSV si se ha solicitado.
        if saveLog and log_csv is not None:
            timestamp = time.time()
            rows = [
                {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "class": str(detection["class_id"]),
                    "confidence": str(detection["confidence"]),
                    "bbox": str(detection["bbox"]),
                    "mask": "present" if detection["mask"] is not None else "None",
                }
                for detection in detections
            ]
            if rows:
                pd.DataFrame(rows).to_csv(log_csv, mode="a", header=False, index=False)

        # Guardamos las inferencias (clases y coordenadas) en un archivo JSON si se ha solicitado.
        if saveInference:
            annotated_frame = yoloModel.drawResults(frame, yolo_results)
            image_path = inference_dir / f"frame_{frame_id}.jpg"
            if not cv2.imwrite(str(image_path), annotated_frame):
                logger.warning("No se pudo guardar la inferencia en %s", image_path)
