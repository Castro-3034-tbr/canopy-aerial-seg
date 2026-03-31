"""
Archivo que encapsula la logica de procesamiento del stream RTSP y publicación de resultados en MQTT,
para que se pueda ejecutar en tiempo real de forma asincrona y no bloquee la API.
"""

import time
import logging
from pathlib import Path

import cv2
import pandas as pd

logger = logging.getLogger(__name__)

def _appendDetectionsToLog(log_csv: Path, frame_id: int, detections: list[dict]) -> None:
    """Añade las detecciones del frame al CSV de log.
    Args:
        log_csv (Path): Ruta del archivo CSV de log.
        frame_id (int): ID del frame procesado.
        detections (list[dict]): Lista de detecciones a añadir al log.
    """
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


def processorThread(
    sharedData,
    projectData,
    saveLog,
    saveInference,
    confidenceClass,
    mqttClient,
    classYolo
):
    """Función que se ejecuta en un hilo separado para procesar el stream RTSP y publicar los resultados en MQTT."""

    # Obtenemos el directorio del proyecto para guardar los logs y las inferencias.
    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / projectData.getSavePathLogs()
    inference_dir = project_root / projectData.getSavePathInference()

    log_csv = None
    if saveLog:
        log_csv = _initializeLogFile(logs_dir)

    if saveInference:
        inference_dir.mkdir(parents=True, exist_ok=True)

    # Bucle de procesamiento.
    while projectData.getProcessorThreadRunning():
        # Intentamos obtener un frame del sharedData con un timeout para evitar bloqueos indefinidos.
        package = sharedData.getFrame(timeout=1)
        if package is None:
            continue

        frame = package["img"]
        frame_id = package["frame_id"]

        # Realizamos la deteccion con el modelo YOLO y obtenemos los resultados.
        yolo_results = classYolo.predict(frame)

        # Obtenemos los resultados en formato serializable.
        detections = classYolo.extractDetections(
            yolo_results,
            confidence_threshold=confidenceClass,
        )

        # Publicamos los resultados en MQTT.
        mqttClient.publish(detections)

        # Guardamos los logs de detección en el CSV si se ha solicitado.
        if saveLog and log_csv is not None:
            _appendDetectionsToLog(log_csv, frame_id, detections)

        # Guardamos las inferencias (clases y coordenadas) en un archivo JSON si se ha solicitado.
        if saveInference:
            annotated_frame = classYolo.drawResults(frame, yolo_results)
            image_path = inference_dir / f"frame_{frame_id}.jpg"
            if not cv2.imwrite(str(image_path), annotated_frame):
                logger.warning("No se pudo guardar la inferencia en %s", image_path)
