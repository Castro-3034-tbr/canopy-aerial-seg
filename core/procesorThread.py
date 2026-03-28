"""
Archivo que encapsula la logica de procesamiento del stream RTSP y publicación de resultados en MQTT,
para que se pueda ejecutar en tiempo real de forma asincrona y no bloquee la API.
"""

import pandas as pd
import time
import cv2
from pathlib import Path


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

    #Obtenemos el directorio del proyecto para guardar los logs y las inferencias
    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "logs"
    inference_dir = project_root / "inference"
    
    logCSV = None
    if saveLog:
        logs_dir.mkdir(parents=True, exist_ok=True)
        #Creacion del archivo csv de log para este hilo
        logCSV = logs_dir / f"log_{int(time.time())}.csv"
        dfLog = pd.DataFrame(columns=["timestamp", "frame_id", "class", "confidence", "bbox", "mask"])
        dfLog.to_csv(logCSV, index=False)

    if saveInference:
        inference_dir.mkdir(parents=True, exist_ok=True)

    #Bucle de procesamiento
    while projectData.getProcessorThreadRunning():

        # Intentamos obtener un frame del sharedData con un timeout para evitar bloqueos indefinidos
        package = sharedData.get_frame(timeout=1)
        frame = package["img"]
        frame_id = package["frame_id"]

        #Realizamos la deteccion con el modelo YOLO y obtenemos los resultados
        yolo_results = classYolo.predict(frame)

        #Obtenemos los resultados en formato serializable
        detections = classYolo.extractDetections(yolo_results, confidence_threshold=confidenceClass)

        #Logica de publicacion de resultados en MQTT
        mqttClient.publish(detections)

        #Escribimos el frame procesado
        if saveLog:
            #Obtenemos el timestamp actual
            timestamp = time.time()
            for detection in detections:
                dfLog = pd.DataFrame([{
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "class": str(detection["class_id"]),
                    "confidence": str(detection["confidence"]),
                    "bbox": str(detection["bbox"]),
                    "mask": "present" if detection["mask"] is not None else "None"
                }])
                dfLog.to_csv(logCSV, mode='a', header=False, index=False)


        if saveInference:
            #Guardamos el frame con las detecciones dibujadas
            annotated_frame = classYolo.drawResults(frame, yolo_results)
            cv2.imwrite(str(inference_dir / f"frame_{frame_id}.jpg"), annotated_frame)