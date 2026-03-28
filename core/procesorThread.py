"""
Archivo que encapsula la logica de procesamiento del stream RTSP y publicación de resultados en MQTT,
para que se pueda ejecutar en tiempo real de forma asincrona y no bloquee la API.
"""

import pandas as pd
import time
import cv2


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
    
    if saveLog:
        #Creacion del archivo csv de log para este hilo
        logCSV = f"logs/log_{int(time.time())}.csv"
        dfLog = pd.DataFrame(columns=["timestamp", "frame_id", "class", "confidence", "bbox", "mask"])
        dfLog.to_csv(logCSV, index=False)
    

    while projectData.getProcessorThreadRunning():

        # Intentamos obtener un frame del sharedData con un timeout para evitar bloqueos indefinidos
        package = sharedData.get_frame(timeout=1)
        frame = package["img"]

        #Realizamos la deteccion con el modelo YOLO y obtenemos los resultados
        results = classYolo.predict(frame)

        #Obtenemos los resultados
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        bbox = results[0].boxes.xyxy.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
        detecciones = []
        for i in range(len(classes)):
            if confidences[i] >= confidenceClass:
                detecciones[i] = {
                    "class": classes[i],
                    "confidence": confidences[i],
                    "bbox": bbox[i].tolist(),
                    "mask": masks[i].tolist() if masks is not None else None
                }

        #Construimos el diccionario de resultados para publicarlo en MQTT
        results = {
            "frame_id": package["frame_id"],
            "detections": detecciones
        }

        #Logica de publicacion de resultados en MQTT
        mqttClient.publish(results)

        #Escribimos el frame procesado
        if saveLog:
            dfLog = pd.DataFrame([{
                "timestamp": time.time(),
                "frame_id": package["frame_id"],
                "class": classes,
                "confidence": confidences,
                "bbox": bbox,
                "mask": masks
            }])
            dfLog.to_csv(logCSV, mode='a', header=False, index=False)


        if saveInference:
            #Guardamos el frame con las detecciones dibujadas
            annotated_frame = classYolo.drawResults(frame, results)
            cv2.imwrite(f"inference/frame_{package['frame_id']}.jpg", annotated_frame)