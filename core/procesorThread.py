"""
Archivo que encapsula la logica de procesamiento del stream RTSP y publicación de resultados en MQTT,
para que se pueda ejecutar en tiempo real de forma asincrona y no bloquee la API.
"""

import pandas as pd


def processorThread(
    sharedData,
    projectData,
    saveLog,
    saveInference,
    confidenceClass,
    mqttClient
):
    """Función que se ejecuta en un hilo separado para procesar el stream RTSP y publicar los resultados en MQTT."""
    
    if saveLog:
        #Creacion del archivo csv de log para este hilo
        logCSV = f"logs/log_{int(time.time())}.csv"
        dfLog = pd.DataFrame(columns=["timestamp", "frame_id", "class", "confidence", "bbox"])
        dfLog.to_csv(logCSV, index=False)
    

    while projectData.getProcessorThreadRunning():

        # Intentamos obtener un frame del sharedData con un timeout para evitar bloqueos indefinidos
        package = sharedData.get_frame(timeout=1)
        frame = package["img"]

        #Logica de procesamiento del frame

        #Prueba de resultados de detección (simulados)
        results = {
            "frame_id": package["frame_id"],
            "detections": [
                {
                    "class": "person",
                    "confidence": 0.85,
                    "bbox": [100, 150, 200, 300],
                    "mask": [0, 1, 0, 1]  # Ejemplo de máscara binaria (simplificada)
                }
            ]
        }

        #Logica de publicacion de resultados en MQTT
        mqttClient.publish(results)

        #Escribimos el frame procesado
        if saveLog:
            # Guardamos los resultados de detección en el archivo CSV de log
            pass

        if saveInference:
            # Guardamos las inferencias (clases y coordenadas) en un archivo JSON en la carpeta de inferencias
            pass