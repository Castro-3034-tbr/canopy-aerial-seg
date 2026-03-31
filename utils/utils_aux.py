import cv2
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import HTTPException

def processImage(yoloModel, contents: bytes, confidence_threshold: float) -> tuple[Path, str]:
    """Procesa una imagen y devuelve la ruta temporal de salida y su media type."""
    
    #Leemos la imagen desde los bytes recibidos
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.")
    
    #Realizamos la predicción y anotamos la imagen
    results = yoloModel.predict(frame, confidence_threshold)
    
    #Dibujamos los resultados en la imagen
    annotated_frame = yoloModel.drawResults(frame, results)
    
    #Guardamos la imagen anotada en un archivo temporal
    with NamedTemporaryFile(delete=False, suffix=".jpg") as output_file:
        output_path = Path(output_file.name)
    if not cv2.imwrite(str(output_path), annotated_frame):
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="No se pudo generar la imagen resultante.")

    return output_path, "image/jpeg"


def processVideo(yoloModel,contents: bytes, confidence_threshold: float) -> tuple[Path, str]:
    """Procesa un video y devuelve la ruta temporal de salida y su media type."""
    
    #Guardamos el video recibido en un archivo temporal para poder procesarlo con OpenCV
    with NamedTemporaryFile(delete=False, suffix=".mp4") as input_file:
        input_file.write(contents)
        input_path = Path(input_file.name)

    #Creamos un archivo temporal para guardar el video anotado
    with NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
        output_path = Path(output_file.name)

    #Abrimos el video con OpenCV
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        #Si no se pudo abrir el video, liberamos recursos y eliminamos los archivos temporales
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No se pudo abrir el video enviado.")

    #Obtenemos el FPS y las dimensiones del video para configurar el VideoWriter
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        #Si no se pudieron obtener las dimensiones del video, liberamos recursos y eliminamos los archivos temporales
        capture.release()
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No se pudieron obtener las dimensiones del video.")

    #Creamos el VideoWriter para guardar el video anotado
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        #Si no se pudo crear el VideoWriter, liberamos recursos y eliminamos los archivos temporales
        capture.release()
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="No se pudo generar el video resultante.")

    try:
        #Bucle de procesamiento de cada frame del video
        while True:
            #Leemos un frame del video
            success, frame = capture.read()
            if not success:
                break
            #Realizamos la predicción y anotamos el frame
            results = yoloModel.predict(frame)
            annotated_frame = yoloModel.drawResults(frame, results)
            writer.write(annotated_frame)
    finally:
        #Liberamos los recursos y eliminamos el archivo temporal de entrada
        capture.release()
        writer.release()
        input_path.unlink(missing_ok=True)

    return output_path, "video/mp4"
