"""
Archivo que encapsula la logica de procesamiento de frames
"""

import ultralytics
import cv2
import numpy as np



class ClassYOLO:
    def __init__(self, model_path: str = "yolov8m.pt", device: str = "cpu"):
        
        self.model = ultralytics.YOLO(model_path)
        self.model.to(device)

    def predict(self, frame):
        results = self.model(frame)
        return results
    
    
    def drawResults(self, frame, results):
        #Comprobamos si se han detectado objetos con mascara en el frame
        if results[0].masks is None:
            return frame.copy()  #Devolvemos una copia del frame original si no se han detectado objetos con mascara
        
        #Obtenemos los bbox y las clases detectadas, y las dibujamos en el frame
        masks = results[0].masks.data.cpu().numpy()
        
        #Hacemos una copia del frame para dibujar los resultados
        annotated_frame = frame.copy()
        
        #Dibujamos los bbox y las clases detectadas en el frame
        for i in range(len(masks) ):
            #Calculamos el centroide de la mascara para dibujar el nombre de la clase
            centroid = self.calcularCentroid(masks[i])
            
            colorMask = (0, 255, 0)
            colorCentroid = (0, 0, 255)
            #Dibujamos el centroide y el nombre de la clase si el centroide es valido
            if centroid is not None:
                cX, cY = centroid
                cv2.circle(annotated_frame, (cX, cY), 5, colorCentroid, -1)
            
            #Dibujamos la mascara si existe
            if masks is not None:
                mask = masks[i]
                colored_mask = np.zeros_like(annotated_frame)
                colored_mask[mask > 0.5] = colorMask
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)
        
        return annotated_frame
    
    def calcularCentroid(self, mask):
        """Funcion para calcular el centroide de una mascara binaria

        Args:
            mask (numpy.ndarray): mascara binaria de la deteccion

        Returns:
            tuple: coordenadas del centroide (x, y)
        """
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            return (cX, cY)
        else:
            return None

    
    def processFrame(self,results):
        """Funcion para calcular el area que tiene la mascara respecto a la imagen 
        para despues calcular el area real de la mascara dependiendo del GSD de la imagen

        Args:
            results (ultralytics.yolo.engine.results.Results): Resultados de la inferencia del modelo YOLO
        """
        
        #Obtenemos el tamaño del frame y la mascara de cada deteccion
        frame_width = results[0].orig_shape[1]
        frame_height = results[0].orig_shape[0]
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
        
        
        #Calculamos el area de cada mascara y la relacion con el area total del frame
        if masks is not None:
            for i in range(masks.shape[0]):
                mask = masks[i]
                mask_area = np.sum(mask > 0.5)
                frame_area = frame_width * frame_height
                area_ratio = mask_area / frame_area
                print(f"Mask {i}: Area={mask_area}, Frame Area={frame_area}, Area Ratio={area_ratio:.4f}")