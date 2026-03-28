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
        
        #Obtenemos los bbox y las clases detectadas, y las dibujamos en el frame
        bbox = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
        
        #Hacemos una copia del frame para dibujar los resultados
        annotated_frame = frame.copy()
        
        #Dibujamos los bbox y las clases detectadas en el frame
        for i in range(len(bbox)):
            x1, y1, x2, y2 = bbox[i]
            cls = int(classes[i])
            conf = confidences[i]
            
            #Dibujamos el bbox
            color = (0, 255, 0) if cls == 0 else (255, 255, 0) # Verde para complete, amarillo para incomplete
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            #Dibujamos la clase y la confianza
            label = f"{'complete' if cls == 0 else 'incomplete'}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            #Dibujamos la mascara si existe
            if masks is not None:
                mask = masks[i]
                colored_mask = np.zeros_like(annotated_frame)
                colored_mask[mask > 0.5] = color
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)
        
        return annotated_frame
    
    
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