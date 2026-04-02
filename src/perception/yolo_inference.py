"""YOLO inference wrapper for the project."""

from __future__ import annotations

import cv2
import numpy as np
import ultralytics

from src.core.constants import (
    CENTROID_COLOR,
    CENTROID_RADIUS,
    DEFAULT_MASK_THRESHOLD,
    MASK_COLOR,
    MASK_OVERLAY_ALPHA,
    MASK_OVERLAY_BETA,
    MASK_OVERLAY_GAMMA,
)
from src.perception.postprocessing import (
    calculate_centroid,
    extract_vertices
)


class YoloInference:
    """Encapsula el modelo YOLO y proporciona metodos complementarios"""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """Inicializa el modelo YOLO con la ruta y el dispositivo especificados.

        Args:
            model_path (str): Ruta al archivo del modelo YOLO.
            device (str, optional): Dispositivo para la inferencia (e.g., "cpu", "cuda"). Defaults to "cpu".
        """
        self.model = ultralytics.YOLO(model_path)
        self.model.to(device)

    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.0,
        debug: bool = False,
    ):
        """Realiza la inferencia en un frame dado utilizando el modelo YOLO.

        Args:
            frame (np.ndarray): Frame sobre el que se realiza la inferencia.
            confidence_threshold (float, optional): Umbral de confianza para filtrar detecciones. Defaults to 0.0.
            debug (bool, optional): Indicador para mostrar información de depuración. Defaults to False.

        Returns:
            ultralytics.engine.results.Results: Resultados de la inferencia del modelo YOLO.
        """
        
        return self.model(frame, conf=confidence_threshold, verbose=debug)

    def extract_detections(
        self,
        results
    ) -> list[dict]:
        """Adapta los resultados a un diccionario con toda la informacion relevante

        Args:
            results ( ultralytics.engine.results.Results): Resultados de la inferencia del modelo YOLO.

        Returns:
            list[dict[str, Any]]: Lista de diccionarios con la información de cada detección, incluyendo clase, confianza, bounding box, mascara, vertices y centroide.
        """
        #FIXME: Revisar si es correcta la estructura de extracion de resultados
        
        #Comprobar si hay resultados
        if not results:
            return []

        
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        masks = results.masks.data.cpu().numpy() if results.masks is not None else []

        detections= []
        for index, class_id in enumerate(classes):
            confidence = float(confidences[index])

            mask_array = masks[index] if index < len(masks) else None
            centroid = calculate_centroid(mask_array)
            vertices = (
                extract_vertices(mask_array).tolist() if mask_array is not None else []
            )
            detections.append(
                {
                    "class_id": int(class_id),
                    "confidence": confidence,
                    "bbox": boxes[index].tolist(),
                    "mask": mask_array.tolist()
                    if isinstance(mask_array, np.ndarray)
                    else [],
                    "vertices": vertices,
                    "centroid": list(centroid) if centroid is not None else [],
                }
            )

        return detections

    def draw_results(self, frame: np.ndarray, results) -> np.ndarray:
        """Dibuja las mascaras de segmentacion y los centroides en el frame original.

        Args:
            frame (np.ndarray): Frame original sobre el que se dibujaran las mascaras y centroides.
            results (ultralytics.engine.results.Results): Resultados de la inferencia del modelo YOLO.

        Returns:
            np.ndarray: Frame con las mascaras y centroides dibujados.
        """
        
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            return frame.copy()

        masks = result.masks.data.cpu().numpy()
        annotated_frame = frame.copy()

        for mask in masks:
            if mask.shape != annotated_frame.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (annotated_frame.shape[1], annotated_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            centroid = calculate_centroid(mask)
            colored_mask = np.zeros_like(annotated_frame)
            colored_mask[mask > DEFAULT_MASK_THRESHOLD] = MASK_COLOR
            annotated_frame = cv2.addWeighted(
                annotated_frame,
                MASK_OVERLAY_BETA,
                colored_mask,
                MASK_OVERLAY_ALPHA,
                MASK_OVERLAY_GAMMA,
            )

            if centroid is not None:
                cv2.circle(
                    annotated_frame,
                    centroid,
                    CENTROID_RADIUS,
                    CENTROID_COLOR,
                    -1,
                )

        return annotated_frame
