"""Utilidades para inferencia, visualizacion y postproceso con YOLO."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import ultralytics


class ClassYOLO:
    """Encapsula el uso del modelo YOLO dentro del proyecto."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """Carga el modelo YOLO desde la ruta especificada y lo mueve al dispositivo indicado.

        Args:
            model_path (str): Ruta al archivo del modelo YOLO.
            device (str, optional): Dispositivo en el que se ejecutará el modelo. Defaults to "cpu".
        """
        self.model = ultralytics.YOLO(model_path)
        self.model.to(device)

    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.0,
        debug: bool = False,
    ):
        """Ejecuta inferencia sobre un frame BGR."""
        return self.model(frame, conf=confidence_threshold, verbose=debug)

    def extractVertices(self, mask: np.ndarray) -> np.ndarray:
        """Aplica el algoritmo de Canny para obtener las coordenadas de los vértices de los bordes.
        Args:
            mask (np.ndarray): Máscara binaria.
        Returns:
            np.ndarray: Coordenadas de los vértices de los bordes."""

        #Extracion de los contornos de la mascara utilizando el algoritmo de Canny
        contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
        #Si no se encuentran contornos, se devuelve un array vacío
        if not contours:
            return np.array([])

        # Elegir el contorno más grande
        contour = max(contours, key=cv2.contourArea)

        return contour.squeeze()

    def extractDetections(
        self,
        results,
        confidence_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Extrae detecciones a un formato serializable y estable.
        Args:
            results: Salida del modelo YOLO.
            confidence_threshold (float, optional): Umbral de confianza para filtrar detecciones. Defaults to 0.0.
        Returns:
            list[dict[str, Any]]: Lista de detecciones con clase, confianza, bbox, mascara y centroide.
        """
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        #Obtenemos los elementos de las detecciones
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []

        #Construimos la lista de detecciones filtrando por el umbral de confianza
        detections: list[dict[str, Any]] = []
        for index, class_id in enumerate(classes):
            confidence = float(confidences[index])
            if confidence < confidence_threshold:
                continue

            class_idx = int(class_id)
            #Obtenemos los vertices de la mascara si esta presente
            mask_arr = masks[index] if index < len(masks) else None
            centroid = self.calculateCentroid(mask_arr) if mask_arr is not None else None

            #Calculamos el vertice de la mascara utilizando el algoritmo de Canny
            vertices = self.extractVertices(mask_arr).tolist() if mask_arr is not None else []

            detections.append(
                {
                    "class_id": class_idx,
                    "confidence": confidence,
                    "bbox": boxes[index].tolist(),
                    "mask": mask_arr.tolist() if isinstance(mask_arr, np.ndarray) else [],
                    "vertices": vertices,
                    "centroid": list(centroid) if centroid is not None else [],
                }
            )

        #Si no hay detecciones que superen el umbral de confianza, se devuelve una lista vacía
        if not detections:
            return []

        return detections

    def drawResults(self, frame: np.ndarray, results) -> np.ndarray:
        """Dibuja mascaras y centroides sobre una copia del frame.
        Args:
            frame (np.ndarray): Imagen en formato BGR sobre la que se dibujarán los resultados.
            results: Salida del modelo YOLO.
        Returns:
            np.ndarray: Imagen con las mascaras y centroides dibujados.
            Si no hay mascaras, se devuelve una copia del frame original."""
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            return frame.copy()

        #Obtenemos la mascara formato de pixel a pixel y la dibujamos sobre el frame
        masks = result.masks.data.cpu().numpy()
        annotated_frame = frame.copy()

        for mask in masks:
            # Redimensionar la máscara si es necesario
            if mask.shape != annotated_frame.shape[:2]:
                mask = cv2.resize(mask, (annotated_frame.shape[1], annotated_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            #Calculamos el centroide de la mascara y dibujamos un circulo rojo en esa posicion
            centroid = self.calculateCentroid(mask)
            colored_mask = np.zeros_like(annotated_frame)
            colored_mask[mask > 0.5] = (0, 255, 0)
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)

            if centroid is not None:
                cv2.circle(annotated_frame, centroid, 5, (0, 0, 255), -1)

        return annotated_frame

    def calculateCentroid(self, mask: np.ndarray | None) -> tuple[int, int] | None:
        """Calcula el centroide de una mascara binaria.
        Args:
            mask (np.ndarray | None): Mascara binaria de la que se calculará el
            centroide. Si es None, se devuelve None.
        Returns:
                tuple[int, int] | None: Coordenadas (x, y) del centroide o None si no se pudo calcular."""
        if mask is None:
            return None

        #Calculamos los momentos de la mascara para obtener el centroide
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None

        #Calculamos las coordenadas del centroide a partir de los momentos
        c_x = int(moments["m10"] / moments["m00"])
        c_y = int(moments["m01"] / moments["m00"])
        return (c_x, c_y)

    def processFrame(self, results) -> list[dict[str, float | int]]:
        """Calcula estadisticas de area para cada mascara detectada.
        Args:
            results: Salida del modelo YOLO.
        Returns:
            list[dict[str, float | int]]: Lista de diccionarios con las metricas de area para cada mascara detectada. Cada diccionario contiene:
            - mask_index: Índice de la mascara en el resultado del modelo.
            - mask_area: Area de la mascara en pixeles.
            - frame_area: Area total del frame en pixeles.
            - area_ratio: Ratio entre el area de la mascara y el area total del frame.
        """
        result = results[0]

        #Obtenemos la mascara formato de pixel a pixel
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        if masks is None:
            return []

        #Calculamos el area total del frame para luego calcular el ratio de area de cada mascara
        frame_height, frame_width = result.orig_shape
        frame_area = frame_width * frame_height

        #Calculamos las metricas de area para cada mascara detectada y las almacenamos en una lista de diccionarios
        mask_metrics: list[dict[str, float | int]] = []
        for index, mask in enumerate(masks):
            mask_area = int(np.sum(mask > 0.5))
            area_ratio = float(mask_area / frame_area) if frame_area else 0.0
            mask_metrics.append(
                {
                    "mask_index": index,
                    "mask_area": mask_area,
                    "frame_area": frame_area,
                    "area_ratio": area_ratio,
                }
            )

        return mask_metrics
