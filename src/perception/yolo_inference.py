"""Envoltura del modelo YOLO usada por el proyecto."""

from __future__ import annotations

from typing import Any

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
from src.perception.postprocessing import calculate_centroid, extract_vertices


class YoloInference:
    """Encapsula el modelo YOLO y sus utilidades de postproceso."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """Inicializa el modelo YOLO con la ruta y el dispositivo indicados."""
        # Carga el modelo una sola vez y lo mueve al dispositivo elegido.
        self.model = ultralytics.YOLO(model_path)
        self.model.to(device)

    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.0,
        debug: bool = False,
    ):
        """Realiza la inferencia sobre un frame."""
        # Mantiene la salida nativa de Ultralytics para reutilizar su API.
        return self.model(frame, conf=confidence_threshold, verbose=debug)

    def extract_detections(self, results: Any) -> list[dict]:
        """Adapta los resultados del modelo a un formato serializable."""
        # Si no hay resultados, devuelve una lista vacia compatible con JSON.
        if not results:
            return []

        # El pipeline procesa un unico frame por llamada.
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        masks = (
            result.masks.data.cpu().numpy()
            if result.masks is not None
            else []
        )

        detections = []
        # Convierte cada deteccion a tipos simples para serializacion.
        for index, class_id in enumerate(classes):
            confidence = float(confidences[index])
            mask_array = masks[index] if index < len(masks) else None
            centroid = calculate_centroid(mask_array)
            vertices = (
                extract_vertices(mask_array).tolist()
                if mask_array is not None
                else []
            )
            detections.append(
                {
                    "class_id": int(class_id),
                    "confidence": confidence,
                    "bbox": boxes[index].tolist(),
                    "mask": (
                        mask_array.tolist()
                        if isinstance(mask_array, np.ndarray)
                        else []
                    ),
                    "vertices": vertices,
                    "centroid": list(centroid) if centroid is not None else [],
                }
            )

        return detections

    def draw_results(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """Dibuja mascaras y centroides sobre el frame original."""
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            return frame.copy()

        masks = result.masks.data.cpu().numpy()
        annotated_frame = frame.copy()

        # Superpone cada mascara y marca su centroide si existe.
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
