"""Utilidades para inferencia, visualizacion y postproceso con YOLO."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import ultralytics


class ClassYOLO:
    """Encapsula el uso del modelo YOLO dentro del proyecto."""

    def __init__(self, model_path: str = "yolov8m.pt", device: str = "cpu") -> None:
        self.model = ultralytics.YOLO(model_path)
        self.model.to(device)

    def predict(self, frame: np.ndarray):
        """Ejecuta inferencia sobre un frame BGR."""
        return self.model(frame)

    def extractDetections(
        self,
        results,
        confidence_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Extrae detecciones a un formato serializable y estable."""
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None

        detections: list[dict[str, Any]] = []
        for index, class_id in enumerate(classes):
            confidence = float(confidences[index])
            if confidence < confidence_threshold:
                continue

            class_idx = int(class_id)
            mask = masks[index] if masks is not None else None
            centroid = self.calculateCentroid(mask) if mask is not None else None
            detections.append(
                {
                    "class_id": class_idx,
                    "confidence": confidence,
                    "bbox": boxes[index].tolist(),
                    "mask": mask.tolist() if mask is not None else None,
                    "centroid": list(centroid) if centroid is not None else None,
                }
            )

        return detections

    def drawResults(self, frame: np.ndarray, results) -> np.ndarray:
        """Dibuja mascaras y centroides sobre una copia del frame."""
        result = results[0]
        if result.masks is None:
            return frame.copy()

        masks = result.masks.data.cpu().numpy()
        annotated_frame = frame.copy()

        for mask in masks:
            centroid = self.calculateCentroid(mask)
            colored_mask = np.zeros_like(annotated_frame)
            colored_mask[mask > 0.5] = (0, 255, 0)
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)

            if centroid is not None:
                cv2.circle(annotated_frame, centroid, 5, (0, 0, 255), -1)

        return annotated_frame

    def calculateCentroid(self, mask: np.ndarray | None) -> tuple[int, int] | None:
        """Calcula el centroide de una mascara binaria."""
        if mask is None:
            return None

        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None

        c_x = int(moments["m10"] / moments["m00"])
        c_y = int(moments["m01"] / moments["m00"])
        return (c_x, c_y)

    def processFrame(self, results) -> list[dict[str, float | int]]:
        """Calcula estadisticas de area para cada mascara detectada."""
        result = results[0]
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        if masks is None:
            return []

        frame_height, frame_width = result.orig_shape
        frame_area = frame_width * frame_height
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