"""Envoltura del modelo YOLO usada por el proyecto."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import ultralytics

from api.core.constants import (
    CENTROID_COLOR,
    CENTROID_RADIUS,
    DEFAULT_MASK_THRESHOLD,
    MASK_COLOR,
    MASK_OVERLAY_ALPHA,
    MASK_OVERLAY_BETA,
    MASK_OVERLAY_GAMMA,
)
from common.types.geometry import BoundingBox
from common.types.model import InferenceDetection
from api.perception.postprocessing import calculate_centroid, extract_vertices
from common.types.geometry import Coordinates
from common.types.media import FrameArray


def initialize_model(model_path: str, device: str = "cpu") -> ultralytics.YOLO:
    """Inicializa el modelo YOLO con la ruta y el dispositivo indicados."""
    # Carga el modelo una sola vez y lo mueve al dispositivo elegido.
    model = ultralytics.YOLO(model_path)
    model.to(device)
    return model

def predict(
    model: ultralytics.YOLO,
    frame: FrameArray,
    confidence_threshold: float = 0.0,
    debug: bool = False,
) -> Any:
    """Realiza la inferencia sobre un frame."""
    # Mantiene la salida nativa de Ultralytics para reutilizar su API.
    results = model(frame, conf=confidence_threshold, verbose=debug)

    #Conversion de los rultados a un formato comun
    return extract_detections(results=results)

def extract_detections(results: Any) -> list[InferenceDetection]:
    """Adapta los resultados del modelo a un formato serializable."""
    # Si no hay resultados, devuelve una lista vacia compatible con JSON.
    if not results:
        return []

    # Extraccion de los resultados de una iteraccion
    result_yolo = results[0]
    boxes = result_yolo.boxes.xyxy.cpu().numpy()
    classes = result_yolo.boxes.cls.cpu().numpy()
    confidences = result_yolo.boxes.conf.cpu().numpy()
    masks = result_yolo.masks.data.cpu().numpy() if result_yolo.masks is not None else []
    frame_height, frame_width = result_yolo.orig_shape[:2]

    detections: list[InferenceDetection] = []
    # Convierte cada deteccion a tipos simples para serializacion.
    for index, class_id in enumerate(classes):
        x1, y1, x2, y2 = boxes[index]
        x1_norm = float(x1 / frame_width)
        y1_norm = float(y1 / frame_height)
        x2_norm = float(x2 / frame_width)
        y2_norm = float(y2 / frame_height)
        vertices = extract_vertices(mask=masks[index])

        detection = InferenceDetection(
            class_id=int(class_id),
            confidence=float(confidences[index]),
            bbox=BoundingBox(
                p1=Coordinates(x=x1_norm, y=y1_norm),
                p2=Coordinates(x=x2_norm, y=y2_norm),
                width=float((x2 - x1) / frame_width),
                height=float((y2 - y1) / frame_height),
            ),
            mask=vertices,
            frame_mask=masks[index],
            centroid=calculate_centroid(polygon=vertices)
        )
        detections.append(detection)

    return detections

def draw_results(frame: FrameArray, results: list[InferenceDetection]) -> FrameArray:
    """Dibuja mascaras y centroides sobre el frame original."""
    # Obtencion del tamaño del frame para escalar las coordenadas normalizadas a pixeles.
    height, width = frame.shape[:2]

    # Iteraccion por cada deteccion
    for detection in results:
        pt1 = (
            int(detection.bbox.p1.x * width),
            int(detection.bbox.p1.y * height),
        )
        pt2 = (
            int(detection.bbox.p2.x * width),
            int(detection.bbox.p2.y * height),
        )

        # Dibujo del bounding box
        cv2.rectangle(
            frame,
            pt1,
            pt2,
            (0, 255, 0),
            2,
        )

        # Dibujo del centroide
        cv2.circle(
            frame,
            (int(detection.centroid.x * width), int(detection.centroid.y * height)),
            CENTROID_RADIUS,
            CENTROID_COLOR,
            -1,
        )

        # Dibujo de la mascara
        colored_mask = np.zeros_like(frame)
        colored_mask[detection.frame_mask > DEFAULT_MASK_THRESHOLD] = MASK_COLOR
        frame = np.asarray(
            cv2.addWeighted(
            frame,
            MASK_OVERLAY_BETA,
            colored_mask,
            MASK_OVERLAY_ALPHA,
            MASK_OVERLAY_GAMMA,
            ),
            dtype=np.uint8,
        )

        # Dibujo de la etiqueta de clase y confianza
        label = f"{detection.class_id}: {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (pt1[0], pt1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return np.asarray(frame, dtype=np.uint8)
