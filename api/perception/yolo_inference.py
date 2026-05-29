"""Envoltura del modelo YOLO usada por el proyecto.

Proporciona utilidades para inicializar el modelo, ejecutar predicciones,
adaptar los resultados a `InferenceDetection` y dibujar resultados sobre frames.
"""

from __future__ import annotations

from typing import Any

import logging
import cv2
import numpy as np
import ultralytics

from api.core.constants import (
    CENTROID_COLOR,
    CENTROID_RADIUS,
    MASK_COLOR,
    MASK_OVERLAY_ALPHA,
    MASK_OVERLAY_BETA,
    MASK_OVERLAY_GAMMA,
)
from common.types.geometry import BoundingBox
from common.types.model import InferenceDetection
from api.perception.postprocessing import calculate_centroid, extract_vertices
from common.types.geometry import Coordinates
from common.types.media import Imagen
from common.types.model import YoloModel

from api.perception.analisis import remove_overlap


def initialize_model(model_path: str, device: str = "cpu") -> YoloModel:
    """Inicializa y devuelve una instancia del modelo YOLO.

    Args:
        model_path (str): Ruta al archivo de pesos del modelo.
        device (str, optional): Dispositivo para inferencia (ej. 'cpu' o 'cuda').

    Returns:
        YoloModel: Modelo YOLO cargado y preparado en el dispositivo indicado.
    """
    model = ultralytics.YOLO(model_path)
    model.to(device)
    return model

def predict(
    model: YoloModel,
    frame: Imagen,
    confidence_threshold: float = 0.0,
    overlap: tuple[float, float] = (0.1, 0.1),
    debug: bool = False,
) -> list[InferenceDetection]:
    """Ejecuta inferencia sobre un frame y adapta los resultados.

    Args:
        model (YoloModel): Modelo preparado para inferencia.
        frame (Imagen): Frame en formato BGR (np.ndarray).
        confidence_threshold (float, optional): Umbral mínimo de confianza.
        overlap (tuple[float,float], optional): Fracción a recortar para eliminar solapes.
        debug (bool, optional): Si True, activa salida verbosa del modelo.

    Returns:
        list[InferenceDetection]: Lista de detecciones serializables.
    """
    results = model(frame, conf=confidence_threshold, verbose=debug)
    return extract_detections(results=results, overlap=overlap)

def extract_detections(results: Any, overlap: tuple[float, float]) -> list[InferenceDetection]:
    """Adapta los resultados del modelo Ultralytics a `InferenceDetection`.

    Args:
        results (Any): Salida del modelo Ultralytics para un solo frame.
        overlap (tuple[float,float]): Fracción a recortar en (x, y) para máscaras.

    Returns:
        list[InferenceDetection]: Lista de detecciones con máscaras y centroides.
    """
    if not results:
        return []

    result_yolo = results[0]
    boxes = result_yolo.boxes.xyxy.cpu().numpy()
    classes = result_yolo.boxes.cls.cpu().numpy()
    confidences = result_yolo.boxes.conf.cpu().numpy()
    masks = (
        result_yolo.masks.data.cpu().numpy()
        if result_yolo.masks is not None
        else []
    )
    frame_height, frame_width = result_yolo.orig_shape[:2]

    detections: list[InferenceDetection] = []
    masks_iter = (
        masks
        if len(masks) != 0
        else [
            np.zeros((frame_height, frame_width), dtype=np.uint8)
            for _ in range(len(boxes))
        ]
    )

    for index, class_id in enumerate(classes):
        mask_cropped = remove_overlap(mask=masks_iter[index], overlap=overlap)

        x1, y1, x2, y2 = boxes[index]
        x1_norm = float(x1 / frame_width)
        y1_norm = float(y1 / frame_height)
        x2_norm = float(x2 / frame_width)
        y2_norm = float(y2 / frame_height)
        vertices = extract_vertices(mask=mask_cropped)

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
            frame_mask=mask_cropped,
            centroid=calculate_centroid(polygon=vertices),
        )
        detections.append(detection)

    return detections

def draw_results(frame: Imagen, results: list[InferenceDetection]) -> Imagen:
    """Dibuja centroides y máscaras sobre un frame en BGR.

    Args:
        frame (Imagen): Imagen de entrada (H x W x 3) en uint8.
        results (list[InferenceDetection]): Detecciones con centroid y máscara.

    Returns:
        Imagen: Imagen con overlays aplicados (modificación in-place del frame).
    """

    logger = logging.getLogger(__name__)

    # Asegura tipo y canales compatibles con OpenCV
    if frame.dtype != np.uint8:
        try:
            frame = frame.astype(np.uint8)
        except Exception:
            logger.exception("No se pudo convertir el frame a uint8; se retorna sin modificar")
            return frame

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] < 3:
        # Extiende canales menores a 3 duplicando el último canal
        channels = [frame[..., i] if i < frame.shape[2] else frame[..., -1] for i in range(3)]
        frame = np.stack(channels, axis=-1)

    height, width = frame.shape[:2]

    # Máscara acumulada (optimización: una sola por frame), uint8
    combined_mask = np.zeros_like(frame, dtype=np.uint8)

    # Normaliza y valida constantes visuales
    try:
        centroid_color = tuple(int(c) for c in CENTROID_COLOR)
        if len(centroid_color) != 3:
            raise ValueError
    except Exception:
        logger.warning("CENTROID_COLOR inválido; usando (0,0,255)")
        centroid_color = (0, 0, 255)

    try:
        mask_color = tuple(int(c) for c in MASK_COLOR)
        if len(mask_color) != 3:
            raise ValueError
    except Exception:
        logger.warning("MASK_COLOR inválido; usando (203,192,255)")
        mask_color = (203, 192, 255)

    try:
        centroid_radius = int(max(0, CENTROID_RADIUS))
    except Exception:
        centroid_radius = 5

    for detection in results:
        # Centroide
        if detection.centroid is not None:
            # Normaliza centroides a rango válido
            cx = int(np.clip(float(detection.centroid.x) * width, 0, width - 1))
            cy = int(np.clip(float(detection.centroid.y) * height, 0, height - 1))
            cv2.circle(frame, (cx, cy), centroid_radius, centroid_color, cv2.FILLED)

        # Máscara
        vertices = detection.mask

        if vertices and len(vertices) >= 3:
            pts = np.array(
                [
                    (
                        int(np.clip(p.x * width, 0, width - 1)),
                        int(np.clip(p.y * height, 0, height - 1)),
                    )
                    for p in vertices
                ],
                dtype=np.int32,
            )
        else:
            return frame

        try:
            cv2.fillPoly(combined_mask, [pts], mask_color)
        except Exception:
            logger.exception("Error al dibujar la máscara; se omite esta detección")

    # Aplica la máscara combinada con transparencia sobre el frame original.
    try:
        cv2.addWeighted(
            frame,
            MASK_OVERLAY_BETA,
            combined_mask,
            MASK_OVERLAY_ALPHA,
            MASK_OVERLAY_GAMMA,
            dst=frame,  # <-- clave: in-place
        )
    except Exception:
        logger.exception("Error aplicando overlay de máscara; se devuelve el frame sin overlay")

    return frame
