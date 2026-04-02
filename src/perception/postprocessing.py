from __future__ import annotations

import cv2
import numpy as np

from src.core.constants import DEFAULT_MASK_THRESHOLD


def calculate_centroid(mask: np.ndarray | None) -> tuple[int, int] | None:
    """Calculo de los centroides de las mascaras de segmentación.

    Args:
        mask (np.ndarray | None): Mascara de segmentación binaria.

    Returns:
        tuple[int, int] | None: Coordenadas (x, y) del centroide o None si no se puede calcular.
    """
    
    #Comprobar si la mascara es nula
    if mask is None:
        return None

    #Calculo de los momentos de la mascara para obtener el centroide
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return None

    #Calculo de las coordenadas del centroide a partir de los momentos
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return (center_x, center_y)


def extract_vertices(mask: np.ndarray | None) -> np.ndarray:
    """Extrae los vértices de la mascara de segmentación utilizando contornos.

    Args:
        mask (np.ndarray | None): Mascara de segmentación binaria.

    Returns:
        np.ndarray: Array de coordenadas de los vértices del contorno más grande encontrado en la mascara. Si no se encuentra ningún contorno, devuelve un array vacío.
    """
    
    #Comprobar si la mascara es nula
    if mask is None:
        return np.array([])
    
    #Encontrar los contornos en la mascara
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    
    #Si no se encuentran contornos, devolver un array vacío
    if not contours:
        return np.array([])

    #Seleccionar el contorno más grande basado en el área y devolver sus vértices
    contour = max(contours, key=cv2.contourArea)
    return contour.squeeze()


def process_mask(masks) -> list[dict[str, float | int]]:
    """Calcula metricas realacionadas con las mascaras de segmentacion,
    Se calcula:
        - Area de cada mascara
        - Area del frame
        - Ratio entre el area de la mascara y el area del frame

    Args:
        masks (np.ndarray | None): Array de mascaras de segmentacion binarias.

    Returns:
        list[dict[str, float | int]]: Lista de diccionarios con las metricas de cada mascara
    """
    #Comprobacion de que hay mascaras para procesar
    if masks is None:
        return []
    
    #Analisis para cada mascara, FIXME: Revisar si es correcta la estructura de extracion de resultados

    frame_height, frame_width = masks.shape[1:3]
    frame_area = frame_width * frame_height
    mask_metrics = []

    #Calculo de las metricas para cada mascara
    for index, mask in enumerate(masks):
        mask_area = int(np.sum(mask > DEFAULT_MASK_THRESHOLD))
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
