"""
Módulo de utilidades para el cálculo del Ground Sample Distance (GSD) en imágenes aéreas.
"""

from __future__ import annotations

def calculate_gsd(image_width:tuple[int, int], sensor_width: tuple[float, float], focal_length:float, altitude:float) -> tuple[float, float]:
    """
    Calculo del Ground Sample Distance (GSD) para una imagen aérea.

    Args:
        image_width (tuple[int, int]): Ancho de la imagen en píxeles (ancho, alto).
        sensor_width (tuple[float, float]): Ancho del sensor en milímetros (ancho, alto).
        focal_length (float): Longitud focal de la cámara en milímetros.
        altitude (float): Altitud de vuelo en metros.
    """
    
    # Conversion de medidas a metros
    sensor_width_m = sensor_width[0] / 1000.0
    sensor_height_m = sensor_width[1] / 1000.0
    focal_length_m = focal_length / 1000.0
    
    # Calculo del GSD tanto ancho como a alto
    gsd_width = (altitude * sensor_width_m) / (focal_length_m * image_width[0])
    gsd_height = (altitude * sensor_height_m) / (focal_length_m * image_width[1])
    
    return (gsd_width , gsd_height)