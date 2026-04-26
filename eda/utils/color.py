import numpy as np
import cv2

def color_to_gray_array(image_array: np.ndarray) -> np.ndarray:
    """Convierte una imagen RGB/RGBA/gris en array gris de numpy.
    Args:
        image_array (np.ndarray): Imagen de entrada como array numpy.

    Returns:
        np.ndarray: Imagen en escala de grises como array numpy.
    """
    if image_array.ndim == 2:
        gray = image_array
    elif image_array.shape[2] == 4:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    return np.asarray(gray)