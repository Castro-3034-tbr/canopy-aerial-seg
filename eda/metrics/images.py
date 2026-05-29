"""Métricas EDA para imágenes."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Sequence

import cv2
import numpy as np

from eda.core.types import ImageData
from common.types.geometry import ImageSize
from eda.io.loaders import iter_images
from eda.utils.color import color_to_gray_array


def count_image_types(images: Sequence[ImageData]) -> dict[str, int]:
    """Cuenta imágenes por extensión de archivo."""
    return dict(Counter(image.path.suffix.lower() for image in images))


def count_image_sizes(images: Sequence[ImageData]) -> dict[ImageSize, int]:
    """Cuenta imágenes por tamaño `(ancho, alto)`."""
    sizes = {}

    for image in images:
        size = (image.width, image.height)
        sizes[size] = sizes.get(size, 0) + 1

    return sizes



def count_image_aspect_ratios(images: Sequence[ImageData]) -> dict[str, int]:
    """Cuenta imágenes por relación de aspecto simplificada."""
    ratios: list[str] = []
    for image in images:
        width = image.width
        height = image.height
        fraction = Fraction(width, height)
        ratios.append(f"{fraction.numerator}/{fraction.denominator}")

    return dict(Counter(ratios))


def compute_images_brightness(images: Sequence[ImageData]) -> list[float]:
    """Calcula el brillo medio de cada imagen."""
    brightness_values: list[float] = []

    for image_array in iter_images(list(images)):
        gray = color_to_gray_array(image_array)
        brightness_values.append(float(np.mean(gray)))

    return brightness_values


def compute_images_contrast(images: Sequence[ImageData]) -> list[float]:
    """Calcula el contraste de cada imagen como desviación típica en gris."""
    contrast_values: list[float] = []

    for image_array in iter_images(list(images)):
        gray = color_to_gray_array(image_array)
        contrast_values.append(float(np.std(gray)))

    return contrast_values


def compute_images_blur(images: Sequence[ImageData]) -> list[float]:
    """Calcula el desenfoque usando la varianza del Laplaciano."""
    blur_values: list[float] = []

    for image_array in iter_images(list(images)):
        gray = color_to_gray_array(image_array)
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    return blur_values
