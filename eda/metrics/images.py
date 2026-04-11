"""Métricas EDA para imágenes."""

#TODO: Revisar que esten correctos

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Sequence

import cv2
import numpy as np

from eda.core.types import AspectRatioCounts, ImageData, ImageSize, MetricValues


def count_image_types(images: Sequence[ImageData]) -> dict[str, int]:
    """Cuenta imágenes por extensión de archivo."""
    return dict(Counter(image.path.suffix.lower() for image in images))


def count_image_sizes(images: Sequence[ImageData]) -> dict[ImageSize, int]:
    """Cuenta imágenes por tamaño `(ancho, alto)`."""
    sizes = ((image.data.shape[1], image.data.shape[0]) for image in images)
    return dict(Counter(sizes))


def count_image_aspect_ratios(images: Sequence[ImageData]) -> AspectRatioCounts:
    """Cuenta imágenes por relación de aspecto simplificada."""
    ratios: list[str] = []
    for image in images:
        width = image.data.shape[1]
        height = image.data.shape[0]
        fraction = Fraction(width, height)
        ratios.append(f"{fraction.numerator}/{fraction.denominator}")

    return dict(Counter(ratios))


def compute_images_brightness(images: Sequence[ImageData]) -> MetricValues:
    """Calcula el brillo medio de cada imagen."""
    brightness_values: MetricValues = []

    for image in images:
        gray = cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))

    return brightness_values


def compute_images_contrast(images: Sequence[ImageData]) -> MetricValues:
    """Calcula el contraste de cada imagen como desviación típica en gris."""
    contrast_values: MetricValues = []

    for image in images:
        gray = cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY)
        contrast_values.append(float(np.std(gray)))

    return contrast_values


def compute_images_blur(images: Sequence[ImageData]) -> MetricValues:
    """Calcula el desenfoque usando la varianza del Laplaciano."""
    blur_values: MetricValues = []

    for image in images:
        gray = cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY)
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    return blur_values
