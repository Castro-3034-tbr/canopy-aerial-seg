from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from eda.core.types import (
    BboxLabel,
    ImageData,
    ImagesLoaderResult,
    LabelData,
    LabelsLoaderResult,
    MaskLabel,
)

from eda.io.parses import parse_polygon_label, parse_bbox

# Extensiones válidas como frozenset para O(1) lookup
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


def load_images(path: Path) -> ImagesLoaderResult:
    """Carga imágenes válidas desde un directorio.

    Itera el directorio una sola vez, separa archivos válidos de inválidos
    y devuelve el resultado ordenado por nombre de archivo.

    Args:
        path (Path): Directorio que contiene las imágenes.

    Returns:
        ImagesLoaderResult: Imágenes cargadas y rutas incorrectas.

    Raises:
        ValueError: Si ``path`` no es un directorio existente.
    """
    if not path.is_dir():
        raise ValueError(f"La ruta no es un directorio válido: {path}")

    images: list[ImageData] = []
    incorrect: list[Path] = []

    # Path.iterdir() es más eficiente que os.listdir() + os.path.join()
    # porque ya devuelve objetos Path completos
    for file_path in sorted(path.iterdir(), key=lambda p: p.name):
        if not file_path.is_file():
            incorrect.append(file_path)
            continue

        if file_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            incorrect.append(file_path)
            continue

        # Carga la imagen usando OpenCV para obtener sus dimensiones y canales
        with Image.open(file_path) as img:
            width, height = img.size
            channels = len(img.getbands())

        images.append(ImageData(path=file_path, width=width, height=height, channels=channels))

    return ImagesLoaderResult(images=images, incorrect_images=incorrect)

def iter_images(images: list[ImageData]) -> Iterator[np.ndarray]:
    """Genera imágenes individuales a partir de una lista de ImageData.

    Args:
        images (list[ImageData]): Lista de objetos ImageData.
        batch_size (int): Tamaño del lote.

    Yields:
        np.ndarray: Imagen como array numpy.
    """

    for image_data in images:
        with Image.open(image_data.path) as img:
            yield np.array(img)




def load_labels(path: Path) -> LabelsLoaderResult:
    """Carga y valida etiquetas de segmentación desde un directorio.

    Por cada archivo ``.txt`` parsea cada línea como etiqueta de polígono,
    deriva su bounding box y agrupa los resultados. Las líneas con formato
    incorrecto se registran en ``incorrect_labels``.

    Args:
        path (Path): Directorio que contiene los archivos de etiquetas.

    Returns:
        LabelsLoaderResult: Etiquetas válidas e incorrectas.

    Raises:
        ValueError: Si ``path`` no es un directorio existente.
    """
    if not path.is_dir():
        raise ValueError(f"La ruta no es un directorio válido: {path}")

    labels: list[LabelData] = []
    incorrect: list[Path] = []

    for file_path in sorted(path.iterdir(), key=lambda p: p.name):
        if not file_path.is_file() or file_path.suffix.lower() != ".txt":
            incorrect.append(file_path)
            continue

        masks: list[MaskLabel] = []
        bboxes: list[BboxLabel] = []

        # Lectura del archivo
        lines = file_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            parsed = parse_polygon_label(line)
            if parsed is None:
                # Guardamos el archivo y la línea problemática para diagnóstico
                incorrect.append(file_path)
                continue

            #Construimos la bbox a partir del polígono
            bbox = parse_bbox(parsed[1])

            masks.append((parsed[0], parsed[1]))
            bboxes.append((parsed[0], bbox))

        labels.append(
            LabelData(path=file_path, masks=masks, bboxes=bboxes)
        )

    return LabelsLoaderResult(labels=labels, incorrect_labels=incorrect)
