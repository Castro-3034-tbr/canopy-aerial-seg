"""Divide el dataset en conjuntos de entrenamiento, validacion y prueba."""

import random
import shutil
from pathlib import Path
from typing import Iterable


def read_files(dataset_path: Path) -> list[str]:
    """Devuelve los nombres de archivo contenidos en un directorio.

    Args:
        dataset_path (Path): Ruta del directorio a inspeccionar.

    Returns:
        list[str]: Lista de nombres de archivo presentes en el directorio.
    """
    return [f.name for f in dataset_path.iterdir() if f.is_file()]


def make_pairs(
    images: Iterable[str],
    labels: Iterable[str],
) -> list[tuple[str, str]]:
    """Empareja imágenes con etiquetas usando su nombre base.

    Utiliza un diccionario de búsqueda para evitar bucles anidados y
    solo devuelve pares donde ambos archivos existen.

    Args:
        images (Iterable[str]): Lista de archivos de imagen.
        labels (Iterable[str]): Lista de archivos de etiqueta.

    Returns:
        list[tuple[str, str]]: Lista de tuplas ``(imagen, etiqueta)``.
    """
    # Construcción de un diccionario para búsqueda rápida de etiquetas por nombre base
    label_map = {Path(lbl).stem: lbl for lbl in labels}

    # Imágenes sin etiqueta correspondiente quedan excluidas por el filtro
    return [
        (img, label_map[stem])
        for img in images
        if (stem := Path(img).stem) in label_map
    ]


def split_dataset(
    dataset_path: str,
    output_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> None:
    """Divide un dataset YOLO en entrenamiento, validación y prueba.

    Args:
        dataset_path (str): Ruta al directorio raíz del dataset.
        output_path (str): Ruta donde se crearán los conjuntos divididos.
        train_ratio (float): Proporción del conjunto de entrenamiento.
        val_ratio (float): Proporción del conjunto de validación.
        test_ratio (float): Proporción del conjunto de prueba.

    Raises:
        ValueError: Si las proporciones no suman 1.0 o si el dataset
            está vacío.
        RuntimeError: Si ocurre un error durante la lectura, mezcla
            o copia de archivos.
    """

    dataset_root = Path(dataset_path)
    output_root = Path(output_path)

    # Validacion de proporciones
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError("Las proporciones deben sumar 1.0")

    # Listar archivos en el dataset
    try:
        images = read_files(dataset_path=dataset_root / "images")
        labels = read_files(dataset_path=dataset_root / "labels")
        if not images or not labels:
            raise ValueError("El dataset esta vacio")
    except Exception as exc:
        raise RuntimeError(f"Error al listar archivos del dataset: {exc}") from exc

    # Mezclamos los archivos para evitar sesgos
    try:
        pairs = make_pairs(images=images, labels=labels)
        if not pairs:
            raise ValueError(
                "No se encontraron pares imagen-etiqueta. "
                "Verifica que los nombres base entre images/ y labels/ coincidan."
            )
        random.shuffle(pairs)
    except Exception as exc:
        raise RuntimeError(f"Error durante la mezcla de archivos: {exc}") from exc

    # Division del dataset
    try:
        total_files = len(pairs)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]

        # Creacion de directorios de salida
        for split_name in ("train", "val", "test"):
            (output_root / split_name / "images").mkdir(
                parents=True,
                exist_ok=True,
            )
            (output_root / split_name / "labels").mkdir(
                parents=True,
                exist_ok=True,
            )

        # Copia de archivos a los directorios correspondientes
        for img, lbl in train_pairs:
            shutil.copy2(
                dataset_root / "images" / img,
                output_root / "train" / "images" / img,
            )
            shutil.copy2(
                dataset_root / "labels" / lbl,
                output_root / "train" / "labels" / lbl,
            )

        for img, lbl in val_pairs:
            shutil.copy2(
                dataset_root / "images" / img,
                output_root / "val" / "images" / img,
            )
            shutil.copy2(
                dataset_root / "labels" / lbl,
                output_root / "val" / "labels" / lbl,
            )

        for img, lbl in test_pairs:
            shutil.copy2(
                dataset_root / "images" / img,
                output_root / "test" / "images" / img,
            )
            shutil.copy2(
                dataset_root / "labels" / lbl,
                output_root / "test" / "labels" / lbl,
            )

    except Exception as exc:
        raise RuntimeError(f"Error durante la division del dataset: {exc}") from exc
