"""Divide el dataset en conjuntos de entrenamiento, validacion y prueba."""

import random
import shutil
from pathlib import Path


def read_files(dataset_path: Path) -> list:
    """Lectura de archivos del dataset.

    Args:
        dataset_path (str): Ruta al dataset original.

    Returns:
        list: Lista de archivos en el dataset.
    """
    return [f.name for f in dataset_path.iterdir() if f.is_file()]


def make_pairs(images: list, labels: list) -> list:
    """Empareja archivos de imagen con sus etiquetas correspondientes por nombre base.

    Utiliza un diccionario de búsqueda para lograr complejidad O(n+m) en lugar
    del enfoque O(n*m) de bucles anidados. Solo se incluyen pares donde ambos
    archivos existen.

    Args:
        images (list): Lista de rutas (str o Path) de archivos de imagen.
        labels (list): Lista de rutas (str o Path) de archivos de etiqueta.

    Returns:
        list[tuple]: Lista de tuplas ``(imagen, etiqueta)`` para cada par válido.
                    El orden sigue el de la lista ``images``.

    Raises:
        TypeError: Si ``images`` o ``labels`` no son iterables.

    Example:
        >>> images = ["data/cat.jpg", "data/dog.jpg", "data/bird.jpg"]
        >>> labels = ["data/cat.txt", "data/dog.txt"]
        >>> make_pairs(images, labels)
        [('data/cat.jpg', 'data/cat.txt'), ('data/dog.jpg', 'data/dog.txt')]
    """
    # Diccionario stem -> ruta completa para O(1) lookup por nombre base
    label_map: dict[str, any] = {Path(lbl).stem: lbl for lbl in labels}

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
    """
    Divide el dataset en conjuntos de entrenamiento, validacion y prueba.

    Args:
        dataset_path (str): Ruta al dataset original.
        output_path (str): Ruta donde se guardaran los conjuntos divididos.
        train_ratio (float): Proporcion del conjunto de entrenamiento.
        val_ratio (float): Proporcion del conjunto de validacion.
        test_ratio (float): Proporcion del conjunto de prueba.

    Raises:
        ValueError: Si las proporciones no suman 1.0 o si el dataset esta vacio.
        RuntimeError: Si ocurre un error durante la division del dataset.
    """

    dataset_root = Path(dataset_path)
    output_root = Path(output_path)

    # Validacion de proporciones
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError("Las proporciones deben sumar 1.0")

    # Listar archivos en el dataset
    try:
        images = read_files(dataset_root / "images")
        labels = read_files(dataset_root / "labels")
        if not images or not labels:
            raise ValueError("El dataset esta vacio")
    except Exception as exc:
        raise RuntimeError(f"Error al listar archivos del dataset: {exc}") from exc

    # Mezaclamos los archivos para evitar sesgos
    try:
        # Creacion de las parejas entre imagenes y etiquetas
        pairs = make_pairs(images, labels)

        # Mezclado de la lista de parejas
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
        (output_root / "train" / "images").mkdir(parents=True, exist_ok=True)
        (output_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (output_root / "val" / "images").mkdir(parents=True, exist_ok=True)
        (output_root / "val" / "labels").mkdir(parents=True, exist_ok=True)
        (output_root / "test" / "images").mkdir(parents=True, exist_ok=True)
        (output_root / "test" / "labels").mkdir(parents=True, exist_ok=True)

        # Copia de archivos a los directorios correspondientes
        for img, lbl in train_pairs:
            shutil.copy2(
                dataset_root / "images" / img, output_root / "train" / "images" / img
            )
            shutil.copy2(
                dataset_root / "labels" / lbl, output_root / "train" / "labels" / lbl
            )

        for img, lbl in val_pairs:
            shutil.copy2(
                dataset_root / "images" / img, output_root / "val" / "images" / img
            )
            shutil.copy2(
                dataset_root / "labels" / lbl, output_root / "val" / "labels" / lbl
            )

        for img, lbl in test_pairs:
            shutil.copy2(
                dataset_root / "images" / img, output_root / "test" / "images" / img
            )
            shutil.copy2(
                dataset_root / "labels" / lbl, output_root / "test" / "labels" / lbl
            )

    except Exception as exc:
        raise RuntimeError(f"Error durante la division del dataset: {exc}") from exc
