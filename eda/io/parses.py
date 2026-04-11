from __future__ import annotations

import numpy as np

from eda.core.types import BoundingBox, Coordinates, MaskLabel, Polygon
import logging

logger = logging.getLogger(__name__)

def parse_polygon_label(label: str) -> MaskLabel:
    """Parsea una línea de etiqueta YOLO de segmentación.

    Formato esperado: ``clase x1 y1 x2 y2 ... xN yN``
    donde clase es un entero >= 0 y todas las coordenadas
    son valores normalizados en [0.0, 1.0].

    Requisitos mínimos:
        - Al menos 3 puntos (6 coordenadas) para formar un polígono.
        - Número de coordenadas par: (len - 1) % 2 == 0.
        - Todas las coordenadas en [0.0, 1.0].
        - class_id >= 0.

    Args:
        label (str): Línea de texto con la etiqueta YOLO.

    Returns:
        tuple[int, Polygon]: ``(class_id, lista de Coordinates)`` si es válida.
        None: Si el formato es incorrecto o los valores están fuera de rango.
    """
    #Division de la linea
    parts = label.strip().split()

    # Mínimo: 1 class_id + 6 coordenadas (3 puntos) = 7 tokens
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        logger.debug(f"Formato incorrecto en línea: {label}")
        return MaskLabel((1, Polygon([])))

    try:
        # Conversion de class_id a entero y coordenadas a float
        class_id = int(float(parts[0]))
        raw_coords = list(map(float, parts[1:]))
    except ValueError:
        logger.debug(f"Error al convertir coordenadas en línea: {label}")
        return MaskLabel((1, Polygon([])))

    if class_id < 0:
        # class_id negativo no es válido para clasificación de objetos
        logger.debug(f"ID de clase negativo en línea: {label}")
        return MaskLabel((1, Polygon([])))

    # Verificar que estan todas en el rango [0.0, 1.0] antes de construir Coordinates
    coords_array = np.array(raw_coords, dtype=np.float64).reshape(-1, 2)
    if not np.all((coords_array >= 0.0) & (coords_array <= 1.0)):
        logger.debug(f"Coordenadas fuera de rango en línea: {label}")
        return MaskLabel((1, Polygon([])))

    #Conversion a lista de coordenadas normalizadas
    try:
        polygon: Polygon = [
            Coordinates(x=float(row[0]), y=float(row[1]))
            for row in coords_array
        ]
    except ValueError as exc:
        logger.debug(f"Error al construir Coordinates: {exc}")
        return MaskLabel((1, Polygon([])))

    return MaskLabel((class_id, polygon))

def parse_bbox(polygon: Polygon) -> BoundingBox:
    """Deriva una bounding box mínima a partir de un polígono de Coordinates.

    Usa operaciones vectorizadas con numpy para eficiencia. La bbox
    resultante es el rectángulo mínimo que contiene todos los puntos
    del polígono.

    Args:
        polygon (Polygon): Lista de ``Coordinates`` normalizadas en [0.0, 1.0].

    Returns:
        BoundingBox: Caja delimitadora validada por Pydantic.

    Raises:
        ValueError: Si el polígono está vacío.
    """
    if not polygon:
        raise ValueError("El polígono está vacío — no se puede derivar una bbox.")

    # Extraemos las coordenadas x e y
    xs = np.array([c.x for c in polygon], dtype=np.float64)
    ys = np.array([c.y for c in polygon], dtype=np.float64)

    #  Calculamos los extremos del bounding box
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    # max(0.0, ...) evita valores negativos por errores de precisión float
    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        width=max(0.0, x_max - x_min),
        height=max(0.0, y_max - y_min),
    )