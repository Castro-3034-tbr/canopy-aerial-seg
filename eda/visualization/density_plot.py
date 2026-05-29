from __future__ import annotations

import logging
import os
from os import PathLike

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.types.geometry import Coordinates

logger = logging.getLogger(__name__)


def plot_density_center(
    labels_centers: list[Coordinates],
    output_path: str | PathLike[str],
) -> None:
    """Genera y guarda un gráfico de densidad de centros de etiquetas."""

    # Obtenemos las posiciones X e Y de las etiquetas
    if not labels_centers:
        logger.warning(
            "No se encontraron centros de etiquetas para generar el gráfico de densidad."
        )
        return

    # Convertimos las posiciones de las etiquetas a un array de NumPy para facilitar el manejo
    x: list[float] = []
    y: list[float] = []
    for center in labels_centers:
        x.append(center.x)
        y.append(center.y)

    # Creamos el grafico de densidad
    plt.figure(figsize=(8, 6))
    plt.hexbin(x, y, gridsize=30, cmap="Blues")
    plt.colorbar(label="Número de etiquetas")
    plt.xlabel("Posición horizontal (x_center)")
    plt.ylabel("Posición vertical (y_center)")
    plt.title("Densidad de posiciones de etiquetas")

    # Guardamos el gráfico en la ruta especificada
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
