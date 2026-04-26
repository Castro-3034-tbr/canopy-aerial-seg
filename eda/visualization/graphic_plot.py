from __future__ import annotations

import logging
import os
from os import PathLike
from collections.abc import Sequence
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from eda.core.types import MetricValues

NumericSeries = Sequence[float | int]

logger = logging.getLogger(__name__)


def _ensure_parent_dir(output_path: str | PathLike[str]) -> None:
    """Crea la carpeta destino si no existe."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def plot_continuous_distribution(
    values: NumericSeries,
    output_path: str | PathLike[str],
    title: str,
    x_label: str,
    bins: int = 30,
    use_log_x: bool = False,
) -> None:
    """Genera histograma de una métrica continua y opcionalmente usa escala log."""
    if not values:
        logger.warning("No hay datos para generar el gráfico: %s", title)
        return

    _ensure_parent_dir(output_path)
    data = np.asarray(values, dtype=np.float64)

    # Si se solicita escala logarítmica, filtramos valores no positivos
    if use_log_x:
        data = data[data > 0]
        if data.size == 0:
            logger.warning("No hay valores positivos para escala log en: %s", title)
            return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficamos el histograma con densidad para mostrar la forma de la distribución
    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.7,
        color="#4c72b0",
        edgecolor="black",
        label="Datos",
    )

    # Agregamos la curva de densidad normal teórica para referencia visual
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        x_vals = np.linspace(np.min(data), np.max(data), 200)
        normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_vals - mean) / std) ** 2
        )
        ax.plot(x_vals, normal_pdf, "r-", linewidth=2, label="Normal teórica")

    if use_log_x:
        ax.set_xscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Densidad")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_boxplot(
    values: NumericSeries,
    output_path: str | PathLike[str],
    title: str,
    y_label: str,
) -> None:
    """Genera un boxplot para resumir dispersión y outliers."""
    if not values:
        logger.warning("No hay datos para generar boxplot: %s", title)
        return

    _ensure_parent_dir(output_path)

    # Convertimos a array de NumPy para asegurar compatibilidad con matplotlib
    data = np.asarray(values, dtype=np.float64)

    # Creamos el boxplot con estilo personalizado
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, vert=True, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks([])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_count_bars(
    counts: Mapping[str, int],
    output_path: str | PathLike[str],
    title: str,
    x_label: str,
    y_label: str,
    top_n: int = 15,
) -> None:
    """Genera barras horizontales para conteos categóricos."""
    if not counts:
        logger.warning("No hay datos para generar barras: %s", title)
        return

    _ensure_parent_dir(output_path)

    # Ordenamos por frecuencia y limitamos a top_n categorías, agrupando el resto como "Otros"
    sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if top_n > 0 and len(sorted_items) > top_n:
        top_items = sorted_items[:top_n]
        remaining_sum = sum(value for _, value in sorted_items[top_n:])
        top_items.append(("Otros", remaining_sum))
    else:
        top_items = sorted_items

    # Separamos etiquetas y valores para el gráfico
    labels = [key for key, _ in top_items]
    values = [value for _, value in top_items]

    # Ajustamos la altura de la figura según el número de categorías
    fig_height = max(5, 0.5 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(labels, values, color="#4c72b0", edgecolor="black")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Ajustamos el layout para evitar solapamientos
    fig.tight_layout()

    # Guardamos el gráfico
    fig.savefig(output_path)
    plt.close(fig)