"""Escritura de resultados de análisis EDA."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np

from eda.core.types import AnalysisResult


def _write_summary_stats(file_obj, values: list[float] | list[int], title: str) -> None:
    """Escribe estadísticas básicas para una lista de valores."""
    file_obj.write(
        f"{title}:\n"
        f"\tMedia: {np.mean(values):.2f}\n"
        f"\tMediana: {np.median(values):.2f}\n"
        f"\tMínimo: {np.min(values):.2f}\n"
        f"\tMáximo: {np.max(values):.2f}\n"
        f"\tDesviación estándar: {np.std(values):.2f}\n"
        f"\tCuartiles: 25%={np.percentile(values, 25):.2f}, 50%={np.percentile(values, 50):.2f}, 75%={np.percentile(values, 75):.2f}\n"
        f"\n"
    )


def _write_dict_counts(file_obj, title: str, values: dict) -> None:
    """Escribe un diccionario de conteos ordenado por frecuencia."""
    file_obj.write(f"{title}:\n")
    for key, value in sorted(values.items(), key=lambda item: item[1], reverse=True):
        file_obj.write(f"\t{key}: {value}\n")
    file_obj.write("\n")


def _write_centers_summary(file_obj, results: AnalysisResult) -> None:
    """Escribe resumen estadístico de coordenadas de centros de etiquetas."""
    centers_x = [center.x for center in results.labels_centers]
    centers_y = [center.y for center in results.labels_centers]

    if centers_x and centers_y:
        _write_summary_stats(file_obj, centers_x, "Centroide X de etiquetas")
        _write_summary_stats(file_obj, centers_y, "Centroide Y de etiquetas")
    else:
        file_obj.write("No se han obtenido centroides de etiquetas.\n")


def save_results(
    results: AnalysisResult,
    output_path: str | PathLike[str],
) -> None:
    """Guarda los resultados del análisis EDA en un archivo de texto."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("Resultados del Análisis EDA:\n\n")

        file_obj.write("=== Resumen de carga ===\n")
        if results.images:
            file_obj.write(f"Número total de imágenes: {len(results.images)}\n")
        else:
            file_obj.write("No se han cargado imágenes correctamente.\n")

        total_labels = sum(len(label_file.masks) for label_file in results.labels)
        if total_labels > 0:
            file_obj.write(f"Número total de etiquetas: {total_labels}\n")
        else:
            file_obj.write("No se han cargado etiquetas correctamente.\n")

        file_obj.write("\n=== Archivos incorrectos ===\n")

        if results.incorrect_images:
            file_obj.write(
                "Número de imágenes con formato incorrecto: "
                f"{len(results.incorrect_images)}\n"
            )
            file_obj.write("Imágenes con formato incorrecto:\n")
            for image_path in results.incorrect_images:
                file_obj.write(f"\t{image_path}\n")
        else:
            file_obj.write("No se han encontrado imágenes con formato incorrecto.\n")

        if results.incorrect_labels:
            file_obj.write(
                "Número de etiquetas con formato incorrecto: "
                f"{len(results.incorrect_labels)}\n"
            )
            file_obj.write("Etiquetas con formato incorrecto:\n")
            for label_path in results.incorrect_labels:
                file_obj.write(f"\t{label_path}\n")
        else:
            file_obj.write("No se han encontrado etiquetas con formato incorrecto.\n")

        file_obj.write("\n=== Métricas de imágenes ===\n")

        if results.image_types:
            _write_dict_counts(file_obj, "Tipos de archivos de imágenes", results.image_types)
        else:
            file_obj.write(
                "No se han cargado imágenes para analizar los tipos de archivo.\n"
            )

        if results.image_sizes:
            _write_dict_counts(file_obj, "Tamaños de imágenes", results.image_sizes)
        else:
            file_obj.write("No se han obtenido tamaños de imágenes.\n")

        if results.image_aspect_ratios:
            _write_dict_counts(
                file_obj,
                "Relaciones de aspecto de imágenes",
                results.image_aspect_ratios,
            )
        else:
            file_obj.write("No se han obtenido relaciones de aspecto de imágenes.\n")

        if results.images_brightness:
            _write_summary_stats(
                file_obj,
                results.images_brightness,
                "Brillo de imágenes",
            )
        else:
            file_obj.write("No se han obtenido datos sobre el brillo de las imágenes.\n")

        if results.images_contrast:
            _write_summary_stats(
                file_obj,
                results.images_contrast,
                "Contraste de imágenes",
            )
        else:
            file_obj.write(
                "No se han obtenido datos sobre el contraste de las imágenes.\n"
            )

        if results.images_blur:
            _write_summary_stats(
                file_obj,
                results.images_blur,
                "Desenfoque (varianza de Laplacian) de imágenes",
            )
        else:
            file_obj.write(
                "No se han obtenido datos sobre el desenfoque de las imágenes.\n"
            )

        file_obj.write("\n=== Métricas de etiquetas ===\n")

        if results.num_labels_per_image:
            _write_summary_stats(
                file_obj,
                results.num_labels_per_image,
                "Número de etiquetas por imagen",
            )
        else:
            file_obj.write(
                "No se han obtenido datos sobre el número de etiquetas por imagen.\n"
            )

        if results.label_aspect_ratios:
            _write_dict_counts(
                file_obj,
                "Relaciones de aspecto de etiquetas",
                results.label_aspect_ratios,
            )
        else:
            file_obj.write("No se han obtenido relaciones de aspecto de etiquetas.\n")

        if results.label_areas:
            _write_summary_stats(
                file_obj,
                results.label_areas,
                "Áreas de etiquetas",
            )
        else:
            file_obj.write("No se han obtenido áreas de etiquetas.\n")

        if results.labels_areas:
            _write_summary_stats(
                file_obj,
                results.label_areas,
                "Áreas de etiquetas",
            )
        else:
            file_obj.write("No se han obtenido áreas de etiquetas.\n")

        _write_centers_summary(file_obj, results)

        if any(results.label_quadrants_x.values()):
            file_obj.write(f"Cuadrantes X de etiquetas: {results.label_quadrants_x}\n")
        else:
            file_obj.write(
                "No se han obtenido datos sobre los cuadrantes X de etiquetas.\n"
            )

        if any(results.label_quadrants_y.values()):
            file_obj.write(f"Cuadrantes Y de etiquetas: {results.label_quadrants_y}\n\n")
        else:
            file_obj.write(
                "No se han obtenido datos sobre los cuadrantes Y de etiquetas.\n"
            )

        if results.labels_iou:
            _write_summary_stats(
                file_obj,
                results.labels_iou,
                "Valores de IoU entre etiquetas",
            )
        else:
            file_obj.write("No se han obtenido valores de IoU entre etiquetas.\n")
