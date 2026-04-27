
import os
from common.logger import configure_logging
from eda.io.loaders import load_images, load_labels
from eda.io.writers import save_results
from eda.visualization.density_plot import plot_density_center
from eda.visualization.graphic_plot import (
    plot_boxplot,
    plot_continuous_distribution,
    plot_count_bars,
)

from eda.core.types import (
    AnalysisResult
)

from eda.metrics.images import (
    count_image_types,
    count_image_sizes,
    count_image_aspect_ratios,
    compute_images_brightness,
    compute_images_contrast,
    compute_images_blur,
)

from eda.metrics.labels import (
    compute_label_areas,
    compute_labels_areas,
    count_label_aspect_ratios,
    compute_label_centers,
    count_label_quadrants_x,
    count_label_quadrants_y,
    count_labels_per_image,
    compute_labels_iou,
)


from pathlib import Path

def _generate_plots(results: AnalysisResult, dir : Path) -> None:
    """Genera gráficos EDA a partir de las métricas calculadas."""

    if not dir.exists():
        os.makedirs(dir)

    plot_count_bars(
        counts={str(k): v for k, v in results.image_types.items()},
        output_path=dir / "image_types_bar.png",
        title="Tipos de archivo de imágenes",
        x_label="Frecuencia",
        y_label="Tipo",
    )
    plot_count_bars(
        counts={f"{w}x{h}": c for (w, h), c in results.image_sizes.items()},
        output_path=dir / "image_sizes_bar.png",
        title="Top tamaños de imágenes",
        x_label="Frecuencia",
        y_label="Resolución",
        top_n=15,
    )
    plot_count_bars(
        counts={str(k): v for k, v in results.image_aspect_ratios.items()},
        output_path=dir / "image_aspect_ratios_bar.png",
        title="Relaciones de aspecto de imágenes",
        x_label="Frecuencia",
        y_label="Relación",
    )

    plot_continuous_distribution(
        values=results.images_brightness,
        output_path=dir / "images_brightness_hist.png",
        title="Distribución de brillo",
        x_label="Brillo medio",
    )
    plot_boxplot(
        values=results.images_brightness,
        output_path=dir / "images_brightness_box.png",
        title="Boxplot de brillo",
        y_label="Brillo medio",
    )
    plot_continuous_distribution(
        values=results.images_contrast,
        output_path=dir / "images_contrast_hist.png",
        title="Distribución de contraste",
        x_label="Contraste",
    )
    plot_boxplot(
        values=results.images_contrast,
        output_path=dir / "images_contrast_box.png",
        title="Boxplot de contraste",
        y_label="Contraste",
    )
    plot_continuous_distribution(
        values=results.images_blur,
        output_path=dir / "images_blur_hist.png",
        title="Distribución de desenfoque",
        x_label="Varianza de Laplaciano",
        use_log_x=False,
    )
    plot_boxplot(
        values=results.images_blur,
        output_path=dir / "images_blur_box.png",
        title="Boxplot de desenfoque",
        y_label="Varianza de Laplaciano",
    )

    plot_continuous_distribution(
        values=results.num_labels_per_image,
        output_path=dir / "labels_per_image_hist.png",
        title="Etiquetas por imagen",
        x_label="Cantidad de etiquetas",
    )
    plot_boxplot(
        values=results.num_labels_per_image,
        output_path=dir / "labels_per_image_box.png",
        title="Boxplot de etiquetas por imagen",
        y_label="Cantidad de etiquetas",
    )

    plot_count_bars(
        counts={str(k): v for k, v in results.label_aspect_ratios.items()},
        output_path=dir / "label_aspect_ratios_bar.png",
        title="Relaciones de aspecto de etiquetas",
        x_label="Frecuencia",
        y_label="Relación",
    )

    plot_continuous_distribution(
        values=results.label_areas,
        output_path=dir / "label_areas_hist.png",
        title="Distribución de áreas de etiquetas",
        x_label="Área de etiqueta",
        use_log_x=False,
    )
    plot_boxplot(
        values=results.label_areas,
        output_path=dir / "label_areas_box.png",
        title="Boxplot de áreas de etiquetas",
        y_label="Área de etiqueta",
    )

    plot_continuous_distribution(
        values=results.labels_areas,
        output_path=dir / "labels_areas_hist.png",
        title="Distribución de áreas de etiquetas",
        x_label="Área de etiqueta",
        use_log_x=False,
    )
    plot_boxplot(
        values=results.labels_areas,
        output_path=dir / "labels_areas_box.png",
        title="Boxplot de áreas de etiquetas",
        y_label="Área de etiqueta",
    )

    plot_count_bars(
        counts={str(k): v for k, v in results.label_quadrants_x.items()},
        output_path=dir / "label_quadrants_x_bar.png",
        title="Centroides por cuadrante X",
        x_label="Frecuencia",
        y_label="Cuadrante",
    )
    plot_count_bars(
        counts={str(k): v for k, v in results.label_quadrants_y.items()},
        output_path=dir / "label_quadrants_y_bar.png",
        title="Centroides por cuadrante Y",
        x_label="Frecuencia",
        y_label="Cuadrante",
    )

    plot_continuous_distribution(
        values=results.labels_iou,
        output_path=dir / "labels_iou_hist.png",
        title="Distribución de IoU entre etiquetas",
        x_label="IoU",
    )
    plot_boxplot(
        values=results.labels_iou,
        output_path=dir / "labels_iou_box.png",
        title="Boxplot de IoU",
        y_label="IoU",
    )

    plot_density_center(
        labels_centers=results.labels_centers,
        output_path=dir / "labels_centers_density.png",
    )

def main() -> None:
    """Punto de entrada principal del programa."""
    # Definicion de rutas de datos y resultados
    DATASET_PATH = Path("./data/train")
    RESULTS_DIR = Path("./output/eda")
    RESULTS_FILE = RESULTS_DIR / "eda_results.txt"
    PLOTS_DIR = RESULTS_DIR / "plots"

    # Comprobacion de existencia de directorios
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"El directorio de datos no existe: {DATASET_PATH}")
    if not RESULTS_DIR.exists():
        os.makedirs(RESULTS_DIR)

    # Configura el sistema de logging
    configure_logging()

    # Carga de datos
    images_loader = load_images(DATASET_PATH / "images")
    labels_loader = load_labels(DATASET_PATH / "labels")

    images = images_loader.images
    incorrect_images = images_loader.incorrect_images

    labels = labels_loader.labels
    incorrect_labels = labels_loader.incorrect_labels
    
    print(f"Cargadas {len(images)} imágenes y {len(labels)} archivos de etiquetas.")

    # Calculo de métricas para imágenes
    image_types = count_image_types(images)
    print("Calculado el conteo de tipos de imágenes.")
    image_sizes = count_image_sizes(images)
    print("Calculado el conteo de tamaños de imágenes.")
    image_aspect_ratios = count_image_aspect_ratios(images)
    print("Calculado el conteo de relaciones de aspecto de imágenes.")
    images_brightness = compute_images_brightness(images)
    print("Calculado el brillo medio de las imágenes.")
    images_contrast = compute_images_contrast(images)
    print("Calculado el contraste de las imágenes.")
    images_blur = compute_images_blur(images)
    print("Calculado el desenfoque de las imágenes.")
    
    # Calculo de métricas para etiquetas
    num_labels_per_image = count_labels_per_image(labels)
    print("Calculado el conteo de etiquetas por imagen.")
    label_areas = compute_label_areas(labels)
    print("Calculado el área de las etiquetas individuales.")
    labels_areas = compute_labels_areas(labels)
    print("Calculado el área total de las etiquetas por imagen.")
    label_aspect_ratios = count_label_aspect_ratios(labels)
    print("Calculado el conteo de relaciones de aspecto de las etiquetas.")
    labels_centers = compute_label_centers(labels)
    print("Calculado el centroide de las etiquetas.")
    label_quadrants_x = count_label_quadrants_x(labels_centers)
    print("Calculado el conteo de etiquetas por cuadrante X.")
    label_quadrants_y = count_label_quadrants_y(labels_centers)
    print("Calculado el conteo de etiquetas por cuadrante Y.")
    labels_iou = compute_labels_iou(labels)
    print("Calculado el IoU entre etiquetas.")

    # Guardar resultados

    results = AnalysisResult(
        images=images,
        labels=labels,
        incorrect_images = incorrect_images,
        incorrect_labels = incorrect_labels,
        image_types=image_types,
        image_sizes=image_sizes,
        image_aspect_ratios=image_aspect_ratios,
        images_brightness=images_brightness,
        images_contrast=images_contrast,
        images_blur=images_blur,
        num_labels_per_image=num_labels_per_image,
        label_areas=label_areas,
        labels_areas=labels_areas,
        label_aspect_ratios=label_aspect_ratios,
        labels_centers=labels_centers,
        label_quadrants_x=label_quadrants_x,
        label_quadrants_y=label_quadrants_y,
        labels_iou=labels_iou,
    )  
    save_results(results, RESULTS_FILE)
    _generate_plots(results=results, dir=PLOTS_DIR)
    

if __name__ == "__main__":
    main()
