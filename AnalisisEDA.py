import os

import cv2

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)
import pathlib
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np


class EDA:
    """Clase que usaremos para analizar el conjunto de datos"""

    def __init__(self):

        #Definicion de las variables que usaremos para almacenar la informacion de las imagenes y etiquetas
        self.images = []
        self.labels = []
        self.incorrect_images = []
        self.incorrect_labels = []
        
        self.imageTypes = {}
        self.imageSizes = {}
        self.imagesAspectRatios = {}
        self.labelSizes = []
        self.labelsCenters = []
        self.labelAscpectRatios = {}
        self.labelPositionsX = []
        self.labelPositionsY = []
        self.labelCuadrantesX = {"Izquierda": 0, "Centro": 0, "Derecha": 0}
        self.labelCuadrantesY = {"Arriba": 0, "Centro": 0, "Abajo": 0}
        self.numLabelsPerImage = []
        self.labelsUIO = []
        self.imagesBrightness = []
        self.imagesContrast = []
        self.imagesBlur = []

    def loadImages(self, path):
        """Carga las imágenes y etiquetas desde el directorio especificado.

        Args:
            path (str): Ruta al directorio que contiene las imágenes organizadas en subdirectorios por clase.
        """

        # Leemos todos los archivos en el directorio
        for files in os.listdir(path):
            # Obtenemos la ruta completa del archivo
            file_path = os.path.join(path, files)

            # Verificamos si es un archivo de imagen (puedes ajustar esto según tus necesidades)
            if os.path.isfile(file_path) and file_path.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                # Leemos la imagen usando OpenCV
                image = cv2.imread(file_path)

                # Agregamos la imagen a la lista de imágenes
                self.images.append((file_path, image))
            else:
                self.incorrect_images.append(file_path)

        # Ordenamos las imágenes por nombre de archivo para facilitar la comparación con las etiquetas
        self.images.sort(key=lambda x: os.path.basename(x[0]))

    def loadLabels(self, path):
        """Carga las etiquetas desde el directorio especificado.

        Args:
            path (str): Ruta al directorio que contiene las imágenes organizadas en subdirectorios por clase.
        """

        # Leemos todos los archivos en el directorio
        for files in os.listdir(path):
            # Obtenemos la ruta completa del archivo
            file_path = os.path.join(path, files)

            # Verificamos si es un archivo de texto (puedes ajustar esto según tus necesidades)
            if os.path.isfile(file_path) and file_path.lower().endswith(".txt"):
                # Leemos el contenido del archivo de texto
                with open(file_path, "r") as f:
                    labels = f.readlines()
                # Agregamos la etiqueta a la lista de etiquetas
                labelsComprobadas = []
                # Verificamos que el formato de las etiquetas sea correcto (puedes ajustar esto según tus necesidades)
                for label in labels:
                    text = label.strip().split()
                    # Verificar que cumpla con el formato esperado: clase (entero), x_center (float), y_center (float), width (float), height (float)
                    if (
                        len(text) < 5
                        and float(text[0]) < 0
                        and not all(0 <= float(coord) <= 1 for coord in text[1:5])
                    ):
                        self.incorrect_labels.append((file_path, label))
                    else:
                        labelsComprobadas.append(label)
                self.labels.append((file_path, labelsComprobadas))
            else:
                self.incorrect_labels.append(file_path)

        # Ordenamos las etiquetas por nombre de archivo para facilitar la comparación con las imágenes
        self.labels.sort(key=lambda x: os.path.basename(x[0]))

    def analyzeTypeFiles(self):
        """Realizacion de un analisis del tipo de archivo de las imagenes"""

        # Analizamos el tipo de archivo de cada imagen
        for file_path, _ in self.images:
            # Obtenemos la extensión del archivo
            ext = os.path.splitext(file_path)[1].lower()

            # Contamos el número de imágenes por tipo de archivo
            if ext in self.imageTypes:
                self.imageTypes[ext] += 1
            else:
                self.imageTypes[ext] = 1

    def analyzeImageSizes(self):
        """Realizacion de un analisis del tamaño de las imagenes"""

        # Analizamos el tamaño de cada imagen
        for _, image in self.images:
            # Obtenemos el tamaño de la imagen (ancho x alto)
            size = (image.shape[1], image.shape[0])  # (ancho, alto)

            # Contamos el número de imágenes por tamaño
            if size in self.imageSizes:
                self.imageSizes[size] += 1
            else:
                self.imageSizes[size] = 1

    def analyzeAspectRatio(self):
        """Realizacion de un analisis de la relacion de aspecto de las imagenes"""

        # Analizamos la relacion de aspecto de cada imagen
        for _, image in self.images:
            # Obtenemos el tamaño de la imagen (ancho x alto)
            width, height = image.shape[1], image.shape[0]

            # Calculamos la relacion de aspecto como fracción simplificada (ej. 16/9)
            frac = Fraction(width, height)
            aspect_ratio = f"{frac.numerator}/{frac.denominator}"

            # Contamos el número de imágenes por relación de aspecto
            if aspect_ratio in self.imagesAspectRatios:
                self.imagesAspectRatios[aspect_ratio] += 1
            else:
                self.imagesAspectRatios[aspect_ratio] = 1

    def analyzeLabelsSize(self):
        """Realizacion de un analisis del tamaño relativo de las etiquetas respecto a las imagenes"""

        # Analizamos el tamaño relativo de cada etiqueta respecto a su imagen correspondiente
        for _, labels in self.labels:
            # Analizamos cada etiqueta en el archivo de etiquetas
            for label in labels:
                text = label.strip().split()
                if (
                    len(text) < 5
                ):  # Verificar que la etiqueta tenga al menos 6 elementos (clase + 4 coordenadas)
                    continue  # Saltar etiquetas mal formateadas

                # Obtenemos el tamaño de la imagen correspondiente a la etiqueta
                width = float(text[3]) * 100
                height = float(text[4]) * 100

                # Agregamos el tamaño relativo de la etiqueta respecto a la imagen a las listas correspondientes
                self.labelSizes.append((width, height))

    def analyzePositionLabels(self):
        """Realizacion de un analisis de la distribucion de las etiquetas especialmente (Verticalmente y horizontalmente)"""

        for _, labels in self.labels:
            for label in labels:
                text = label.strip().split()
                if (
                    len(text) < 5
                ):  # Verificar que la etiqueta tenga al menos 6 elementos (clase + 4 coordenadas)
                    continue  # Saltar etiquetas mal formateadas

                # Obtenemos el tamaño de la imagen correspondiente a la etiqueta
                x_center = float(text[1])
                y_center = float(text[2])

                # Agregamos la posición de la etiqueta a las listas correspondientes
                self.labelsCenters.append((x_center, y_center))
                self.labelPositionsX.append(x_center)
                self.labelPositionsY.append(y_center)

                # Clasificamos la posición horizontal de la etiqueta en cuadrantes
                if x_center < 0.33:
                    self.labelCuadrantesX["Izquierda"] += 1
                elif x_center < 0.66:
                    self.labelCuadrantesX["Centro"] += 1
                else:
                    self.labelCuadrantesX["Derecha"] += 1

                # Clasificamos la posición vertical de la etiqueta en cuadrantes
                if y_center < 0.33:
                    self.labelCuadrantesY["Arriba"] += 1
                elif y_center < 0.66:
                    self.labelCuadrantesY["Centro"] += 1
                else:
                    self.labelCuadrantesY["Abajo"] += 1

    def analyzeNumLabelsPerImage(self):
        """Realizacion de un analisis del numero de etiquetas por imagen"""

        # Analizamos el número de etiquetas por imagen
        for _, labels in self.labels:
            num_labels = len(labels)
            self.numLabelsPerImage.append(num_labels)

    def generateDensityCenterPlot(self, pathOutput=None):
        """Generacion de un grafico de densidad para la posición de las etiquetas"""

        # Obtenemos las posiciones X e Y de las etiquetas
        if not self.labelsCenters:
            self.analyzePositionLabels()  # Asegurarnos de que se hayan analizado las posiciones de las etiquetas

        # Convertimos las posiciones de las etiquetas a un array de NumPy para facilitar el manejo
        centers = np.array(self.labelsCenters)
        x = centers[:, 0]
        y = centers[:, 1]

        # Creamos el grafico de densidad
        plt.figure(figsize=(8, 6))
        plt.hexbin(x, y, gridsize=30, cmap="Blues")
        plt.colorbar(label="Número de etiquetas")
        plt.xlabel("Posición horizontal (x_center)")
        plt.ylabel("Posición vertical (y_center)")
        plt.title("Densidad de posiciones de etiquetas")

        # Guardamos el grafico si se especifica un path de salida
        if pathOutput:
            plt.savefig(pathOutput)

    def analyzeLabelsAspectRatio(self):
        """Realizacion de un analisis de la relacion de aspecto de las etiquetas"""

        #Validamos si ya se han analizado los tamaños de las etiquetas, si no es así, los analizamos
        if not self.labelSizes:
            self.analyzeLabelsSize()

        # Analizamos la relacion de aspecto de cada etiqueta
        for width, height in self.labelSizes:
            # Calculamos la relacion de aspecto como fracción simplificada (ej. 16/9)
            frac = Fraction(int(width), int(height))
            aspect_ratio = f"{frac.numerator}/{frac.denominator}"

            # Contamos el número de etiquetas por relación de aspecto
            if aspect_ratio in self.labelAscpectRatios:
                self.labelAscpectRatios[aspect_ratio] += 1
            else:
                self.labelAscpectRatios[aspect_ratio] = 1

    def calculateIoU(self, bbox1, bbox2):
        """Función para calcular el Intersection over Union (IoU) entre dos bounding boxes"""
        x_min1, y_min1, x_max1, y_max1 , width1, height1 = bbox1
        x_min2, y_min2, x_max2, y_max2 , width2, height2 = bbox2

        # Calculamos las coordenadas del área de intersección
        x_min_inter = max(x_min1, x_min2)
        y_min_inter = max(y_min1, y_min2)
        x_max_inter = min(x_max1, x_max2)
        y_max_inter = min(y_max1, y_max2)

        # Calculamos el área de intersección
        inter_width = max(0, x_max_inter - x_min_inter)
        inter_height = max(0, y_max_inter - y_min_inter)
        area_inter = inter_width * inter_height

        # Calculamos el área de cada bounding box
        area_bbox1 = width1 * height1
        area_bbox2 = width2 * height2

        # Calculamos el área de unión
        area_union = area_bbox1 + area_bbox2 - area_inter

        # Calculamos el IoU
        iou = area_inter / area_union if area_union > 0 else 0

        return iou

    def analyzeLabelsSolapamiento(self):
        """Realizacion de un analisis del solapamiento entre etiquetas (IoU)"""

        if not self.labels:
            return # No hay etiquetas para analizar

        #Obtemos las etiquetas para  cada imagen
        for _, labels in self.labels:
            #Analizamos el solapamiento cuando hay más de una etiqueta en la imagen
            if len(labels) > 1:
                #Obtenemos las coordenadas de cada etiqueta
                bboxes = []
                for label in labels:
                    text = label.strip().split()

                    x_center = float(text[1])
                    y_center = float(text[2])
                    width = float(text[3])
                    height = float(text[4])
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    bboxes.append((x_min, y_min, x_max, y_max, width, height))

                #Calculamos el IoU entre cada par de etiquetas
                for i in range(len(bboxes)):
                    for j in range(i + 1, len(bboxes)):
                        bbox1 = bboxes[i]
                        bbox2 = bboxes[j]
                        iou = self.calculateIoU(bbox1, bbox2)
                        #Aquí podrías almacenar o analizar el IoU según tus necesidades
                        self.labelsUIO.append(iou)

    def analyzeBrightness(self):
        """Realizacion de un analisis del brillo de las imagenes"""
        if not self.images:
            return # No hay imágenes para analizar

        # Analizamos el brillo de cada imagen
        for _, image in self.images:
            # Convertimos la imagen a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculamos el brillo promedio de la imagen
            brightness = np.mean(gray)
            self.imagesBrightness.append(brightness)

    def analyzeContrast(self):
        """Realizacion de un analisis del contraste de las imagenes
            Notas de rango
            <20 = Bajo contraste
            20-50 = Contraste medio
            >50 = Alto contraste
        """
        if not self.images:
            return # No hay imágenes para analizar

        # Analizamos el contraste de cada imagen
        for _, image in self.images:
            # Convertimos la imagen a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculamos el contraste usando la varianza de Laplacian
            contrast = float(np.std(gray))
            self.imagesContrast.append(contrast)

    def generateContinuosPlots(self, dato, output_dir, filename):
        """Genera y guarda un gráfico de histograma con curva normal superpuesta."""
        # Creamos el directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Conversion del dato a array de numpy para facilitar el manejo
        data = np.array(dato)

        # Histograma + curva normal para evaluar similitud con distribución normal
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(data, bins=20, density=True, alpha=0.7, color="#4c72b0", edgecolor="black", label="Datos")

        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            x_vals = np.linspace(np.min(data), np.max(data), 200)
            normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_vals - mean) / std) ** 2
            )
            ax.plot(x_vals, normal_pdf, "r-", linewidth=2, label="Normal teórica")

        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad")
        ax.set_title("Distribución vs Curva Normal")
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.close(fig)

    def analyzeDesenfoque(self):
        """Realizacion de un analisis del desenfoque de las imagenes usando la varianza de Laplacian"""
        if not self.images:
            return # No hay imágenes para analizar

        # Analizamos el desenfoque de cada imagen
        for _, image in self.images:
            # Convertimos la imagen a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculamos la varianza de Laplacian para evaluar el desenfoque
            blur_metric = cv2.Laplacian(gray, cv2.CV_64F).var()
            self.imagesBlur.append(blur_metric)

    def saveResults(self, output_dir):
        """Guarda los resultados del análisis en un archivo de texto."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "eda_results.txt"), "w") as f:
            f.write("Resultados del Análisis EDA:\n\n")

            #Imagenes cargadas
            if len(self.images) > 0:
                f.write(f"Número total de imágenes: {len(self.images)}\n")
            else:
                f.write("No se han cargado imágenes correctamente.\n")

            #Etiquetas cargadas
            if len(self.labels) > 0:
                f.write(f"Número total de etiquetas: {len(self.labels)}\n")
            else:
                f.write("No se han cargado etiquetas correctamente.\n")

            #Images incorrectas
            if len(self.incorrect_images) > 0:
                f.write(f"Número de imágenes con formato incorrecto: {len(self.incorrect_images)}\n")
                f.write("Imágenes con formato incorrecto:\n")
                for img in self.incorrect_images:
                    f.write(f"\t{img}\n")
            else:
                f.write("No se han encontrado imágenes con formato incorrecto.\n")

            #Etiquetas incorrectas
            if len(self.incorrect_labels) > 0:
                f.write(f"Número de etiquetas con formato incorrecto: {len(self.incorrect_labels)}\n")
                f.write("Etiquetas con formato incorrecto:\n")
                for label in self.incorrect_labels:
                    f.write(f"\t{label}\n")
            else:
                f.write("No se han encontrado etiquetas con formato incorrecto.\n")

            #Estadísticas de las imágenes
            if len(self.imageTypes) > 0:
                f.write(f"Tipos de archivos de imágenes: {self.imageTypes}\n")
            else:
                f.write("No se han cargado imágenes para analizar los tipos de archivo.\n")

            #Tamaños de las imagenes
            if len(self.imageSizes) > 0:
                f.write(f"Tamaños de imágenes: {self.imageSizes}\n")
            else:
                f.write("No se han obtenido tamaños de imágenes.\n")

            #Relaciones de aspecto de las imagenes
            if len(self.imagesAspectRatios) > 0:
                f.write(f"Relaciones de aspecto de imágenes: {self.imagesAspectRatios}\n")
            else:
                f.write("No se han obtenido relaciones de aspecto de imágenes.\n")

            #Número de etiquetas por imagen
            if len(self.numLabelsPerImage) > 0:
                f.write(f"Número de etiquetas por imagen:\n\t Media: {np.mean(self.numLabelsPerImage):.2f}\n\t Mediana: {np.median(self.numLabelsPerImage)}\n\t Mínimo: {np.min(self.numLabelsPerImage)}\n\t Máximo: {np.max(self.numLabelsPerImage)}\n")
            else:
                f.write("No se han obtenido datos sobre el número de etiquetas por imagen.\n")

            #Relaciones de aspecto de las etiquetas
            if len(self.labelAscpectRatios) > 0:
                f.write(f"Relaciones de aspecto de etiquetas: {self.labelAscpectRatios}\n")
            else:
                f.write("No se han obtenido relaciones de aspecto de etiquetas.\n")

            #Cuadrantes X de las etiquetas
            if len(self.labelCuadrantesX) > 0:
                f.write(f"Cuadrantes X de etiquetas: {self.labelCuadrantesX}\n")
            else:
                f.write("No se han obtenido datos sobre los cuadrantes X de etiquetas.\n")

            #Cuadrantes Y de las etiquetas
            if len(self.labelCuadrantesY) > 0:
                f.write(f"Cuadrantes Y de etiquetas: {self.labelCuadrantesY}\n")
            else:
                f.write("No se han obtenido datos sobre los cuadrantes Y de etiquetas.\n")

            #Valores de IoU entre etiquetas
            if len(self.labelsUIO) > 0:
                f.write(f"Valores de IoU entre etiquetas:\n\t Media: {np.mean(self.labelsUIO):.2f}\n\t Mediana: {np.median(self.labelsUIO):.2f}\n\t Mínimo: {np.min(self.labelsUIO):.2f}\n\t Máximo: {np.max(self.labelsUIO):.2f}\n")
            else:
                f.write("No se han obtenido valores de IoU entre etiquetas.\n")

            #Brillo de las imagenes
            if len(self.imagesBrightness) > 0:
                f.write(f"Brillo de imágenes: \n\t Media: {np.mean(self.imagesBrightness):.2f}\n\t Mediana: {np.median(self.imagesBrightness):.2f}\n\t Mínimo: {np.min(self.imagesBrightness):.2f}\n\t Máximo: {np.max(self.imagesBrightness):.2f}\n")
            else:
                f.write("No se han obtenido datos sobre el brillo de las imágenes.\n")

            #Contraste de las imagenes
            if len(self.imagesContrast) > 0:
                f.write(f"Contraste de imágenes: \n\t Media: {np.mean(self.imagesContrast):.2f}\n\t Mediana: {np.median(self.imagesContrast):.2f}\n\t Mínimo: {np.min(self.imagesContrast):.2f}\n\t Máximo: {np.max(self.imagesContrast):.2f}\n")
            else:
                f.write("No se han obtenido datos sobre el contraste de las imágenes.\n")

            #Desenfoque de las imagenes
            if len(self.imagesBlur) > 0:
                f.write(f"Desenfoque (varianza de Laplacian) de imágenes: \n\t Media: {np.mean(self.imagesBlur):.2f}\n\t Mediana: {np.median(self.imagesBlur):.2f}\n\t Mínimo: {np.min(self.imagesBlur):.2f}\n\t Máximo: {np.max(self.imagesBlur):.2f}\n")
            else:
                f.write("No se han obtenido datos sobre el desenfoque de las imágenes.\n")

        #Cerramos el archivo de resultados
        f.close()


# Definicion de path
BASE_DIR = pathlib.Path(__file__).parent
pathImages = os.path.join(BASE_DIR, "dataPruebas/images")
pathLabels = os.path.join(BASE_DIR, "dataPruebas/labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

print("Path de las imágenes:", pathImages)
print("Path de las etiquetas:", pathLabels)

# Creación de la instancia de la clase EDA
eda = EDA()

# Carga de las imágenes y etiquetas
eda.loadImages(pathImages)
eda.loadLabels(pathLabels)

print("Número de imágenes cargadas:", len(eda.images))
print("Número de etiquetas cargadas:", len(eda.labels))