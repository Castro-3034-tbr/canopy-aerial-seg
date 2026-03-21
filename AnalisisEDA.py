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

# print("Número de imágenes cargadas:", len(eda.images))
# print("Número de etiquetas cargadas:", len(eda.labels))

# #Obtenemos el tipo de archivo de las imágenes
# eda.analyzeTypeFiles()
# print("Número de imágenes por tipo de archivo:", eda.imageTypes)

# #Obtenemos el tamaño de las imágenes
# eda.analyzeImageSizes()
# print("Número de imágenes por tamaño:", eda.imageSizes)

# #Obtenemos la relación de aspecto de las imágenes
# eda.analyzeAspectRatio()
# print("Número de imágenes por relación de aspecto:", eda.imagesAspectRatios)

# #Obtenemos el tamaño relativo de las etiquetas respecto a las imágenes
# eda.analyzeLabelsSize()
# labelSizeWidths = [eda.labelSizes[i][0] for i in range(len(eda.labelSizes))]
# labelSizeHeights = [eda.labelSizes[i][1] for i in range(len(eda.labelSizes))]
# print("Ancho de las etiquetas \n\t Media: {} \n\t Mediana: {} \n\t Mínimo: {} \n\t Máximo: {}".format(np.mean(labelSizeWidths), np.median(labelSizeWidths), np.min(labelSizeWidths), np.max(labelSizeWidths)))
# print("Alto de las etiquetas \n\t Media: {} \n\t Mediana: {} \n\t Mínimo: {} \n\t Máximo: {}".format(np.mean(labelSizeHeights), np.median(labelSizeHeights), np.min(labelSizeHeights), np.max(labelSizeHeights)))

# # Obtenemos la distribución de las etiquetas especialmente (Verticalmente y horizontalmente)
# eda.analyzePositionLabels()
# print("Posición horizontal \n\t Media: {}, \n\t Mediana: {}, \n\t Mínimo: {}, \n\t Máximo: {}".format(np.mean(eda.labelPositionsX), np.median(eda.labelPositionsX), np.min(eda.labelPositionsX), np.max(eda.labelPositionsX)))
# print("Posición vertical \n\t Media: {}, \n\t Mediana: {}, \n\t Mínimo: {}, \n\t Máximo: {}".format(np.mean(eda.labelPositionsY), np.median(eda.labelPositionsY), np.min(eda.labelPositionsY), np.max(eda.labelPositionsY)))
# print("Número de etiquetas por cuadrante horizontal:", eda.labelCuadrantesX)
# print("Número de etiquetas por cuadrante vertical:", eda.labelCuadrantesY)

# eda.generateDensityCenterPlot(pathOutput=os.path.join(OUTPUT_DIR, "density_center_plot.png"))

# # Obtenemos el número de etiquetas por imagen
# eda.analyzeNumLabelsPerImage()
# print("Número de etiquetas por imagen \n\t Media: {}, \n\t Mediana: {}, \n\t Mínimo: {}, \n\t Máximo: {}".format(np.mean(eda.numLabelsPerImage), np.median(eda.numLabelsPerImage), np.min(eda.numLabelsPerImage), np.max(eda.numLabelsPerImage)))

# # Obtenemos las imágenes que no cuentan con el formato correcto
# print("Número de imágenes incorrectas:", len(eda.incorrect_images))
# for image_path in eda.incorrect_images:
#     print(f"Imagen incorrecta: {image_path}")

# #Obtenemos las etiquetas que no cuentan con el formato correcto
# print("Número de etiquetas incorrectas:", len(eda.incorrect_labels))
# for label_path in eda.incorrect_labels:
#     print(f"Etiqueta incorrecta en {label_path}")

#Obtenemos el aspect ratio de las etiquetas
eda.analyzeLabelsAspectRatio()
print("Número de etiquetas por relación de aspecto:", eda.labelAscpectRatios)
