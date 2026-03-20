import os
import numpy as np
import cv2
import pathlib
from fractions import Fraction

class EDA:
    """Clase que usaremos para analizar el conjunto de datos 
    """
    
    def __init__(self):
        self.images = []
        self.labels = []
        
    def loadImages(self, path):
        """Carga las imágenes y etiquetas desde el directorio especificado.
        
        Args:
            path (str): Ruta al directorio que contiene las imágenes organizadas en subdirectorios por clase.
        """
        
        #Leemos todos los archivos en el directorio
        for files in os.listdir(path):
            #Obtenemos la ruta completa del archivo
            file_path = os.path.join(path, files)
            
            #Verificamos si es un archivo de imagen (puedes ajustar esto según tus necesidades)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                #Leemos la imagen usando OpenCV
                image = cv2.imread(file_path)
                
                #Agregamos la imagen a la lista de imágenes
                self.images.append((file_path, image))
        
    def loadLabels(self, path):
        """Carga las etiquetas desde el directorio especificado.
        
        Args:
            path (str): Ruta al directorio que contiene las imágenes organizadas en subdirectorios por clase.
        """
        #Leemos todos los archivos en el directorio
        for files in os.listdir(path):
            #Obtenemos la ruta completa del archivo
            file_path = os.path.join(path, files)
            
            #Verificamos si es un archivo de texto (puedes ajustar esto según tus necesidades)
            if os.path.isfile(file_path) and file_path.lower().endswith('.txt'):
                #Leemos el contenido del archivo de texto
                with open(file_path, 'r') as f:
                    labels = f.readlines()
                #Agregamos la etiqueta a la lista de etiquetas
                self.labels.append((file_path, labels))
    
    def analyzeTypeFiles(self):
        """Realizacion de un analisis del tipo de archivo de las imagenes
        """
        
        self.image_types = {}
        
        #Analizamos el tipo de archivo de cada imagen
        for file_path, _ in self.images:
            #Obtenemos la extensión del archivo
            ext = os.path.splitext(file_path)[1].lower()
            
            #Contamos el número de imágenes por tipo de archivo
            if ext in self.image_types:
                self.image_types[ext] += 1
            else:
                self.image_types[ext] = 1
    
    def analyzeImageSizes(self):
        """Realizacion de un analisis del tamaño de las imagenes
        """
        
        self.image_sizes = {}
        
        #Analizamos el tamaño de cada imagen
        for _, image in self.images:
            #Obtenemos el tamaño de la imagen (ancho x alto)
            size = (image.shape[1], image.shape[0])  # (ancho, alto)
            
            #Contamos el número de imágenes por tamaño
            if size in self.image_sizes:
                self.image_sizes[size] += 1
            else:
                self.image_sizes[size] = 1
    
    def analyzeAspectRatio(self):
        """Realizacion de un analisis de la relacion de aspecto de las imagenes
        """
        
        self.aspect_ratios = {}
        
        #Analizamos la relacion de aspecto de cada imagen
        for _, image in self.images:
            #Obtenemos el tamaño de la imagen (ancho x alto)
            width, height = image.shape[1], image.shape[0]
            
            # Calculamos la relacion de aspecto como fracción simplificada (ej. 16/9)
            frac = Fraction(width, height)
            aspect_ratio = f"{frac.numerator}/{frac.denominator}"
            
            #Contamos el número de imágenes por relación de aspecto
            if aspect_ratio in self.aspect_ratios:
                self.aspect_ratios[aspect_ratio] += 1
            else:
                self.aspect_ratios[aspect_ratio] = 1
    
    def analyzeLabelsSize(self):
        """Realizacion de un analisis del tamaño relativo de las etiquetas respecto a las imagenes
        """
        
        self.labelSizesWidth = []
        self.labelSizesHeight = []
        
        #Analizamos el tamaño relativo de cada etiqueta respecto a su imagen correspondiente
        for label_path, labels in self.labels:
            #Analizamos cada etiqueta en el archivo de etiquetas
            for label in labels:
                text = label.strip().split()
                if len(text) < 5:  # Verificar que la etiqueta tenga al menos 6 elementos (clase + 4 coordenadas)
                    continue  # Saltar etiquetas mal formateadas
                
                #Obtenemos el tamaño de la imagen correspondiente a la etiqueta
                width = float(text[3])*100 
                height = float(text[4])*100
                
                #Agregamos el tamaño relativo de la etiqueta respecto a la imagen a las listas correspondientes
                self.labelSizesWidth.append(width)
                self.labelSizesHeight.append(height)
    
    def analyzePositionLabels(self):
        """Realizacion de un analisis de la distribucion de las etiquetas especialmente (Verticalmente y horizontalmente)
        """
        
        self.labelPositionsX = []
        self.labelPositionsY = []
        self.labelCuadrantesX = {
            "Izquierda": 0,
            "Centro": 0,
            "Derecha": 0
        }
        self.labelCuadrantesY = {
            "Arriba": 0,
            "Centro": 0,
            "Abajo": 0
        }
        
        for label_path, labels in self.labels:
            for label in labels:
                text = label.strip().split()
                if len(text) < 5:  # Verificar que la etiqueta tenga al menos 6 elementos (clase + 4 coordenadas)
                    continue  # Saltar etiquetas mal formateadas
                
                #Obtenemos el tamaño de la imagen correspondiente a la etiqueta
                x_center = float(text[1])
                y_center = float(text[2])
                
                #Agregamos la posición de la etiqueta a las listas correspondientes
                self.labelPositionsX.append(x_center)
                self.labelPositionsY.append(y_center)
                
                #Clasificamos la posición horizontal de la etiqueta en cuadrantes
                if x_center < 0.33:
                    self.labelCuadrantesX["Izquierda"] += 1
                elif x_center < 0.66:
                    self.labelCuadrantesX["Centro"] += 1
                else:
                    self.labelCuadrantesX["Derecha"] += 1
                
                #Clasificamos la posición vertical de la etiqueta en cuadrantes
                if y_center < 0.33:
                    self.labelCuadrantesY["Arriba"] += 1
                elif y_center < 0.66:
                    self.labelCuadrantesY["Centro"] += 1
                else:
                    self.labelCuadrantesY["Abajo"] += 1
    




#Definicion de path
BASE_DIR = pathlib.Path(__file__).parent
pathImages = os.path.join(BASE_DIR, 'dataPruebas/images')
pathLabels = os.path.join(BASE_DIR, 'dataPruebas/labels')

print("Path de las imágenes:", pathImages)
print("Path de las etiquetas:", pathLabels)

#Creación de la instancia de la clase EDA
eda = EDA()

#Carga de las imágenes y etiquetas
eda.loadImages(pathImages)
eda.loadLabels(pathLabels)

# print("Número de imágenes cargadas:", len(eda.images))
# print("Número de etiquetas cargadas:", len(eda.labels))

# #Obtenemos el tipo de archivo de las imágenes
# eda.analyzeTypeFiles()
# print("Número de imágenes por tipo de archivo:", eda.image_types)

# #Obtenemos el tamaño de las imágenes
# eda.analyzeImageSizes()
# print("Número de imágenes por tamaño:", eda.image_sizes)

# #Obtenemos la relación de aspecto de las imágenes
# eda.analyzeAspectRatio()
# print("Número de imágenes por relación de aspecto:", eda.aspect_ratios)

# #Obtenemos el tamaño relativo de las etiquetas respecto a las imágenes
# eda.analyzeLabelsSize()
# print("Ancho de las etiquetas \n\t Media: {} \n\t Mediana: {} \n\t Mínimo: {} \n\t Máximo: {}".format(np.mean(eda.labelSizesWidth), np.median(eda.labelSizesWidth), np.min(eda.labelSizesWidth), np.max(eda.labelSizesWidth)))
# print("Alto de las etiquetas \n\t Media: {} \n\t Mediana: {} \n\t Mínimo: {} \n\t Máximo: {}".format(np.mean(eda.labelSizesHeight), np.median(eda.labelSizesHeight), np.min(eda.labelSizesHeight), np.max(eda.labelSizesHeight)))

#Obtenemos la distribución de las etiquetas especialmente (Verticalmente y horizontalmente)
eda.analyzePositionLabels()
print("Posición horizontal \n\t Media: {}, \n\t Mediana: {}, \n\t Mínimo: {}, \n\t Máximo: {}".format(np.mean(eda.labelPositionsX), np.median(eda.labelPositionsX), np.min(eda.labelPositionsX), np.max(eda.labelPositionsX)))
print("Posición vertical \n\t Media: {}, \n\t Mediana: {}, \n\t Mínimo: {}, \n\t Máximo: {}".format(np.mean(eda.labelPositionsY), np.median(eda.labelPositionsY), np.min(eda.labelPositionsY), np.max(eda.labelPositionsY)))
print("Número de etiquetas por cuadrante horizontal:", eda.labelCuadrantesX)
print("Número de etiquetas por cuadrante vertical:", eda.labelCuadrantesY)

