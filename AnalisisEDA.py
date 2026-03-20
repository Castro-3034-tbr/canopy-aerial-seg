import os
import numpy as np
import cv2
import pathlib

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
                    label = f.read().strip()
                
                #Agregamos la etiqueta a la lista de etiquetas
                self.labels.append((file_path, label))
    


