
from ultralytics import YOLO
import os


class YOLOTrainer:
    """Clase para gestionar el entrenamiento, validación y testing de modelos YOLO."""
    
    def __init__(self, modelPath, dataPath, outputPath, logger):
        """Inicializa el trainer con un modelo YOLO.
        
        Args:
            modelPath (str): Ruta al archivo del modelo YOLO
            dataPath (str): Ruta al archivo de datos
            outputPath (str): Ruta al directorio de salida
            logger (logging.Logger): Logger para registrar mensajes
        """

        
        self.logger = logger
        
        self.dataPath = dataPath
        # Limpiamos cache antes de cargar el modelo
        self.cleanCache(os.path.dirname(modelPath))
        
        self.outputPath = outputPath
        
        #Cargamos el modelo para evitar problemas de cache
        try:
            self.model = YOLO(modelPath)
            self.logger.info("Modelo cargado exitosamente")
        except Exception as e:
            self.logger.error(f"Error al cargar el modelo: {e}")
            raise

    def findCacheFiles(self, directory):
        """Encuentra y elimina archivos de cache.
        
        Args:
            directory (str): Directorio donde buscar archivos de cache
            
        Returns:
            list: Lista de rutas de archivos de cache eliminados
        """
        cache_paths = []

        for dirpath, _, filenames in os.walk(directory):
            for file in filenames:
                if file.endswith(".cache"):
                    full_path = os.path.join(dirpath, file)
                    self.logger.info(f"Archivo de cache encontrado: {full_path}")
                    try:
                        os.remove(full_path)
                        cache_paths.append(full_path)
                        self.logger.info("Cache eliminado exitosamente")
                    except Exception as e:
                        self.logger.error(f"Error al eliminar {full_path}: {e}")
        return cache_paths
    
    def cleanCache(self, directory):
        """Limpia los archivos de cache en un directorio.
        
        Args:
            directory (str): Directorio donde limpiar cache
        """
        self.logger.info(f"Buscando archivos de cache en {directory}")
        cache_paths = self.findCacheFiles(directory)
        if cache_paths:
            self.logger.info(f"Archivos de cache eliminados: {len(cache_paths)}")
        else:
            self.logger.info("No se encontraron archivos de cache")
    
    def train(self, config):
        """Entrena el modelo.
        
        Args:
            config (dict): Configuración de entrenamiento
            
        Returns:
            dict: Resultados del entrenamiento
        """
        self.logger.info("Iniciando entrenamiento")
        try:
            batch = config.get("batch_size", config.get("batch", 16))
            imgsz = config.get("img_size", config.get("imgsz", 640))
            device = config.get("device", "cpu")
            save_period = config.get("save_period", -1)
            seed = config.get("seed", 42)
            patience = config.get("patience", 10)
            workers = config.get("workers", 4)

            if config.get("augmentation", False):
                results = self.model.train(
                    data=self.dataPath,                         # Configuracion de los datos
                    epochs=config["epochs"],                    # Numero de epochs
                    batch=batch,                                 # Batch size
                    imgsz=imgsz,                                 # Tamaño de la imagen
                    device=device,                               # Dispositivo a utilizar
                    save_period=save_period,                     # Periodo de guardado
                    seed=seed,                                   # Semilla
                    patience=patience,                           # Paciencia
                    workers=workers,                             # Numero de workers
                    hsv_h=0,                                    # HSV-Hue augmentation
                    hsv_s=0,                                    # HSV-Saturation augmentation
                    hsv_v=0,                                    # HSV-Value augmentation
                    degrees=0,                                  # Image rotation
                    translate=0,                                # Image translation
                    scale=0,                                    # Image scaling
                    shear=0,                                    # Image shearing
                    perspective=0,                              # Perspective transformation
                    flipud=0,                                   # Vertical flip
                    fliplr=0,                                   # Horizontal flip
                    mosaic=0,                                   # Mosaic augmentation
                    mixup=0,                                    # Mixup augmentation
                    name=self.outputPath+"/train",              # Ruta de guardado
                )
            else:
                results = self.model.train(
                    data=self.dataPath,                          # Ruta al archivo de datos
                    epochs=config["epochs"],                     # Numero de epochs
                    batch=batch,                                  # Batch size
                    imgsz=imgsz,                                  # Tamaño de la imagen
                    device=device,                                # Dispositivo a utilizar
                    save_period=save_period,                      # Periodo de guardado
                    seed=seed,                                    # Semilla
                    patience=patience,                            # Paciencia
                    workers=workers,                              # Numero de workers
                    mosaic=False,                                # No usar mosaico
                    name=self.outputPath+"/train",               # Ruta de guardado
                )
            
            #Seleccionamos el mejor modelo 
            self.model = YOLO(self.model.trainer.best)
            return results
            
        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento: {e}")
            raise
    
    def validate(self, config):
        """Valida el modelo.
        
        Args:
            config (dict): Diccionario con los parámetros para model.val()
            
        Returns:
            dict: Resultados de la validación
        """
        self.logger.info("Iniciando validación")
        try:
            results = self.model.val(
                data=self.dataPath,                              # Configuracion de los datos
                batch=config.get("batch_size", config.get("batch", 16)),
                imgsz=config.get("img_size", config.get("imgsz", 640)),
                device=config.get("device", "cpu"),
                workers=config.get("workers", 4),
                name=self.outputPath+"/validation",              # Nombre de la carpeta de salida
            )
            return results
        except Exception as e:
            self.logger.error(f"Error durante la validación: {e}")
            raise
    
    def test(self, config):
        """Realiza testing del modelo.
        
        Args:
            config (dict): Diccionario con los parámetros para model.predict()
        
        Returns:
            dict: Resultados del testing
        """
        self.logger.info("Iniciando testing")
        try:
            results = self.model.predict(
                source=self.dataPath,                              # Ruta de la imagen
                conf=config.get("conf_threshold", 0.25),          # Confianza mínima para la detección
                imgsz=config.get("img_size", 640),                # Tamaño de la imagen
                device=config.get("device", "cpu"),              # Dispositivo a utilizar
                save=config.get("save_results", True),            # Guardar resultados
                save_txt=config.get("save_results", True),        # Guardar etiquetas en txt
                save_conf=config.get("save_conf", True),          # Guardar confianza
                name=self.outputPath+"/detect",                   # Nombre de la carpeta de salida
                exist_ok=True,                                     # Sobreescribir carpeta si existe
            )
            return results
        except Exception as e:
            self.logger.error(f"Error durante el testing: {e}")
            raise

