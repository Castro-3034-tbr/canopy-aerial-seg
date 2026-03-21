from pathlib import Path
import os
import logging
import json
import sys

from config.config import loadConfig

from utils.util_YOLO import YOLOTrainer



#Definicion de rutas
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "config.json"

#Cargamos la configuración
config = loadConfig(CONFIG_PATH)
print("Configuración cargada:", config)

# Configurar logging
if not os.path.exists(config["pathLog"]):
    os.mkdir(config["pathLog"])
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config["pathLog"] + "/yolo_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

#Creacion de la clase de entrenamiento
Yolo = YOLOTrainer(config["model"]["path"], config["pathData"], config["pathResult"], logger)

#Analizamos las opciones 
train = config["task"]["train"]
val = config["task"]["val"]
test = config["task"]["test"]
logging.info(f"Opciones de tarea - Train: {train}, Val: {val}, Test: {test}")

if train:
    logger.info("Iniciando entrenamiento...")
    Yolo.train(config["training"])
    logger.info("Entrenamiento finalizado")

if val:
    logger.info("Iniciando validación...")
    Yolo.validate(config["validation"])
    logger.info("Validación finalizada")

if test:
    logger.info("Iniciando testing...")
    test_config = config.get("testing", config.get("test", {}))
    Yolo.test(test_config)
    logger.info("Testing finalizado")
