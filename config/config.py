import json
import sys


def validateConfig(config):
    """Valida que la configuración tenga la estructura y tipos esperados.

    Además, normaliza claves de evaluación para mantener compatibilidad con el
    flujo actual de `main.py`.
    """
    if not isinstance(config, dict):
        raise ValueError("La configuración debe ser un objeto JSON (diccionario).")

    # Validación de claves y tipos básicos
    required_top_level = ["pathData", "pathResult", "pathLog", "task", "model", "training"]
    missing_top_level = [key for key in required_top_level if key not in config]
    if missing_top_level:
        raise ValueError(f"Faltan claves obligatorias en config: {missing_top_level}")

    # Validar rutas y que existan
    for key in ["pathData", "pathResult", "pathLog"]:
        if not isinstance(config[key], str) or not config[key].strip(): 
            raise ValueError(f"La clave '{key}' debe ser un string no vacío y la ruta debe existir: {config[key]}")

    # Validar estructura de tareas
    if not isinstance(config["task"], dict):
        raise ValueError("'task' debe ser un objeto.")
    for key in ["train", "val", "test"]:
        if key not in config["task"]:
            raise ValueError(f"Falta la clave 'task.{key}'.")
        if not isinstance(config["task"][key], bool):
            raise ValueError(f"'task.{key}' debe ser booleano.")
    
    # Validar modelo
    if not isinstance(config["model"], dict):
        raise ValueError("'model' debe ser un objeto.")
    for key in ["name", "path"]:
        if key not in config["model"]:
            raise ValueError(f"Falta la clave 'model.{key}'.")
        if not isinstance(config["model"][key], str) or not config["model"][key].strip():
            raise ValueError(f"'model.{key}' debe ser un string no vacío.")

    # Validar entrenamiento
    if not isinstance(config["training"], dict):
        raise ValueError("'training' debe ser un objeto.")
    required_training = ["augmentation", "epochs", "batch_size", "img_size", "device"]
    missing_training = [key for key in required_training if key not in config["training"]]
    if missing_training:
        raise ValueError(f"Faltan claves obligatorias en 'training': {missing_training}")

    # Validar tipos específicos de entrenamiento
    int_fields = ["epochs", "batch_size", "img_size"]
    for key in int_fields:
        if not isinstance(config["training"][key], int) or config["training"][key] <= 0:
            raise ValueError(f"'training.{key}' debe ser un entero positivo.")

    # Validar booleanos y strings
    if not isinstance(config["training"]["augmentation"], bool):
        raise ValueError("'training.augmentation' debe ser booleano.")
    if not isinstance(config["training"]["device"], str) or not config["training"]["device"].strip():
        raise ValueError("'training.device' debe ser un string no vacío.")

    # Compatibilidad: si existe 'evaluation', usarla como base para validación y testing.
    if "evaluation" in config:
        if not isinstance(config["evaluation"], dict):
            raise ValueError("'evaluation' debe ser un objeto.")
        config.setdefault("validation", dict(config["evaluation"]))
        config.setdefault("testing", dict(config["evaluation"]))

    for section in ["validation", "testing"]:
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"'{section}' debe ser un objeto.")

    return config


def loadConfig(config_path):
    """Carga la configuración desde un archivo JSON y la valida.

    Args:
        config_path (str): Ruta al archivo de configuración

    Returns:
        dict: Diccionario con la configuración validada
    """
    try:
        # Cargar el archivo JSON
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Validar la configuración y normalizar claves de evaluación
        config = validateConfig(config)
        return config
    except FileNotFoundError:
        print(f"Archivo de configuración no encontrado: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error al parsear JSON: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error de validación en configuración: {e}")
        sys.exit(1)
