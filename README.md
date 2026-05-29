# Segmentación de objetos mediante análisis de imágenes aérea​

Sistema de segmentación de instancias sobre imágenes fotogramétricas aéreas orientado a la detección y delimitación de zonas arboladas. Construido sobre YOLO (Ultralytics) con una arquitectura modular que cubre el flujo completo:

```
análisis exploratorio → entrenamiento → inferencia en producción vía API REST de streaming RTSP y imagenes estáticas.
```
---

## Stack tecnológico

| Capa          | Tecnología                            |
|---------------|---------------------------------------|
| Lenguaje      | Python 3.10+                          |
| Modelo        | Ultralytics YOLO (`.pt`), PyTorch     |
| API / ASGI    | FastAPI, Starlette, Uvicorn           |
| Visión        | OpenCV, NumPy                         |
| Streaming     | PyAV / FFmpeg, MediaMTX (RTSP/RTMP)   |
| Telemetría    | MQTT (paho-mqtt)                      |
| Análisis      | pandas, matplotlib                    |
| Validación    | Pydantic v2                           |
| Config        | JSON, YAML                            |

---

## Arquitectura del sistema

El pipeline se divide en tres módulos independientes y ejecutables por separado:

```
TFM/
├── eda/            # Análisis exploratorio del dataset
├── train/          # Entrenamiento y validación del modelo
├── api/            # API REST para inferencia en producción
├── config/         # Archivos de configuración JSON
├── common/         # Funciones y utilidades compartidas
├── data/           # Dataset fotogramétrico (imágenes + etiquetas)
├── logs/           # Logs de entrenamiento y ejecución
├── models/         # Pesos de modelos preentrenados y entrenados
└── output/         # Artefactos generados (plots, pesos, logs)
```

---

## Módulo 1 — EDA (Exploratory Data Analysis)

Caracteriza el dataset antes del entrenamiento. Detecta problemas de calidad que impactan directamente en el rendimiento del modelo: desbalanceo de clases, anotaciones erróneas, sesgo espacial, imágenes degradadas.

### Estructura

```
eda/
├── core/types.py                  # Tipos: Dataset, ImageRecord, LabelRecord, Metrics
├── io/
│   ├── loaders.py                 # Carga de imágenes y etiquetas desde disco
│   ├── parsers.py                 # Parseo de formatos YOLO / COCO
│   └── writers.py                 # Persistencia de resultados y logs
├── metrics/
│   ├── image_metrics.py           # Calidad visual: brillo, contraste, blur, resolución
│   └── label_metrics.py           # Anotaciones: área, densidad, IoU, distribución espacial
├── utils/
│   ├── colors.py                  # Normalización y conversión de espacios de color
│   └── geometry.py                # IoU, centroides, bounding boxes
├── visualization/
│   ├── density_plots.py           # Distribución espacial de etiquetas
│   └── statistical_plots.py       # Histogramas, boxplots, scatter
└── output/
    ├── eda_results.txt
    └── plots/
run_eda.py                         # Punto de entrada
```

### Métricas — Imágenes

| Métrica       | Función                       | Descripción                           |
|---------------|-------------------------------|---------------------------------------|
| Tipo          | `count_image_types`           | Distribución de formatos (JPEG, PNG…) |
| Resolución    | `count_image_sizes`           | Distribución width × height           |
| Aspect ratio  | `count_image_aspect_ratios`   | Detección de deformaciones            |
| Brillo        | `compute_images_brightness`   | Intensidad media de píxeles           |
| Contraste     | `compute_images_contrast`     | Variabilidad de intensidades          |
| Blur          | `compute_images_blur`         | Varianza del Laplaciano               |

### Métricas — Etiquetas

| Métrica               | Función                       | Descripción                           |
|-----------------------|-------------------------------|---------------------------------------|
| Densidad              | `count_labels_per_image`      | Objetos anotados por imagen           |
| Área individual       | `compute_label_areas`         | Tamaño de cada instancia              |
| Área total            | `compute_labels_areas`        | Ocupación agregada por imagen         |
| Aspect ratio          | `count_label_aspect_ratios`   | Detección de formas anómalas          |
| Centroide             | `compute_label_centers`       | Base para análisis espacial           |
| Distribución X        | `count_label_quadrants_x`     | Sesgo lateral                         |
| Distribución Y        | `count_label_quadrants_y`     | Sesgo vertical                        |
| IoU entre etiquetas   | `compute_labels_iou`          | Solapamientos / etiquetas redundantes |

### Configuración de rutas

La configuración del EDA se realiza directamente en el script principal mediante rutas locales del dataset y directorios de salida.


```python
DATASET_PATH = Path("/media/castro/Castro/Fotogrametrias/Moeche")
RESULTS_DIR = Path("./output/eda")
RESULTS_FILE = RESULTS_DIR / "eda_results.txt"
PLOTS_DIR = RESULTS_DIR / "plots"
images_loader = load_images(DATASET_PATH / "images")
labels_loader = load_labels(DATASET_PATH / "labels")
```

### Ejecución

Para ejecutar el análisis completo:

```bash
python3 run_eda.py
```

---

## Módulo 2 — Entrenamiento

Preprocesa el dataset, configura el modelo YOLO para segmentación de instancias, ejecuta el ciclo de entrenamiento y persiste los pesos resultantes.

### Estructura

```
train/
├── core/
│   ├── config.py                  # Carga de config_train.json
│   └── types.py                   # Tipos: TrainConfig, ModelRecord, Metrics
├── dataset/
│   ├── loaders.py                 # Carga y preprocesado de datos
│   └── split.py                   # División train / val / test
├── inference/
│   └── predictor.py               # Inferencia con el modelo entrenado
├── training/
│   ├── pipeline.py                # Orquestación del ciclo completo
│   ├── trainer.py                 # Bucle de entrenamiento y optimización
│   └── validator.py               # Cálculo de métricas durante validación
└── utils/
    └── filesystem.py              # Resolución de paths, limpieza de caché
run_train.py                       # Punto de entrada
```

### Configuración

El entrenamiento se parametriza íntegramente desde `config/config_train.json`:

```json
{
  "model": "yolo11n-seg.pt",
  "data": "config/data.yaml",
  "epochs": 100,
  "imgsz": 640,
  "batch": 16,
  "device": "cuda",
  "project": "output/runs",
  "name": "moeche_seg"
}
```

### Ejecución

Para ejecutar el entrenamiento del modelo:

```bash
python3 run_train.py
```
Los pesos del modelo se guardan en `output/runs/<name>/weights/best.pt`.

---

## Módulo 3 — Inferencia / Producción

API REST para inferencia bajo demanda sobre imágenes estáticas o streaming RTSP. El sistema procesa las entradas, ejecuta la inferencia con el modelo entrenado y devuelve resultados estructurados (JSON) con las zonas detectadas y sus áreas reales.

### Estructura

```
api/
├── core/
|    ├── config.py                  # Carga de config_api.json
|    ├── constants.py               # Constantes globales (e.g., rutas, parámetros)
|    ├── data_init.py               # Inicialización de recursos compartidos
|    ├── dependencies.py            # Definición de dependencias para FastAPI
|    └── types.py                   # Tipos Pydantic para request/response
├── mqtt/
|   ├── connection.py               # Configuración y gestión de la conexión MQTT
│   └── publisher.py                # Funciones para publicar mensajes de telemetría
├── perception/
|   ├── analisis.py                 # Funciones para analizar los resultados de inferencia
|   ├── postprocessing.py           # Funciones de postprocesado
|   └── yolo_inference.py           # Interfaz con el modelo YOLO para inferencia
├── routes/
|   └── routes.py                   # Definición de endpoints y lógica de manejo de solicitudes
|
└── utils/
    └── file_utils.py               # Funciones para manejo de archivos de imagen y archivos zip
run_api.py                       # Punto de entrada
```

### Endpoints principales

Para la comunicacion con la API se han definido los siguientes endpoints:

| Método    | Ruta              | Descripción                                                      |
|-----------|-------------------|------------------------------------------------------------------|
| `POST`    | `/stream/start`   |Inicia sesión de streaming RTSP                                   |
| `POST`    | `/stream/stop`    |Finaliza sesión de streaming RTSP                                 |
| `POST`    | `/analysis/file`  |Analiza una imagen o un archivo zip                               |
| `POST`    | `/test/detection` |Realiza la deteccion sobre una imagen de prueba                   |
| `GET`     | `/health`         |Devuelve el estado del servicio y de los stream que están activos |


**Parámetros de Endpoints**

- **`/stream/start` (POST)**:
    - **`streamId`**: (`session_id`: `str | None`) — Query alias `streamId`. Identificador único del stream. Por defecto `None` (se genera uno automáticamente).
    - **`rtspUrl`**: (`rtsp_url`: `str`) — Query alias `rtspUrl`. URL RTSP del flujo de video. Requerido.
    - **`saveLog`**: (`save_log`: `bool`) — Query alias `saveLog`. Por defecto `False`. Guarda un log del stream si es `True`.
    - **`saveInference`**: (`save_inference`: `bool`) — Query alias `saveInference`. Por defecto `False`. Guarda inferencias si es `True`.
    - **`confidenceThreshold`**: (`confidence_threshold`: `float`) — Query alias `confidenceThreshold`. Umbral de confianza; por defecto `DEFAULT_CONFIDENCE_THRESHOLD`.
    - **`mqttHost`**: (`mqtt_host`: `str`) — Query alias `mqttHost`. Host del broker MQTT. Requerido.
    - **`mqttPort`**: (`mqtt_port`: `str`) — Query alias `mqttPort`. Puerto del broker MQTT. Requerido.
    - **`mqttTopic`**: (`mqtt_topic`: `str`) — Query alias `mqttTopic`. Topic MQTT; por defecto `DEFAULT_MQTT_TOPIC`.
    - **`overlap`**: (`overlap`: `tuple[float, float]`) — Query alias `overlap`. Fracción de máscara para recortar solapes; por defecto `(0.1, 0.1)`.
    - **`gsd`**: (`gsd`: `float`) — Query alias `gsd`. Ground Sample Distance en m/px; por defecto `0.1`.

- **`/stream/stop` (POST)**:
    - **`sessionId`**: (`session_id`: `str | None`) — Query alias `sessionId`. Si se indica detiene sólo esa sesión; si no, detiene todas. Por defecto `None`.

- **`/analysis/file` (POST)**:
    - **`file`**: `UploadFile` — multipart file. Soporta `image/*` y `application/zip`.
    - **`confidenceThreshold`**: (`confidence_threshold`: `float`) — Query alias `confidenceThreshold`. Umbral de confianza; por defecto `DEFAULT_CONFIDENCE_THRESHOLD`.
    - **`overlap`**: (`overlap`: `tuple[float, float]`) — Query alias `overlap`. Por defecto `(0.1, 0.1)`.
    - **`gsd`**: (`gsd`: `float`) — Query alias `gsd`. Por defecto `0.1`.

- **`/test/detection` (POST)**:
    - **`file`**: `UploadFile` — multipart file. Debe ser `image/*`.
    - **`confidenceThreshold`**: (`confidence_threshold`: `float`) — Query alias `confidenceThreshold`. Por defecto `DEFAULT_CONFIDENCE_THRESHOLD`.
    - **`overlap`**: (`overlap`: `tuple[float, float]`) — Query alias `overlap`. Por defecto `(0.1, 0.1)`.

- **`/health` (GET)**: sin parámetros.

### Configuración y ejecución

Para simplificar el despliegue, se ha centralizado la configuracion de la API en `config/config_api.json`:

Por lo cual para ejecutar la API se ha centralizado en el siguiente comando:

```bash
python3 run_api.py
```

---

## Instalación

### Requisitos

- Python 3.10+
- CUDA 11.8+ (recomendado para entrenamiento)
- FFmpeg (requerido para el módulo de streaming)

### Setup

```bash
git clone https://github.com/Castro-3034-tbr/canopy-aerial-seg.git
cd canopy-aerial-seg

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset

El sistema espera un dataset con estructura compatible con YOLO Segmentation:

```
dataset/
├── images/
└── labels/
```

El archivo `config/data.yaml` debe apuntar a las rutas correspondientes.

---

## Estimación de área real

A partir de las máscaras de segmentación y los metadatos de la misión (altitud de vuelo, resolución del sensor, GSD), el sistema calcula el área real en m² de las zonas detectadas.

El GSD (Ground Sampling Distance) actúa como factor de escala:

```
Área_real [m²] = Área_píxeles × GSD² [m/px]²
```

Este parametro se importa en los atributos de las llamadas a los diferentes endpoints de la API para que el sistema pueda realizar el cálculo de área real en cada inferencia.

---

## Contacto
Para cualquier consulta o colaboración, no dudes en contactarme:
 ** Daniel Castro Gómez **
- Email: danielcastrogomezzz@gmail.com