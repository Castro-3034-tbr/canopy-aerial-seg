# Plan de refactorizacion para unificar `api/`, `train/` y `eda/`

## Objetivo

Unificar las tres carpetas principales del proyecto para evitar duplicidad de codigo, imports cruzados y modelos de datos incompatibles. El foco de esta refactorizacion debe ser:

- centralizar tipos, constantes y utilidades compartidas en `common/`
- dejar `api/`, `train/` y `eda/` como modulos de dominio, no como copias parciales
- eliminar dependencias circulares o engañosas entre carpetas
- separar claramente configuracion comun, configuracion de API y configuracion de entrenamiento

## Situacion actual detectada

- `api/` y `train/` contienen muchos archivos duplicados o practicamente identicos
- varios archivos dentro de `api/` importan desde `train.*` en lugar de usar su propio namespace
- `eda/` tiene su propio `core/` con `types.py`, `constants.py` y `logger.py`, duplicando patrones ya presentes en otros modulos
- `common/` existe, pero ahora mismo esta vacio y no esta actuando como capa compartida
- `config/config.json` mezcla dos esquemas distintos:
  - configuracion de API/inferencia en streaming
  - configuracion de entrenamiento/prediccion offline
- el editor parece seguir mostrando referencias a `src/`, pero en el repo real la carpeta activa es `train/`

## Hallazgos concretos importantes

- `api/core/config.py` y `train/core/config.py` son iguales
- `api/core/dependencies.py` y `train/core/dependencies.py` son iguales
- `api/processes/*.py` y `train/processes/*.py` son iguales
- `api/perception/*.py` y `train/perception/*.py` son iguales
- `api/utils/file_utils.py` y `train/utils/file_utils.py` son iguales
- `api/core/logger.py` importa desde `train.core.constants`, lo que confirma que `api/` no esta desacoplado
- `main_api.py` arranca usando `train.core.*`, no `api.core.*`
- `config/config.json` contiene dos JSON pegados, por lo que ahora mismo no representa una fuente de configuracion limpia para ambos flujos

## Estructura objetivo recomendada

La estructura mas clara para este proyecto seria esta:

```text
common/
  config/
  constants.py
  logger.py
  types/
  utils/

api/
  routes/
  services/
  runtime/
  mqtt/

train/
  training/
  inference/
  dataset/

eda/
  io/
  metrics/
  visualization/

config/
  api.json
  train.json
  eda.json
  shared.json
```

## Que deberia ir a `common/`

### 1. Tipos base compartidos

Mover a `common/types/` lo que realmente se reutiliza entre modulos:

- `StrictModel`
- `Coordinates`
- `BoundingBox`
- aliases de tipos geometricos o de imagen si van a ser compartidos
- tipos de salida genericos como `OutputFile` u `OutputPathResult`

Separacion recomendada:

- `common/types/base.py`: `StrictModel`, tipos comunes de Pydantic
- `common/types/geometry.py`: `Coordinates`, `BoundingBox`, `Polygon`, `Mask`
- `common/types/media.py`: arrays de imagen, tipos de frame, tipos de salida
- `common/types/config.py`: modelos base reutilizables de configuracion

### 2. Constantes globales

Crear `common/constants.py` para:

- `PROJECT_ROOT`
- rutas por defecto compartidas
- formato comun de logs
- nombres estables de carpetas de salida

No mezclar ahi constantes exclusivas de cada dominio:

- las constantes HTTP deben quedarse en `api/`
- las constantes de entrenamiento deben quedarse en `train/`
- las constantes de analisis visual deben quedarse en `eda/`

### 3. Logging comun

Extraer `configure_logging` a `common/logger.py`, parametrizando:

- nombre del fichero log
- nivel de log
- directorio de logs

Despues:

- `api/core/logger.py` puede desaparecer o convertirse en wrapper fino
- `train/core/logger.py` puede desaparecer o convertirse en wrapper fino
- `eda/core/logger.py` puede desaparecer o convertirse en wrapper fino

### 4. Utilidades de filesystem

Mover a `common/utils/filesystem.py`:

- `resolve_path`
- helpers de limpieza
- helpers para directorios de salida

Esto evita que `train/utils/filesystem.py` sea la dependencia real de otros modulos.

## Que deberia quedarse por dominio

### `api/`

Debe quedarse con todo lo relativo a ejecucion online y FastAPI:

- rutas
- ciclo de vida de la app
- gestion de streams
- procesos lector/procesador
- MQTT
- endpoints de subida de imagen/video

### `train/`

Debe quedarse con todo lo relativo a pipeline offline:

- entrenamiento
- validacion
- prediccion por lotes
- split de dataset

### `eda/`

Debe quedarse con todo lo relativo a exploracion y analisis:

- carga de imagenes y labels
- metricas
- visualizaciones
- exportacion de resultados EDA

## Tareas de refactorizacion recomendadas

### Fase 1. Limpiar la arquitectura

- decidir si `api/` o `train/` sera la fuente canonica del runtime compartido actual
- renombrar o eliminar la carpeta duplicada que no vaya a quedar como base
- dejar de usar referencias antiguas a `src/` en el IDE, scripts o documentacion

Mi recomendacion:

- conservar `api/` para todo lo que sea servicio online
- conservar `train/` para entrenamiento y pipeline offline
- mover lo compartido a `common/`
- borrar duplicados solo cuando los imports ya apunten a `common/`

### Fase 2. Extraer codigo comun real

- crear `common/types/`
- crear `common/logger.py`
- crear `common/constants.py`
- crear `common/utils/filesystem.py`
- mover ahi el codigo comun desde `api/core/*`, `train/core/*` y `eda/core/*`

### Fase 3. Separar configuraciones

- dividir `config/config.json` en varios archivos
- crear `config/api.json`
- crear `config/train.json`
- crear `config/eda.json` si EDA necesita configuracion propia
- crear opcionalmente `config/shared.json` para rutas o defaults comunes

Modificaciones necesarias:

- `api/core/config.py` debe cargar `api.json`
- `train/core/config.py` debe cargar `train.json`
- si compartes modelos base, ambos loaders deben importar desde `common.types.config`

### Fase 4. Unificar tipos

- crear modelos base compartidos para geometria, frames y configuracion
- evitar tener dos `ModelConfig` con significados distintos en carpetas distintas
- renombrar modelos ambiguos

Renombres recomendados:

- `api.core.types.ModelConfig` -> `InferenceModelConfig`
- `train.core.types.ModelConfig` -> `TrainingModelConfig`
- `AppConfigModel` -> `ApiAppConfig`
- `PipelineConfig` -> `TrainPipelineConfig`

### Fase 5. Rehacer imports

- reemplazar imports `from train.core...` dentro de `api/`
- reemplazar imports locales duplicados por `common.*`
- dejar cada modulo importando solo:
  - su propio dominio
  - `common`

Objetivo de imports:

- `api/*` no debe importar desde `train/*` salvo una dependencia muy justificada
- `train/*` no debe importar desde `api/*`
- `eda/*` no debe depender de `api/*` ni de `train/*`

### Fase 6. Consolidar modulos duplicados

Archivos candidatos claros a consolidacion:

- `api/core/config.py` y `train/core/config.py`
- `api/core/dependencies.py` y `train/core/dependencies.py`
- `api/perception/postprocessing.py` y `train/perception/postprocessing.py`
- `api/perception/yolo_inference.py` y `train/perception/yolo_inference.py`
- `api/processes/reader_process.py` y `train/processes/reader_process.py`
- `api/processes/processor_process.py` y `train/processes/processor_process.py`
- `api/processes/stream_manager.py` y `train/processes/stream_manager.py`
- `api/utils/file_utils.py` y `train/utils/file_utils.py`

Recomendacion practica:

- si el comportamiento es exactamente el mismo, dejar una sola implementacion
- si cambia el contexto de uso, extraer el nucleo a `common/` y dejar wrappers delgados en cada dominio

### Fase 7. Revisar entry points

- `main_api.py` debe importar desde `api.*`
- `main_train.py` debe importar desde `train.*`
- `main.py` debe importar desde `eda.*`

Ahora mismo `main_api.py` usa `train.core.*`, y eso deberia corregirse cuando `api/` quede desacoplado.

## Modificaciones concretas por carpeta

### Cambios en `common/`

- crear `common/types/base.py`
- crear `common/types/geometry.py`
- crear `common/types/config.py`
- crear `common/types/media.py`
- crear `common/constants.py`
- crear `common/logger.py`
- crear `common/utils/filesystem.py`
- añadir `common/__init__.py` y, si hace falta, `common/types/__init__.py`

### Cambios en `api/`

- dejar de importar desde `train.*`
- mover `core/logger.py` a wrapper o eliminarlo
- mover `core/constants.py` a constantes solo de API
- mover tipos compartidos a `common`
- conservar en `api/core/types.py` solo tipos exclusivos de runtime HTTP y streams

Tipos que probablemente deben quedarse en `api`:

- `StreamSession`
- `StreamStartedResponse`
- `StreamStoppedResponse`
- `StreamsHealthResponse`
- `AppRuntime`
- protocolos de manager, eventos y colas si solo pertenecen al runtime multiproceso de la API

### Cambios en `train/`

- eliminar las copias del runtime de API si no forman parte del entrenamiento offline
- dejar `train/core/types.py` solo con configuracion de entrenamiento e inferencia offline
- mover `resolve_path` a `common`
- revisar si `train/api/` tiene sentido o si en realidad deberia desaparecer

Punto importante:

- la presencia de `train/api/routes.py` indica que `train/` esta mezclando responsabilidades de API y entrenamiento

### Cambios en `eda/`

- mover `StrictModel`, `Coordinates` y `BoundingBox` a `common`
- dejar `eda/core/types.py` solo con modelos propios del analisis
- mover `configure_logging` comun a `common/logger.py`
- mover constantes globales a `common/constants.py`

Tipos que si deben seguir en `eda`:

- `ImageData`
- `LabelData`
- `ImagesLoaderResult`
- `LabelsLoaderResult`
- `AnalysisResult`

## Orden recomendado de ejecucion

Para que la refactorizacion no rompa todo a la vez, haria esto en este orden:

1. Crear `common/` con tipos, logger y filesystem compartidos.
2. Migrar imports de `eda/` a `common/`, porque es el dominio mas aislado.
3. Migrar imports compartidos de `train/`.
4. Migrar imports compartidos de `api/`.
5. Separar `config/config.json` en archivos por dominio.
6. Corregir `main_api.py` para que use `api.*`.
7. Eliminar duplicados una vez comprobado que nadie los importa.

## Riesgos que hay que vigilar

- romper validaciones Pydantic al fusionar modelos con nombres iguales pero semantica distinta
- mezclar configuracion de entrenamiento con configuracion de streaming
- dejar imports residuales a `train.*` dentro de `api/`
- borrar archivos duplicados antes de actualizar los entry points
- asumir que `src/` sigue existiendo cuando el repo real ya usa `train/`

## Resultado esperado al terminar

Al final de la refactorizacion deberias tener:

- una sola capa comun real en `common/`
- `api/`, `train/` y `eda/` con responsabilidades claras
- cero imports cruzados innecesarios entre dominios
- configuraciones separadas por contexto
- menos duplicacion y menos riesgo de tocar una copia y dejar la otra desactualizada

## Siguiente paso recomendado

El mejor siguiente paso tecnico es empezar por crear la nueva estructura de `common/types`, `common/logger.py` y `common/utils/filesystem.py`, y despues mover primero `eda`, porque es el bloque menos acoplado y sirve como prueba de la estrategia antes de tocar `api` y `train`.

## Inventario de tipos a unificar

Esta seccion te sirve como mapa para la segunda parte de la refactorizacion: unificar nombres y mover tipos a una sola fuente de verdad.

### Tipos que claramente representan lo mismo

- `api.core.types.StrictModel`
- `train.core.types.StrictModel`
- `eda.core.types.StrictModel`

Nombre recomendado:

- `BaseStrictModel`

Motivo:

- es el modelo base comun de Pydantic para todo el proyecto
- evita que `StrictModel` parezca un modelo de negocio

Ubicacion recomendada:

- `common/types/base.py`

---

- `api.core.types.Coordinates`
- `eda.core.types.Coordinates`

Nombre recomendado:

- `NormalizedPoint`

Alternativa valida:

- `NormalizedCoordinates`

Motivo:

- el dato representa un punto normalizado en el frame o imagen
- `Coordinates` es correcto, pero demasiado generico

Ubicacion recomendada:

- `common/types/geometry.py`

---

- `api.core.types.Polygon`
- uso equivalente en EDA a traves de `MaskLabel` con `list[Coordinates]`

Nombre recomendado:

- `NormalizedPolygon`

Motivo:

- deja claro que son vertices normalizados y no coordenadas absolutas

Ubicacion recomendada:

- `common/types/geometry.py`

---

- `api.core.types.BoundingBox`
- `eda.core.types.BoundingBox`

Nombre recomendado:

- `NormalizedBoundingBox`

Motivo:

- ambos representan cajas en coordenadas normalizadas
- ahora mismo tienen forma distinta:
  - `api` usa `p1` y `p2`
  - `eda` usa `x_min`, `y_min`, `x_max`, `y_max`
- para unificar scripts, la forma mas clara es la de `x_min`, `y_min`, `x_max`, `y_max`, y opcionalmente mantener `width` y `height`

Recomendacion de esquema:

- `x_min`
- `y_min`
- `x_max`
- `y_max`
- `width`
- `height`

Ubicacion recomendada:

- `common/types/geometry.py`

---

- `api.core.types.ModelConfig`
- `train.core.types.ModelConfig`

No son iguales, pero representan configuracion de modelo YOLO y conviene separarlos con mejores nombres.

Nombres recomendados:

- `InferenceModelConfig` para el actual de `api`
- `TrainingModelConfig` para el actual de `train`

Motivo:

- hoy comparten el mismo nombre y eso induce errores al importar
- sus campos y validaciones no representan exactamente el mismo contrato

Ubicacion recomendada:

- `common/types/config.py` si quieres centralizar modelos de configuracion
- o en `api/types.py` y `train/types.py` si prefieres mantenerlos por dominio

### Tipos compatibles que conviene redefinir como familia comun

- `api.core.types.FrameArray`
- uso implicito de arrays de imagen en `eda`

Nombre recomendado:

- `ImageArray`

Motivo:

- es mas general y sirve tanto para streaming como para EDA

Ubicacion recomendada:

- `common/types/media.py`

---

- `api.core.types.FrameMask`
- uso implicito de mascaras en `eda`

Nombre recomendado:

- `MaskArray`

Motivo:

- describe el dato y evita atarlo solo al concepto de frame en streaming

Ubicacion recomendada:

- `common/types/media.py`

---

- `api.core.types.OutputPathResult`

Nombre recomendado:

- `GeneratedFileInfo`

Alternativa valida:

- `OutputFileInfo`

Motivo:

- `tuple[Path, str]` es poco expresivo
- si se mantiene como alias, al menos el nombre debe indicar que representa ruta mas mime type

Mejor aun:

- convertirlo de alias a modelo:
  - `path: Path`
  - `media_type: str`

Ubicacion recomendada:

- `common/types/media.py`

---

- `api.core.types.OutputFile`

Nombre recomendado:

- `GeneratedFile`

Motivo:

- es el mismo concepto que el alias anterior, pero mejor modelado
- conviene quedarte con una sola representacion y eliminar la duplicidad entre alias y modelo

Ubicacion recomendada:

- `common/types/media.py`

### Tipos que deberian renombrarse para reflejar mejor el dominio

- `api.core.types.Detection`

Nombre recomendado:

- `SegmentationDetection`

Alternativa si quieres algo mas general:

- `ModelDetection`

Motivo:

- el tipo no solo guarda clase y confianza; tambien incluye mascara, bbox y centroide
- el nombre actual es demasiado amplio para futuras tareas

Ubicacion recomendada:

- `common/types/inference.py`

---

- `api.core.types.MaskMetric`

Nombre recomendado:

- `MaskStatistics`

Motivo:

- el tipo contiene varias metricas agregadas de una mascara, no una sola metrica

Ubicacion recomendada:

- `common/types/inference.py`

---

- `api.core.types.FramePackage`

Nombre recomendado:

- `VideoFramePacket`

Alternativa valida:

- `FrameEnvelope`

Motivo:

- representa un frame con metadatos de transporte entre procesos
- `Package` es correcto, pero `Packet` o `Envelope` describe mejor el uso

Ubicacion recomendada:

- `common/types/media.py` si se reutiliza fuera de API
- `api/types/runtime.py` si queda solo en streaming

---

- `eda.core.types.ImageData`

Nombre recomendado:

- `ImageMetadata`

Motivo:

- ese modelo no contiene el array de imagen, sino metadatos de fichero y dimensiones

Ubicacion recomendada:

- `eda/types.py` o `common/types/dataset.py` si se reutiliza en entrenamiento

---

- `eda.core.types.LabelData`

Nombre recomendado:

- `AnnotationFileData`

Alternativa valida:

- `LabelFileData`

Motivo:

- representa el contenido parseado de un fichero de anotaciones, no una sola etiqueta

Ubicacion recomendada:

- `common/types/dataset.py`

### Tipos de resultado que pueden agruparse con una convencion comun

- `eda.core.types.ImagesLoaderResult`
- `eda.core.types.LabelsLoaderResult`

Nombres recomendados:

- `ImageLoadResult`
- `AnnotationLoadResult`

Motivo:

- siguen el mismo patron: elementos validos + elementos invalidos
- si luego creas mas loaders, te conviene mantener la misma convención

Ubicacion recomendada:

- `common/types/io.py` o `eda/types.py`

---

- `train.core.types.PipelineResults`

Nombre recomendado:

- `TrainingPipelineResults`

Motivo:

- deja claro que el resultado pertenece al pipeline offline, no al runtime general del proyecto

Ubicacion recomendada:

- `train/types.py`

### Tipos de configuracion que deberian normalizarse

- `api.core.types.ApiConfig`

Nombre recomendado:

- `ApiServerConfig`

Motivo:

- describe host, puerto y comportamiento del servidor, no la app completa

---

- `api.core.types.SavePathConfigModel`

Nombre recomendado:

- `OutputPathsConfig`

Motivo:

- expresa mejor que agrupa rutas de salida

---

- `api.core.types.AppConfigModel`

Nombre recomendado:

- `ApiAppConfig`

Motivo:

- evita colisionar con otras configuraciones raiz del proyecto

---

- `train.core.types.PipelineConfig`

Nombre recomendado:

- `TrainPipelineConfig`

Motivo:

- deja claro que es la configuracion de entrenamiento/inferencia offline

---

- `train.core.types.TaskConfig`

Nombre recomendado:

- `PipelineTaskSelection`

Motivo:

- representa la seleccion de tareas activas, no una configuracion tecnica compleja

---

- `train.core.types.TrainConfig`

Nombre recomendado:

- `TrainingRunConfig`

---

- `train.core.types.ValidationConfig`

Nombre recomendado:

- `ValidationRunConfig`

---

- `train.core.types.PredictConfig`

Nombre recomendado:

- `PredictionRunConfig`

Motivo comun:

- todos esos modelos describen parametros de ejecucion de una etapa concreta

### Tipos y aliases que conviene eliminar o reducir

- `api.core.types.ModelConfigModel`

Recomendacion:

- eliminarlo

Motivo:

- es solo un alias redundante de `ModelConfig`

---

- `api.core.types.BoundingBoxList`
- `api.core.types.CentroidList`

Recomendacion:

- mantenerlos solo si mejoran mucho la legibilidad
- si no, usar directamente `list[NormalizedBoundingBox]` y `list[NormalizedPoint]`

---

- `eda.core.types.LabelsSizes`
- `eda.core.types.MetricValues`
- `eda.core.types.LabelsPerImage`

Recomendacion:

- evaluar si aportan realmente semantica o si complican la lectura
- si se mantienen, renombrarlos con formato singular coherente:
  - `LabelAreaValues`
  - `MetricSeries`
  - `LabelCountPerImage`

### Tipos que no conviene unificar porque son propios de un dominio

Dejar solo en `api/`:

- `MQTTConfig`
- `StreamSession`
- `StreamStartedResponse`
- `StreamStoppedResponse`
- `StopAllStreamsResponse`
- `StreamHealth`
- `StreamsHealthResponse`
- `HealthResponse`
- `EventLike`
- `FrameQueueLike`
- `SharedData`
- `ProjectData`
- `RuntimeState`
- `GlobalManager`
- `PahoMQTTClient`

Dejar solo en `train/`:

- `TrainingRunConfig`
- `ValidationRunConfig`
- `PredictionRunConfig`
- `TrainingPipelineResults`

Dejar solo en `eda/`:

- `AnalysisResult`
- contadores y agregados estadisticos propios de EDA

### Propuesta final de nombres canonicos

Si quieres una lista corta de nombres objetivo para dejar cerrada la unificacion, esta seria mi propuesta:

- `BaseStrictModel`
- `NormalizedPoint`
- `NormalizedPolygon`
- `NormalizedBoundingBox`
- `ImageArray`
- `MaskArray`
- `VideoFramePacket`
- `SegmentationDetection`
- `MaskStatistics`
- `GeneratedFile`
- `ApiServerConfig`
- `OutputPathsConfig`
- `ApiAppConfig`
- `InferenceModelConfig`
- `TrainingModelConfig`
- `PipelineTaskSelection`
- `TrainPipelineConfig`
- `TrainingRunConfig`
- `ValidationRunConfig`
- `PredictionRunConfig`
- `ImageMetadata`
- `AnnotationFileData`
- `ImageLoadResult`
- `AnnotationLoadResult`
- `TrainingPipelineResults`

### Regla practica para unificar los scripts

Cuando empieces a tocar los scripts, usa esta regla:

- si un tipo describe geometria, media, rutas o modelos base, debe salir de `common`
- si describe runtime HTTP, streams o MQTT, debe quedarse en `api`
- si describe entrenamiento o ejecucion offline, debe quedarse en `train`
- si describe analisis exploratorio, metricas o resultados estadisticos, debe quedarse en `eda`
