#!/bin/bash

set -euo pipefail

# Este codigo divide un conjunto de datos de imagenes y sus etiquetas en tres conjuntos: entrenamiento, validación y test.
# 
# Ejemplo:
#   ./Divisiondata.sh 

DATA_DIR="./data"  						# Directorio de entrada que contiene las imágenes y sus etiquetas (en formato .txt)
OUTPUT_DIR="./dataPruebas"  			# Directorio de salida donde se crearán las subcarpetas train, validation y test
TRAIN_PCT="80"							# Porcentaje de datos para entrenamiento
VAL_PCT="10"							# Porcentaje de datos para validación
TEST_PCT="10"							# Porcentaje de datos para test

TRAIN_DIR="${OUTPUT_DIR}/train"			# Directorio para el conjunto de entrenamiento
VAL_DIR="${OUTPUT_DIR}/validation"		# Directorio para el conjunto de validación
TEST_DIR="${OUTPUT_DIR}/test"			# Directorio para el conjunto de test


#Comprobar que el directorio de entrada existe
if [[ ! -d "${DATA_DIR}" ]]; then
	echo "Error: el directorio '${DATA_DIR}' no existe."
	exit 1
fi

#Comprobar que los porcentajes suman 100
if (( TRAIN_PCT + VAL_PCT + TEST_PCT != 100 )); then
	echo "Error: train_pct + val_pct + test_pct debe sumar 100."
	exit 1
fi

# Obtener la lista de archivos de imagen en el directorio de entrada
readarray -d '' -t IMAGE_FILES < <(
	find "${DATA_DIR}" -maxdepth 1 -type f \
    \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0
)

TOTAL_FILES="${#IMAGE_FILES[@]}"
if (( TOTAL_FILES == 0 )); then
	echo "No se encontraron imágenes en '${DATA_DIR}'."
	exit 0
fi

# Verificar que cada imagen tenga su etiqueta correspondiente
readarray -d '' -t LABEL_FILES < <(find "${DATA_DIR}" -maxdepth 1 -type f -iname "*.txt" -print0)

echo "Cantidad de imagenes: ${TOTAL_FILES}"
echo "Cantidad de etiquetas: ${#LABEL_FILES[@]}"

# Mezclar los archivos de imagen
shuf -z -e "${IMAGE_FILES[@]}" | readarray -d '' -t IMAGE_FILES

# Calcular la cantidad de archivos para cada conjunto
TRAIN_FILES=$((TOTAL_FILES * TRAIN_PCT / 100))
VAL_FILES=$((TOTAL_FILES * VAL_PCT / 100))
TEST_FILES=$((TOTAL_FILES - TRAIN_FILES - VAL_FILES))

echo "Total de archivos: ${TOTAL_FILES}"
echo "Archivos para entrenar: ${TRAIN_FILES}"
echo "Archivos para validar: ${VAL_FILES}"
echo "Archivos para test: ${TEST_FILES}"

# Crear los directorios de salida
mkdir -p "${TRAIN_DIR}" "${VAL_DIR}" "${TEST_DIR}"

# Función para mover una imagen y su etiqueta correspondiente
move_with_label() {
	local image_path="$1"
	local destination="$2"
	local base_name
	local label_path

	# Obtener el nombre base del archivo de imagen (sin extensión)
	base_name="$(basename "${image_path}")"
	base_name="${base_name%.*}"
	label_path="${DATA_DIR}/${base_name}.txt"

	# Mover la imagen al destino
	mv "${image_path}" "${destination}/"
	if [[ -f "${label_path}" ]]; then
		mv "${label_path}" "${destination}/"
	else
		echo "Advertencia: No se encontró la etiqueta para ${image_path}"
	fi
}

# Mover los archivos a sus respectivos directorios
for ((i = 0; i < TRAIN_FILES; i++)); do
  move_with_label "${IMAGE_FILES[${i}]}" "${TRAIN_DIR}"
done

# Mover los archivos de validación
for ((i = TRAIN_FILES; i < TRAIN_FILES + VAL_FILES; i++)); do
  move_with_label "${IMAGE_FILES[${i}]}" "${VAL_DIR}"
done

# Mover los archivos de test
for ((i = TRAIN_FILES + VAL_FILES; i < TOTAL_FILES; i++)); do
  move_with_label "${IMAGE_FILES[${i}]}" "${TEST_DIR}"
done

echo "Archivos movidos a entrenamiento: ${TRAIN_DIR}"
echo "Archivos movidos a validación: ${VAL_DIR}"
echo "Archivos movidos a test: ${TEST_DIR}"