"""Script para redondear todas las coordenadas negativas a 0 en archivos de etiquetas YOLO."""

import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_coordinates_in_file(file_path: Path, tolerance: float = 0) -> bool:
    """
    Corrige todas las coordenadas en un archivo de etiquetas YOLO.
    
    Validaciones y correcciones realizadas:
    - Reemplaza coordenadas negativas por 0.0
    - Reemplaza coordenadas > 1.0 por 1.0
    - Elimina coordenadas impares (polígonos incompletos)
    - Solo guarda líneas válidas (≥7 tokens, número impar de elementos)
    
    Args:
        file_path: Ruta del archivo .txt a procesar
        tolerance: No utilizado (mantener para compatibilidad)
        
    Returns:
        True si se realizaron cambios, False en caso contrario
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error al leer {file_path}: {e}")
        return False
    
    modified = False
    corrected_lines = []
    
    for line_num, line in enumerate(lines, 1):
        line = line.rstrip('\n')
        parts = line.split()
        
        # Validar que sea una línea válida (al menos clase + 6 coordenadas)
        if len(parts) < 1:
            corrected_lines.append(line)
            continue
        
        try:
            class_id = parts[0]
            coords = list(map(float, parts[1:]))
            
            # Revisar cada coordenada: normalizar al rango [0.0, 1.0]
            corrected_coords = []
            for coord in coords:
                if coord < 0:
                    corrected_coords.append(0.0)
                    modified = True
                elif coord > 1.0:
                    corrected_coords.append(1.0)
                    modified = True
                else:
                    corrected_coords.append(coord)
            
            # Si el número de coordenadas es impar, eliminar la última (polígono incompleto)
            if len(corrected_coords) % 2 != 0:
                logger.warning(f"{file_path}:{line_num} - Polígono incompleto, eliminando última coordenada")
                corrected_coords = corrected_coords[:-1]
                modified = True
            
            # Solo incluir líneas válidas: ≥7 tokens (class_id + 6 coordenadas mínimo)
            if len(corrected_coords) >= 6:
                corrected_line = f"{class_id} " + " ".join(f"{c:.10g}" for c in corrected_coords)
                corrected_lines.append(corrected_line)
            else:
                logger.warning(f"{file_path}:{line_num} - Línea descartada por insuficientes coordenadas")
                modified = True
            
        except ValueError:
            logger.warning(f"{file_path}:{line_num} - Línea con formato incorrecto: {line}")
            # No incluir esta línea
            modified = True
    
    # Si hubo cambios, guardar el archivo
    if modified:
        try:
            with open(file_path, 'w') as f:
                if corrected_lines:
                    f.write('\n'.join(corrected_lines) + '\n')
            logger.info(f"✓ Corregido: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar {file_path}: {e}")
            return False
    
    return False


def process_labels_directory(labels_dir: str | Path) -> None:
    """
    Procesa todos los archivos .txt en un directorio de etiquetas.
    
    Args:
        labels_dir: Ruta del directorio que contiene los archivos .txt
    """
    labels_path = Path(labels_dir)
    
    if not labels_path.is_dir():
        logger.error(f"Error: {labels_dir} no es un directorio válido")
        return
    
    # Buscar todos los archivos .txt
    txt_files = list(labels_path.glob("*.txt"))
    
    if not txt_files:
        logger.warning(f"No se encontraron archivos .txt en {labels_dir}")
        return
    
    logger.info(f"Procesando {len(txt_files)} archivo(s) en {labels_dir}")
    
    corrected_count = 0
    for txt_file in sorted(txt_files):
        if fix_coordinates_in_file(txt_file):
            corrected_count += 1
    
    logger.info(f"✓ Proceso completado: {corrected_count}/{len(txt_files)} archivo(s) corregido(s)")


if __name__ == "__main__":
    # Cambiar esta ruta si tus etiquetas están en otro lugar
    labels_directory = Path("/media/castro/Castro/Fotogrametrias/Moeche/labels")
    
    # Alternativamente, especifica la ruta manualmente:
    # labels_directory = Path("ruta/a/tus/etiquetas")
    
    process_labels_directory(labels_directory)
