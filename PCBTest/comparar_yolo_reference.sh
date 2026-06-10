#!/usr/bin/env bash
set -e

WORKSPACE_HOST="$HOME/yolo_ws"

SCRIPT_HOST="$WORKSPACE_HOST/comparar_yolo_reference.py"

PROCESSED_DIR="$WORKSPACE_HOST/results/prueba_box_homography"
REFERENCE_DIR="$WORKSPACE_HOST/referenceBoard"
OUTPUT_DIR=""

MATCH_BY="name"
MIN_IOU="0.20"
MAX_CENTER_DISTANCE="0.035"
MAX_CENTER_DISTANCE_RELAXED="0.060"

MIN_SIZE_RATIO="0.70"
MAX_SIZE_RATIO="1.30"

MIN_REF_COVERAGE="0.70"
MIN_DET_COVERAGE="0.50"
MIN_REF_COVERAGE_RELAXED="0.35"
MIN_DET_COVERAGE_RELAXED="0.25"

MAX_CANDIDATE_CENTER_DISTANCE="0.10"
MIN_CANDIDATE_IOU="0.02"
MIN_CANDIDATE_REF_COVERAGE="0.10"
MIN_CANDIDATE_DET_COVERAGE="0.10"

MIN_CONF="0.0"
NO_REFERENCE_NAMES_FOR_DETECTIONS=false

mostrar_ayuda() {
  echo ""
  echo "Uso:"
  echo "  ./comparar_yolo_reference.sh [opciones]"
  echo ""
  echo "Opciones:"
  echo "  --processed-dir                       Carpeta generada por procesar_pcb_homografia_yolo.sh"
  echo "  --reference-dir                       Carpeta con notes.json y labels/UNICO_FICHERO.txt"
  echo "  --output-dir                          Carpeta de salida. Por defecto: processed-dir/comparison"
  echo "  --match-by                            name|id. Por defecto: name"
  echo ""
  echo "Criterio OK:"
  echo "  --min-iou                             IoU mínimo fuerte. Por defecto: 0.20"
  echo "  --max-center-distance                 Distancia centro fuerte. Por defecto: 0.035"
  echo "  --max-center-distance-relaxed         Distancia centro relajada. Por defecto: 0.060"
  echo "  --min-ref-coverage                    Cobertura fuerte referencia. Por defecto: 0.70"
  echo "  --min-det-coverage                    Cobertura fuerte detección. Por defecto: 0.50"
  echo "  --min-ref-coverage-relaxed            Cobertura relajada referencia. Por defecto: 0.35"
  echo "  --min-det-coverage-relaxed            Cobertura relajada detección. Por defecto: 0.25"
  echo ""
  echo "Tamaño:"
  echo "  --min-size-ratio                      Ratio mínimo tamaño. Solo aviso. Por defecto: 0.70"
  echo "  --max-size-ratio                      Ratio máximo tamaño. Solo aviso. Por defecto: 1.30"
  echo ""
  echo "Puerta inicial de candidatos:"
  echo "  --max-candidate-center-distance       Distancia máxima para aceptar candidato local. Por defecto: 0.10"
  echo "  --min-candidate-iou                   IoU mínimo para aceptar candidato. Por defecto: 0.02"
  echo "  --min-candidate-ref-coverage          Cobertura mínima ref para aceptar candidato. Por defecto: 0.10"
  echo "  --min-candidate-det-coverage          Cobertura mínima det para aceptar candidato. Por defecto: 0.10"
  echo ""
  echo "Otros:"
  echo "  --min-conf                            Confianza mínima de detección. Por defecto: 0.0"
  echo "  --no-reference-names-for-detections   No renombrar detecciones usando notes.json"
  echo "  -h, --help                            Ayuda"
  echo ""
}

expand_path() {
  local p="$1"
  p="${p/#\~/$HOME}"
  realpath -m "$p"
}

to_container_path() {
  local p="$1"

  if [[ "$p" == /workspace* ]]; then
    echo "$p"
    return
  fi

  local abs
  abs="$(expand_path "$p")"

  local workspace_abs
  workspace_abs="$(expand_path "$WORKSPACE_HOST")"

  if [[ "$abs" != "$workspace_abs"* ]]; then
    echo ""
    echo "ERROR: esta ruta debe estar dentro de ~/yolo_ws:"
    echo "  $abs"
    echo ""
    exit 1
  fi

  echo "/workspace${abs#$workspace_abs}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --processed-dir)
      PROCESSED_DIR="$2"
      shift 2
      ;;
    --reference-dir)
      REFERENCE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --match-by)
      MATCH_BY="$2"
      shift 2
      ;;
    --min-iou)
      MIN_IOU="$2"
      shift 2
      ;;
    --max-center-distance)
      MAX_CENTER_DISTANCE="$2"
      shift 2
      ;;
    --max-center-distance-relaxed)
      MAX_CENTER_DISTANCE_RELAXED="$2"
      shift 2
      ;;
    --min-size-ratio)
      MIN_SIZE_RATIO="$2"
      shift 2
      ;;
    --max-size-ratio)
      MAX_SIZE_RATIO="$2"
      shift 2
      ;;
    --min-ref-coverage)
      MIN_REF_COVERAGE="$2"
      shift 2
      ;;
    --min-det-coverage)
      MIN_DET_COVERAGE="$2"
      shift 2
      ;;
    --min-ref-coverage-relaxed)
      MIN_REF_COVERAGE_RELAXED="$2"
      shift 2
      ;;
    --min-det-coverage-relaxed)
      MIN_DET_COVERAGE_RELAXED="$2"
      shift 2
      ;;
    --max-candidate-center-distance)
      MAX_CANDIDATE_CENTER_DISTANCE="$2"
      shift 2
      ;;
    --min-candidate-iou)
      MIN_CANDIDATE_IOU="$2"
      shift 2
      ;;
    --min-candidate-ref-coverage)
      MIN_CANDIDATE_REF_COVERAGE="$2"
      shift 2
      ;;
    --min-candidate-det-coverage)
      MIN_CANDIDATE_DET_COVERAGE="$2"
      shift 2
      ;;
    --min-conf)
      MIN_CONF="$2"
      shift 2
      ;;
    --no-reference-names-for-detections)
      NO_REFERENCE_NAMES_FOR_DETECTIONS=true
      shift
      ;;
    -h|--help)
      mostrar_ayuda
      exit 0
      ;;
    *)
      echo "Opción no reconocida: $1"
      mostrar_ayuda
      exit 1
      ;;
  esac
done

WORKSPACE_HOST="$(expand_path "$WORKSPACE_HOST")"
SCRIPT_HOST="$(expand_path "$SCRIPT_HOST")"
PROCESSED_DIR_HOST="$(expand_path "$PROCESSED_DIR")"
REFERENCE_DIR_HOST="$(expand_path "$REFERENCE_DIR")"

if [[ -n "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR_HOST="$(expand_path "$OUTPUT_DIR")"
else
  OUTPUT_DIR_HOST=""
fi

if [[ ! -f "$SCRIPT_HOST" ]]; then
  echo "No encuentro el script Python:"
  echo "  $SCRIPT_HOST"
  exit 1
fi

if [[ ! -d "$PROCESSED_DIR_HOST" ]]; then
  echo "No existe processed-dir:"
  echo "  $PROCESSED_DIR_HOST"
  exit 1
fi

if [[ ! -d "$REFERENCE_DIR_HOST" ]]; then
  echo "No existe reference-dir:"
  echo "  $REFERENCE_DIR_HOST"
  exit 1
fi

if [[ ! -f "$REFERENCE_DIR_HOST/notes.json" ]]; then
  echo "No existe notes.json:"
  echo "  $REFERENCE_DIR_HOST/notes.json"
  exit 1
fi

if [[ ! -d "$REFERENCE_DIR_HOST/labels" ]]; then
  echo "No existe la carpeta labels:"
  echo "  $REFERENCE_DIR_HOST/labels"
  exit 1
fi

LABEL_COUNT="$(find "$REFERENCE_DIR_HOST/labels" -maxdepth 1 -type f -name "*.txt" | wc -l)"

if [[ "$LABEL_COUNT" -ne 1 ]]; then
  echo "Debe haber exactamente UN fichero .txt en:"
  echo "  $REFERENCE_DIR_HOST/labels"
  echo "Encontrados: $LABEL_COUNT"
  echo ""
  find "$REFERENCE_DIR_HOST/labels" -maxdepth 1 -type f -name "*.txt" -print
  exit 1
fi

mkdir -p "$WORKSPACE_HOST/.ultralytics"
mkdir -p "$WORKSPACE_HOST/.config"
mkdir -p "$WORKSPACE_HOST/.cache"

PROCESSED_DIR_CONT="$(to_container_path "$PROCESSED_DIR_HOST")"
REFERENCE_DIR_CONT="$(to_container_path "$REFERENCE_DIR_HOST")"

DOCKER_CMD=(
  docker run --rm
  --user "$(id -u):$(id -g)"
  -e HOME=/workspace
  -e YOLO_CONFIG_DIR=/workspace/.ultralytics
  -e XDG_CONFIG_HOME=/workspace/.config
  -e XDG_CACHE_HOME=/workspace/.cache
  -v "$WORKSPACE_HOST":/workspace
  -w /workspace
  ultralytics/ultralytics:latest-jetson-jetpack6
  python /workspace/comparar_yolo_reference.py
  --processed-dir "$PROCESSED_DIR_CONT"
  --reference-dir "$REFERENCE_DIR_CONT"
  --match-by "$MATCH_BY"
  --min-iou "$MIN_IOU"
  --max-center-distance "$MAX_CENTER_DISTANCE"
  --max-center-distance-relaxed "$MAX_CENTER_DISTANCE_RELAXED"
  --min-size-ratio "$MIN_SIZE_RATIO"
  --max-size-ratio "$MAX_SIZE_RATIO"
  --min-ref-coverage "$MIN_REF_COVERAGE"
  --min-det-coverage "$MIN_DET_COVERAGE"
  --min-ref-coverage-relaxed "$MIN_REF_COVERAGE_RELAXED"
  --min-det-coverage-relaxed "$MIN_DET_COVERAGE_RELAXED"
  --max-candidate-center-distance "$MAX_CANDIDATE_CENTER_DISTANCE"
  --min-candidate-iou "$MIN_CANDIDATE_IOU"
  --min-candidate-ref-coverage "$MIN_CANDIDATE_REF_COVERAGE"
  --min-candidate-det-coverage "$MIN_CANDIDATE_DET_COVERAGE"
  --min-conf "$MIN_CONF"
)

if [[ "$NO_REFERENCE_NAMES_FOR_DETECTIONS" == true ]]; then
  DOCKER_CMD+=(--no-reference-names-for-detections)
fi

if [[ -n "$OUTPUT_DIR_HOST" ]]; then
  OUTPUT_DIR_CONT="$(to_container_path "$OUTPUT_DIR_HOST")"
  DOCKER_CMD+=(--output-dir "$OUTPUT_DIR_CONT")
fi

echo ""
echo "Configuración comparación:"
echo "  Usuario UID:GID:                    $(id -u):$(id -g)"
echo "  workspace host:                     $WORKSPACE_HOST"
echo "  processed-dir host:                 $PROCESSED_DIR_HOST"
echo "  reference-dir host:                 $REFERENCE_DIR_HOST"
echo "  output-dir host:                    ${OUTPUT_DIR_HOST:-$PROCESSED_DIR_HOST/comparison}"
echo "  match-by:                           $MATCH_BY"
echo "  min-iou:                            $MIN_IOU"
echo "  max-center-distance:                $MAX_CENTER_DISTANCE"
echo "  max-center-distance-relaxed:        $MAX_CENTER_DISTANCE_RELAXED"
echo "  size-ratio warning:                 $MIN_SIZE_RATIO - $MAX_SIZE_RATIO"
echo "  min-ref-coverage:                   $MIN_REF_COVERAGE"
echo "  min-det-coverage:                   $MIN_DET_COVERAGE"
echo "  min-ref-coverage-relaxed:           $MIN_REF_COVERAGE_RELAXED"
echo "  min-det-coverage-relaxed:           $MIN_DET_COVERAGE_RELAXED"
echo "  max-candidate-center-distance:      $MAX_CANDIDATE_CENTER_DISTANCE"
echo "  min-candidate-iou:                  $MIN_CANDIDATE_IOU"
echo "  min-candidate-ref-coverage:         $MIN_CANDIDATE_REF_COVERAGE"
echo "  min-candidate-det-coverage:         $MIN_CANDIDATE_DET_COVERAGE"
echo "  min-conf:                           $MIN_CONF"
echo "  renombrar detecciones por ref:      $([[ "$NO_REFERENCE_NAMES_FOR_DETECTIONS" == true ]] && echo false || echo true)"
echo ""

"${DOCKER_CMD[@]}"

echo ""
echo "Terminado."
echo ""
