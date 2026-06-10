#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_HOST="$SCRIPT_DIR"

PYTHON_SCRIPT_HOST="$WORKSPACE_HOST/pcb_realtime_pipeline.py"

OUTPUT_DIR="$WORKSPACE_HOST/results/realtime_pcb"
REFERENCE_DIR="$WORKSPACE_HOST/referenceBoard"
CONFIG_PATH="$WORKSPACE_HOST/config_fiduciales.json"
ORIENTATION_TEMPLATE="$WORKSPACE_HOST/keypoints/serigrafia.png"
COMPONENT_MODEL=""

CAMERA_SOURCE="0"
CAMERA_SOURCE_CONT="0"
IMG_SIZE="640"
CONF="0.49"

HOMOGRAPHY_METHOD="hough"
MIN_AREA_RATIO="0.02"
SIDE_FRAC="0.18"

ALLOW_MASK_TOUCH_BORDER=false
BORDER_MARGIN_PX="3"
BORDER_MIN_GREEN_PIXELS="20"

ORIENTATION_EXPECTED_QUADRANT=""
ORIENTATION_MIN_SCORE="0.45"
ORIENTATION_MIN_SCALE="0.55"
ORIENTATION_MAX_SCALE="0.80"
ORIENTATION_SCALE_STEP="0.03"
ALLOW_LOW_ORIENTATION_SCORE=false
NO_ORIENTATION=false

MATCH_BY="name"
MIN_IOU="0.20"
MAX_CENTER_DISTANCE="0.035"
MAX_CENTER_DISTANCE_RELAXED="0.060"

INTERVAL="0"
DURATION="0"
LIMIT="1"
WARMUP_FRAMES="5"

CAMERA_WIDTH="0"
CAMERA_HEIGHT="0"

SAVE_HISTORY=false

DOCKER_IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"

declare -a EXTRA_MOUNT_SOURCES=()
declare -a EXTRA_MOUNT_TARGETS=()
CONTAINER_PATH_RESULT=""
EXTRA_MOUNT_RESULT=""

mostrar_ayuda() {
  echo ""
  echo "Uso:"
  echo "  ./pcb_realtime_pipeline.sh [opciones]"
  echo ""
  echo "Rutas:"
  echo "  --output-dir PATH"
  echo "  --reference-dir PATH"
  echo "  --config PATH"
  echo "  --orientation-template PATH"
  echo "  --component-model PATH_ABSOLUTO"
  echo ""
  echo "Cámara:"
  echo "  --camera-source VALUE       Ej: 0, 1, /dev/video0, /dev/video2"
  echo "  --camera-width N"
  echo "  --camera-height N"
  echo ""
  echo "YOLO:"
  echo "  --imgsz N"
  echo "  --conf FLOAT"
  echo ""
  echo "Homografía:"
  echo "  --homography-method hough|lines|box"
  echo "  --min-area-ratio FLOAT"
  echo "  --side-frac FLOAT"
  echo "  --allow-mask-touch-border"
  echo "  --border-margin-px N"
  echo "  --border-min-green-pixels N"
  echo ""
  echo "Orientación:"
  echo "  --orientation-template PATH"
  echo "  --orientation-expected-quadrant tl|tr|br|bl"
  echo "  --orientation-min-score FLOAT"
  echo "  --orientation-min-scale FLOAT"
  echo "  --orientation-max-scale FLOAT"
  echo "  --orientation-scale-step FLOAT"
  echo "  --allow-low-orientation-score"
  echo "  --no-orientation"
  echo ""
  echo "Comparación:"
  echo "  --match-by name|id"
  echo "  --min-iou FLOAT"
  echo "  --max-center-distance FLOAT"
  echo "  --max-center-distance-relaxed FLOAT"
  echo ""
  echo "Ejecución:"
  echo "  --interval FLOAT"
  echo "  --duration FLOAT"
  echo "  --limit N"
  echo "  --warmup-frames N"
  echo "  --save-history"
  echo ""
}

expand_path() {
  local p="$1"
  p="${p/#\~/$HOME}"
  realpath -m "$p"
}

is_inside_workspace() {
  local abs="$1"
  local ws="$2"

  [[ "$abs" == "$ws" || "$abs" == "$ws/"* ]]
}

ensure_extra_mount() {
  local source="$1"

  for i in "${!EXTRA_MOUNT_SOURCES[@]}"; do
    if [[ "${EXTRA_MOUNT_SOURCES[$i]}" == "$source" ]]; then
      EXTRA_MOUNT_RESULT="${EXTRA_MOUNT_TARGETS[$i]}"
      return
    fi
  done

  local idx="${#EXTRA_MOUNT_SOURCES[@]}"
  local target="/external_mount_${idx}"

  EXTRA_MOUNT_SOURCES+=("$source")
  EXTRA_MOUNT_TARGETS+=("$target")
  EXTRA_MOUNT_RESULT="$target"
}

set_container_path() {
  local host_path="$1"
  local kind="$2"

  local abs
  abs="$(expand_path "$host_path")"

  local ws
  ws="$(expand_path "$WORKSPACE_HOST")"

  if is_inside_workspace "$abs" "$ws"; then
    CONTAINER_PATH_RESULT="/workspace${abs#$ws}"
    return
  fi

  if [[ "$kind" == "dir" ]]; then
    ensure_extra_mount "$abs"
    CONTAINER_PATH_RESULT="$EXTRA_MOUNT_RESULT"
  else
    local parent
    local base
    parent="$(dirname "$abs")"
    base="$(basename "$abs")"

    ensure_extra_mount "$parent"
    CONTAINER_PATH_RESULT="$EXTRA_MOUNT_RESULT/$base"
  fi
}

guess_default_model() {
  local candidates=(
    "$WORKSPACE_HOST/train37/weights/best.pt"
    "$WORKSPACE_HOST/train37/weights/best.engine"
    "$WORKSPACE_HOST/weights/best.pt"
    "$WORKSPACE_HOST/weights/best.engine"
  )

  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]]; then
      echo "$p"
      return
    fi
  done

  echo ""
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --reference-dir)
      REFERENCE_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --orientation-template)
      ORIENTATION_TEMPLATE="$2"
      shift 2
      ;;
    --component-model)
      COMPONENT_MODEL="$2"
      shift 2
      ;;
    --camera-source)
      CAMERA_SOURCE="$2"
      shift 2
      ;;
    --camera-width)
      CAMERA_WIDTH="$2"
      shift 2
      ;;
    --camera-height)
      CAMERA_HEIGHT="$2"
      shift 2
      ;;
    --imgsz)
      IMG_SIZE="$2"
      shift 2
      ;;
    --conf)
      CONF="$2"
      shift 2
      ;;
    --homography-method)
      HOMOGRAPHY_METHOD="$2"
      shift 2
      ;;
    --min-area-ratio)
      MIN_AREA_RATIO="$2"
      shift 2
      ;;
    --side-frac)
      SIDE_FRAC="$2"
      shift 2
      ;;
    --allow-mask-touch-border)
      ALLOW_MASK_TOUCH_BORDER=true
      shift
      ;;
    --border-margin-px)
      BORDER_MARGIN_PX="$2"
      shift 2
      ;;
    --border-min-green-pixels)
      BORDER_MIN_GREEN_PIXELS="$2"
      shift 2
      ;;
    --orientation-expected-quadrant)
      ORIENTATION_EXPECTED_QUADRANT="$2"
      shift 2
      ;;
    --orientation-min-score)
      ORIENTATION_MIN_SCORE="$2"
      shift 2
      ;;
    --orientation-min-scale)
      ORIENTATION_MIN_SCALE="$2"
      shift 2
      ;;
    --orientation-max-scale)
      ORIENTATION_MAX_SCALE="$2"
      shift 2
      ;;
    --orientation-scale-step)
      ORIENTATION_SCALE_STEP="$2"
      shift 2
      ;;
    --allow-low-orientation-score)
      ALLOW_LOW_ORIENTATION_SCORE=true
      shift
      ;;
    --no-orientation)
      NO_ORIENTATION=true
      shift
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
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --warmup-frames)
      WARMUP_FRAMES="$2"
      shift 2
      ;;
    --save-history)
      SAVE_HISTORY=true
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
PYTHON_SCRIPT_HOST="$(expand_path "$PYTHON_SCRIPT_HOST")"
OUTPUT_DIR_HOST="$(expand_path "$OUTPUT_DIR")"
REFERENCE_DIR_HOST="$(expand_path "$REFERENCE_DIR")"
CONFIG_PATH_HOST="$(expand_path "$CONFIG_PATH")"
ORIENTATION_TEMPLATE_HOST="$(expand_path "$ORIENTATION_TEMPLATE")"

if [[ -z "$COMPONENT_MODEL" ]]; then
  COMPONENT_MODEL="$(guess_default_model)"
fi

if [[ -z "$COMPONENT_MODEL" ]]; then
  echo ""
  echo "ERROR: no se ha indicado modelo YOLO."
  echo "Usa:"
  echo "  --component-model /ruta/absoluta/modelo.pt"
  echo ""
  exit 1
fi

COMPONENT_MODEL_HOST="$(expand_path "$COMPONENT_MODEL")"

if [[ ! -f "$PYTHON_SCRIPT_HOST" ]]; then
  echo "No encuentro pcb_realtime_pipeline.py:"
  echo "  $PYTHON_SCRIPT_HOST"
  exit 1
fi

if [[ ! -d "$REFERENCE_DIR_HOST" ]]; then
  echo "No existe reference-dir:"
  echo "  $REFERENCE_DIR_HOST"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH_HOST" ]]; then
  echo "No existe config:"
  echo "  $CONFIG_PATH_HOST"
  exit 1
fi

if [[ "$NO_ORIENTATION" == false && ! -f "$ORIENTATION_TEMPLATE_HOST" ]]; then
  echo "No existe orientation-template:"
  echo "  $ORIENTATION_TEMPLATE_HOST"
  exit 1
fi

if [[ ! -f "$COMPONENT_MODEL_HOST" ]]; then
  echo "No existe component-model:"
  echo "  $COMPONENT_MODEL_HOST"
  exit 1
fi

mkdir -p "$OUTPUT_DIR_HOST"
mkdir -p "$WORKSPACE_HOST/.ultralytics"
mkdir -p "$WORKSPACE_HOST/.config"
mkdir -p "$WORKSPACE_HOST/.cache"

set_container_path "$OUTPUT_DIR_HOST" "dir"
OUTPUT_DIR_CONT="$CONTAINER_PATH_RESULT"

set_container_path "$REFERENCE_DIR_HOST" "dir"
REFERENCE_DIR_CONT="$CONTAINER_PATH_RESULT"

set_container_path "$CONFIG_PATH_HOST" "file"
CONFIG_PATH_CONT="$CONTAINER_PATH_RESULT"

set_container_path "$COMPONENT_MODEL_HOST" "file"
COMPONENT_MODEL_CONT="$CONTAINER_PATH_RESULT"

if [[ "$NO_ORIENTATION" == false ]]; then
  set_container_path "$ORIENTATION_TEMPLATE_HOST" "file"
  ORIENTATION_TEMPLATE_CONT="$CONTAINER_PATH_RESULT"
else
  ORIENTATION_TEMPLATE_CONT=""
fi

DOCKER_MOUNTS=(
  -v "$WORKSPACE_HOST":/workspace
)

for i in "${!EXTRA_MOUNT_SOURCES[@]}"; do
  DOCKER_MOUNTS+=(
    -v "${EXTRA_MOUNT_SOURCES[$i]}:${EXTRA_MOUNT_TARGETS[$i]}"
  )
done

DOCKER_DEVICES=()
DOCKER_GROUPS=()
VIDEO_DEVICES_FOUND=false

shopt -s nullglob
for dev in /dev/video*; do
  if [[ -e "$dev" ]]; then
    VIDEO_DEVICES_FOUND=true
    DOCKER_DEVICES+=(--device "$dev:$dev")
  fi
done
shopt -u nullglob

if [[ "$CAMERA_SOURCE" =~ ^[0-9]+$ ]]; then
  EXPECTED_DEVICE="/dev/video${CAMERA_SOURCE}"

  if [[ ! -e "$EXPECTED_DEVICE" ]]; then
    echo ""
    echo "AVISO: se ha pedido cámara '$CAMERA_SOURCE', pero en el host no existe:"
    echo "  $EXPECTED_DEVICE"
    echo ""
    echo "Dispositivos /dev/video* encontrados en el host:"
    ls -l /dev/video* 2>/dev/null || true
    echo ""
  fi

  CAMERA_SOURCE_CONT="$CAMERA_SOURCE"

elif [[ "$CAMERA_SOURCE" == /dev/video* ]]; then
  if [[ ! -e "$CAMERA_SOURCE" ]]; then
    echo ""
    echo "AVISO: se ha pedido cámara '$CAMERA_SOURCE', pero no existe en el host."
    echo ""
    echo "Dispositivos /dev/video* encontrados en el host:"
    ls -l /dev/video* 2>/dev/null || true
    echo ""
  fi

  CAMERA_SOURCE_CONT="$CAMERA_SOURCE"

else
  CAMERA_SOURCE_CONT="$CAMERA_SOURCE"
fi

VIDEO_GID="$(getent group video | cut -d: -f3 || true)"
if [[ -n "$VIDEO_GID" ]]; then
  DOCKER_GROUPS+=(--group-add "$VIDEO_GID")
fi

PYTHON_ARGS=(
  python /workspace/pcb_realtime_pipeline.py
  --camera-source "$CAMERA_SOURCE_CONT"
  --output-dir "$OUTPUT_DIR_CONT"
  --reference-dir "$REFERENCE_DIR_CONT"
  --config "$CONFIG_PATH_CONT"
  --component-model "$COMPONENT_MODEL_CONT"
  --imgsz "$IMG_SIZE"
  --conf "$CONF"
  --homography-method "$HOMOGRAPHY_METHOD"
  --min-area-ratio "$MIN_AREA_RATIO"
  --side-frac "$SIDE_FRAC"
  --border-margin-px "$BORDER_MARGIN_PX"
  --border-min-green-pixels "$BORDER_MIN_GREEN_PIXELS"
  --orientation-min-score "$ORIENTATION_MIN_SCORE"
  --orientation-min-scale "$ORIENTATION_MIN_SCALE"
  --orientation-max-scale "$ORIENTATION_MAX_SCALE"
  --orientation-scale-step "$ORIENTATION_SCALE_STEP"
  --match-by "$MATCH_BY"
  --min-iou "$MIN_IOU"
  --max-center-distance "$MAX_CENTER_DISTANCE"
  --max-center-distance-relaxed "$MAX_CENTER_DISTANCE_RELAXED"
  --interval "$INTERVAL"
  --duration "$DURATION"
  --limit "$LIMIT"
  --warmup-frames "$WARMUP_FRAMES"
  --camera-width "$CAMERA_WIDTH"
  --camera-height "$CAMERA_HEIGHT"
)

if [[ "$NO_ORIENTATION" == true ]]; then
  PYTHON_ARGS+=(--no-orientation)
else
  PYTHON_ARGS+=(--orientation-template "$ORIENTATION_TEMPLATE_CONT")
fi

if [[ -n "$ORIENTATION_EXPECTED_QUADRANT" ]]; then
  PYTHON_ARGS+=(--orientation-expected-quadrant "$ORIENTATION_EXPECTED_QUADRANT")
fi

if [[ "$ALLOW_MASK_TOUCH_BORDER" == true ]]; then
  PYTHON_ARGS+=(--allow-mask-touch-border)
fi

if [[ "$ALLOW_LOW_ORIENTATION_SCORE" == true ]]; then
  PYTHON_ARGS+=(--allow-low-orientation-score)
fi

if [[ "$SAVE_HISTORY" == true ]]; then
  PYTHON_ARGS+=(--save-history)
fi

echo ""
echo "Configuración realtime:"
echo "  Usuario UID:GID:              $(id -u):$(id -g)"
echo "  app/workspace host:           $WORKSPACE_HOST"
echo "  output host:                  $OUTPUT_DIR_HOST"
echo "  reference host:               $REFERENCE_DIR_HOST"
echo "  config host:                  $CONFIG_PATH_HOST"
echo "  orientation host:             $ORIENTATION_TEMPLATE_HOST"
echo "  model host:                   $COMPONENT_MODEL_HOST"
echo "  output container:             $OUTPUT_DIR_CONT"
echo "  reference container:          $REFERENCE_DIR_CONT"
echo "  config container:             $CONFIG_PATH_CONT"
echo "  orientation container:        $ORIENTATION_TEMPLATE_CONT"
echo "  model container:              $COMPONENT_MODEL_CONT"
echo "  camera source host/gui:       $CAMERA_SOURCE"
echo "  camera source container:      $CAMERA_SOURCE_CONT"
echo "  camera width/height:          ${CAMERA_WIDTH}x${CAMERA_HEIGHT}"
echo "  video devices found:          $VIDEO_DEVICES_FOUND"
echo "  video group gid:              ${VIDEO_GID:-none}"
echo "  homography-method:            $HOMOGRAPHY_METHOD"
echo "  conf:                         $CONF"
echo "  max-center-distance:          $MAX_CENTER_DISTANCE"
echo "  max-center-distance-relaxed:  $MAX_CENTER_DISTANCE_RELAXED"
echo "  docker image:                 $DOCKER_IMAGE"
echo ""

if [[ "${#EXTRA_MOUNT_SOURCES[@]}" -gt 0 ]]; then
  echo "Montajes externos:"
  for i in "${!EXTRA_MOUNT_SOURCES[@]}"; do
    echo "  ${EXTRA_MOUNT_SOURCES[$i]} -> ${EXTRA_MOUNT_TARGETS[$i]}"
  done
  echo ""
fi

echo "Dispositivos de vídeo montados:"
if [[ "${#DOCKER_DEVICES[@]}" -gt 0 ]]; then
  printf '  %s\n' "${DOCKER_DEVICES[@]}"
else
  echo "  Ninguno"
fi
echo ""

docker run --rm \
  --user "$(id -u):$(id -g)" \
  "${DOCKER_GROUPS[@]}" \
  --device-cgroup-rule='c 81:* rmw' \
  -e HOME=/workspace \
  -e YOLO_CONFIG_DIR=/workspace/.ultralytics \
  -e XDG_CONFIG_HOME=/workspace/.config \
  -e XDG_CACHE_HOME=/workspace/.cache \
  "${DOCKER_DEVICES[@]}" \
  "${DOCKER_MOUNTS[@]}" \
  -w /workspace \
  "$DOCKER_IMAGE" \
  "${PYTHON_ARGS[@]}"

echo ""
echo "Terminado."
echo "Resultados en:"
echo "  $OUTPUT_DIR_HOST"
echo ""
