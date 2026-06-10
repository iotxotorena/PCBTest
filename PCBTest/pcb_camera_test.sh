#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_HOST="$SCRIPT_DIR"

PYTHON_SCRIPT_HOST="$WORKSPACE_HOST/pcb_camera_test.py"

CAMERA_SOURCE="0"
CAMERA_WIDTH="0"
CAMERA_HEIGHT="0"
OUTPUT_PATH="$WORKSPACE_HOST/results/gui_pcb_inspection/camera_test/latest_camera_test.jpg"

DOCKER_IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"

declare -a EXTRA_MOUNT_SOURCES=()
declare -a EXTRA_MOUNT_TARGETS=()
CONTAINER_PATH_RESULT=""
EXTRA_MOUNT_RESULT=""

mostrar_ayuda() {
  echo ""
  echo "Uso:"
  echo "  ./pcb_camera_test.sh --camera-source 0 --output-path /ruta/captura.jpg"
  echo ""
  echo "Opciones:"
  echo "  --camera-source VALUE     Ej: 0, 1, /dev/video0, /dev/video2"
  echo "  --camera-width N          0 = automático"
  echo "  --camera-height N         0 = automático"
  echo "  --output-path PATH"
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

    mkdir -p "$parent"

    ensure_extra_mount "$parent"
    CONTAINER_PATH_RESULT="$EXTRA_MOUNT_RESULT/$base"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
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
OUTPUT_PATH_HOST="$(expand_path "$OUTPUT_PATH")"

if [[ ! -f "$PYTHON_SCRIPT_HOST" ]]; then
  echo "No encuentro pcb_camera_test.py:"
  echo "  $PYTHON_SCRIPT_HOST"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH_HOST")"
mkdir -p "$WORKSPACE_HOST/.ultralytics"
mkdir -p "$WORKSPACE_HOST/.config"
mkdir -p "$WORKSPACE_HOST/.cache"

set_container_path "$OUTPUT_PATH_HOST" "file"
OUTPUT_PATH_CONT="$CONTAINER_PATH_RESULT"

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
CAMERA_SOURCE_CONT="$CAMERA_SOURCE"

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
fi

VIDEO_GID="$(getent group video | cut -d: -f3 || true)"

if [[ -n "$VIDEO_GID" ]]; then
  DOCKER_GROUPS+=(--group-add "$VIDEO_GID")
fi

PYTHON_ARGS=(
  python /workspace/pcb_camera_test.py
  --camera-source "$CAMERA_SOURCE_CONT"
  --camera-width "$CAMERA_WIDTH"
  --camera-height "$CAMERA_HEIGHT"
  --output-path "$OUTPUT_PATH_CONT"
)

echo ""
echo "Configuración TEST cámara:"
echo "  Usuario UID:GID:         $(id -u):$(id -g)"
echo "  workspace host:          $WORKSPACE_HOST"
echo "  camera source host/gui:  $CAMERA_SOURCE"
echo "  camera source container: $CAMERA_SOURCE_CONT"
echo "  camera width/height:     ${CAMERA_WIDTH}x${CAMERA_HEIGHT}"
echo "  output host:             $OUTPUT_PATH_HOST"
echo "  output container:        $OUTPUT_PATH_CONT"
echo "  video devices found:     $VIDEO_DEVICES_FOUND"
echo "  video group gid:         ${VIDEO_GID:-none}"
echo "  docker image:            $DOCKER_IMAGE"
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
echo "TEST cámara terminado."
echo "Captura guardada en:"
echo "  $OUTPUT_PATH_HOST"
echo ""
