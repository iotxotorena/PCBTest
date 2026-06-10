#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_SCRIPT="$SCRIPT_DIR/pcb_gui_inspeccion.py"

if [[ ! -f "$GUI_SCRIPT" ]]; then
  echo "No encuentro la GUI:"
  echo "  $GUI_SCRIPT"
  exit 1
fi

cd "$SCRIPT_DIR"

python3 "$GUI_SCRIPT"
