#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2


def parse_camera_source(value):
    value = str(value).strip()

    if value.isdigit():
        return int(value)

    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test rápido de cámara: abre fuente OpenCV, captura un frame y lo guarda."
    )

    parser.add_argument("--camera-source", required=True)
    parser.add_argument("--camera-width", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=0)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--warmup-frames", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    source = parse_camera_source(args.camera_source)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("")
    print("TEST cámara")
    print(f"  source:       {source}")
    print(f"  width:        {args.camera_width}")
    print(f"  height:       {args.camera_height}")
    print(f"  output-path:  {output_path}")
    print("")

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara/fuente: {args.camera_source}")

    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)

    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    for i in range(max(0, args.warmup_frames)):
        cap.read()
        time.sleep(0.03)

    ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        raise RuntimeError("La cámara se abrió, pero no devolvió ningún frame válido.")

    h, w = frame.shape[:2]

    ok = cv2.imwrite(str(output_path), frame)

    if not ok:
        raise RuntimeError(f"No se pudo guardar la imagen en: {output_path}")

    print("Captura correcta.")
    print(f"  frame size:   {w}x{h}")
    print(f"  saved:        {output_path}")
    print("")


if __name__ == "__main__":
    main()
