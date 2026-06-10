#!/usr/bin/env python3
import argparse
import csv
import json
import signal
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from procesar_pcb_homografia_yolo import (
    cargar_tamano_desde_config,
    calcular_homografia,
    elegir_orientacion_por_serigrafia,
    dibujar_debug_homografia,
    dibujar_template_debug,
)

from comparar_yolo_reference import (
    cargar_referencia,
    comparar_una_imagen,
    resumen_rows,
    escribir_csv,
    dibujar_overlay,
)


STOP_REQUESTED = False


def signal_handler(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("")
    print("Parada solicitada. Cerrando proceso...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_camera_source(value):
    value = str(value)
    if value.isdigit():
        return int(value)
    return value


def crear_carpetas(output_dir):
    dirs = {
        "raw": output_dir / "raw",
        "corrected": output_dir / "corrected",
        "overlay": output_dir / "overlay",
        "overlay_failures": output_dir / "overlay_failures",
        "components": output_dir / "components",
        "comparison": output_dir / "comparison",
        "json": output_dir / "json",
        "failed": output_dir / "failed",
        "yolo_runs": output_dir / "yolo_runs",
        "debug": output_dir / "debug",
        "debug_mask": output_dir / "debug" / "mask",
        "debug_homography": output_dir / "debug" / "homography",
        "debug_orientation": output_dir / "debug" / "orientation",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def get_model_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if isinstance(names, list) and class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def result_to_detections(result, corrected_frame, model, class_map):
    img_h, img_w = corrected_frame.shape[:2]
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        original_name = get_model_name(model.names, class_id)
        reference_name = class_map.get(class_id, original_name)

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x_center_px = (x1 + x2) / 2.0
        y_center_px = (y1 + y2) / 2.0
        width_px = x2 - x1
        height_px = y2 - y1

        detections.append({
            "det_idx": i,
            "class_id": class_id,
            "class_name": str(reference_name),
            "class_name_original": str(original_name),
            "confidence": confidence,
            "x1_px": x1,
            "y1_px": y1,
            "x2_px": x2,
            "y2_px": y2,
            "x_center_px": x_center_px,
            "y_center_px": y_center_px,
            "width_px": width_px,
            "height_px": height_px,
            "xc": x_center_px / img_w,
            "yc": y_center_px / img_h,
            "w": width_px / img_w,
            "h": height_px / img_h,
        })

    return detections


def escribir_componentes_csv(path, detections):
    fieldnames = [
        "det_idx",
        "class_id",
        "class_name",
        "class_name_original",
        "confidence",
        "x1_px",
        "y1_px",
        "x2_px",
        "y2_px",
        "x_center_px",
        "y_center_px",
        "width_px",
        "height_px",
        "x_center_norm",
        "y_center_norm",
        "width_norm",
        "height_norm",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for det in detections:
            writer.writerow({
                "det_idx": det["det_idx"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "class_name_original": det["class_name_original"],
                "confidence": f"{det['confidence']:.6f}",
                "x1_px": f"{det['x1_px']:.2f}",
                "y1_px": f"{det['y1_px']:.2f}",
                "x2_px": f"{det['x2_px']:.2f}",
                "y2_px": f"{det['y2_px']:.2f}",
                "x_center_px": f"{det['x_center_px']:.2f}",
                "y_center_px": f"{det['y_center_px']:.2f}",
                "width_px": f"{det['width_px']:.2f}",
                "height_px": f"{det['height_px']:.2f}",
                "x_center_norm": f"{det['xc']:.6f}",
                "y_center_norm": f"{det['yc']:.6f}",
                "width_norm": f"{det['w']:.6f}",
                "height_norm": f"{det['h']:.6f}",
            })


def append_csv(path, fieldnames, row):
    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def guardar_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def procesar_frame(
    frame,
    frame_idx,
    args,
    dirs,
    model,
    reference,
    orientation_template,
    out_width,
    out_height,
):
    total_t0 = time.perf_counter()

    if args.save_history:
        stem = f"frame_{frame_idx:06d}"
    else:
        stem = "latest"

    raw_path = dirs["raw"] / f"{stem}_raw.jpg"
    corrected_path = dirs["corrected"] / f"{stem}_corrected.jpg"
    overlay_path = dirs["overlay"] / f"{stem}_result.jpg"
    components_csv_path = dirs["components"] / f"{stem}_components.csv"
    comparison_csv_path = dirs["comparison"] / f"{stem}_comparison.csv"
    json_path = dirs["json"] / f"{stem}.json"

    mask_debug_path = dirs["debug_mask"] / f"{stem}_mask.png"
    homography_debug_path = dirs["debug_homography"] / f"{stem}_homography_debug.jpg"
    orientation_debug_path = dirs["debug_orientation"] / f"{stem}_orientation_debug.jpg"

    cv2.imwrite(str(raw_path), frame)

    hom_t0 = time.perf_counter()

    hom = calcular_homografia(
        image=frame,
        out_width=out_width,
        out_height=out_height,
        min_area_ratio=args.min_area_ratio,
        side_frac=args.side_frac,
        homography_method=args.homography_method,
        reject_mask_touch_border=not args.allow_mask_touch_border,
        border_margin_px=args.border_margin_px,
        border_min_green_pixels=args.border_min_green_pixels,
    )

    homography_time = time.perf_counter() - hom_t0

    cv2.imwrite(str(mask_debug_path), hom["mask"])

    dibujar_debug_homografia(
        image=frame,
        principal=hom["principal"],
        hull=hom["hull"],
        box=hom["box"],
        corners_raw=hom["corners_raw"],
        corners_ordered=hom["corners_ordered"],
        output_path=homography_debug_path,
        method=hom["method"],
        lados_pts=hom["lados_pts"],
        lineas=hom["lineas"],
    )

    corrected = hom["corrected"]

    orientation_time = 0.0
    orientation_info = None
    orientation_all_results = None

    if orientation_template is not None:
        ori_t0 = time.perf_counter()

        mejor_orientacion, resultados_orientacion = elegir_orientacion_por_serigrafia(
            corrected=corrected,
            orientation_template=orientation_template,
            expected_quadrant=args.orientation_expected_quadrant,
            min_score=args.orientation_min_score,
            allow_low_score=args.allow_low_orientation_score,
            min_scale=args.orientation_min_scale,
            max_scale=args.orientation_max_scale,
            scale_step=args.orientation_scale_step,
        )

        orientation_time = time.perf_counter() - ori_t0
        corrected = mejor_orientacion["image"]

        orientation_info = {
            "selected_variant": mejor_orientacion["name"],
            "score": mejor_orientacion["detection"]["score"],
            "scale": mejor_orientacion["detection"]["scale"],
            "quadrant": mejor_orientacion["quadrant"],
            "elapsed_seconds": orientation_time,
        }

        orientation_all_results = []

        for r in resultados_orientacion:
            orientation_all_results.append({
                "variant": r["name"],
                "score": r["detection"]["score"],
                "score_total": r["score_total"],
                "scale": r["detection"]["scale"],
                "quadrant": r["quadrant"],
                "elapsed_seconds": r["elapsed_seconds"],
                "scales_valid": r["detection"]["scales_valid"],
                "scales_tested": r["detection"]["scales_tested"],
            })

        text = (
            f"{orientation_info['selected_variant']} | "
            f"score={orientation_info['score']:.3f} | "
            f"scale={orientation_info['scale']:.2f} | "
            f"t={orientation_info['elapsed_seconds']:.2f}s"
        )

        dibujar_template_debug(
            image=corrected,
            detection=mejor_orientacion["detection"],
            text=text,
            output_path=orientation_debug_path,
        )

    cv2.imwrite(str(corrected_path), corrected)

    yolo_t0 = time.perf_counter()

    results = model.predict(
        source=corrected,
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
        save=False,
        project=str(dirs["yolo_runs"]),
        name=stem,
        exist_ok=True,
    )

    yolo_time = time.perf_counter() - yolo_t0

    result = results[0]

    detections = result_to_detections(
        result=result,
        corrected_frame=corrected,
        model=model,
        class_map=reference["class_map"],
    )

    escribir_componentes_csv(components_csv_path, detections)

    compare_t0 = time.perf_counter()

    comparison_rows = comparar_una_imagen(
        reference_boxes=reference["boxes"],
        detections=detections,
        match_by=args.match_by,
        min_iou=args.min_iou,
        max_center_distance=args.max_center_distance,
        max_center_distance_relaxed=args.max_center_distance_relaxed,
    )

    image_summary = resumen_rows(comparison_rows)

    escribir_csv(comparison_csv_path, comparison_rows)
    dibujar_overlay(corrected_path, comparison_rows, overlay_path, image_summary)

    compare_time = time.perf_counter() - compare_t0

    total_time = time.perf_counter() - total_t0

    data = {
        "frame_idx": frame_idx,
        "status": image_summary["status"],
        "summary": image_summary,
        "paths": {
            "raw": str(raw_path),
            "corrected": str(corrected_path),
            "overlay": str(overlay_path),
            "overlay_failures": str(dirs["overlay_failures"] / f"{stem}_failures.jpg"),
            "components_csv": str(components_csv_path),
            "comparison_csv": str(comparison_csv_path),
            "mask_debug": str(mask_debug_path),
            "homography_debug": str(homography_debug_path),
            "orientation_debug": str(orientation_debug_path) if orientation_template is not None else None,
        },
        "timing": {
            "homography_seconds": homography_time,
            "orientation_seconds": orientation_time,
            "yolo_seconds": yolo_time,
            "comparison_seconds": compare_time,
            "total_seconds": total_time,
        },
        "homography": {
            "method": hom["method"],
            "corners_ordered": hom["corners_ordered"].tolist(),
            "mask_border_check": hom["border_info"],
        },
        "orientation": orientation_info,
        "orientation_all_results": orientation_all_results,
        "detections_count": len(detections),
        "params": {
            "component_model": args.component_model,
            "conf": args.conf,
            "max_center_distance": args.max_center_distance,
            "max_center_distance_relaxed": args.max_center_distance_relaxed,
        },
    }

    guardar_json(json_path, data)

    return {
        "frame_idx": frame_idx,
        "status": image_summary["status"],
        "ok": image_summary["ok"],
        "missing": image_summary["missing"],
        "misplaced": image_summary["misplaced"],
        "extra": image_summary["extra"],
        "detections": len(detections),
        "raw_path": raw_path,
        "corrected_path": corrected_path,
        "overlay_path": overlay_path,
        "components_csv_path": components_csv_path,
        "comparison_csv_path": comparison_csv_path,
        "json_path": json_path,
        "mask_debug_path": mask_debug_path,
        "homography_debug_path": homography_debug_path,
        "orientation_debug_path": orientation_debug_path if orientation_template is not None else "",
        "homography_time": homography_time,
        "orientation_time": orientation_time,
        "yolo_time": yolo_time,
        "comparison_time": compare_time,
        "total_time": total_time,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline en tiempo real: cámara → homografía → YOLO → comparación referenceBoard → CSV + imagen."
    )

    parser.add_argument("--camera-source", default="0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--orientation-template", default=None)
    parser.add_argument("--no-orientation", action="store_true")

    parser.add_argument("--component-model", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.49)

    parser.add_argument("--homography-method", choices=["box", "lines", "hough"], default="hough")
    parser.add_argument("--min-area-ratio", type=float, default=0.02)
    parser.add_argument("--side-frac", type=float, default=0.18)

    parser.add_argument("--allow-mask-touch-border", action="store_true")
    parser.add_argument("--border-margin-px", type=int, default=3)
    parser.add_argument("--border-min-green-pixels", type=int, default=20)

    parser.add_argument("--orientation-expected-quadrant", choices=["tl", "tr", "br", "bl"], default=None)
    parser.add_argument("--orientation-min-score", type=float, default=0.45)
    parser.add_argument("--orientation-min-scale", type=float, default=0.55)
    parser.add_argument("--orientation-max-scale", type=float, default=0.80)
    parser.add_argument("--orientation-scale-step", type=float, default=0.03)
    parser.add_argument("--allow-low-orientation-score", action="store_true")

    parser.add_argument("--match-by", choices=["name", "id"], default="name")
    parser.add_argument("--min-iou", type=float, default=0.20)
    parser.add_argument("--max-center-distance", type=float, default=0.035)
    parser.add_argument("--max-center-distance-relaxed", type=float, default=0.060)

    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=5)

    parser.add_argument("--camera-width", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=0)

    parser.add_argument("--save-history", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dirs = crear_carpetas(output_dir)

    summary_csv_path = output_dir / "summary_realtime.csv"

    summary_fieldnames = [
        "frame_idx",
        "status",
        "ok",
        "missing",
        "misplaced",
        "extra",
        "detections",
        "homography_time_s",
        "orientation_time_s",
        "yolo_time_s",
        "comparison_time_s",
        "total_time_s",
        "overlay",
        "homography_debug",
        "orientation_debug",
        "components_csv",
        "comparison_csv",
        "message",
    ]

    out_width, out_height = cargar_tamano_desde_config(args.config)

    print("")
    print("Cargando referencia:")
    print(f"  {args.reference_dir}")

    reference = cargar_referencia(Path(args.reference_dir))

    print(f"  cajas referencia: {len(reference['boxes'])}")
    print(f"  clases:           {len(reference['class_map'])}")

    orientation_template = None

    if not args.no_orientation:
        if args.orientation_template is None:
            raise RuntimeError("Se ha pedido orientación, pero no se ha indicado --orientation-template")

        orientation_template_path = Path(args.orientation_template)

        if not orientation_template_path.exists():
            raise RuntimeError(f"No existe orientation-template: {orientation_template_path}")

        orientation_template = cv2.imread(str(orientation_template_path))

        if orientation_template is None:
            raise RuntimeError(f"No se pudo abrir orientation-template: {orientation_template_path}")

    print("")
    print("Cargando modelo YOLO:")
    print(f"  {args.component_model}")

    model = YOLO(args.component_model, task="detect")

    source = parse_camera_source(args.camera_source)

    print("")
    print("Abriendo cámara:")
    print(f"  source: {source}")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara/fuente: {args.camera_source}")

    if args.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)

    if args.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    for _ in range(max(0, args.warmup_frames)):
        cap.read()

    print("")
    print("Pipeline en marcha.")
    print("Pulsa Ctrl+C para parar.")
    print("")
    print(f"  output-dir:                  {output_dir}")
    print(f"  reference-dir:               {args.reference_dir}")
    print(f"  config:                      {args.config}")
    print(f"  component-model:             {args.component_model}")
    print(f"  debug-dir:                   {dirs['debug']}")
    print(f"  save-history:                {args.save_history}")
    print(f"  interval:                    {args.interval}s")
    print(f"  duration:                    {args.duration}s")
    print(f"  limit:                       {args.limit}")
    print(f"  homography-method:           {args.homography_method}")
    print(f"  conf:                        {args.conf}")
    print(f"  max-center-distance:         {args.max_center_distance}")
    print(f"  max-center-distance-relaxed: {args.max_center_distance_relaxed}")
    print("")

    start_time = time.perf_counter()
    frame_idx = 0

    try:
        while not STOP_REQUESTED:
            loop_t0 = time.perf_counter()

            if args.duration > 0 and (loop_t0 - start_time) >= args.duration:
                break

            if args.limit > 0 and frame_idx >= args.limit:
                break

            capture_t0 = time.perf_counter()
            ret, frame = cap.read()
            capture_time = time.perf_counter() - capture_t0

            if not ret or frame is None:
                print("No se pudo capturar frame.")
                time.sleep(0.2)
                continue

            frame_idx += 1

            try:
                result = procesar_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    args=args,
                    dirs=dirs,
                    model=model,
                    reference=reference,
                    orientation_template=orientation_template,
                    out_width=out_width,
                    out_height=out_height,
                )

                append_csv(summary_csv_path, summary_fieldnames, {
                    "frame_idx": frame_idx,
                    "status": result["status"],
                    "ok": result["ok"],
                    "missing": result["missing"],
                    "misplaced": result["misplaced"],
                    "extra": result["extra"],
                    "detections": result["detections"],
                    "homography_time_s": f"{result['homography_time']:.3f}",
                    "orientation_time_s": f"{result['orientation_time']:.3f}",
                    "yolo_time_s": f"{result['yolo_time']:.3f}",
                    "comparison_time_s": f"{result['comparison_time']:.3f}",
                    "total_time_s": f"{result['total_time']:.3f}",
                    "overlay": str(result["overlay_path"]),
                    "homography_debug": str(result["homography_debug_path"]),
                    "orientation_debug": str(result["orientation_debug_path"]),
                    "components_csv": str(result["components_csv_path"]),
                    "comparison_csv": str(result["comparison_csv_path"]),
                    "message": "",
                })

                print(
                    f"[{frame_idx:06d}] {result['status']} | "
                    f"OK={result['ok']} "
                    f"MISSING={result['missing']} "
                    f"MISPLACED={result['misplaced']} "
                    f"EXTRA={result['extra']} | "
                    f"cap={capture_time:.3f}s "
                    f"hom={result['homography_time']:.3f}s "
                    f"ori={result['orientation_time']:.3f}s "
                    f"yolo={result['yolo_time']:.3f}s "
                    f"cmp={result['comparison_time']:.3f}s "
                    f"total={result['total_time']:.3f}s"
                )

                print(f"    overlay:     {result['overlay_path']}")
                print(f"    homography:  {result['homography_debug_path']}")
                if result["orientation_debug_path"]:
                    print(f"    orientation: {result['orientation_debug_path']}")

            except Exception as e:
                status = "SKIP" if "Imagen descartada" in str(e) else "FAIL"

                if args.save_history:
                    error_stem = f"frame_{frame_idx:06d}"
                else:
                    error_stem = "latest"

                failed_path = dirs["failed"] / f"{error_stem}_error.txt"
                raw_failed_path = dirs["failed"] / f"{error_stem}_raw.jpg"

                cv2.imwrite(str(raw_failed_path), frame)

                with open(failed_path, "w", encoding="utf-8") as f:
                    f.write(str(e))
                    f.write("\n")

                append_csv(summary_csv_path, summary_fieldnames, {
                    "frame_idx": frame_idx,
                    "status": status,
                    "ok": "",
                    "missing": "",
                    "misplaced": "",
                    "extra": "",
                    "detections": "",
                    "homography_time_s": "",
                    "orientation_time_s": "",
                    "yolo_time_s": "",
                    "comparison_time_s": "",
                    "total_time_s": "",
                    "overlay": "",
                    "homography_debug": "",
                    "orientation_debug": "",
                    "components_csv": "",
                    "comparison_csv": "",
                    "message": str(e),
                })

                print(f"[{frame_idx:06d}] {status}: {e}")

            elapsed_loop = time.perf_counter() - loop_t0

            if args.interval > 0:
                sleep_time = args.interval - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        cap.release()

    total_elapsed = time.perf_counter() - start_time

    print("")
    print("Pipeline terminado.")
    print(f"Frames procesados: {frame_idx}")
    print(f"Tiempo total:      {total_elapsed:.3f}s")
    print(f"Resumen CSV:       {summary_csv_path}")
    print(f"Última imagen:     {dirs['overlay'] / 'latest_result.jpg'}")
    print(f"Últimos fallos:    {dirs['overlay_failures'] / 'latest_failures.jpg'}")
    print("")


if __name__ == "__main__":
    main()
