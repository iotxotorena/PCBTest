import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cargar_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def limpiar_nombre(name):
    if name is None:
        return None
    return str(name).strip()


def cargar_mapa_clases_desde_notes(notes_path):
    notes = cargar_json(notes_path)
    class_map = {}

    if not isinstance(notes, dict):
        raise RuntimeError(f"notes.json no tiene formato de diccionario: {notes_path}")

    categories = notes.get("categories")

    if isinstance(categories, list):
        for idx, item in enumerate(categories):
            if isinstance(item, dict):
                class_id = item.get(
                    "id",
                    item.get("class_id", item.get("category_id", item.get("index", idx)))
                )
                name = item.get(
                    "name",
                    item.get("label", item.get("class", item.get("title")))
                )

                if name is None:
                    continue

                try:
                    class_map[int(class_id)] = limpiar_nombre(name)
                except Exception:
                    class_map[idx] = limpiar_nombre(name)

            elif isinstance(item, str):
                class_map[idx] = limpiar_nombre(item)

    elif isinstance(categories, dict):
        for key, value in categories.items():
            try:
                class_id = int(key)
            except Exception:
                continue

            if isinstance(value, dict):
                name = value.get("name", value.get("label", value.get("class", value.get("title"))))
            else:
                name = value

            if name is not None:
                class_map[class_id] = limpiar_nombre(name)

    for key in ["names", "classes", "labels"]:
        value = notes.get(key)

        if isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, str):
                    class_map.setdefault(idx, limpiar_nombre(item))

                elif isinstance(item, dict):
                    class_id = item.get("id", item.get("class_id", item.get("category_id", idx)))
                    name = item.get("name", item.get("label", item.get("class")))

                    if name is not None:
                        try:
                            class_map.setdefault(int(class_id), limpiar_nombre(name))
                        except Exception:
                            class_map.setdefault(idx, limpiar_nombre(name))

        elif isinstance(value, dict):
            for k, v in value.items():
                try:
                    class_id = int(k)
                except Exception:
                    continue

                if isinstance(v, dict):
                    name = v.get("name", v.get("label", v.get("class")))
                else:
                    name = v

                if name is not None:
                    class_map.setdefault(class_id, limpiar_nombre(name))

    if not class_map:
        raise RuntimeError(
            f"No he podido extraer nombres de clases desde {notes_path}. "
            "Espero encontrar 'categories', 'names', 'classes' o 'labels'."
        )

    return class_map


def encontrar_unico_fichero_labels(reference_dir):
    labels_dir = reference_dir / "labels"

    if not labels_dir.exists() or not labels_dir.is_dir():
        raise RuntimeError(f"No existe la carpeta labels en la referencia: {labels_dir}")

    label_files = sorted([
        p for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ])

    if len(label_files) == 0:
        raise RuntimeError(f"No hay ningún fichero .txt en: {labels_dir}")

    if len(label_files) > 1:
        files_txt = "\n".join([f"  - {p.name}" for p in label_files])
        raise RuntimeError(
            "Debe haber UN SOLO fichero .txt de labels YOLO en referenceBoard/labels.\n"
            f"He encontrado {len(label_files)}:\n{files_txt}"
        )

    return label_files[0]


def cargar_referencia(reference_dir):
    reference_dir = Path(reference_dir)
    notes_path = reference_dir / "notes.json"

    if not notes_path.exists():
        raise RuntimeError(f"No existe notes.json en: {reference_dir}")

    class_map = cargar_mapa_clases_desde_notes(notes_path)
    label_path = encontrar_unico_fichero_labels(reference_dir)

    reference_boxes = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 5:
                raise RuntimeError(
                    f"Línea YOLO inválida en {label_path}, línea {line_number}: {line}"
                )

            try:
                class_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except Exception:
                raise RuntimeError(
                    f"No puedo interpretar línea YOLO en {label_path}, "
                    f"línea {line_number}: {line}"
                )

            class_name = class_map.get(class_id)

            if class_name is None:
                raise RuntimeError(
                    f"El class_id {class_id} aparece en {label_path}, "
                    f"pero no existe en notes.json. "
                    f"IDs disponibles: {sorted(class_map.keys())}"
                )

            reference_boxes.append({
                "class_id": class_id,
                "class_name": class_name,
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
                "source_line": line_number,
            })

    if not reference_boxes:
        raise RuntimeError(f"No se ha cargado ninguna caja de referencia desde: {label_path}")

    return {
        "notes_path": notes_path,
        "label_path": label_path,
        "class_map": class_map,
        "boxes": reference_boxes,
    }


def leer_detections_csv(detections_csv, class_map, min_conf=0.0, use_reference_names=True):
    detections_by_image = {}

    with open(detections_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = [
            "image",
            "class_id",
            "class_name",
            "x_center_norm",
            "y_center_norm",
            "width_norm",
            "height_norm",
        ]

        for r in required:
            if r not in reader.fieldnames:
                raise RuntimeError(
                    f"Falta columna '{r}' en detections.csv. "
                    f"Columnas encontradas: {reader.fieldnames}"
                )

        for row in reader:
            try:
                confidence = float(row.get("confidence", 1.0))
            except Exception:
                confidence = 1.0

            if confidence < min_conf:
                continue

            image_name = row["image"]
            class_id = int(float(row["class_id"]))

            detected_name_original = limpiar_nombre(row.get("class_name", None))

            if use_reference_names and class_id in class_map:
                class_name = class_map[class_id]
            else:
                class_name = detected_name_original

            item = {
                "image": image_name,
                "class_id": class_id,
                "class_name": class_name,
                "class_name_original": detected_name_original,
                "confidence": confidence,
                "xc": float(row["x_center_norm"]),
                "yc": float(row["y_center_norm"]),
                "w": float(row["width_norm"]),
                "h": float(row["height_norm"]),
            }

            detections_by_image.setdefault(image_name, []).append(item)

    return detections_by_image


def xywh_to_xyxy(box):
    xc = float(box["xc"])
    yc = float(box["yc"])
    w = float(box["w"])
    h = float(box["h"])

    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0

    return x1, y1, x2, y2


def box_metrics_xywh(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    iou = 0.0
    if union > 0:
        iou = inter / union

    coverage_a = 0.0
    if area_a > 0:
        coverage_a = inter / area_a

    coverage_b = 0.0
    if area_b > 0:
        coverage_b = inter / area_b

    return {
        "iou": iou,
        "intersection": inter,
        "area_a": area_a,
        "area_b": area_b,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
    }


def center_distance(a, b):
    dx = float(a["xc"]) - float(b["xc"])
    dy = float(a["yc"]) - float(b["yc"])
    return math.sqrt(dx * dx + dy * dy)


def safe_ratio(det_value, ref_value):
    ref_value = float(ref_value)
    det_value = float(det_value)

    if ref_value <= 1e-9:
        return 0.0

    return det_value / ref_value


def obtener_clave_match(item, match_by="name"):
    if match_by == "id":
        return ("id", str(item.get("class_id")))

    class_name = item.get("class_name")

    if class_name is None or str(class_name).strip() == "":
        return ("name", "")

    return ("name", str(class_name).strip())


def evaluar_match_geometrico(
    ref,
    det,
    min_iou,
    max_center_distance,
    min_size_ratio,
    max_size_ratio,
    min_ref_coverage=0.70,
    min_det_coverage=0.50,
    max_center_distance_relaxed=0.060,
    min_ref_coverage_relaxed=0.35,
    min_det_coverage_relaxed=0.25,
):
    metrics = box_metrics_xywh(ref, det)

    iou = metrics["iou"]
    ref_coverage = metrics["coverage_a"]
    det_coverage = metrics["coverage_b"]

    dist = center_distance(ref, det)

    width_ratio = safe_ratio(det["w"], ref["w"])
    height_ratio = safe_ratio(det["h"], ref["h"])

    ref_area = float(ref["w"]) * float(ref["h"])
    det_area = float(det["w"]) * float(det["h"])
    area_ratio = safe_ratio(det_area, ref_area)

    ok_iou = iou >= min_iou
    ok_center = dist <= max_center_distance
    ok_center_relaxed = dist <= max_center_distance_relaxed

    ok_ref_coverage = ref_coverage >= min_ref_coverage
    ok_det_coverage = det_coverage >= min_det_coverage

    ok_ref_coverage_relaxed = ref_coverage >= min_ref_coverage_relaxed
    ok_det_coverage_relaxed = det_coverage >= min_det_coverage_relaxed

    ok_width = min_size_ratio <= width_ratio <= max_size_ratio
    ok_height = min_size_ratio <= height_ratio <= max_size_ratio

    size_warning = not (ok_width and ok_height)

    strict_overlap = ok_iou or ok_ref_coverage or ok_det_coverage
    relaxed_overlap = ok_ref_coverage_relaxed and ok_det_coverage_relaxed

    ok_strict = ok_center and strict_overlap
    ok_relaxed = ok_center_relaxed and relaxed_overlap

    ok = ok_strict or ok_relaxed

    fail_reasons = []

    if not ok_center and not ok_center_relaxed:
        fail_reasons.append("center")

    if not strict_overlap and not relaxed_overlap:
        fail_reasons.append("overlap")

    if size_warning:
        fail_reasons.append("size_warning")

    return {
        "ok": ok,
        "iou": iou,
        "center_distance": dist,
        "center_distance_relaxed": max_center_distance_relaxed,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "area_ratio": area_ratio,
        "ref_coverage": ref_coverage,
        "det_coverage": det_coverage,
        "ok_iou": ok_iou,
        "ok_center": ok_center,
        "ok_center_relaxed": ok_center_relaxed,
        "ok_width": ok_width,
        "ok_height": ok_height,
        "ok_ref_coverage": ok_ref_coverage,
        "ok_det_coverage": ok_det_coverage,
        "ok_ref_coverage_relaxed": ok_ref_coverage_relaxed,
        "ok_det_coverage_relaxed": ok_det_coverage_relaxed,
        "size_warning": size_warning,
        "ok_strict": ok_strict,
        "ok_relaxed": ok_relaxed,
        "fail_reasons": ",".join(fail_reasons),
    }


def pasa_puerta_candidato(
    geom,
    max_candidate_center_distance,
    min_candidate_iou,
    min_candidate_ref_coverage,
    min_candidate_det_coverage,
):
    """
    Esta puerta evita el desaguisado:
    no permite emparejar componentes de la misma clase si están muy lejos
    y no tienen solape real.
    """
    center_gate = geom["center_distance"] <= max_candidate_center_distance
    iou_gate = geom["iou"] >= min_candidate_iou
    ref_cov_gate = geom["ref_coverage"] >= min_candidate_ref_coverage
    det_cov_gate = geom["det_coverage"] >= min_candidate_det_coverage

    candidate_valid = center_gate or iou_gate or ref_cov_gate or det_cov_gate

    reasons = []

    if not center_gate:
        reasons.append("candidate_center_far")

    if not iou_gate and not ref_cov_gate and not det_cov_gate:
        reasons.append("candidate_no_overlap")

    return {
        "candidate_valid": candidate_valid,
        "candidate_center_gate": center_gate,
        "candidate_iou_gate": iou_gate,
        "candidate_ref_cov_gate": ref_cov_gate,
        "candidate_det_cov_gate": det_cov_gate,
        "candidate_gate_reasons": ",".join(reasons),
    }


def fila_missing(ref_idx, ref, reason):
    return {
        "status": "MISSING",
        "ref_idx": ref_idx,
        "det_idx": "",
        "class_id_ref": ref.get("class_id"),
        "class_name_ref": ref.get("class_name"),
        "class_id_det": "",
        "class_name_det": "",
        "class_name_det_original": "",
        "confidence": "",
        "iou": "",
        "center_distance": "",
        "center_distance_relaxed": "",
        "width_ratio": "",
        "height_ratio": "",
        "area_ratio": "",
        "ref_coverage": "",
        "det_coverage": "",
        "ok_iou": "",
        "ok_center": "",
        "ok_center_relaxed": "",
        "ok_width": "",
        "ok_height": "",
        "ok_ref_coverage": "",
        "ok_det_coverage": "",
        "ok_ref_coverage_relaxed": "",
        "ok_det_coverage_relaxed": "",
        "size_warning": "",
        "ok_strict": "",
        "ok_relaxed": "",
        "candidate_valid": "",
        "candidate_center_gate": "",
        "candidate_iou_gate": "",
        "candidate_ref_cov_gate": "",
        "candidate_det_cov_gate": "",
        "candidate_gate_reasons": "",
        "fail_reasons": reason,
        "ref_xc": ref["xc"],
        "ref_yc": ref["yc"],
        "ref_w": ref["w"],
        "ref_h": ref["h"],
        "det_xc": "",
        "det_yc": "",
        "det_w": "",
        "det_h": "",
    }


def fila_extra(det_idx, det):
    return {
        "status": "EXTRA",
        "ref_idx": "",
        "det_idx": det_idx,
        "class_id_ref": "",
        "class_name_ref": "",
        "class_id_det": det.get("class_id"),
        "class_name_det": det.get("class_name"),
        "class_name_det_original": det.get("class_name_original", ""),
        "confidence": det.get("confidence", ""),
        "iou": "",
        "center_distance": "",
        "center_distance_relaxed": "",
        "width_ratio": "",
        "height_ratio": "",
        "area_ratio": "",
        "ref_coverage": "",
        "det_coverage": "",
        "ok_iou": "",
        "ok_center": "",
        "ok_center_relaxed": "",
        "ok_width": "",
        "ok_height": "",
        "ok_ref_coverage": "",
        "ok_det_coverage": "",
        "ok_ref_coverage_relaxed": "",
        "ok_det_coverage_relaxed": "",
        "size_warning": "",
        "ok_strict": "",
        "ok_relaxed": "",
        "candidate_valid": "",
        "candidate_center_gate": "",
        "candidate_iou_gate": "",
        "candidate_ref_cov_gate": "",
        "candidate_det_cov_gate": "",
        "candidate_gate_reasons": "",
        "fail_reasons": "extra_detection",
        "ref_xc": "",
        "ref_yc": "",
        "ref_w": "",
        "ref_h": "",
        "det_xc": det["xc"],
        "det_yc": det["yc"],
        "det_w": det["w"],
        "det_h": det["h"],
    }


def fila_match(ref_idx, ref, det_idx, det, geom, gate):
    status = "OK" if geom["ok"] else "MISPLACED"

    return {
        "status": status,
        "ref_idx": ref_idx,
        "det_idx": det_idx,
        "class_id_ref": ref.get("class_id"),
        "class_name_ref": ref.get("class_name"),
        "class_id_det": det.get("class_id"),
        "class_name_det": det.get("class_name"),
        "class_name_det_original": det.get("class_name_original", ""),
        "confidence": det.get("confidence", ""),
        "iou": geom["iou"],
        "center_distance": geom["center_distance"],
        "center_distance_relaxed": geom["center_distance_relaxed"],
        "width_ratio": geom["width_ratio"],
        "height_ratio": geom["height_ratio"],
        "area_ratio": geom["area_ratio"],
        "ref_coverage": geom["ref_coverage"],
        "det_coverage": geom["det_coverage"],
        "ok_iou": geom["ok_iou"],
        "ok_center": geom["ok_center"],
        "ok_center_relaxed": geom["ok_center_relaxed"],
        "ok_width": geom["ok_width"],
        "ok_height": geom["ok_height"],
        "ok_ref_coverage": geom["ok_ref_coverage"],
        "ok_det_coverage": geom["ok_det_coverage"],
        "ok_ref_coverage_relaxed": geom["ok_ref_coverage_relaxed"],
        "ok_det_coverage_relaxed": geom["ok_det_coverage_relaxed"],
        "size_warning": geom["size_warning"],
        "ok_strict": geom["ok_strict"],
        "ok_relaxed": geom["ok_relaxed"],
        "candidate_valid": gate["candidate_valid"],
        "candidate_center_gate": gate["candidate_center_gate"],
        "candidate_iou_gate": gate["candidate_iou_gate"],
        "candidate_ref_cov_gate": gate["candidate_ref_cov_gate"],
        "candidate_det_cov_gate": gate["candidate_det_cov_gate"],
        "candidate_gate_reasons": gate["candidate_gate_reasons"],
        "fail_reasons": geom["fail_reasons"],
        "ref_xc": ref["xc"],
        "ref_yc": ref["yc"],
        "ref_w": ref["w"],
        "ref_h": ref["h"],
        "det_xc": det["xc"],
        "det_yc": det["yc"],
        "det_w": det["w"],
        "det_h": det["h"],
    }


def comparar_una_imagen(
    reference_boxes,
    detections,
    match_by="name",
    min_iou=0.20,
    max_center_distance=0.035,
    min_size_ratio=0.70,
    max_size_ratio=1.30,
    min_ref_coverage=0.70,
    min_det_coverage=0.50,
    max_center_distance_relaxed=0.060,
    min_ref_coverage_relaxed=0.35,
    min_det_coverage_relaxed=0.25,
    max_candidate_center_distance=0.10,
    min_candidate_iou=0.02,
    min_candidate_ref_coverage=0.10,
    min_candidate_det_coverage=0.10,
):
    """
    Comparación robusta one-to-one.

    Cambio importante:
    - Ya no empareja una referencia con cualquier detección de la misma clase.
    - Primero exige una puerta local:
        centro razonablemente cerca O solape/cobertura mínima.
    - Si no pasa esa puerta, la referencia queda MISSING.
    """

    candidates = []
    same_class_seen_by_ref = {i: False for i in range(len(reference_boxes))}

    for ref_idx, ref in enumerate(reference_boxes):
        ref_key = obtener_clave_match(ref, match_by=match_by)

        for det_idx, det in enumerate(detections):
            det_key = obtener_clave_match(det, match_by=match_by)

            if det_key != ref_key:
                continue

            same_class_seen_by_ref[ref_idx] = True

            geom = evaluar_match_geometrico(
                ref=ref,
                det=det,
                min_iou=min_iou,
                max_center_distance=max_center_distance,
                min_size_ratio=min_size_ratio,
                max_size_ratio=max_size_ratio,
                min_ref_coverage=min_ref_coverage,
                min_det_coverage=min_det_coverage,
                max_center_distance_relaxed=max_center_distance_relaxed,
                min_ref_coverage_relaxed=min_ref_coverage_relaxed,
                min_det_coverage_relaxed=min_det_coverage_relaxed,
            )

            gate = pasa_puerta_candidato(
                geom=geom,
                max_candidate_center_distance=max_candidate_center_distance,
                min_candidate_iou=min_candidate_iou,
                min_candidate_ref_coverage=min_candidate_ref_coverage,
                min_candidate_det_coverage=min_candidate_det_coverage,
            )

            if not gate["candidate_valid"]:
                continue

            candidates.append({
                "ref_idx": ref_idx,
                "det_idx": det_idx,
                "ref": ref,
                "det": det,
                "geom": geom,
                "gate": gate,
            })

    candidates = sorted(
        candidates,
        key=lambda c: (
            not c["geom"]["ok"],
            not c["geom"]["ok_strict"],
            not c["geom"]["ok_relaxed"],
            -c["geom"]["ref_coverage"],
            -c["geom"]["det_coverage"],
            -c["geom"]["iou"],
            c["geom"]["center_distance"],
        )
    )

    matched_refs = {}
    used_dets = set()

    for cand in candidates:
        ref_idx = cand["ref_idx"]
        det_idx = cand["det_idx"]

        if ref_idx in matched_refs:
            continue

        if det_idx in used_dets:
            continue

        matched_refs[ref_idx] = cand
        used_dets.add(det_idx)

    rows = []

    for ref_idx, ref in enumerate(reference_boxes):
        cand = matched_refs.get(ref_idx)

        if cand is None:
            if same_class_seen_by_ref.get(ref_idx, False):
                reason = "no_valid_candidate_same_class"
            else:
                reason = "no_detection_same_class"

            rows.append(fila_missing(ref_idx, ref, reason))
        else:
            rows.append(
                fila_match(
                    ref_idx=ref_idx,
                    ref=cand["ref"],
                    det_idx=cand["det_idx"],
                    det=cand["det"],
                    geom=cand["geom"],
                    gate=cand["gate"],
                )
            )

    for det_idx, det in enumerate(detections):
        if det_idx not in used_dets:
            rows.append(fila_extra(det_idx, det))

    return rows


def norm_box_to_px(row, image_w, image_h, prefix):
    xc = float(row[f"{prefix}xc"])
    yc = float(row[f"{prefix}yc"])
    w = float(row[f"{prefix}w"])
    h = float(row[f"{prefix}h"])

    x1 = int((xc - w / 2.0) * image_w)
    y1 = int((yc - h / 2.0) * image_h)
    x2 = int((xc + w / 2.0) * image_w)
    y2 = int((yc + h / 2.0) * image_h)

    return x1, y1, x2, y2


def center_to_px(xc, yc, image_w, image_h):
    return int(float(xc) * image_w), int(float(yc) * image_h)


def draw_label(img, text, x, y, color, scale=0.5, thickness=1, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x = max(0, int(x))
    y = max(th + baseline + 2, int(y))

    if bg:
        cv2.rectangle(
            img,
            (x, y - th - baseline - 4),
            (x + tw + 6, y + 2),
            (0, 0, 0),
            -1,
        )

    cv2.putText(
        img,
        text,
        (x + 3, y - 3),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_legend_on_bottom_strip(img, strip_y, mode="all"):
    if mode == "failures":
        items = [
            ("MISSING", (0, 0, 255)),
            ("MISPLACED", (255, 0, 255)),
        ]
    else:
        items = [
            ("OK", (0, 255, 0)),
            ("MISSING", (0, 0, 255)),
            ("MISPLACED", (255, 0, 255)),
            ("EXTRA", (0, 165, 255)),
        ]

    x = 10
    y = strip_y + 72

    for text, color in items:
        cv2.rectangle(img, (x, y - 12), (x + 18, y + 6), color, -1)
        cv2.putText(
            img,
            text,
            (x + 25, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        x += 150


def crear_canvas_con_cinta_inferior(image, strip_h=95):
    h, w = image.shape[:2]
    strip_y = h

    canvas = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    canvas[0:h, 0:w] = image
    cv2.rectangle(canvas, (0, strip_y), (w, h + strip_h), (20, 20, 20), -1)

    return canvas, strip_y, w, h


def calcular_ruta_overlay_fallos(output_path):
    output_path = Path(output_path)

    if output_path.parent.name == "overlay":
        failures_dir = output_path.parent.parent / "overlay_failures"
    else:
        failures_dir = output_path.parent / "overlay_failures"

    failures_dir.mkdir(parents=True, exist_ok=True)

    stem = output_path.stem

    if stem.endswith("_result"):
        new_stem = stem[:-len("_result")] + "_failures"
    elif stem.endswith("_comparison"):
        new_stem = stem[:-len("_comparison")] + "_failures"
    else:
        new_stem = stem + "_failures"

    return failures_dir / f"{new_stem}{output_path.suffix}"


def dibujar_overlay_fallos(image_path, rows, output_path, image_summary):
    image = cv2.imread(str(image_path))

    if image is None:
        return False

    canvas, strip_y, w, h = crear_canvas_con_cinta_inferior(image, strip_h=95)

    failures = [r for r in rows if r["status"] in ["MISSING", "MISPLACED"]]

    status_color = (0, 220, 0) if len(failures) == 0 else (0, 0, 255)

    cv2.putText(
        canvas,
        f"{image_path.name} | OVERLAY FALLOS: {'OK' if len(failures) == 0 else 'FAIL'}",
        (10, strip_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            f"MISSING={image_summary['missing']}   "
            f"MISPLACED={image_summary['misplaced']}   "
            f"No se muestran OK ni EXTRA"
        ),
        (10, strip_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    draw_legend_on_bottom_strip(canvas, strip_y, mode="failures")

    if len(failures) == 0:
        cv2.putText(
            canvas,
            "SIN MISSING NI MISPLACED",
            (10, max(35, h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for row in failures:
        status = row["status"]

        if status == "MISSING" and row["ref_xc"] != "":
            rx1, ry1, rx2, ry2 = norm_box_to_px(row, w, h, prefix="ref_")
            rcx, rcy = center_to_px(row["ref_xc"], row["ref_yc"], w, h)

            ref_color = (0, 0, 255)

            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), ref_color, 3)
            cv2.circle(canvas, (rcx, rcy), 5, ref_color, -1)

            draw_label(
                canvas,
                f"MISSING {row['class_name_ref']}",
                rx1,
                max(ry1 - 8, 18),
                ref_color,
                scale=0.55,
                thickness=1,
                bg=True,
            )

            reason = row.get("fail_reasons", "")
            if reason:
                draw_label(
                    canvas,
                    reason,
                    rx1,
                    min(ry2 + 22, h - 4),
                    ref_color,
                    scale=0.45,
                    thickness=1,
                    bg=True,
                )

        if status == "MISPLACED":
            if row["ref_xc"] != "":
                rx1, ry1, rx2, ry2 = norm_box_to_px(row, w, h, prefix="ref_")
                rcx, rcy = center_to_px(row["ref_xc"], row["ref_yc"], w, h)

                ref_color = (0, 0, 255)

                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), ref_color, 2)
                cv2.circle(canvas, (rcx, rcy), 5, ref_color, -1)

                draw_label(
                    canvas,
                    f"REF {row['class_name_ref']}",
                    rx1,
                    max(ry1 - 8, 18),
                    ref_color,
                    scale=0.45,
                    thickness=1,
                    bg=True,
                )

            if row["det_xc"] != "":
                dx1, dy1, dx2, dy2 = norm_box_to_px(row, w, h, prefix="det_")
                dcx, dcy = center_to_px(row["det_xc"], row["det_yc"], w, h)

                det_color = (255, 0, 255)

                cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), det_color, 3)
                cv2.circle(canvas, (dcx, dcy), 5, det_color, -1)

                det_name = row["class_name_det"]
                conf = row["confidence"]

                if conf != "":
                    label = f"MISPLACED {det_name} {float(conf):.2f}"
                else:
                    label = f"MISPLACED {det_name}"

                draw_label(
                    canvas,
                    label,
                    dx1,
                    min(dy2 + 22, h - 4),
                    det_color,
                    scale=0.50,
                    thickness=1,
                    bg=True,
                )

            if row["ref_xc"] != "" and row["det_xc"] != "":
                rcx, rcy = center_to_px(row["ref_xc"], row["ref_yc"], w, h)
                dcx, dcy = center_to_px(row["det_xc"], row["det_yc"], w, h)

                cv2.line(canvas, (rcx, rcy), (dcx, dcy), (0, 255, 255), 2)

                reason = row.get("fail_reasons", "")
                text = (
                    f"IoU={float(row['iou']):.2f} "
                    f"d={float(row['center_distance']):.3f} "
                    f"refCov={float(row['ref_coverage']):.2f} "
                    f"detCov={float(row['det_coverage']):.2f}"
                )

                if reason:
                    text += f" [{reason}]"

                mx = int((rcx + dcx) / 2)
                my = int((rcy + dcy) / 2)

                draw_label(
                    canvas,
                    text,
                    mx,
                    my,
                    (0, 255, 255),
                    scale=0.42,
                    thickness=1,
                    bg=True,
                )

    cv2.imwrite(str(output_path), canvas)
    return True


def dibujar_overlay(image_path, rows, output_path, image_summary):
    image = cv2.imread(str(image_path))

    if image is None:
        return False

    canvas, strip_y, w, h = crear_canvas_con_cinta_inferior(image, strip_h=95)

    status_color = (0, 220, 0) if image_summary["status"] == "OK" else (0, 0, 255)

    cv2.putText(
        canvas,
        f"{image_path.name} | RESULTADO: {image_summary['status']}",
        (10, strip_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            f"OK={image_summary['ok']}   "
            f"MISSING={image_summary['missing']}   "
            f"MISPLACED={image_summary['misplaced']}   "
            f"EXTRA={image_summary['extra']}"
        ),
        (10, strip_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    draw_legend_on_bottom_strip(canvas, strip_y, mode="all")

    for row in rows:
        status = row["status"]

        if status in ["OK", "MISPLACED", "MISSING"] and row["ref_xc"] != "":
            rx1, ry1, rx2, ry2 = norm_box_to_px(row, w, h, prefix="ref_")
            rcx, rcy = center_to_px(row["ref_xc"], row["ref_yc"], w, h)

            ref_color = (0, 180, 0) if status == "OK" else (0, 0, 255)

            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), ref_color, 2)
            cv2.circle(canvas, (rcx, rcy), 4, ref_color, -1)

            draw_label(
                canvas,
                f"REF {row['class_name_ref']}",
                rx1,
                max(ry1 - 6, 18),
                ref_color,
                scale=0.45,
                thickness=1,
                bg=True,
            )

        if status in ["OK", "MISPLACED", "EXTRA"] and row["det_xc"] != "":
            dx1, dy1, dx2, dy2 = norm_box_to_px(row, w, h, prefix="det_")
            dcx, dcy = center_to_px(row["det_xc"], row["det_yc"], w, h)

            if status == "OK":
                det_color = (0, 255, 0)
            elif status == "MISPLACED":
                det_color = (255, 0, 255)
            else:
                det_color = (0, 165, 255)

            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), det_color, 2)
            cv2.circle(canvas, (dcx, dcy), 4, det_color, -1)

            det_name = row["class_name_det"]
            original_name = row.get("class_name_det_original", "")
            conf = row["confidence"]

            if original_name and original_name != det_name:
                shown_name = f"{det_name}({original_name})"
            else:
                shown_name = det_name

            if conf != "":
                label = f"DET {shown_name} {float(conf):.2f}"
            else:
                label = f"DET {shown_name}"

            draw_label(
                canvas,
                label,
                dx1,
                min(dy2 + 18, h - 4),
                det_color,
                scale=0.45,
                thickness=1,
                bg=True,
            )

        if status == "MISPLACED" and row["ref_xc"] != "" and row["det_xc"] != "":
            rcx, rcy = center_to_px(row["ref_xc"], row["ref_yc"], w, h)
            dcx, dcy = center_to_px(row["det_xc"], row["det_yc"], w, h)

            cv2.line(canvas, (rcx, rcy), (dcx, dcy), (0, 255, 255), 2)

            reason = row.get("fail_reasons", "")
            text = (
                f"IoU={float(row['iou']):.2f} "
                f"d={float(row['center_distance']):.3f} "
                f"refCov={float(row['ref_coverage']):.2f} "
                f"detCov={float(row['det_coverage']):.2f}"
            )

            if reason:
                text += f" [{reason}]"

            mx = int((rcx + dcx) / 2)
            my = int((rcy + dcy) / 2)

            draw_label(
                canvas,
                text,
                mx,
                my,
                (0, 255, 255),
                scale=0.42,
                thickness=1,
                bg=True,
            )

        if status == "MISSING":
            rx1, ry1, rx2, ry2 = norm_box_to_px(row, w, h, prefix="ref_")
            draw_label(
                canvas,
                f"MISSING {row['class_name_ref']}",
                rx1,
                min(ry2 + 18, h - 4),
                (0, 0, 255),
                scale=0.45,
                thickness=1,
                bg=True,
            )

        if status == "EXTRA":
            dx1, dy1, dx2, dy2 = norm_box_to_px(row, w, h, prefix="det_")
            draw_label(
                canvas,
                f"EXTRA {row['class_name_det']}",
                dx1,
                max(dy1 - 6, 18),
                (0, 165, 255),
                scale=0.45,
                thickness=1,
                bg=True,
            )

    cv2.imwrite(str(output_path), canvas)

    failures_output_path = calcular_ruta_overlay_fallos(output_path)
    dibujar_overlay_fallos(image_path, rows, failures_output_path, image_summary)

    return True


def escribir_csv(path, rows):
    fieldnames = [
        "status",
        "ref_idx",
        "det_idx",
        "class_id_ref",
        "class_name_ref",
        "class_id_det",
        "class_name_det",
        "class_name_det_original",
        "confidence",
        "iou",
        "center_distance",
        "center_distance_relaxed",
        "width_ratio",
        "height_ratio",
        "area_ratio",
        "ref_coverage",
        "det_coverage",
        "ok_iou",
        "ok_center",
        "ok_center_relaxed",
        "ok_width",
        "ok_height",
        "ok_ref_coverage",
        "ok_det_coverage",
        "ok_ref_coverage_relaxed",
        "ok_det_coverage_relaxed",
        "size_warning",
        "ok_strict",
        "ok_relaxed",
        "candidate_valid",
        "candidate_center_gate",
        "candidate_iou_gate",
        "candidate_ref_cov_gate",
        "candidate_det_cov_gate",
        "candidate_gate_reasons",
        "fail_reasons",
        "ref_xc",
        "ref_yc",
        "ref_w",
        "ref_h",
        "det_xc",
        "det_yc",
        "det_w",
        "det_h",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            formatted = {}

            for key in fieldnames:
                value = row.get(key, "")

                if isinstance(value, float):
                    formatted[key] = f"{value:.6f}"
                else:
                    formatted[key] = value

            writer.writerow(formatted)


def resumen_rows(rows):
    counts = {
        "OK": 0,
        "MISSING": 0,
        "MISPLACED": 0,
        "EXTRA": 0,
    }

    for row in rows:
        status = row["status"]
        if status in counts:
            counts[status] += 1

    total_issues = counts["MISSING"] + counts["MISPLACED"] + counts["EXTRA"]

    return {
        "ok": counts["OK"],
        "missing": counts["MISSING"],
        "misplaced": counts["MISPLACED"],
        "extra": counts["EXTRA"],
        "status": "OK" if total_issues == 0 else "FAIL",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara detecciones YOLO corregidas contra referenceBoard/notes.json + referenceBoard/labels/*.txt"
    )

    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--reference-dir", default="/workspace/referenceBoard")
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--match-by", choices=["name", "id"], default="name")

    parser.add_argument("--min-iou", type=float, default=0.20)
    parser.add_argument("--max-center-distance", type=float, default=0.035)

    parser.add_argument(
        "--max-center-distance-relaxed",
        type=float,
        default=0.060,
        help="Distancia de centro relajada para casos con cobertura parcial suficiente."
    )

    parser.add_argument(
        "--min-size-ratio",
        type=float,
        default=0.70,
        help="Ratio mínimo tamaño. Solo genera size_warning."
    )

    parser.add_argument(
        "--max-size-ratio",
        type=float,
        default=1.30,
        help="Ratio máximo tamaño. Solo genera size_warning."
    )

    parser.add_argument(
        "--min-ref-coverage",
        type=float,
        default=0.70,
        help="Cobertura fuerte mínima de la referencia."
    )

    parser.add_argument(
        "--min-det-coverage",
        type=float,
        default=0.50,
        help="Cobertura fuerte mínima de la detección."
    )

    parser.add_argument(
        "--min-ref-coverage-relaxed",
        type=float,
        default=0.35,
        help="Cobertura relajada mínima de la referencia."
    )

    parser.add_argument(
        "--min-det-coverage-relaxed",
        type=float,
        default=0.25,
        help="Cobertura relajada mínima de la detección."
    )

    parser.add_argument(
        "--max-candidate-center-distance",
        type=float,
        default=0.10,
        help="Puerta inicial: distancia máxima para considerar candidato local."
    )

    parser.add_argument(
        "--min-candidate-iou",
        type=float,
        default=0.02,
        help="Puerta inicial: IoU mínimo para considerar candidato aunque el centro esté lejos."
    )

    parser.add_argument(
        "--min-candidate-ref-coverage",
        type=float,
        default=0.10,
        help="Puerta inicial: cobertura mínima de referencia para considerar candidato."
    )

    parser.add_argument(
        "--min-candidate-det-coverage",
        type=float,
        default=0.10,
        help="Puerta inicial: cobertura mínima de detección para considerar candidato."
    )

    parser.add_argument("--min-conf", type=float, default=0.0)

    parser.add_argument(
        "--no-reference-names-for-detections",
        action="store_true",
        help="No renombra las detecciones usando notes.json. No recomendado."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    processed_dir = Path(args.processed_dir)
    reference_dir = Path(args.reference_dir)

    corrected_dir = processed_dir / "corrected"
    detections_csv = processed_dir / "detections.csv"

    if args.output_dir is None:
        output_dir = processed_dir / "comparison"
    else:
        output_dir = Path(args.output_dir)

    if not processed_dir.exists():
        raise RuntimeError(f"No existe processed-dir: {processed_dir}")

    if not corrected_dir.exists():
        raise RuntimeError(f"No existe corrected/: {corrected_dir}")

    if not detections_csv.exists():
        raise RuntimeError(f"No existe detections.csv: {detections_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = output_dir / "overlay"
    overlay_failures_dir = output_dir / "overlay_failures"
    per_image_dir = output_dir / "per_image"

    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_failures_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir.mkdir(parents=True, exist_ok=True)

    reference = cargar_referencia(reference_dir)
    reference_boxes = reference["boxes"]
    class_map = reference["class_map"]

    detections_by_image = leer_detections_csv(
        detections_csv,
        class_map=class_map,
        min_conf=args.min_conf,
        use_reference_names=not args.no_reference_names_for_detections,
    )

    corrected_images = sorted([
        p for p in corrected_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not corrected_images:
        raise RuntimeError(f"No hay imágenes corregidas en: {corrected_dir}")

    summary_rows = []
    global_counts = {
        "images": 0,
        "ok_images": 0,
        "fail_images": 0,
        "ok": 0,
        "missing": 0,
        "misplaced": 0,
        "extra": 0,
    }

    print("")
    print("Referencia cargada:")
    print(f"  notes.json:                    {reference['notes_path']}")
    print(f"  labels referencia:             {reference['label_path']}")
    print(f"  cajas referencia:              {len(reference_boxes)}")
    print(f"  clases notes.json:             {len(reference['class_map'])}")
    print(f"  match-by:                      {args.match_by}")
    print(f"  min-iou:                       {args.min_iou}")
    print(f"  max-center-distance:           {args.max_center_distance}")
    print(f"  max-center-distance-relaxed:   {args.max_center_distance_relaxed}")
    print(f"  size-ratio warning:            {args.min_size_ratio} - {args.max_size_ratio}")
    print(f"  min-ref-coverage:              {args.min_ref_coverage}")
    print(f"  min-det-coverage:              {args.min_det_coverage}")
    print(f"  min-ref-coverage-relaxed:      {args.min_ref_coverage_relaxed}")
    print(f"  min-det-coverage-relaxed:      {args.min_det_coverage_relaxed}")
    print(f"  max-candidate-center-distance: {args.max_candidate_center_distance}")
    print(f"  min-candidate-iou:             {args.min_candidate_iou}")
    print(f"  min-candidate-ref-coverage:    {args.min_candidate_ref_coverage}")
    print(f"  min-candidate-det-coverage:    {args.min_candidate_det_coverage}")
    print(f"  renombrar detecciones por ref: {not args.no_reference_names_for_detections}")
    print("")

    for image_path in corrected_images:
        image_name = image_path.name
        detections = detections_by_image.get(image_name, [])

        rows = comparar_una_imagen(
            reference_boxes=reference_boxes,
            detections=detections,
            match_by=args.match_by,
            min_iou=args.min_iou,
            max_center_distance=args.max_center_distance,
            min_size_ratio=args.min_size_ratio,
            max_size_ratio=args.max_size_ratio,
            min_ref_coverage=args.min_ref_coverage,
            min_det_coverage=args.min_det_coverage,
            max_center_distance_relaxed=args.max_center_distance_relaxed,
            min_ref_coverage_relaxed=args.min_ref_coverage_relaxed,
            min_det_coverage_relaxed=args.min_det_coverage_relaxed,
            max_candidate_center_distance=args.max_candidate_center_distance,
            min_candidate_iou=args.min_candidate_iou,
            min_candidate_ref_coverage=args.min_candidate_ref_coverage,
            min_candidate_det_coverage=args.min_candidate_det_coverage,
        )

        image_summary = resumen_rows(rows)

        per_image_csv = per_image_dir / f"{image_path.stem}_comparison.csv"
        overlay_path = overlay_dir / f"{image_path.stem}_comparison.jpg"
        overlay_failures_path = calcular_ruta_overlay_fallos(overlay_path)

        escribir_csv(per_image_csv, rows)
        dibujar_overlay(image_path, rows, overlay_path, image_summary)

        summary_rows.append({
            "image": image_name,
            "status": image_summary["status"],
            "ok": image_summary["ok"],
            "missing": image_summary["missing"],
            "misplaced": image_summary["misplaced"],
            "extra": image_summary["extra"],
            "detections": len(detections),
            "reference_boxes": len(reference_boxes),
            "overlay": str(overlay_path),
            "overlay_failures": str(overlay_failures_path),
            "csv": str(per_image_csv),
        })

        global_counts["images"] += 1
        global_counts["ok"] += image_summary["ok"]
        global_counts["missing"] += image_summary["missing"]
        global_counts["misplaced"] += image_summary["misplaced"]
        global_counts["extra"] += image_summary["extra"]

        if image_summary["status"] == "OK":
            global_counts["ok_images"] += 1
        else:
            global_counts["fail_images"] += 1

        print(
            f"{image_name}: {image_summary['status']} | "
            f"OK={image_summary['ok']} "
            f"MISSING={image_summary['missing']} "
            f"MISPLACED={image_summary['misplaced']} "
            f"EXTRA={image_summary['extra']}"
        )

    summary_csv = output_dir / "summary.csv"

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "status",
            "ok",
            "missing",
            "misplaced",
            "extra",
            "detections",
            "reference_boxes",
            "overlay",
            "overlay_failures",
            "csv",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    global_summary_path = output_dir / "global_summary.json"

    with open(global_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "reference": {
                "notes_path": str(reference["notes_path"]),
                "label_path": str(reference["label_path"]),
                "num_reference_boxes": len(reference_boxes),
                "class_map": reference["class_map"],
            },
            "params": {
                "match_by": args.match_by,
                "min_iou": args.min_iou,
                "max_center_distance": args.max_center_distance,
                "max_center_distance_relaxed": args.max_center_distance_relaxed,
                "min_size_ratio": args.min_size_ratio,
                "max_size_ratio": args.max_size_ratio,
                "min_ref_coverage": args.min_ref_coverage,
                "min_det_coverage": args.min_det_coverage,
                "min_ref_coverage_relaxed": args.min_ref_coverage_relaxed,
                "min_det_coverage_relaxed": args.min_det_coverage_relaxed,
                "max_candidate_center_distance": args.max_candidate_center_distance,
                "min_candidate_iou": args.min_candidate_iou,
                "min_candidate_ref_coverage": args.min_candidate_ref_coverage,
                "min_candidate_det_coverage": args.min_candidate_det_coverage,
                "min_conf": args.min_conf,
                "use_reference_names_for_detections": not args.no_reference_names_for_detections,
            },
            "summary": global_counts,
        }, f, indent=4)

    print("")
    print("Comparación terminada.")
    print(f"  output:           {output_dir}")
    print(f"  summary.csv:      {summary_csv}")
    print(f"  global_summary:   {global_summary_path}")
    print(f"  overlay:          {overlay_dir}")
    print(f"  overlay_failures: {overlay_failures_dir}")
    print("")


if __name__ == "__main__":
    main()
