import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cargar_tamano_desde_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return int(config["out_width"]), int(config["out_height"])


def crear_mascara_green(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 25, 25])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((21, 21), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    return mask


def comprobar_mascara_toca_borde(mask, margin_px=3, min_green_pixels=20):
    h, w = mask.shape[:2]

    margin_px = max(1, int(margin_px))
    margin_px = min(margin_px, max(1, h // 2), max(1, w // 2))

    top = mask[:margin_px, :]
    bottom = mask[h - margin_px:h, :]
    left = mask[:, :margin_px]
    right = mask[:, w - margin_px:w]

    counts = {
        "top": int(np.count_nonzero(top)),
        "bottom": int(np.count_nonzero(bottom)),
        "left": int(np.count_nonzero(left)),
        "right": int(np.count_nonzero(right)),
    }

    touching_edges = [
        edge for edge, count in counts.items()
        if count >= min_green_pixels
    ]

    return {
        "touches_border": len(touching_edges) > 0,
        "touching_edges": touching_edges,
        "counts": counts,
        "margin_px": margin_px,
        "min_green_pixels": int(min_green_pixels),
    }


def obtener_contorno_principal(mask, min_area_ratio=0.02):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise RuntimeError("No se encontraron contornos verdes de la PCB")

    h, w = mask.shape[:2]
    min_area = h * w * min_area_ratio

    validos = [
        c for c in contours
        if cv2.contourArea(c) >= min_area
    ]

    if not validos:
        raise RuntimeError(
            "No hay contornos suficientemente grandes. "
            "Prueba a bajar --min-area-ratio"
        )

    principal = max(validos, key=cv2.contourArea)
    hull = cv2.convexHull(principal)

    return principal, hull


def obtener_rectangulo_minimo(hull):
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    center = np.array(rect[0], dtype=np.float32)
    return rect, center, box


def obtener_ejes_rectangulo(hull):
    rect, center, box = obtener_rectangulo_minimo(hull)

    p0, p1, p2, _ = box

    v1 = p1 - p0
    v2 = p2 - p1

    if np.linalg.norm(v1) >= np.linalg.norm(v2):
        u = v1
        v = v2
    else:
        u = v2
        v = v1

    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    return center, u, v, box


def proyectar_puntos(points, center, u, v):
    pts = points.reshape(-1, 2).astype(np.float32)
    rel = pts - center

    pu = rel @ u
    pv = rel @ v

    return pts, pu, pv


def seleccionar_puntos_lado(pts, pu, pv, lado, frac=0.18, min_points=6):
    umin, umax = np.min(pu), np.max(pu)
    vmin, vmax = np.min(pv), np.max(pv)

    urange = umax - umin
    vrange = vmax - vmin

    if lado == "left":
        limite = umin + frac * urange
        mask = pu <= limite
        distancia_lado = np.abs(pu - umin)

    elif lado == "right":
        limite = umax - frac * urange
        mask = pu >= limite
        distancia_lado = np.abs(pu - umax)

    elif lado == "top":
        limite = vmin + frac * vrange
        mask = pv <= limite
        distancia_lado = np.abs(pv - vmin)

    elif lado == "bottom":
        limite = vmax - frac * vrange
        mask = pv >= limite
        distancia_lado = np.abs(pv - vmax)

    else:
        raise ValueError("lado no válido")

    sel = pts[mask]

    if len(sel) < min_points:
        n = min(max(min_points, 12), len(pts))
        indices = np.argsort(distancia_lado)[:n]
        sel = pts[indices]

    if len(sel) < 2:
        raise RuntimeError(f"No hay suficientes puntos para ajustar el lado {lado}")

    return sel


def ajustar_linea(points):
    pts = points.reshape(-1, 1, 2).astype(np.float32)

    line = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    ).reshape(-1)

    return {
        "vx": float(line[0]),
        "vy": float(line[1]),
        "x0": float(line[2]),
        "y0": float(line[3])
    }


def linea_desde_segmento(x1, y1, x2, y2):
    pts = np.array([
        [x1, y1],
        [x2, y2]
    ], dtype=np.float32)
    return ajustar_linea(pts)


def interseccion_lineas(l1, l2):
    p1 = np.array([l1["x0"], l1["y0"]], dtype=np.float32)
    d1 = np.array([l1["vx"], l1["vy"]], dtype=np.float32)

    p2 = np.array([l2["x0"], l2["y0"]], dtype=np.float32)
    d2 = np.array([l2["vx"], l2["vy"]], dtype=np.float32)

    A = np.array([
        [d1[0], -d2[0]],
        [d1[1], -d2[1]]
    ], dtype=np.float32)

    b = p2 - p1

    det = np.linalg.det(A)

    if abs(det) < 1e-8:
        raise RuntimeError("Dos líneas son casi paralelas; no se puede calcular intersección")

    t_s = np.linalg.solve(A, b)
    t = t_s[0]

    return p1 + t * d1


def ordenar_esquinas_visualmente(points):
    pts = np.array(points, dtype=np.float32)

    if pts.shape != (4, 2):
        raise RuntimeError(f"Se esperaban 4 esquinas, recibidas: {pts.shape}")

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def aplicar_padding_cuadrilatero(corners, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0):
    """
    corners: TL, TR, BR, BL
    """
    corners = corners.astype(np.float32).copy()

    tl, tr, br, bl = corners

    ux = tr - tl
    uy = bl - tl

    ux_norm = np.linalg.norm(ux)
    uy_norm = np.linalg.norm(uy)

    if ux_norm < 1e-6 or uy_norm < 1e-6:
        return corners

    ux = ux / ux_norm
    uy = uy / uy_norm

    tl = tl - ux * pad_left - uy * pad_top
    tr = tr + ux * pad_right - uy * pad_top
    br = br + ux * pad_right + uy * pad_bottom
    bl = bl - ux * pad_left + uy * pad_bottom

    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_pcb(image, corners, out_width, out_height):
    destino = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        corners.astype(np.float32),
        destino
    )

    corrected = cv2.warpPerspective(
        image,
        H,
        (out_width, out_height)
    )

    return corrected, H, destino


def dibujar_linea(img, linea, color, thickness=2):
    vx = linea["vx"]
    vy = linea["vy"]
    x0 = linea["x0"]
    y0 = linea["y0"]

    p1 = (int(x0 - 2000 * vx), int(y0 - 2000 * vy))
    p2 = (int(x0 + 2000 * vx), int(y0 + 2000 * vy))

    cv2.line(img, p1, p2, color, thickness)


def dibujar_debug_homografia(
    image,
    principal,
    hull,
    box,
    corners_raw,
    corners_ordered,
    output_path,
    method="box",
    lados_pts=None,
    lineas=None,
):
    debug = image.copy()

    cv2.drawContours(debug, [principal], -1, (255, 0, 0), 1)
    cv2.drawContours(debug, [hull], -1, (0, 255, 255), 2)

    box_int = np.int32(box)
    cv2.polylines(debug, [box_int], isClosed=True, color=(255, 255, 0), thickness=2)

    if method in ["lines", "hough"] and lineas is not None:
        colores_lineas = {
            "top": (0, 255, 0),
            "right": (255, 0, 255),
            "bottom": (0, 165, 255),
            "left": (255, 255, 255),
        }

        for lado, linea in lineas.items():
            dibujar_linea(debug, linea, colores_lineas.get(lado, (255, 255, 255)), 2)

    if method == "lines" and lados_pts is not None:
        colores_puntos = {
            "top": (0, 255, 0),
            "right": (255, 0, 255),
            "bottom": (0, 165, 255),
            "left": (255, 255, 255),
        }

        for lado, pts in lados_pts.items():
            color = colores_puntos.get(lado, (255, 255, 255))
            step = max(1, len(pts) // 80)
            for p in pts[::step]:
                cv2.circle(debug, (int(p[0]), int(p[1])), 2, color, -1)

    for p in corners_raw:
        x, y = int(p[0]), int(p[1])
        cv2.circle(debug, (x, y), 5, (0, 165, 255), -1)

    labels = ["TL", "TR", "BR", "BL"]
    for label, p in zip(labels, corners_ordered):
        x, y = int(p[0]), int(p[1])
        cv2.circle(debug, (x, y), 9, (0, 0, 255), -1)
        cv2.putText(
            debug,
            label,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.putText(
        debug,
        f"homography={method}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imwrite(str(output_path), debug)


def preparar_edges_hough(image, mask, hull, roi_pad_px=35):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    x, y, w, h = cv2.boundingRect(hull)

    ih, iw = mask.shape[:2]

    x1 = max(0, x - roi_pad_px)
    y1 = max(0, y - roi_pad_px)
    x2 = min(iw - 1, x + w + roi_pad_px)
    y2 = min(ih - 1, y + h + roi_pad_px)

    roi = np.zeros_like(mask)
    cv2.rectangle(roi, (x1, y1), (x2, y2), 255, -1)

    edges = cv2.bitwise_and(edges, roi)

    return edges, {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }


def seleccionar_lineas_hough(
    image,
    mask,
    principal,
    hull,
    center,
    u,
    v,
    hough_threshold=60,
    hough_min_line_length_ratio=0.35,
    hough_max_line_gap=35,
    hough_angle_tolerance_deg=18,
    hough_roi_pad_px=35,
):
    pts, pu, pv = proyectar_puntos(hull, center, u, v)

    board_width = float(np.max(pu) - np.min(pu))
    board_height = float(np.max(pv) - np.min(pv))

    if board_width < board_height:
        board_width, board_height = board_height, board_width

    min_line_length = max(30, int(min(board_width, board_height) * hough_min_line_length_ratio))

    edges, roi_info = preparar_edges_hough(
        image=image,
        mask=mask,
        hull=hull,
        roi_pad_px=hough_roi_pad_px,
    )

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(hough_threshold),
        minLineLength=int(min_line_length),
        maxLineGap=int(hough_max_line_gap),
    )

    if raw_lines is None:
        raise RuntimeError("Hough no encontró líneas. Prueba a bajar hough_threshold o min_line_length_ratio.")

    cos_tol = math.cos(math.radians(hough_angle_tolerance_deg))

    horizontal_candidates = []
    vertical_candidates = []

    for item in raw_lines:
        x1, y1, x2, y2 = item[0].astype(float)

        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)

        d = p2 - p1
        length = float(np.linalg.norm(d))

        if length < min_line_length:
            continue

        d = d / length

        align_u = abs(float(np.dot(d, u)))
        align_v = abs(float(np.dot(d, v)))

        rel1 = p1 - center
        rel2 = p2 - center

        p1u = float(rel1 @ u)
        p2u = float(rel2 @ u)
        p1v = float(rel1 @ v)
        p2v = float(rel2 @ v)

        avg_u = (p1u + p2u) / 2.0
        avg_v = (p1v + p2v) / 2.0

        span_u = abs(p2u - p1u)
        span_v = abs(p2v - p1v)

        line_dict = linea_desde_segmento(x1, y1, x2, y2)

        candidate = {
            "segment": [float(x1), float(y1), float(x2), float(y2)],
            "line": line_dict,
            "length": length,
            "avg_u": avg_u,
            "avg_v": avg_v,
            "span_u": span_u,
            "span_v": span_v,
            "align_u": align_u,
            "align_v": align_v,
        }

        if align_u >= cos_tol and span_u >= board_width * hough_min_line_length_ratio:
            horizontal_candidates.append(candidate)

        if align_v >= cos_tol and span_v >= board_height * hough_min_line_length_ratio:
            vertical_candidates.append(candidate)

    if len(horizontal_candidates) < 2:
        raise RuntimeError(
            f"Hough no encontró suficientes líneas horizontales exteriores. "
            f"Encontradas={len(horizontal_candidates)}"
        )

    if len(vertical_candidates) < 2:
        raise RuntimeError(
            f"Hough no encontró suficientes líneas verticales exteriores. "
            f"Encontradas={len(vertical_candidates)}"
        )

    top = min(horizontal_candidates, key=lambda c: c["avg_v"])
    bottom = max(horizontal_candidates, key=lambda c: c["avg_v"])
    left = min(vertical_candidates, key=lambda c: c["avg_u"])
    right = max(vertical_candidates, key=lambda c: c["avg_u"])

    lineas = {
        "top": top["line"],
        "bottom": bottom["line"],
        "left": left["line"],
        "right": right["line"],
    }

    return lineas, {
        "raw_lines_count": int(len(raw_lines)),
        "horizontal_candidates": int(len(horizontal_candidates)),
        "vertical_candidates": int(len(vertical_candidates)),
        "roi": roi_info,
        "min_line_length": int(min_line_length),
        "selected": {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
        },
    }


def preprocess_gray_for_template(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def buscar_template_multiescala(
    image,
    template,
    min_scale=0.60,
    max_scale=1.50,
    scale_step=0.05,
):
    t0 = time.perf_counter()

    image_gray = preprocess_gray_for_template(image)
    template_gray_base = preprocess_gray_for_template(template)

    ih, iw = image_gray.shape[:2]

    mejor = None
    scales_tested = 0
    scales_valid = 0

    scales = np.arange(min_scale, max_scale + 0.0001, scale_step)

    for scale in scales:
        scales_tested += 1

        template_gray = cv2.resize(
            template_gray_base,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

        th, tw = template_gray.shape[:2]

        if th < 5 or tw < 5:
            continue

        if th >= ih or tw >= iw:
            continue

        scales_valid += 1

        result = cv2.matchTemplate(
            image_gray,
            template_gray,
            cv2.TM_CCOEFF_NORMED
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        x1 = int(max_loc[0])
        y1 = int(max_loc[1])
        x2 = int(x1 + tw)
        y2 = int(y1 + th)

        x_center = x1 + tw / 2
        y_center = y1 + th / 2

        deteccion = {
            "score": float(max_val),
            "x": float(x_center),
            "y": float(y_center),
            "bbox": [x1, y1, x2, y2],
            "template_width": int(tw),
            "template_height": int(th),
            "scale": float(scale),
        }

        if mejor is None or deteccion["score"] > mejor["score"]:
            mejor = deteccion

    elapsed = time.perf_counter() - t0

    if mejor is None:
        raise RuntimeError(
            "No se pudo buscar la serigrafía en ninguna escala. "
            "Revisa el tamaño del template o los parámetros de escala."
        )

    mejor["elapsed_seconds"] = float(elapsed)
    mejor["scales_tested"] = int(scales_tested)
    mejor["scales_valid"] = int(scales_valid)

    return mejor


def obtener_cuadrante(x, y, width, height):
    izquierda = x < width / 2
    arriba = y < height / 2

    if arriba and izquierda:
        return "tl"
    if arriba and not izquierda:
        return "tr"
    if not arriba and not izquierda:
        return "br"
    return "bl"


def generar_variantes_orientacion(image):
    return [
        {"name": "normal", "image": image.copy()},
        {"name": "rot180", "image": cv2.rotate(image, cv2.ROTATE_180)},
        {"name": "flip_horizontal", "image": cv2.flip(image, 1)},
        {"name": "flip_vertical", "image": cv2.flip(image, 0)},
    ]


def elegir_orientacion_por_serigrafia(
    corrected,
    orientation_template,
    expected_quadrant=None,
    min_score=0.45,
    quadrant_bonus=0.20,
    allow_low_score=False,
    min_scale=0.60,
    max_scale=1.50,
    scale_step=0.05,
):
    total_t0 = time.perf_counter()

    variantes = generar_variantes_orientacion(corrected)
    resultados = []

    for variante in variantes:
        variant_t0 = time.perf_counter()

        img = variante["image"]
        h, w = img.shape[:2]

        det = buscar_template_multiescala(
            image=img,
            template=orientation_template,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_step=scale_step,
        )

        variant_elapsed = time.perf_counter() - variant_t0

        quadrant = obtener_cuadrante(det["x"], det["y"], w, h)

        score_total = det["score"]
        if expected_quadrant is not None and quadrant == expected_quadrant:
            score_total += quadrant_bonus

        resultados.append({
            "name": variante["name"],
            "image": img,
            "detection": det,
            "quadrant": quadrant,
            "score_total": float(score_total),
            "elapsed_seconds": float(variant_elapsed),
        })

    total_elapsed = time.perf_counter() - total_t0

    mejor = sorted(
        resultados,
        key=lambda r: r["score_total"],
        reverse=True
    )[0]

    print("")
    print("Resultados orientación por serigrafía multiescala:")
    for r in resultados:
        print(
            f"  {r['name']}: "
            f"score={r['detection']['score']:.3f}, "
            f"scale={r['detection']['scale']:.2f}, "
            f"score_total={r['score_total']:.3f}, "
            f"quad={r['quadrant']}, "
            f"tiempo={r['elapsed_seconds']:.3f}s, "
            f"escalas={r['detection']['scales_valid']}/{r['detection']['scales_tested']}"
        )

    print(f"  TOTAL orientación: {total_elapsed:.3f}s")

    if mejor["detection"]["score"] < min_score:
        msg = (
            f"Score de serigrafía demasiado bajo: "
            f"{mejor['detection']['score']:.3f} < {min_score:.3f}. "
            f"No se acepta la orientación."
        )

        if not allow_low_score:
            raise RuntimeError(msg)

        print("")
        print("AVISO:")
        print("  " + msg)
        print("  Se continúa porque se ha usado --allow-low-orientation-score.")

    mejor["orientation_total_elapsed_seconds"] = float(total_elapsed)

    return mejor, resultados


def dibujar_template_debug(image, detection, text, output_path):
    debug = image.copy()

    x1, y1, x2, y2 = detection["bbox"]

    cv2.rectangle(
        debug,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.circle(
        debug,
        (int(detection["x"]), int(detection["y"])),
        6,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        debug,
        text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imwrite(str(output_path), debug)


def calcular_homografia_por_box(
    image,
    out_width,
    out_height,
    min_area_ratio,
    reject_mask_touch_border=True,
    border_margin_px=3,
    border_min_green_pixels=20,
):
    mask = crear_mascara_green(image)

    border_info = comprobar_mascara_toca_borde(
        mask=mask,
        margin_px=border_margin_px,
        min_green_pixels=border_min_green_pixels,
    )

    if reject_mask_touch_border and border_info["touches_border"]:
        raise RuntimeError(
            "Imagen descartada: la máscara verde toca el borde de la imagen. "
            f"bordes={border_info['touching_edges']}, conteos={border_info['counts']}"
        )

    principal, hull = obtener_contorno_principal(mask, min_area_ratio=min_area_ratio)

    _, _, box = obtener_rectangulo_minimo(hull)

    corners_raw = box.copy()
    corners_ordered = ordenar_esquinas_visualmente(corners_raw)

    corrected, H, destino = warp_pcb(image, corners_ordered, out_width, out_height)

    return {
        "mask": mask,
        "border_info": border_info,
        "principal": principal,
        "hull": hull,
        "box": box,
        "lados_pts": None,
        "lineas": None,
        "corners_raw": corners_raw,
        "corners_ordered": corners_ordered,
        "corrected": corrected,
        "homography": H,
        "destino": destino,
        "method": "box",
    }


def calcular_homografia_por_lineas(
    image,
    out_width,
    out_height,
    min_area_ratio,
    side_frac,
    reject_mask_touch_border=True,
    border_margin_px=3,
    border_min_green_pixels=20,
):
    mask = crear_mascara_green(image)

    border_info = comprobar_mascara_toca_borde(
        mask=mask,
        margin_px=border_margin_px,
        min_green_pixels=border_min_green_pixels,
    )

    if reject_mask_touch_border and border_info["touches_border"]:
        raise RuntimeError(
            "Imagen descartada: la máscara verde toca el borde de la imagen. "
            f"bordes={border_info['touching_edges']}, conteos={border_info['counts']}"
        )

    principal, hull = obtener_contorno_principal(mask, min_area_ratio=min_area_ratio)

    center, u, v, box = obtener_ejes_rectangulo(hull)
    pts, pu, pv = proyectar_puntos(principal, center, u, v)

    lados_pts = {
        "left": seleccionar_puntos_lado(pts, pu, pv, "left", side_frac),
        "right": seleccionar_puntos_lado(pts, pu, pv, "right", side_frac),
        "top": seleccionar_puntos_lado(pts, pu, pv, "top", side_frac),
        "bottom": seleccionar_puntos_lado(pts, pu, pv, "bottom", side_frac),
    }

    lineas = {
        "left": ajustar_linea(lados_pts["left"]),
        "right": ajustar_linea(lados_pts["right"]),
        "top": ajustar_linea(lados_pts["top"]),
        "bottom": ajustar_linea(lados_pts["bottom"]),
    }

    c1 = interseccion_lineas(lineas["top"], lineas["left"])
    c2 = interseccion_lineas(lineas["top"], lineas["right"])
    c3 = interseccion_lineas(lineas["bottom"], lineas["right"])
    c4 = interseccion_lineas(lineas["bottom"], lineas["left"])

    corners_raw = np.array([c1, c2, c3, c4], dtype=np.float32)
    corners_ordered = ordenar_esquinas_visualmente(corners_raw)

    corrected, H, destino = warp_pcb(image, corners_ordered, out_width, out_height)

    return {
        "mask": mask,
        "border_info": border_info,
        "principal": principal,
        "hull": hull,
        "box": box,
        "lados_pts": lados_pts,
        "lineas": lineas,
        "corners_raw": corners_raw,
        "corners_ordered": corners_ordered,
        "corrected": corrected,
        "homography": H,
        "destino": destino,
        "method": "lines",
    }


def calcular_homografia_por_hough(
    image,
    out_width,
    out_height,
    min_area_ratio,
    reject_mask_touch_border=True,
    border_margin_px=3,
    border_min_green_pixels=20,
    hough_threshold=60,
    hough_min_line_length_ratio=0.30,
    hough_max_line_gap=40,
    hough_angle_tolerance_deg=20,
    hough_roi_pad_px=45,
    quad_pad_top_px=0,
    quad_pad_bottom_px=0,
    quad_pad_left_px=0,
    quad_pad_right_px=0,
):
    mask = crear_mascara_green(image)

    border_info = comprobar_mascara_toca_borde(
        mask=mask,
        margin_px=border_margin_px,
        min_green_pixels=border_min_green_pixels,
    )

    if reject_mask_touch_border and border_info["touches_border"]:
        raise RuntimeError(
            "Imagen descartada: la máscara verde toca el borde de la imagen. "
            f"bordes={border_info['touching_edges']}, conteos={border_info['counts']}"
        )

    principal, hull = obtener_contorno_principal(mask, min_area_ratio=min_area_ratio)

    center, u, v, box = obtener_ejes_rectangulo(hull)

    lineas, hough_info = seleccionar_lineas_hough(
        image=image,
        mask=mask,
        principal=principal,
        hull=hull,
        center=center,
        u=u,
        v=v,
        hough_threshold=hough_threshold,
        hough_min_line_length_ratio=hough_min_line_length_ratio,
        hough_max_line_gap=hough_max_line_gap,
        hough_angle_tolerance_deg=hough_angle_tolerance_deg,
        hough_roi_pad_px=hough_roi_pad_px,
    )

    tl = interseccion_lineas(lineas["top"], lineas["left"])
    tr = interseccion_lineas(lineas["top"], lineas["right"])
    br = interseccion_lineas(lineas["bottom"], lineas["right"])
    bl = interseccion_lineas(lineas["bottom"], lineas["left"])

    corners_raw = np.array([tl, tr, br, bl], dtype=np.float32)
    corners_ordered = ordenar_esquinas_visualmente(corners_raw)

    corners_ordered = aplicar_padding_cuadrilatero(
        corners_ordered,
        pad_top=quad_pad_top_px,
        pad_bottom=quad_pad_bottom_px,
        pad_left=quad_pad_left_px,
        pad_right=quad_pad_right_px,
    )

    corrected, H, destino = warp_pcb(image, corners_ordered, out_width, out_height)

    return {
        "mask": mask,
        "border_info": border_info,
        "principal": principal,
        "hull": hull,
        "box": box,
        "lados_pts": None,
        "lineas": lineas,
        "corners_raw": corners_raw,
        "corners_ordered": corners_ordered,
        "corrected": corrected,
        "homography": H,
        "destino": destino,
        "method": "hough",
        "hough_info": hough_info,
    }


def calcular_homografia(
    image,
    out_width,
    out_height,
    min_area_ratio,
    side_frac,
    homography_method,
    reject_mask_touch_border=True,
    border_margin_px=3,
    border_min_green_pixels=20,
):
    if homography_method == "box":
        return calcular_homografia_por_box(
            image=image,
            out_width=out_width,
            out_height=out_height,
            min_area_ratio=min_area_ratio,
            reject_mask_touch_border=reject_mask_touch_border,
            border_margin_px=border_margin_px,
            border_min_green_pixels=border_min_green_pixels,
        )

    if homography_method == "lines":
        return calcular_homografia_por_lineas(
            image=image,
            out_width=out_width,
            out_height=out_height,
            min_area_ratio=min_area_ratio,
            side_frac=side_frac,
            reject_mask_touch_border=reject_mask_touch_border,
            border_margin_px=border_margin_px,
            border_min_green_pixels=border_min_green_pixels,
        )

    if homography_method == "hough":
        return calcular_homografia_por_hough(
            image=image,
            out_width=out_width,
            out_height=out_height,
            min_area_ratio=min_area_ratio,
            reject_mask_touch_border=reject_mask_touch_border,
            border_margin_px=border_margin_px,
            border_min_green_pixels=border_min_green_pixels,
        )

    raise RuntimeError(f"Método de homografía no soportado: {homography_method}")


def get_model_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def guardar_labels_y_csv(result, corrected_frame, label_path, csv_writer, image_name, model, save_crop, crops_dir):
    img_h, img_w = corrected_frame.shape[:2]

    with open(label_path, "w", encoding="utf-8") as label_file:
        if result.boxes is None or len(result.boxes) == 0:
            return

        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            class_name = get_model_name(model.names, class_id)

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x_center_px = (x1 + x2) / 2
            y_center_px = (y1 + y2) / 2
            width_px = x2 - x1
            height_px = y2 - y1

            x_center_norm = x_center_px / img_w
            y_center_norm = y_center_px / img_h
            width_norm = width_px / img_w
            height_norm = height_px / img_h

            label_file.write(
                f"{class_id} "
                f"{x_center_norm:.6f} "
                f"{y_center_norm:.6f} "
                f"{width_norm:.6f} "
                f"{height_norm:.6f} "
                f"{confidence:.6f}\n"
            )

            csv_writer.writerow([
                image_name,
                class_id,
                class_name,
                f"{confidence:.6f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                f"{x_center_px:.2f}",
                f"{y_center_px:.2f}",
                f"{width_px:.2f}",
                f"{height_px:.2f}",
                f"{x_center_norm:.6f}",
                f"{y_center_norm:.6f}",
                f"{width_norm:.6f}",
                f"{height_norm:.6f}",
            ])

            if save_crop:
                class_crop_dir = crops_dir / class_name
                class_crop_dir.mkdir(parents=True, exist_ok=True)

                x1i = max(0, int(x1))
                y1i = max(0, int(y1))
                x2i = min(img_w, int(x2))
                y2i = min(img_h, int(y2))

                crop = corrected_frame[y1i:y2i, x1i:x2i]

                if crop.size > 0:
                    crop_path = class_crop_dir / f"{Path(image_name).stem}_crop_{i:02d}.jpg"
                    cv2.imwrite(str(crop_path), crop)


def preparar_carpetas(output_dir):
    dirs = {
        "original": output_dir / "original",
        "mask": output_dir / "mask",
        "line_debug": output_dir / "line_debug",
        "orientation_debug": output_dir / "orientation_debug",
        "corrected": output_dir / "corrected",
        "annotated": output_dir / "annotated",
        "labels": output_dir / "labels",
        "json": output_dir / "json",
        "failed": output_dir / "failed",
        "crops": output_dir / "crops",
        "yolo_runs": output_dir / "yolo_runs",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def procesar_imagen(image_path, args, model, orientation_template, csv_writer, dirs, out_width, out_height):
    image_t0 = time.perf_counter()

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"No se pudo abrir la imagen: {image_path}")

    stem = image_path.stem

    original_path = dirs["original"] / f"{stem}.jpg"
    mask_path = dirs["mask"] / f"{stem}_mask.png"
    line_debug_path = dirs["line_debug"] / f"{stem}_homography_debug.jpg"
    orientation_debug_path = dirs["orientation_debug"] / f"{stem}_orientation_debug.jpg"
    corrected_path = dirs["corrected"] / f"{stem}.jpg"
    annotated_path = dirs["annotated"] / f"{stem}_annotated.jpg"
    label_path = dirs["labels"] / f"{stem}.txt"
    json_path = dirs["json"] / f"{stem}.json"

    cv2.imwrite(str(original_path), image)

    hom_t0 = time.perf_counter()

    hom = calcular_homografia(
        image=image,
        out_width=out_width,
        out_height=out_height,
        min_area_ratio=args.min_area_ratio,
        side_frac=args.side_frac,
        homography_method=args.homography_method,
        reject_mask_touch_border=not args.allow_mask_touch_border,
        border_margin_px=args.border_margin_px,
        border_min_green_pixels=args.border_min_green_pixels,
    )

    hom_elapsed = time.perf_counter() - hom_t0

    cv2.imwrite(str(mask_path), hom["mask"])

    dibujar_debug_homografia(
        image=image,
        principal=hom["principal"],
        hull=hom["hull"],
        box=hom["box"],
        corners_raw=hom["corners_raw"],
        corners_ordered=hom["corners_ordered"],
        output_path=line_debug_path,
        method=hom["method"],
        lados_pts=hom["lados_pts"],
        lineas=hom["lineas"],
    )

    corrected = hom["corrected"]

    orientation_info = None
    orientation_all_results = None
    orientation_elapsed = 0.0

    if orientation_template is not None:
        orientation_t0 = time.perf_counter()

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

        orientation_elapsed = time.perf_counter() - orientation_t0
        corrected = mejor_orientacion["image"]

        orientation_info = {
            "selected_variant": mejor_orientacion["name"],
            "score": mejor_orientacion["detection"]["score"],
            "score_total": mejor_orientacion["score_total"],
            "scale": mejor_orientacion["detection"]["scale"],
            "quadrant": mejor_orientacion["quadrant"],
            "bbox": mejor_orientacion["detection"]["bbox"],
            "x": mejor_orientacion["detection"]["x"],
            "y": mejor_orientacion["detection"]["y"],
            "elapsed_seconds": orientation_elapsed,
        }

        orientation_all_results = []

        for r in resultados_orientacion:
            orientation_all_results.append({
                "variant": r["name"],
                "score": r["detection"]["score"],
                "score_total": r["score_total"],
                "scale": r["detection"]["scale"],
                "quadrant": r["quadrant"],
                "bbox": r["detection"]["bbox"],
                "x": r["detection"]["x"],
                "y": r["detection"]["y"],
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
            output_path=orientation_debug_path
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

    yolo_elapsed = time.perf_counter() - yolo_t0

    result = results[0]

    annotated = result.plot()
    cv2.imwrite(str(annotated_path), annotated)

    guardar_labels_y_csv(
        result=result,
        corrected_frame=corrected,
        label_path=label_path,
        csv_writer=csv_writer,
        image_name=f"{stem}.jpg",
        model=model,
        save_crop=args.save_crop,
        crops_dir=dirs["crops"],
    )

    image_elapsed = time.perf_counter() - image_t0

    data = {
        "input_image": str(image_path),
        "outputs": {
            "original": str(original_path),
            "mask": str(mask_path),
            "homography_debug": str(line_debug_path),
            "orientation_debug": str(orientation_debug_path) if orientation_template is not None else None,
            "corrected": str(corrected_path),
            "annotated": str(annotated_path),
            "labels": str(label_path),
        },
        "timing": {
            "total_image_seconds": image_elapsed,
            "homography_seconds": hom_elapsed,
            "orientation_seconds": orientation_elapsed,
            "yolo_seconds": yolo_elapsed,
        },
        "mask_border_check": hom["border_info"],
        "homography": {
            "method": hom["method"],
            "corners_raw": hom["corners_raw"].tolist(),
            "corners_ordered": hom["corners_ordered"].tolist(),
            "destino": hom["destino"].tolist(),
            "matrix": hom["homography"].tolist(),
            "hough_info": hom.get("hough_info"),
        },
        "orientation": orientation_info,
        "orientation_all_results": orientation_all_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return {
        "image": image_path.name,
        "status": "OK",
        "corrected": str(corrected_path),
        "annotated": str(annotated_path),
        "json": str(json_path),
        "orientation": orientation_info["selected_variant"] if orientation_info else "",
        "orientation_score": orientation_info["score"] if orientation_info else "",
        "orientation_scale": orientation_info["scale"] if orientation_info else "",
        "orientation_time": orientation_elapsed,
        "homography_time": hom_elapsed,
        "yolo_time": yolo_elapsed,
        "total_time": image_elapsed,
    }


def listar_imagenes(input_dir):
    paths = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Procesa imágenes raw de PCB: homografía + orientación multiescala + YOLO."
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--config", default="/workspace/config_fiduciales.json")
    parser.add_argument("--orientation-template", default=None)

    parser.add_argument(
        "--homography-method",
        choices=["box", "lines", "hough"],
        default="hough",
        help="Método de homografía: box, lines o hough."
    )

    parser.add_argument("--orientation-expected-quadrant", choices=["tl", "tr", "br", "bl"], default=None)
    parser.add_argument("--orientation-min-score", type=float, default=0.45)
    parser.add_argument("--orientation-min-scale", type=float, default=0.55)
    parser.add_argument("--orientation-max-scale", type=float, default=0.80)
    parser.add_argument("--orientation-scale-step", type=float, default=0.03)

    parser.add_argument("--allow-low-orientation-score", action="store_true")
    parser.add_argument("--allow-mask-touch-border", action="store_true")
    parser.add_argument("--border-margin-px", type=int, default=3)
    parser.add_argument("--border-min-green-pixels", type=int, default=20)

    parser.add_argument("--component-model", default="/workspace/train37/weights/best.engine")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--min-area-ratio", type=float, default=0.02)
    parser.add_argument("--side-frac", type=float, default=0.18)
    parser.add_argument("--save-crop", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def main():
    process_t0 = time.perf_counter()
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise RuntimeError(f"No existe input-dir: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = preparar_carpetas(output_dir)

    out_width, out_height = cargar_tamano_desde_config(args.config)

    orientation_template = None
    if args.orientation_template:
        orientation_template = cv2.imread(str(args.orientation_template))
        if orientation_template is None:
            raise RuntimeError(f"No se pudo abrir orientation-template: {args.orientation_template}")

    print("")
    print("Cargando modelo de componentes:")
    print(f"  {args.component_model}")
    print(f"  homography-method: {args.homography_method}")
    print("")

    model = YOLO(args.component_model, task="detect")

    images = listar_imagenes(input_dir)
    if args.limit is not None:
        images = images[:args.limit]

    if not images:
        raise RuntimeError(f"No se han encontrado imágenes en: {input_dir}")

    detections_csv_path = output_dir / "detections.csv"
    summary_csv_path = output_dir / "summary.csv"

    with open(detections_csv_path, "w", newline="", encoding="utf-8") as csv_file, \
         open(summary_csv_path, "w", newline="", encoding="utf-8") as summary_file:

        csv_writer = csv.writer(csv_file)
        summary_writer = csv.writer(summary_file)

        csv_writer.writerow([
            "image",
            "class_id",
            "class_name",
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
        ])

        summary_writer.writerow([
            "image",
            "status",
            "orientation",
            "orientation_score",
            "orientation_scale",
            "homography_time_s",
            "orientation_time_s",
            "yolo_time_s",
            "total_time_s",
            "message",
        ])

        print(f"Procesando {len(images)} imágenes...")
        print("")

        for idx, image_path in enumerate(images, start=1):
            print(f"[{idx}/{len(images)}] {image_path.name}")

            try:
                result = procesar_imagen(
                    image_path=image_path,
                    args=args,
                    model=model,
                    orientation_template=orientation_template,
                    csv_writer=csv_writer,
                    dirs=dirs,
                    out_width=out_width,
                    out_height=out_height,
                )

                summary_writer.writerow([
                    result["image"],
                    "OK",
                    result["orientation"],
                    result["orientation_score"],
                    result["orientation_scale"],
                    f"{result['homography_time']:.3f}",
                    f"{result['orientation_time']:.3f}",
                    f"{result['yolo_time']:.3f}",
                    f"{result['total_time']:.3f}",
                    "",
                ])

                print(
                    f"  OK → {result['annotated']} | "
                    f"hom={result['homography_time']:.2f}s "
                    f"ori={result['orientation_time']:.2f}s "
                    f"yolo={result['yolo_time']:.2f}s "
                    f"total={result['total_time']:.2f}s"
                )

            except Exception as e:
                failed_path = dirs["failed"] / f"{image_path.stem}_error.txt"

                with open(failed_path, "w", encoding="utf-8") as f:
                    f.write(str(e))
                    f.write("\n")

                summary_writer.writerow([
                    image_path.name,
                    "FAIL",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    str(e),
                ])

                print(f"  FAIL: {e}")

    total_process_elapsed = time.perf_counter() - process_t0

    print("")
    print("Proceso terminado.")
    print(f"Tiempo total proceso: {total_process_elapsed:.3f}s")
    print("")
    print("Resultados:")
    print(f"  corrected:          {dirs['corrected']}")
    print(f"  annotated:          {dirs['annotated']}")
    print(f"  labels:             {dirs['labels']}")
    print(f"  homography_debug:   {dirs['line_debug']}")
    print(f"  orientation_debug:  {dirs['orientation_debug']}")
    print(f"  failed:             {dirs['failed']}")
    print(f"  detections.csv:     {detections_csv_path}")
    print(f"  summary.csv:        {summary_csv_path}")
    print("")


if __name__ == "__main__":
    main()
