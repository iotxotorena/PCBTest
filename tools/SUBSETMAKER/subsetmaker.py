SubsetMaker — GUI tool to create label-balanced subsets of YOLO CV datasets.

Supports standard YOLO directory layout:
    dataset/
        images/
            train/  (or directly here)
            val/
        labels/
            train/
            val/
        data.yaml   (optional, for class names)

Usage:
    python subsetmaker.py
"""

import json
import os
import re
import shutil
import random
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def parse_yaml(content: str) -> dict:
    """
    Simple YAML parser for YOLO data.yaml files.

    Supports:
    - key: value pairs (strings, integers)
    - flow sequence: key: [item1, item2]
    - block sequence: key:\\n- item1\\n- item2
    - block mapping under a key: key:\\n  0: val0\\n  1: val1
    - comments (#) and empty lines
    - quoted strings (single or double quotes)
    """
    result: dict = {}
    lines = content.splitlines()
    i = 0
    n = len(lines)

    def _strip_comment(s: str) -> str:
        """Remove inline YAML comment (# not inside quotes)."""
        in_sq = in_dq = False
        for idx, ch in enumerate(s):
            if ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif ch == '#' and not in_sq and not in_dq:
                return s[:idx].rstrip()
        return s

    def _unquote(s: str) -> str:
        """Strip surrounding quotes and unescape escape sequences."""
        s = s.strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or
                            (s[0] == "'" and s[-1] == "'")):
            inner = s[1:-1]
            if s[0] == '"':
                inner = inner.replace('\\\\', '\\').replace('\\"', '"')
            return inner
        return s

    def _parse_flow_sequence(s: str) -> list:
        """Parse a YAML flow sequence like [a, b, "c: d"] into a list."""
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        items = []
        current = ''
        in_sq = in_dq = False
        for ch in s:
            if ch == "'" and not in_dq:
                in_sq = not in_sq
                current += ch
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
                current += ch
            elif ch == ',' and not in_sq and not in_dq:
                items.append(_unquote(current.strip()))
                current = ''
            else:
                current += ch
        if current.strip():
            items.append(_unquote(current.strip()))
        return items

    while i < n:
        line = lines[i]
        # Skip empty lines and comments
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            i += 1
            continue

        # Top-level key: value
        if ':' in line and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('-'):
            colon_idx = line.index(':')
            key = line[:colon_idx].strip()
            raw_value = line[colon_idx + 1:]
            value_stripped = _strip_comment(raw_value).strip()

            if value_stripped == '':
                # Value is on subsequent lines — could be block sequence or block mapping
                block_items = []
                block_map: dict = {}
                j = i + 1
                while j < n:
                    next_line = lines[j]
                    next_stripped = next_line.strip()
                    if not next_stripped or next_stripped.startswith('#'):
                        j += 1
                        continue
                    # Block sequence item
                    if next_stripped.startswith('- ') or next_stripped == '-':
                        item_text = next_stripped[1:].strip() if next_stripped != '-' else ''
                        block_items.append(_unquote(item_text))
                        j += 1
                    # Block mapping item (indented key: value)
                    elif ':' in next_stripped and (next_line.startswith(' ') or next_line.startswith('\t')):
                        colon2 = next_stripped.index(':')
                        mk = next_stripped[:colon2].strip()
                        mv = _unquote(_strip_comment(next_stripped[colon2 + 1:]).strip())
                        try:
                            block_map[int(mk)] = mv
                        except ValueError:
                            block_map[mk] = mv
                        j += 1
                    else:
                        break

                i = j
                if block_items:
                    result[key] = block_items
                elif block_map:
                    result[key] = block_map
                else:
                    result[key] = None
            elif value_stripped.startswith('['):
                # Flow sequence
                result[key] = _parse_flow_sequence(value_stripped)
                i += 1
            else:
                # Scalar value
                val = _unquote(value_stripped)
                try:
                    result[key] = int(val)
                except ValueError:
                    result[key] = val
                i += 1
        else:
            i += 1

    return result

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
LABEL_EXTENSION = ".txt"


def _is_zone_identifier(fname: str) -> bool:
    """Return True for Windows Zone.Identifier metadata files that should be ignored."""
    return "zone.identifier" in fname.lower()


def _image_stems_in_dir(directory: str) -> set[str]:
    """Return lowercase image stems present in *directory* (non-recursive).

    Ignores Zone.Identifier files and is case-insensitive for both stem and
    extension, so ``001.JPG`` and ``001.jpg`` both contribute ``"001"``.
    """
    stems: set[str] = set()
    if not os.path.isdir(directory):
        return stems
    for fname in os.listdir(directory):
        if _is_zone_identifier(fname):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() in IMAGE_EXTENSIONS:
            stems.add(stem.lower())
    return stems


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def find_splits(dataset_dir: str) -> list[str]:
    """Return available split names (e.g. ['train', 'val']) inside dataset_dir."""
    splits = []
    images_dir = os.path.join(dataset_dir, "images")
    if os.path.isdir(images_dir):
        for entry in sorted(os.listdir(images_dir)):
            if os.path.isdir(os.path.join(images_dir, entry)):
                splits.append(entry)
    if not splits:
        # Flat layout: images and labels sit directly under dataset_dir
        splits = [""]
    return splits


def images_dir_for_split(dataset_dir: str, split: str) -> str:
    if split:
        return os.path.join(dataset_dir, "images", split)
    return os.path.join(dataset_dir, "images")


def labels_dir_for_split(dataset_dir: str, split: str) -> str:
    if split:
        return os.path.join(dataset_dir, "labels", split)
    return os.path.join(dataset_dir, "labels")


def find_label_path(dataset_dir: str, split: str, image_stem: str) -> str | None:
    label_dir = labels_dir_for_split(dataset_dir, split)
    candidate = os.path.join(label_dir, image_stem + LABEL_EXTENSION)
    return candidate if os.path.isfile(candidate) else None


def load_class_names(dataset_dir: str) -> list[str]:
    """Load class names from data.yaml if present."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.isfile(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = parse_yaml(fh.read())
        except OSError:
            return []
        names = data.get("names", [])
        if isinstance(names, list):
            return [str(n) for n in names]
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
    return []


def parse_label_file(path: str) -> list[int]:
    """Return list of class ids found in a YOLO label file."""
    ids: list[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                ids.append(int(parts[0]))
            except (ValueError, IndexError):
                pass
    return ids


def scan_dataset(
    dataset_dir: str,
    split: str,
    progress_cb=None,
) -> dict[int, list[str]]:
    """
    Return a dict mapping class_id -> list of image file paths that contain
    at least one annotation with that class_id.
    """
    images_dir = images_dir_for_split(dataset_dir, split)
    if not os.path.isdir(images_dir):
        return {}

    image_files = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    class_to_images: dict[int, list[str]] = {}
    total = len(image_files)
    for idx, img_path in enumerate(image_files):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = find_label_path(dataset_dir, split, stem)
        if label_path:
            for cls_id in set(parse_label_file(label_path)):
                class_to_images.setdefault(cls_id, []).append(img_path)
        if progress_cb and total:
            progress_cb(int((idx + 1) / total * 100))

    return class_to_images


def filter_label_file(src_label_path: str, kept_classes: set[int]) -> str:
    """
    Return the filtered content of a label file keeping only lines whose
    class id is in *kept_classes*.
    """
    lines = []
    with open(src_label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            try:
                if int(parts[0]) in kept_classes:
                    lines.append(stripped)
            except (ValueError, IndexError):
                pass
    return "\n".join(lines)


def copy_image(src: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


def write_label(content: str, dst_dir: str, stem: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, stem + LABEL_EXTENSION)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(content)



def create_subset(
    dataset_dir: str,
    split: str,
    selected_class_ids: list[int],
    max_per_class: int,
    output_dir: str,
    class_names: list[str],
    remap_ids: bool,
    log_cb=None,
    progress_cb=None,
) -> dict[str, int]:
    """
    Core subset creation logic.

    Returns a summary dict: {class_name: count_of_images_included}.
    """
    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    kept_classes = set(selected_class_ids)

    # --- Scan ----------------------------------------------------------------
    log("Scanning dataset...")
    class_to_images = scan_dataset(dataset_dir, split, progress_cb)

    # --- Sample --------------------------------------------------------------
    selected_images: set[str] = set()
    counts: dict[int, int] = {}
    for cls_id in selected_class_ids:
        pool = class_to_images.get(cls_id, [])
        random.shuffle(pool)
        chosen = pool[:max_per_class]
        counts[cls_id] = len(chosen)
        selected_images.update(chosen)

    log(f"Selected {len(selected_images)} unique image(s) covering "
        f"{len(selected_class_ids)} class(es).")

    # --- Remap class ids -----------------------------------------------------
    sorted_ids = sorted(selected_class_ids)
    id_remap: dict[int, int] = (
        {old: new for new, old in enumerate(sorted_ids)}
        if remap_ids
        else {i: i for i in sorted_ids}
    )

    # --- Output dirs ---------------------------------------------------------
    out_images = (
        os.path.join(output_dir, "images", split)
        if split
        else os.path.join(output_dir, "images")
    )
    out_labels = (
        os.path.join(output_dir, "labels", split)
        if split
        else os.path.join(output_dir, "labels")
    )
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # --- Copy files ----------------------------------------------------------
    total = len(selected_images)
    for idx, img_path in enumerate(sorted(selected_images)):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = find_label_path(dataset_dir, split, stem)

        copy_image(img_path, out_images)

        if label_path:
            content = filter_and_remap_label(label_path, kept_classes, id_remap)
            write_label(content, out_labels, stem)
        else:
            write_label("", out_labels, stem)

        if progress_cb and total:
            progress_cb(int((idx + 1) / total * 100))

    # --- data.yaml -----------------------------------------------------------
    kept_class_names = {
        id_remap[cid]: (
            class_names[cid] if cid < len(class_names) else str(cid)
        )
        for cid in selected_class_ids
    }
    _write_data_yaml(dataset_dir, output_dir, kept_class_names)

    summary = {
        (class_names[cid] if cid < len(class_names) else str(cid)): counts.get(cid, 0)
        for cid in selected_class_ids
    }
    log("Done.")
    return summary


def filter_and_remap_label(
    src_label_path: str, kept_classes: set[int], id_remap: dict[int, int]
) -> str:
    """Filter and optionally remap class ids in a label file."""
    lines = []
    with open(src_label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            try:
                cls_id = int(parts[0])
                if cls_id in kept_classes:
                    new_id = id_remap.get(cls_id, cls_id)
                    lines.append(" ".join([str(new_id)] + parts[1:]))
            except (ValueError, IndexError):
                pass
    return "\n".join(lines)


def check_dataset(dataset_dir: str, split: str) -> dict[str, list[str]]:
    """
    Check a YOLO dataset for integrity issues.

    When *split* is empty the check is performed recursively across the entire
    ``images/`` and ``labels/`` subdirectory trees.  When a split is given
    only the corresponding leaf directory is checked (non-recursively).

    The comparison is case-insensitive for both image extensions and stems so
    that files created on Windows (e.g. ``001.JPG``) are correctly matched
    against their label counterpart on case-sensitive Linux filesystems.
    Zone.Identifier metadata files created by Windows are silently ignored.

    Returns a dict with:
        "missing_labels": image paths that have no corresponding label file.
        "orphan_labels":  label paths that have no corresponding image file.
    """
    images_dir = images_dir_for_split(dataset_dir, split)
    labels_dir = labels_dir_for_split(dataset_dir, split)

    missing_labels: list[str] = []
    orphan_labels: list[str] = []

    if split:
        # Non-recursive scan of a single split directory
        if os.path.isdir(images_dir):
            for fname in sorted(os.listdir(images_dir)):
                if _is_zone_identifier(fname):
                    continue
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                    stem = os.path.splitext(fname)[0]
                    lbl = os.path.join(labels_dir, stem + LABEL_EXTENSION)
                    if not os.path.isfile(lbl):
                        missing_labels.append(os.path.join(images_dir, fname))

        if os.path.isdir(labels_dir):
            img_stems = _image_stems_in_dir(images_dir)
            for fname in sorted(os.listdir(labels_dir)):
                if _is_zone_identifier(fname):
                    continue
                if fname.lower().endswith(LABEL_EXTENSION):
                    stem = os.path.splitext(fname)[0]
                    if stem.lower() not in img_stems:
                        orphan_labels.append(os.path.join(labels_dir, fname))
    else:
        # Recursive scan across the whole images/ and labels/ trees
        if os.path.isdir(images_dir):
            for root, _dirs, files in os.walk(images_dir):
                for fname in sorted(files):
                    if _is_zone_identifier(fname):
                        continue
                    if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                        stem = os.path.splitext(fname)[0]
                        rel = os.path.relpath(root, images_dir)
                        lbl_dir = labels_dir if rel in (".", "") else os.path.join(labels_dir, rel)
                        lbl = os.path.join(lbl_dir, stem + LABEL_EXTENSION)
                        if not os.path.isfile(lbl):
                            missing_labels.append(os.path.join(root, fname))

        if os.path.isdir(labels_dir):
            for root, _dirs, files in os.walk(labels_dir):
                rel = os.path.relpath(root, labels_dir)
                img_dir = images_dir if rel in (".", "") else os.path.join(images_dir, rel)
                img_stems = _image_stems_in_dir(img_dir)
                for fname in sorted(files):
                    if _is_zone_identifier(fname):
                        continue
                    if fname.lower().endswith(LABEL_EXTENSION):
                        stem = os.path.splitext(fname)[0]
                        if stem.lower() not in img_stems:
                            orphan_labels.append(os.path.join(root, fname))

    return {"missing_labels": missing_labels, "orphan_labels": orphan_labels}


def fix_missing_labels(image_paths: list[str], dataset_dir: str, split: str) -> int:
    """
    Create an empty label file for every image that lacks one.
    Returns the number of label files created.
    """
    images_dir = images_dir_for_split(dataset_dir, split)
    labels_dir = labels_dir_for_split(dataset_dir, split)
    os.makedirs(labels_dir, exist_ok=True)
    count = 0
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img_dir = os.path.dirname(img_path)
        rel = os.path.relpath(img_dir, images_dir)
        lbl_dir = labels_dir if rel in (".", "") else os.path.join(labels_dir, rel)
        lbl = os.path.join(lbl_dir, stem + LABEL_EXTENSION)
        if not os.path.isfile(lbl):
            os.makedirs(lbl_dir, exist_ok=True)
            with open(lbl, "w", encoding="utf-8") as fh:
                fh.write("")
            count += 1
    return count


def fix_orphan_labels(label_paths: list[str]) -> int:
    """
    Delete label files that have no corresponding image.
    Returns the number of files deleted.
    """
    count = 0
    for path in label_paths:
        if os.path.isfile(path):
            os.remove(path)
            count += 1
    return count


def split_dataset(
    dataset_dir: str,
    split: str,
    output_dir: str,
    train_pct: float,
    seed: int = 42,
    log_cb=None,
    progress_cb=None,
) -> dict[str, int]:
    """
    Split a YOLO dataset split into train/val subsets.

    Images (and their label files) from *split* inside *dataset_dir* are
    randomly divided so that *train_pct* % go to ``images/train`` /
    ``labels/train`` and the remainder go to ``images/val`` / ``labels/val``
    inside *output_dir*.

    Parameters
    ----------
    dataset_dir : str
        Root directory of the source dataset.
    split : str
        Source split to read from (e.g. ``"train"``, ``""`` for flat layout).
    output_dir : str
        Destination directory; train/val sub-directories are created here.
    train_pct : float
        Percentage of images assigned to the train split (0 < train_pct < 100).
    seed : int
        Random seed for reproducibility.
    log_cb : callable, optional
        Called with progress message strings.
    progress_cb : callable, optional
        Called with an integer 0-100 representing overall progress.

    Returns
    -------
    dict[str, int]
        ``{"train": <n_train>, "val": <n_val>}`` counts.
    """
    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    images_dir = images_dir_for_split(dataset_dir, split)
    labels_dir = labels_dir_for_split(dataset_dir, split)

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Collect all image files
    all_images = sorted(
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    if not all_images:
        raise ValueError(f"No images found in: {images_dir}")

    log(f"Found {len(all_images)} image(s) in source split '{split or 'flat'}'.")

    # Shuffle and split
    random.seed(seed)
    shuffled = list(all_images)
    random.shuffle(shuffled)
    total = len(shuffled)
    n_train = round(total * train_pct / 100)
    # Ensure at least one image in each split when there are enough images
    n_train = max(1, min(n_train, total - 1))
    n_val = total - n_train

    train_images = shuffled[:n_train]
    val_images = shuffled[n_train:]

    log(f"Split: {len(train_images)} train / {len(val_images)} val.")

    # Create output directories
    out_dirs: dict[str, tuple[str, str]] = {
        "train": (
            os.path.join(output_dir, "images", "train"),
            os.path.join(output_dir, "labels", "train"),
        ),
        "val": (
            os.path.join(output_dir, "images", "val"),
            os.path.join(output_dir, "labels", "val"),
        ),
    }
    for img_d, lbl_d in out_dirs.values():
        os.makedirs(img_d, exist_ok=True)
        os.makedirs(lbl_d, exist_ok=True)

    # Copy files
    done = 0
    for dest_split, image_list in (("train", train_images), ("val", val_images)):
        img_out, lbl_out = out_dirs[dest_split]
        for fname in image_list:
            src_img = os.path.join(images_dir, fname)
            shutil.copy2(src_img, os.path.join(img_out, fname))

            stem = os.path.splitext(fname)[0]
            src_lbl = os.path.join(labels_dir, stem + LABEL_EXTENSION)
            if os.path.isfile(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_out, stem + LABEL_EXTENSION))

            done += 1
            if progress_cb and total:
                progress_cb(int(done / total * 100))

    # Copy data.yaml if present
    yaml_src = os.path.join(dataset_dir, "data.yaml")
    if os.path.isfile(yaml_src):
        shutil.copy2(yaml_src, os.path.join(output_dir, "data.yaml"))
        log("Copied data.yaml to output folder.")

    log("Done.")
    return {"train": len(train_images), "val": len(val_images)}


def remap_labels(
    labels_dir: str,
    id_remap: dict[int, int],
    output_dir: str | None = None,
) -> int:
    """Apply *id_remap* to every YOLO label file in *labels_dir*.

    If *output_dir* is given and differs from *labels_dir*, rewritten files are
    placed there; otherwise files are modified in-place.  Only files whose
    content actually changes are written when working in-place (all files are
    written when copying to a different *output_dir*).

    Returns the number of label files written.
    """
    in_place = output_dir is None or (
        os.path.realpath(output_dir) == os.path.realpath(labels_dir)
    )
    dst_dir = labels_dir if in_place else output_dir
    if not in_place:
        os.makedirs(dst_dir, exist_ok=True)

    written = 0
    for fname in sorted(os.listdir(labels_dir)):
        if not fname.lower().endswith(LABEL_EXTENSION):
            continue
        src = os.path.join(labels_dir, fname)
        new_lines: list[str] = []
        changed = False
        with open(src, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                try:
                    cls_id = int(parts[0])
                    new_id = id_remap.get(cls_id, cls_id)
                    new_lines.append(" ".join([str(new_id)] + parts[1:]))
                    if new_id != cls_id:
                        changed = True
                except (ValueError, IndexError):
                    new_lines.append(stripped)
        if changed or not in_place:
            dst = os.path.join(dst_dir, fname)
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write("\n".join(new_lines))
                if new_lines:
                    fh.write("\n")
            written += 1
    return written


def remap_yaml_classes(
    yaml_path: str,
    id_remap: dict[int, int],
    output_path: str,
) -> None:
    """Rewrite *yaml_path* with class names remapped by *id_remap*.

    The updated YAML is written to *output_path* (which may equal *yaml_path*
    for an in-place update).  ``train``, ``val``, and ``test`` paths from the
    original YAML are preserved.  Classes whose IDs are not present in
    *id_remap* keep their original ID.  When two classes are mapped to the
    same new ID (merge), the last name in sorted order of the original IDs
    wins.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            data = parse_yaml(fh.read())
    except OSError:
        return

    names = data.get("names", [])
    if isinstance(names, list):
        old_names: dict[int, str] = {i: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        old_names = {int(k): str(v) for k, v in names.items()}
    else:
        old_names = {}

    # Apply id_remap: map each old class id to new id, keeping the name
    new_names: dict[int, str] = {}
    for old_id in sorted(old_names):
        new_id = id_remap.get(old_id, old_id)
        new_names[new_id] = old_names[old_id]

    sorted_ids = sorted(new_names)
    nc = len(sorted_ids)
    names_list = [new_names[i] for i in sorted_ids]
    names_block = "".join(f"- {_yaml_scalar(name)}\n" for name in names_list)
    yaml_content = f"nc: {nc}\nnames:\n{names_block}"

    # Preserve train/val/test paths from the original YAML
    extra_lines = []
    for key in ("train", "val", "test"):
        if key in data:
            extra_lines.append(f"{key}: {data[key]}")
    if extra_lines:
        yaml_content += "\n".join(extra_lines) + "\n"

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(yaml_content)


def load_yaml_file(yaml_path: str) -> dict:
    """
    Load a YOLO-style YAML configuration file.

    Returns:
        {
            "names": list[str] – class names,
            "nc":    int       – number of classes,
            "raw":   dict      – full parsed YAML contents,
        }
    """
    if not os.path.isfile(yaml_path):
        return {"names": [], "nc": 0, "raw": {}}
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            data = parse_yaml(fh.read())
    except OSError:
        return {"names": [], "nc": 0, "raw": {}}
    names = data.get("names", [])
    if isinstance(names, list):
        names = [str(n) for n in names]
    elif isinstance(names, dict):
        names = [str(names[k]) for k in sorted(names)]
    else:
        names = []
    nc = int(data.get("nc", len(names)))
    return {"names": names, "nc": nc, "raw": data}


def yaml_from_json(json_path: str) -> dict[int, str]:
    """
    Extract class names from a JSON file and return as a {class_id: name} mapping.

    Supported JSON formats
    ----------------------
    1. COCO annotation file — has a ``"categories"`` list whose items each have
       at least a ``"name"`` key (and an optional ``"id"`` key)::

           {"categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]}

    2. Names-keyed object — same structure as a YOLO data.yaml converted to
       JSON::

           {"names": ["cat", "dog"]}          # list form
           {"names": {"0": "cat", "1": "dog"}} # dict form

    3. Plain list of strings::

           ["cat", "dog", "bird"]

    Returns
    -------
    dict[int, str]
        Class-id → class-name mapping (ids are 0-based unless the source
        provides explicit ids, as in COCO categories).

    Raises
    ------
    ValueError
        If the JSON does not match any supported format.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # 1. COCO format: {"categories": [{"id": ..., "name": ...}, ...]}
    if isinstance(data, dict) and "categories" in data:
        cats = data["categories"]
        if isinstance(cats, list) and cats and isinstance(cats[0], dict) and "name" in cats[0]:
            result = {}
            for i, cat in enumerate(cats):
                if isinstance(cat, dict) and "name" in cat:
                    result[int(cat.get("id", i))] = str(cat["name"])
            if result:
                return result

    # 2. Names-keyed object: {"names": [...]} or {"names": {...}}
    if isinstance(data, dict) and "names" in data:
        names = data["names"]
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}

    # 3. Plain list of strings: ["cat", "dog"]
    if isinstance(data, list):
        result = {i: str(n) for i, n in enumerate(data) if isinstance(n, str)}
        if result:
            return result

    raise ValueError(
        "Cannot extract class names from the JSON file. "
        "Expected a COCO 'categories' list, a 'names' key, or a plain list of strings."
    )


def _yaml_scalar(value: str) -> str:
    """
    Return a YAML-safe representation of *value* for use in a block sequence.

    Plain (unquoted) scalars are used when safe; double-quoted otherwise.
    This produces valid YAML even for names that contain special characters.
    """
    # Characters that require quoting when they lead or appear in a scalar
    _NEEDS_QUOTE_START = set(':?|>!&*#%@`')
    _NEEDS_QUOTE_ANY = {': ', ' #'}

    if (not value
            or value[0] in _NEEDS_QUOTE_START
            or value[0].isspace()
            or value[-1].isspace()
            or any(token in value for token in _NEEDS_QUOTE_ANY)):
        # Double-quote and escape internal double-quotes and backslashes
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _write_data_yaml(
    dataset_dir: str,
    output_dir: str,
    kept_class_names: dict[int, str],
    *,
    train_path: str | None = None,
    val_path: str | None = None,
    test_path: str | None = None,
) -> None:
    """Write data.yaml for the new subset using YAML block-sequence format.

    When *train_path* / *val_path* / *test_path* are provided they are written
    directly into the output YAML.  Otherwise the function falls back to
    reading the source ``data.yaml`` inside *dataset_dir*; if that file does
    not exist either, the YOLO default relative paths (``images/train`` and
    ``images/val``) are used.
    """
    sorted_ids = sorted(kept_class_names)
    nc = len(sorted_ids)
    names_list = [kept_class_names[i] for i in sorted_ids]

    # Build names as a proper YAML block sequence (one entry per line),
    # quoting any name that contains YAML-special characters.
    names_block = "".join(f"- {_yaml_scalar(name)}\n" for name in names_list)
    yaml_content = f"nc: {nc}\nnames:\n{names_block}"

    if train_path is not None or val_path is not None or test_path is not None:
        # Use explicitly supplied paths.
        explicit: dict[str, str] = {}
        if train_path is not None:
            explicit["train"] = train_path
        if val_path is not None:
            explicit["val"] = val_path
        if test_path is not None:
            explicit["test"] = test_path
        extra_lines = [f"{k}: {v}" for k, v in explicit.items() if v]
        if extra_lines:
            yaml_content += "\n".join(extra_lines) + "\n"
    else:
        src_yaml = os.path.join(dataset_dir, "data.yaml")
        if os.path.isfile(src_yaml):
            try:
                with open(src_yaml, "r", encoding="utf-8") as fh:
                    orig = parse_yaml(fh.read())
            except OSError:
                orig = {}
            extra_lines = []
            for key in ("train", "val", "test"):
                if key in orig:
                    extra_lines.append(f"{key}: {orig[key]}")
            if extra_lines:
                yaml_content += "\n".join(extra_lines) + "\n"
        else:
            # No source YAML — use YOLO default relative paths.
            yaml_content += "train: images/train\nval: images/val\n"

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(yaml_content)


# ---------------------------------------------------------------------------
# Theme toggle widget
# ---------------------------------------------------------------------------

class ThemeToggle(tk.Canvas):
    """Sliding toggle switch for selecting dark or light theme."""

    _W = 46   # canvas width
    _H = 24   # canvas height
    _PAD = 3  # padding around knob

    # Fixed track colours (independent of app theme)
    _TRACK_ON = "#89b4fa"   # blue  – dark mode active
    _TRACK_OFF = "#9ca0b0"  # grey  – light mode active

    def __init__(self, parent, *, dark: bool = True, command=None, **kwargs):
        super().__init__(
            parent,
            width=self._W, height=self._H,
            highlightthickness=0, cursor="hand2",
            **kwargs,
        )
        self._dark = dark
        self._command = command
        self.bind("<Button-1>", self._click)
        self._draw()

    # ------------------------------------------------------------------
    def _draw(self) -> None:
        self.delete("all")
        w, h, p = self._W, self._H, self._PAD
        r = h // 2
        track = self._TRACK_ON if self._dark else self._TRACK_OFF
        knob_cx = w - r if self._dark else r

        # Rounded track: two end-caps + filled rectangle
        self.create_oval(0, 0, h, h, fill=track, outline="")
        self.create_oval(w - h, 0, w, h, fill=track, outline="")
        self.create_rectangle(r, 0, w - r, h, fill=track, outline="")

        # Knob (white circle)
        self.create_oval(
            knob_cx - r + p, p,
            knob_cx + r - p, h - p,
            fill="white", outline="",
        )

    def _click(self, _event) -> None:
        self._dark = not self._dark
        self._draw()
        if self._command:
            self._command()

    def set_dark(self, dark: bool) -> None:
        if self._dark != dark:
            self._dark = dark
            self._draw()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SubsetMakerApp(tk.Tk):
    """Main application window."""

    # ── Colour palettes ──────────────────────────────────────────────
    DARK_THEME: dict[str, str] = {
        "BG":        "#1e1e2e",
        "FG":        "#cdd6f4",
        "ACCENT":    "#89b4fa",
        "BTN_BG":    "#313244",
        "BTN_ACTIVE":"#45475a",
        "ENTRY_BG":  "#313244",
        "SELECT_BG": "#89b4fa",
        "SELECT_FG": "#1e1e2e",
        "GREEN":     "#a6e3a1",
        "RED":       "#f38ba8",
        "YELLOW":    "#f9e2af",
        "GREY":      "#585b70",
    }

    LIGHT_THEME: dict[str, str] = {
        "BG":        "#eff1f5",
        "FG":        "#4c4f69",
        "ACCENT":    "#1e66f5",
        "BTN_BG":    "#dce0e8",
        "BTN_ACTIVE":"#ccd0da",
        "ENTRY_BG":  "#dce0e8",
        "SELECT_BG": "#1e66f5",
        "SELECT_FG": "#eff1f5",
        "GREEN":     "#40a02b",
        "RED":       "#d20f39",
        "YELLOW":    "#df8e1d",
        "GREY":      "#9ca0b0",
    }

    def __init__(self):
        super().__init__()

        # Load saved config (sets self._dark_mode)
        self._load_config()

        # Initialise theme instance attributes from the chosen palette
        palette = self.DARK_THEME if self._dark_mode else self.LIGHT_THEME
        for k, v in palette.items():
            setattr(self, k, v)

        self.title("SubsetMaker - YOLO Dataset Subset Tool")
        self.configure(bg=self.BG)
        self.resizable(True, True)
        self.minsize(780, 620)

        self._dataset_dir = tk.StringVar()

        # Check-dataset tab state
        self._check_split = tk.StringVar(value="train")
        self._check_results: dict[str, list[str]] = {}

        # YAML-info tab state
        self._yaml_path = tk.StringVar()

        # JSON → YAML tab state
        self._json_path = tk.StringVar()
        self._json_output_dir = tk.StringVar()
        self._yaml_train_path = tk.StringVar(value="images/train")
        self._yaml_val_path = tk.StringVar(value="images/val")
        self._yaml_test_path = tk.StringVar(value="")

        # Renumber Labels tab state
        self._renum_yaml_path = tk.StringVar()
        self._renum_output_dir = tk.StringVar()
        self._renum_class_ids: list[int] = []
        self._renum_new_id_vars: dict[int, tk.StringVar] = {}
        self._swap_a = tk.StringVar()
        self._swap_b = tk.StringVar()

        # Split Dataset tab state
        self._splitds_dataset_dir = tk.StringVar()
        self._splitds_output_dir = tk.StringVar()
        self._splitds_source_split = tk.StringVar(value="")
        self._splitds_train_pct = tk.StringVar(value="80")
        self._splitds_seed = tk.StringVar(value="42")

        self._build_ui()
        self._ttk_style = ttk.Style(self)
        self._apply_ttk_theme()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        # ── Header bar ────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=self.ACCENT, height=4)
        hdr.grid(row=0, column=0, sticky="ew")

        # ── Title row (title left, theme toggle right) ─────────────────
        title_row = tk.Frame(self, bg=self.BG)
        title_row.grid(row=1, column=0, sticky="ew")
        title_row.columnconfigure(0, weight=1)

        tk.Label(
            title_row,
            text="SubsetMaker",
            font=("Helvetica", 23, "bold"),
            bg=self.BG,
            fg=self.ACCENT,
            pady=10,
        ).grid(row=0, column=0, sticky="w", padx=20)

        # Theme toggle
        toggle_frame = tk.Frame(title_row, bg=self.BG)
        toggle_frame.grid(row=0, column=1, sticky="e", padx=20)

        self._theme_icon_lbl = tk.Label(
            toggle_frame,
            text="Dark Mode" if self._dark_mode else "Light Mode",
            bg=self.BG,
            fg=self.FG,
            font=("Helvetica", 15),
        )
        self._theme_icon_lbl.pack(side="left", padx=(0, 6))

        self._theme_toggle = ThemeToggle(
            toggle_frame,
            dark=self._dark_mode,
            command=self._toggle_theme,
            bg=self.BG,
        )
        self._theme_toggle.pack(side="left")

        subtitle = tk.Label(
            self,
            text="YOLO dataset tools: create subsets, check integrity, and inspect YAML configs",
            font=("Helvetica", 15),
            bg=self.BG,
            fg=self.GREY,
        )
        subtitle.grid(row=2, column=0, sticky="w", padx=22, pady=(0, 8))

        # ── Notebook ──────────────────────────────────────────────────
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 8))

        # Tab 1: Check Dataset
        check_frame = tk.Frame(self._notebook, bg=self.BG)
        self._notebook.add(check_frame, text="Check Dataset")
        self._build_check_tab(check_frame)

        # Tab 2: YAML / JSON
        yaml_frame = tk.Frame(self._notebook, bg=self.BG)
        self._notebook.add(yaml_frame, text="YAML / JSON")
        self._build_yaml_tab(yaml_frame)

        # Tab 3: Renumber Labels
        renum_frame = tk.Frame(self._notebook, bg=self.BG)
        self._notebook.add(renum_frame, text="Renumber Labels")
        self._build_renum_tab(renum_frame)

        # Tab 4: Split Dataset
        splitds_frame = tk.Frame(self._notebook, bg=self.BG)
        self._notebook.add(splitds_frame, text="Split Dataset")
        self._build_splitds_tab(splitds_frame)

        # ── Status bar ────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready.")
        status = tk.Label(
            self,
            textvariable=self._status_var,
            bg=self.GREY,
            fg=self.BG,
            anchor="w",
            font=("Helvetica", 14),
            padx=8,
        )
        status.grid(row=4, column=0, sticky="ew")

    def _build_check_tab(self, parent: tk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        row = 0

        def section(text: str) -> None:
            nonlocal row
            tk.Label(
                parent,
                text=text,
                font=("Helvetica", 15, "bold"),
                bg=self.BG,
                fg=self.ACCENT,
                anchor="w",
            ).grid(row=row, column=0, sticky="w", pady=(12, 2), padx=4)
            row += 1

        section("1. Dataset")

        # Dataset folder
        folder_frame = tk.Frame(parent, bg=self.BG)
        folder_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        folder_frame.columnconfigure(0, weight=1)
        tk.Label(
            folder_frame, text="Dataset folder", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15), anchor="w", width=16,
        ).grid(row=0, column=0, sticky="w")
        tk.Entry(
            folder_frame, textvariable=self._dataset_dir,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 15),
        ).grid(row=0, column=1, sticky="ew", padx=4)
        folder_frame.columnconfigure(1, weight=1)
        tk.Button(
            folder_frame, text="...", command=self._browse_dataset,
            bg=self.BTN_BG, fg=self.FG, activebackground=self.BTN_ACTIVE,
            relief="flat", padx=8, font=("Helvetica", 15),
        ).grid(row=0, column=2)
        row += 1

        row += 1

        # Check button
        tk.Button(
            parent, text="Check Dataset",
            command=self._check_dataset_action,
            bg=self.ACCENT, fg=self.BG,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 15, "bold"),
            relief="flat", pady=4,
        ).grid(row=row, column=0, sticky="ew", padx=4, pady=(6, 2))
        row += 1

        section("2. Results")

        # Results text area
        results_frame = tk.Frame(parent, bg=self.BG)
        results_frame.grid(row=row, column=0, sticky="nsew", padx=4, pady=(0, 4))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)

        results_scroll = ttk.Scrollbar(results_frame, orient="vertical")
        self._check_text = tk.Text(
            results_frame,
            bg=self.ENTRY_BG, fg=self.FG,
            font=("Courier", 14),
            state="disabled",
            relief="flat",
            yscrollcommand=results_scroll.set,
            height=12,
        )
        results_scroll.config(command=self._check_text.yview)
        results_scroll.pack(side="right", fill="y")
        self._check_text.pack(side="left", fill="both", expand=True)
        row += 1

        # Fix buttons
        fix_frame = tk.Frame(parent, bg=self.BG)
        fix_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=(4, 8))

        self._fix_missing_btn = tk.Button(
            fix_frame,
            text="Create empty labels for unlabeled images",
            command=self._fix_missing_labels_action,
            bg=self.BTN_BG, fg=self.FG,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 14),
            relief="flat", pady=3,
            state="disabled",
        )
        self._fix_missing_btn.pack(side="left", padx=(0, 4))

        self._fix_orphan_btn = tk.Button(
            fix_frame,
            text="Delete orphan labels",
            command=self._fix_orphan_labels_action,
            bg=self.BTN_BG, fg=self.RED,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 14),
            relief="flat", pady=3,
            state="disabled",
        )
        self._fix_orphan_btn.pack(side="left")

    def _build_yaml_tab(self, parent: tk.Frame) -> None:
        """Build the merged YAML / JSON tab (YAML Info + JSON → YAML)."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        paned.grid(row=0, column=0, sticky="nsew")

        # ── Top pane: Load YAML ────────────────────────────────────────
        top = tk.Frame(paned, bg=self.BG)
        top.columnconfigure(0, weight=1)
        top.rowconfigure(3, weight=1)
        paned.add(top, weight=1)

        trow = 0

        def tsec(text: str) -> None:
            nonlocal trow
            tk.Label(
                top, text=text,
                font=("Helvetica", 15, "bold"),
                bg=self.BG, fg=self.ACCENT, anchor="w",
            ).grid(row=trow, column=0, sticky="w", pady=(12, 2), padx=4)
            trow += 1

        tsec("1. Load YAML File")

        file_frame = tk.Frame(top, bg=self.BG)
        file_frame.grid(row=trow, column=0, sticky="ew", padx=4, pady=2)
        file_frame.columnconfigure(0, weight=1)
        tk.Entry(
            file_frame, textvariable=self._yaml_path,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 15),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        tk.Button(
            file_frame, text="...", command=self._browse_yaml,
            bg=self.BTN_BG, fg=self.FG, activebackground=self.BTN_ACTIVE,
            relief="flat", padx=8, font=("Helvetica", 15),
        ).grid(row=0, column=1)
        trow += 1

        tk.Button(
            top, text="Load YAML",
            command=self._load_yaml_action,
            bg=self.ACCENT, fg=self.BG,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 15, "bold"),
            relief="flat", pady=4,
        ).grid(row=trow, column=0, sticky="ew", padx=4, pady=(6, 2))
        trow += 1

        tsec("  Class Names")
        top.rowconfigure(trow, weight=1)

        yaml_results_frame = tk.Frame(top, bg=self.BG)
        yaml_results_frame.grid(row=trow, column=0, sticky="nsew", padx=4, pady=(0, 8))
        yaml_results_frame.columnconfigure(0, weight=1)
        yaml_results_frame.rowconfigure(0, weight=1)

        yaml_scroll = ttk.Scrollbar(yaml_results_frame, orient="vertical")
        self._yaml_text = tk.Text(
            yaml_results_frame,
            bg=self.ENTRY_BG, fg=self.FG,
            font=("Courier", 15),
            state="disabled",
            relief="flat",
            yscrollcommand=yaml_scroll.set,
        )
        yaml_scroll.config(command=self._yaml_text.yview)
        yaml_scroll.pack(side="right", fill="y")
        self._yaml_text.pack(side="left", fill="both", expand=True)

        # ── Bottom pane: Generate data.yaml from JSON ──────────────────
        bot = tk.Frame(paned, bg=self.BG)
        bot.columnconfigure(0, weight=1)
        paned.add(bot, weight=1)

        brow = 0

        def bsec(text: str) -> None:
            nonlocal brow
            tk.Label(
                bot, text=text,
                font=("Helvetica", 15, "bold"),
                bg=self.BG, fg=self.ACCENT, anchor="w",
            ).grid(row=brow, column=0, sticky="w", pady=(12, 2), padx=4)
            brow += 1

        bsec("2. Generate data.yaml from JSON")

        # JSON file row
        json_frame = tk.Frame(bot, bg=self.BG)
        json_frame.grid(row=brow, column=0, sticky="ew", padx=4, pady=2)
        json_frame.columnconfigure(1, weight=1)
        tk.Label(
            json_frame, text="JSON file", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15), anchor="w", width=13,
        ).grid(row=0, column=0, sticky="w")
        tk.Entry(
            json_frame, textvariable=self._json_path,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 15),
        ).grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(
            json_frame, text="...", command=self._browse_json,
            bg=self.BTN_BG, fg=self.FG, activebackground=self.BTN_ACTIVE,
            relief="flat", padx=8, font=("Helvetica", 15),
        ).grid(row=0, column=2)
        brow += 1

        # Output folder row
        out_frame = tk.Frame(bot, bg=self.BG)
        out_frame.grid(row=brow, column=0, sticky="ew", padx=4, pady=2)
        out_frame.columnconfigure(1, weight=1)
        tk.Label(
            out_frame, text="Output folder", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15), anchor="w", width=13,
        ).grid(row=0, column=0, sticky="w")
        tk.Entry(
            out_frame, textvariable=self._json_output_dir,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 15),
        ).grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(
            out_frame, text="...", command=self._browse_json_output,
            bg=self.BTN_BG, fg=self.FG, activebackground=self.BTN_ACTIVE,
            relief="flat", padx=8, font=("Helvetica", 15),
        ).grid(row=0, column=2)
        brow += 1

        # Split paths row (train / val / test)
        paths_frame = tk.Frame(bot, bg=self.BG)
        paths_frame.grid(row=brow, column=0, sticky="ew", padx=4, pady=2)
        paths_frame.columnconfigure(1, weight=1)
        paths_frame.columnconfigure(3, weight=1)
        paths_frame.columnconfigure(5, weight=1)
        tk.Label(
            paths_frame, text="train path", bg=self.BG, fg=self.FG,
            font=("Helvetica", 14), anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, 2))
        tk.Entry(
            paths_frame, textvariable=self._yaml_train_path,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 14),
        ).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        tk.Label(
            paths_frame, text="val path", bg=self.BG, fg=self.FG,
            font=("Helvetica", 14), anchor="w",
        ).grid(row=0, column=2, sticky="w", padx=(0, 2))
        tk.Entry(
            paths_frame, textvariable=self._yaml_val_path,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 14),
        ).grid(row=0, column=3, sticky="ew", padx=(0, 8))
        tk.Label(
            paths_frame, text="test path", bg=self.BG, fg=self.FG,
            font=("Helvetica", 14), anchor="w",
        ).grid(row=0, column=4, sticky="w", padx=(0, 2))
        tk.Entry(
            paths_frame, textvariable=self._yaml_test_path,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 14),
        ).grid(row=0, column=5, sticky="ew")
        brow += 1

        tk.Button(
            bot, text="Generate data.yaml",
            command=self._generate_yaml_from_json,
            bg=self.ACCENT, fg=self.BG,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 15, "bold"),
            relief="flat", pady=4,
        ).grid(row=brow, column=0, sticky="ew", padx=4, pady=(6, 2))
        brow += 1

        bsec("  Result")
        bot.rowconfigure(brow, weight=1)

        result_frame = tk.Frame(bot, bg=self.BG)
        result_frame.grid(row=brow, column=0, sticky="nsew", padx=4, pady=(0, 8))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        result_scroll = ttk.Scrollbar(result_frame, orient="vertical")
        self._json_yaml_text = tk.Text(
            result_frame,
            bg=self.ENTRY_BG, fg=self.FG,
            font=("Courier", 15),
            state="disabled",
            relief="flat",
            yscrollcommand=result_scroll.set,
        )
        result_scroll.config(command=self._json_yaml_text.yview)
        result_scroll.pack(side="right", fill="y")
        self._json_yaml_text.pack(side="left", fill="both", expand=True)

    def _build_renum_tab(self, parent: tk.Frame) -> None:
        """Build the Renumber Labels tab UI."""
        parent.columnconfigure(0, weight=1)

        row = 0

        def section(text: str) -> None:
            nonlocal row
            tk.Label(
                parent,
                text=text,
                font=("Helvetica", 15, "bold"),
                bg=self.BG,
                fg=self.ACCENT,
                anchor="w",
            ).grid(row=row, column=0, sticky="w", pady=(12, 2), padx=4)
            row += 1

        section("1. Dataset")

        # Dataset folder
        folder_frame = tk.Frame(parent, bg=self.BG)
        folder_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        folder_frame.columnconfigure(1, weight=1)
        tk.Label(
            folder_frame, text="Dataset folder", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15), anchor="w", width=14,
        ).grid(row=0, column=0, sticky="w")
        tk.Entry(
            folder_frame, textvariable=self._dataset_dir,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 15),
        ).grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(
            folder_frame, text="...", command=self._browse_dataset,
            bg=self.BTN_BG, fg=self.FG, activebackground=self.BTN_ACTIVE,
            relief="flat", padx=8, font=("Helvetica", 15),
        ).grid(row=0, column=2)
        row += 1

        # YAML chooser
        yaml_frame = tk.Frame(parent, bg=self.BG)
        yaml_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        yaml_frame.columnconfigure(1, weight=1)
        tk.Label(
            yaml_frame, text="YAML file", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15), anchor="w", width=14,
        ).grid(row=0, column=0, sticky="w")
        tk.Entry(
            yaml_frame, textvariable=self._renum_yaml_path,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 15),
        ).grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(
            yaml_frame, text="...", command=self._browse_renum_yaml,
            bg=self.BTN_BG, fg=self.FG, activebackground=self.BTN_ACTIVE,
            relief="flat", padx=8, font=("Helvetica", 15),
        ).grid(row=0, column=2)
        row += 1

        # Output folder
        out_frame = tk.Frame(parent, bg=self.BG)
        out_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        out_frame.columnconfigure(1, weight=1)
        tk.Label(
            out_frame, text="Output folder", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15), anchor="w", width=14,
        ).grid(row=0, column=0, sticky="w")
        tk.Entry(
            out_frame, textvariable=self._renum_output_dir,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 15),
        ).grid(row=0, column=1, sticky="ew", padx=4)
        tk.Button(
            out_frame, text="...", command=self._browse_renum_output,
            bg=self.BTN_BG, fg=self.FG, activebackground=self.BTN_ACTIVE,
            relief="flat", padx=8, font=("Helvetica", 15),
        ).grid(row=0, column=2)
        row += 1

        tk.Label(
            parent,
            text="Leave blank to modify label files in-place.",
            bg=self.BG, fg=self.GREY, font=("Helvetica", 13), anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=4)
        row += 1

        # Scan button
        tk.Button(
            parent, text="Scan Labels",
            command=self._renum_scan_action,
            bg=self.ACCENT, fg=self.BG,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 15, "bold"),
            relief="flat", pady=4,
        ).grid(row=row, column=0, sticky="ew", padx=4, pady=(6, 2))
        row += 1

        section("2. Renumber")

        # Controls frame that hosts swap_panel or free_panel
        controls_frame = tk.Frame(parent, bg=self.BG)
        controls_frame.grid(row=row, column=0, sticky="nsew", padx=4, pady=2)
        controls_frame.columnconfigure(0, weight=1)
        row += 1

        # ── Swap panel (default) ──────────────────────────────────────
        self._swap_panel = tk.Frame(controls_frame, bg=self.BG)
        self._swap_panel.grid(row=0, column=0, sticky="ew")
        self._swap_panel.columnconfigure(1, weight=1)
        self._swap_panel.columnconfigure(3, weight=1)

        tk.Label(
            self._swap_panel, text="Swap class:", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15),
        ).grid(row=0, column=0, sticky="w", padx=4, pady=4)

        self._swap_a_combo = ttk.Combobox(
            self._swap_panel, textvariable=self._swap_a,
            state="readonly", font=("Helvetica", 15), width=10,
        )
        self._swap_a_combo.grid(row=0, column=1, sticky="ew", padx=4)

        tk.Label(
            self._swap_panel, text="<->", bg=self.BG, fg=self.FG,
            font=("Helvetica", 19),
        ).grid(row=0, column=2, padx=4)

        self._swap_b_combo = ttk.Combobox(
            self._swap_panel, textvariable=self._swap_b,
            state="readonly", font=("Helvetica", 15), width=10,
        )
        self._swap_b_combo.grid(row=0, column=3, sticky="ew", padx=4)

        swap_btns = tk.Frame(self._swap_panel, bg=self.BG)
        swap_btns.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 4))
        self._swap_btn = tk.Button(
            swap_btns, text="Swap Labels",
            command=self._renum_swap_action,
            bg=self.GREEN, fg=self.BG,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 15, "bold"),
            relief="flat", pady=4,
        )
        self._swap_btn.pack(side="left", padx=(0, 8))
        tk.Button(
            swap_btns, text="Free Renumber...",
            command=self._show_free_panel,
            bg=self.BTN_BG, fg=self.FG,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 14),
            relief="flat", pady=4,
        ).pack(side="left")

        # ── Free renumber panel (shown after clicking "Free Renumber") ──
        self._free_panel = tk.Frame(controls_frame, bg=self.BG)
        self._free_panel.columnconfigure(0, weight=1)

        tk.Label(
            self._free_panel,
            text=(
                "Each class is mapped to a new ID. "
                "Assigning two classes to the same ID merges them."
            ),
            bg=self.BG, fg=self.RED,
            font=("Helvetica", 14, "italic"),
            anchor="w", wraplength=540,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        # Scrollable table
        table_outer = tk.Frame(self._free_panel, bg=self.BG)
        table_outer.grid(row=1, column=0, sticky="nsew")
        table_outer.columnconfigure(0, weight=1)

        free_scroll = ttk.Scrollbar(table_outer, orient="vertical")
        self._free_canvas = tk.Canvas(
            table_outer, bg=self.ENTRY_BG,
            highlightthickness=0, height=130,
            yscrollcommand=free_scroll.set,
        )
        free_scroll.config(command=self._free_canvas.yview)
        free_scroll.pack(side="right", fill="y")
        self._free_canvas.pack(side="left", fill="both", expand=True)

        self._free_inner = tk.Frame(self._free_canvas, bg=self.ENTRY_BG)
        self._free_canvas_win = self._free_canvas.create_window(
            (0, 0), window=self._free_inner, anchor="nw",
        )
        self._free_inner.bind(
            "<Configure>",
            lambda e: self._free_canvas.configure(
                scrollregion=self._free_canvas.bbox("all")
            ),
        )
        self._free_canvas.bind(
            "<Configure>",
            lambda e: self._free_canvas.itemconfig(
                self._free_canvas_win, width=e.width
            ),
        )

        free_btns = tk.Frame(self._free_panel, bg=self.BG)
        free_btns.grid(row=2, column=0, sticky="ew", pady=(6, 4))
        self._apply_free_btn = tk.Button(
            free_btns, text="Apply Renumber",
            command=self._renum_apply_action,
            bg=self.GREEN, fg=self.BG,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 15, "bold"),
            relief="flat", pady=4,
        )
        self._apply_free_btn.pack(side="left", padx=(0, 8))
        tk.Button(
            free_btns, text="Back to Swap",
            command=self._show_swap_panel,
            bg=self.BTN_BG, fg=self.FG,
            activebackground=self.BTN_ACTIVE,
            font=("Helvetica", 14),
            relief="flat", pady=4,
        ).pack(side="left")

        section("3. Results")

        results_frame = tk.Frame(parent, bg=self.BG)
        results_frame.grid(row=row, column=0, sticky="nsew", padx=4, pady=(0, 8))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)

        results_scroll = ttk.Scrollbar(results_frame, orient="vertical")
        self._renum_text = tk.Text(
            results_frame,
            bg=self.ENTRY_BG, fg=self.FG,
            font=("Courier", 14),
            state="disabled",
            relief="flat",
            yscrollcommand=results_scroll.set,
            height=6,
        )
        results_scroll.config(command=self._renum_text.yview)
        results_scroll.pack(side="right", fill="y")
        self._renum_text.pack(side="left", fill="both", expand=True)

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------

    _CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".subsetmaker", "config.json")

    def _load_config(self) -> None:
        """Load saved theme preference from config file. Defaults to dark mode."""
        self._dark_mode = True
        try:
            with open(self._CONFIG_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
            self._dark_mode = data.get("theme", "dark") != "light"
        except (OSError, json.JSONDecodeError):
            pass

    def _save_config(self) -> None:
        """Persist the current theme preference to config file."""
        try:
            os.makedirs(os.path.dirname(self._CONFIG_PATH), exist_ok=True)
            with open(self._CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump({"theme": "dark" if self._dark_mode else "light"}, fh)
        except OSError:
            pass

    def _toggle_theme(self) -> None:
        """Switch between dark and light themes."""
        old_theme = self.DARK_THEME if self._dark_mode else self.LIGHT_THEME
        new_theme = self.LIGHT_THEME if self._dark_mode else self.DARK_THEME
        self._dark_mode = not self._dark_mode

        # Build mapping from old hex values to new hex values
        color_map = {v: new_theme[k] for k, v in old_theme.items()}

        # Update instance colour attributes
        for k, v in new_theme.items():
            setattr(self, k, v)

        # Retheme all widgets and apply TTK styles
        self._retheme_widgets(self, color_map)
        self._apply_ttk_theme()

        # Update the toggle icon label and the toggle widget itself
        self._theme_icon_lbl.configure(
            text="Dark Mode" if self._dark_mode else "Light Mode"
        )
        self._theme_toggle.set_dark(self._dark_mode)
        self._save_config()

    def _retheme_widgets(self, widget: tk.Widget, color_map: dict) -> None:
        """Recursively update every widget's colour options using *color_map*."""
        for opt in (
            "bg", "fg",
            "activebackground", "activeforeground",
            "selectcolor", "insertbackground",
            "highlightbackground", "disabledforeground",
        ):
            try:
                current = str(widget.cget(opt))
                if current in color_map:
                    widget.configure(**{opt: color_map[current]})
            except tk.TclError:
                pass
        for child in widget.winfo_children():
            self._retheme_widgets(child, color_map)

    def _apply_ttk_theme(self) -> None:
        """Configure TTK widget styles to match the current colour palette."""
        style = self._ttk_style
        style.configure("TNotebook", background=self.BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=self.BTN_BG,
            foreground=self.FG,
            padding=[10, 4],
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.BG)],
            foreground=[("selected", self.ACCENT)],
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.ENTRY_BG,
            background=self.BTN_BG,
            foreground=self.FG,
            selectbackground=self.SELECT_BG,
            selectforeground=self.SELECT_FG,
            arrowcolor=self.FG,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.ENTRY_BG)],
            foreground=[("readonly", self.FG)],
            selectbackground=[("readonly", self.ENTRY_BG)],
            selectforeground=[("readonly", self.FG)],
        )
        style.configure(
            "Vertical.TScrollbar",
            background=self.BTN_BG,
            troughcolor=self.BG,
            arrowcolor=self.FG,
        )
        style.configure(
            "TProgressbar",
            background=self.ACCENT,
            troughcolor=self.ENTRY_BG,
        )

    def _browse_dataset(self) -> None:
        path = filedialog.askdirectory(title="Select dataset root folder")
        if path:
            self._dataset_dir.set(path)

    def _browse_yaml(self) -> None:
        path = filedialog.askopenfilename(
            title="Select YAML file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self._yaml_path.set(path)

    def _browse_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Select JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._json_path.set(path)

    def _browse_json_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder for data.yaml")
        if path:
            self._json_output_dir.set(path)

    def _browse_renum_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder for renumbered labels")
        if path:
            self._renum_output_dir.set(path)

    def _browse_renum_yaml(self) -> None:
        path = filedialog.askopenfilename(
            title="Select YAML file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self._renum_yaml_path.set(path)
            self._renum_show_yaml_classes()

    def _renum_load_class_names(self) -> list[str]:
        """Load class names from the user-chosen YAML, or return empty list."""
        yaml_path = self._renum_yaml_path.get().strip()
        if not yaml_path or not os.path.isfile(yaml_path):
            return []
        try:
            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = parse_yaml(fh.read())
        except Exception:
            return []
        names = data.get("names", [])
        if isinstance(names, list):
            return [str(n) for n in names]
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=str)]
        return []

    def _renum_show_yaml_classes(self) -> None:
        """Display class names from the chosen YAML in the output window."""
        class_names = self._renum_load_class_names()
        self._renum_text.configure(state="normal")
        self._renum_text.delete("1.0", "end")
        if class_names:
            lines = ["Class names (from YAML):"]
            lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
            self._renum_text.insert("end", "\n".join(lines) + "\n")
        else:
            yaml_path = self._renum_yaml_path.get().strip()
            if yaml_path:
                self._renum_text.insert("end", "No class names found in the selected YAML.\n")
        self._renum_text.configure(state="disabled")

    def _browse_splitds_dataset(self) -> None:
        path = filedialog.askdirectory(title="Select source dataset folder")
        if path:
            self._splitds_dataset_dir.set(path)

    def _browse_splitds_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder for split dataset")
        if path:
            self._splitds_output_dir.set(path)

    # ------------------------------------------------------------------
    # Renumber Labels helpers
    # ------------------------------------------------------------------

    def _show_swap_panel(self) -> None:
        self._free_panel.grid_remove()
        self._swap_panel.grid(row=0, column=0, sticky="ew")

    def _show_free_panel(self) -> None:
        if not self._renum_class_ids:
            messagebox.showwarning(
                "No labels scanned",
                "Click 'Scan Labels' first to discover the class IDs.",
            )
            return
        self._swap_panel.grid_remove()
        self._free_panel.grid(row=0, column=0, sticky="nsew")

    def _renum_scan_action(self) -> None:
        dataset_dir = self._dataset_dir.get().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            messagebox.showerror("Error", "Please select a valid dataset folder.")
            return
        labels_dir = os.path.join(dataset_dir, "labels")
        if not os.path.isdir(labels_dir):
            messagebox.showerror(
                "Error", f"Labels directory not found:\n{labels_dir}"
            )
            return

        # Collect unique class IDs from all label files
        class_ids: set[int] = set()
        for fname in os.listdir(labels_dir):
            if not fname.lower().endswith(LABEL_EXTENSION):
                continue
            ids = parse_label_file(os.path.join(labels_dir, fname))
            class_ids.update(ids)

        if not class_ids:
            messagebox.showinfo("No labels", "No class labels found in the labels directory.")
            return

        self._renum_class_ids = sorted(class_ids)

        # Load class names from the chosen YAML (if any)
        class_names = self._renum_load_class_names()

        def _label_str(cid: int) -> str:
            if cid < len(class_names):
                return f"{cid}  ({class_names[cid]})"
            return str(cid)

        combo_values = [_label_str(cid) for cid in self._renum_class_ids]

        # Update swap comboboxes
        self._swap_a_combo.configure(values=combo_values)
        self._swap_b_combo.configure(values=combo_values)
        if combo_values:
            self._swap_a.set(combo_values[0])
            self._swap_b.set(combo_values[-1] if len(combo_values) > 1 else combo_values[0])

        # Rebuild free-renumber table
        for widget in self._free_inner.winfo_children():
            widget.destroy()
        self._renum_new_id_vars.clear()

        header = tk.Frame(self._free_inner, bg=self.ENTRY_BG)
        header.pack(fill="x", padx=4, pady=(4, 2))
        tk.Label(
            header, text="Current ID", bg=self.ENTRY_BG, fg=self.GREY,
            font=("Helvetica", 14, "bold"), width=20, anchor="w",
        ).pack(side="left")
        tk.Label(
            header, text="New ID", bg=self.ENTRY_BG, fg=self.GREY,
            font=("Helvetica", 14, "bold"), width=10, anchor="w",
        ).pack(side="left")

        for cid in self._renum_class_ids:
            row_frame = tk.Frame(self._free_inner, bg=self.ENTRY_BG)
            row_frame.pack(fill="x", padx=4, pady=1)
            tk.Label(
                row_frame, text=_label_str(cid), bg=self.ENTRY_BG, fg=self.FG,
                font=("Helvetica", 15), width=20, anchor="w",
            ).pack(side="left")
            var = tk.StringVar(value=str(cid))
            self._renum_new_id_vars[cid] = var
            tk.Entry(
                row_frame, textvariable=var,
                bg=self.BG, fg=self.FG, insertbackground=self.FG,
                relief="flat", bd=3, font=("Helvetica", 15), width=8,
            ).pack(side="left", padx=4)

        self._free_canvas.configure(scrollregion=self._free_canvas.bbox("all"))

        # Show result in text area
        lines = [
            f"Labels dir : {labels_dir}",
            "",
            f"Found {len(self._renum_class_ids)} class ID(s) : "
            + ", ".join(str(c) for c in self._renum_class_ids),
        ]
        if class_names:
            lines += ["", "Class names (from YAML):"]
            lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
        self._renum_text.configure(state="normal")
        self._renum_text.delete("1.0", "end")
        self._renum_text.insert("end", "\n".join(lines) + "\n")
        self._renum_text.configure(state="disabled")
        self._set_status(
            f"Scan complete: {len(self._renum_class_ids)} class ID(s) found."
        )

    def _renum_labels_dir(self) -> str | None:
        """Return labels_dir if valid, else show error and return None."""
        dataset_dir = self._dataset_dir.get().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            messagebox.showerror("Error", "Please select a valid dataset folder.")
            return None
        labels_dir = os.path.join(dataset_dir, "labels")
        if not os.path.isdir(labels_dir):
            messagebox.showerror("Error", f"Labels directory not found:\n{labels_dir}")
            return None
        return labels_dir

    def _renum_output_dir_val(self) -> str | None:
        """Return the output dir string (may be empty for in-place)."""
        return self._renum_output_dir.get().strip() or None

    def _renum_swap_action(self) -> None:
        if not self._renum_class_ids:
            messagebox.showwarning("No labels", "Scan labels first.")
            return
        labels_dir = self._renum_labels_dir()
        if labels_dir is None:
            return

        def _combo_to_id(s: str) -> int:
            return int(s.split()[0])

        try:
            a = _combo_to_id(self._swap_a.get())
            b = _combo_to_id(self._swap_b.get())
        except (ValueError, IndexError):
            messagebox.showerror("Error", "Please select two classes to swap.")
            return

        if a == b:
            messagebox.showwarning("Same class", "The two classes are identical - nothing to swap.")
            return

        output_dir = self._renum_output_dir_val()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        id_remap = {a: b, b: a}
        written = remap_labels(labels_dir, id_remap, output_dir)

        yaml_path = self._renum_yaml_path.get().strip()
        if yaml_path and os.path.isfile(yaml_path):
            yaml_out = os.path.join(output_dir, "data.yaml") if output_dir else yaml_path
            remap_yaml_classes(yaml_path, id_remap, yaml_out)

        msg = (
            f"Swapped class {a} <-> {b} in {written} label file(s).\n"
            f"{'Output: ' + output_dir if output_dir else 'Modified in-place.'}"
        )
        self._renum_text.configure(state="normal")
        self._renum_text.insert("end", "\n" + msg + "\n")
        self._renum_text.see("end")
        self._renum_text.configure(state="disabled")
        self._set_status(f"Swap complete: {written} file(s) updated.")
        messagebox.showinfo("Swap complete", msg)

    def _renum_apply_action(self) -> None:
        if not self._renum_class_ids:
            messagebox.showwarning("No labels", "Scan labels first.")
            return
        labels_dir = self._renum_labels_dir()
        if labels_dir is None:
            return

        # Parse the new ID entries
        id_remap: dict[int, int] = {}
        for old_id, var in self._renum_new_id_vars.items():
            raw = var.get().strip()
            try:
                new_id = int(raw)
                if new_id < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid ID",
                    f"New ID for class {old_id} must be a non-negative integer (got: {raw!r}).",
                )
                return
            id_remap[old_id] = new_id

        # Warn if any merge will occur
        targets = list(id_remap.values())
        if len(targets) != len(set(targets)):
            if not messagebox.askyesno(
                "Confirm merge",
                "Some classes map to the same new ID - this will merge them.\n\n"
                "The original class IDs that map to that ID will be lost.\n"
                "Continue?",
            ):
                return

        # Warn for in-place edit
        output_dir = self._renum_output_dir_val()
        if output_dir is None:
            if not messagebox.askyesno(
                "Confirm in-place",
                "No output folder selected - label files will be modified in-place.\n"
                "This cannot be undone.\n\nContinue?",
            ):
                return
        else:
            os.makedirs(output_dir, exist_ok=True)

        written = remap_labels(labels_dir, id_remap, output_dir)

        yaml_path = self._renum_yaml_path.get().strip()
        if yaml_path and os.path.isfile(yaml_path):
            yaml_out = os.path.join(output_dir, "data.yaml") if output_dir else yaml_path
            remap_yaml_classes(yaml_path, id_remap, yaml_out)

        mapping_lines = [f"  {old} -> {new}" for old, new in sorted(id_remap.items())]
        msg = (
            f"Renumber complete: {written} file(s) updated.\n"
            + "\n".join(mapping_lines)
            + f"\n{'Output: ' + output_dir if output_dir else 'Modified in-place.'}"
        )
        self._renum_text.configure(state="normal")
        self._renum_text.insert("end", "\n" + msg + "\n")
        self._renum_text.see("end")
        self._renum_text.configure(state="disabled")
        self._set_status(f"Renumber complete: {written} file(s) updated.")
        messagebox.showinfo("Renumber complete", msg)

    def _set_status(self, message: str) -> None:
        self._status_var.set(message)

    # ------------------------------------------------------------------
    # Dataset checking
    # ------------------------------------------------------------------

    def _check_dataset_action(self) -> None:
        dataset_dir = self._dataset_dir.get().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            messagebox.showerror("Error", "Please select a valid dataset folder.")
            return
        split = ""
        self._set_status("Checking dataset...")

        results = check_dataset(dataset_dir, split)
        self._check_results = results

        missing = results["missing_labels"]
        orphans = results["orphan_labels"]

        lines = []
        lines.append(f"Dataset : {dataset_dir}  [recursive]")
        lines.append("")
        lines.append(
            f"Images without a label file : {len(missing)}"
            + (" [OK]" if not missing else " [!]")
        )
        for p in missing:
            lines.append(f"    {p}")
        lines.append("")
        lines.append(
            f"Label files without an image : {len(orphans)}"
            + (" [OK]" if not orphans else " [!]")
        )
        for p in orphans:
            lines.append(f"    {p}")

        self._check_text.configure(state="normal")
        self._check_text.delete("1.0", "end")
        self._check_text.insert("end", "\n".join(lines) + "\n")
        self._check_text.configure(state="disabled")

        self._fix_missing_btn.configure(
            state="normal" if missing else "disabled"
        )
        self._fix_orphan_btn.configure(
            state="normal" if orphans else "disabled"
        )
        self._set_status(
            f"Check complete: {len(missing)} missing label(s), {len(orphans)} orphan(s)."
        )

    def _fix_missing_labels_action(self) -> None:
        missing = self._check_results.get("missing_labels", [])
        if not missing:
            return
        dataset_dir = self._dataset_dir.get().strip()
        if messagebox.askyesno(
            "Confirm",
            f"Create empty label files for {len(missing)} unlabeled image(s)?",
        ):
            count = fix_missing_labels(missing, dataset_dir, "")
            self._set_status(f"Created {count} empty label file(s).")
            self._check_dataset_action()

    def _fix_orphan_labels_action(self) -> None:
        orphans = self._check_results.get("orphan_labels", [])
        if not orphans:
            return
        if messagebox.askyesno(
            "Confirm",
            f"Permanently delete {len(orphans)} orphan label file(s)?\n"
            "This cannot be undone.",
        ):
            count = fix_orphan_labels(orphans)
            self._set_status(f"Deleted {count} orphan label file(s).")
            self._check_dataset_action()

    # ------------------------------------------------------------------
    # YAML info
    # ------------------------------------------------------------------

    def _load_yaml_action(self) -> None:
        yaml_path = self._yaml_path.get().strip()
        if not yaml_path or not os.path.isfile(yaml_path):
            messagebox.showerror("Error", "Please select a valid YAML file.")
            return

        info = load_yaml_file(yaml_path)
        names = info["names"]
        nc = info["nc"]

        lines = []
        lines.append(f"File : {yaml_path}")
        lines.append("")
        lines.append(f"Number of classes (nc) : {nc}")
        lines.append("")
        lines.append(f"Class names ({len(names)}) :")
        for idx, name in enumerate(names):
            lines.append(f"  [{idx:>4}]  {name}")

        self._yaml_text.configure(state="normal")
        self._yaml_text.delete("1.0", "end")
        self._yaml_text.insert("end", "\n".join(lines) + "\n")
        self._yaml_text.configure(state="disabled")
        self._set_status(f"YAML loaded: {nc} class(es) found.")

    # ------------------------------------------------------------------
    # JSON → YAML generation
    # ------------------------------------------------------------------

    def _generate_yaml_from_json(self) -> None:
        json_path = self._json_path.get().strip()
        output_dir = self._json_output_dir.get().strip()

        if not json_path or not os.path.isfile(json_path):
            messagebox.showerror("Error", "Please select a valid JSON file.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        try:
            class_map = yaml_from_json(json_path)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            messagebox.showerror("Error", f"Could not read JSON:\n{exc}")
            return

        try:
            # Pass explicit split paths from the UI fields.
            _write_data_yaml(
                output_dir, output_dir, class_map,
                train_path=self._yaml_train_path.get().strip() or None,
                val_path=self._yaml_val_path.get().strip() or None,
                test_path=self._yaml_test_path.get().strip() or None,
            )
        except OSError as exc:
            messagebox.showerror("Error", f"Could not write data.yaml:\n{exc}")
            return

        nc = len(class_map)
        sorted_ids = sorted(class_map)
        lines = []
        lines.append(f"Source : {json_path}")
        lines.append(f"Output : {os.path.join(output_dir, 'data.yaml')}")
        lines.append("")
        lines.append(f"Number of classes (nc) : {nc}")
        lines.append("")
        lines.append(f"Class names ({nc}) :")
        for cid in sorted_ids:
            lines.append(f"  [{cid:>4}]  {class_map[cid]}")

        self._json_yaml_text.configure(state="normal")
        self._json_yaml_text.delete("1.0", "end")
        self._json_yaml_text.insert("end", "\n".join(lines) + "\n")
        self._json_yaml_text.configure(state="disabled")
        self._set_status(f"data.yaml generated: {nc} class(es) -> {output_dir}")
        messagebox.showinfo(
            "Success",
            f"data.yaml generated successfully!\n\nOutput: {os.path.join(output_dir, 'data.yaml')}",
        )

    # ------------------------------------------------------------------
    # Split Dataset tab
    # ------------------------------------------------------------------

    def _build_splitds_tab(self, parent: tk.Frame) -> None:
        """Build the Split Dataset tab UI."""
        parent.columnconfigure(0, weight=1)
        row = 0

        def section(text: str) -> None:
            nonlocal row
            tk.Label(
                parent,
                text=text,
                font=("Helvetica", 15, "bold"),
                bg=self.BG,
                fg=self.ACCENT,
                anchor="w",
            ).grid(row=row, column=0, sticky="w", pady=(12, 2), padx=4)
            row += 1

        def folder_row(label: str, var: tk.StringVar, browse_cmd) -> None:
            nonlocal row
            frame = tk.Frame(parent, bg=self.BG)
            frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
            frame.columnconfigure(1, weight=1)
            tk.Label(
                frame, text=label, bg=self.BG, fg=self.FG,
                font=("Helvetica", 15), anchor="w", width=14,
            ).grid(row=0, column=0, sticky="w")
            tk.Entry(
                frame, textvariable=var,
                bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
                relief="flat", bd=4, font=("Helvetica", 15),
            ).grid(row=0, column=1, sticky="ew", padx=4)
            tk.Button(
                frame, text="...", command=browse_cmd,
                bg=self.BTN_BG, fg=self.FG, activebackground=self.BTN_ACTIVE,
                relief="flat", padx=8, font=("Helvetica", 15),
            ).grid(row=0, column=2)
            row += 1

        section("1. Source Dataset")
        folder_row("Dataset folder", self._splitds_dataset_dir, self._browse_splitds_dataset)
        folder_row("Output folder", self._splitds_output_dir, self._browse_splitds_output)

        # Source split selector
        split_frame = tk.Frame(parent, bg=self.BG)
        split_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        split_frame.columnconfigure(1, weight=1)
        tk.Label(
            split_frame, text="Source split", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15), anchor="w", width=14,
        ).grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            split_frame, textvariable=self._splitds_source_split,
            state="readonly", font=("Helvetica", 15),
            values=["", "train", "val", "test"],
        ).grid(row=0, column=1, sticky="ew", padx=4)
        row += 1

        tk.Label(
            parent,
            text="Choose the source split to divide (leave blank for flat layout).",
            bg=self.BG, fg=self.GREY, font=("Helvetica", 13), anchor="w",
        ).grid(row=row, column=0, sticky="w", padx=4)
        row += 1

        section("2. Split Ratio")

        # Train % entry + live label
        pct_frame = tk.Frame(parent, bg=self.BG)
        pct_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        pct_frame.columnconfigure(1, weight=1)
        tk.Label(
            pct_frame, text="Train %", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15), anchor="w", width=14,
        ).grid(row=0, column=0, sticky="w")
        pct_entry = tk.Entry(
            pct_frame, textvariable=self._splitds_train_pct,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 15), width=6,
        )
        pct_entry.grid(row=0, column=1, sticky="w", padx=4)

        self._splitds_ratio_lbl = tk.Label(
            pct_frame, text="80 % train / 20 % val",
            bg=self.BG, fg=self.GREY, font=("Helvetica", 14),
        )
        self._splitds_ratio_lbl.grid(row=0, column=2, sticky="w", padx=(8, 4))
        row += 1

        def _update_ratio_lbl(*_) -> None:
            try:
                pct = float(self._splitds_train_pct.get())
                if 1 <= pct <= 99:
                    self._splitds_ratio_lbl.config(
                        text=f"{pct:.0f} % train / {100 - pct:.0f} % val"
                    )
                else:
                    self._splitds_ratio_lbl.config(text="must be between 1 and 99")
            except ValueError:
                self._splitds_ratio_lbl.config(text="enter a number 1-99")

        self._splitds_train_pct.trace_add("write", _update_ratio_lbl)
        _update_ratio_lbl()

        # Seed
        seed_frame = tk.Frame(parent, bg=self.BG)
        seed_frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        seed_frame.columnconfigure(1, weight=1)
        tk.Label(
            seed_frame, text="Random seed", bg=self.BG, fg=self.FG,
            font=("Helvetica", 15), anchor="w", width=14,
        ).grid(row=0, column=0, sticky="w")
        tk.Entry(
            seed_frame, textvariable=self._splitds_seed,
            bg=self.ENTRY_BG, fg=self.FG, insertbackground=self.FG,
            relief="flat", bd=4, font=("Helvetica", 15), width=8,
        ).grid(row=0, column=1, sticky="w", padx=4)
        tk.Button(
            seed_frame, text="Randomize",
            command=lambda: self._splitds_seed.set(str(random.randint(0, 99999))),
            bg=self.BTN_BG, fg=self.FG, activebackground=self.BTN_ACTIVE,
            relief="flat", padx=8, font=("Helvetica", 15),
        ).grid(row=0, column=2, padx=(0, 4))
        row += 1

        section("3. Action")

        self._splitds_progress = ttk.Progressbar(parent, length=200, mode="determinate")
        self._splitds_progress.grid(row=row, column=0, sticky="ew", padx=4, pady=(0, 4))
        row += 1

        self._splitds_btn = tk.Button(
            parent,
            text="Split Dataset",
            command=self._splitds_action,
            bg=self.GREEN, fg=self.BG,
            activebackground="#79d483",
            font=("Helvetica", 16, "bold"),
            relief="flat", pady=6,
        )
        self._splitds_btn.grid(row=row, column=0, sticky="ew", padx=4, pady=(4, 4))
        row += 1

        # Log area
        section("4. Log")

        log_frame = tk.Frame(parent, bg=self.BG)
        log_frame.grid(row=row, column=0, sticky="nsew", padx=4, pady=2)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)

        self._splitds_log = tk.Text(
            log_frame,
            bg=self.ENTRY_BG, fg=self.FG,
            font=("Courier", 14),
            relief="flat", bd=4,
            state="disabled",
            wrap="word",
            height=8,
        )
        self._splitds_log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self._splitds_log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._splitds_log.config(yscrollcommand=sb.set)
        row += 1

    def _splitds_log_msg(self, msg: str) -> None:
        self._splitds_log.config(state="normal")
        self._splitds_log.insert("end", msg + "\n")
        self._splitds_log.see("end")
        self._splitds_log.config(state="disabled")

    def _splitds_action(self) -> None:
        dataset_dir = self._splitds_dataset_dir.get().strip()
        output_dir = self._splitds_output_dir.get().strip()

        if not dataset_dir or not os.path.isdir(dataset_dir):
            messagebox.showerror("Error", "Please select a valid dataset folder.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        try:
            train_pct = float(self._splitds_train_pct.get())
            if not (1 <= train_pct <= 99):
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Train % must be a number between 1 and 99.")
            return

        try:
            seed = int(self._splitds_seed.get())
        except ValueError:
            seed = 42
            self._splitds_log_msg("Note: invalid seed value; using default seed=42.")

        source_split = self._splitds_source_split.get().strip()

        self._splitds_btn.config(state="disabled")
        self._splitds_progress["value"] = 0
        self._splitds_log.config(state="normal")
        self._splitds_log.delete("1.0", "end")
        self._splitds_log.config(state="disabled")
        self._splitds_log_msg("-" * 40)
        self._splitds_log_msg(
            f"Splitting dataset - source split='{source_split or 'flat'}', "
            f"train={train_pct:.0f}%, seed={seed}"
        )
        self._set_status("Splitting dataset...")

        def worker() -> None:
            try:
                counts = split_dataset(
                    dataset_dir=dataset_dir,
                    split=source_split,
                    output_dir=output_dir,
                    train_pct=train_pct,
                    seed=seed,
                    log_cb=lambda m: self.after(0, self._splitds_log_msg, m),
                    progress_cb=lambda v: self.after(
                        0, self._splitds_progress.__setitem__, "value", v
                    ),
                )
                self.after(0, self._on_splitds_done, counts, output_dir)
            except Exception as exc:
                self.after(0, self._splitds_log_msg, f"ERROR: {exc}")
                self.after(0, self._set_status, "Dataset split failed.")
            finally:
                self.after(0, lambda: self._splitds_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_splitds_done(self, counts: dict[str, int], output_dir: str) -> None:
        self._splitds_progress["value"] = 100
        self._splitds_log_msg(
            f"Result: {counts['train']} train image(s), {counts['val']} val image(s)."
        )
        self._set_status(f"Dataset split to {output_dir}")
        messagebox.showinfo(
            "Success",
            f"Dataset split successfully!\n\n"
            f"Train: {counts['train']} image(s)\n"
            f"Val:   {counts['val']} image(s)\n\n"
            f"Output: {output_dir}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = SubsetMakerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
