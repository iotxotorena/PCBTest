# SubsetMaker

  > 🌐 Other Languages: [Euskera](README.md) | [Español](LEEME.md)
A GUI application to create label-balanced subsets of YOLO Computer Vision datasets.
  
![SubsetMaker screenshot](https://github.com/user-attachments/assets/96bebab4-5fe4-416b-a2c5-462bf297fa62)

## What it does

SubsetMaker is a desktop tool for **YOLO dataset management** with six built-in features:

- **Create Subset** — crop a dataset down to a desired number of images per class, selecting which classes to keep and optionally remapping class IDs.
- **Check Dataset** — detect and fix integrity issues such as missing label files or orphan labels without a matching image.
- **YAML Info** — inspect any `data.yaml` file to review class names and configuration.
- **Split Dataset** — randomly divide a dataset split into `train` / `val` subsets with a configurable ratio and reproducible seed.
- **Renumber Labels** — remap class IDs across every label file in a directory, in-place or to a new output directory.
- **JSON → YAML** — convert a COCO JSON annotation file (or a plain JSON names list) into a YOLO-compatible `data.yaml`.

The app supports **dark and light themes** and remembers your preference between sessions.

## How it works

### YOLO label format

Each image in a YOLO dataset has a companion `.txt` file with the same base name. Every line in that file describes one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to the range `[0, 1]` relative to the image dimensions. SubsetMaker reads and rewrites only the `<class_id>` field (the first token on each line); the bounding-box coordinates are never modified.

### Create Subset — algorithm

1. **Scan** — the app walks every image in the selected split and maps each `class_id` found in the matching label file to the image path that contains it. An image appears in the map for every class it contains.
2. **Sample** — for each selected class the pool of images that contain that class is shuffled (using the provided random seed) and the first `max_per_class` images are picked.
3. **Union** — the selected images from all classes are merged into a single set, so an image annotated with multiple classes is never duplicated.
4. **Filter labels** — when writing the output label file for a copied image, only annotation lines whose `class_id` is in the kept set are written. Lines for excluded classes are dropped silently.
5. **Remap IDs (optional)** — if *Remap class IDs* is checked, the kept class IDs are sorted and renumbered from `0` consecutively. For example, if you keep original classes `2`, `5`, and `7`, they become `0`, `1`, and `2` in the output. The mapping is reflected in the generated `data.yaml`.
6. **Write output** — images are copied with `shutil.copy2` (preserving metadata), label files are written with filtered/remapped content, and a new `data.yaml` is generated listing only the kept classes.

### Check Dataset — algorithm

The checker compares the image and label directories file by file:

- **Missing labels** — for each image file, it looks for a `.txt` file with the same stem in the labels directory. Any image without a matching label is reported as missing.
- **Orphan labels** — for each `.txt` label file, it checks whether an image with the same stem exists (trying every supported extension). A label without a matching image is reported as an orphan.

When no split is specified the check is performed **recursively** across the entire `images/` and `labels/` trees, preserving relative subdirectory structure. When a specific split is given, only the corresponding leaf directory is scanned (non-recursively).

The fix actions are safe by design: creating empty labels only touches files that are genuinely absent, and deleting orphans only removes files that were already reported.

### Split Dataset — algorithm

1. All image filenames in the source split are collected and sorted, then shuffled with `random.seed(seed)` for reproducibility.
2. The first `round(total * train_pct / 100)` images go to `train`; the rest go to `val`. At least one image is always guaranteed in each split when the total is ≥ 2.
3. Each image is copied to `images/train` or `images/val` inside the output folder. Its matching label file (if any) is copied to the corresponding `labels/train` or `labels/val`. Images without a label are copied without error.
4. If a `data.yaml` is present in the source dataset root it is also copied to the output folder unchanged.

### Renumber Labels — algorithm

The remapper reads every `.txt` file in the selected labels directory. For each annotation line it replaces the `class_id` with the value looked up in the user-supplied mapping table. Class IDs not present in the table are kept as-is.

- **In-place mode** (output folder equals the labels folder, or is empty): only files whose content actually changes are written, avoiding unnecessary disk writes.
- **Copy mode** (different output folder): all label files are written to the destination, whether or not their content changed.

### Custom YAML parser

SubsetMaker includes a lightweight YAML parser (`parse_yaml`) instead of depending on the full PyYAML library. It supports the subset of YAML features used in YOLO `data.yaml` files:

- `key: value` pairs (strings and integers)
- Flow sequences: `names: [cat, dog, bird]`
- Block sequences: `names:\n  - cat\n  - dog`
- Block mappings under a key: `names:\n  0: cat\n  1: dog`
- Inline comments (`#`) and quoted strings (single and double quotes)

YAML features not found in typical YOLO configs (anchors, multi-document files, complex nesting, etc.) are not supported.

## Supported dataset layout

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml        ← optional (used for class names)
```

Flat layouts (images and labels directly under `images/` and `labels/`) are also supported.

**Supported image formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

## Requirements

- Python 3.10+
- `Pillow` ≥ 9.0 — for image file handling
- `tkinter` — included with most Python distributions (install `python3-tk` on Linux)

```bash
pip install -r requirements.txt
# Linux only, if tkinter is missing:
sudo apt-get install python3-tk
```

## Usage

```bash
python subsetmaker.py
```

---

### ✂ Create Subset

Copies a filtered and (optionally) rebalanced subset of your dataset to a new output directory, including a regenerated `data.yaml`.

**Workflow:**

1. **Dataset folder** — select the root of your YOLO dataset.
2. **Output folder** — choose where the subset will be written.
3. **Split** — pick `train`, `val`, `test`, or leave blank for flat layouts.
4. Click **🔍 Load Dataset** — the app scans the labels and lists every class with its image count.
5. **Classes panel** — check/uncheck the classes you want to keep.
6. **Max images per class** — set the upper limit of images to include for each selected class.
7. **Random seed** — set an integer seed for reproducible sampling.
8. **Remap class IDs** — when checked, output label files will have class IDs renumbered from 0.
9. Click **✂ Create Subset** — images and filtered labels are copied to the output folder.

---

### 🔍 Check Dataset

Scans a split for common integrity problems and offers one-click fixes.

**Workflow:**

1. **Dataset folder** — select (or reuse) the root of your YOLO dataset.
2. **Split** — pick the split to check (`train`, `val`, `test`, or blank for flat).
3. Click **🔍 Check Dataset** — the results panel lists:
   - **Missing labels** — image files that have no corresponding `.txt` label file.
   - **Orphan labels** — `.txt` label files that have no corresponding image.
4. Use the fix buttons as needed:
   - **➕ Create empty labels for unlabeled images** — writes an empty `.txt` for every image that lacks one (marks them as background/negative samples).
   - **🗑 Delete orphan labels** — removes label files that have no matching image.

---

### 📄 YAML Info

Quickly inspect any YOLO-style `data.yaml` configuration file.

**Workflow:**

1. Click **…** to browse to a `data.yaml` file (or type the path directly).
2. Click **📄 Load YAML** — the panel shows:
   - **nc** — number of classes declared in the file.
   - **names** — the full list of class names, one per line, with their index.

---

### 🔀 Split Dataset

Randomly divides a dataset split into separate `train` and `val` subsets.

**Workflow:**

1. **Dataset folder** — select the root of your YOLO dataset.
2. **Output folder** — choose where the new `train` / `val` sub-directories will be created.
3. **Split** — pick the source split to read from (`train`, `val`, `test`, or blank for flat layouts).
4. **Train %** — set the percentage of images that go to the training split (the rest go to validation).
5. **Random seed** — set an integer seed for reproducible shuffling.
6. Click **🔀 Split Dataset** — images and their label files are copied into `images/train`, `images/val`, `labels/train`, and `labels/val` inside the output folder. The source `data.yaml` is also copied when present.

---

### 🔢 Renumber Labels

Applies a custom class-ID remapping to every YOLO label file in a directory.

**Workflow:**

1. **Labels folder** — select the directory containing `.txt` label files.
2. **Output folder** — choose a destination folder, or leave it pointing to the same directory to remap in-place.
3. **Mapping** — enter one remapping rule per line in the format `old_id → new_id` (e.g. `2 → 0`).
4. Click **🔢 Renumber Labels** — the app rewrites only the files whose content changes (in-place mode) or copies all files to the output directory with updated IDs.

---

### 📋 JSON → YAML

Converts a COCO JSON annotation file (or a plain JSON class-names list) into a YOLO-compatible `data.yaml`.

**Supported JSON formats:**

| Format | Example |
|--------|---------|
| COCO annotations | `{"categories": [{"id": 1, "name": "cat"}, …]}` |
| Names list | `["cat", "dog", "bird"]` |
| Names object | `{"names": ["cat", "dog"]}` or `{"names": {"0": "cat", "1": "dog"}}` |

**Workflow:**

1. Click **…** next to **JSON file** to browse to your JSON file (or type the path directly).
2. Click **📋 Load JSON** — the class mapping is displayed in the panel.
3. Optionally edit the **Output YAML path**.
4. Click **💾 Save YAML** — a `data.yaml` is written with the extracted class names.

---

## Running tests

```bash
pip install pytest
pytest test_subsetmaker.py -v
```
