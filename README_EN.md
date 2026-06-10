# PCBTest

> 🌐 Other languages: [Euskera](README.md) | [Castellano](README_ES.md)

Tools for visual inspection of PCB boards and YOLO dataset management.

---

## Repository structure

```
PCBTest/
├── PCBTest/          # Main PCB board inspection application
└── tools/
    ├── 2dDatasetCreator/   # Synthetic 2D dataset generator for YOLO
    └── SUBSETMAKER/        # GUI for YOLO dataset management
```

---

## PCBTest

Application for visual inspection of PCB boards via camera.

**Pipeline:**
```
Camera → homography → orientation → YOLO → comparison → OK / FAIL
```

The application captures the image of a board with a camera, corrects the perspective using homography, detects components with a YOLO model and compares them against a reference board. It finally reports whether the board is **OK** or **FAIL**.

Designed to run on a **Jetson Orin Nano** with Docker.

See [`PCBTest/GUIA_DE_USO.md`](PCBTest/GUIA_DE_USO.md) for detailed usage instructions.

---

## tools/2dDatasetCreator

Script (`yodaut.py`) that generates synthetic 2D image datasets for training YOLO models.

It takes component images from the `input/` folder, combines them with configurable parameters (number of elements, scale, rotation angle) and produces a dataset with images and YOLO labels ready for training.

See [`tools/2dDatasetCreator/README.md`](tools/2dDatasetCreator/README.md) for more information.

---

## tools/SUBSETMAKER

Desktop application (`subsetmaker.py`) for managing YOLO datasets.

Main features:

- **Create subset** — filters a dataset by classes and maximum number of images per class.
- **Verify dataset** — detects orphan labels or unlabelled images.
- **Split dataset** — splits a split into `train` / `val` with a reproducible seed.
- **Renumber labels** — remaps class IDs across all label files.
- **JSON → YAML** — converts COCO JSON annotations to YOLO `data.yaml` format.
- **YAML Info** — inspects any `data.yaml` file.

See [`tools/SUBSETMAKER/README.md`](tools/SUBSETMAKER/README.md) for more information.

---

## License

The source code is distributed under **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`).  
Documentation and explanatory materials are distributed under **Creative Commons Attribution-ShareAlike 4.0 International** (`CC-BY-SA-4.0`).
