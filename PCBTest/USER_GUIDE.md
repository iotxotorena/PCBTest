# pcbTest – User Guide

> 🌐 Other languages: [Euskera](ERABILTZAILE_GIDA.md) | [Castellano](GUIA_DE_USO.md)

*GPL-3.0-or-later / CC-BY-SA-4.0*

---

## Purpose

Visual inspection of PCB boards using a camera, applying homography to correct the image, detecting components with YOLO, and comparing them against a reference board.

**Pipeline:**
```
Camera → homography → orientation → YOLO → comparison → OK / FAIL
```

**Licence:** Code: GPL-3.0-or-later. Documentation and materials: CC-BY-SA-4.0.

---

## Table of Contents

1. [What pcbTest does](#1-what-pcbtest-does)
2. [Project structure](#2-project-structure)
3. [First run](#3-first-run)
4. [Paths tab](#4-paths-tab)
5. [Camera tab](#5-camera-tab)
6. [Inspection configuration](#6-inspection-configuration)
7. [Running an inspection](#7-running-an-inspection)
8. [Interpreting results](#8-interpreting-results)
9. [Recommended tuning](#9-recommended-tuning)
10. [Common problems](#10-common-problems)
11. [Licence and usage notes](#11-licence-and-usage-notes)

---

## 1. What pcbTest does

pcbTest is a tool for analysing the state of a PCB. A camera captures an image of the board, the program corrects the perspective, detects the components, and compares them against a known-good reference board. Finally, it reports to the user whether the board is **OK** or **FAIL**.

The program does not only run YOLO inference. It first prepares the image, places the board in a flat plane, and attempts to correct its orientation. Understanding the full pipeline is therefore important.

```
Camera
  → capture image
  → detect board
  → apply homography
  → verify orientation via silkscreen
  → detect components with YOLO
  → compare against referenceBoard/
  → result: BOARD OK or BOARD FAIL
```

> **Important:** pcbTest is intended for prototyping and educational use. Before deploying in an industrial environment, lighting, camera, model, tolerances and false positives/negatives must be properly validated.

---

## 2. Project structure

When copying the program to another Jetson, a clean folder layout is recommended:

```
pcbTest/
├── pcb_gui_inspeccion.py
├── pcb_gui_inspeccion.sh
├── pcb_realtime_pipeline.py
├── pcb_realtime_pipeline.sh
├── pcb_camera_test.py
├── pcb_camera_test.sh
├── procesar_pcb_homografia_yolo.py
├── comparar_yolo_reference.py
├── config_homografia.json
├── keypoints/
│   └── serigrafia.png
├── referenceBoard/
│   ├── notes.json
│   └── labels/
│       └── referencia.txt
├── weights/
│   └── best.pt
├── results/
│   └── .gitkeep
├── README.md
├── install_notes.md
└── .gitignore
```

**Key files:**

| File | Description |
|------|-------------|
| `pcb_gui_inspeccion.py` | Main graphical interface. |
| `pcb_realtime_pipeline.py` | Full inspection pipeline. |
| `pcb_camera_test.py` | Quick camera-test script. |
| `procesar_pcb_homografia_yolo.py` | Homography, orientation, and image processing. |
| `comparar_yolo_reference.py` | Compares YOLO detections against the reference. |
| `config_homografia.json` | Defines the size of the corrected image. |
| `referenceBoard/` | Geometric reference and class names of the good board. |
| `weights/best.pt` | YOLO model. |

---

## 3. First run

Before starting the program, make sure the Jetson has permission to use Docker and that the camera appears in the system.

**Set permissions:**

```bash
cd pcbTest
chmod +x pcb_gui_inspeccion.py
chmod +x pcb_gui_inspeccion.sh
chmod +x pcb_realtime_pipeline.py
chmod +x pcb_realtime_pipeline.sh
chmod +x pcb_camera_test.py
chmod +x pcb_camera_test.sh
```

**Launch the GUI:**

```bash
cd pcbTest
./pcb_gui_inspeccion.sh
```

When the interface opens you will see four main tabs: **Inspección**, **Rutas**, **Cámara**, and **Configuración de inspección**. Some labels may still appear in Spanish; their function is explained in the following sections.

> **Note:** The `gui_config.json` file should **not** be copied to another machine. It stores absolute paths specific to each machine.

---

## 4. Paths tab

On this tab you verify or select the paths required by the program. When using it for the first time, this is the first section to configure.

| Field | Description |
|-------|-------------|
| **YOLO model** | The YOLO model. Recommended: `weights/best.pt`. If it is stored elsewhere, select the absolute path. |
| **Output folder** | Folder where results will be saved. Typical: `results/gui_pcb_inspection/`. |
| **referenceBoard** | Folder containing the reference of the good board. |
| **config_homografia.json** | Defines the width and height of the corrected image. |
| **Silkscreen orientation** | Silkscreen image used to determine orientation. |

### config_homografia.json

This file does **not** detect the board. It defines the size of the image that will be generated after homography. Minimum content:

```json
{
  "out_width": 1355,
  "out_height": 774
}
```

> **Caution:** If this size is changed, the coordinates in `referenceBoard/labels/referencia.txt` are tied to the corrected-image dimensions.

### referenceBoard folder

```
referenceBoard/
├── notes.json
└── labels/
    └── referencia.txt
```

- `notes.json` – contains the class names.
- `labels/referencia.txt` – contains the positions of the good board's components in YOLO format.
- There must be **exactly one** `.txt` file inside `labels/`.

---

## 5. Camera tab

On the Camera tab you select the camera source and can run a capture test. It is recommended to do this before starting a full inspection.

**Common camera sources:**

```
0
1
/dev/video0
/dev/video1
/dev/video2
```

To find out which device your camera is on, run in a terminal:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

**TEST camera button**

The *TEST camera* button takes an instant capture and displays the image in the same tab. If the capture looks correct, you are ready for a full inspection.

The test image is saved here:

```
results/gui_pcb_inspection/camera_test/latest_camera_test.jpg
```

**Resolution**

If *Camera width* and *Camera height* are left at `0`, OpenCV will use the camera's default resolution. If there are issues, try:

- `1280 × 720`
- `1920 × 1080`

> **Important:** If the camera test fails, do **not** proceed with an inspection. First check the camera source, Docker permissions, and `/dev/video*` devices.

---

## 6. Inspection configuration

On this tab you adjust the detection and comparison parameters. These values directly affect the **MISSING**, **MISPLACED**, and **EXTRA** results.

| Parameter | Description |
|-----------|-------------|
| **Method** | Homography method. Recommended: `hough`. |
| **YOLO confidence** | Minimum confidence to accept a detection. Higher values give fewer false positives. |
| **Centre distance** | Maximum distance between the centres of the reference and the detection. |
| **Relaxed distance** | Wider tolerance used to avoid discarding candidates in some cases. |
| **EXTRA as failure** | When enabled, any extra component will mark the board as FAIL. |
| **Capture limit** | Normally `1` for button-triggered use. |

**Recommended starting values:**

| Parameter | Value |
|-----------|-------|
| Method | `hough` |
| YOLO confidence | `0.49` |
| Max centre distance | `0.035` |
| Relaxed centre distance | `0.060` |
| Capture limit | `1` |
| Duration | `0` |
| Interval | `0` |
| EXTRA as failure | disabled |

> Change **one parameter at a time**. If there is a problem, adjust a single value and test again so you can identify what improved or worsened the result.

---

## 7. Running an inspection

Before running an inspection, verify that the camera can see correctly and that the full board is visible. The board should **not** touch the edges of the image.

**Recommended procedure:**

1. Open the GUI: `./pcb_gui_inspeccion.sh`
2. Go to the **Paths** tab and verify all files.
3. Go to the **Camera** tab and click **TEST camera**.
4. If the capture is correct, go to the **Inspección** tab.
5. Place the board under the camera, fully visible and well-lit.
6. Click **Analizar placa**.
7. Wait for the result: **BOARD OK** or **BOARD FAIL**.

**What happens internally:**

1. Camera capture is taken.
2. The board is detected.
3. Homography is applied.
4. Orientation is verified via the silkscreen.
5. YOLO inference is run.
6. Detections are compared against the reference.
7. Image, CSVs, and summary are generated.

---

## 8. Interpreting results

After an inspection, the program generates an image and several CSV files. The GUI normally shows `latest_failures.jpg` – the image highlighting the detected failures.

| Status | Meaning |
|--------|---------|
| **OK** | The reference component was found and its position is acceptable. |
| **MISSING** | A component expected in the reference was not validly detected. |
| **MISPLACED** | A component of the correct class was detected, but its position or geometry is not good enough. |
| **EXTRA** | YOLO made a detection that has no matching component in the reference. |

**OK / FAIL criteria (default):**

The GUI considers a board correct if:

- `MISSING = 0`
- `MISPLACED = 0`

EXTRA detections are treated as warnings by default. If **EXTRA as failure** is enabled in the configuration, even a single EXTRA will mark the board as FAIL.

**Results folder:**

```
results/gui_pcb_inspection/
├── raw/latest_raw.jpg
├── corrected/latest_corrected.jpg
├── overlay/latest_result.jpg
├── overlay_failures/latest_failures.jpg
├── components/latest_components.csv
├── comparison/latest_comparison.csv
├── camera_test/latest_camera_test.jpg
├── debug/
└── summary_realtime.csv
```

---

## 9. Recommended tuning

**Too many false positives**

Increase YOLO confidence. This will discard weak detections.

```
0.49 → 0.55 → 0.60
```

**Correct components appearing as MISPLACED**

Increase centre distance slightly. This widens the geometric tolerance.

```
0.035 → 0.045 → 0.060
```

**Real components appearing as MISSING**

Decrease YOLO confidence slightly, or review lighting and focus.

```
0.60 → 0.55 → 0.49
```

**Bad homography**

- Make sure the full board is visible in the image.
- Avoid harsh glare and heavy shadows.
- The board must not touch the image edges.
- Check images in the `debug/homography/` folder.
- Keep the `hough` method – it is usually the most robust when edges are visible.

**Wrong orientation**

- Verify that `keypoints/serigrafia.png` is correct.
- The silkscreen must always be visible.
- Do not use a silkscreen that is very small or has a lot of glare.
- Check images in the `debug/orientation/` folder.

---

## 10. Common problems

**Error: cannot open camera**

First check that the system can see the camera:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

In the GUI, try other sources: `0`, `1`, `/dev/video0`, `/dev/video1`, `/dev/video2`.

**Docker gives a permission error**

The user must be in the `docker` group:

```bash
sudo usermod -aG docker $USER
```

Then log out and back in.

**Files are created as root**

New scripts run Docker as the regular user. However, if folders were previously created as root:

```bash
sudo chown -R $USER:$USER results .ultralytics .config .cache
```

**`no_valid_candidate_same_class` message**

This message means the program found detections of the same class but could not find a valid candidate to match with a specific reference component. It is usually related to centre distance, overlap, size, or homography.

---

## 11. Licence and usage notes

The pcbTest source code is distributed under:

> **GNU General Public License v3.0 or later**
> `SPDX-License-Identifier: GPL-3.0-or-later`

This means the program can be used, studied, modified and redistributed, but modified distributed versions must retain the GPLv3 or a compatible licence.

Documentation, images and explanatory materials are distributed under:

> **Creative Commons Attribution-ShareAlike 4.0 International**
> `SPDX-License-Identifier: CC-BY-SA-4.0`

**Critical usage notes:**

- This program is intended for educational and prototyping use.
- Before deploying in an industrial environment, thorough validation is required.
- YOLO model results depend on the quality of the training data.
- Camera, lighting, and board positioning must be repeatable.
- When sharing `best.pt`, take into account the origin and licence of the training data.

---

*Summary for correct use: first test the camera, then review the homography and orientation debug images, and finally adjust YOLO and comparison parameters gradually.*
