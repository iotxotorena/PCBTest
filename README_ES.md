# PCBTest

> 🌐 Otros idiomas: [Euskera](README.md) | [English](README_EN.md)

Herramientas para inspección visual de placas PCB y gestión de datasets YOLO.

---

## Estructura del repositorio

```
PCBTest/
├── PCBTest/          # Aplicación principal de inspección de placas PCB
└── tools/
    ├── 2dDatasetCreator/   # Generador sintético de datasets 2D para YOLO
    └── SUBSETMAKER/        # GUI para gestión de datasets YOLO
```

---

## PCBTest

Aplicación para inspección visual de placas PCB mediante cámara.

**Pipeline:**
```
Cámara → homografía → orientación → YOLO → comparación → OK / MAL
```

La aplicación captura la imagen de una placa con una cámara, corrige la perspectiva mediante homografía, detecta los componentes con un modelo YOLO y los compara con una placa de referencia. Al final informa si la placa está **OK** o **MAL**.

Diseñado para ejecutarse en una **Jetson Orin Nano** con Docker.

Consulta [`PCBTest/GUIA_DE_USO.md`](PCBTest/GUIA_DE_USO.md) para instrucciones detalladas de uso.

---

## tools/2dDatasetCreator

Script (`yodaut.py`) que genera datasets sintéticos de imágenes 2D para entrenar modelos YOLO.

Toma imágenes de componentes de la carpeta `input/`, las combina con parámetros configurables (número de elementos, escala, ángulo de rotación) y genera un dataset con imágenes y etiquetas YOLO listo para entrenar.

Consulta [`tools/2dDatasetCreator/README.md`](tools/2dDatasetCreator/README.md) para más información.

---

## tools/SUBSETMAKER

Aplicación de escritorio (`subsetmaker.py`) para gestionar datasets YOLO.

Funcionalidades principales:

- **Crear subconjunto** — filtra un dataset por clases y número máximo de imágenes por clase.
- **Verificar dataset** — detecta etiquetas huérfanas o imágenes sin etiqueta.
- **Dividir dataset** — divide un split en `train` / `val` con semilla reproducible.
- **Renumerar etiquetas** — remapea los IDs de clase en todos los ficheros de etiquetas.
- **JSON → YAML** — convierte anotaciones COCO JSON a formato `data.yaml` de YOLO.
- **Info YAML** — inspecciona cualquier fichero `data.yaml`.

Consulta [`tools/SUBSETMAKER/README.md`](tools/SUBSETMAKER/README.md) para más información.

---

## Licencia

El código fuente se distribuye bajo **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`).  
La documentación y materiales explicativos se distribuyen bajo **Creative Commons Attribution-ShareAlike 4.0 International** (`CC-BY-SA-4.0`).
