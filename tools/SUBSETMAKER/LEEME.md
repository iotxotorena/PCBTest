# SubsetMaker

> 🌐 Otros idiomas: [Euskera](README.md) | [English](README_EN.md)

Una aplicación con interfaz gráfica para crear subconjuntos equilibrados por etiqueta de conjuntos de datos YOLO de Visión Artificial.

![Captura de pantalla de SubsetMaker](https://github.com/user-attachments/assets/96bebab4-5fe4-416b-a2c5-462bf297fa62)

## Qué hace

SubsetMaker es una herramienta de escritorio para la **gestión de conjuntos de datos YOLO** con seis funcionalidades integradas:

- **Crear subconjunto** — recortar un conjunto de datos al número de imágenes deseado por clase, seleccionando qué clases conservar y, opcionalmente, reasignando los IDs de clase.
- **Verificar conjunto de datos** — detectar y corregir problemas de integridad como archivos de etiquetas faltantes o etiquetas huérfanas sin imagen correspondiente.
- **Información YAML** — inspeccionar cualquier archivo `data.yaml` para revisar los nombres de las clases y la configuración.
- **Dividir conjunto de datos** — dividir aleatoriamente una partición de un conjunto de datos en subconjuntos `train` / `val` con una proporción configurable y semilla reproducible.
- **Renumerar etiquetas** — reasignar los IDs de clase en todos los archivos de etiquetas de un directorio, en el mismo lugar o en un nuevo directorio de salida.
- **JSON → YAML** — convertir un archivo de anotaciones COCO JSON (o una lista JSON de nombres simple) en un `data.yaml` compatible con YOLO.

La aplicación admite **temas oscuros y claros** y recuerda tu preferencia entre sesiones.

## Cómo funciona

### Formato de etiquetas YOLO

Cada imagen en un conjunto de datos YOLO tiene un archivo `.txt` complementario con el mismo nombre base. Cada línea de ese archivo describe un objeto:

```
<class_id> <x_center> <y_center> <width> <height>
```

Todas las coordenadas están normalizadas al rango `[0, 1]` relativo a las dimensiones de la imagen. SubsetMaker lee y reescribe únicamente el campo `<class_id>` (el primer token de cada línea); las coordenadas del cuadro delimitador nunca se modifican.

### Crear subconjunto — algoritmo

1. **Escaneo** — la aplicación recorre cada imagen en la partición seleccionada y mapea cada `class_id` encontrado en el archivo de etiquetas correspondiente a la ruta de la imagen que lo contiene. Una imagen aparece en el mapa por cada clase que contiene.
2. **Muestreo** — para cada clase seleccionada, el grupo de imágenes que contiene esa clase se mezcla (usando la semilla aleatoria proporcionada) y se seleccionan las primeras `max_per_class` imágenes.
3. **Unión** — las imágenes seleccionadas de todas las clases se combinan en un único conjunto, de modo que una imagen anotada con varias clases nunca se duplica.
4. **Filtrado de etiquetas** — al escribir el archivo de etiquetas de salida para una imagen copiada, solo se escriben las líneas de anotación cuyo `class_id` está en el conjunto conservado. Las líneas de clases excluidas se descartan silenciosamente.
5. **Reasignación de IDs (opcional)** — si se marca *Reasignar IDs de clase*, los IDs de clase conservados se ordenan y renumeran desde `0` consecutivamente. Por ejemplo, si conservas las clases originales `2`, `5` y `7`, se convierten en `0`, `1` y `2` en la salida. La asignación se refleja en el `data.yaml` generado.
6. **Escritura de salida** — las imágenes se copian con `shutil.copy2` (conservando los metadatos), los archivos de etiquetas se escriben con contenido filtrado/reasignado, y se genera un nuevo `data.yaml` que lista solo las clases conservadas.

### Verificar conjunto de datos — algoritmo

El verificador compara los directorios de imágenes y etiquetas archivo por archivo:

- **Etiquetas faltantes** — para cada archivo de imagen, busca un archivo `.txt` con el mismo nombre base en el directorio de etiquetas. Cualquier imagen sin una etiqueta correspondiente se reporta como faltante.
- **Etiquetas huérfanas** — para cada archivo de etiquetas `.txt`, comprueba si existe una imagen con el mismo nombre base (probando todas las extensiones admitidas). Una etiqueta sin imagen correspondiente se reporta como huérfana.

Cuando no se especifica ninguna partición, la verificación se realiza **de forma recursiva** en todos los árboles `images/` y `labels/`, conservando la estructura de subdirectorios relativa. Cuando se proporciona una partición específica, solo se escanea el directorio hoja correspondiente (de forma no recursiva).

Las acciones de corrección son seguras por diseño: crear etiquetas vacías solo toca los archivos que están genuinamente ausentes, y eliminar huérfanas solo elimina los archivos que ya fueron reportados.

### Dividir conjunto de datos — algoritmo

1. Se recopilan y ordenan todos los nombres de archivos de imagen en la partición fuente, luego se mezclan con `random.seed(seed)` para reproducibilidad.
2. Las primeras `round(total * train_pct / 100)` imágenes van a `train`; el resto va a `val`. Al menos una imagen está siempre garantizada en cada partición cuando el total es ≥ 2.
3. Cada imagen se copia a `images/train` o `images/val` dentro de la carpeta de salida. Su archivo de etiquetas correspondiente (si existe) se copia al `labels/train` o `labels/val` correspondiente. Las imágenes sin etiqueta se copian sin error.
4. Si hay un `data.yaml` en la raíz del conjunto de datos fuente, también se copia a la carpeta de salida sin cambios.

### Renumerar etiquetas — algoritmo

El reasignador lee todos los archivos `.txt` en el directorio de etiquetas seleccionado. Para cada línea de anotación, reemplaza el `class_id` con el valor buscado en la tabla de mapeo proporcionada por el usuario. Los IDs de clase que no están en la tabla se mantienen tal cual.

- **Modo en el mismo lugar** (la carpeta de salida coincide con la carpeta de etiquetas, o está vacía): solo se escriben los archivos cuyo contenido realmente cambia, evitando escrituras de disco innecesarias.
- **Modo de copia** (carpeta de salida diferente): todos los archivos de etiquetas se escriben en el destino con IDs actualizados, independientemente de si su contenido cambió.

### Analizador YAML personalizado

SubsetMaker incluye un analizador YAML ligero (`parse_yaml`) en lugar de depender de la biblioteca PyYAML completa. Admite el subconjunto de características YAML utilizadas en los archivos YOLO `data.yaml`:

- Pares `key: value` (cadenas y enteros)
- Secuencias de flujo: `names: [cat, dog, bird]`
- Secuencias de bloque: `names:\n  - cat\n  - dog`
- Mapeos de bloque bajo una clave: `names:\n  0: cat\n  1: dog`
- Comentarios en línea (`#`) y cadenas entre comillas (simples y dobles)

Las características YAML que no se encuentran en las configuraciones YOLO típicas (anclajes, archivos multi-documento, anidamiento complejo, etc.) no están admitidas.

## Estructura del conjunto de datos admitida

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml        ← opcional (usado para los nombres de clases)
```

Las estructuras planas (imágenes y etiquetas directamente bajo `images/` y `labels/`) también están admitidas.

**Formatos de imagen admitidos:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

## Requisitos

- Python 3.10+
- `Pillow` ≥ 9.0 — para el manejo de archivos de imagen
- `tkinter` — incluido con la mayoría de las distribuciones de Python (instala `python3-tk` en Linux)

```bash
pip install -r requirements.txt
# Solo en Linux, si falta tkinter:
sudo apt-get install python3-tk
```

## Uso

```bash
python subsetmaker.py
```

---

### ✂ Crear subconjunto

Copia un subconjunto filtrado y (opcionalmente) reequilibrado de tu conjunto de datos en un nuevo directorio de salida, incluyendo un `data.yaml` regenerado.

**Flujo de trabajo:**

1. **Carpeta del conjunto de datos** — selecciona la raíz de tu conjunto de datos YOLO.
2. **Carpeta de salida** — elige dónde se escribirá el subconjunto.
3. **Partición** — elige `train`, `val`, `test`, o deja en blanco para estructuras planas.
4. Haz clic en **🔍 Cargar conjunto de datos** — la aplicación escanea las etiquetas y lista cada clase con su número de imágenes.
5. **Panel de clases** — marca/desmarca las clases que quieres conservar.
6. **Máximo de imágenes por clase** — establece el límite superior de imágenes a incluir para cada clase seleccionada.
7. **Semilla aleatoria** — establece una semilla entera para un muestreo reproducible.
8. **Reasignar IDs de clase** — cuando está marcado, los archivos de etiquetas de salida tendrán los IDs de clase renumerados desde 0.
9. Haz clic en **✂ Crear subconjunto** — las imágenes y las etiquetas filtradas se copian en la carpeta de salida.

---

### 🔍 Verificar conjunto de datos

Escanea una partición en busca de problemas de integridad comunes y ofrece correcciones con un solo clic.

**Flujo de trabajo:**

1. **Carpeta del conjunto de datos** — selecciona (o reutiliza) la raíz de tu conjunto de datos YOLO.
2. **Partición** — elige la partición a verificar (`train`, `val`, `test`, o en blanco para plana).
3. Haz clic en **🔍 Verificar conjunto de datos** — el panel de resultados lista:
   - **Etiquetas faltantes** — archivos de imagen que no tienen un archivo de etiquetas `.txt` correspondiente.
   - **Etiquetas huérfanas** — archivos de etiquetas `.txt` que no tienen una imagen correspondiente.
4. Usa los botones de corrección según sea necesario:
   - **➕ Crear etiquetas vacías para imágenes sin etiquetar** — escribe un `.txt` vacío para cada imagen que no tiene uno (las marca como muestras de fondo/negativas).
   - **🗑 Eliminar etiquetas huérfanas** — elimina los archivos de etiquetas que no tienen imagen correspondiente.

---

### 📄 Información YAML

Inspecciona rápidamente cualquier archivo de configuración `data.yaml` de estilo YOLO.

**Flujo de trabajo:**

1. Haz clic en **…** para navegar hasta un archivo `data.yaml` (o escribe la ruta directamente).
2. Haz clic en **📄 Cargar YAML** — el panel muestra:
   - **nc** — número de clases declaradas en el archivo.
   - **names** — la lista completa de nombres de clases, una por línea, con su índice.

---

### 🔀 Dividir conjunto de datos

Divide aleatoriamente una partición de un conjunto de datos en subconjuntos `train` y `val` separados.

**Flujo de trabajo:**

1. **Carpeta del conjunto de datos** — selecciona la raíz de tu conjunto de datos YOLO.
2. **Carpeta de salida** — elige dónde se crearán los nuevos subdirectorios `train` / `val`.
3. **Partición** — elige la partición fuente desde la que leer (`train`, `val`, `test`, o en blanco para estructuras planas).
4. **Train %** — establece el porcentaje de imágenes que van a la partición de entrenamiento (el resto va a validación).
5. **Semilla aleatoria** — establece una semilla entera para una mezcla reproducible.
6. Haz clic en **🔀 Dividir conjunto de datos** — las imágenes y sus archivos de etiquetas se copian en `images/train`, `images/val`, `labels/train` y `labels/val` dentro de la carpeta de salida. El `data.yaml` fuente también se copia cuando está presente.

---

### 🔢 Renumerar etiquetas

Aplica una reasignación personalizada de IDs de clase a todos los archivos de etiquetas YOLO en un directorio.

**Flujo de trabajo:**

1. **Carpeta de etiquetas** — selecciona el directorio que contiene los archivos de etiquetas `.txt`.
2. **Carpeta de salida** — elige una carpeta de destino, o déjala apuntando al mismo directorio para reasignar en el mismo lugar.
3. **Mapeo** — introduce una regla de reasignación por línea en el formato `id_viejo → id_nuevo` (p. ej. `2 → 0`).
4. Haz clic en **🔢 Renumerar etiquetas** — la aplicación reescribe solo los archivos cuyo contenido cambia (modo en el mismo lugar) o copia todos los archivos al directorio de salida con IDs actualizados.

---

### 📋 JSON → YAML

Convierte un archivo de anotaciones COCO JSON (o una lista JSON de nombres de clases simple) en un `data.yaml` compatible con YOLO.

**Formatos JSON admitidos:**

| Formato | Ejemplo |
|---------|---------|
| Anotaciones COCO | `{"categories": [{"id": 1, "name": "cat"}, …]}` |
| Lista de nombres | `["cat", "dog", "bird"]` |
| Objeto de nombres | `{"names": ["cat", "dog"]}` o `{"names": {"0": "cat", "1": "dog"}}` |

**Flujo de trabajo:**

1. Haz clic en **…** junto a **Archivo JSON** para navegar hasta tu archivo JSON (o escribe la ruta directamente).
2. Haz clic en **📋 Cargar JSON** — el mapeo de clases se muestra en el panel.
3. Opcionalmente edita la **Ruta de salida YAML**.
4. Haz clic en **💾 Guardar YAML** — se escribe un `data.yaml` con los nombres de clases extraídos.

---

## Ejecutar pruebas

```bash
pip install pytest
pytest test_subsetmaker.py -v
```
