## Licencia

El código fuente de pcbTest se distribuye bajo la licencia:

GNU General Public License v3.0 or later

SPDX-License-Identifier: GPL-3.0-or-later

Esto significa que puedes usar, estudiar, modificar y redistribuir el programa, siempre que las versiones modificadas que distribuyas mantengan la misma licencia GPLv3 o compatible.

La documentación, imágenes y material explicativo del proyecto se distribuyen bajo:

Creative Commons Attribution-ShareAlike 4.0 International

SPDX-License-Identifier: CC-BY-SA-4.0


# pcbTest

pcbTest es una aplicación para inspección visual de placas PCB mediante cámara, homografía, detección YOLO y comparación contra una placa de referencia.

El flujo principal es:

Cámara
→ captura de imagen
→ detección de la placa
→ corrección de perspectiva mediante homografía
→ orientación mediante serigrafía
→ detección de componentes con YOLO
→ comparación contra referencia
→ resultado OK / MAL + imagen de fallos

---

## 1. Estructura del proyecto

La carpeta del proyecto debe tener esta estructura:

pcbTest/  
├── pcb_gui_inspeccion.py  
├── pcb_gui_inspeccion.sh  
│  
├── pcb_realtime_pipeline.py  
├── pcb_realtime_pipeline.sh  
│  
├── pcb_camera_test.py  
├── pcb_camera_test.sh  
│  
├── procesar_pcb_homografia_yolo.py  
├── comparar_yolo_reference.py  
│  
├── config_homografia.json  
├── keypoints/  
│   └── serigrafia.png  
│  
├── referenceBoard/  
│   ├── notes.json  
│   └── labels/  
│       └── referencia.txt  
│  
├── weights/  
│   └── best.pt  
│  
├── results/  
│   └── .gitkeep  
│  
├── README.md  
├── install_notes.md  
└── .gitignore  

---

## 2. Ficheros principales

### pcb_gui_inspeccion.py

Interfaz gráfica principal de la aplicación.

Incluye pestañas para:

- Inspección
- Rutas
- Cámara
- Configuración de inspección

### pcb_gui_inspeccion.sh

Lanzador de la interfaz gráfica.

Ejecuta:

./pcb_gui_inspeccion.py

desde la carpeta del proyecto.

### pcb_realtime_pipeline.py

Pipeline principal:

captura → homografía → orientación → YOLO → comparación → resultados

### pcb_realtime_pipeline.sh

Lanzador Docker del pipeline principal.

Se encarga de:

- montar la carpeta del proyecto dentro del contenedor
- montar el modelo YOLO aunque esté en una ruta externa
- montar los dispositivos /dev/video*
- ejecutar el pipeline dentro de Docker
- mantener los ficheros creados con el UID/GID del usuario que lanza el script

### procesar_pcb_homografia_yolo.py

Funciones de procesamiento de imagen:

- detección de placa
- cálculo de homografía
- orientación mediante serigrafía
- generación de imágenes de depuración

### comparar_yolo_reference.py

Compara las detecciones YOLO contra la referencia de la placa.

Genera:

- componentes detectados
- elementos correctos
- elementos missing
- elementos misplaced
- elementos extra
- overlays visuales

### pcb_camera_test.py

Script de prueba de cámara.

Permite comprobar que la cámara funciona antes de lanzar una inspección completa.

### pcb_camera_test.sh

Lanzador Docker del test de cámara.

Monta los dispositivos /dev/video* y guarda una captura de prueba.

---

## 3. Configuración de homografía

El fichero config_homografia.json define el tamaño de salida de la imagen corregida.

Contenido mínimo:

{
  "out_width": 1355,
  "out_height": 774
}

Este tamaño es importante porque la referencia YOLO de la placa está normalizada respecto a la imagen corregida.

config_homografia.json no detecta la placa.

La detección de placa se realiza mediante el método seleccionado en la GUI, normalmente hough.

El flujo es:

imagen de cámara
→ detección de placa con Hough
→ cálculo de las 4 esquinas
→ homografía
→ imagen corregida con tamaño fijo definido en config_homografia.json
→ YOLO
→ comparación contra referenceBoard

---

## 4. Carpeta referenceBoard

La carpeta referenceBoard contiene la referencia de la placa correcta.

Estructura:

referenceBoard/  
├── notes.json  
└── labels/  
    └── referencia.txt  

### notes.json

Contiene los nombres de las clases.

Ejemplo:

{
  "categories": [
    {
      "id": 0,
      "name": "C"
    },
    {
      "id": 8,
      "name": "LED"
    },
    {
      "id": 10,
      "name": "R"
    }
  ]
}

notes.json es la fuente buena de nombres de clases.

### labels/referencia.txt

Contiene las cajas YOLO de los componentes esperados en la placa de referencia.

Debe haber un solo fichero .txt dentro de referenceBoard/labels/.

Formato YOLO:

class_id x_center y_center width height

Ejemplo:

8 0.056492 0.069376 0.079258 0.126944
10 0.057014 0.288343 0.038009 0.195186

---

## 5. Modelo YOLO

El modelo recomendado para compartir el proyecto es:

weights/best.pt

Aunque el programa permite seleccionar el modelo desde la GUI, dejarlo en weights/best.pt facilita la instalación.

No se recomienda depender solo de un .engine, porque los modelos TensorRT pueden depender de:

- versión de JetPack
- versión de CUDA
- versión de TensorRT
- arquitectura de la máquina
- configuración concreta del entorno

El .pt es más portable.

Si se quiere usar TensorRT, lo recomendable es generar el .engine en la Jetson donde se va a ejecutar.

---

## 6. Requisitos

Probado para uso en Jetson Orin Nano con Docker.

Requisitos mínimos:

- Python 3
- python3-tk
- python3-pip
- Docker
- Pillow
- cámara visible como /dev/video*
- imagen Docker de Ultralytics para Jetson

Instalación básica:

sudo apt update
sudo apt install python3 python3-pip python3-tk docker.io

python3 -m pip install pillow

Añadir el usuario al grupo Docker:

sudo usermod -aG docker $USER

Después de ejecutar ese comando, cerrar sesión y volver a entrar.

---

## 7. Imagen Docker

El proyecto usa una imagen Docker de Ultralytics para Jetson.

Por defecto, en los scripts se usa:

ultralytics/ultralytics:latest-jetson-jetpack6

Si la Jetson usa otra versión de JetPack, puede ser necesario cambiar la imagen Docker en:

pcb_realtime_pipeline.sh
pcb_camera_test.sh

Variable:

DOCKER_IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"

---

## 8. Permisos

Después de copiar el proyecto en la Jetson:

cd pcbTest

chmod +x pcb_gui_inspeccion.py
chmod +x pcb_gui_inspeccion.sh
chmod +x pcb_realtime_pipeline.py
chmod +x pcb_realtime_pipeline.sh
chmod +x pcb_camera_test.py
chmod +x pcb_camera_test.sh

---

## 9. Ejecución

Lanzar la aplicación:

cd pcbTest
./pcb_gui_inspeccion.sh

---

## 10. Uso desde la GUI

### 1. Pestaña Rutas

Comprobar o seleccionar:

- modelo YOLO
- carpeta de salida
- carpeta referenceBoard
- config_homografia.json
- imagen de serigrafía

El modelo YOLO puede estar fuera de la carpeta del proyecto, pero debe seleccionarse con ruta absoluta.

### 2. Pestaña Cámara

Configurar la cámara.

Ejemplos de fuente:

0
1
/dev/video0
/dev/video2

Pulsar:

TEST cámara

para comprobar que la cámara captura correctamente.

La captura de prueba se guarda en:

results/gui_pcb_inspection/camera_test/latest_camera_test.jpg

### 3. Pestaña Configuración de inspección

Ajustar parámetros:

- método de homografía
- confianza mínima YOLO
- distancia máxima de centro
- distancia máxima de centro relajada
- límite de capturas
- duración
- intervalo
- si los elementos EXTRA cuentan como fallo o solo como aviso

Valores iniciales recomendados:

Método de homografía: hough
Confianza YOLO: 0.49
Distancia máxima centro: 0.035
Distancia máxima centro relajada: 0.060
Límite de capturas: 1
Intervalo: 0
Duración: 0

### 4. Pestaña Inspección

Pulsar:

Analizar placa

El programa mostrará:

- imagen de fallos
- estado de la placa
- resumen de detecciones
- tiempos de proceso

---

## 11. Resultados generados

Los resultados se guardan por defecto en:

results/gui_pcb_inspection/

Estructura típica:

results/gui_pcb_inspection/  
├── raw/  
│   └── latest_raw.jpg  
├── corrected/  
│   └── latest_corrected.jpg  
├── overlay/  
│   └── latest_result.jpg  
├── overlay_failures/  
│   └── latest_failures.jpg  
├── components/  
│   └── latest_components.csv  
├── comparison/  
│   └── latest_comparison.csv  
├── camera_test/  
│   └── latest_camera_test.jpg  
├── debug/  
└── summary_realtime.csv  

---

## 12. Criterio de resultado

La GUI considera la placa correcta si:

MISSING = 0
MISPLACED = 0

Los elementos EXTRA pueden tratarse como aviso o como fallo según la configuración.

Por defecto:

EXTRA = aviso

Si se activa la opción "Considerar EXTRA como fallo de placa", entonces cualquier EXTRA hará que la placa se marque como MAL.

---

## 13. Ajustes recomendados

### Muchos falsos positivos

Subir la confianza mínima YOLO.

Ejemplo:

0.49 → 0.55 → 0.60

### Componentes correctos aparecen como MISPLACED

Subir ligeramente la distancia máxima de centro.

Ejemplo:

0.035 → 0.045 → 0.060

### Faltan componentes reales

Bajar ligeramente la confianza mínima YOLO.

Ejemplo:

0.60 → 0.55 → 0.49

### Problemas de cámara

Probar distintas fuentes:

0
1
/dev/video0
/dev/video1
/dev/video2

También se puede fijar resolución:

1280 x 720
1920 x 1080

Si la cámara no aparece, comprobar en la Jetson:

ls -l /dev/video*

Si no aparece ningún /dev/video*, el problema no está en pcbTest, sino en la detección de cámara del sistema.

---

## 14. Ficheros que no conviene subir al repositorio

No subir:

results/*
gui_config.json
.ultralytics/
.config/
.cache/
__pycache__/
*.pyc
*.log

gui_config.json contiene rutas absolutas de cada máquina, por eso no debe compartirse.

---

## 15. .gitignore recomendado

Crear un fichero .gitignore con este contenido:

# Resultados generados
results/*
!results/.gitkeep

# Configuración local de cada máquina
gui_config.json

# Cachés Python
__pycache__/
*.pyc
*.pyo

# Cachés de Ultralytics / sistema
.ultralytics/
.config/
.cache/

# Logs
*.log

# Temporales
*.tmp

---

## 16. Preparar carpeta results

Para que la carpeta results exista pero esté vacía:

mkdir -p results
touch results/.gitkeep

---

## 17. Notas importantes

- config_homografia.json no detecta la placa; solo define el tamaño de salida de la imagen corregida.
- La detección de la placa se realiza mediante el método seleccionado en la GUI, normalmente hough.
- referenceBoard/notes.json es la fuente de nombres de clases.
- referenceBoard/labels/referencia.txt es la referencia geométrica de los componentes.
- El modelo .pt es más portable que un .engine.
- Si se quiere usar TensorRT, lo recomendable es generar el .engine en la Jetson donde se va a ejecutar.
- La carpeta referenceBoard/labels/ debe contener un solo fichero .txt.
- No compartir gui_config.json entre máquinas.
- Los scripts Docker se ejecutan con el UID/GID del usuario que los lanza, para evitar que las carpetas generadas queden como root.

---

## 18. Comprobación rápida

Después de copiar el proyecto:

cd pcbTest
./pcb_gui_inspeccion.sh

En la GUI:

1. Ir a Rutas.
2. Comprobar que el modelo apunta a weights/best.pt.
3. Comprobar que referenceBoard apunta a referenceBoard/.
4. Comprobar que config_homografia.json existe.
5. Comprobar que serigrafia.png existe.
6. Ir a Cámara.
7. Pulsar TEST cámara.
8. Si la captura es correcta, ir a Inspección.
9. Pulsar Analizar placa.

---

## 19. Problemas frecuentes

### Error: no se pudo abrir la cámara

Posibles causas:

- La cámara no está conectada.
- El índice no es correcto.
- Hay que probar /dev/video0, /dev/video1, etc.
- Docker no tiene acceso a la cámara.
- El host no ve ningún /dev/video*.

Comprobar:

ls -l /dev/video*

### Error: modelo no encontrado

Comprobar en la pestaña Rutas que el modelo existe y tiene ruta absoluta.

Ejemplo válido:

/home/usuario/pcbTest/weights/best.pt

### Error: referenceBoard no encontrado

Comprobar que existe:

referenceBoard/notes.json
referenceBoard/labels/referencia.txt

### Error: comparación incorrecta

Comprobar:

- que config_homografia.json tiene el tamaño correcto
- que referencia.txt corresponde a la imagen corregida con ese tamaño
- que los nombres de clases de notes.json coinciden con las clases del modelo
- que no hay varios ficheros .txt dentro de referenceBoard/labels/

### Error: muchos EXTRA

Subir confianza YOLO.

Ejemplo:

0.49 → 0.55

### Error: muchos MISSING

Bajar confianza YOLO o revisar iluminación/cámara.

### Error: muchos MISPLACED

Aumentar distancia máxima de centro o revisar la homografía.

---

## 20. Licencia / uso

Este proyecto está pensado como herramienta educativa y de prototipado para inspección visual de placas PCB.

Antes de usarlo en un entorno industrial real, habría que validar:

- robustez del modelo
- condiciones de iluminación
- repetibilidad de cámara
- tolerancias geométricas
- tasa de falsos positivos
- tasa de falsos negativos
- comportamiento con placas parcialmente visibles
- criterios de aceptación/rechazo


## Licencia

El código fuente de pcbTest se distribuye bajo la licencia:

GNU General Public License v3.0 or later

SPDX-License-Identifier: GPL-3.0-or-later

Esto significa que puedes usar, estudiar, modificar y redistribuir el programa, siempre que las versiones modificadas que distribuyas mantengan la misma licencia GPLv3 o compatible.

La documentación, imágenes y material explicativo del proyecto se distribuyen bajo:

Creative Commons Attribution-ShareAlike 4.0 International

SPDX-License-Identifier: CC-BY-SA-4.0
