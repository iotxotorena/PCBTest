# install_notes.md

# pcbTest - Guía de instalación en Jetson Orin Nano

Esta guía describe cómo preparar una Jetson Orin Nano para ejecutar pcbTest.

pcbTest usa:

- Python 3 para la GUI.
- Tkinter para la interfaz gráfica.
- Pillow para mostrar imágenes en la GUI.
- Docker para ejecutar el pipeline de visión.
- OpenCV dentro del contenedor.
- Ultralytics YOLO dentro del contenedor.
- Cámara accesible como /dev/video*.

---

## 1. Sistema recomendado

Hardware recomendado:

- NVIDIA Jetson Orin Nano
- Cámara USB o cámara compatible con V4L2
- Fuente de alimentación adecuada para la Jetson
- Pantalla, teclado y ratón para usar la GUI

Software recomendado:

- JetPack 6.x
- Docker funcionando
- NVIDIA Container Runtime funcionando
- Imagen Docker de Ultralytics para Jetson

Imagen Docker usada por defecto en los scripts:

ultralytics/ultralytics:latest-jetson-jetpack6

Si se usa otra versión de JetPack, puede ser necesario cambiar la imagen Docker en:

- pcb_realtime_pipeline.sh
- pcb_camera_test.sh

Variable:

DOCKER_IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6"

---

## 2. Actualizar el sistema

Ejecutar:

sudo apt update
sudo apt upgrade -y

Reiniciar si el sistema lo pide.

---

## 3. Instalar dependencias del sistema

Instalar Python, pip y Tkinter:

sudo apt install -y python3 python3-pip python3-tk

Instalar herramientas útiles de cámara:

sudo apt install -y v4l-utils

Comprobar versión de Python:

python3 --version

---

## 4. Instalar Pillow

La GUI necesita Pillow para cargar y mostrar imágenes.

Instalar:

python3 -m pip install pillow

Comprobar:

python3 - <<'PY'
from PIL import Image
print("Pillow OK")
PY

---

## 5. Comprobar Docker

En Jetson con JetPack, Docker normalmente ya viene instalado.

Comprobar:

docker --version

Comprobar que Docker responde:

docker info

Si Docker no existe, instalar:

sudo apt install -y docker.io

Activar Docker:

sudo systemctl enable docker
sudo systemctl start docker

---

## 6. Permitir usar Docker sin sudo

Añadir el usuario actual al grupo docker:

sudo usermod -aG docker $USER

Después de este comando:

1. Cerrar sesión.
2. Volver a entrar.

Comprobar:

groups

Debe aparecer:

docker

Probar Docker sin sudo:

docker run --rm hello-world

Si da error de permisos, cerrar sesión y volver a entrar de nuevo.

---

## 7. Comprobar NVIDIA Container Runtime

Comprobar paquetes NVIDIA relacionados con contenedores:

dpkg --get-selections | grep nvidia-container

Comprobar información Docker:

docker info | grep -i nvidia

En Jetson, si JetPack está correctamente instalado, el runtime de NVIDIA debería estar disponible.

---

## 8. Descargar imagen Docker de Ultralytics

La imagen usada por defecto es:

ultralytics/ultralytics:latest-jetson-jetpack6

Descargar:

docker pull ultralytics/ultralytics:latest-jetson-jetpack6

Probar que arranca:

docker run --rm -it ultralytics/ultralytics:latest-jetson-jetpack6 python -c "import cv2; print('OpenCV OK')"

Probar Ultralytics:

docker run --rm -it ultralytics/ultralytics:latest-jetson-jetpack6 python -c "from ultralytics import YOLO; print('Ultralytics OK')"

---

## 9. Copiar el proyecto pcbTest

Copiar la carpeta completa a la Jetson.

Ejemplo:

/home/usuario/pcbTest

La estructura esperada es:

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

---

## 10. Dar permisos de ejecución

Entrar en la carpeta:

cd ~/pcbTest

Dar permisos:

chmod +x pcb_gui_inspeccion.py
chmod +x pcb_gui_inspeccion.sh
chmod +x pcb_realtime_pipeline.py
chmod +x pcb_realtime_pipeline.sh
chmod +x pcb_camera_test.py
chmod +x pcb_camera_test.sh

---

## 11. Verificar ficheros necesarios

Comprobar que existe el modelo:

ls -lh weights/best.pt

Comprobar config de homografía:

cat config_homografia.json

Debe tener algo parecido a:

{
  "out_width": 1355,
  "out_height": 774
}

Comprobar referenceBoard:

ls -lh referenceBoard/
ls -lh referenceBoard/labels/

Debe existir:

referenceBoard/notes.json
referenceBoard/labels/referencia.txt

Importante:

Dentro de referenceBoard/labels/ debe haber un solo fichero .txt.

Comprobar:

find referenceBoard/labels -name "*.txt"

---

## 12. Comprobar cámara en el host

Conectar la cámara.

Listar dispositivos:

ls -l /dev/video*

Ver información de cámaras:

v4l2-ctl --list-devices

Si no aparece ningún /dev/video*, el sistema no está detectando la cámara.

Probar cámara con OpenCV en el host:

python3 - <<'PY'
import cv2

cap = cv2.VideoCapture(0)
print("opened:", cap.isOpened())

ret, frame = cap.read()
print("ret:", ret)

if ret:
    print("shape:", frame.shape)

cap.release()
PY

Si falla con 0, probar con 1, 2, etc.

---

## 13. Lanzar la GUI

Desde la carpeta del proyecto:

cd ~/pcbTest
./pcb_gui_inspeccion.sh

---

## 14. Primera configuración en la GUI

### Pestaña Rutas

Comprobar:

- Modelo YOLO: weights/best.pt
- Carpeta de salida: results/gui_pcb_inspection
- referenceBoard: referenceBoard/
- config_homografia.json
- Serigrafía: keypoints/serigrafia.png

Guardar configuración.

### Pestaña Cámara

Probar fuente:

0

Pulsar:

TEST cámara

Si falla, probar:

1
2
/dev/video0
/dev/video1
/dev/video2

Si la imagen sale incorrecta, probar resolución:

1280 x 720

o:

1920 x 1080

### Pestaña Configuración de inspección

Valores recomendados iniciales:

Método de homografía: hough
Confianza YOLO: 0.49
Distancia máxima centro: 0.035
Distancia máxima centro relajada: 0.060
Límite de capturas: 1
Duración: 0
Intervalo: 0
EXTRA como fallo: desactivado

### Pestaña Inspección

Pulsar:

Analizar placa

---

## 15. Resultados generados

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

## 16. Comprobación por terminal del test de cámara

También se puede probar la cámara sin abrir la GUI:

cd ~/pcbTest

./pcb_camera_test.sh \
  --camera-source 0 \
  --camera-width 0 \
  --camera-height 0 \
  --output-path results/gui_pcb_inspection/camera_test/latest_camera_test.jpg

Si funciona, debería guardar una imagen en:

results/gui_pcb_inspection/camera_test/latest_camera_test.jpg

---

## 17. Comprobación por terminal del pipeline

También se puede probar el pipeline sin GUI:

cd ~/pcbTest

./pcb_realtime_pipeline.sh \
  --camera-source 0 \
  --output-dir results/prueba_terminal \
  --reference-dir referenceBoard \
  --config config_homografia.json \
  --orientation-template keypoints/serigrafia.png \
  --component-model weights/best.pt \
  --limit 1 \
  --interval 0 \
  --homography-method hough \
  --conf 0.49 \
  --max-center-distance 0.035 \
  --max-center-distance-relaxed 0.060

---

## 18. Propiedad de carpetas y ficheros

Los scripts Docker se ejecutan con:

--user "$(id -u):$(id -g)"

Esto evita que los resultados se creen como root.

Comprobar propietario:

ls -lh results/

Si aparecen ficheros propiedad de root, probablemente se ejecutó Docker manualmente con sudo o con otro script antiguo.

Corregir propietario:

sudo chown -R $USER:$USER results .ultralytics .config .cache

---

## 19. Problemas frecuentes

### Problema: Docker pide permisos

Error típico:

permission denied while trying to connect to the Docker daemon socket

Solución:

sudo usermod -aG docker $USER

Cerrar sesión y volver a entrar.

---

### Problema: no se pudo abrir la cámara

Comprobar:

ls -l /dev/video*

Probar otras fuentes en la GUI:

0
1
2
/dev/video0
/dev/video1
/dev/video2

Comprobar que otra aplicación no esté usando la cámara.

Comprobar permisos:

groups

Debe aparecer:

video

Si no aparece:

sudo usermod -aG video $USER

Cerrar sesión y volver a entrar.

---

### Problema: Docker no ve la cámara

Los scripts montan automáticamente /dev/video*.

En la salida debe aparecer algo como:

Dispositivos de vídeo montados:
  --device
  /dev/video0:/dev/video0

Si aparece:

Dispositivos de vídeo montados:
  Ninguno

entonces el host no está viendo la cámara.

---

### Problema: modelo no encontrado

Comprobar:

ls -lh weights/best.pt

En la GUI, pestaña Rutas, seleccionar el modelo con ruta absoluta si está fuera del proyecto.

Ejemplo:

/home/usuario/modelos/best.pt

---

### Problema: referenceBoard no encontrado

Comprobar:

ls -lh referenceBoard/
ls -lh referenceBoard/labels/

Debe existir:

referenceBoard/notes.json
referenceBoard/labels/referencia.txt

---

### Problema: muchos EXTRA

Subir la confianza mínima YOLO.

Ejemplo:

0.49 → 0.55 → 0.60

---

### Problema: muchos MISSING

Bajar la confianza mínima YOLO.

Ejemplo:

0.60 → 0.55 → 0.49

También revisar:

- iluminación
- enfoque
- distancia cámara-placa
- orientación
- que el modelo corresponde a esa placa
- que notes.json coincide con las clases del modelo

---

### Problema: muchos MISPLACED

Subir distancia máxima de centro.

Ejemplo:

0.035 → 0.045 → 0.060

También revisar:

- calidad de homografía
- que config_homografia.json tenga el tamaño correcto
- que referenceBoard/labels/referencia.txt corresponde a ese tamaño corregido
- que la placa esté completa en la imagen

---

### Problema: homografía incorrecta

Revisar imágenes de debug:

results/gui_pcb_inspection/debug/

Comprobar:

- que la placa se ve completa
- que los bordes están bien iluminados
- que no hay reflejos fuertes
- que la placa no toca los bordes de la imagen
- que el método hough detecta correctamente el contorno

---

### Problema: la placa sale girada o espejada

Revisar:

keypoints/serigrafia.png

La serigrafía debe ser una zona visible y estable de la placa.

Revisar debug de orientación:

results/gui_pcb_inspection/debug/orientation/

---

## 20. Generar TensorRT engine opcional

El modelo recomendado para compartir es:

weights/best.pt

Para acelerar inferencia, se puede generar un .engine en la propia Jetson.

Ejemplo dentro de Python/Ultralytics:

from ultralytics import YOLO

model = YOLO("weights/best.pt")
model.export(format="engine", imgsz=640)

Después se puede seleccionar el .engine desde la GUI.

Importante:

El .engine debe generarse en la misma Jetson o en un entorno compatible.

---

## 21. Limpieza de resultados

Para limpiar resultados generados:

rm -rf results/gui_pcb_inspection

Crear carpeta vacía de nuevo:

mkdir -p results
touch results/.gitkeep

---

## 22. Reinstalación limpia

Si algo queda mal configurado:

1. Cerrar la GUI.
2. Borrar config local:

rm -f gui_config.json

3. Borrar resultados:

rm -rf results/gui_pcb_inspection

4. Abrir de nuevo:

./pcb_gui_inspeccion.sh

La GUI volverá a usar valores por defecto.

---

## 23. Checklist final

Antes de usar:

[ ] Docker funciona sin sudo.
[ ] El usuario pertenece al grupo docker.
[ ] El usuario pertenece al grupo video.
[ ] Existe /dev/video0 o similar.
[ ] TEST cámara funciona.
[ ] Existe weights/best.pt.
[ ] Existe config_homografia.json.
[ ] Existe keypoints/serigrafia.png.
[ ] Existe referenceBoard/notes.json.
[ ] Existe un único .txt en referenceBoard/labels/.
[ ] La placa aparece completa en la captura.
[ ] La homografía se ve correcta.
[ ] La imagen de fallos se genera correctamente.

