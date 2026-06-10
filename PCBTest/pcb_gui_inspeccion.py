#!/usr/bin/env python3
import csv
import json
import subprocess
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit(
        "Falta Pillow.\n"
        "Instálalo con:\n"
        "  python3 -m pip install pillow\n"
    )


class PCBInspectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("pcbTest")
        self.root.geometry("1280x920")
        self.root.minsize(980, 720)

        self.app_dir = Path(__file__).resolve().parent
        self.pipeline_script = self.app_dir / "pcb_realtime_pipeline.sh"
        self.camera_test_script = self.app_dir / "pcb_camera_test.sh"
        self.config_path = self.app_dir / "gui_config.json"

        default_model = self.find_default_model()
        default_homography_config = self.find_default_homography_config()

        self.default_config = {
            "output_dir": str(self.app_dir / "results" / "gui_pcb_inspection"),
            "reference_dir": str(self.app_dir / "referenceBoard"),
            "homography_config": str(default_homography_config),
            "orientation_template": str(self.app_dir / "keypoints" / "serigrafia.png"),
            "model_path": str(default_model) if default_model else "",
            "camera_source": "0",
            "camera_width": 0,
            "camera_height": 0,
            "homography_method": "hough",
            "yolo_conf": 0.49,
            "max_center_distance": 0.035,
            "max_center_distance_relaxed": 0.060,
            "duration": 0.0,
            "limit": 1,
            "interval": 0.0,
            "treat_extra_as_error": False,
        }

        self.config = self.load_config()
        self.output_dir = Path(self.config["output_dir"])
        self.refresh_paths()

        self.current_image = None
        self.current_photo = None

        self.camera_test_image = None
        self.camera_test_photo = None

        self.worker_thread = None
        self.camera_test_thread = None

        self.is_processing = False
        self.last_run_start_time = 0.0
        self.last_camera_test_start_time = 0.0

        self.create_widgets()
        self.apply_config_to_widgets()
        self.refresh_paths()
        self.check_environment()

    def find_default_model(self):
        candidates = [
            self.app_dir / "train37" / "weights" / "best.pt",
            self.app_dir / "train37" / "weights" / "best.engine",
            self.app_dir / "weights" / "best.pt",
            self.app_dir / "weights" / "best.engine",
        ]

        for p in candidates:
            if p.exists():
                return p.resolve()

        return None

    def find_default_homography_config(self):
        candidates = [
            self.app_dir / "config_homografia.json",
            self.app_dir / "config_fiduciales.json",
        ]

        for p in candidates:
            if p.exists():
                return p.resolve()

        return self.app_dir / "config_homografia.json"

    def load_config(self):
        if not self.config_path.exists():
            return dict(self.default_config)

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            # Migración desde versiones anteriores:
            # antes se llamaba fiducial_config.
            if "homography_config" not in loaded and "fiducial_config" in loaded:
                loaded["homography_config"] = loaded["fiducial_config"]

            config = dict(self.default_config)
            config.update(loaded)
            return config

        except Exception:
            return dict(self.default_config)

    def save_config(self):
        if not self.read_config_from_widgets(validate=True):
            return

        self.refresh_paths()

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)

            self.write_info(f"Configuración guardada en: {self.config_path}\n")

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"No se pudo guardar la configuración:\n{self.config_path}\n\n{e}"
            )

    def refresh_paths(self):
        self.output_dir = Path(self.config["output_dir"]).expanduser()

        self.raw_image_path = self.output_dir / "raw" / "latest_raw.jpg"
        self.corrected_image_path = self.output_dir / "corrected" / "latest_corrected.jpg"
        self.failures_image_path = self.output_dir / "overlay_failures" / "latest_failures.jpg"
        self.normal_overlay_path = self.output_dir / "overlay" / "latest_result.jpg"
        self.summary_csv_path = self.output_dir / "summary_realtime.csv"
        self.comparison_csv_path = self.output_dir / "comparison" / "latest_comparison.csv"
        self.components_csv_path = self.output_dir / "components" / "latest_components.csv"

        self.camera_test_image_path = self.output_dir / "camera_test" / "latest_camera_test.jpg"

    def create_config_variables(self):
        self.output_dir_var = tk.StringVar()
        self.reference_dir_var = tk.StringVar()
        self.homography_config_var = tk.StringVar()
        self.orientation_template_var = tk.StringVar()
        self.model_path_var = tk.StringVar()

        self.camera_source_var = tk.StringVar()
        self.camera_width_var = tk.IntVar()
        self.camera_height_var = tk.IntVar()

        self.homography_method_var = tk.StringVar()
        self.yolo_conf_var = tk.DoubleVar()
        self.max_center_distance_var = tk.DoubleVar()
        self.max_center_distance_relaxed_var = tk.DoubleVar()
        self.duration_var = tk.DoubleVar()
        self.limit_var = tk.IntVar()
        self.interval_var = tk.DoubleVar()
        self.treat_extra_as_error_var = tk.BooleanVar()

    def create_widgets(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.create_config_variables()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tab_inspection = ttk.Frame(self.notebook)
        self.tab_routes = ttk.Frame(self.notebook)
        self.tab_camera = ttk.Frame(self.notebook)
        self.tab_inspection_config = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_inspection, text="Inspección")
        self.notebook.add(self.tab_routes, text="Rutas")
        self.notebook.add(self.tab_camera, text="Cámara")
        self.notebook.add(self.tab_inspection_config, text="Configuración de inspección")

        self.create_inspection_tab()
        self.create_routes_tab()
        self.create_camera_tab()
        self.create_inspection_config_tab()

    def create_inspection_tab(self):
        self.tab_inspection.columnconfigure(0, weight=1)
        self.tab_inspection.rowconfigure(1, weight=1)

        top_frame = ttk.Frame(self.tab_inspection, padding=10)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(6, weight=1)

        self.btn_analyze = ttk.Button(
            top_frame,
            text="Analizar placa",
            command=self.start_analysis
        )
        self.btn_analyze.grid(row=0, column=0, padx=(0, 10))

        self.btn_open_folder = ttk.Button(
            top_frame,
            text="Abrir resultados",
            command=self.open_results_folder
        )
        self.btn_open_folder.grid(row=0, column=1, padx=(0, 10))

        self.btn_show_raw = ttk.Button(
            top_frame,
            text="Raw",
            command=self.show_raw_image
        )
        self.btn_show_raw.grid(row=0, column=2, padx=(0, 10))

        self.btn_show_corrected = ttk.Button(
            top_frame,
            text="Corregida",
            command=self.show_corrected_image
        )
        self.btn_show_corrected.grid(row=0, column=3, padx=(0, 10))

        self.btn_show_failures = ttk.Button(
            top_frame,
            text="Fallos",
            command=self.show_failures_image
        )
        self.btn_show_failures.grid(row=0, column=4, padx=(0, 10))

        self.btn_show_overlay = ttk.Button(
            top_frame,
            text="Overlay completo",
            command=self.show_normal_overlay_image
        )
        self.btn_show_overlay.grid(row=0, column=5, padx=(0, 10))

        self.status_label = tk.Label(
            top_frame,
            text="ESPERANDO",
            font=("Arial", 18, "bold"),
            bg="#444444",
            fg="white",
            padx=15,
            pady=8
        )
        self.status_label.grid(row=0, column=6, sticky="e")

        image_frame = ttk.Frame(self.tab_inspection, padding=(10, 0, 10, 10))
        image_frame.grid(row=1, column=0, sticky="nsew")
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(
            image_frame,
            bg="#202020",
            highlightthickness=0
        )
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        self.image_canvas.bind("<Configure>", self.on_canvas_resize)

        bottom_frame = ttk.Frame(self.tab_inspection, padding=10)
        bottom_frame.grid(row=2, column=0, sticky="ew")
        bottom_frame.columnconfigure(0, weight=1)

        self.info_text = tk.Text(
            bottom_frame,
            height=10,
            wrap="word",
            font=("Consolas", 10)
        )
        self.info_text.grid(row=0, column=0, sticky="ew")

        scrollbar = ttk.Scrollbar(
            bottom_frame,
            orient="vertical",
            command=self.info_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.info_text.configure(yscrollcommand=scrollbar.set)

        self.write_info("GUI preparada.\n")
        self.write_info(f"Carpeta programa: {self.app_dir}\n")
        self.write_info(f"Config GUI:       {self.config_path}\n")

    def create_routes_tab(self):
        self.tab_routes.columnconfigure(0, weight=1)
        self.tab_routes.rowconfigure(0, weight=1)

        main = ttk.Frame(self.tab_routes, padding=24)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)

        row = 0

        title = ttk.Label(
            main,
            text="Rutas del proyecto",
            font=("Arial", 18, "bold")
        )
        title.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 20))
        row += 1

        ttk.Label(
            main,
            text=(
                "Aquí se configuran los ficheros y carpetas que necesita el sistema. "
                "El modelo YOLO puede estar fuera de la carpeta del programa, pero debe tener ruta absoluta."
            ),
            wraplength=1000
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 20))
        row += 1

        row = self.add_path_selector(
            main,
            row,
            "Modelo YOLO (.pt/.engine)",
            self.model_path_var,
            self.browse_model_file,
            required=True
        )

        row = self.add_path_selector(
            main,
            row,
            "Carpeta de salida",
            self.output_dir_var,
            self.browse_output_dir,
            required=True
        )

        row = self.add_path_selector(
            main,
            row,
            "referenceBoard",
            self.reference_dir_var,
            self.browse_reference_dir,
            required=True
        )

        row = self.add_path_selector(
            main,
            row,
            "config_homografia.json",
            self.homography_config_var,
            self.browse_homography_config,
            required=True
        )

        row = self.add_path_selector(
            main,
            row,
            "Serigrafía orientación",
            self.orientation_template_var,
            self.browse_orientation_template,
            required=True
        )

        ttk.Separator(main).grid(row=row, column=0, columnspan=3, sticky="ew", pady=20)
        row += 1

        buttons = ttk.Frame(main)
        buttons.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Button(
            buttons,
            text="Guardar configuración",
            command=self.save_config
        ).grid(row=0, column=0, padx=(0, 10))

        ttk.Button(
            buttons,
            text="Aplicar sin guardar",
            command=self.apply_config_without_saving
        ).grid(row=0, column=1, padx=(0, 10))

        ttk.Button(
            buttons,
            text="Restaurar valores por defecto",
            command=self.restore_default_config
        ).grid(row=0, column=2, padx=(0, 10))

        row += 1

        help_text = tk.Text(
            main,
            height=10,
            wrap="word",
            font=("Consolas", 10)
        )
        help_text.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(20, 0))

        help_text.insert(
            "end",
            "Estructura esperada:\n"
            "- referenceBoard/notes.json\n"
            "- referenceBoard/labels/UN_SOLO_FICHERO.txt\n"
            "- config_homografia.json\n"
            "- keypoints/serigrafia.png\n"
            "- modelo YOLO .pt o .engine\n\n"
            "Notas:\n"
            "- config_homografia.json define el tamaño de salida de la imagen corregida por homografía.\n"
            "- El modelo se puede llamar como quieras.\n"
            "- El modelo puede estar fuera del programa.\n"
            "- Si usas .engine, debe ser compatible con TensorRT/CUDA de la máquina actual.\n"
        )
        help_text.configure(state="disabled")

    def create_camera_tab(self):
        self.tab_camera.columnconfigure(0, weight=1)
        self.tab_camera.rowconfigure(0, weight=1)

        main = ttk.Frame(self.tab_camera, padding=24)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)

        row = 0

        title = ttk.Label(
            main,
            text="Configuración de cámara",
            font=("Arial", 18, "bold")
        )
        title.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 20))
        row += 1

        ttk.Label(
            main,
            text=(
                "Aquí se configura qué cámara debe usar OpenCV dentro del contenedor Docker. "
                "El botón TEST hace una captura instantánea y la muestra debajo."
            ),
            wraplength=1000
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 20))
        row += 1

        ttk.Label(main, text="Fuente de cámara").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Entry(main, textvariable=self.camera_source_var).grid(row=row, column=1, sticky="ew", padx=10)
        ttk.Label(main, text="Ej: 0, 1, /dev/video0, /dev/video2").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(main, text="Ancho cámara").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Spinbox(
            main,
            from_=0,
            to=4096,
            increment=1,
            textvariable=self.camera_width_var,
            width=10
        ).grid(row=row, column=1, sticky="w", padx=10)
        ttk.Label(main, text="0 = automático").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(main, text="Alto cámara").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Spinbox(
            main,
            from_=0,
            to=4096,
            increment=1,
            textvariable=self.camera_height_var,
            width=10
        ).grid(row=row, column=1, sticky="w", padx=10)
        ttk.Label(main, text="0 = automático").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Separator(main).grid(row=row, column=0, columnspan=3, sticky="ew", pady=16)
        row += 1

        buttons = ttk.Frame(main)
        buttons.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)

        self.btn_camera_test = ttk.Button(
            buttons,
            text="TEST cámara",
            command=self.start_camera_test
        )
        self.btn_camera_test.grid(row=0, column=0, padx=(0, 10))

        ttk.Button(
            buttons,
            text="Guardar configuración",
            command=self.save_config
        ).grid(row=0, column=1, padx=(0, 10))

        ttk.Button(
            buttons,
            text="Aplicar sin guardar",
            command=self.apply_config_without_saving
        ).grid(row=0, column=2, padx=(0, 10))

        ttk.Button(
            buttons,
            text="Restaurar valores por defecto",
            command=self.restore_default_config
        ).grid(row=0, column=3, padx=(0, 10))

        row += 1

        self.camera_test_status_label = ttk.Label(
            main,
            text="Última captura de prueba: sin capturar todavía",
            font=("Arial", 11, "bold")
        )
        self.camera_test_status_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 8))
        row += 1

        main.rowconfigure(row, weight=1)

        preview_frame = ttk.Frame(main)
        preview_frame.grid(row=row, column=0, columnspan=3, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.camera_test_canvas = tk.Canvas(
            preview_frame,
            bg="#202020",
            highlightthickness=0,
            height=320
        )
        self.camera_test_canvas.grid(row=0, column=0, sticky="nsew")
        self.camera_test_canvas.bind("<Configure>", self.on_camera_test_canvas_resize)

    def create_inspection_config_tab(self):
        self.tab_inspection_config.columnconfigure(0, weight=1)
        self.tab_inspection_config.rowconfigure(0, weight=1)

        main = ttk.Frame(self.tab_inspection_config, padding=24)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)

        row = 0

        title = ttk.Label(
            main,
            text="Configuración de inspección",
            font=("Arial", 18, "bold")
        )
        title.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 20))
        row += 1

        ttk.Label(
            main,
            text=(
                "Aquí se ajustan los parámetros de detección, homografía y comparación. "
                "Estos valores afectan directamente a los falsos positivos, missing y misplaced."
            ),
            wraplength=1000
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 20))
        row += 1

        ttk.Label(main, text="Método de homografía").grid(row=row, column=0, sticky="w", pady=8)
        homography_combo = ttk.Combobox(
            main,
            textvariable=self.homography_method_var,
            values=["hough", "lines", "box"],
            state="readonly",
            width=15
        )
        homography_combo.grid(row=row, column=1, sticky="w", padx=10)
        row += 1

        ttk.Label(main, text="Confianza mínima YOLO").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Spinbox(
            main,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.yolo_conf_var,
            width=10,
            format="%.2f"
        ).grid(row=row, column=1, sticky="w", padx=10)
        ttk.Label(main, text="Ejemplo: 0.49").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(main, text="Distancia máxima centro").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Spinbox(
            main,
            from_=0.0,
            to=1.0,
            increment=0.005,
            textvariable=self.max_center_distance_var,
            width=10,
            format="%.3f"
        ).grid(row=row, column=1, sticky="w", padx=10)
        ttk.Label(main, text="Ejemplo: 0.035").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(main, text="Distancia máxima centro relajada").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Spinbox(
            main,
            from_=0.0,
            to=1.0,
            increment=0.005,
            textvariable=self.max_center_distance_relaxed_var,
            width=10,
            format="%.3f"
        ).grid(row=row, column=1, sticky="w", padx=10)
        ttk.Label(main, text="Ejemplo: 0.060").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Checkbutton(
            main,
            text="Considerar EXTRA como fallo de placa",
            variable=self.treat_extra_as_error_var
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 8))
        row += 1

        ttk.Separator(main).grid(row=row, column=0, columnspan=3, sticky="ew", pady=20)
        row += 1

        ttk.Label(main, text="Límite de capturas").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Spinbox(
            main,
            from_=1,
            to=100,
            increment=1,
            textvariable=self.limit_var,
            width=10
        ).grid(row=row, column=1, sticky="w", padx=10)
        ttk.Label(main, text="Para botón manual: 1").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(main, text="Duración").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Spinbox(
            main,
            from_=0.0,
            to=999.0,
            increment=1.0,
            textvariable=self.duration_var,
            width=10,
            format="%.1f"
        ).grid(row=row, column=1, sticky="w", padx=10)
        ttk.Label(main, text="0 = no usar duración").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(main, text="Intervalo entre capturas").grid(row=row, column=0, sticky="w", pady=8)
        ttk.Spinbox(
            main,
            from_=0.0,
            to=60.0,
            increment=0.1,
            textvariable=self.interval_var,
            width=10,
            format="%.1f"
        ).grid(row=row, column=1, sticky="w", padx=10)
        ttk.Label(main, text="Para botón manual: 0").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Separator(main).grid(row=row, column=0, columnspan=3, sticky="ew", pady=20)
        row += 1

        buttons = ttk.Frame(main)
        buttons.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Button(
            buttons,
            text="Guardar configuración",
            command=self.save_config
        ).grid(row=0, column=0, padx=(0, 10))

        ttk.Button(
            buttons,
            text="Aplicar sin guardar",
            command=self.apply_config_without_saving
        ).grid(row=0, column=1, padx=(0, 10))

        ttk.Button(
            buttons,
            text="Restaurar valores por defecto",
            command=self.restore_default_config
        ).grid(row=0, column=2, padx=(0, 10))

        row += 1

        help_text = tk.Text(
            main,
            height=12,
            wrap="word",
            font=("Consolas", 10)
        )
        help_text.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(20, 0))

        help_text.insert(
            "end",
            "Ajustes recomendados:\n"
            "- Si tienes muchos falsos positivos: sube Confianza mínima YOLO. Ej: 0.55, 0.60.\n"
            "- Si componentes correctos salen como MISPLACED: sube un poco la distancia máxima centro.\n"
            "- Si faltan componentes reales: baja un poco la confianza YOLO.\n"
            "- Hough suele ser el método más robusto si los bordes de la placa están bien definidos.\n"
            "- EXTRA como fallo solo conviene activarlo cuando el modelo sea muy fiable y no genere falsos positivos.\n"
        )
        help_text.configure(state="disabled")

    def add_path_selector(self, parent, row, label, variable, command, required=False):
        suffix = " *" if required else ""
        ttk.Label(parent, text=label + suffix).grid(row=row, column=0, sticky="w", pady=8)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=10)
        ttk.Button(parent, text="Examinar", command=command).grid(row=row, column=2, sticky="e")
        return row + 1

    def browse_model_file(self):
        path = filedialog.askopenfilename(
            title="Seleccionar modelo YOLO",
            filetypes=[
                ("Modelos YOLO", "*.pt *.engine *.onnx"),
                ("PyTorch", "*.pt"),
                ("TensorRT", "*.engine"),
                ("Todos", "*.*"),
            ]
        )
        if path:
            self.model_path_var.set(str(Path(path).resolve()))

    def browse_output_dir(self):
        path = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if path:
            self.output_dir_var.set(str(Path(path).resolve()))

    def browse_reference_dir(self):
        path = filedialog.askdirectory(title="Seleccionar carpeta referenceBoard")
        if path:
            self.reference_dir_var.set(str(Path(path).resolve()))

    def browse_homography_config(self):
        path = filedialog.askopenfilename(
            title="Seleccionar config_homografia.json",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")]
        )
        if path:
            self.homography_config_var.set(str(Path(path).resolve()))

    def browse_orientation_template(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen de serigrafía",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("Todos", "*.*"),
            ]
        )
        if path:
            self.orientation_template_var.set(str(Path(path).resolve()))

    def apply_config_to_widgets(self):
        self.output_dir_var.set(str(self.config["output_dir"]))
        self.reference_dir_var.set(str(self.config["reference_dir"]))
        self.homography_config_var.set(str(self.config["homography_config"]))
        self.orientation_template_var.set(str(self.config["orientation_template"]))
        self.model_path_var.set(str(self.config["model_path"]))

        self.camera_source_var.set(str(self.config["camera_source"]))
        self.camera_width_var.set(int(self.config["camera_width"]))
        self.camera_height_var.set(int(self.config["camera_height"]))

        self.homography_method_var.set(str(self.config["homography_method"]))
        self.yolo_conf_var.set(float(self.config["yolo_conf"]))
        self.max_center_distance_var.set(float(self.config["max_center_distance"]))
        self.max_center_distance_relaxed_var.set(float(self.config["max_center_distance_relaxed"]))
        self.duration_var.set(float(self.config["duration"]))
        self.limit_var.set(int(self.config["limit"]))
        self.interval_var.set(float(self.config["interval"]))
        self.treat_extra_as_error_var.set(bool(self.config["treat_extra_as_error"]))

    def read_config_from_widgets(self, validate=False):
        try:
            output_dir = Path(self.output_dir_var.get().strip()).expanduser()
            reference_dir = Path(self.reference_dir_var.get().strip()).expanduser()
            homography_config = Path(self.homography_config_var.get().strip()).expanduser()
            orientation_template = Path(self.orientation_template_var.get().strip()).expanduser()
            model_path = Path(self.model_path_var.get().strip()).expanduser()

            if validate:
                if not self.model_path_var.get().strip():
                    messagebox.showerror("Modelo requerido", "Debes seleccionar un modelo YOLO.")
                    self.notebook.select(self.tab_routes)
                    return False

                if not model_path.is_absolute():
                    messagebox.showerror(
                        "Ruta no válida",
                        "La ruta del modelo debe ser absoluta."
                    )
                    self.notebook.select(self.tab_routes)
                    return False

                if not model_path.exists() or not model_path.is_file():
                    messagebox.showerror(
                        "Modelo no encontrado",
                        f"No existe el modelo:\n{model_path}"
                    )
                    self.notebook.select(self.tab_routes)
                    return False

                if not reference_dir.exists() or not reference_dir.is_dir():
                    messagebox.showerror(
                        "referenceBoard no encontrado",
                        f"No existe la carpeta:\n{reference_dir}"
                    )
                    self.notebook.select(self.tab_routes)
                    return False

                if not homography_config.exists() or not homography_config.is_file():
                    messagebox.showerror(
                        "Config de homografía no encontrada",
                        f"No existe el fichero:\n{homography_config}"
                    )
                    self.notebook.select(self.tab_routes)
                    return False

                if not orientation_template.exists() or not orientation_template.is_file():
                    messagebox.showerror(
                        "Serigrafía no encontrada",
                        f"No existe la imagen:\n{orientation_template}"
                    )
                    self.notebook.select(self.tab_routes)
                    return False

                if not self.camera_source_var.get().strip():
                    messagebox.showerror(
                        "Cámara no configurada",
                        "Debes indicar una fuente de cámara: 0, 1, /dev/video0..."
                    )
                    self.notebook.select(self.tab_camera)
                    return False

            self.config["output_dir"] = str(output_dir.resolve())
            self.config["reference_dir"] = str(reference_dir.resolve())
            self.config["homography_config"] = str(homography_config.resolve())
            self.config["orientation_template"] = str(orientation_template.resolve())

            if self.model_path_var.get().strip():
                self.config["model_path"] = str(model_path.resolve())
            else:
                self.config["model_path"] = ""

            self.config["camera_source"] = self.camera_source_var.get().strip()
            self.config["camera_width"] = int(self.camera_width_var.get())
            self.config["camera_height"] = int(self.camera_height_var.get())

            self.config["homography_method"] = self.homography_method_var.get().strip()
            self.config["yolo_conf"] = float(self.yolo_conf_var.get())
            self.config["max_center_distance"] = float(self.max_center_distance_var.get())
            self.config["max_center_distance_relaxed"] = float(self.max_center_distance_relaxed_var.get())
            self.config["duration"] = float(self.duration_var.get())
            self.config["limit"] = int(self.limit_var.get())
            self.config["interval"] = float(self.interval_var.get())
            self.config["treat_extra_as_error"] = bool(self.treat_extra_as_error_var.get())

            return True

        except Exception as e:
            messagebox.showerror("Configuración inválida", str(e))
            return False

    def apply_config_without_saving(self):
        if not self.read_config_from_widgets(validate=True):
            return

        self.refresh_paths()
        self.write_info("Configuración aplicada sin guardar.\n")
        self.write_current_config_to_info()

    def restore_default_config(self):
        self.config = dict(self.default_config)
        self.apply_config_to_widgets()
        self.refresh_paths()
        self.write_info("Valores por defecto restaurados. Pulsa guardar si quieres conservarlos.\n")

    def check_environment(self):
        if not self.app_dir.exists():
            self.set_status("ERROR", "#aa0000")
            self.write_info(f"ERROR: no existe {self.app_dir}\n")
            return

        if not self.pipeline_script.exists():
            self.set_status("ERROR", "#aa0000")
            self.write_info(f"ERROR: no existe {self.pipeline_script}\n")
            return

        if not self.camera_test_script.exists():
            self.write_info(f"AVISO: no existe {self.camera_test_script}. El botón TEST cámara no funcionará.\n")

        self.write_info("Entorno correcto.\n")
        self.write_current_config_to_info()

    def write_current_config_to_info(self):
        self.write_info("\nConfiguración actual:\n")
        self.write_info(f"  app_dir:                       {self.app_dir}\n")
        self.write_info(f"  output_dir:                    {self.config['output_dir']}\n")
        self.write_info(f"  reference_dir:                 {self.config['reference_dir']}\n")
        self.write_info(f"  homography_config:             {self.config['homography_config']}\n")
        self.write_info(f"  orientation_template:          {self.config['orientation_template']}\n")
        self.write_info(f"  model_path:                    {self.config['model_path']}\n")
        self.write_info(f"  camera_source:                 {self.config['camera_source']}\n")
        self.write_info(f"  camera_width:                  {self.config['camera_width']}\n")
        self.write_info(f"  camera_height:                 {self.config['camera_height']}\n")
        self.write_info(f"  homography_method:             {self.config['homography_method']}\n")
        self.write_info(f"  yolo_conf:                     {self.config['yolo_conf']}\n")
        self.write_info(f"  max_center_distance:           {self.config['max_center_distance']}\n")
        self.write_info(f"  max_center_distance_relaxed:   {self.config['max_center_distance_relaxed']}\n")
        self.write_info(f"  limit:                         {self.config['limit']}\n")
        self.write_info(f"  duration:                      {self.config['duration']}\n")
        self.write_info(f"  interval:                      {self.config['interval']}\n")
        self.write_info(f"  treat_extra_as_error:          {self.config['treat_extra_as_error']}\n")

    def write_info(self, text):
        self.info_text.insert("end", text)
        self.info_text.see("end")
        self.root.update_idletasks()

    def clear_info(self):
        self.info_text.delete("1.0", "end")

    def set_status(self, text, color):
        self.status_label.configure(text=text, bg=color)

    def set_buttons_state(self, state):
        self.btn_analyze.configure(state=state)
        self.btn_open_folder.configure(state=state)
        self.btn_show_raw.configure(state=state)
        self.btn_show_corrected.configure(state=state)
        self.btn_show_failures.configure(state=state)
        self.btn_show_overlay.configure(state=state)

        if hasattr(self, "btn_camera_test"):
            self.btn_camera_test.configure(state=state)

    def clear_canvas(self):
        self.current_image = None
        self.current_photo = None
        self.image_canvas.delete("all")

    def clear_camera_test_canvas(self):
        self.camera_test_image = None
        self.camera_test_photo = None
        self.camera_test_canvas.delete("all")

    def clean_previous_latest_outputs(self):
        paths_to_remove = [
            self.raw_image_path,
            self.corrected_image_path,
            self.failures_image_path,
            self.normal_overlay_path,
            self.comparison_csv_path,
            self.components_csv_path,
        ]

        for p in paths_to_remove:
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                self.write_info(f"AVISO: no pude borrar {p}: {e}\n")

        failed_dir = self.output_dir / "failed"
        if failed_dir.exists():
            try:
                for p in failed_dir.glob("latest*"):
                    if p.is_file():
                        p.unlink()
            except Exception as e:
                self.write_info(f"AVISO: no pude limpiar failed/: {e}\n")

        try:
            if self.summary_csv_path.exists():
                self.summary_csv_path.unlink()
        except Exception as e:
            self.write_info(f"AVISO: no pude borrar summary_realtime.csv: {e}\n")

    def clean_previous_camera_test_outputs(self):
        try:
            if self.camera_test_image_path.exists():
                self.camera_test_image_path.unlink()
        except Exception as e:
            self.write_info(f"AVISO: no pude borrar captura de test anterior: {e}\n")

    def start_camera_test(self):
        if self.is_processing:
            return

        if not self.camera_test_script.exists():
            messagebox.showerror(
                "Error",
                f"No encuentro el script de test de cámara:\n{self.camera_test_script}"
            )
            return

        if not self.read_config_from_widgets(validate=False):
            return

        if not self.camera_source_var.get().strip():
            messagebox.showerror(
                "Cámara no configurada",
                "Debes indicar una fuente de cámara: 0, 1, /dev/video0..."
            )
            self.notebook.select(self.tab_camera)
            return

        self.refresh_paths()
        self.camera_test_image_path.parent.mkdir(parents=True, exist_ok=True)
        self.clean_previous_camera_test_outputs()
        self.clear_camera_test_canvas()

        self.is_processing = True
        self.set_buttons_state("disabled")
        self.set_status("TEST CÁMARA...", "#005a9c")
        self.camera_test_status_label.configure(text="Capturando imagen de prueba...")

        self.last_camera_test_start_time = time.time()

        self.write_info("\nIniciando TEST de cámara...\n")
        self.write_info(f"  Cámara:      {self.config['camera_source']}\n")
        self.write_info(f"  Resolución:  {self.config['camera_width']}x{self.config['camera_height']}\n")
        self.write_info(f"  Salida:      {self.camera_test_image_path}\n\n")

        self.camera_test_thread = threading.Thread(
            target=self.run_camera_test_once,
            daemon=True
        )
        self.camera_test_thread.start()

    def run_camera_test_once(self):
        start_time = time.perf_counter()

        cmd = [
            str(self.camera_test_script),
            "--camera-source", str(self.config["camera_source"]),
            "--camera-width", str(int(self.config["camera_width"])),
            "--camera-height", str(int(self.config["camera_height"])),
            "--output-path", str(self.camera_test_image_path),
        ]

        try:
            process = subprocess.run(
                cmd,
                cwd=str(self.app_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False
            )

            elapsed = time.perf_counter() - start_time

            self.root.after(
                0,
                lambda: self.on_camera_test_finished(process.returncode, process.stdout, elapsed, cmd)
            )

        except Exception as e:
            self.root.after(
                0,
                lambda: self.on_camera_test_error(str(e))
            )

    def on_camera_test_error(self, error_text):
        self.is_processing = False
        self.set_buttons_state("normal")
        self.set_status("ERROR CÁMARA", "#aa0000")
        self.camera_test_status_label.configure(text="ERROR en test de cámara")
        self.write_info("\nERROR ejecutando test de cámara:\n")
        self.write_info(error_text + "\n")

    def on_camera_test_finished(self, returncode, output, elapsed, cmd):
        self.is_processing = False
        self.set_buttons_state("normal")

        self.write_info("Comando TEST cámara:\n")
        self.write_info(" ".join(cmd) + "\n\n")

        self.write_info("Salida TEST cámara:\n")
        self.write_info(output)
        self.write_info("\n")
        self.write_info(f"Tiempo TEST cámara: {elapsed:.3f} s\n\n")

        if returncode != 0:
            self.set_status("ERROR CÁMARA", "#aa0000")
            self.camera_test_status_label.configure(text="ERROR: no se pudo capturar imagen")
            return

        if not self.camera_test_image_path.exists():
            self.set_status("SIN CAPTURA", "#aa0000")
            self.camera_test_status_label.configure(text="ERROR: no se generó imagen de test")
            self.write_info(f"No existe la captura esperada:\n{self.camera_test_image_path}\n")
            return

        if not self.is_camera_test_file_new():
            self.set_status("CAPTURA ANTIGUA", "#aa0000")
            self.camera_test_status_label.configure(text="ERROR: la imagen de test no parece nueva")
            return

        self.set_status("CÁMARA OK", "#008000")
        self.camera_test_status_label.configure(
            text=f"Captura OK: {self.camera_test_image_path}"
        )
        self.load_camera_test_image(self.camera_test_image_path)

    def is_camera_test_file_new(self):
        try:
            return self.camera_test_image_path.stat().st_mtime >= self.last_camera_test_start_time - 1.0
        except Exception:
            return False

    def start_analysis(self):
        if self.is_processing:
            return

        if not self.pipeline_script.exists():
            messagebox.showerror(
                "Error",
                f"No encuentro el script:\n{self.pipeline_script}"
            )
            return

        if not self.read_config_from_widgets(validate=True):
            return

        self.refresh_paths()

        self.is_processing = True
        self.set_buttons_state("disabled")
        self.set_status("PROCESANDO...", "#b36b00")
        self.clear_info()
        self.clear_canvas()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clean_previous_latest_outputs()

        self.last_run_start_time = time.time()

        self.write_info("Iniciando análisis de una placa...\n")
        self.write_info("Se borran los latest anteriores para evitar imágenes antiguas.\n\n")
        self.write_current_config_to_info()
        self.write_info("\n")

        self.worker_thread = threading.Thread(
            target=self.run_pipeline_once,
            daemon=True
        )
        self.worker_thread.start()

    def build_pipeline_command(self):
        cmd = [
            str(self.pipeline_script),
            "--camera-source", str(self.config["camera_source"]),
            "--camera-width", str(int(self.config["camera_width"])),
            "--camera-height", str(int(self.config["camera_height"])),
            "--output-dir", str(self.output_dir),
            "--reference-dir", str(Path(self.config["reference_dir"])),
            "--config", str(Path(self.config["homography_config"])),
            "--orientation-template", str(Path(self.config["orientation_template"])),
            "--component-model", str(Path(self.config["model_path"])),
            "--limit", str(int(self.config["limit"])),
            "--interval", str(float(self.config["interval"])),
            "--homography-method", str(self.config["homography_method"]),
            "--conf", str(float(self.config["yolo_conf"])),
            "--max-center-distance", str(float(self.config["max_center_distance"])),
            "--max-center-distance-relaxed", str(float(self.config["max_center_distance_relaxed"])),
        ]

        duration = float(self.config["duration"])
        if duration > 0:
            cmd.extend(["--duration", str(duration)])

        return cmd

    def run_pipeline_once(self):
        start_time = time.perf_counter()
        cmd = self.build_pipeline_command()

        try:
            process = subprocess.run(
                cmd,
                cwd=str(self.app_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False
            )

            elapsed = time.perf_counter() - start_time

            self.root.after(
                0,
                lambda: self.on_pipeline_finished(process.returncode, process.stdout, elapsed, cmd)
            )

        except Exception as e:
            self.root.after(
                0,
                lambda: self.on_pipeline_error(str(e))
            )

    def on_pipeline_error(self, error_text):
        self.is_processing = False
        self.set_buttons_state("normal")
        self.set_status("ERROR", "#aa0000")
        self.write_info("\nERROR ejecutando el pipeline:\n")
        self.write_info(error_text + "\n")

    def on_pipeline_finished(self, returncode, output, elapsed, cmd):
        self.is_processing = False
        self.set_buttons_state("normal")

        self.write_info("Comando ejecutado:\n")
        self.write_info(" ".join(cmd) + "\n\n")

        self.write_info("Salida del pipeline:\n")
        self.write_info(output)
        self.write_info("\n")
        self.write_info(f"Tiempo total GUI/pipeline: {elapsed:.3f} s\n\n")

        if returncode != 0:
            self.set_status("ERROR PIPELINE", "#aa0000")
            self.write_info(f"El pipeline terminó con código: {returncode}\n")
            self.show_failure_text_if_exists()
            return

        summary = self.read_last_summary()

        if summary is None:
            self.set_status("SIN RESUMEN", "#aa0000")
            self.write_info("No he podido leer un resumen nuevo.\n")
            self.show_failure_text_if_exists()
            return

        if not self.failures_image_path.exists():
            self.set_status("SIN IMAGEN", "#aa0000")
            self.write_info("El pipeline terminó, pero no se generó latest_failures.jpg.\n")
            self.write_info(f"Esperado: {self.failures_image_path}\n")
            self.show_failure_text_if_exists()
            return

        if not self.is_file_newer_than_start(self.failures_image_path):
            self.set_status("IMAGEN ANTIGUA", "#aa0000")
            self.write_info("La imagen latest_failures.jpg existe, pero no parece haberse regenerado en este análisis.\n")
            self.write_info(f"Imagen: {self.failures_image_path}\n")
            return

        self.update_status_from_summary(summary)
        self.show_failures_image()
        self.show_summary_info(summary)

    def show_failure_text_if_exists(self):
        failed_dir = self.output_dir / "failed"
        error_file = failed_dir / "latest_error.txt"

        if error_file.exists():
            try:
                self.write_info("\nÚltimo error del pipeline:\n")
                self.write_info(error_file.read_text(encoding="utf-8"))
                self.write_info("\n")
            except Exception as e:
                self.write_info(f"No pude leer {error_file}: {e}\n")

    def is_file_newer_than_start(self, path):
        try:
            return path.stat().st_mtime >= self.last_run_start_time - 1.0
        except Exception:
            return False

    def read_last_summary(self):
        if not self.summary_csv_path.exists():
            return None

        try:
            with open(self.summary_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return None

            return rows[-1]

        except Exception as e:
            self.write_info(f"ERROR leyendo resumen: {e}\n")
            return None

    def update_status_from_summary(self, summary):
        missing = self.to_int(summary.get("missing", 0))
        misplaced = self.to_int(summary.get("misplaced", 0))
        extra = self.to_int(summary.get("extra", 0))

        treat_extra_as_error = bool(self.config.get("treat_extra_as_error", False))

        if missing == 0 and misplaced == 0:
            if extra > 0 and treat_extra_as_error:
                self.set_status("PLACA MAL", "#b00000")
            elif extra > 0:
                self.set_status("PLACA OK*", "#008000")
            else:
                self.set_status("PLACA OK", "#008000")
        else:
            self.set_status("PLACA MAL", "#b00000")

    def show_summary_info(self, summary):
        status = summary.get("status", "").strip()
        ok = self.to_int(summary.get("ok", 0))
        missing = self.to_int(summary.get("missing", 0))
        misplaced = self.to_int(summary.get("misplaced", 0))
        extra = self.to_int(summary.get("extra", 0))

        treat_extra_as_error = bool(self.config.get("treat_extra_as_error", False))
        gui_ok = missing == 0 and misplaced == 0 and not (extra > 0 and treat_extra_as_error)

        self.write_info("Resumen del último análisis:\n")
        self.write_info(f"  status pipeline:  {status}\n")
        self.write_info(f"  status GUI:       {'PLACA OK' if gui_ok else 'PLACA MAL'}\n")
        self.write_info(f"  frame_idx:        {summary.get('frame_idx', '')}\n")
        self.write_info(f"  OK:               {ok}\n")
        self.write_info(f"  MISSING:          {missing}\n")
        self.write_info(f"  MISPLACED:        {misplaced}\n")
        self.write_info(f"  EXTRA:            {extra}\n")
        self.write_info(f"  detections:       {summary.get('detections', '')}\n")

        self.write_info("\nConfiguración usada:\n")
        self.write_info(f"  Modelo:                 {self.config['model_path']}\n")
        self.write_info(f"  Cámara:                 {self.config['camera_source']}\n")
        self.write_info(f"  Resolución cámara:      {self.config['camera_width']}x{self.config['camera_height']}\n")
        self.write_info(f"  Config homografía:      {self.config['homography_config']}\n")
        self.write_info(f"  YOLO conf:              {self.config['yolo_conf']}\n")
        self.write_info(f"  max center distance:    {self.config['max_center_distance']}\n")
        self.write_info(f"  relaxed center dist:    {self.config['max_center_distance_relaxed']}\n")
        self.write_info(f"  homography method:      {self.config['homography_method']}\n")
        self.write_info(f"  EXTRA como fallo:       {self.config['treat_extra_as_error']}\n")

        if extra > 0 and not treat_extra_as_error:
            self.write_info("\nAVISO:\n")
            self.write_info("  Hay detecciones EXTRA, pero no se consideran fallo de placa.\n")

        self.write_info("\nTiempos:\n")
        self.write_info(f"  homography:       {summary.get('homography_time_s', '')} s\n")
        self.write_info(f"  orientation:      {summary.get('orientation_time_s', '')} s\n")
        self.write_info(f"  yolo:             {summary.get('yolo_time_s', '')} s\n")
        self.write_info(f"  comparison:       {summary.get('comparison_time_s', '')} s\n")
        self.write_info(f"  total:            {summary.get('total_time_s', '')} s\n")

        self.write_info("\nFicheros:\n")
        self.write_info(f"  raw:              {self.raw_image_path}\n")
        self.write_info(f"  corrected:        {self.corrected_image_path}\n")
        self.write_info(f"  failures image:   {self.failures_image_path}\n")
        self.write_info(f"  overlay completo: {self.normal_overlay_path}\n")
        self.write_info(f"  comparison csv:   {self.comparison_csv_path}\n")
        self.write_info(f"  components csv:   {self.components_csv_path}\n")

    def show_raw_image(self):
        self.load_image(self.raw_image_path)

    def show_corrected_image(self):
        self.load_image(self.corrected_image_path)

    def show_failures_image(self):
        self.load_image(self.failures_image_path)

    def show_normal_overlay_image(self):
        self.load_image(self.normal_overlay_path)

    def load_image(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            self.write_info(f"No existe imagen: {image_path}\n")
            return

        try:
            with Image.open(image_path) as img:
                self.current_image = img.convert("RGB").copy()

            self.render_image()
            mtime = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(image_path.stat().st_mtime)
            )
            self.write_info(f"Imagen mostrada: {image_path}\n")
            self.write_info(f"Fecha imagen:    {mtime}\n")

        except Exception as e:
            self.write_info(f"ERROR abriendo imagen {image_path}: {e}\n")

    def load_camera_test_image(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            self.write_info(f"No existe imagen test cámara: {image_path}\n")
            return

        try:
            with Image.open(image_path) as img:
                self.camera_test_image = img.convert("RGB").copy()

            self.render_camera_test_image()

            mtime = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(image_path.stat().st_mtime)
            )

            self.write_info(f"Imagen TEST cámara mostrada: {image_path}\n")
            self.write_info(f"Fecha imagen TEST:           {mtime}\n")

        except Exception as e:
            self.write_info(f"ERROR abriendo imagen de test {image_path}: {e}\n")

    def render_image(self):
        if self.current_image is None:
            return

        canvas_w = self.image_canvas.winfo_width()
        canvas_h = self.image_canvas.winfo_height()

        if canvas_w <= 5 or canvas_h <= 5:
            return

        img_w, img_h = self.current_image.size

        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = self.current_image.resize((new_w, new_h), Image.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(resized)

        self.image_canvas.delete("all")

        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2

        self.image_canvas.create_image(
            x,
            y,
            anchor="nw",
            image=self.current_photo
        )

    def render_camera_test_image(self):
        if self.camera_test_image is None:
            return

        canvas_w = self.camera_test_canvas.winfo_width()
        canvas_h = self.camera_test_canvas.winfo_height()

        if canvas_w <= 5 or canvas_h <= 5:
            return

        img_w, img_h = self.camera_test_image.size

        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = self.camera_test_image.resize((new_w, new_h), Image.LANCZOS)
        self.camera_test_photo = ImageTk.PhotoImage(resized)

        self.camera_test_canvas.delete("all")

        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2

        self.camera_test_canvas.create_image(
            x,
            y,
            anchor="nw",
            image=self.camera_test_photo
        )

    def on_canvas_resize(self, event):
        self.render_image()

    def on_camera_test_canvas_resize(self, event):
        self.render_camera_test_image()

    def open_results_folder(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.Popen(["xdg-open", str(self.output_dir)])
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"No se pudo abrir la carpeta:\n{self.output_dir}\n\n{e}"
            )

    @staticmethod
    def to_int(value):
        try:
            if value is None or value == "":
                return 0
            return int(float(value))
        except Exception:
            return 0


def main():
    root = tk.Tk()
    PCBInspectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
