from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pyproj import Transformer
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
VIDEO_DIR = PROJECT_ROOT / "Separate_4by3"
CSV_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SETTINGS_FILE = APP_DIR / "calibration_gui_v4_settings.json"
LEGACY_SETTINGS_FILE = PROJECT_ROOT / "calibration_tool" / "calibration_gui_settings.json"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v",".dav")


def resolve_settings_path(value: Any, default: Path) -> Path:
    if not value:
        return default
    path = Path(str(value))
    return path if path.is_absolute() else APP_DIR / path


def portable_settings_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(APP_DIR))
    except ValueError:
        return str(path)


def location_key(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    match = re.match(r"^(.*?)_main_", stem, flags=re.IGNORECASE)
    prefix = match.group(1) if match else stem
    return re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_")


def display_location(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    match = re.match(r"^(.*?)_main_", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


def transform_norm_points(points_norm: list[list[float]], h_matrix: list[list[float]]) -> list[list[float]]:
    pts = np.array(points_norm, dtype=np.float32).reshape(-1, 1, 2)
    h = np.array(h_matrix, dtype=np.float64)
    world = cv2.perspectiveTransform(pts, h).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in world]


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


def base_to_final_world_matrix(pixel_axis_info: dict[str, Any] | None) -> np.ndarray:
    if not pixel_axis_info:
        return np.eye(3, dtype=np.float64)
    origin = pixel_axis_info["origin_base_m"]
    x_axis = pixel_axis_info["x_axis_base_unit"]
    y_axis = pixel_axis_info["y_axis_base_unit"]
    origin_x = float(origin[0])
    origin_y = float(origin[1])
    x0 = float(x_axis[0])
    x1 = float(x_axis[1])
    y0 = float(y_axis[0])
    y1 = float(y_axis[1])
    return np.array(
        [
            [x0, x1, -(origin_x * x0 + origin_y * x1)],
            [y0, y1, -(origin_x * y0 + origin_y * y1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def final_homography_matrices(
    h_matrix: list[list[float]],
    pixel_axis_info: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, list[list[float]]]:
    h_base_norm = np.array(h_matrix, dtype=np.float64)
    base_to_final = base_to_final_world_matrix(pixel_axis_info)
    h_final_norm = base_to_final @ h_base_norm
    norm_from_px = np.array(
        [
            [1.0 / float(width), 0.0, 0.0],
            [0.0, 1.0 / float(height), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return {
        "base_to_final_world": matrix_to_list(base_to_final),
        "image_norm_to_final_world": matrix_to_list(h_final_norm),
        "image_px_to_final_world": matrix_to_list(h_final_norm @ norm_from_px),
    }


def gps_to_utm(lat: float, lon: float) -> dict[str, Any]:
    zone = int((lon + 180) / 6) + 1
    north = lat >= 0
    epsg = 32600 + zone if north else 32700 + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return {
        "lat": float(lat),
        "lon": float(lon),
        "easting": float(easting),
        "northing": float(northing),
        "zone": int(zone),
        "north": bool(north),
        "epsg": int(epsg),
    }


def parse_lat_lon(text: str) -> tuple[float, float]:
    value = text.strip()
    dms_pattern = re.compile(
        r"([+-]?\d+(?:\.\d+)?)\s*[°º]\s*"
        r"(?:(\d+(?:\.\d+)?)\s*['′’]\s*)?"
        r"(?:(\d+(?:\.\d+)?)\s*[\"″”]\s*)?"
        r"([NSEW])",
        flags=re.IGNORECASE,
    )
    dms_matches = list(dms_pattern.finditer(value))
    if dms_matches:
        if len(dms_matches) != 2:
            raise ValueError("Enter one latitude and one longitude in DMS format.")
        coordinates: dict[str, float] = {}
        for match in dms_matches:
            degrees = abs(float(match.group(1)))
            minutes = float(match.group(2) or 0.0)
            seconds = float(match.group(3) or 0.0)
            direction = match.group(4).upper()
            if minutes >= 60 or seconds >= 60:
                raise ValueError("DMS minutes and seconds must be less than 60.")
            decimal = degrees + minutes / 60.0 + seconds / 3600.0
            if direction in {"S", "W"}:
                decimal = -decimal
            key = "lat" if direction in {"N", "S"} else "lon"
            if key in coordinates:
                raise ValueError("DMS input must contain one N/S latitude and one E/W longitude.")
            coordinates[key] = decimal
        if "lat" not in coordinates or "lon" not in coordinates:
            raise ValueError("DMS input must contain one N/S latitude and one E/W longitude.")
        lat, lon = coordinates["lat"], coordinates["lon"]
    else:
        parts = [p.strip() for p in re.split(r"[, ]+", value) if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                "Use decimal coordinates (23.742866, 90.395765) or Google Earth DMS "
                "(23°44'44.61\"N 90°23'41.02\"E)."
            )
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError as exc:
            raise ValueError(
                "Use decimal coordinates (23.742866, 90.395765) or Google Earth DMS "
                "(23°44'44.61\"N 90°23'41.02\"E)."
            ) from exc
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Latitude must be -90..90 and longitude must be -180..180.")
    return lat, lon


def world_to_gps(world_x: float, world_y: float, origin_info: dict[str, Any]) -> tuple[float, float]:
    epsg = int(origin_info["epsg"])
    x_axis = origin_info.get("x_axis_utm")
    y_axis = origin_info.get("y_axis_utm")
    if x_axis and y_axis:
        easting = float(origin_info["E0"]) + world_x * float(x_axis[0]) + world_y * float(y_axis[0])
        northing = float(origin_info["N0"]) + world_x * float(x_axis[1]) + world_y * float(y_axis[1])
    else:
        easting = float(origin_info["E0"]) + world_x
        northing = float(origin_info["N0"]) + world_y
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return float(lat), float(lon)


class CalibrationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video Homography Calibration V5")
        self.geometry("1280x820")
        self.minsize(720, 500)

        self.source_video_dir = VIDEO_DIR
        self.output_dir = OUTPUT_DIR
        self.video_paths: list[Path] = []
        self.csv_paths: list[Path] = []

        self.current_video: Path | None = None
        self.current_frame_index = tk.IntVar(value=0)
        self.current_frame_rgb: np.ndarray | None = None
        self.current_image_size: tuple[int, int] | None = None
        self.current_fps: float | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.tk_image_size: tuple[int, int] | None = None
        self.tk_image_frame_token: int | None = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.h_matrix: list[list[float]] | None = None
        self.mean_error_m: float | None = None
        self.reprojection_errors_m: list[float] = []
        self.inspect_marker_px: list[float] | None = None
        self.inspect_world_m: list[float] | None = None
        self.origin_info: dict[str, Any] | None = None
        self.measure_points_px: list[list[float]] = []
        self.measure_distance_m: float | None = None
        self.drag_point_index: int | None = None
        self.active_drag_index: int | None = None
        self.pixel_axis_drag_handle: str | None = None
        self.panning = False
        self.drag_started = False
        self.loading_json = False
        self.autosave_job: str | None = None
        self.autosave_dirty = False
        self.grid_redraw_job: str | None = None
        self.drag_redraw_job: str | None = None
        self.gps_axis_definition: dict[str, Any] | None = None
        self.pixel_axis_info: dict[str, Any] | None = None
        self.axis_capture: dict[str, Any] | None = None
        self.calibration_input_type: str | None = None

        self.state: dict[str, Any] = {
            "calibration_points": [],
            "active_area_polygon_px": [],
            "start_line_px": [],
        }
        self.history: list[dict[str, Any]] = []
        self.redo_history: list[dict[str, Any]] = []

        self.mode = tk.StringVar(value="calibrate")
        self.coord_input_mode = tk.StringVar(value="xy")
        self.source_folder_var = tk.StringVar(value="")
        self.output_folder_var = tk.StringVar(value="")
        self.show_grid_var = tk.BooleanVar(value=False)
        self.autosave_var = tk.BooleanVar(value=False)
        self.autosave_interval_var = tk.IntVar(value=10)
        self.grid_spacing_var = tk.DoubleVar(value=5.0)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.name_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Load a video frame to begin.")
        self.h_status_var = tk.StringVar(value="Homography: not computed")
        self.inspect_var = tk.StringVar(value="Inspect: click a point after homography is ready.")

        self.load_app_settings()
        self.save_app_settings()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._build_ui()
        self.bind_all("<Control-s>", self.on_ctrl_s)
        self.bind_all("<Control-S>", self.on_ctrl_s)
        self.bind_all("<Control-z>", self.on_ctrl_z)
        self.bind_all("<Control-Z>", self.on_ctrl_z)
        self.bind_all("<Control-y>", self.on_ctrl_y)
        self.bind_all("<Control-Y>", self.on_ctrl_y)
        self.load_video_sources(initial=True)
        if self.autosave_var.get():
            self.ensure_autosave_timer()

    def load_app_settings(self) -> None:
        source_dir = VIDEO_DIR
        output_dir = OUTPUT_DIR
        settings_to_read = SETTINGS_FILE if SETTINGS_FILE.exists() else LEGACY_SETTINGS_FILE
        if settings_to_read.exists():
            try:
                with open(settings_to_read, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                source_dir = resolve_settings_path(data.get("source_video_folder"), source_dir)
                output_dir = resolve_settings_path(data.get("output_folder"), output_dir)
                self.autosave_var.set(bool(data.get("autosave_json", False)))
                self.autosave_interval_var.set(max(2, int(data.get("autosave_interval_seconds", 10))))
            except (OSError, json.JSONDecodeError):
                pass
        self.source_video_dir = source_dir
        self.output_dir = output_dir
        self.source_folder_var.set(str(source_dir))
        self.output_folder_var.set(str(output_dir))

    def save_app_settings(self) -> None:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_video_folder": portable_settings_path(self.source_video_dir),
                    "output_folder": portable_settings_path(self.output_dir),
                    "autosave_json": bool(self.autosave_var.get()),
                    "autosave_interval_seconds": max(2, int(self.autosave_interval_var.get())),
                },
                f,
                indent=2,
            )

    def browse_source_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Select source video folder",
            initialdir=str(self.source_video_dir if self.source_video_dir.exists() else PROJECT_ROOT),
        )
        if not folder:
            return
        self.source_folder_var.set(folder)

    def browse_output_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Select output JSON folder",
            initialdir=str(self.output_dir if self.output_dir.exists() else PROJECT_ROOT),
        )
        if not folder:
            return
        self.output_folder_var.set(folder)

    def reset_default_folders(self) -> None:
        self.source_folder_var.set(str(VIDEO_DIR))
        self.output_folder_var.set(str(OUTPUT_DIR))
        self.load_video_sources()

    def load_video_sources(self, initial: bool = False) -> None:
        source_dir = Path(self.source_folder_var.get().strip() or VIDEO_DIR)
        output_dir = Path(self.output_folder_var.get().strip() or OUTPUT_DIR)
        if not source_dir.exists() or not source_dir.is_dir():
            if not initial:
                messagebox.showerror("Source folder not found", f"Could not find source video folder:\n{source_dir}")
            return

        self.source_video_dir = source_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_folder_var.set(str(self.source_video_dir))
        self.output_folder_var.set(str(self.output_dir))
        self.save_app_settings()

        self.video_paths = sorted(
            [p for p in self.source_video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )

        csv_candidates: list[Path] = []
        for csv_dir in [self.source_video_dir, CSV_DIR, self.source_video_dir.parent / "data"]:
            if csv_dir.exists() and csv_dir.is_dir():
                csv_candidates.extend(sorted(csv_dir.glob("*.csv"), key=lambda p: p.name.lower()))
        seen_csv = set()
        self.csv_paths = []
        for path in csv_candidates:
            key = str(path.resolve()).lower()
            if key not in seen_csv:
                seen_csv.add(key)
                self.csv_paths.append(path)

        self.reset_calibration_workspace(clear_frame=True)
        self.current_video = None
        self._refresh_file_lists()
        if self.video_paths:
            self.status_var.set(f"Loaded {len(self.video_paths)} video(s). Select a video or use the first one.")
        elif not initial:
            self.status_var.set("No supported videos found in the selected source folder.")

    def _build_ui(self) -> None:
        root = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True)

        left_shell = ttk.Frame(root, padding=8, width=340)
        center = ttk.Frame(root, padding=(0, 8, 8, 8))
        right = ttk.Frame(root, padding=8, width=380)
        root.add(left_shell, weight=0)
        root.add(center, weight=1)
        root.add(right, weight=0)

        left_canvas = tk.Canvas(left_shell, width=320, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_shell, orient=tk.VERTICAL, command=left_canvas.yview)
        left = ttk.Frame(left_canvas)
        left.bind("<Configure>", lambda event: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left, anchor=tk.NW)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        folders = ttk.LabelFrame(left, text="Step 1-2: Folders", padding=8)
        folders.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(folders, text="Video source folder").pack(anchor=tk.W)
        source_row = ttk.Frame(folders)
        source_row.pack(fill=tk.X, pady=(2, 6))
        ttk.Entry(source_row, textvariable=self.source_folder_var, width=34).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(source_row, text="Browse", command=self.browse_source_folder).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(folders, text="Output JSON folder").pack(anchor=tk.W)
        output_row = ttk.Frame(folders)
        output_row.pack(fill=tk.X, pady=(2, 8))
        ttk.Entry(output_row, textvariable=self.output_folder_var, width=34).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_row, text="Browse", command=self.browse_output_folder).pack(side=tk.LEFT, padx=(4, 0))

        folder_buttons = ttk.Frame(folders)
        folder_buttons.pack(fill=tk.X)
        ttk.Button(folder_buttons, text="Load", command=self.load_video_sources).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(folder_buttons, text="Reset defaults", command=self.reset_default_folders).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        video_step = ttk.LabelFrame(left, text="Step 3: Video", padding=8)
        video_step.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(video_step, text="Source video").pack(anchor=tk.W)
        self.video_combo = ttk.Combobox(video_step, width=44, state="readonly")
        self.video_combo.pack(fill=tk.X, pady=(2, 8))
        self.video_combo.bind("<<ComboboxSelected>>", self._on_video_selected)

        frame_row = ttk.Frame(video_step)
        frame_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(frame_row, text="Frame").pack(side=tk.LEFT)
        ttk.Spinbox(frame_row, from_=0, to=9999999, textvariable=self.current_frame_index, width=10).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(frame_row, text="Load", command=self.load_frame).pack(side=tk.LEFT)
        ttk.Button(frame_row, text="Open Current JSON", command=self.open_current_json).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(video_step, text="JSON name follows selected video").pack(anchor=tk.W)
        ttk.Entry(video_step, textvariable=self.name_var, width=44, state="readonly").pack(fill=tk.X, pady=(2, 0))

        coord_step = ttk.LabelFrame(left, text="Step 4: Coordinate Input", padding=8)
        coord_step.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(coord_step, text="World X/Y meters", value="xy", variable=self.coord_input_mode, command=self.schedule_autosave).pack(anchor=tk.W)
        ttk.Radiobutton(coord_step, text="GPS latitude/longitude", value="gps", variable=self.coord_input_mode, command=self.schedule_autosave).pack(anchor=tk.W)
        ttk.Button(coord_step, text="Set/Edit GPS Custom XY Axis", command=self.set_gps_custom_axis).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(coord_step, text="Clear Image Axis", command=self.clear_pixel_axis).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(coord_step, text="Flip Perpendicular Axis (+/-)", command=self.flip_perpendicular_axis).pack(
            fill=tk.X, pady=(6, 0)
        )
        ttk.Button(coord_step, text="GPS / XY Axis Guide", command=self.show_xy_axis_help).pack(fill=tk.X, pady=(6, 0))

        mode_step = ttk.LabelFrame(left, text="Step 5: Click Mode", padding=8)
        mode_step.pack(fill=tk.X, pady=(0, 8))
        for text, value in [
            ("Add calibration point", "calibrate"),
            ("Measure distance", "measure"),
            ("Draw active area", "active"),
            ("Set start line", "start"),
            ("Set/edit axis from image", "axis"),
            ("Inspect real-world coordinate", "inspect"),
        ]:
            ttk.Radiobutton(mode_step, text=text, value=value, variable=self.mode, command=self.on_mode_changed).pack(anchor=tk.W)

        tools_step = ttk.LabelFrame(left, text="Step 6: View And Tools", padding=8)
        tools_step.pack(fill=tk.X, pady=(0, 8))
        zoom_row = ttk.Frame(tools_step)
        zoom_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(zoom_row, text="Zoom -", command=self.zoom_out).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(zoom_row, text="Fit", command=self.zoom_fit).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(zoom_row, text="Zoom +", command=self.zoom_in).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.zoom_label = ttk.Label(tools_step, text="Zoom: 100%")
        self.zoom_label.pack(anchor=tk.W, pady=(0, 4))

        grid_row = ttk.Frame(tools_step)
        grid_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(grid_row, text="Show plane grid", variable=self.show_grid_var, command=self.on_view_changed).pack(
            side=tk.LEFT
        )
        ttk.Label(grid_row, text="m").pack(side=tk.RIGHT)
        self.grid_spacing_spinbox = ttk.Spinbox(
            grid_row,
            from_=0.5,
            to=100.0,
            increment=0.5,
            textvariable=self.grid_spacing_var,
            width=6,
            command=self.on_grid_spacing_changed,
        )
        self.grid_spacing_spinbox.pack(side=tk.RIGHT, padx=(4, 2))
        self.grid_spacing_spinbox.bind("<KeyRelease>", self.on_grid_spacing_changed)
        self.grid_spacing_spinbox.bind("<FocusOut>", self.on_grid_spacing_changed)
        self.grid_spacing_spinbox.bind("<Return>", self.on_grid_spacing_changed)

        autosave_row = ttk.Frame(tools_step)
        autosave_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(
            autosave_row, text="Autosave", variable=self.autosave_var, command=self.on_autosave_toggle
        ).pack(side=tk.LEFT)
        ttk.Label(autosave_row, text="every").pack(side=tk.LEFT, padx=(10, 4))
        autosave_interval = ttk.Spinbox(
            autosave_row,
            from_=2,
            to=3600,
            increment=1,
            textvariable=self.autosave_interval_var,
            width=6,
            command=self.on_autosave_interval_changed,
        )
        autosave_interval.pack(side=tk.LEFT)
        autosave_interval.bind("<FocusOut>", self.on_autosave_interval_changed)
        autosave_interval.bind("<Return>", self.on_autosave_interval_changed)
        ttk.Label(autosave_row, text="seconds").pack(side=tk.LEFT, padx=(4, 0))

        action_grid = ttk.Frame(tools_step)
        action_grid.pack(fill=tk.X, pady=10)
        ttk.Button(action_grid, text="Compute H", command=self.compute_homography).grid(row=0, column=0, columnspan=2, sticky="ew", padx=2)
        ttk.Button(action_grid, text="Undo", command=self.undo).grid(row=1, column=0, sticky="ew", padx=2, pady=4)
        ttk.Button(action_grid, text="Redo", command=self.redo).grid(row=1, column=1, sticky="ew", padx=2, pady=4)
        ttk.Button(action_grid, text="Clear ROI", command=self.clear_active_area).grid(row=2, column=0, sticky="ew", padx=2)
        ttk.Button(action_grid, text="Clear Line", command=self.clear_start_line).grid(row=2, column=1, sticky="ew", padx=2)
        ttk.Button(action_grid, text="Clear Measure", command=self.clear_measure).grid(row=3, column=0, sticky="ew", padx=2, pady=4)
        ttk.Button(action_grid, text="Open Other JSON", command=self.open_json).grid(row=3, column=1, sticky="ew", padx=2, pady=4)
        ttk.Button(action_grid, text="Save Video JSON", command=self.save_json).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=2
        )
        action_grid.columnconfigure(0, weight=1)
        action_grid.columnconfigure(1, weight=1)

        status_step = ttk.LabelFrame(left, text="Status", padding=8)
        status_step.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(status_step, textvariable=self.h_status_var, wraplength=330).pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(status_step, textvariable=self.inspect_var, wraplength=330).pack(anchor=tk.W)

        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(center, bg="#16181d", highlightthickness=0)
        canvas_y = ttk.Scrollbar(center, orient=tk.VERTICAL, command=self.canvas.yview)
        canvas_x = ttk.Scrollbar(center, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=canvas_y.set, xscrollcommand=canvas_x.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        canvas_y.grid(row=0, column=1, sticky="ns")
        canvas_x.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel_zoom)
        self.canvas.bind("<Button-4>", self.on_mousewheel_zoom)
        self.canvas.bind("<Button-5>", self.on_mousewheel_zoom)
        self.canvas.bind("<ButtonPress-2>", self.on_middle_press)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_release)
        self.canvas.bind("<Control-ButtonPress-1>", self.on_pan_press)
        self.canvas.bind("<Control-B1-Motion>", self.on_pan_drag)
        self.canvas.bind("<Control-ButtonRelease-1>", self.on_pan_release)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel_pan)
        self.bind("<space>", self.on_space_press)
        self.bind("<KeyRelease-space>", self.on_space_release)
        self.canvas.bind("<Configure>", lambda _event: self.redraw_canvas())

        ttk.Label(right, text="Calibration Points").pack(anchor=tk.W)
        columns = ("idx", "px", "world", "axis_world", "input")
        self.point_tree = ttk.Treeview(right, columns=columns, show="headings", height=16)
        self.point_tree.heading("idx", text="#")
        self.point_tree.heading("px", text="Image px")
        self.point_tree.heading("world", text="Base world m")
        self.point_tree.heading("axis_world", text="New axis m")
        self.point_tree.heading("input", text="Input")
        self.point_tree.column("idx", width=36, anchor=tk.CENTER)
        self.point_tree.column("px", width=90)
        self.point_tree.column("world", width=100)
        self.point_tree.column("axis_world", width=100)
        self.point_tree.column("input", width=60)
        self.point_tree.pack(fill=tk.BOTH, expand=True)
        self.point_tree.bind("<Double-1>", lambda _event: self.edit_selected_point())

        point_buttons = ttk.Frame(right)
        point_buttons.pack(fill=tk.X, pady=6)
        ttk.Button(point_buttons, text="Edit Selected", command=self.edit_selected_point).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(point_buttons, text="Delete Selected", command=self.delete_selected_point).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        ttk.Separator(right).pack(fill=tk.X, pady=10)
        ttk.Label(right, text="Status").pack(anchor=tk.W)
        ttk.Label(right, textvariable=self.status_var, wraplength=300).pack(anchor=tk.W, pady=(2, 0))

    def _refresh_file_lists(self) -> None:
        names = [self.video_display_name(p) for p in self.video_paths]
        self.video_combo["values"] = names
        if names:
            self.video_combo.current(0)
            self._on_video_selected()
        else:
            self.video_combo.set("")
            self.name_var.set("")

    def video_display_name(self, path: Path) -> str:
        marker = "* " if any(candidate.exists() for candidate in self.json_paths_for_video(path)) else "  "
        return f"{marker}{path.name}"

    @staticmethod
    def safe_json_stem(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "calibration"

    def json_paths_for_video(self, video_path: Path) -> list[Path]:
        """Return canonical and legacy JSON names for a video."""
        exact_path = self.output_dir / f"{video_path.stem}.json"
        old_safe_path = self.output_dir / f"{self.safe_json_stem(video_path.stem)}.json"
        return [exact_path] if exact_path == old_safe_path else [exact_path, old_safe_path]

    def reset_calibration_workspace(self, clear_frame: bool = True) -> None:
        self.state = {
            "calibration_points": [],
            "active_area_polygon_px": [],
            "start_line_px": [],
        }
        self.history = []
        self.redo_history = []
        self.h_matrix = None
        self.mean_error_m = None
        self.reprojection_errors_m = []
        self.origin_info = None
        self.calibration_input_type = None
        self.inspect_marker_px = None
        self.inspect_world_m = None
        self.measure_points_px = []
        self.measure_distance_m = None
        self.drag_point_index = None
        self.active_drag_index = None
        self.pixel_axis_drag_handle = None
        self.panning = False
        self.drag_started = False
        self.gps_axis_definition = None
        self.pixel_axis_info = None
        self.axis_capture = None
        self.h_status_var.set("Homography: not computed")
        self.inspect_var.set("Inspect: click a point after homography is ready.")
        self.refresh_point_tree()
        if clear_frame:
            self.clear_loaded_frame()
        else:
            self.redraw_canvas()

    def clear_loaded_frame(self) -> None:
        self.current_frame_rgb = None
        self.current_image_size = None
        self.current_fps = None
        self.tk_image = None
        self.tk_image_size = None
        self.tk_image_frame_token = None
        self.inspect_marker_px = None
        self.inspect_world_m = None
        self.measure_points_px = []
        self.measure_distance_m = None
        self.redraw_canvas()

    def _on_video_selected(self, _event: tk.Event | None = None, sync_location: bool = True) -> None:
        idx = self.video_combo.current()
        if idx < 0:
            return
        selected_video = self.video_paths[idx]
        previous_video = self.current_video
        if previous_video != selected_video:
            self.reset_calibration_workspace(clear_frame=True)

        self.current_video = selected_video
        self.name_var.set(selected_video.stem)
        self.status_var.set(f"Started blank calibration for {selected_video.name}. Load a frame or open its JSON.")
        if not self.loading_json:
            loaded = self.try_auto_load_current_json()
            if not loaded:
                self.status_var.set(f"No saved JSON found for {selected_video.name}. Load a frame to begin.")

    def current_json_path(self) -> Path | None:
        if self.current_video is None:
            return None
        candidates = self.json_paths_for_video(self.current_video)
        return next((path for path in candidates if path.exists()), candidates[0])

    def try_auto_load_current_json(self) -> bool:
        path = self.current_json_path()
        if path is None or not path.exists():
            return False
        self.load_json_file(path)
        return True

    def push_history(self) -> None:
        self.history.append(self.history_snapshot())
        self.redo_history.clear()
        if len(self.history) > 100:
            self.history.pop(0)

    def history_snapshot(self) -> dict[str, Any]:
        return deepcopy(
            {
                "workspace_state": self.state,
                "origin_info": self.origin_info,
                "gps_axis_definition": self.gps_axis_definition,
                "pixel_axis_info": self.pixel_axis_info,
                "calibration_input_type": self.calibration_input_type,
            }
        )

    def restore_history_snapshot(self, snapshot: dict[str, Any]) -> None:
        # Accept history entries created by earlier v4 builds as well.
        if "workspace_state" not in snapshot:
            self.state = deepcopy(snapshot)
            return
        self.state = deepcopy(snapshot["workspace_state"])
        self.origin_info = deepcopy(snapshot.get("origin_info"))
        self.gps_axis_definition = deepcopy(snapshot.get("gps_axis_definition"))
        self.pixel_axis_info = deepcopy(snapshot.get("pixel_axis_info"))
        self.calibration_input_type = snapshot.get("calibration_input_type")

    def on_mode_changed(self) -> None:
        if self.mode.get() != "axis" and self.axis_capture:
            self.axis_capture = None
            self.redraw_canvas()
            self.status_var.set("Image-axis capture cancelled.")
        self.schedule_autosave()

    def on_view_changed(self) -> None:
        self.redraw_canvas()
        self.schedule_autosave()

    def on_autosave_toggle(self) -> None:
        self.save_app_settings()
        if self.autosave_var.get():
            self.autosave_dirty = True
            self.ensure_autosave_timer(restart=True)
        elif self.autosave_job is not None:
            self.after_cancel(self.autosave_job)
            self.autosave_job = None

    def on_autosave_interval_changed(self, _event: tk.Event | None = None) -> None:
        try:
            interval = max(2, int(self.autosave_interval_var.get()))
        except (tk.TclError, TypeError, ValueError):
            return
        self.autosave_interval_var.set(interval)
        self.save_app_settings()
        if self.autosave_var.get():
            self.ensure_autosave_timer(restart=True)

    def ensure_autosave_timer(self, restart: bool = False) -> None:
        if not self.autosave_var.get():
            return
        if restart and self.autosave_job is not None:
            self.after_cancel(self.autosave_job)
            self.autosave_job = None
        if self.autosave_job is None:
            try:
                seconds = max(2, int(self.autosave_interval_var.get()))
            except (tk.TclError, TypeError, ValueError):
                seconds = 10
            self.autosave_job = self.after(seconds * 1000, self.run_autosave)

    def schedule_autosave(self) -> None:
        if self.loading_json or not self.autosave_var.get():
            return
        self.autosave_dirty = True
        self.ensure_autosave_timer()

    def run_autosave(self) -> None:
        self.autosave_job = None
        if self.autosave_var.get() and self.autosave_dirty:
            if self.save_json(silent=True):
                self.autosave_dirty = False
        self.ensure_autosave_timer()

    def on_ctrl_s(self, _event: tk.Event | None = None) -> str:
        if self.save_json():
            self.autosave_dirty = False
        return "break"

    def undo(self) -> None:
        if not self.history:
            self.status_var.set("Nothing to undo.")
            return
        self.redo_history.append(self.history_snapshot())
        self.restore_history_snapshot(self.history.pop())
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()
        self.status_var.set("Undid last edit.")

    def redo(self) -> None:
        if not self.redo_history:
            self.status_var.set("Nothing to redo.")
            return
        self.history.append(self.history_snapshot())
        self.restore_history_snapshot(self.redo_history.pop())
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()
        self.status_var.set("Redid last edit.")

    def on_ctrl_z(self, _event: tk.Event | None = None) -> str:
        self.undo()
        return "break"

    def on_ctrl_y(self, _event: tk.Event | None = None) -> str:
        self.redo()
        return "break"

    def load_frame(self) -> None:
        if self.current_video is None:
            messagebox.showwarning("No video", "Select a video first.")
            return

        cap = cv2.VideoCapture(str(self.current_video))
        if not cap.isOpened():
            messagebox.showerror("Video error", f"Could not open:\n{self.current_video}")
            return

        frame_index = max(0, int(self.current_frame_index.get()))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        self.current_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()

        if not ok or frame is None:
            messagebox.showerror("Frame error", f"Could not read frame {frame_index}.")
            return

        self.current_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.tk_image = None
        self.tk_image_size = None
        self.tk_image_frame_token = None
        h, w = self.current_frame_rgb.shape[:2]
        self.current_image_size = (w, h)
        self.status_var.set(f"Loaded {self.current_video.name}, frame {frame_index}, size {w}x{h}.")
        self.redraw_canvas()

    def redraw_canvas(self) -> None:
        self.canvas.delete("all")
        if self.current_frame_rgb is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                fill="#cfd4dc",
                text="Load a video frame",
                font=("Segoe UI", 18),
            )
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        img_h, img_w = self.current_frame_rgb.shape[:2]
        fit_scale = min(canvas_w / img_w, canvas_h / img_h)
        requested_zoom = float(self.zoom_var.get())
        scale = fit_scale * requested_zoom

        # ImageTk keeps a full uncompressed bitmap. Cap only the raster backing
        # image; otherwise an 8x zoom of a 1080p frame can require hundreds of
        # megabytes and fail during repeated drag redraws.
        max_render_pixels = 8_000_000
        max_render_dimension = 8192
        safe_scale = min(
            scale,
            math.sqrt(max_render_pixels / max(1.0, float(img_w * img_h))),
            max_render_dimension / float(img_w),
            max_render_dimension / float(img_h),
        )
        if safe_scale < scale:
            scale = safe_scale
            safe_zoom = scale / max(fit_scale, 1e-12)
            self.zoom_var.set(safe_zoom)
            requested_zoom = safe_zoom
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))

        frame_token = id(self.current_frame_rgb)
        cache_valid = (
            self.tk_image is not None
            and self.tk_image_size == (disp_w, disp_h)
            and self.tk_image_frame_token == frame_token
        )
        if not cache_valid:
            self.tk_image = None
            self.tk_image_size = None
            self.tk_image_frame_token = None
            rendered = False
            for _attempt in range(5):
                try:
                    image = Image.fromarray(self.current_frame_rgb)
                    display = image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
                    new_tk_image = ImageTk.PhotoImage(display)
                    self.tk_image = new_tk_image
                    self.tk_image_size = (disp_w, disp_h)
                    self.tk_image_frame_token = frame_token
                    rendered = True
                    break
                except MemoryError:
                    scale *= 0.75
                    disp_w = max(1, int(img_w * scale))
                    disp_h = max(1, int(img_h * scale))
            if not rendered:
                self.status_var.set("Could not render frame: insufficient memory. Use Fit or reduce zoom.")
                return
            safe_zoom = scale / max(fit_scale, 1e-12)
            if safe_zoom < requested_zoom:
                self.zoom_var.set(safe_zoom)
                requested_zoom = safe_zoom

        offset_x = max(0, (canvas_w - disp_w) // 2)
        offset_y = max(0, (canvas_h - disp_h) // 2)
        self.display_scale = scale
        self.display_offset = (offset_x, offset_y)
        self.canvas.configure(scrollregion=(0, 0, max(canvas_w, disp_w + offset_x), max(canvas_h, disp_h + offset_y)))
        if hasattr(self, "zoom_label"):
            self.zoom_label.configure(text=f"Zoom: {int(requested_zoom * 100)}%")

        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor=tk.NW)

        self._draw_active_area()
        self._draw_plane_grid()
        self._draw_active_area_outline()
        self._draw_start_line()
        self._draw_measurement()
        self._draw_pixel_axis()
        self._draw_calibration_points()
        self._draw_inspect_marker()

    def request_drag_redraw(self) -> None:
        """Coalesce high-frequency mouse motion into at most ~30 redraws/sec."""
        if self.drag_redraw_job is None:
            self.drag_redraw_job = self.after(33, self._run_drag_redraw)

    def _run_drag_redraw(self) -> None:
        self.drag_redraw_job = None
        self.redraw_canvas()

    def finish_drag_redraw(self) -> None:
        if self.drag_redraw_job is not None:
            self.after_cancel(self.drag_redraw_job)
            self.drag_redraw_job = None

    def on_grid_spacing_changed(self, _event: tk.Event | None = None) -> None:
        """Debounce typed spinbox edits and redraw only when the value is valid."""
        if self.grid_redraw_job is not None:
            self.after_cancel(self.grid_redraw_job)
        self.grid_redraw_job = self.after(120, self._apply_grid_spacing_change)

    def _apply_grid_spacing_change(self) -> None:
        self.grid_redraw_job = None
        try:
            spacing = float(self.grid_spacing_var.get())
        except (tk.TclError, ValueError):
            return
        if spacing <= 0:
            return
        self.redraw_canvas()
        self.schedule_autosave()

    def image_to_canvas(self, pt: list[float] | tuple[float, float]) -> tuple[float, float]:
        ox, oy = self.display_offset
        return ox + pt[0] * self.display_scale, oy + pt[1] * self.display_scale

    def canvas_to_image(self, x: float, y: float) -> list[float] | None:
        if self.current_image_size is None:
            return None
        ox, oy = self.display_offset
        img_x = (x - ox) / self.display_scale
        img_y = (y - oy) / self.display_scale
        w, h = self.current_image_size
        if img_x < 0 or img_y < 0 or img_x > w or img_y > h:
            return None
        return [float(img_x), float(img_y)]

    def event_to_image(self, event: tk.Event) -> list[float] | None:
        return self.canvas_to_image(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def image_px_to_base_world(self, img_pt: list[float]) -> list[float] | None:
        if self.h_matrix is None or self.current_image_size is None:
            return None
        w, h = self.current_image_size
        return transform_norm_points([[img_pt[0] / w, img_pt[1] / h]], self.h_matrix)[0]

    def base_to_output_world(self, base_pt: list[float] | tuple[float, float]) -> list[float]:
        if not self.pixel_axis_info:
            return [float(base_pt[0]), float(base_pt[1])]
        origin = self.pixel_axis_info["origin_base_m"]
        x_axis = self.pixel_axis_info["x_axis_base_unit"]
        y_axis = self.pixel_axis_info["y_axis_base_unit"]
        dx = float(base_pt[0]) - float(origin[0])
        dy = float(base_pt[1]) - float(origin[1])
        return [
            dx * float(x_axis[0]) + dy * float(x_axis[1]),
            dx * float(y_axis[0]) + dy * float(y_axis[1]),
        ]

    def output_to_base_world(self, output_pt: list[float] | tuple[float, float]) -> list[float]:
        if not self.pixel_axis_info:
            return [float(output_pt[0]), float(output_pt[1])]
        origin = self.pixel_axis_info["origin_base_m"]
        x_axis = self.pixel_axis_info["x_axis_base_unit"]
        y_axis = self.pixel_axis_info["y_axis_base_unit"]
        return [
            float(origin[0]) + float(output_pt[0]) * float(x_axis[0]) + float(output_pt[1]) * float(y_axis[0]),
            float(origin[1]) + float(output_pt[0]) * float(x_axis[1]) + float(output_pt[1]) * float(y_axis[1]),
        ]

    def image_px_to_output_world(self, img_pt: list[float]) -> list[float] | None:
        base = self.image_px_to_base_world(img_pt)
        return self.base_to_output_world(base) if base else None

    def _draw_calibration_points(self) -> None:
        for idx, point in enumerate(self.state["calibration_points"], start=1):
            x, y = self.image_to_canvas(point["image_px"])
            is_dragging = self.drag_point_index == idx - 1
            r = 8 if is_dragging else 5
            self.canvas.create_oval(
                x - r,
                y - r,
                x + r,
                y + r,
                fill="#a7ff83" if is_dragging else "#ff4f5e",
                outline="black" if is_dragging else "white",
                width=2 if is_dragging else 1,
            )
            self.canvas.create_text(x + 12, y - 12, fill="white", text=str(idx), font=("Segoe UI", 10, "bold"))

    def _draw_active_area(self) -> None:
        pts = self.state["active_area_polygon_px"]
        if not pts:
            return
        canvas_pts = [self.image_to_canvas(pt) for pt in pts]
        flat = [coord for pt in canvas_pts for coord in pt]
        if len(canvas_pts) >= 3:
            self.canvas.create_polygon(*flat, outline="#f4a340", fill="#f4a340", stipple="gray25", width=2)
        if len(canvas_pts) >= 2:
            self.canvas.create_line(*flat, fill="#f4a340", width=2)
        for x, y in canvas_pts:
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#f4a340", outline="white")

    def _draw_active_area_outline(self) -> None:
        pts = self.state["active_area_polygon_px"]
        if len(pts) < 2:
            return
        canvas_pts = [self.image_to_canvas(pt) for pt in pts]
        flat = [coord for pt in canvas_pts for coord in pt]
        self.canvas.create_line(*flat, fill="#f4a340", width=2)
        for x, y in canvas_pts:
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#f4a340", outline="white")

    def _draw_pixel_axis(self) -> None:
        if self.axis_capture:
            pts = self.axis_capture.get("points_px", [])
            canvas_pts = [self.image_to_canvas(pt) for pt in pts]
            for x, y in canvas_pts:
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ffffff", outline="#111111", width=2)
            if len(canvas_pts) == 1:
                self.canvas.create_text(
                    canvas_pts[0][0] + 10,
                    canvas_pts[0][1] + 10,
                    anchor=tk.NW,
                    fill="#ffffff",
                    text=f"Origin, now click +{self.axis_capture['axis'].upper()}",
                    font=("Segoe UI", 11, "bold"),
                )
            return
        if not self.pixel_axis_info:
            return
        origin_px = self.pixel_axis_info["origin_px"]
        axis_px = self.pixel_axis_info["axis_point_px"]
        o = self.image_to_canvas(origin_px)
        a = self.image_to_canvas(axis_px)
        axis_len = max(float(self.pixel_axis_info.get("axis_length_base_m") or 1.0), 1.0)
        draw_len = axis_len * 1.25
        x_axis_pts = self.world_to_image_px([[-draw_len, 0.0], [draw_len, 0.0]])
        y_axis_pts = self.world_to_image_px([[0.0, -draw_len], [0.0, draw_len]])
        image_w, image_h = self.current_image_size or (1, 1)
        left, top = self.display_offset
        right = left + image_w * self.display_scale
        bottom = top + image_h * self.display_scale

        def clipped_ray(endpoint: tuple[float, float], margin: float = 18.0) -> tuple[float, float]:
            dx, dy = endpoint[0] - o[0], endpoint[1] - o[1]
            limit = 1.0
            if dx > 0:
                limit = min(limit, (right - margin - o[0]) / dx)
            elif dx < 0:
                limit = min(limit, (left + margin - o[0]) / dx)
            if dy > 0:
                limit = min(limit, (bottom - margin - o[1]) / dy)
            elif dy < 0:
                limit = min(limit, (top + margin - o[1]) / dy)
            limit = max(0.08, min(1.0, limit))
            return o[0] + dx * limit, o[1] + dy * limit

        def draw_axis_ray(endpoint_px: list[float], label: str, color: str) -> None:
            endpoint = clipped_ray(self.image_to_canvas(endpoint_px))
            self.canvas.create_line(*o, *endpoint, fill="#111318", width=7, arrow=tk.LAST, arrowshape=(14, 16, 6))
            self.canvas.create_line(*o, *endpoint, fill=color, width=4, arrow=tk.LAST, arrowshape=(14, 16, 6))
            lx = min(right - 22, max(left + 22, endpoint[0]))
            ly = min(bottom - 14, max(top + 14, endpoint[1]))
            self.canvas.create_rectangle(lx - 18, ly - 12, lx + 18, ly + 12, fill="#111318", outline=color, width=2)
            self.canvas.create_text(lx, ly, fill=color, text=label, font=("Segoe UI", 11, "bold"))

        if len(x_axis_pts) == 2:
            draw_axis_ray(x_axis_pts[0], "−X", "#ff4f5e")
            draw_axis_ray(x_axis_pts[1], "+X", "#ff4f5e")
        if len(y_axis_pts) == 2:
            draw_axis_ray(y_axis_pts[0], "−Y", "#32d2ff")
            draw_axis_ray(y_axis_pts[1], "+Y", "#32d2ff")

        color = "#ff4f5e" if self.pixel_axis_info["axis"] == "x" else "#32d2ff"
        self.canvas.create_oval(o[0] - 8, o[1] - 8, o[0] + 8, o[1] + 8, fill="#ffffff", outline=color, width=3)
        self.canvas.create_text(o[0] + 10, o[1] - 18, anchor=tk.NW, fill="#ffffff", text="origin", font=("Segoe UI", 10, "bold"))
        self.canvas.create_oval(a[0] - 8, a[1] - 8, a[0] + 8, a[1] + 8, fill=color, outline="#ffffff", width=2)
        self.canvas.create_text(
            a[0] + 10, a[1] + 10, anchor=tk.NW, fill=color,
            text=f"drag +{self.pixel_axis_info['axis'].upper()} handle", font=("Segoe UI", 10, "bold"),
        )

        legend_x = self.canvas.canvasx(16)
        legend_y = self.canvas.canvasy(16)
        defined = self.pixel_axis_info.get("axis", "y").upper()
        perpendicular = "X" if defined == "Y" else "Y"
        flipped = int(self.pixel_axis_info.get("perpendicular_sign", 1)) < 0
        legend_text = (
            f"CUSTOM AXES   X = RED   Y = CYAN\n"
            f"Clicked direction: +{defined}   |   +{perpendicular}: {'FLIPPED' if flipped else 'DEFAULT'}"
        )
        self.canvas.create_rectangle(
            legend_x, legend_y, legend_x + 330, legend_y + 48,
            fill="#111318", outline="#ffffff", width=1,
        )
        self.canvas.create_text(
            legend_x + 10, legend_y + 7, anchor=tk.NW, fill="#ffffff",
            text=legend_text, font=("Segoe UI", 10, "bold"),
        )

    def _draw_start_line(self) -> None:
        pts = self.state["start_line_px"]
        if not pts:
            return
        canvas_pts = [self.image_to_canvas(pt) for pt in pts]
        for x, y in canvas_pts:
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#32d2ff", outline="white")
        if len(canvas_pts) == 2:
            self.canvas.create_line(*canvas_pts[0], *canvas_pts[1], fill="#32d2ff", width=3)

    def world_to_image_px(self, world_points: list[list[float]]) -> list[list[float]]:
        if self.h_matrix is None or self.current_image_size is None:
            return []
        try:
            h_inv = np.linalg.inv(np.array(self.h_matrix, dtype=np.float64))
        except np.linalg.LinAlgError:
            return []
        base_points = [self.output_to_base_world(pt) for pt in world_points]
        pts = np.array(base_points, dtype=np.float32).reshape(-1, 1, 2)
        norm = cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2)
        w, h = self.current_image_size
        return [[float(x * w), float(y * h)] for x, y in norm]

    def image_corners_world(self) -> list[list[float]]:
        if self.h_matrix is None:
            return []
        base = transform_norm_points([[0, 0], [1, 0], [1, 1], [0, 1]], self.h_matrix)
        return [self.base_to_output_world(pt) for pt in base]

    def grid_reference_world_points(self) -> list[list[float]]:
        if self.current_image_size is None:
            return []
        w, h = self.current_image_size
        points_px = [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]
        world_points: list[list[float]] = []
        for point_px in points_px:
            base = self.image_px_to_base_world(point_px)
            if base is not None:
                world_points.append(self.base_to_output_world(base))
        return world_points

    def grid_clip_polygon_norm(self) -> list[tuple[float, float]]:
        return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    def image_norm_line_to_canvas_segments(
        self,
        line: np.ndarray,
        polygon_norm: list[tuple[float, float]],
    ) -> list[list[tuple[float, float]]]:
        if self.current_image_size is None or len(polygon_norm) < 3:
            return []
        a, b, c = [float(v) for v in line]
        norm = math.hypot(a, b)
        if norm < 1e-12:
            return []
        a /= norm
        b /= norm
        c /= norm
        direction = (-b, a)
        if abs(a) >= abs(b):
            origin = (-c / a, 0.0) if abs(a) > 1e-12 else (0.0, 0.0)
        else:
            origin = (0.0, -c / b)

        intersections: list[tuple[float, float, float]] = []
        eps = 1e-8
        count = len(polygon_norm)
        for idx in range(count):
            p = polygon_norm[idx]
            q = polygon_norm[(idx + 1) % count]
            sp = a * p[0] + b * p[1] + c
            sq = a * q[0] + b * q[1] + c
            if abs(sp) <= eps:
                t = (p[0] - origin[0]) * direction[0] + (p[1] - origin[1]) * direction[1]
                intersections.append((t, p[0], p[1]))
            if abs(sq) <= eps:
                t = (q[0] - origin[0]) * direction[0] + (q[1] - origin[1]) * direction[1]
                intersections.append((t, q[0], q[1]))
            if sp * sq < 0:
                edge_t = sp / (sp - sq)
                x = p[0] + edge_t * (q[0] - p[0])
                y = p[1] + edge_t * (q[1] - p[1])
                t = (x - origin[0]) * direction[0] + (y - origin[1]) * direction[1]
                intersections.append((t, x, y))

        unique: list[tuple[float, float, float]] = []
        for item in sorted(intersections, key=lambda v: v[0]):
            if not unique or math.hypot(item[1] - unique[-1][1], item[2] - unique[-1][2]) > 1e-6:
                unique.append(item)
        if len(unique) < 2:
            return []

        segments: list[list[tuple[float, float]]] = []
        w, h = self.current_image_size
        for start, end in zip(unique[0::2], unique[1::2]):
            p0 = self.image_to_canvas([start[1] * w, start[2] * h])
            p1 = self.image_to_canvas([end[1] * w, end[2] * h])
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= 1.0:
                segments.append([p0, p1])
        return segments

    def draw_world_grid_line(
        self,
        final_h_norm: np.ndarray,
        world_line: list[float],
        polygon_norm: list[tuple[float, float]],
        color: str,
        width: int,
        density_state: dict[str, Any] | None = None,
        family: str = "grid",
        force: bool = False,
    ) -> None:
        image_line = final_h_norm.T @ np.array(world_line, dtype=np.float64)
        for segment in self.image_norm_line_to_canvas_segments(image_line, polygon_norm):
            if density_state is not None and not force:
                ordered = sorted(segment)
                signature = (ordered[0][0], ordered[0][1], ordered[1][0], ordered[1][1])
                previous = density_state.get(f"last_{family}")
                # Consecutive world lines that differ by less than three screen
                # pixels are indistinguishable near the vanishing region.
                if previous is not None and max(abs(a - b) for a, b in zip(signature, previous)) < 3.0:
                    continue
                region_counts = density_state.setdefault("regions", {})
                x0, y0 = ordered[0]
                x1, y1 = ordered[1]
                region_keys = {
                    (int((x0 + (x1 - x0) * t) // 32), int((y0 + (y1 - y0) * t) // 32))
                    for t in (0.0, 0.25, 0.5, 0.75, 1.0)
                }
                # Strict local cap: no 32x32 screen region receives more than
                # eight ordinary grid lines. Important calibrated lines bypass it.
                if any(int(region_counts.get(key, 0)) >= 8 for key in region_keys):
                    continue
                if int(density_state.get("count", 0)) >= 350:
                    continue
                density_state[f"last_{family}"] = signature
                density_state["count"] = int(density_state.get("count", 0)) + 1
                for key in region_keys:
                    region_counts[key] = int(region_counts.get(key, 0)) + 1
            self.canvas.create_line(
                *[coord for point in segment for coord in point],
                fill=color,
                width=width,
                dash=None if width > 1 else (4, 4),
            )

    def _draw_plane_grid(self) -> None:
        if not self.show_grid_var.get() or self.h_matrix is None or self.current_image_size is None:
            return
        try:
            spacing = float(self.grid_spacing_var.get())
        except (tk.TclError, ValueError):
            spacing = 5.0
        if spacing <= 0:
            return

        reference_points = self.grid_reference_world_points()
        if len(reference_points) < 3:
            return
        xs = [p[0] for p in reference_points]
        ys = [p[1] for p in reference_points]
        min_x = math.floor((min(xs) - spacing) / spacing) * spacing
        max_x = math.ceil((max(xs) + spacing) / spacing) * spacing
        min_y = math.floor((min(ys) - spacing) / spacing) * spacing
        max_y = math.ceil((max(ys) + spacing) / spacing) * spacing

        # If the homography denominator changes sign inside the image, the
        # world plane has a projective pole there. Corner-only bounds then miss
        # arbitrarily large coordinates and grid lines appear/disappear after
        # tiny calibration edits. Use a stable, symmetric emergency range.
        h_base = np.asarray(self.h_matrix, dtype=np.float64)
        denominator_at_corners = [
            float(h_base[2, 0] * x + h_base[2, 1] * y + h_base[2, 2])
            for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        ]
        if min(denominator_at_corners) <= 0.0 <= max(denominator_at_corners):
            stable_cells = 500
            min_x = min_y = -stable_cells * spacing
            max_x = max_y = stable_cells * spacing
        if max_x - min_x > 1000 * spacing or max_y - min_y > 1000 * spacing:
            self.status_var.set("Grid hidden: homography plane is too large. Check calibration points.")
            return

        x_values = np.arange(min_x, max_x + spacing * 0.5, spacing)
        y_values = np.arange(min_y, max_y + spacing * 0.5, spacing)
        # A projective road grid can legitimately contain hundreds of visible
        # lines near the horizon. The former 140-line budget created obvious
        # gaps. Keep a generous emergency limit; the 1000-cells-per-axis guard
        # above still protects the GUI from pathological homographies.
        max_lines = 2500
        if len(x_values) + len(y_values) > max_lines:
            skip = math.ceil((len(x_values) + len(y_values)) / max_lines)
            calibration_world = [
                self.base_to_output_world(point["world_m"])
                for point in self.state.get("calibration_points", [])
                if point.get("world_m") and len(point["world_m"]) >= 2
            ]
            if calibration_world:
                calibration_min_x = min(point[0] for point in calibration_world) - spacing * 1e-7
                calibration_max_x = max(point[0] for point in calibration_world) + spacing * 1e-7
                calibration_min_y = min(point[1] for point in calibration_world) - spacing * 1e-7
                calibration_max_y = max(point[1] for point in calibration_world) + spacing * 1e-7
            else:
                calibration_min_x = calibration_min_y = math.inf
                calibration_max_x = calibration_max_y = -math.inf
            # Anchor thinning to world-grid indices, not the changing image
            # bounds. This keeps the same lines visible after small point moves
            # and guarantees axes plus lines inside the calibrated range survive.
            x_values = np.asarray(
                [
                    value
                    for value in x_values
                    if calibration_min_x <= value <= calibration_max_x
                    or int(round(float(value) / spacing)) % skip == 0
                ],
                dtype=np.float64,
            )
            y_values = np.asarray(
                [
                    value
                    for value in y_values
                    if calibration_min_y <= value <= calibration_max_y
                    or int(round(float(value) / spacing)) % skip == 0
                ],
                dtype=np.float64,
            )

        final_h_norm = base_to_final_world_matrix(self.pixel_axis_info) @ np.array(self.h_matrix, dtype=np.float64)
        polygon_norm = self.grid_clip_polygon_norm()
        density_state: dict[str, Any] = {"count": 0}
        calibration_world = [
            self.base_to_output_world(point["world_m"])
            for point in self.state.get("calibration_points", [])
            if point.get("world_m") and len(point["world_m"]) >= 2
        ]
        calibration_x_bounds = (
            (min(point[0] for point in calibration_world), max(point[0] for point in calibration_world))
            if calibration_world
            else None
        )
        calibration_y_bounds = (
            (min(point[1] for point in calibration_world), max(point[1] for point in calibration_world))
            if calibration_world
            else None
        )
        for x in x_values:
            color = "#ecf0f3" if abs(x) > spacing * 0.25 else "#ff4f5e"
            width = 1 if abs(x) > spacing * 0.25 else 3
            important = width > 1 or (
                calibration_x_bounds is not None
                and calibration_x_bounds[0] - spacing * 1e-7 <= x <= calibration_x_bounds[1] + spacing * 1e-7
            )
            self.draw_world_grid_line(
                final_h_norm, [1.0, 0.0, -float(x)], polygon_norm, color, width,
                density_state=density_state, family="x", force=important,
            )
        for y in y_values:
            color = "#b7ffda" if abs(y) > spacing * 0.25 else "#32d2ff"
            width = 1 if abs(y) > spacing * 0.25 else 3
            important = width > 1 or (
                calibration_y_bounds is not None
                and calibration_y_bounds[0] - spacing * 1e-7 <= y <= calibration_y_bounds[1] + spacing * 1e-7
            )
            self.draw_world_grid_line(
                final_h_norm, [0.0, 1.0, -float(y)], polygon_norm, color, width,
                density_state=density_state, family="y", force=important,
            )

    def _draw_world_polyline(self, world_line: list[list[float]], color: str, width: int) -> None:
        if self.current_image_size is None:
            return
        image_pts = self.world_to_image_px(world_line)
        if not image_pts:
            return
        img_w, img_h = self.current_image_size
        segment: list[tuple[float, float]] = []
        for px, py in image_pts:
            inside = -img_w * 0.1 <= px <= img_w * 1.1 and -img_h * 0.1 <= py <= img_h * 1.1
            if inside:
                segment.append(self.image_to_canvas([px, py]))
            else:
                if len(segment) >= 2:
                    self.canvas.create_line(*[coord for pt in segment for coord in pt], fill=color, width=width, stipple="gray50")
                segment = []
        if len(segment) >= 2:
            self.canvas.create_line(*[coord for pt in segment for coord in pt], fill=color, width=width, stipple="gray50")

    def _draw_measurement(self) -> None:
        if not self.measure_points_px:
            return
        canvas_pts = [self.image_to_canvas(pt) for pt in self.measure_points_px]
        for x, y in canvas_pts:
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#a7ff83", outline="black", width=1)
        if len(canvas_pts) == 2:
            self.canvas.create_line(*canvas_pts[0], *canvas_pts[1], fill="#a7ff83", width=3)
            if self.measure_distance_m is not None:
                mid_x = (canvas_pts[0][0] + canvas_pts[1][0]) / 2
                mid_y = (canvas_pts[0][1] + canvas_pts[1][1]) / 2
                self.canvas.create_text(
                    mid_x + 10,
                    mid_y + 10,
                    anchor=tk.NW,
                    fill="#a7ff83",
                    text=f"{self.measure_distance_m:.3f} m",
                    font=("Segoe UI", 12, "bold"),
                )

    def _draw_inspect_marker(self) -> None:
        if self.inspect_marker_px is None or self.inspect_world_m is None:
            return
        x, y = self.image_to_canvas(self.inspect_marker_px)
        self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#a7ff83", width=3)
        label = f"({self.inspect_world_m[0]:.2f}, {self.inspect_world_m[1]:.2f}) m"
        self.canvas.create_text(x + 12, y + 12, anchor=tk.NW, fill="#a7ff83", text=label, font=("Segoe UI", 11, "bold"))

    def nearest_calibration_point_index(self, img_pt: list[float], max_screen_px: float = 35.0) -> tuple[int | None, float | None]:
        points = self.state["calibration_points"]
        if not points:
            return None, None
        distances = [
            math.hypot(float(p["image_px"][0]) - img_pt[0], float(p["image_px"][1]) - img_pt[1])
            for p in points
        ]
        idx = int(np.argmin(distances))
        max_image_px = max_screen_px / max(self.display_scale, 0.01)
        if distances[idx] <= max_image_px:
            return idx, distances[idx]
        return None, distances[idx]

    def nearest_active_area_point_index(self, img_pt: list[float], max_screen_px: float = 35.0) -> tuple[int | None, float | None]:
        points = self.state["active_area_polygon_px"]
        if not points:
            return None, None
        distances = [math.hypot(float(p[0]) - img_pt[0], float(p[1]) - img_pt[1]) for p in points]
        idx = int(np.argmin(distances))
        max_image_px = max_screen_px / max(self.display_scale, 0.01)
        if distances[idx] <= max_image_px:
            return idx, distances[idx]
        return None, distances[idx]

    def nearest_pixel_axis_handle(self, img_pt: list[float], max_screen_px: float = 35.0) -> str | None:
        if not self.pixel_axis_info:
            return None
        max_image_px = max_screen_px / max(self.display_scale, 0.01)
        handles = {
            "origin": self.pixel_axis_info["origin_px"],
            "axis_point": self.pixel_axis_info["axis_point_px"],
        }
        nearest_name = None
        nearest_distance = None
        for name, pt in handles.items():
            distance = math.hypot(float(pt[0]) - img_pt[0], float(pt[1]) - img_pt[1])
            if nearest_distance is None or distance < nearest_distance:
                nearest_name = name
                nearest_distance = distance
        return nearest_name if nearest_distance is not None and nearest_distance <= max_image_px else None

    def start_pixel_axis_capture(self) -> None:
        if self.h_matrix is None:
            if not self.compute_homography():
                messagebox.showinfo("Need homography", "Compute or load a homography before setting an image-picked axis.")
                return
        self.mode.set("axis")
        axis_config = self.ask_axis_from_image()
        if axis_config is None:
            return
        axis = axis_config["axis"]
        self.axis_capture = {
            "axis": axis,
            "perpendicular_sign": axis_config["perpendicular_sign"],
            "points_px": [],
        }
        self.status_var.set(f"Axis capture started. Click origin (0,0), then click a point on +{axis.upper()}.")
        self.redraw_canvas()

    def ask_axis_from_image(self) -> dict[str, Any] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Image-picked custom axis")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        selected = tk.StringVar(value=(self.pixel_axis_info or {}).get("axis", "y"))
        flip_perpendicular = tk.BooleanVar(
            value=int((self.pixel_axis_info or {}).get("perpendicular_sign", 1)) < 0
        )
        result: dict[str, Any] = {"config": None}

        ttk.Label(
            dialog,
            text="Choose which output axis the second clicked image point will define.",
            wraplength=420,
        ).pack(anchor=tk.W, padx=12, pady=(12, 6))
        ttk.Radiobutton(dialog, text="Second point is on +X axis", value="x", variable=selected).pack(anchor=tk.W, padx=12)
        ttk.Radiobutton(dialog, text="Second point is on +Y axis", value="y", variable=selected).pack(anchor=tk.W, padx=12)
        ttk.Checkbutton(
            dialog,
            text="Flip the automatically created perpendicular axis",
            variable=flip_perpendicular,
        ).pack(anchor=tk.W, padx=12, pady=(8, 0))
        ttk.Label(
            dialog,
            text="Example: when the second point defines +Y, this reverses +X without changing +Y.",
            wraplength=420,
        ).pack(anchor=tk.W, padx=30, pady=(2, 0))

        if self.pixel_axis_info:
            info = self.pixel_axis_info
            summary = (
                f"Current axis: +{info.get('axis', '?').upper()}\n"
                f"Origin px: {info['origin_px'][0]:.1f}, {info['origin_px'][1]:.1f}\n"
                f"Axis point px: {info['axis_point_px'][0]:.1f}, {info['axis_point_px'][1]:.1f}"
            )
            ttk.Label(dialog, text=summary, wraplength=420).pack(anchor=tk.W, padx=12, pady=(8, 0))

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=12)

        def ok() -> None:
            result["config"] = {
                "axis": selected.get(),
                "perpendicular_sign": -1 if flip_perpendicular.get() else 1,
            }
            dialog.destroy()

        ttk.Button(buttons, text="Start clicking", command=ok).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self.wait_window(dialog)
        return result["config"]

    def add_pixel_axis_capture_point(self, img_pt: list[float]) -> None:
        if not self.axis_capture:
            return
        self.axis_capture["points_px"].append(img_pt)
        if len(self.axis_capture["points_px"]) == 1:
            self.status_var.set(f"Origin set. Click a point on +{self.axis_capture['axis'].upper()}.")
            self.redraw_canvas()
            return
        origin_px, axis_px = self.axis_capture["points_px"][:2]
        self.apply_pixel_axis(
            origin_px,
            axis_px,
            self.axis_capture["axis"],
            int(self.axis_capture.get("perpendicular_sign", 1)),
        )
        self.axis_capture = None
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()

    def apply_pixel_axis(
        self,
        origin_px: list[float],
        axis_px: list[float],
        axis: str,
        perpendicular_sign: int | None = None,
    ) -> None:
        if self.current_image_size is None:
            return
        origin_base = self.image_px_to_base_world(origin_px)
        axis_base = self.image_px_to_base_world(axis_px)
        if origin_base is None or axis_base is None:
            messagebox.showerror("Need homography", "Could not transform axis points. Compute homography first.")
            return
        vx = axis_base[0] - origin_base[0]
        vy = axis_base[1] - origin_base[1]
        length = math.hypot(vx, vy)
        if length == 0:
            messagebox.showerror("Invalid axis", "Axis point is too close to origin.")
            return
        unit = [vx / length, vy / length]
        if perpendicular_sign is None:
            perpendicular_sign = int((self.pixel_axis_info or {}).get("perpendicular_sign", 1))
        perpendicular_sign = -1 if perpendicular_sign < 0 else 1
        if axis == "x":
            x_axis = unit
            y_axis = [perpendicular_sign * -unit[1], perpendicular_sign * unit[0]]
        else:
            y_axis = unit
            x_axis = [perpendicular_sign * unit[1], perpendicular_sign * -unit[0]]
        w, h = self.current_image_size
        self.pixel_axis_info = {
            "type": "image_picked_axis",
            "axis": axis,
            "perpendicular_sign": perpendicular_sign,
            "origin_px": [float(origin_px[0]), float(origin_px[1])],
            "axis_point_px": [float(axis_px[0]), float(axis_px[1])],
            "origin_norm": [float(origin_px[0] / w), float(origin_px[1] / h)],
            "axis_point_norm": [float(axis_px[0] / w), float(axis_px[1] / h)],
            "origin_base_m": [float(origin_base[0]), float(origin_base[1])],
            "axis_point_base_m": [float(axis_base[0]), float(axis_base[1])],
            "x_axis_base_unit": [float(x_axis[0]), float(x_axis[1])],
            "y_axis_base_unit": [float(y_axis[0]), float(y_axis[1])],
            "axis_length_base_m": float(length),
        }
        self.status_var.set(f"Image custom axis set from clicked +{axis.upper()} direction.")

    def on_canvas_press(self, event: tk.Event) -> None:
        if self.panning:
            self.canvas.scan_mark(event.x, event.y)
            return
        img_pt = self.event_to_image(event)
        if img_pt is None:
            return
        mode = self.mode.get()
        if mode == "axis":
            if self.axis_capture:
                self.add_pixel_axis_capture_point(img_pt)
                return
            axis_handle = self.nearest_pixel_axis_handle(img_pt)
            if axis_handle is not None:
                self.push_history()
                self.pixel_axis_drag_handle = axis_handle
                self.status_var.set(f"Dragging image-axis {axis_handle.replace('_', ' ')}.")
                return
            self.start_pixel_axis_capture()
            if self.axis_capture:
                self.add_pixel_axis_capture_point(img_pt)
            return
        if mode == "calibrate":
            idx, _distance = self.nearest_calibration_point_index(img_pt)
            if idx is not None:
                self.push_history()
                self.drag_point_index = idx
                self.drag_started = True
                self.status_var.set(f"Dragging calibration point {idx + 1}. Release mouse to finish.")
                return
            self.add_calibration_point(img_pt)
        elif mode == "measure":
            self.add_measure_point(img_pt)
        elif mode == "active":
            idx, _distance = self.nearest_active_area_point_index(img_pt)
            if idx is not None:
                self.push_history()
                self.active_drag_index = idx
                self.status_var.set(f"Dragging active-area vertex {idx + 1}.")
                return
            self.push_history()
            self.state["active_area_polygon_px"].append(img_pt)
            self.status_var.set(f"Active area has {len(self.state['active_area_polygon_px'])} point(s).")
            self.redraw_canvas()
            self.schedule_autosave()
        elif mode == "start":
            self.push_history()
            if len(self.state["start_line_px"]) >= 2:
                self.state["start_line_px"] = []
            self.state["start_line_px"].append(img_pt)
            self.status_var.set(f"Start line has {len(self.state['start_line_px'])}/2 point(s).")
            self.redraw_canvas()
            self.schedule_autosave()
        elif mode == "inspect":
            self.inspect_point(img_pt)

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            return
        if self.pixel_axis_drag_handle is not None:
            img_pt = self.event_to_image(event)
            if img_pt is None or not self.pixel_axis_info:
                return
            origin_px = self.pixel_axis_info["origin_px"]
            axis_px = self.pixel_axis_info["axis_point_px"]
            if self.pixel_axis_drag_handle == "origin":
                origin_px = img_pt
            else:
                axis_px = img_pt
            self.apply_pixel_axis(origin_px, axis_px, self.pixel_axis_info["axis"])
            self.refresh_point_tree()
            self.request_drag_redraw()
            return
        if self.active_drag_index is not None:
            img_pt = self.event_to_image(event)
            if img_pt is None:
                return
            pts = self.state["active_area_polygon_px"]
            if 0 <= self.active_drag_index < len(pts):
                pts[self.active_drag_index] = img_pt
                self.request_drag_redraw()
            return
        if self.drag_point_index is None:
            return
        img_pt = self.event_to_image(event)
        if img_pt is None:
            return
        points = self.state["calibration_points"]
        if not 0 <= self.drag_point_index < len(points):
            return
        points[self.drag_point_index]["image_px"] = img_pt
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.request_drag_redraw()

    def on_canvas_release(self, _event: tk.Event) -> None:
        self.finish_drag_redraw()
        if self.panning:
            return
        if self.pixel_axis_drag_handle is not None:
            handle = self.pixel_axis_drag_handle
            self.pixel_axis_drag_handle = None
            self.refresh_point_tree()
            self.redraw_canvas()
            self.schedule_autosave()
            self.status_var.set(f"Moved image-axis {handle.replace('_', ' ')}.")
            return
        if self.active_drag_index is not None:
            idx = self.active_drag_index
            self.active_drag_index = None
            self.redraw_canvas()
            self.schedule_autosave()
            self.status_var.set(f"Moved active-area vertex {idx + 1}.")
            return
        if self.drag_point_index is None:
            return
        idx = self.drag_point_index
        self.drag_point_index = None
        self.drag_started = False
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()
        self.status_var.set(f"Moved calibration point {idx + 1}.")

    def on_canvas_right_click(self, event: tk.Event) -> None:
        img_pt = self.event_to_image(event)
        if img_pt is None:
            return
        if self.mode.get() == "calibrate":
            self.delete_nearest_calibration_point(img_pt)
        elif self.mode.get() == "active" and self.state["active_area_polygon_px"]:
            self.delete_nearest_active_area_point(img_pt)
        elif self.mode.get() == "start" and self.state["start_line_px"]:
            self.push_history()
            self.state["start_line_px"].pop()
            self.status_var.set("Removed last start-line point.")
            self.redraw_canvas()
            self.schedule_autosave()
        elif self.mode.get() == "measure":
            self.clear_measure()

    def delete_nearest_active_area_point(self, img_pt: list[float]) -> None:
        points = self.state["active_area_polygon_px"]
        if not points:
            return
        idx, _distance = self.nearest_active_area_point_index(img_pt)
        if idx is None:
            self.status_var.set("Right-click closer to an active-area point to delete it.")
            return
        self.push_history()
        del points[idx]
        self.status_var.set(f"Removed active-area point {idx + 1}.")
        self.redraw_canvas()
        self.schedule_autosave()

    def delete_last_start_line_point(self) -> None:
        if self.state["start_line_px"]:
            self.push_history()
            self.state["start_line_px"].pop()
            self.status_var.set("Removed last start-line point.")
            self.redraw_canvas()
            self.schedule_autosave()

    def delete_nearest_calibration_point(self, img_pt: list[float]) -> None:
        points = self.state["calibration_points"]
        if not points:
            return
        idx, _distance = self.nearest_calibration_point_index(img_pt)
        if idx is None:
            self.status_var.set("Right-click closer to a calibration point to delete it.")
            return
        self.push_history()
        del points[idx]
        self.gps_axis_definition = None
        self.rebuild_gps_relative_world()
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()
        self.status_var.set(f"Deleted calibration point {idx + 1}.")

    def add_measure_point(self, img_pt: list[float]) -> None:
        if self.h_matrix is None:
            if not self.compute_homography():
                return
        if len(self.measure_points_px) >= 2:
            self.measure_points_px = []
            self.measure_distance_m = None
        self.measure_points_px.append(img_pt)
        if len(self.measure_points_px) == 2:
            assert self.current_image_size is not None
            w, h = self.current_image_size
            norm = [[pt[0] / w, pt[1] / h] for pt in self.measure_points_px]
            base_pts = transform_norm_points(norm, self.h_matrix)
            world_pts = [self.base_to_output_world(pt) for pt in base_pts]
            dx = world_pts[1][0] - world_pts[0][0]
            dy = world_pts[1][1] - world_pts[0][1]
            self.measure_distance_m = math.hypot(dx, dy)
            self.status_var.set(f"Measured distance: {self.measure_distance_m:.3f} m.")
        else:
            self.status_var.set("Measure: click the second point.")
        self.redraw_canvas()

    def axis_reference_text(self, key: str) -> str:
        if not self.gps_axis_definition:
            return ""
        ref = self.gps_axis_definition.get(key) or {}
        raw = ref.get("raw_gps")
        if not raw:
            return ""
        source = ref.get("source", "manual")
        if source == "calibration_point" and "point_index" in ref:
            return str(int(ref["point_index"]) + 1)
        return f"{raw[0]}, {raw[1]}"

    def parse_gps_axis_reference(self, text: str) -> dict[str, Any]:
        value = text.strip()
        if not value:
            raise ValueError("Enter either a GPS point number or latitude, longitude.")
        if re.fullmatch(r"\d+", value):
            point_number = int(value)
            if not 1 <= point_number <= len(self.state["calibration_points"]):
                raise ValueError(f"Point {point_number} does not exist.")
            point = self.state["calibration_points"][point_number - 1]
            if point.get("coordinate_input") != "gps" or not point.get("raw_gps"):
                raise ValueError(f"Point {point_number} is not a GPS calibration point.")
            return {
                "source": "calibration_point",
                "point_index": point_number - 1,
                "raw_gps": [float(point["raw_gps"][0]), float(point["raw_gps"][1])],
            }
        lat, lon = parse_lat_lon(value)
        return {"source": "manual", "raw_gps": [lat, lon]}

    def set_gps_custom_axis(self) -> None:
        if self.calibration_input_type and self.calibration_input_type != "gps":
            messagebox.showwarning(
                "GPS axis unavailable",
                "This calibration uses X/Y points. GPS custom axis is only available for GPS-based calibrations.",
            )
            return
        gps_point_numbers = [
            idx + 1
            for idx, point in enumerate(self.state["calibration_points"])
            if point.get("coordinate_input") == "gps" and point.get("raw_gps")
        ]
        if not gps_point_numbers:
            messagebox.showinfo(
                "Need GPS calibration",
                "Add GPS calibration points first, or use 'Set/Edit Axis From Image' if you only want to define axes by pixel clicks.",
            )
            return
        values = self.ask_gps_axis_dialog(gps_point_numbers)
        if values is None:
            return
        try:
            origin_ref = self.parse_gps_axis_reference(values["origin"])
            y_ref = self.parse_gps_axis_reference(values["positive_y"])
        except ValueError as exc:
            messagebox.showerror("Invalid GPS axis", str(exc))
            return

        self.push_history()
        self.gps_axis_definition = {"origin": origin_ref, "positive_y": y_ref}
        try:
            self.rebuild_gps_relative_world()
        except ValueError as exc:
            self.state = self.history.pop()
            messagebox.showerror("GPS axis error", str(exc))
            return
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()
        self.status_var.set("GPS custom XY axis updated.")

    def resolve_gps_axis_reference(self, ref: dict[str, Any]) -> list[float]:
        if ref.get("source") == "calibration_point":
            idx = int(ref.get("point_index", -1))
            if not 0 <= idx < len(self.state["calibration_points"]):
                raise ValueError("A GPS custom-axis calibration point was deleted. Edit the GPS axis.")
            point = self.state["calibration_points"][idx]
            if point.get("coordinate_input") != "gps" or not point.get("raw_gps"):
                raise ValueError("A GPS custom-axis point is no longer a GPS point. Edit the GPS axis.")
            ref["raw_gps"] = [float(point["raw_gps"][0]), float(point["raw_gps"][1])]
        raw = ref.get("raw_gps")
        if not raw:
            raise ValueError("GPS custom-axis reference is missing coordinates.")
        return [float(raw[0]), float(raw[1])]

    def ask_gps_axis_dialog(self, gps_point_numbers: list[int]) -> dict[str, str] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Set/Edit GPS custom XY axis")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        result: dict[str, str] | None = None
        origin_var = tk.StringVar(value=self.axis_reference_text("origin"))
        y_var = tk.StringVar(value=self.axis_reference_text("positive_y"))
        hint = (
            "Each field accepts either an existing GPS calibration point number "
            "or a new latitude, longitude value."
        )
        available = f"Available GPS point numbers: {gps_point_numbers or 'none'}"

        ttk.Label(dialog, text=hint, wraplength=480).pack(anchor=tk.W, padx=12, pady=(12, 2))
        ttk.Label(dialog, text=available, wraplength=480).pack(anchor=tk.W, padx=12, pady=(0, 10))

        form = ttk.Frame(dialog)
        form.pack(fill=tk.X, padx=12)
        ttk.Label(form, text="Origin (0,0)").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=origin_var, width=52).grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=4)
        ttk.Label(form, text="+Y direction point").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=y_var, width=52).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)
        form.columnconfigure(1, weight=1)

        ttk.Label(
            dialog,
            text="Example: 23.742866010823988, 90.39576489469023",
            wraplength=480,
        ).pack(anchor=tk.W, padx=12, pady=(6, 0))

        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=12, pady=12)

        def save() -> None:
            nonlocal result
            result = {"origin": origin_var.get().strip(), "positive_y": y_var.get().strip()}
            dialog.destroy()

        ttk.Button(buttons, text="Save axis", command=save).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self.wait_window(dialog)
        return result

    def rebuild_gps_relative_world(self) -> None:
        gps_points = [p for p in self.state["calibration_points"] if p.get("coordinate_input") == "gps" and p.get("raw_gps")]
        if not gps_points:
            self.origin_info = None
            return

        if self.gps_axis_definition:
            origin_raw = self.resolve_gps_axis_reference(self.gps_axis_definition["origin"])
            y_raw = self.resolve_gps_axis_reference(self.gps_axis_definition["positive_y"])
        else:
            origin_raw = gps_points[0]["raw_gps"]
            y_raw = None

        origin_utm = gps_to_utm(float(origin_raw[0]), float(origin_raw[1]))
        x_axis = [1.0, 0.0]
        y_axis = [0.0, 1.0]
        origin_type = "gps_utm_origin"
        if y_raw:
            y_utm = gps_to_utm(float(y_raw[0]), float(y_raw[1]))
            if y_utm["epsg"] != origin_utm["epsg"]:
                raise ValueError("GPS custom +Y point must be in the same UTM zone as the origin.")
            dy_e = y_utm["easting"] - origin_utm["easting"]
            dy_n = y_utm["northing"] - origin_utm["northing"]
            axis_len = math.hypot(dy_e, dy_n)
            if axis_len == 0:
                raise ValueError("GPS custom +Y point is too close to the origin.")
            y_axis = [dy_e / axis_len, dy_n / axis_len]
            x_axis = [y_axis[1], -y_axis[0]]
            origin_type = "gps_custom_xy_origin"

        self.origin_info = {
            "type": origin_type,
            "lat": origin_utm["lat"],
            "lon": origin_utm["lon"],
            "E0": origin_utm["easting"],
            "N0": origin_utm["northing"],
            "zone": origin_utm["zone"],
            "north": origin_utm["north"],
            "epsg": origin_utm["epsg"],
            "x_axis_utm": x_axis,
            "y_axis_utm": y_axis,
            "gps_axis_definition": self.gps_axis_definition,
        }

        for point in gps_points:
            lat, lon = point["raw_gps"]
            utm = gps_to_utm(float(lat), float(lon))
            if utm["epsg"] != origin_utm["epsg"]:
                raise ValueError("All GPS calibration points must be in the same UTM zone.")
            point["utm"] = {
                "easting": utm["easting"],
                "northing": utm["northing"],
                "zone": utm["zone"],
                "north": utm["north"],
                "epsg": utm["epsg"],
            }
            delta_e = utm["easting"] - origin_utm["easting"]
            delta_n = utm["northing"] - origin_utm["northing"]
            point["world_m"] = [
                delta_e * x_axis[0] + delta_n * x_axis[1],
                delta_e * y_axis[0] + delta_n * y_axis[1],
            ]

    def add_calibration_point(self, img_pt: list[float]) -> None:
        point: dict[str, Any]
        selected_mode = self.coord_input_mode.get()
        if self.calibration_input_type and selected_mode != self.calibration_input_type:
            self.coord_input_mode.set(self.calibration_input_type)
            messagebox.showwarning(
                "Coordinate type locked",
                f"This calibration already uses {self.calibration_input_type.upper()} points. "
                "Start a new video/calibration or clear points before switching coordinate type.",
            )
            return
        if selected_mode == "gps":
            text = simpledialog.askstring(
                "GPS coordinate",
                "Enter Google Maps decimal or Google Earth DMS coordinates:\n"
                "23.742866010823988, 90.39576489469023\n"
                "23°44'44.61\"N 90°23'41.02\"E",
                parent=self,
            )
            if text is None:
                return
            try:
                lat, lon = parse_lat_lon(text)
            except ValueError as exc:
                messagebox.showerror("Invalid GPS coordinate", str(exc))
                return
            point = {
                "image_px": img_pt,
                "coordinate_input": "gps",
                "raw_gps": [lat, lon],
                "world_m": [0.0, 0.0],
            }
        else:
            world_x = simpledialog.askfloat("World X", "Enter real-world X coordinate in meters:", parent=self)
            if world_x is None:
                return
            world_y = simpledialog.askfloat("World Y", "Enter real-world Y coordinate in meters:", parent=self)
            if world_y is None:
                return
            point = {
                "image_px": img_pt,
                "coordinate_input": "xy",
                "world_m": [float(world_x), float(world_y)],
            }

        self.push_history()
        self.state["calibration_points"].append(point)
        self.calibration_input_type = selected_mode
        try:
            self.rebuild_gps_relative_world()
        except ValueError as exc:
            self.state = self.history.pop()
            messagebox.showerror("GPS conversion error", str(exc))
            return
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()
        self.status_var.set(f"Added calibration point {len(self.state['calibration_points'])}.")

    def refresh_point_tree(self) -> None:
        for item in self.point_tree.get_children():
            self.point_tree.delete(item)
        if self.state["calibration_points"]:
            self.calibration_input_type = self.state["calibration_points"][0].get("coordinate_input", "xy")
            self.coord_input_mode.set(self.calibration_input_type)
        else:
            self.calibration_input_type = None
        for idx, point in enumerate(self.state["calibration_points"], start=1):
            px = point["image_px"]
            world = point["world_m"]
            axis_world_text = ""
            if self.pixel_axis_info:
                axis_world = self.base_to_output_world(world)
                axis_world_text = f"{axis_world[0]:.3f}, {axis_world[1]:.3f}"
            input_label = "GPS" if point.get("coordinate_input") == "gps" else "X/Y"
            self.point_tree.insert(
                "",
                tk.END,
                iid=str(idx - 1),
                values=(
                    idx,
                    f"{px[0]:.1f}, {px[1]:.1f}",
                    f"{world[0]:.3f}, {world[1]:.3f}",
                    axis_world_text,
                    input_label,
                ),
            )

    def selected_point_index(self) -> int | None:
        selection = self.point_tree.selection()
        if not selection:
            return None
        return int(selection[0])

    def edit_selected_point(self) -> None:
        idx = self.selected_point_index()
        if idx is None:
            messagebox.showinfo("Select point", "Select a calibration point first.")
            return
        point = self.state["calibration_points"][idx]
        self.push_history()
        if point.get("coordinate_input") == "gps":
            raw = point.get("raw_gps") or ["", ""]
            text = simpledialog.askstring(
                "GPS coordinate",
                "Enter decimal or DMS latitude/longitude:\n"
                "Example: 23°44'44.61\"N 90°23'41.02\"E",
                initialvalue=f"{raw[0]}, {raw[1]}",
                parent=self,
            )
            if text is None:
                self.history.pop()
                return
            try:
                lat, lon = parse_lat_lon(text)
                point["raw_gps"] = [lat, lon]
                self.rebuild_gps_relative_world()
            except ValueError as exc:
                self.state = self.history.pop()
                messagebox.showerror("Invalid GPS coordinate", str(exc))
                return
        else:
            world_x = simpledialog.askfloat(
                "World X",
                "Enter real-world X coordinate in meters:",
                initialvalue=point["world_m"][0],
                parent=self,
            )
            if world_x is None:
                self.history.pop()
                return
            world_y = simpledialog.askfloat(
                "World Y",
                "Enter real-world Y coordinate in meters:",
                initialvalue=point["world_m"][1],
                parent=self,
            )
            if world_y is None:
                self.history.pop()
                return
            point["world_m"] = [float(world_x), float(world_y)]
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()

    def delete_selected_point(self) -> None:
        idx = self.selected_point_index()
        if idx is None:
            messagebox.showinfo("Select point", "Select a calibration point first.")
            return
        self.push_history()
        del self.state["calibration_points"][idx]
        self.gps_axis_definition = None
        self.rebuild_gps_relative_world()
        self.compute_homography(show_errors=False)
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()

    def clear_active_area(self) -> None:
        self.push_history()
        self.state["active_area_polygon_px"] = []
        self.redraw_canvas()
        self.schedule_autosave()

    def clear_start_line(self) -> None:
        self.push_history()
        self.state["start_line_px"] = []
        self.redraw_canvas()
        self.schedule_autosave()

    def clear_measure(self) -> None:
        self.measure_points_px = []
        self.measure_distance_m = None
        self.redraw_canvas()
        self.status_var.set("Measurement cleared.")

    def clear_pixel_axis(self) -> None:
        if self.pixel_axis_info is not None:
            self.push_history()
        self.pixel_axis_info = None
        self.axis_capture = None
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()
        self.status_var.set("Image custom axis cleared.")

    def flip_perpendicular_axis(self) -> None:
        if not self.pixel_axis_info:
            messagebox.showinfo(
                "No image axis",
                "Set an axis from the image first, then use this button to reverse the perpendicular axis.",
            )
            return
        self.push_history()
        defined_axis = self.pixel_axis_info.get("axis", "y")
        if defined_axis == "y":
            self.pixel_axis_info["x_axis_base_unit"] = [
                -float(value) for value in self.pixel_axis_info["x_axis_base_unit"]
            ]
            flipped_name = "X"
        else:
            self.pixel_axis_info["y_axis_base_unit"] = [
                -float(value) for value in self.pixel_axis_info["y_axis_base_unit"]
            ]
            flipped_name = "Y"
        current_sign = int(self.pixel_axis_info.get("perpendicular_sign", 1))
        self.pixel_axis_info["perpendicular_sign"] = -1 if current_sign > 0 else 1
        self.refresh_point_tree()
        self.redraw_canvas()
        self.schedule_autosave()
        self.status_var.set(f"Flipped +{flipped_name}/-{flipped_name}; the defined +{defined_axis.upper()} direction is unchanged.")

    def zoom_in(self) -> None:
        self.zoom_var.set(min(8.0, float(self.zoom_var.get()) * 1.25))
        self.redraw_canvas()
        self.schedule_autosave()

    def zoom_out(self) -> None:
        self.zoom_var.set(max(0.25, float(self.zoom_var.get()) / 1.25))
        self.redraw_canvas()
        self.schedule_autosave()

    def zoom_fit(self) -> None:
        self.zoom_var.set(1.0)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.redraw_canvas()
        self.schedule_autosave()

    def on_mousewheel_zoom(self, event: tk.Event) -> str:
        if self.current_frame_rgb is None:
            return "break"
        old_zoom = float(self.zoom_var.get())
        if getattr(event, "num", None) in (4, 5):
            steps = 1.0 if event.num == 4 else -1.0
        else:
            delta = float(getattr(event, "delta", 0))
            steps = delta / 120.0 if abs(delta) >= 120 else (1.0 if delta > 0 else -1.0)
        factor = 1.12 ** steps
        new_zoom = min(8.0, max(0.25, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return "break"

        # Keep the same image pixel under the mouse. Fractions of the old
        # canvas bounding box drift badly when the image is centered or panned.
        anchor_image = self.event_to_image(event)
        self.zoom_var.set(new_zoom)
        self.redraw_canvas()
        self.update_idletasks()
        if anchor_image is not None:
            anchor_x, anchor_y = self.image_to_canvas(anchor_image)
            region = self.canvas.cget("scrollregion").split()
            if len(region) == 4:
                x0, y0, x1, y1 = map(float, region)
                region_w = max(1.0, x1 - x0)
                region_h = max(1.0, y1 - y0)
                self.canvas.xview_moveto((anchor_x - event.x - x0) / region_w)
                self.canvas.yview_moveto((anchor_y - event.y - y0) / region_h)
        self.schedule_autosave()
        return "break"

    def on_middle_press(self, event: tk.Event) -> None:
        self.panning = True
        self.canvas.configure(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)

    def on_middle_drag(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_middle_release(self, _event: tk.Event) -> None:
        self.panning = False
        self.canvas.configure(cursor="")

    def on_pan_press(self, event: tk.Event) -> str:
        self.panning = True
        self.canvas.configure(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)
        return "break"

    def on_pan_drag(self, event: tk.Event) -> str:
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        return "break"

    def on_pan_release(self, _event: tk.Event) -> str:
        self.panning = False
        self.canvas.configure(cursor="")
        return "break"

    def on_space_press(self, _event: tk.Event) -> None:
        self.panning = True
        self.canvas.configure(cursor="fleur")

    def on_space_release(self, _event: tk.Event) -> None:
        self.panning = False
        self.canvas.configure(cursor="")

    def on_shift_mousewheel_pan(self, event: tk.Event) -> str:
        units = -1 if event.delta > 0 else 1
        self.canvas.xview_scroll(units * 5, "units")
        return "break"

    def show_xy_axis_help(self) -> None:
        messagebox.showinfo(
            "Custom XY axes",
            "X/Y mode:\n"
            "Choose your own origin and axes directly. For queue work, use (0, 0) near the start line, +Y along the queue direction, and +X across the lane.\n\n"
            "GPS mode without custom axis:\n"
            "GPS is converted to local UTM meters. X is roughly East and Y is roughly North.\n\n"
            "GPS mode with custom axis:\n"
            "Add GPS calibration points, then click 'Set GPS Custom XY Axis'. Choose one GPS point as (0, 0), then another GPS point as the +Y road direction. The app will keep GPS input but convert it into your road-aligned XY plane.\n\n"
            "The plane grid overlay shows the final X/Y axes on the video. Red is X=0 and cyan is Y=0 when those axes are visible.",
        )

    def compute_homography(self, show_errors: bool = True) -> bool:
        points = self.state["calibration_points"]
        try:
            self.rebuild_gps_relative_world()
        except ValueError as exc:
            self.h_matrix = None
            self.h_status_var.set("Homography: GPS conversion failed")
            if show_errors:
                messagebox.showerror("GPS conversion error", str(exc))
            return False
        if len(points) < 4 or self.current_image_size is None:
            self.h_matrix = None
            self.mean_error_m = None
            self.reprojection_errors_m = []
            self.h_status_var.set(f"Homography: need at least 4 points ({len(points)} now)")
            return False

        w, h = self.current_image_size
        image_norm = np.array(
            [[p["image_px"][0] / w, p["image_px"][1] / h] for p in points],
            dtype=np.float32,
        )
        world = np.array([p["world_m"] for p in points], dtype=np.float32)
        method = cv2.RANSAC if len(points) > 4 else 0
        h_matrix, _mask = cv2.findHomography(image_norm, world, method=method, ransacReprojThreshold=0.5)
        if h_matrix is None:
            self.h_matrix = None
            self.h_status_var.set("Homography: failed to compute")
            if show_errors:
                messagebox.showerror("Homography failed", "OpenCV could not compute homography from these points.")
            return False

        projected = cv2.perspectiveTransform(image_norm.reshape(-1, 1, 2), h_matrix).reshape(-1, 2)
        errors = np.linalg.norm(projected - world, axis=1)
        self.h_matrix = h_matrix.tolist()
        self.reprojection_errors_m = [float(v) for v in errors]
        self.mean_error_m = float(np.mean(errors))
        self.h_status_var.set(f"Homography: ready, mean error {self.mean_error_m:.3f} m")
        return True

    def inspect_point(self, img_pt: list[float]) -> None:
        if self.h_matrix is None:
            if not self.compute_homography():
                return
        assert self.current_image_size is not None
        base_world = self.image_px_to_base_world(img_pt)
        if base_world is None:
            return
        world = self.base_to_output_world(base_world)
        self.inspect_marker_px = img_pt
        self.inspect_world_m = world
        w, h = self.current_image_size
        norm_x = img_pt[0] / w
        norm_y = img_pt[1] / h
        message = (
            f"Inspect: image px ({img_pt[0]:.1f}, {img_pt[1]:.1f})"
            f" | image norm ({norm_x:.6f}, {norm_y:.6f})"
            f" | base world ({base_world[0]:.3f}, {base_world[1]:.3f}) m"
        )
        if self.pixel_axis_info:
            message += f" | new axis world ({world[0]:.3f}, {world[1]:.3f}) m"
        if self.origin_info:
            lat, lon = world_to_gps(base_world[0], base_world[1], self.origin_info)
            message += f" | GPS ({lat:.8f}, {lon:.8f})"
        self.inspect_var.set(message)
        self.redraw_canvas()

    def selected_names(self, listbox: tk.Listbox, paths: list[Path]) -> list[str]:
        return [paths[int(i)].name for i in listbox.curselection()]

    def save_json(self, silent: bool = False) -> bool:
        if self.current_image_size is None:
            if not silent:
                messagebox.showwarning("No frame", "Load a frame before saving.")
            return False
        if not self.compute_homography(show_errors=not silent):
            return False

        self.output_dir = Path(self.output_folder_var.get().strip() or OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_folder_var.set(str(self.output_dir))
        self.save_app_settings()
        w, h = self.current_image_size
        active_px = self.state["active_area_polygon_px"]
        start_px = self.state["start_line_px"]
        name = self.current_video.stem if self.current_video else (self.name_var.get().strip() or "calibration")
        json_name = name if self.current_video else self.safe_json_stem(name)
        snapshot_name = f"{json_name}_frame_{int(self.current_frame_index.get())}.png"
        snapshot_path = self.output_dir / snapshot_name
        if self.current_frame_rgb is not None:
            Image.fromarray(self.current_frame_rgb).save(snapshot_path)

        final_world_system = "custom_image_axis_meters" if self.pixel_axis_info else ("relative_utm_meters" if self.origin_info else "relative_meters")
        final_matrices = final_homography_matrices(self.h_matrix, self.pixel_axis_info, w, h)

        payload = {
            "schema_version": 3,
            "metadata": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_video": self.current_video.name if self.current_video else None,
                "source_video_path": str(self.current_video) if self.current_video else None,
                "source_frame_index": int(self.current_frame_index.get()),
                "source_frame_image": snapshot_name if self.current_frame_rgb is not None else None,
            },
            "video_size": {"width": w, "height": h},
            "coordinate_reference": {
                "world_coordinate_system": final_world_system,
                "origin_info": self.origin_info,
                "pixel_custom_axis": self.pixel_axis_info,
            },
            "homography": {
                "image_coordinate_system": "normalized_0_1",
                "base_world_coordinate_system": "relative_utm_meters" if self.origin_info else "relative_meters",
                "final_world_coordinate_system": final_world_system,
                "output_world_coordinate_system": final_world_system,
                "H_image_norm_to_base_world": self.h_matrix,
                "H_base_world_to_final_world": final_matrices["base_to_final_world"],
                "H_image_norm_to_final_world": final_matrices["image_norm_to_final_world"],
                "H_image_px_to_final_world": final_matrices["image_px_to_final_world"],
                "H_image_norm_to_world": final_matrices["image_norm_to_final_world"],
                "matrix_usage": {
                    "H_image_norm_to_final_world": "Use with normalized image coordinates x,y in 0..1. Output is final real-world X,Y meters, including custom axis if present.",
                    "H_image_px_to_final_world": "Use with image pixel coordinates x,y. Output is final real-world X,Y meters, including custom axis if present.",
                    "H_image_norm_to_base_world": "Base calibration plane before optional image-picked custom axis.",
                    "H_base_world_to_final_world": "Optional axis transform from base meters to final output meters. Identity when no custom axis is used.",
                },
                "mean_error_m": self.mean_error_m,
                "reprojection_errors_m": self.reprojection_errors_m,
            },
            "calibration_points": [
                {
                    "image_px": [float(p["image_px"][0]), float(p["image_px"][1])],
                    "image_norm": [float(p["image_px"][0] / w), float(p["image_px"][1] / h)],
                    "world_m": [float(p["world_m"][0]), float(p["world_m"][1])],
                    "coordinate_input": p.get("coordinate_input", "xy"),
                    "raw_gps": p.get("raw_gps"),
                    "utm": p.get("utm"),
                }
                for p in self.state["calibration_points"]
            ],
            "active_area": {
                "polygon_px": [[float(x), float(y)] for x, y in active_px],
                "polygon_norm": [[float(x / w), float(y / h)] for x, y in active_px],
            },
            "start_line": {
                "line_px": [[float(x), float(y)] for x, y in start_px],
                "line_norm": [[float(x / w), float(y / h)] for x, y in start_px],
                "fallback": "video_bottom_if_missing",
            },
            "workspace": {
                "show_plane_grid": bool(self.show_grid_var.get()),
                "grid_spacing_m": float(self.grid_spacing_var.get()),
                "zoom": float(self.zoom_var.get()),
                "click_mode": self.mode.get(),
                "coordinate_input_mode": self.coord_input_mode.get(),
                "measure_points_px": [[float(x), float(y)] for x, y in self.measure_points_px],
                "measure_points_norm": [[float(x / w), float(y / h)] for x, y in self.measure_points_px],
                "inspect_marker_px": self.inspect_marker_px,
                "inspect_marker_norm": (
                    [float(self.inspect_marker_px[0] / w), float(self.inspect_marker_px[1] / h)]
                    if self.inspect_marker_px is not None
                    else None
                ),
            },
        }

        save_path = self.output_dir / f"{json_name}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.video_combo["values"] = [self.video_display_name(p) for p in self.video_paths]
        if self.current_video in self.video_paths:
            self.video_combo.current(self.video_paths.index(self.current_video))
        if silent:
            self.status_var.set(f"Autosaved calibration JSON: {save_path}")
        else:
            self.status_var.set(f"Saved calibration JSON: {save_path}")
        return True

    def open_current_json(self) -> None:
        if self.current_video is None:
            messagebox.showwarning("No video", "Select a video first.")
            return
        path = self.current_json_path()
        if path is None:
            return
        if not path.exists():
            messagebox.showinfo("JSON not found", f"No matching JSON found:\n{path}")
            return
        self.load_json_file(path)

    def open_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Open calibration JSON",
            initialdir=str(self.output_dir),
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        self.load_json_file(Path(path))

    def find_video_for_json(self, json_path: Path, data: dict[str, Any]) -> Path | None:
        """Resolve a JSON to a video without trusting stale embedded metadata first.

        V5 deliberately gives the JSON filename priority. This supports copying a
        calibration JSON to another time-period video and renaming the copy to the
        new video's stem, even when metadata.source_video(_path) still names the
        original recording. No JSON content is modified while resolving it.
        """
        json_stem = json_path.stem.casefold()
        for video_path in self.video_paths:
            candidate_stems = {
                video_path.stem.casefold(),
                self.safe_json_stem(video_path.stem).casefold(),
            }
            if json_stem in candidate_stems:
                return video_path

        # When the user selected a video and opened a differently named/template
        # JSON explicitly, retain that selection before consulting old metadata.
        if self.current_video is not None:
            current_stems = {
                self.current_video.stem.casefold(),
                self.safe_json_stem(self.current_video.stem).casefold(),
            }
            if json_stem in current_stems:
                return self.current_video

        metadata = data.get("metadata") or {}
        source_path = metadata.get("source_video_path")
        if source_path:
            candidate = Path(source_path)
            if candidate.exists():
                return candidate
        source_name = metadata.get("source_video")
        if source_name:
            for video_path in self.video_paths:
                if video_path.name.lower() == str(source_name).lower():
                    return video_path
        return None

    def load_saved_snapshot(self, json_path: Path, data: dict[str, Any]) -> bool:
        """Load the saved calibration frame without seeking through the video."""
        snapshot_name = (data.get("metadata") or {}).get("source_frame_image")
        if not snapshot_name:
            return False
        snapshot_path = json_path.parent / str(snapshot_name)
        if not snapshot_path.exists():
            return False
        try:
            with Image.open(snapshot_path) as snapshot:
                frame_rgb = np.asarray(snapshot.convert("RGB"), dtype=np.uint8).copy()
        except (OSError, ValueError, MemoryError):
            return False
        self.current_frame_rgb = frame_rgb
        height, width = frame_rgb.shape[:2]
        self.current_image_size = (width, height)
        self.tk_image = None
        self.tk_image_size = None
        self.tk_image_frame_token = None
        self.status_var.set(f"Loaded saved frame snapshot: {snapshot_path.name}")
        return True

    def load_json_file(self, path: Path) -> None:
        self.loading_json = True
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            video_size = data.get("video_size") or {}
            width = int(video_size.get("width") or 0)
            height = int(video_size.get("height") or 0)
            target_video = self.find_video_for_json(path, data)
            if target_video:
                for idx, video_path in enumerate(self.video_paths):
                    if video_path == target_video:
                        self.video_combo.current(idx)
                        self._on_video_selected()
                        break
                saved_frame = int((data.get("metadata") or {}).get("source_frame_index") or 0)
                self.current_frame_index.set(saved_frame)
                if not self.load_saved_snapshot(path, data):
                    self.load_frame()
            else:
                self.reset_calibration_workspace(clear_frame=True)

            self.name_var.set(target_video.stem if target_video else path.stem)

            if self.current_image_size is None and width and height:
                self.current_image_size = (width, height)

            w, h = self.current_image_size or (width, height)
            if not w or not h:
                messagebox.showwarning("Need frame", "Load the matching video frame before editing this JSON.")
                return

            def point_from_json(item: dict[str, Any], px_key: str = "image_px", norm_key: str = "image_norm") -> list[float] | None:
                norm = item.get(norm_key)
                if norm and len(norm) == 2:
                    return [float(norm[0]) * w, float(norm[1]) * h]
                px = item.get(px_key)
                if px and len(px) == 2:
                    return [float(px[0]), float(px[1])]
                return None

            def points_from_json(section: dict[str, Any], px_key: str, norm_key: str) -> list[list[float]]:
                norm_points = section.get(norm_key) or []
                if norm_points:
                    return [[float(x) * w, float(y) * h] for x, y in norm_points]
                return [[float(x), float(y)] for x, y in section.get(px_key, [])]

            self.state["calibration_points"] = [
                {
                    "image_px": point_from_json(item),
                    "world_m": item.get("world_m"),
                    "coordinate_input": item.get("coordinate_input", "gps" if item.get("raw_gps") else "xy"),
                    "raw_gps": item.get("raw_gps"),
                    "utm": item.get("utm"),
                }
                for item in data.get("calibration_points", [])
                if point_from_json(item) and item.get("world_m")
            ]
            active = data.get("active_area", {})
            start = data.get("start_line", {})
            self.state["active_area_polygon_px"] = points_from_json(active, "polygon_px", "polygon_norm")
            self.state["start_line_px"] = points_from_json(start, "line_px", "line_norm")
            workspace = data.get("workspace") or {}
            self.show_grid_var.set(bool(workspace.get("show_plane_grid", False)))
            try:
                self.grid_spacing_var.set(float(workspace.get("grid_spacing_m", 5.0)))
                self.zoom_var.set(min(8.0, max(0.25, float(workspace.get("zoom", 1.0)))))
            except (tk.TclError, TypeError, ValueError):
                pass
            saved_mode = workspace.get("click_mode")
            if saved_mode in {"calibrate", "measure", "active", "start", "axis", "inspect"}:
                self.mode.set(saved_mode)
            saved_coord_mode = workspace.get("coordinate_input_mode")
            if saved_coord_mode in {"xy", "gps"}:
                self.coord_input_mode.set(saved_coord_mode)
            self.measure_points_px = points_from_json(workspace, "measure_points_px", "measure_points_norm")[:2]
            self.inspect_marker_px = point_from_json(workspace, "inspect_marker_px", "inspect_marker_norm")
            homography = data.get("homography", {})
            self.origin_info = (data.get("coordinate_reference") or {}).get("origin_info")
            self.pixel_axis_info = (data.get("coordinate_reference") or {}).get("pixel_custom_axis")
            if self.pixel_axis_info:
                origin_px = point_from_json(self.pixel_axis_info, "origin_px", "origin_norm")
                axis_px = point_from_json(self.pixel_axis_info, "axis_point_px", "axis_point_norm")
                if origin_px:
                    self.pixel_axis_info["origin_px"] = origin_px
                if axis_px:
                    self.pixel_axis_info["axis_point_px"] = axis_px
            self.gps_axis_definition = (self.origin_info or {}).get("gps_axis_definition")
            if self.gps_axis_definition is None and (self.origin_info or {}).get("gps_axis_point_indices"):
                axis_indices = self.origin_info["gps_axis_point_indices"]
                self.gps_axis_definition = {
                    key: {
                        "source": "calibration_point",
                        "point_index": idx,
                        "raw_gps": self.state["calibration_points"][idx].get("raw_gps"),
                    }
                    for key, idx in axis_indices.items()
                    if 0 <= idx < len(self.state["calibration_points"])
                }
            self.h_matrix = homography.get("H_image_norm_to_base_world") or homography.get("H_image_norm_to_world")
            self.mean_error_m = homography.get("mean_error_m")
            self.reprojection_errors_m = homography.get("reprojection_errors_m") or []
            self.compute_homography(show_errors=False)
            self.measure_distance_m = None
            if len(self.measure_points_px) == 2 and self.h_matrix is not None:
                base_points = [self.image_px_to_base_world(point) for point in self.measure_points_px]
                if all(point is not None for point in base_points):
                    output_points = [self.base_to_output_world(point) for point in base_points if point is not None]
                    self.measure_distance_m = math.hypot(
                        output_points[1][0] - output_points[0][0],
                        output_points[1][1] - output_points[0][1],
                    )
            self.inspect_world_m = None
            if self.inspect_marker_px is not None and self.h_matrix is not None:
                inspect_base = self.image_px_to_base_world(self.inspect_marker_px)
                if inspect_base is not None:
                    self.inspect_world_m = self.base_to_output_world(inspect_base)
            self.refresh_point_tree()
            self.redraw_canvas()
            self.history.clear()
            self.redo_history.clear()
            embedded_name = str((data.get("metadata") or {}).get("source_video") or "")
            if target_video and embedded_name and embedded_name.casefold() != target_video.name.casefold():
                self.status_var.set(
                    f"Loaded calibration JSON: {path} | V5 opened {target_video.name} from the JSON filename; "
                    f"ignored stale embedded source name {embedded_name}."
                )
            else:
                self.status_var.set(f"Loaded calibration JSON: {path}")
        finally:
            self.loading_json = False
            self.autosave_dirty = False


def main() -> None:
    app = CalibrationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
