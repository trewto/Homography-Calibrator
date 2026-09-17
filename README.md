# Video Homography Calibration V5

**Turn video-frame points into real-world road-plane coordinates in metres.**

A Python desktop application for interactively calibrating a planar scene from a video frame. Match image points to known **local X/Y coordinates** or **GPS latitude/longitude**, define a useful coordinate system, inspect distances and positions, and export the calibration for use in other analysis pipelines.

**Local X/Y or GPS · Custom road-aligned axes · Ground-plane grid · Reusable JSON output**

Developed by **Arnob Protim Roy**.

> This is a geometric calibration tool—not an object detector or tracker. It creates the coordinate transformation and scene geometry that a separate video-analysis pipeline can use.

## Contents

[Features](#features) · [Installation](#installation) · [Calibration workflow](#calibration-workflow) · [Custom axes](#custom-axes) · [Controls](#controls) · [Exported files](#exported-files) · [Using the calibration in Python](#using-the-calibration-in-python) · [Calibration quality and limitations](#calibration-quality-and-limitations) · [Troubleshooting](#troubleshooting)

## Features

| Capability | What you can do |
| --- | --- |
| **Video-frame calibration** | Select a source folder, choose a video, and load a frame by its zero-based index. |
| **Two coordinate-input modes** | Enter local X/Y values in metres, or GPS coordinates in decimal degrees or degrees–minutes–seconds (DMS). |
| **GPS-to-metre conversion** | Convert WGS 84 latitude/longitude into a local coordinate system derived from UTM. |
| **Custom coordinate axes** | Define a GPS-based origin and +Y direction, or choose an origin and +X/+Y direction directly on the calibrated image. |
| **Interactive editing** | Add, drag, edit and delete calibration points; edit active-area vertices and image-axis handles; undo and redo workspace edits. |
| **Scene inspection** | Overlay a metre-spaced plane grid, inspect real-world coordinates, and measure the planar distance between two clicked points. |
| **Scene geometry** | Draw an active-area polygon and a two-point start line for downstream processing. |
| **Reusable output** | Save calibration matrices, control points, coordinate-reference information and workspace settings in JSON, together with an unannotated PNG frame snapshot. |
| **Workspace convenience** | Use cursor-centred zoom, panning, automatic loading of matching JSON files and optional timed autosave. |

## Installation

### Requirements

Use a desktop Python environment with **Tkinter/Tcl–Tk** available. Python **3.10 or newer** is a suggested environment; the source does not provide a tested-version matrix.

The third-party packages imported by the application are:

```text
numpy
opencv-python
pyproj
Pillow
```

Tkinter is supplied through your Python installation or operating system rather than through this package list. The script does not require YOLO weights, PyTorch or a GPU-specific dependency.

### Install and launch

From the directory containing the script:

```bash
python -m pip install numpy opencv-python pyproj Pillow
python "calibration_gui_v5(1).py"
```

Alternatively, with the accompanying `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python "calibration_gui_v5(1).py"
```

The commands use the supplied filename, including `(1)`. After renaming the script, use its new filename in the launch command.

Check that Tkinter can open a window with:

```bash
python -m tkinter
```

A virtual environment is optional. Create one with `python -m venv .venv`, activate it using the command appropriate to your shell, and run the installation commands inside it. On systems where Python is invoked as `python3`, use that command consistently.

### Video formats

The source-folder browser recognises `.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v` and `.dav`.

**Recognition of an extension does not guarantee decoding.** The application reads frames through OpenCV's `VideoCapture`; a file must be readable by the video backend available in your environment. This is particularly important for recorder-specific DAV files.

## Calibration workflow

### 1. Select folders and load a frame

In **Step 1–2: Folders**, choose the **Video source folder** and **Output JSON folder**, then click **Load**. Videos are listed from the selected folder itself; subfolders are not scanned recursively.

In **Step 3: Video**, select a video, set the **Frame** index and click **Load**. Frame `0` is the first frame. Choose a frame in which the required control points are visible.

A video with a matching calibration JSON is marked with `*` in the selector. Selecting it attempts to load the saved calibration automatically. **Open Current JSON** and **Open Other JSON** are also available.

> Save your current calibration before switching videos or reloading the source folder. These actions reset the calibration workspace; autosave is not a replacement for confirming that your work has been saved.

### 2. Choose one coordinate-input mode

| Mode | Input | Resulting base coordinates |
| --- | --- | --- |
| **World X/Y meters** | The known X and Y coordinates of each clicked point, entered in separate dialogs. | Your supplied local metric coordinate system. |
| **GPS latitude/longitude** | Latitude first, longitude second. | Local UTM-derived metres, optionally aligned using a GPS custom axis. |

Supported GPS examples, as shown by the application:

```text
Decimal degrees: 23.742866010823988, 90.39576489469023
DMS:             23°44'44.61"N 90°23'41.02"E
```

These are independent examples of the two input formats, not equivalent coordinates or control points for your video.

Without a GPS custom axis, the **first GPS calibration point becomes the local origin**. X follows UTM easting and Y follows UTM northing. All GPS points and GPS-axis references must resolve to the same UTM coordinate reference system.

**One calibration uses one input type.** After points have been added, the application prevents mixing GPS and local-X/Y inputs. Remove the existing points or begin another calibration to change input type.

### 3. Add corresponding control points

Select **Add calibration point**, click a known ground-plane location in the video frame, and enter its corresponding real-world coordinates. Repeat for at least **four distinct, non-collinear point correspondences**.

The point table shows each point's image position, base-world coordinates, optional new-axis coordinates and input type. Drag a point in calibration mode to refine its image position. Double-click its table row, or use **Edit Selected**, to change its world-coordinate input.

### 4. Compute and inspect the homography

Click **Compute H**. The application also recomputes the homography during calibration-point edits.

| Number of correspondences | Implemented estimation method |
| --- | --- |
| Fewer than 4 | No homography is computed. |
| Exactly 4 | `cv2.findHomography(..., method=0)`. |
| More than 4 | `cv2.findHomography(..., method=cv2.RANSAC, ransacReprojThreshold=0.5)`. |

Image coordinates are normalised as `(x / width, y / height)` before estimation. Destination coordinates are in metres, so the RANSAC threshold is **0.5 in the destination metre coordinate system**.

The status panel displays the **mean control-point reprojection error in metres**. Individual control-point errors are saved in JSON. The displayed mean includes all supplied control points, not only RANSAC inliers; the inlier mask itself is not exported.

### 5. Define the axes and scene geometry

Set custom axes when needed, then draw the **active area** and **start line** using their click modes. An active-area polygon needs at least three vertices to form an area; the start line uses two points.

Turn on **Show plane grid** to inspect the mapping visually. Adjust its spacing in metres. Use **Inspect real-world coordinate** to examine an individual location, or **Measure distance** to click two points and display their planar separation.

The active area and start line are exported geometry. The calibrator does **not** itself filter detections, count crossings or measure queue lengths from a video.

### 6. Save the calibration

Click **Save Video JSON** or press **Ctrl+S**. Saving requires a loaded frame and a successfully computed homography; incomplete calibrations with fewer than four valid point correspondences are not saved by this action.

Optional **Autosave** is off by default. Its initial interval is 10 seconds, with a minimum of 2 seconds. Autosave uses the same successful-calibration requirement as manual saving.

## Custom axes

### GPS-defined axes

After adding GPS control points, click **Set/Edit GPS Custom XY Axis**. Supply an **Origin (0,0)** and a **+Y direction point**. Each field accepts either an existing GPS control-point number or a new latitude/longitude pair.

The application projects the references to UTM, sets +Y towards the direction point and constructs perpendicular +X. GPS inputs are retained while the base coordinates become road-aligned metres. A useful arrangement for queue analysis is an origin near the start line, +Y along the queue and +X across the lane.

### Image-picked axes

After computing the base homography:

1. Select **Set/edit axis from image** and click the intended origin on the image.
2. In the dialog, choose whether the direction point will define **+X** or **+Y**, then click **Start clicking**. The initial image click is used as the origin.
3. Click another image point in the chosen positive direction.

The other axis is constructed perpendicular in the calibrated world plane—not merely perpendicular on the perspective image. Both handles can subsequently be dragged in axis mode.

**Flip Perpendicular Axis (+/-)** reverses only the automatically generated perpendicular axis. **Clear Image Axis** removes this optional image-picked transform; it does not remove the GPS-defined base axis.

Finalise the base calibration before defining an image-picked axis. After changing base control coordinates or the GPS reference system, redefine the image-picked axis so its stored metric basis matches the revised calibration.

> The image-axis transform is already included in the exported **final-world** homography matrices. Do not apply it a second time in downstream code.

## Controls

| Action | Mouse or keyboard control |
| --- | --- |
| Add a point | Left-click in the corresponding click mode. |
| Move a calibration point | Left-drag near that point in calibration mode. |
| Edit point coordinates | Double-click its table row or use **Edit Selected**. |
| Delete a calibration point | Right-click near it in calibration mode, or use **Delete Selected**. |
| Edit an active-area vertex | Drag it in **Draw active area** mode; right-click near it to delete. |
| Set the start line | Click two endpoints. A further click starts a new line; right-click removes the last endpoint. |
| Measure a distance | Click two points in **Measure distance** mode; right-click clears the measurement. |
| Zoom | Mouse wheel, **Zoom +**, **Zoom -**, or **Fit**. |
| Pan | Middle-button drag, **Ctrl+left-drag**, or **Space+left-drag**. |
| Pan horizontally | **Shift+mouse wheel**. |
| Save / Undo / Redo | **Ctrl+S** / **Ctrl+Z** / **Ctrl+Y**. |

Undo/redo stores up to 100 undo snapshots for calibration workspace edits. View changes and temporary measurement actions are not all part of that history.

## Exported files

For an illustrative video named `intersection.mp4`, saving frame `120` produces:

```text
selected-output-folder/
├── intersection.json
└── intersection_frame_120.png
```

The PNG is the source frame **without canvas overlays**. The JSON uses **schema version 3** and contains:

| Section | Contents |
| --- | --- |
| `metadata` | Creation timestamp, video name/path, frame index and snapshot filename. |
| `video_size` | Calibration-frame width and height. |
| `coordinate_reference` | Coordinate-system identifier, GPS/UTM reference information and optional image-picked axes. |
| `homography` | Base and final matrices, matrix-usage notes, mean error and per-point errors. |
| `calibration_points` | Pixel and normalised image coordinates, base `world_m` values, input type, and GPS/UTM details where applicable. |
| `active_area` | Polygon vertices in pixels and normalised coordinates. |
| `start_line` | Endpoints in pixels and normalised coordinates, plus the `video_bottom_if_missing` fallback identifier. |
| `workspace` | Grid, zoom, click-mode, coordinate-input, measurement-point and inspection-marker settings. |

The start-line fallback is a saved identifier; a separate consumer must implement any corresponding fallback behaviour.

### Choose the correct matrix

All the following keys are inside the JSON's `homography` object:

| Matrix key | Expected input | Output |
| --- | --- | --- |
| `H_image_px_to_final_world` | Image pixels at the saved calibration resolution. | Final X/Y in metres, including the image-picked axis if present. |
| `H_image_norm_to_final_world` | Normalised image coordinates `(x / width, y / height)`. | The same final X/Y in metres. |
| `H_image_norm_to_world` | Normalised image coordinates. | Alias of the final-world normalised matrix in schema v3. |
| `H_image_norm_to_base_world` | Normalised image coordinates. | Base calibration plane, before the optional image-picked axis. |
| `H_base_world_to_final_world` | Base-world X/Y in metres. | Final coordinates; identity without an image-picked axis. |

The pixel matrix uses the saved frame dimensions. For a differently sized version of the **same uncropped camera view**, normalise using that version's width and height and use the normalised-input matrix. A crop, letterbox border or camera-view change requires the corresponding coordinate adjustment or recalibration.

### Settings and reopening

The script stores application settings beside itself as `calibration_gui_v4_settings.json`. **The `v4` filename is retained by the V5 implementation**. These settings include the source/output folders and autosave preferences.

Default source and output folders are `Separate_4by3` and `output` under the parent of the script's directory. Both can be changed in the GUI; they do not need to match your repository layout.

V5 gives a matching **JSON filename** priority over old embedded video metadata. This supports copying a calibration to another recording from the same unchanged camera view and renaming the JSON to the destination video's stem. Verify the actual target frame before reusing it. Saved snapshots may still show the original calibration frame; click the video's frame **Load** button to inspect the destination recording.

## Using the calibration in Python

The example below applies the exported final-world matrix directly to pixel coordinates. It is a downstream usage example, not an additional GUI feature.

```python
import json
from pathlib import Path

import numpy as np


def pixels_to_world(json_path: str | Path, points_px) -> np.ndarray:
    """Map an (N, 2) array of calibration-resolution pixels to final metres."""
    with Path(json_path).open("r", encoding="utf-8-sig") as handle:
        calibration = json.load(handle)

    matrix_data = calibration.get("homography", {}).get(
        "H_image_px_to_final_world"
    )
    if matrix_data is None:
        raise ValueError("A schema-v3 final pixel-to-world matrix is required.")

    matrix = np.asarray(matrix_data, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("The homography must be a finite 3 x 3 matrix.")

    points = np.asarray(points_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Provide points as an (N, 2) array of [x, y] pixels.")
    if not np.isfinite(points).all():
        raise ValueError("Pixel coordinates must be finite.")

    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = homogeneous @ matrix.T
    denominator = mapped[:, 2]
    if np.any(np.isclose(denominator, 0.0)):
        raise ValueError("A point maps near the homography's projective horizon.")

    world = mapped[:, :2] / denominator[:, None]
    if not np.isfinite(world).all():
        raise ValueError("The transformation produced non-finite coordinates.")
    return world


# Illustrative pixel coordinates; replace with points from your own video.
world_xy = pixels_to_world("output/intersection.json", [[640, 720], [960, 720]])
print("Final coordinates in metres:\n", world_xy)
print("Planar distance in metres:", np.linalg.norm(world_xy[1] - world_xy[0]))
```

Apply any active-area inclusion test separately. This function transforms the supplied points; it does not decide whether those points are inside a reliable calibrated region.

## Calibration quality and limitations

**Use the transformation as a planar measurement model, not as a guarantee of survey accuracy.** The following are practical safeguards for interpreting and using its output.

- **Use one approximately planar surface.** Select control points on the road or other target plane. The implementation does not reconstruct 3D positions or correct radial lens distortion.
- **Check geometry independently.** Spread control points across the intended area and check a known distance not used to fit the homography. The reported reprojection error is a fit diagnostic, not an independent accuracy test.
- **Keep the camera geometry consistent.** Moving, rotating, zooming or cropping the view can invalidate the saved mapping. Normalised coordinates address image-size conventions, not camera motion.
- **Check the coordinate reference.** GPS input must be latitude then longitude, within one UTM reference system. The tool transforms the coordinates supplied; it does not assess their measurement accuracy.
- **Recheck axes after calibration edits.** In particular, deleting a calibration point clears the GPS custom-axis definition. Confirm the intended origin, directions and any image-picked axis again before exporting.
- **Keep the scope clear.** This GUI does not perform detection, tracking, speed estimation, video synchronisation, bulk trajectory-CSV transformation or automatic selection of ground-control points. Those belong to downstream tools.

The grid is a visual check, not an accuracy certificate. It is drawn over the image, with density safeguards near the horizon; it is not automatically clipped to the active-area polygon.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Install the packages using the same Python interpreter used to launch the script. |
| Tkinter cannot open a window | Confirm that Python has working Tcl/Tk support and that a graphical desktop/display is available. |
| No videos appear | Check the selected folder, supported extensions and **Load** button. Nested folders are not scanned. |
| Video or frame cannot be opened | Try frame `0`; confirm the file is readable by your OpenCV backend. An accepted extension alone is insufficient. |
| Homography is unavailable | Load a frame and add at least four suitable image/world correspondences. Check for repeated or collinear points. |
| GPS coordinate type is locked | The current calibration already contains one input type. Begin a fresh calibration or remove its points before switching. |
| GPS conversion reports a zone mismatch | Check that all control points and axis references use the same UTM zone and hemisphere. |
| Coordinates have unexpected scale or direction | Check metre units, latitude/longitude order, correspondence order, the final axis and whether the selected matrix expects pixels or normalised inputs. |
| Grid is crowded or hidden | Increase grid spacing and inspect the point geometry. The renderer limits excessive line density and pathological plane extents. |
| Changes were not autosaved | Enable autosave and confirm a valid homography exists. Save manually before changing videos or closing. |
| A copied JSON opens an old-looking frame | Check `metadata.source_frame_image`; it may refer to the original snapshot. Load a frame from the destination video to verify the reused calibration. |

## Author and contributions

**Arnob Protim Roy**

For a reproducible issue report, include your Python and package versions, operating system, video format, steps to reproduce, and the error message. Share a minimal non-sensitive example when possible.

Calibration JSON can contain source-video paths and GPS coordinates. Review those fields, along with any exported frame image, before publishing them.

## Licence

The supplied script does not declare a licence. This README does not assign one; add the chosen licence separately before presenting the repository as licensed for reuse.
