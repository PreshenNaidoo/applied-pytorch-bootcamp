"""
============================================================
EXERCISE 17: CAMERA GEOMETRY BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a small camera-calibration mini-project.

Camera calibration means:

```text
known 3D point on a calibration board -> observed 2D pixel in a real image
```

Your job is to fit/check a camera model so projected 3D points land close to
the observed pixels.

Plain meaning:
- a 3D point is a point on the real checkerboard, shaped `[x, y, z]`
- a pixel point is where that checkerboard corner appears in the image, shaped `[u, v]`
- intrinsics are camera settings such as focal length and image center
- extrinsics are the board/camera pose, meaning where the board is relative to the camera
- reprojection error is the pixel distance between observed corners and projected corners

WHY THIS EXERCISE EXISTS
Embodied agents need to connect image pixels to 3D space.
Before 3D navigation, SLAM, depth, or robot control, you need the basic camera
equation:

```text
3D point -> camera coordinates -> image pixel
```

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
real calibration board              OpenCV chessboard sample image
measured checkerboard corners       CSV of detected corner pixels
known board geometry                regular 9 by 6 corner grid
camera model                        small pinhole camera in PyTorch
calibration quality                 reprojection error in pixels

WHY THIS IS NOT A SYNTHETIC DATASET
The default image is a real OpenCV sample calibration photo:

```text
https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/left01.jpg
```

The observed 2D points should come from detected checkerboard corners in that
real image.

This exercise does not use random 2D points.
The known 3D board points are not fake labels. They are the measured layout of
the calibration board: every checkerboard square is the same physical size.

DATASET
Default dataset:
- image: OpenCV sample `left01.jpg`
- board pattern: 9 columns by 6 rows of inner checkerboard corners
- measured corner CSV: `data/camera_calibration/opencv_left01_corners.csv`

The CSV should have this exact header:

```text
corner_id,u,v
0, ...
1, ...
```

How to create the CSV:
1. Download the image from `config.image_url`.
2. Use OpenCV `cv2.findChessboardCorners(gray, (9, 6))`.
3. Save each detected corner row as `corner_id,u,v`.
4. Keep the same order OpenCV returns. That order matches the object-point grid.

This exercise does not require OpenCV as a main dependency. The core learning
task uses the saved CSV, PyTorch, and Matplotlib.

WHAT IS INCLUDED AND WHAT IS SKIPPED
Included:
- loading real measured 2D corner observations
- building known 3D checkerboard points
- projecting 3D points into pixels
- fitting simple camera parameters by minimizing reprojection error
- plotting observed vs projected corners

Skipped:
- full multi-image Zhang calibration
- lens distortion
- automatic corner detection code

Those are important in production, but this file teaches the first camera-model
loop before adding more machinery.

INPUTS AND OUTPUTS
Object points:
- shape: `[num_points, 3]`
- units: meters
- example: `[0.00, 0.00, 0.0]`, `[0.025, 0.00, 0.0]`

Observed pixels:
- shape: `[num_points, 2]`
- units: pixels
- example: `[u, v]`

Projected pixels:
- shape: `[num_points, 2]`
- units: pixels

CAMERA FORMULA

```text
x_camera = R @ x_world + t
u = fx * x_camera / z_camera + cx
v = fy * y_camera / z_camera + cy
```

Meaning:
- `R` rotates points into the camera view
- `t` moves points into the camera view
- `fx` and `fy` are focal lengths in pixels
- `cx` and `cy` are the image center in pixels
- `z_camera` must be positive because the point must be in front of the camera

LOSS FUNCTION

```text
reprojection_loss = mean((projected_pixels - observed_pixels) ** 2)
```

METRICS TO TRACK
- train mean reprojection error in pixels: lower is better
- validation mean reprojection error in pixels: lower is better
- final loss: lower is better

VISUALS TO CREATE
Save these files:
- `outputs/camera_geometry/reprojection_overlay.png`
- `outputs/camera_geometry/reprojection_error_histogram.png`
- `outputs/camera_geometry/calibration_loss_curve.png`

COMMON MISTAKES
- using one hand-made 3D point instead of many real measured matches
- mixing up `u, v` image coordinates with `x, y, z` world coordinates
- dividing by `z` when `z <= 0`
- thinking low reprojection error proves the full camera is calibrated
- forgetting that this simplified version skips lens distortion

STEP-BY-STEP TODO
1. Download or load the real OpenCV checkerboard image.
2. Load measured checkerboard corner pixels from CSV.
3. Build known 3D checkerboard points.
4. Project 3D points through a pinhole camera.
5. Compute reprojection error.
6. Fit simple camera parameters with PyTorch optimization.
7. Report train and validation pixel error.
8. Save overlay, histogram, and loss-curve plots.

DEFINITION OF DONE
You are done when:
- observed pixels come from a real checkerboard image, not random points
- more than one calibration match is used
- train and validation reprojection error are reported
- all three visual files are saved
- you can explain `fx`, `fy`, `cx`, `cy`, rotation, translation, and reprojection error
- every completed stub keeps its original `raise NotImplementedError(...)` line commented out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import sys
from typing import TypedDict

import torch
import torch.nn as nn

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed


# Learning order for this exercise:
# 1. Load real measured 2D corner points.
# 2. Build the known 3D checkerboard points.
# 3. Project 3D points with a pinhole camera.
# 4. Fit camera parameters by reducing reprojection error.
# 5. Save metrics and visuals.


class CalibrationSplits(TypedDict):
    """Train and validation calibration matches."""

    train_object_points: torch.Tensor
    train_pixels: torch.Tensor
    val_object_points: torch.Tensor
    val_pixels: torch.Tensor


class CameraCalibrationMetrics(TypedDict):
    """Final metrics and visual paths."""

    final_loss: float
    train_reprojection_error_px: float
    val_reprojection_error_px: float
    overlay_path: str
    histogram_path: str
    loss_curve_path: str


@dataclass
class CameraGeometryConfig:
    """Settings for the camera geometry mini-project."""

    image_url: str = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/left01.jpg"
    data_dir: Path = PROJECT_ROOT / "data" / "camera_calibration"
    image_path: Path = PROJECT_ROOT / "data" / "camera_calibration" / "left01.jpg"
    corner_csv_path: Path = PROJECT_ROOT / "data" / "camera_calibration" / "opencv_left01_corners.csv"
    output_dir: Path = PROJECT_ROOT / "outputs" / "camera_geometry"
    board_cols: int = 9
    board_rows: int = 6
    square_size_m: float = 0.025
    image_width: int = 640
    image_height: int = 480
    train_fraction: float = 0.8
    steps: int = 500
    learning_rate: float = 5e-2
    random_state: int = 42


def download_calibration_image(config: CameraGeometryConfig) -> Path:
    """Download the real OpenCV checkerboard image if it is missing."""
    # Where this fits:
    # This gets the real image used to create the measured corner CSV.
    #
    # Goal:
    # Ensure `config.image_path` exists.
    #
    # Method:
    # 1. If `config.image_path` exists, return it.
    # 2. Create `config.data_dir`.
    # 3. Import `requests` inside this function.
    # 4. Download `config.image_url` with `requests.get(config.image_url, timeout=30)`.
    # 5. Call `response.raise_for_status()`.
    # 6. Write `response.content` to `config.image_path`.
    # 7. Return `config.image_path`.
    #
    # Checks:
    # - The URL should be the OpenCV sample image URL from the file header.
    # - Do not generate an image in code.
    raise NotImplementedError("Download the real OpenCV calibration image.")


def load_measured_corners(config: CameraGeometryConfig) -> torch.Tensor:
    """Load measured checkerboard corner pixels from CSV."""
    # Where this fits:
    # These are the real 2D observations the camera model must match.
    #
    # Input file:
    # `config.corner_csv_path`
    #
    # Required CSV columns:
    # - `corner_id`
    # - `u`
    # - `v`
    #
    # Output:
    # - tensor shaped `[board_rows * board_cols, 2]`
    # - dtype `torch.float32`
    #
    # Method:
    # 1. Check that `config.corner_csv_path` exists.
    # 2. Read it with Python's standard `csv.DictReader`.
    # 3. Sort rows by integer `corner_id`.
    # 4. Convert `u` and `v` to floats.
    # 5. Return a tensor with columns `[u, v]`.
    #
    # If the CSV is missing:
    # Raise `FileNotFoundError` with a message that says to create it from
    # `cv2.findChessboardCorners(gray, (config.board_cols, config.board_rows))`.
    _ = csv
    raise NotImplementedError("Load measured checkerboard corner pixels.")


def build_checkerboard_object_points(config: CameraGeometryConfig) -> torch.Tensor:
    """Return known 3D checkerboard corner coordinates."""
    # Where this fits:
    # These are the known physical 3D points on the calibration board.
    #
    # Goal:
    # Build a regular grid of 3D points on the z=0 checkerboard plane.
    #
    # Output:
    # - tensor shaped `[board_rows * board_cols, 3]`.
    #
    # Method:
    # 1. Loop over row from `0` to `config.board_rows - 1`.
    # 2. Loop over col from `0` to `config.board_cols - 1`.
    # 3. Create point:
    #    `[col * config.square_size_m, row * config.square_size_m, 0.0]`.
    # 4. Return all points as `torch.float32`.
    #
    # Checks:
    # - This is not a random dataset. It is the known board geometry.
    # - The number of object points must match the number of measured corners.
    raise NotImplementedError("Build 3D checkerboard object points.")


def split_calibration_matches(
    object_points: torch.Tensor,
    pixels: torch.Tensor,
    config: CameraGeometryConfig,
) -> CalibrationSplits:
    """Split calibration matches into train and validation sets."""
    # Goal:
    # Keep some corner matches held out for validation.
    #
    # Method:
    # 1. Check object points and pixels have the same first dimension.
    # 2. Build shuffled indices with `torch.Generator().manual_seed(config.random_state)`.
    # 3. Use `config.train_fraction` for the train count.
    # 4. Return train and validation tensors.
    #
    # Check:
    # - Train and validation rows must not overlap.
    raise NotImplementedError("Split calibration matches.")


class PinholeCameraModel(nn.Module):
    """Small differentiable pinhole camera model."""

    def __init__(self, config: CameraGeometryConfig) -> None:
        # Where this fits:
        # These are the camera parameters optimized from corner matches.
        #
        # Goal:
        # Create trainable parameters for a simple camera model.
        #
        # Method:
        # 1. Call `super().__init__()`.
        # 2. Create `nn.Parameter`s for:
        #    - `log_fx`
        #    - `log_fy`
        #    - `cx`
        #    - `cy`
        #    - `rotation_vector` shaped `[3]`
        #    - `translation` shaped `[3]`
        #
        # Suggested initial values:
        # - `fx` and `fy`: about `600.0`, stored as `torch.log(torch.tensor(600.0))`
        # - `cx`: `config.image_width / 2`
        # - `cy`: `config.image_height / 2`
        # - rotation vector: zeros
        # - translation: `[0.0, 0.0, 1.0]`, so the board starts in front of the camera
        super().__init__()
        raise NotImplementedError("Build the differentiable pinhole camera model.")

    def rotation_matrix(self) -> torch.Tensor:
        """Convert the small rotation vector into a rotation matrix."""
        # Goal:
        # Return a `[3, 3]` rotation matrix.
        #
        # Method:
        # For this beginner exercise, use the small-angle approximation:
        #
        # ```text
        # R = I + skew(rotation_vector)
        # ```
        #
        # Steps:
        # 1. Read `rx`, `ry`, and `rz`.
        # 2. Build the skew-symmetric matrix:
        #    `[[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]`.
        # 3. Return `torch.eye(3) + skew`.
        #
        # Note:
        # This is a teaching simplification. Full calibration would use a proper
        # Rodrigues rotation formula.
        raise NotImplementedError("Build camera rotation matrix.")

    def forward(self, object_points: torch.Tensor) -> torch.Tensor:
        """Project 3D object points into pixel coordinates."""
        # Input:
        # - `object_points`: `[num_points, 3]`.
        #
        # Output:
        # - projected pixels: `[num_points, 2]`.
        #
        # Method:
        # 1. Compute `fx = torch.exp(self.log_fx)` and same for `fy`.
        # 2. Get `R = self.rotation_matrix()`.
        # 3. Move points to camera coordinates:
        #    `points_camera = object_points @ R.T + self.translation`.
        # 4. Read `x`, `y`, `z`.
        # 5. Clamp z with `z.clamp_min(1e-6)`.
        # 6. Compute:
        #    `u = fx * x / z + self.cx`
        #    `v = fy * y / z + self.cy`
        # 7. Return `torch.stack([u, v], dim=1)`.
        raise NotImplementedError("Project object points into pixels.")


def reprojection_error_pixels(projected_pixels: torch.Tensor, observed_pixels: torch.Tensor) -> torch.Tensor:
    """Return one reprojection error per point in pixels."""
    # Formula:
    # `error_i = sqrt((u_pred - u_obs)^2 + (v_pred - v_obs)^2)`.
    #
    # Method:
    # Use `torch.linalg.norm(projected_pixels - observed_pixels, dim=1)`.
    #
    # Output:
    # - tensor shaped `[num_points]`.
    raise NotImplementedError("Compute per-point reprojection errors.")


def fit_camera_model(
    splits: CalibrationSplits,
    config: CameraGeometryConfig,
) -> tuple[PinholeCameraModel, list[float]]:
    """Fit camera parameters by minimizing reprojection loss."""
    # Method:
    # 1. Call `set_seed(config.random_state)`.
    # 2. Build `model = PinholeCameraModel(config)`.
    # 3. Build `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 4. For `config.steps`:
    #    - project train object points
    #    - compute MSE against train pixels
    #    - clear gradients
    #    - backpropagate
    #    - step optimizer
    #    - append loss to a list
    # 5. Return the model and loss history.
    _ = set_seed
    raise NotImplementedError("Fit camera model from calibration matches.")


def evaluate_camera_model(model: PinholeCameraModel, splits: CalibrationSplits) -> dict[str, float]:
    """Return train and validation reprojection errors."""
    # Method:
    # 1. Use `torch.no_grad()`.
    # 2. Project train and validation object points.
    # 3. Compute per-point pixel errors.
    # 4. Return:
    #    - `train_reprojection_error_px`
    #    - `val_reprojection_error_px`
    #
    # Check:
    # - These are pixel distances, not squared losses.
    raise NotImplementedError("Evaluate camera reprojection error.")


def save_reprojection_overlay(
    model: PinholeCameraModel,
    splits: CalibrationSplits,
    config: CameraGeometryConfig,
) -> str:
    """Save observed vs projected pixel locations."""
    # Goal:
    # Save `outputs/camera_geometry/reprojection_overlay.png`.
    #
    # Method:
    # 1. Load or display the calibration image if available.
    # 2. Plot observed pixels and projected pixels.
    # 3. Use different markers and a legend.
    # 4. Save and close the figure.
    raise NotImplementedError("Save reprojection overlay plot.")


def save_error_histogram(errors: torch.Tensor, config: CameraGeometryConfig) -> str:
    """Save a histogram of reprojection errors."""
    # Goal:
    # Save `outputs/camera_geometry/reprojection_error_histogram.png`.
    #
    # Method:
    # Plot per-corner pixel errors with Matplotlib.
    raise NotImplementedError("Save reprojection error histogram.")


def save_loss_curve(losses: list[float], config: CameraGeometryConfig) -> str:
    """Save camera fitting loss curve."""
    # Goal:
    # Save `outputs/camera_geometry/calibration_loss_curve.png`.
    #
    # Method:
    # Plot optimization step on the x-axis and MSE loss on the y-axis.
    raise NotImplementedError("Save calibration loss curve.")


def run_camera_calibration_mini_project(config: CameraGeometryConfig) -> CameraCalibrationMetrics:
    """Run the full camera-calibration mini-project."""
    # Method:
    # 1. Download or confirm the real calibration image.
    # 2. Load observed checkerboard corners from CSV.
    # 3. Build known object points.
    # 4. Split matches into train and validation.
    # 5. Fit the camera model.
    # 6. Evaluate train and validation reprojection error.
    # 7. Save overlay, histogram, and loss-curve plots.
    # 8. Return a `CameraCalibrationMetrics` dictionary.
    #
    # Checks:
    # - Use more than one corner match.
    # - Return paths for every visual.
    raise NotImplementedError("Run camera calibration mini-project.")


def explain_camera_parts() -> str:
    """Return a short plain-English explanation."""
    return (
        "Intrinsics describe the camera itself, such as focal length and image "
        "center. Extrinsics describe where the calibration board is relative to "
        "the camera. Reprojection error measures how far projected points land "
        "from the measured image corners."
    )


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the camera geometry scaffold."""
    config = CameraGeometryConfig()
    metrics, ok = report_task_progress("run_camera_calibration_mini_project", run_camera_calibration_mini_project, config)
    if ok:
        print(metrics)
        print(explain_camera_parts())


if __name__ == "__main__":
    main()
