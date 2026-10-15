"""
============================================================
EXERCISE 01: TINY STRUCTURE FROM MOTION
============================================================

WHAT YOU ARE BUILDING
You are building a tiny Structure from Motion pipeline on simple point data.

Plain-language meanings:
- structure means the 3D points in the scene
- motion means how the camera moved between views
- SfM means recovering both from several 2D views
- reprojection checks measure whether the recovered 3D points explain the 2D image points

WHY THIS EXERCISE EXISTS
SfM is one path from camera images to a 3D map.
That map is useful for navigation and embodied agents.

This should be a complete small mapping workflow, not only a shape demo.
The data is a set of matched 2D point tracks across camera views:

```text
same scene point in view 1, view 2, view 3 -> recovered 3D point
```

DATASET OPTIONS
Default: use a small local rendered multi-view point set.

Real dataset option after the default works:
- KITTI Odometry: image sequences, camera calibration, and ground-truth camera poses.

MAIN LOOP IN PLAIN WORDS

```text
collect matched 2D points from several camera views
use known or estimated camera poses
triangulate 3D points
project the 3D points back into each view
measure reprojection error
```

DEFINITION OF DONE
You are done when:
- several camera views are represented
- a small 3D point cloud is recovered
- reprojection error is measured
- the same point order is checked across all views
- you can explain why more views help
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict
import torch


class CameraView(TypedDict):
    """One camera view with pose and matched 2D points."""

    points_2d: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor


@dataclass
class SfMConfig:
    """Settings for the tiny SfM exercise."""

    num_points: int = 8
    num_views: int = 3
    focal_length_pixels: float = 100.0
    center_x: float = 64.0
    center_y: float = 64.0


def build_tiny_views(config: SfMConfig) -> tuple[torch.Tensor, list[CameraView]]:
    """Return intrinsics and several tiny camera views."""
    # Goal:
    # Create a small multi-view point dataset.
    #
    # Input:
    # - `config.num_points`: number of matched scene points.
    # - `config.num_views`: number of camera views.
    #
    # Output:
    # - `intrinsics`: tensor shaped `[3, 3]`
    # - `views`: list of CameraView dictionaries.
    # - Each view has:
    #   - `points_2d`: `[num_points, 2]`
    #   - `rotation`: `[3, 3]`
    #   - `translation`: `[3]`
    #
    # Method:
    # Use simple hand-written camera poses and shifted 2D points.
    #
    # Steps:
    # 1. Build intrinsics.
    # 2. Build at least 3 camera views.
    # 3. Keep point order the same in every view.
    # 4. Return the intrinsics and view list.
    #
    # Checks:
    # - Every view should contain the same number of 2D points.
    raise NotImplementedError("Build tiny SfM views.")


def triangulate_point_cloud(intrinsics: torch.Tensor, views: list[CameraView]) -> torch.Tensor:
    """Recover a small 3D point cloud from several views."""
    # Goal:
    # Build a tiny point cloud from multiple 2D views.
    #
    # Input:
    # - `intrinsics`: `[3, 3]`
    # - `views`: list from `build_tiny_views`
    #
    # Output:
    # - tensor shaped `[num_points, 3]`
    #
    # Method:
    # Simple first version:
    # - triangulate from the first two views with the same stereo-style formula as the related exercise
    # - later, improve it by averaging estimates from more view pairs
    #
    # Steps:
    # 1. Read the first two views.
    # 2. Compute disparity from their x coordinates.
    # 3. Compute depth from focal length and camera baseline.
    # 4. Back-project first-view pixels into 3D points.
    # 5. Return the point cloud.
    #
    # Checks:
    # - The result should have 3 columns.
    # - The number of points should match each view's number of 2D points.
    raise NotImplementedError("Triangulate a tiny 3D point cloud.")


def compute_reprojection_errors(
    point_cloud: torch.Tensor,
    intrinsics: torch.Tensor,
    views: list[CameraView],
) -> dict[str, float]:
    """Measure how well the point cloud explains the views."""
    # Goal:
    # Project the recovered 3D points back into each camera view and measure error.
    #
    # Input:
    # - `point_cloud`: `[num_points, 3]`
    # - `intrinsics`: `[3, 3]`
    # - `views`: camera view list
    #
    # Output:
    # - dictionary with:
    #   - `"mean_reprojection_error"`
    #   - `"max_reprojection_error"`
    #
    # Method:
    # For each view:
    # 1. Move points into camera coordinates.
    # 2. Project them into pixels.
    # 3. Compare with `view["points_2d"]`.
    # 4. Store pixel distances.
    #
    # Checks:
    # - Errors should be finite numbers.
    # - Mean error should be lower when the point cloud and poses are more consistent.
    raise NotImplementedError("Compute multi-view reprojection errors.")


def explain_more_views() -> str:
    """Return a short explanation of why more views help."""
    return "More views give more checks on the same 3D point, so mistakes are easier to notice and reduce."


def run_sfm_mini_project(config: SfMConfig) -> dict[str, float]:
    """Build a tiny point cloud and score it with reprojection error."""
    # Goal:
    # Run the full Structure from Motion workflow on a small multi-view dataset.
    #
    # Output:
    # - dictionary with:
    #   - `"mean_reprojection_error"`
    #   - `"max_reprojection_error"`
    #   - `"num_views"`
    #   - `"num_points"`
    #
    # Method:
    # 1. Build or load the camera views.
    # 2. Triangulate a point cloud.
    # 3. Reproject the point cloud into every view.
    # 4. Report mean and max reprojection error.
    #
    # Checks:
    # - Do not shuffle point order between views.
    # - More consistent views should give lower reprojection error.
    raise NotImplementedError("Run the tiny SfM mini-project.")


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
    """Run the full tiny SfM scaffold."""
    config = SfMConfig()
    metrics, ok = report_task_progress("run_sfm_mini_project", run_sfm_mini_project, config)
    if ok:
        print(metrics)


if __name__ == "__main__":
    main()
