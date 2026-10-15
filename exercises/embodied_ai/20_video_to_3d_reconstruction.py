"""
============================================================
EXERCISE 20: VIDEO TO 3D RECONSTRUCTION
============================================================

WHAT YOU ARE BUILDING
You are building a small 3D reconstruction system from a real video.

The job challenge asks for:

```text
short video of a small indoor area -> coherent 3D scene reconstruction
```

Plain meaning:
- the video is a real `.mp4`, not a made-up tensor scene
- frames are real images decoded from the video
- matched points are the same visual spots found in two frames
- pose means how the camera moved between those two frames
- triangulation means turning two matched 2D points into one 3D point
- reprojection error checks whether the 3D point lands back near the matched
  pixels when projected into the frames
- the output is a sparse colored point cloud saved as a `.ply` file
- the final folder includes an HTML viewer you can open in a browser

This first version reconstructs a sparse point cloud. Sparse means it recovers
points at matched image features, not every pixel.

REAL SYSTEM VS THIS EXERCISE

real applicant system                                      this exercise
---------------------------------------------------------------------------
phone video of a small room                                real local `.mp4` frames
camera intrinsics from calibration or phone metadata        estimated pinhole intrinsics
feature tracks across many frames                          ORB matches between frame pairs
camera pose from SfM or SLAM                                OpenCV essential-matrix pose
3D points from multi-view triangulation                     two-view triangulated sparse points
dense mesh or dense point cloud                             sparse colored PLY point cloud
semantic labels in 3D                                      optional extension after geometry works
polished submission                                        saved PLY, plots, run notes, design note
easy result inspection                                     visual index page + HTML point viewer

WHY THIS EXERCISE USES REAL VIDEO
The applicant task is about reconstruction from real video, so the default
exercise should touch real video frames.

The repo already has local real MP4 files under:

```text
data/coin/videos/
```

This exercise uses one of those videos by default if you do not provide your
own phone video. For the strongest portfolio version, record a short phone
video of a small room and set:

```python
config.video_path = PROJECT_ROOT / "data" / "video_to_3d" / "room.mp4"
```

WHAT IS INCLUDED AND WHAT IS SKIPPED
Included:
- choose a real `.mp4` input
- decode real frames with PyAV
- resize frames to a small working size
- estimate simple camera intrinsics
- detect and match image features with OpenCV ORB
- estimate relative camera pose with an essential matrix
- triangulate sparse 3D points
- color 3D points from the source frame
- save a `.ply` point cloud
- save a match preview and a top-down point plot
- save an interactive 3D point cloud viewer
- save a visual index page that says what to open first
- report reprojection error
- write run instructions and a short design note

Skipped in the first pass:
- full COLMAP-style global reconstruction
- bundle adjustment
- loop closure
- dense depth estimation
- dense meshing
- real semantic segmentation

Those are important later. This exercise teaches the first working real-video
geometry path.

FULL WORKFLOW

```text
1. choose a real video
2. decode a few frames
3. estimate a camera intrinsics matrix
4. match visual features between two frames
5. estimate how the camera moved
6. triangulate matched points into 3D
7. remove points that are clearly invalid
8. color the 3D points from the image
9. save a PLY point cloud
10. check reprojection error
11. save a static point cloud plot
12. save an interactive browser viewer
13. write a visual index page
14. save notes for a small submission
```

THIS EXERCISE COVERS
This exercise covers a small real-data SfM pipeline:

```text
real frames -> feature matches -> camera motion -> 3D points -> point cloud
```

It does not use synthetic frames, synthetic depth, or known toy poses.

INPUTS AND OUTPUTS
RGB frame:
- shape: `[3, image_height, image_width]`
- values: floats from `0.0` to `1.0`

Matched pixels:
- frame 1 points shape: `[num_matches, 2]`
- frame 2 points shape: `[num_matches, 2]`
- each row is `[u, v]`

Camera intrinsics:
- shape: `[3, 3]`
- contains `fx`, `fy`, `cx`, and `cy`

Relative pose:
- rotation shape: `[3, 3]`
- translation shape: `[3]`
- maps frame 1 camera coordinates into frame 2 camera coordinates

Point cloud:
- points shape: `[num_points, 3]`
- colors shape: `[num_points, 3]`
- source pixels shape: `[num_points, 2]`

CORE IDEA
One camera view gives a ray, not a full 3D point.

Plain meaning:

```text
one frame:  pixel says direction, but not distance
two frames: matching rays can meet in 3D
```

The camera movement gives the baseline. Baseline means the distance between
the two camera positions. Without enough camera movement, triangulation is weak.

METRICS TO TRACK
- `num_frames`: real frames decoded from the video
- `num_raw_matches`: feature matches before filtering
- `num_inlier_matches`: matches that agree with the estimated camera motion
- `num_points_3d`: valid 3D points after triangulation
- `mean_reprojection_error_px`: lower is better
- `median_depth_m`: rough scale in the reconstructed coordinate system
- `point_cloud_viewer_path`: browser file for rotating the 3D points
- `visual_index_path`: first file to open when checking the result

COMMON MISTAKES
- using two frames that barely changed
- accepting all feature matches without filtering bad matches
- forgetting that monocular video has unknown absolute scale
- treating sparse SfM as a dense room mesh
- reporting a PLY file without reporting reprojection error
- using semantic labels before the geometry is working

STEP-BY-STEP TODO
1. Find a real video file.
2. Decode real frames from the video.
3. Estimate a simple intrinsics matrix.
4. Detect and match ORB features between two frames.
5. Estimate relative pose with `cv2.findEssentialMat(...)` and `cv2.recoverPose(...)`.
6. Triangulate matched points with `cv2.triangulatePoints(...)`.
7. Filter invalid 3D points.
8. Build a colored sparse point cloud.
9. Save the point cloud as a `.ply` file.
10. Compute reprojection error.
11. Save match and point-cloud plots.
12. Save an interactive point-cloud viewer.
13. Write a visual index page.
14. Write run instructions and a design note.

DEFINITION OF DONE
You are done when:
- the input comes from a real `.mp4`
- at least two real frames are used
- matched features are filtered before triangulation
- a sparse 3D point cloud is saved as `.ply`
- reprojection error is reported
- example input and output plots are saved
- `VISUAL_INDEX.html` is saved and tells you what to open first
- `point_cloud_viewer.html` opens in a browser as an interactive 3D view
- run instructions and a design note are saved
- you can explain why two views are needed for 3D
- every completed stub keeps its original `raise NotImplementedError(...)` line commented out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    import numpy as np

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Learning order for this exercise:
# 1. Decode real frames from a video.
# 2. Match feature points between two frames.
# 3. Estimate camera motion from those matches.
# 4. Triangulate sparse 3D points.
# 5. Save the point cloud, visual previews, and reprojection error.


class VideoFrames(TypedDict):
    """Real video frames loaded for reconstruction."""

    video_path: Path
    frames: list[torch.Tensor]


class MatchedFeatures(TypedDict):
    """Matched 2D points between two real frames."""

    frame1_points: torch.Tensor
    frame2_points: torch.Tensor
    match_preview_path: str
    num_raw_matches: int


class RelativePose(TypedDict):
    """Camera 2 pose relative to camera 1."""

    rotation: torch.Tensor
    translation: torch.Tensor
    inlier_mask: torch.Tensor


class SparsePointCloud(TypedDict):
    """A sparse colored 3D point cloud."""

    points: torch.Tensor
    colors: torch.Tensor
    source_pixels: torch.Tensor


class ReconstructionMetrics(TypedDict):
    """Output metrics and saved artifact paths."""

    video_path: str
    num_frames: int
    num_raw_matches: int
    num_inlier_matches: int
    num_points_3d: int
    mean_reprojection_error_px: float
    median_depth_m: float
    ply_path: str
    match_preview_path: str
    top_down_plot_path: str
    point_cloud_viewer_path: str
    visual_index_path: str
    run_instructions_path: str
    design_note_path: str


@dataclass
class Video3DReconstructionConfig:
    """Settings for real-video 3D reconstruction."""

    video_path: Path | None = None
    fallback_video_dir: Path = PROJECT_ROOT / "data" / "coin" / "videos"
    output_dir: Path = PROJECT_ROOT / "outputs" / "video_to_3d_reconstruction"
    image_height: int = 240
    image_width: int = 320
    num_frames: int = 5
    frame_stride: int = 15
    frame_pair: tuple[int, int] = (0, 2)
    focal_length_scale: float = 1.2
    max_features: int = 2000
    lowe_ratio: float = 0.75
    ransac_threshold_px: float = 1.5
    min_inlier_matches: int = 30
    min_points_3d: int = 25
    max_reprojection_error_px: float = 5.0
    max_depth_m: float = 50.0
    max_preview_points: int = 50_000


def find_default_real_video(config: Video3DReconstructionConfig) -> Path:
    """Return a real video path to use for the exercise."""
    # Where this fits:
    # This replaces the old synthetic-data default.
    #
    # Goal:
    # Choose a real `.mp4` file.
    #
    # Input:
    # - `config.video_path`: optional user-provided video
    # - `config.fallback_video_dir`: local folder of real videos
    #
    # Output:
    # - a `Path` pointing to a real video file
    #
    # Method:
    # 1. If `config.video_path` is not `None`, check that it exists and return it.
    # 2. Search `config.fallback_video_dir` for `*.mp4` files with
    #    `sorted(config.fallback_video_dir.glob("*.mp4"))`.
    # 3. Return the first file.
    # 4. If no file exists, raise `FileNotFoundError` with a message that says
    #    to add a phone video at `data/video_to_3d/room.mp4` and set
    #    `config.video_path`.
    #
    # Checks:
    # - Do not create fake frames here.
    # - The returned path must point to a real file.
    raise NotImplementedError("Find a real input video.")


def decode_video_frames(config: Video3DReconstructionConfig) -> VideoFrames:
    """Decode real RGB frames from a video file."""
    # Where this fits:
    # This is the real-data input step.
    #
    # Goal:
    # Return real frames loaded from an `.mp4`.
    #
    # Output:
    # - `"video_path"`: path used
    # - `"frames"`: list of tensors shaped `[3, image_height, image_width]`
    #
    # Method:
    # Use PyAV, imported as `av`.
    #
    # Exact call shape:
    # - `container = av.open(str(video_path))`
    # - `for frame in container.decode(video=0): ...`
    # - `array = frame.to_rgb().to_ndarray()`
    #
    # Resize method:
    # Use `torch.nn.functional.interpolate(...)`.
    # Call shape:
    # `F.interpolate(image_batch, size=(config.image_height, config.image_width), mode="bilinear", align_corners=False)`.
    #
    # Steps:
    # 1. Get `video_path = find_default_real_video(config)`.
    # 2. Decode video frames with PyAV.
    # 3. Keep every `config.frame_stride` frame.
    # 4. Convert each RGB array to a tensor with shape `[3, H, W]`.
    # 5. Divide by `255.0` so values are from `0.0` to `1.0`.
    # 6. Resize to `(config.image_height, config.image_width)`.
    # 7. Stop after `config.num_frames`.
    # 8. Return the dictionary.
    #
    # Checks:
    # - Return at least 2 frames.
    # - These frames must come from the video, not from random tensors.
    raise NotImplementedError("Decode real frames from video.")


def build_intrinsics(config: Video3DReconstructionConfig) -> torch.Tensor:
    """Build a simple intrinsics matrix for the resized frames."""
    # Where this fits:
    # Essential-matrix pose estimation and triangulation both need intrinsics.
    #
    # Real system:
    # Use phone calibration, EXIF metadata, ARKit/ARCore data, or a calibration
    # board.
    #
    # Exercise version:
    # Estimate focal length from image size. This is not perfect, but it keeps
    # the real-video pipeline moving.
    #
    # Output:
    # - tensor shaped `[3, 3]`
    #
    # Method:
    # 1. Set `fx = config.focal_length_scale * config.image_width`.
    # 2. Set `fy = fx`.
    # 3. Set `cx = (config.image_width - 1) / 2`.
    # 4. Set `cy = (config.image_height - 1) / 2`.
    # 5. Return:
    #
    # ```python
    # torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    # ```
    #
    # Check:
    # - The principal point should be near the image center.
    raise NotImplementedError("Build camera intrinsics matrix.")


def tensor_frame_to_uint8_numpy(frame: torch.Tensor) -> "np.ndarray":
    """Convert one `[3, H, W]` tensor frame to a uint8 NumPy image."""
    # Where this fits:
    # OpenCV feature detection expects NumPy images, not PyTorch tensors.
    #
    # Input:
    # - `frame`: tensor shaped `[3, H, W]`, values from `0.0` to `1.0`
    #
    # Output:
    # - NumPy array shaped `[H, W, 3]`
    # - dtype `uint8`
    #
    # Method:
    # 1. Clamp the tensor to `[0.0, 1.0]`.
    # 2. Move channels last with `frame.permute(1, 2, 0)`.
    # 3. Multiply by `255`.
    # 4. Convert to `torch.uint8`.
    # 5. Move to CPU and call `.numpy()`.
    raise NotImplementedError("Convert tensor frame to uint8 NumPy image.")


def match_features_between_frames(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    config: Video3DReconstructionConfig,
) -> MatchedFeatures:
    """Detect and match ORB features between two real frames."""
    # Where this fits:
    # This finds the same visual points in two frames.
    #
    # Goal:
    # Return matched pixel coordinates for frame 1 and frame 2.
    #
    # Method:
    # Use OpenCV, imported as `cv2`.
    #
    # Exact OpenCV calls:
    # - `orb = cv2.ORB_create(nfeatures=config.max_features)`
    # - `keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)`
    # - `keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)`
    # - `matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)`
    # - `knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)`
    #
    # Lowe ratio test:
    # For each pair `(best, second_best)`, keep `best` only when:
    #
    # ```text
    # best.distance < config.lowe_ratio * second_best.distance
    # ```
    #
    # This removes many weak matches.
    #
    # Output:
    # - `"frame1_points"`: `[num_matches, 2]`
    # - `"frame2_points"`: `[num_matches, 2]`
    # - `"match_preview_path"`: path to a saved match preview image
    # - `"num_raw_matches"`: number before the ratio test
    #
    # Steps:
    # 1. Convert both tensor frames to NumPy RGB images.
    # 2. Convert RGB to grayscale with `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`.
    # 3. Detect ORB keypoints and descriptors.
    # 4. Match descriptors with BFMatcher and `knnMatch`.
    # 5. Apply Lowe's ratio test.
    # 6. Convert kept keypoint locations to float tensors.
    # 7. Save a match preview with `cv2.drawMatches(...)`.
    #
    # Checks:
    # - If descriptors are missing, raise `ValueError` with a clear message.
    # - More matches usually gives a more stable pose estimate.
    raise NotImplementedError("Match ORB features between real frames.")


def estimate_relative_pose(
    matches: MatchedFeatures,
    intrinsics: torch.Tensor,
    config: Video3DReconstructionConfig,
) -> RelativePose:
    """Estimate camera motion between two frames."""
    # Where this fits:
    # Camera motion is needed so two matching rays can triangulate a 3D point.
    #
    # Goal:
    # Estimate rotation and translation from matched points.
    #
    # Method:
    # Use OpenCV:
    # - `cv2.findEssentialMat(...)`
    # - `cv2.recoverPose(...)`
    #
    # Exact call shape:
    # ```python
    # essential, mask = cv2.findEssentialMat(
    #     points1,
    #     points2,
    #     cameraMatrix=K,
    #     method=cv2.RANSAC,
    #     prob=0.999,
    #     threshold=config.ransac_threshold_px,
    # )
    # _, rotation, translation, pose_mask = cv2.recoverPose(essential, points1, points2, K, mask=mask)
    # ```
    #
    # Input:
    # - `points1`: NumPy array shaped `[num_matches, 2]`
    # - `points2`: NumPy array shaped `[num_matches, 2]`
    # - `K`: NumPy intrinsics matrix shaped `[3, 3]`
    #
    # Output:
    # - `"rotation"`: `[3, 3]`
    # - `"translation"`: `[3]`
    # - `"inlier_mask"`: boolean tensor shaped `[num_matches]`
    #
    # Important:
    # The translation scale from monocular video is arbitrary. Its direction is
    # useful, but the true meter scale is unknown unless you add calibration or a
    # known object size.
    #
    # Checks:
    # - Require at least `config.min_inlier_matches`.
    # - Raise `ValueError` if pose estimation fails.
    raise NotImplementedError("Estimate relative camera pose from matched points.")


def triangulate_sparse_points(
    matches: MatchedFeatures,
    pose: RelativePose,
    intrinsics: torch.Tensor,
    config: Video3DReconstructionConfig,
) -> SparsePointCloud:
    """Triangulate sparse 3D points from two real frames."""
    # Where this fits:
    # This is the step that creates 3D structure.
    #
    # Goal:
    # Turn inlier matches into 3D points.
    #
    # Method:
    # Use OpenCV `cv2.triangulatePoints(...)`.
    #
    # Projection matrices:
    # - frame 1: `P1 = K @ [I | 0]`
    # - frame 2: `P2 = K @ [R | t]`
    #
    # Exact call shape:
    # `points_h = cv2.triangulatePoints(P1, P2, points1.T, points2.T)`
    #
    # `points_h` uses homogeneous coordinates. Plain meaning:
    # it has 4 numbers per point, and you divide the first 3 by the 4th:
    #
    # ```python
    # points_3d = points_h[:3] / points_h[3:4]
    # ```
    #
    # Color:
    # This function only creates geometry and stores `source_pixels`.
    # `build_colored_point_cloud(...)` attaches colors from frame 1.
    #
    # Checks:
    # - Keep only points with positive depth in front of both cameras.
    # - Drop points with depth larger than `config.max_depth_m`.
    # - Require at least `config.min_points_3d`.
    raise NotImplementedError("Triangulate sparse 3D points from inlier matches.")


def build_colored_point_cloud(point_cloud: SparsePointCloud, frame1: torch.Tensor) -> SparsePointCloud:
    """Attach RGB colors from frame 1 to each 3D point."""
    # Where this fits:
    # Colored points make the output easier to inspect.
    #
    # Input:
    # - `point_cloud["points"]`: `[num_points, 3]`
    # - `point_cloud["source_pixels"]`: `[num_points, 2]`, from frame 1
    # - `frame1`: `[3, H, W]`
    #
    # Output:
    # - same point cloud with `"colors"` filled
    #
    # Method:
    # 1. Round source pixels to integer pixel indices.
    # 2. Clamp `u` to `[0, W - 1]` and `v` to `[0, H - 1]`.
    # 3. Gather colors with `frame1[:, v, u].T`.
    # 4. Return a new dictionary.
    #
    # Checks:
    # - Colors should have one row per 3D point.
    raise NotImplementedError("Attach colors to sparse point cloud.")


def project_points(
    points_3d: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project 3D points into one camera."""
    # Where this fits:
    # Reprojection checks whether the 3D points explain the original image
    # matches.
    #
    # Input:
    # - `points_3d`: `[num_points, 3]` in frame 1 coordinates
    # - `rotation`: `[3, 3]`
    # - `translation`: `[3]`
    # - `intrinsics`: `[3, 3]`
    #
    # For frame 1, use:
    # - `rotation = torch.eye(3)`
    # - `translation = torch.zeros(3)`
    #
    # For frame 2, use the estimated pose.
    #
    # Method:
    # 1. Move points into camera coordinates:
    #    `camera_points = points_3d @ rotation.T + translation`
    # 2. Read `x`, `y`, and `z`.
    # 3. Compute:
    #    `u = fx * x / z + cx`
    #    `v = fy * y / z + cy`
    # 4. Return `torch.stack([u, v], dim=1)`.
    #
    # Checks:
    # - Use positive depths only.
    raise NotImplementedError("Project 3D points into a camera frame.")


def compute_reprojection_error(
    point_cloud: SparsePointCloud,
    matches: MatchedFeatures,
    pose: RelativePose,
    intrinsics: torch.Tensor,
) -> float:
    """Compute mean reprojection error in pixels."""
    # Where this fits:
    # This is the main geometry quality check.
    #
    # Goal:
    # Project triangulated points back into both frames and compare them with
    # the inlier matched pixels.
    #
    # Method:
    # 1. Use `pose["inlier_mask"]` to select inlier `frame1_points` and
    #    `frame2_points`.
    # 2. If triangulation filtered points further, use `point_cloud["source_pixels"]`
    #    for frame 1 and keep matching frame 2 points in the same order.
    # 3. Project points into frame 1 with identity pose.
    # 4. Project points into frame 2 with estimated pose.
    # 5. Compute pixel distances with `torch.linalg.norm(..., dim=1)`.
    # 6. Return the mean error across both frames.
    #
    # Checks:
    # - Error is measured in pixels.
    # - Lower is better.
    raise NotImplementedError("Compute mean reprojection error.")


def save_point_cloud_ply(point_cloud: SparsePointCloud, output_path: Path) -> str:
    """Save a colored sparse point cloud as an ASCII PLY file."""
    # Where this fits:
    # The `.ply` file is the main 3D output for the challenge.
    #
    # PLY row format:
    # `x y z red green blue`
    #
    # Method:
    # 1. Create `output_path.parent`.
    # 2. Clamp colors to `[0, 1]`.
    # 3. Convert colors to integer values from `0` to `255`.
    # 4. Write a valid ASCII PLY header.
    # 5. Write one point row at a time.
    #
    # Checks:
    # - Header vertex count must equal the number of point rows.
    raise NotImplementedError("Save sparse point cloud as PLY.")


def save_top_down_plot(point_cloud: SparsePointCloud, output_path: Path) -> str:
    """Save a simple top-down plot of the sparse point cloud."""
    # Where this fits:
    # This is a quick visual check of the reconstruction shape.
    #
    # Method:
    # 1. Use Matplotlib with the non-GUI backend:
    #    `matplotlib.use("Agg")`.
    # 2. Plot `x = points[:, 0]` and `z = points[:, 2]`.
    # 3. Use point colors if available.
    # 4. Label axes.
    # 5. Save and close the figure.
    raise NotImplementedError("Save top-down sparse point plot.")


def save_point_cloud_viewer_html(
    point_cloud: SparsePointCloud,
    output_path: Path,
    config: Video3DReconstructionConfig,
) -> str:
    """Save an interactive HTML viewer for the sparse point cloud."""
    # Where this fits:
    # A `.ply` file is useful, but it usually needs a separate viewer program.
    # This HTML file opens in a browser, so you can quickly see whether the 3D
    # points look like a real reconstruction.
    #
    # Goal:
    # Save `point_cloud_viewer.html`.
    #
    # Input:
    # - `point_cloud["points"]`: tensor shaped `[num_points, 3]`
    # - `point_cloud["colors"]`: tensor shaped `[num_points, 3]`
    # - `output_path`: where to save the HTML file
    #
    # Output:
    # - return `str(output_path)`
    #
    # Method:
    # 1. Import Plotly:
    #    `import plotly.graph_objects as go`
    # 2. Move tensors to CPU with `.detach().cpu()`.
    # 3. If there are more than `config.max_preview_points`, keep evenly spaced
    #    points so the HTML file does not become too large.
    # 4. Split the point tensor into:
    #    - `x = points[:, 0]`
    #    - `y = points[:, 1]`
    #    - `z = points[:, 2]`
    # 5. Convert colors from floats in `[0, 1]` to strings like:
    #    `"rgb(120, 80, 200)"`
    # 6. Create the trace:
    #    `go.Scatter3d(x=x, y=y, z=z, mode="markers", marker={...})`
    # 7. Use a small marker size, such as `2`.
    # 8. Use:
    #    `fig.update_layout(scene={"aspectmode": "data"})`
    #    so the cloud is not stretched into a cube.
    # 9. Save with:
    #    `fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)`
    #
    # Checks:
    # - Open the HTML file in a browser.
    # - You should be able to rotate, zoom, and pan the point cloud.
    # - If the page is blank, print `point_cloud["points"].shape`.
    _ = (point_cloud, output_path, config)
    raise NotImplementedError("Save interactive point cloud viewer HTML.")


def write_visual_index(metrics: ReconstructionMetrics, output_path: Path) -> str:
    """Write a small HTML page that links to the visible outputs."""
    # Where this fits:
    # The exercise should end with a file you can open first.
    #
    # Goal:
    # Save `VISUAL_INDEX.html`.
    #
    # Input:
    # - `metrics`: the final metrics dictionary
    # - `output_path`: where to save the HTML page
    #
    # Output:
    # - return `str(output_path)`
    #
    # Method:
    # 1. Build a small HTML string.
    # 2. Put this sentence near the top:
    #    `Open point_cloud_viewer.html first.`
    # 3. Add links to:
    #    - `point_cloud_viewer.html`
    #    - `top_down_map.png`
    #    - the match preview image
    #    - `reconstruction.ply`
    #    - `RUN.md`
    #    - `DESIGN_NOTE.md`
    # 4. Add the key numbers:
    #    - frames used
    #    - raw matches
    #    - inlier matches
    #    - 3D points
    #    - reprojection error
    # 5. For links, use paths relative to `output_path.parent` when possible.
    # 6. Write the page with `output_path.write_text(html, encoding="utf-8")`.
    #
    # Checks:
    # - Opening `VISUAL_INDEX.html` should tell you exactly what to inspect.
    # - The page should link to the interactive viewer and the static plot.
    _ = (metrics, output_path)
    raise NotImplementedError("Write visual index HTML.")


def write_run_instructions(config: Video3DReconstructionConfig, output_path: Path) -> str:
    """Write clear run instructions."""
    # Where this fits:
    # The challenge asks for clear instructions on how to run the system.
    #
    # Include:
    # - the run command
    # - where the default real video comes from
    # - how to use your own phone video
    # - expected output files
    # - open `VISUAL_INDEX.html` first
    # - open `point_cloud_viewer.html` to rotate the sparse 3D points
    # - use `top_down_map.png` as the quick static check
    #
    # Suggested run command:
    #
    # ```text
    # uv run python exercises/embodied_ai/20_video_to_3d_reconstruction.py
    # ```
    #
    # Suggested own-video path:
    #
    # ```text
    # data/video_to_3d/room.mp4
    # ```
    _ = config
    raise NotImplementedError("Write run instructions markdown.")


def write_design_note(
    metrics: ReconstructionMetrics,
    config: Video3DReconstructionConfig,
    output_path: Path,
) -> str:
    """Write a short design note and tradeoff summary."""
    # Where this fits:
    # The design note explains what the simple system does and where it is weak.
    #
    # Include:
    # - this is real video, not synthetic data
    # - the method is sparse two-view SfM
    # - monocular scale is unknown
    # - reprojection error is the main geometry check
    # - what the visual viewer showed
    # - dense reconstruction would need COLMAP, multi-view stereo, or depth models
    # - semantic labels should be added only after geometry is stable
    #
    # Use the actual metric values from `metrics`.
    _ = config
    raise NotImplementedError("Write design note markdown.")


def check_reconstruction_quality(
    metrics: ReconstructionMetrics,
    config: Video3DReconstructionConfig,
) -> bool:
    """Return True when the sparse reconstruction passes basic checks."""
    return (
        metrics["num_frames"] >= 2
        and metrics["num_inlier_matches"] >= config.min_inlier_matches
        and metrics["num_points_3d"] >= config.min_points_3d
        and metrics["mean_reprojection_error_px"] <= config.max_reprojection_error_px
    )


def run_video_to_3d_reconstruction(
    config: Video3DReconstructionConfig,
) -> ReconstructionMetrics:
    """Run the real-video sparse reconstruction pipeline."""
    # Main workflow:
    # real video -> real frames -> feature matches -> pose -> sparse 3D points
    # -> PLY output -> visual viewer -> reprojection check.
    #
    # Method:
    # 1. Create `config.output_dir`.
    # 2. Decode frames with `decode_video_frames(config)`.
    # 3. Build intrinsics with `build_intrinsics(config)`.
    # 4. Choose two frames with `config.frame_pair`.
    # 5. Match features with `match_features_between_frames(...)`.
    # 6. Estimate pose with `estimate_relative_pose(...)`.
    # 7. Triangulate sparse points with `triangulate_sparse_points(...)`.
    # 8. Attach colors with `build_colored_point_cloud(...)`.
    # 9. Compute reprojection error.
    # 10. Save:
    #    - `reconstruction.ply`
    #    - `top_down_map.png`
    #    - `point_cloud_viewer.html`
    #    - `VISUAL_INDEX.html`
    #    - match preview image
    #    - `RUN.md`
    #    - `DESIGN_NOTE.md`
    # 11. Return a metrics dictionary with paths to all visible outputs.
    #
    # Checks:
    # - The video path must be real.
    # - Do not fall back to synthetic data.
    # - The final folder should contain `VISUAL_INDEX.html`.
    # - A beginner should be able to open the viewer without extra software.
    # - Print whether `check_reconstruction_quality(metrics, config)` passes.
    raise NotImplementedError("Run real-video 3D reconstruction pipeline.")


def explain_real_video_pipeline() -> str:
    """Return a short explanation of the method."""
    return (
        "This exercise uses real video frames. It matches visual points between "
        "two frames, estimates how the camera moved, triangulates sparse 3D "
        "points, and checks the result by projecting those points back into the "
        "frames."
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
    """Run the real-video reconstruction scaffold one step at a time."""
    config = Video3DReconstructionConfig()
    completed_steps = 0
    total_steps = 6

    def count_or_stop(ok: bool) -> bool:
        nonlocal completed_steps
        if ok:
            completed_steps += 1
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return ok

    video_path, ok = report_task_progress("find_default_real_video", find_default_real_video, config)
    if not count_or_stop(ok):
        return
    print(f"Using video: {video_path}")

    video_data, ok = report_task_progress("decode_video_frames", decode_video_frames, config)
    if not count_or_stop(ok):
        return

    intrinsics, ok = report_task_progress("build_intrinsics", build_intrinsics, config)
    if not count_or_stop(ok):
        return

    first_index, second_index = config.frame_pair
    frame1 = video_data["frames"][first_index]
    frame2 = video_data["frames"][second_index]

    matches, ok = report_task_progress("match_features_between_frames", match_features_between_frames, frame1, frame2, config)
    if not count_or_stop(ok):
        return

    pose, ok = report_task_progress("estimate_relative_pose", estimate_relative_pose, matches, intrinsics, config)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_video_to_3d_reconstruction", run_video_to_3d_reconstruction, config)
    if count_or_stop(ok):
        print(metrics)
        print(explain_real_video_pipeline())


if __name__ == "__main__":
    main()
