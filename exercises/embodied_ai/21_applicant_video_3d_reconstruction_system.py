"""
============================================================
EXERCISE 21: APPLICANT VIDEO 3D RECONSTRUCTION SYSTEM
============================================================

WHAT YOU ARE BUILDING
You are building a small codebase that is close to what you could submit for
the applicant challenge.

The challenge says:

```text
short video of a small indoor area -> geometrically coherent 3D scene
```

the related exercise teaches the smallest real-video version:

```text
two frames -> OpenCV feature matches -> sparse point cloud
```

This exercise is the next step:

```text
real video -> extracted frames -> COLMAP reconstruction -> point cloud
           -> visual viewer -> quality report -> 3D semantic labels
           -> submission folder
```

Plain meaning:
- use a real phone-style video, not synthetic data
- let a proven SfM tool estimate camera poses and 3D points
- save outputs a reviewer can inspect
- measure whether the geometry is coherent
- add semantic labels only after the geometry works

WHY THIS IS CLOSER TO A REAL APPLICANT SYSTEM
A serious submission should not be only one script that prints one metric.
It should have:

- a clear input folder
- a repeatable command
- cached intermediate files
- a point cloud output
- visual previews
- an interactive viewer you can open in a browser
- a geometry quality report
- a short design note
- a simple way to swap in a new video

This exercise uses COLMAP as the main reconstruction engine.
COLMAP is a widely used Structure from Motion and multi-view stereo tool.
Structure from Motion means:

```text
many images -> camera poses + sparse 3D points
```

Multi-view stereo means:

```text
known camera poses + many images -> denser 3D points
```

REAL SYSTEM VS THIS EXERCISE

real applicant system                                      this exercise
---------------------------------------------------------------------------
phone video of a room                                      real `.mp4` input
frame extraction with metadata                             PyAV frame extraction
camera model from phone or calibration                     COLMAP camera model estimate
SfM with feature tracks and bundle adjustment              COLMAP mapper
dense reconstruction                                       optional COLMAP stereo fusion
geometric quality checks                                   COLMAP stats + reprojection checks
semantic segmentation                                      image labels lifted to the same 3D points
submission package                                         saved output folder with notes

WHAT IS INCLUDED
- input video discovery
- frame extraction to an image folder
- COLMAP availability check
- COLMAP sparse reconstruction commands
- optional COLMAP dense reconstruction commands
- output artifact collection
- point cloud summary checks
- simple screenshot/contact-sheet outputs
- semantic labels in 3D, attached to the same COLMAP points
- final submission folder report

WHAT IS SKIPPED
- training a depth model
- writing a custom bundle-adjustment solver
- real-time performance
- perfect object segmentation
- a web app

Those can be later extensions. The goal here is a clean, usable, defensible
geometry system.

FULL WORKFLOW

```text
1. choose a real video
2. extract frames
3. run COLMAP feature extraction
4. run COLMAP feature matching
5. run COLMAP sparse mapper
6. convert COLMAP sparse output to text
7. optionally run dense stereo and fusion
8. collect the best point cloud
9. compute and save quality checks
10. segment frames or load label maps and lift labels to 3D
11. save a static point cloud preview image
12. save an interactive point cloud viewer HTML file
13. write a visual index page that links to the main outputs
14. write run instructions and design note
15. create a submission summary
```

RECOMMENDED PUBLIC DATA
Use your own phone video if you can. That is closest to the job advert.

If you want a public indoor sequence, use the TUM RGB-D dataset:

```text
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
```

Good starting sequence:

```text
fr3/long_office_household
```

Why this one is useful:
- it is indoors
- it has household and office structure
- it has camera motion through the scene
- it has RGB images, depth images, and ground-truth trajectory

The file is large, so this exercise does not download it automatically unless
you fill in the download helper.

For checking your COLMAP install on a known image collection, use COLMAP's
official dataset page:

```text
https://colmap.github.io/datasets.html
```

Those datasets are image collections, not phone videos, but they are useful for
debugging the COLMAP part of the system.

In the final report, say which input you actually used. A reviewer should know
whether the result came from your own phone video, the local repo videos, TUM
RGB-D, or a COLMAP sample image set.

RECOMMENDED PRETRAINED MODELS
Semantic labels:
- Start with `nvidia/segformer-b0-finetuned-ade-512-512`.
  It is small and good enough for a first indoor semantic pass.
- Upgrade to `facebook/mask2former-swin-small-ade-semantic` if you want a
  stronger segmentation model.

Depth, if you add a learned-depth fallback:
- Use `depth-anything/Depth-Anything-V2-Small-hf` first.
  It gives relative depth. Relative depth means the near/far shape is useful,
  but the meter scale is not guaranteed.

Important:
COLMAP geometry is still the main reconstruction path here. A depth model is a
fallback or extra output, not proof that the 3D geometry is correct by itself.

INPUTS AND OUTPUTS
Input:
- one real `.mp4` file

Extracted frames:
- folder of real images such as `frame_000000.jpg`

COLMAP workspace:
- `database.db`
- `images/`
- `sparse/`
- optional `dense/`
- `semantics/labels/`
- optional `depth/`
- `visuals/`
- `reports/`
- `logs/`

Main outputs:
- `reconstruction_sparse.ply`
- optional `reconstruction_dense.ply`
- `quality_report.json`
- `frame_contact_sheet.png`
- `sparse_point_cloud_preview.png`
- `semantic_point_cloud_preview.png`
- `point_cloud_viewer.html`
- `VISUAL_INDEX.html`
- `RUN.md`
- `DESIGN_NOTE.md`
- `semantic_point_cloud.ply`
- `semantic_alignment_report.json`

METRICS TO TRACK
- number of extracted frames
- number of registered frames
- number of sparse 3D points
- mean reprojection error in pixels
- whether dense reconstruction ran
- number of semantic-labeled 3D points
- semantic label coverage
- semantic disagreement rate across views
- whether the visual preview and viewer were created

COMMON MISTAKES
- trying dense reconstruction before sparse reconstruction works
- using a video with too much motion blur
- using frames that are almost identical
- moving the camera with no side-to-side motion
- submitting only a PLY file with no run instructions
- adding semantic labels that are not tied to the 3D geometry
- finishing with only a numeric report and no picture to inspect

DEFINITION OF DONE
You are done when:
- the system starts from a real `.mp4`
- frames are extracted into a clean workspace
- COLMAP sparse reconstruction is run or clearly reported as missing
- a point cloud is saved or a clear failure report explains why not
- the quality report includes registered frames, point count, and reprojection error
- `VISUAL_INDEX.html` is written and tells you what to open first
- `sparse_point_cloud_preview.png` shows a non-empty point cloud preview
- `point_cloud_viewer.html` opens in a browser as an interactive 3D point cloud
- run instructions and a design note are written
- semantic labels are projected onto the same 3D points
- `semantic_point_cloud_preview.png` is written when semantics are enabled
- semantic coverage and semantic disagreement are reported
- every completed stub keeps its original `raise NotImplementedError(...)` line commented out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import subprocess
import sys
from typing import TypedDict

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Learning order for this exercise:
# 1. Make a clean real-video workspace.
# 2. Extract frames from the input video.
# 3. Run COLMAP sparse reconstruction.
# 4. Collect point clouds and quality numbers.
# 5. Save visual previews so you can see the result.
# 6. Add semantic labels only after geometry works.


class FrameExtractionSummary(TypedDict):
    """Summary of extracted real frames."""

    video_path: str
    image_dir: str
    num_frames: int
    contact_sheet_path: str


class ColmapCommandResult(TypedDict):
    """One COLMAP command result."""

    command_name: str
    command: list[str]
    return_code: int
    stdout_path: str
    stderr_path: str


class ColmapSparseSummary(TypedDict):
    """Summary of COLMAP sparse reconstruction output."""

    sparse_model_dir: str
    sparse_text_dir: str
    registered_images: int
    sparse_points: int
    mean_reprojection_error_px: float
    sparse_ply_path: str


class DenseReconstructionSummary(TypedDict):
    """Summary of optional dense reconstruction output."""

    ran_dense: bool
    dense_workspace_dir: str
    dense_ply_path: str
    dense_points: int


class SemanticPointSummary(TypedDict):
    """Summary of 3D semantic labels."""

    enabled: bool
    semantic_ply_path: str
    labeled_points: int
    total_points: int
    label_coverage: float
    disagreement_rate: float
    label_names: list[str]
    alignment_report_path: str


class PointCloudPreviewSummary(TypedDict):
    """Summary of files used to inspect the reconstruction."""

    sparse_preview_path: str
    semantic_preview_path: str
    point_cloud_viewer_path: str
    visual_index_path: str
    preview_points: int


class ApplicantSystemReport(TypedDict):
    """Final applicant-style system report."""

    video_path: str
    output_dir: str
    num_frames: int
    registered_images: int
    sparse_points: int
    mean_reprojection_error_px: float
    dense_points: int
    semantic_labeled_points: int
    semantic_label_coverage: float
    semantic_disagreement_rate: float
    passed_basic_geometry_check: bool
    passed_semantic_alignment_check: bool
    sparse_preview_path: str
    semantic_preview_path: str
    point_cloud_viewer_path: str
    visual_index_path: str
    run_instructions_path: str
    design_note_path: str
    quality_report_path: str


@dataclass
class ApplicantReconstructionConfig:
    """Settings for the applicant-style reconstruction system."""

    video_path: Path | None = None
    fallback_video_dir: Path = PROJECT_ROOT / "data" / "coin" / "videos"
    output_dir: Path = PROJECT_ROOT / "outputs" / "applicant_video_3d_system"
    recommended_tum_rgbd_url: str = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download"
    recommended_tum_sequence: str = "fr3/long_office_household"
    colmap_dataset_url: str = "https://colmap.github.io/datasets.html"
    frame_stride: int = 12
    max_frames: int = 80
    image_width: int = 960
    image_height: int = 540
    colmap_binary: str = "colmap"
    camera_model: str = "SIMPLE_RADIAL"
    matcher: str = "sequential"
    run_dense: bool = False
    enable_semantics: bool = True
    use_pretrained_semantics: bool = True
    semantic_model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    stronger_semantic_model_id: str = "facebook/mask2former-swin-small-ade-semantic"
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    semantic_label_names: tuple[str, ...] = ("unknown", "floor", "wall", "table", "chair", "object")
    min_semantic_label_coverage: float = 0.30
    max_semantic_disagreement_rate: float = 0.35
    min_registered_images: int = 8
    min_sparse_points: int = 500
    max_mean_reprojection_error_px: float = 4.0
    max_preview_points: int = 50_000


def choose_input_video(config: ApplicantReconstructionConfig) -> Path:
    """Choose a real input video."""
    # Where this fits:
    # A real applicant system should start from a real video path.
    #
    # Goal:
    # Return a real `.mp4` path.
    #
    # Method:
    # 1. If `config.video_path` is set, check it exists and return it.
    # 2. Otherwise, search `config.fallback_video_dir` for `.mp4` files.
    # 3. Return the first sorted file.
    # 4. If no file exists, raise `FileNotFoundError`.
    #
    # Checks:
    # - Do not create synthetic data.
    # - The error message should tell the user to put a room video at
    #   `data/video_to_3d/room.mp4`.
    raise NotImplementedError("Choose real input video.")


def prepare_tum_rgbd_sequence(
    archive_path: Path,
    output_image_dir: Path,
    config: ApplicantReconstructionConfig,
) -> FrameExtractionSummary:
    """Prepare RGB frames from a downloaded TUM RGB-D sequence."""
    # Where this fits:
    # This is the recommended online-data path.
    #
    # The TUM RGB-D dataset page is:
    # `config.recommended_tum_rgbd_url`
    #
    # Recommended sequence:
    # `config.recommended_tum_sequence`, usually `fr3/long_office_household`.
    #
    # Goal:
    # Convert the downloaded TUM archive into the same image folder format used
    # by the rest of this exercise.
    #
    # Input:
    # - `archive_path`: path to a `.tgz` file the user downloaded
    # - `output_image_dir`: where to write `frame_000000.jpg` files
    #
    # Why this is not a direct auto-download:
    # TUM sequences are large. The user should choose the sequence and confirm
    # the download from the official page.
    #
    # Expected TUM files after extraction:
    # - `rgb.txt`: timestamps and RGB image paths
    # - `depth.txt`: timestamps and depth image paths
    # - `groundtruth.txt`: optional camera trajectory
    # - `rgb/`: folder of RGB images
    #
    # Method:
    # 1. Extract the `.tgz` with Python's standard `tarfile` module.
    # 2. Find `rgb.txt`.
    # 3. Read rows that do not start with `#`.
    # 4. Every row has:
    #    `timestamp relative_rgb_path`
    # 5. Keep every `config.frame_stride` row.
    # 6. Copy or resize RGB images into `output_image_dir`.
    # 7. Save names as `frame_000000.jpg`, `frame_000001.jpg`, and so on.
    # 8. Stop at `config.max_frames`.
    # 9. Save a contact sheet.
    #
    # Checks:
    # - Use real RGB images from the archive.
    # - Save at least `config.min_registered_images` frames when possible.
    _ = (archive_path, output_image_dir, config)
    raise NotImplementedError("Prepare frames from a downloaded TUM RGB-D sequence.")


def prepare_workspace(config: ApplicantReconstructionConfig) -> dict[str, Path]:
    """Create the output workspace folder layout."""
    # Where this fits:
    # A submission should have predictable folders.
    #
    # Goal:
    # Create and return paths for:
    # - `images`
    # - `colmap`
    # - `colmap/sparse`
    # - `colmap/sparse_text`
    # - `colmap/dense`
    # - `semantics`
    # - `semantics/labels`
    # - `depth`
    # - `visuals`
    # - `reports`
    # - `logs`
    #
    # Method:
    # 1. Use `config.output_dir.mkdir(parents=True, exist_ok=True)`.
    # 2. Create each child directory.
    # 3. Return a dictionary of `Path` objects.
    #
    # Checks:
    # - Do not delete an existing workspace in this helper.
    # - Existing files may be useful cached outputs.
    raise NotImplementedError("Prepare applicant-system workspace.")


def extract_real_video_frames(
    video_path: Path,
    image_dir: Path,
    config: ApplicantReconstructionConfig,
) -> FrameExtractionSummary:
    """Extract real video frames as image files."""
    # Where this fits:
    # COLMAP works on images, not directly on videos.
    #
    # Goal:
    # Save real frames as JPEG images.
    #
    # Method:
    # Use PyAV for decoding and OpenCV for writing.
    #
    # Exact call shapes:
    # - `container = av.open(str(video_path))`
    # - `for frame in container.decode(video=0):`
    # - `rgb = frame.to_rgb().to_ndarray()`
    # - `cv2.resize(rgb, (config.image_width, config.image_height), interpolation=cv2.INTER_AREA)`
    # - `cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))`
    #
    # Steps:
    # 1. Decode the video.
    # 2. Keep every `config.frame_stride` frame.
    # 3. Resize each kept frame.
    # 4. Save as `frame_000000.jpg`, `frame_000001.jpg`, and so on.
    # 5. Stop at `config.max_frames`.
    # 6. Save a contact sheet with `save_frame_contact_sheet(...)`.
    #
    # Checks:
    # - Save at least 2 frames.
    # - If too few frames are saved, raise `ValueError`.
    raise NotImplementedError("Extract real video frames for COLMAP.")


def save_frame_contact_sheet(image_dir: Path, output_path: Path, max_images: int = 12) -> str:
    """Save a contact sheet of extracted frames."""
    # Where this fits:
    # The reviewer should see what the input looked like.
    #
    # Goal:
    # Save a PNG preview of several extracted frames.
    #
    # Method:
    # 1. Read up to `max_images` JPEGs from `image_dir`.
    # 2. Use Matplotlib with `matplotlib.use("Agg")`.
    # 3. Plot frames in a grid.
    # 4. Save and close.
    raise NotImplementedError("Save extracted-frame contact sheet.")


def check_colmap_available(config: ApplicantReconstructionConfig) -> bool:
    """Return True if the COLMAP executable is available."""
    # Where this fits:
    # COLMAP is an external tool. A clean system should check for it before
    # running commands.
    #
    # Method:
    # Use `shutil.which(config.colmap_binary)`.
    #
    # Output:
    # - `True` if found
    # - `False` otherwise
    #
    # If this returns `False`, the pipeline should still write a clear report
    # explaining that COLMAP must be installed.
    return shutil.which(config.colmap_binary) is not None


def run_command_and_log(command_name: str, command: list[str], log_dir: Path) -> ColmapCommandResult:
    """Run one external command and save stdout and stderr."""
    # Where this fits:
    # A real applicant codebase should keep command logs for debugging.
    #
    # Goal:
    # Run a command and save:
    # - stdout to `<command_name>.out.txt`
    # - stderr to `<command_name>.err.txt`
    #
    # Method:
    # Use:
    # `subprocess.run(command, capture_output=True, text=True, check=False)`.
    #
    # Steps:
    # 1. Create `log_dir`.
    # 2. Run the command.
    # 3. Write stdout and stderr.
    # 4. Return a `ColmapCommandResult`.
    #
    # Checks:
    # - Do not use `shell=True`.
    # - If `return_code != 0`, the caller should raise a clear error.
    _ = subprocess
    raise NotImplementedError("Run external command and save logs.")


def run_colmap_sparse_reconstruction(
    image_dir: Path,
    workspace: dict[str, Path],
    config: ApplicantReconstructionConfig,
) -> list[ColmapCommandResult]:
    """Run COLMAP sparse reconstruction."""
    # Where this fits:
    # This is the main geometry engine for the applicant-style system.
    #
    # COLMAP commands to run:
    #
    # 1. Feature extraction:
    # ```text
    # colmap feature_extractor
    #   --database_path <database.db>
    #   --image_path <images>
    #   --ImageReader.camera_model SIMPLE_RADIAL
    # ```
    #
    # 2. Matching:
    # If `config.matcher == "sequential"`:
    # ```text
    # colmap sequential_matcher --database_path <database.db>
    # ```
    #
    # If `config.matcher == "exhaustive"`:
    # ```text
    # colmap exhaustive_matcher --database_path <database.db>
    # ```
    #
    # 3. Sparse mapping:
    # ```text
    # colmap mapper
    #   --database_path <database.db>
    #   --image_path <images>
    #   --output_path <sparse>
    # ```
    #
    # 4. Convert model to text:
    # ```text
    # colmap model_converter
    #   --input_path <sparse/0>
    #   --output_path <sparse_text>
    #   --output_type TXT
    # ```
    #
    # 5. Export sparse PLY:
    # ```text
    # colmap model_converter
    #   --input_path <sparse/0>
    #   --output_path <reconstruction_sparse.ply>
    #   --output_type PLY
    # ```
    #
    # Checks:
    # - Call `check_colmap_available(config)` before running.
    # - Raise a clear error if COLMAP is missing.
    # - Raise a clear error if no sparse model folder is created.
    # - Save all command logs under `workspace["logs"]`.
    raise NotImplementedError("Run COLMAP sparse reconstruction commands.")


def parse_colmap_sparse_text(sparse_text_dir: Path, sparse_ply_path: Path) -> ColmapSparseSummary:
    """Parse COLMAP text output and return sparse reconstruction stats."""
    # Where this fits:
    # The final report needs numbers, not only files.
    #
    # Files to read:
    # - `images.txt`
    # - `points3D.txt`
    #
    # `images.txt` plain meaning:
    # It lists registered images and their estimated camera poses.
    #
    # `points3D.txt` plain meaning:
    # It lists reconstructed 3D points and their reprojection error.
    #
    # Method:
    # 1. Count non-comment image header rows in `images.txt`.
    # 2. Count non-comment rows in `points3D.txt`.
    # 3. Read the final numeric field from each `points3D.txt` row as error.
    # 4. Compute mean reprojection error.
    #
    # Checks:
    # - If `points3D.txt` is missing, raise `FileNotFoundError`.
    # - If no points exist, return `0` points and infinite error.
    raise NotImplementedError("Parse COLMAP sparse text output.")


def run_colmap_dense_reconstruction(
    image_dir: Path,
    workspace: dict[str, Path],
    config: ApplicantReconstructionConfig,
) -> DenseReconstructionSummary:
    """Optionally run COLMAP dense reconstruction."""
    # Where this fits:
    # Dense output is closer to the applicant challenge, but it is slower and
    # more fragile. Run it only after sparse reconstruction works.
    #
    # If `config.run_dense` is `False`:
    # Return:
    # - `ran_dense=False`
    # - empty path strings
    # - `dense_points=0`
    #
    # If enabled, run:
    # 1. `image_undistorter`
    # 2. `patch_match_stereo`
    # 3. `stereo_fusion`
    #
    # Important:
    # COLMAP dense reconstruction can be slow. Keep `config.max_frames` modest.
    #
    # Checks:
    # - Dense output is useful for a stronger submission, but sparse
    #   reconstruction should pass first.
    _ = image_dir
    raise NotImplementedError("Optionally run COLMAP dense reconstruction.")


def predict_semantic_maps_with_pretrained_model(
    image_dir: Path,
    label_dir: Path,
    config: ApplicantReconstructionConfig,
) -> dict[str, Path]:
    """Predict 2D semantic maps with a pretrained segmentation model."""
    # Where this fits:
    # This is the recommended semantic-label path for a real applicant system.
    #
    # Default model:
    # `config.semantic_model_id`, usually:
    # `nvidia/segformer-b0-finetuned-ade-512-512`
    #
    # Stronger model to try later:
    # `config.stronger_semantic_model_id`, usually:
    # `facebook/mask2former-swin-small-ade-semantic`
    #
    # Goal:
    # Save one label-map PNG per extracted frame.
    #
    # Method with SegFormer:
    # Use Hugging Face Transformers:
    #
    # ```python
    # from PIL import Image
    # from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    #
    # processor = AutoImageProcessor.from_pretrained(config.semantic_model_id)
    # model = SegformerForSemanticSegmentation.from_pretrained(config.semantic_model_id)
    # inputs = processor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # logits = outputs.logits
    # ```
    #
    # `logits` shape:
    # - `[1, num_labels, low_height, low_width]`
    #
    # Steps:
    # 1. Load each extracted frame with `PIL.Image.open(...)`.
    # 2. Run the processor and model.
    # 3. Resize logits back to the frame size with
    #    `torch.nn.functional.interpolate(...)`.
    # 4. Use `argmax(dim=1)` to get one label id per pixel.
    # 5. Save the label map as an 8-bit PNG.
    # 6. Return a dictionary from image filename to label-map path.
    #
    # Important:
    # ADE20K labels are more detailed than this exercise's small label list.
    # Keep a mapping from ADE20K ids to your smaller labels. For example:
    # - floor-like labels -> `floor`
    # - wall-like labels -> `wall`
    # - table-like labels -> `table`
    # - chair-like labels -> `chair`
    # - everything else with a useful object class -> `object`
    #
    # Checks:
    # - Label-map size must match the frame size.
    # - Save one label map for every extracted frame used by COLMAP.
    _ = (image_dir, label_dir, config)
    raise NotImplementedError("Predict semantic label maps with a pretrained model.")


def predict_depth_maps_with_pretrained_model(
    image_dir: Path,
    depth_dir: Path,
    config: ApplicantReconstructionConfig,
) -> dict[str, Path]:
    """Predict relative depth maps with a pretrained depth model."""
    # Where this fits:
    # Depth prediction is not the main COLMAP path, but it can help with:
    # - visual inspection
    # - dense-looking preview images
    # - a fallback when COLMAP fails on low-texture video
    #
    # Recommended first model:
    # `config.depth_model_id`, usually:
    # `depth-anything/Depth-Anything-V2-Small-hf`
    #
    # Method:
    # Use Hugging Face Transformers:
    #
    # ```python
    # from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    #
    # processor = AutoImageProcessor.from_pretrained(config.depth_model_id)
    # model = AutoModelForDepthEstimation.from_pretrained(config.depth_model_id)
    # inputs = processor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # predicted_depth = outputs.predicted_depth
    # ```
    #
    # Important:
    # This is relative depth. It tells you what is nearer or farther, but not
    # exact meters unless the model is metric-depth fine-tuned and calibrated.
    #
    # Output:
    # - dictionary from image filename to saved depth-map path
    #
    # Checks:
    # - Do not replace COLMAP quality checks with depth-model output alone.
    # - Save a note in the design doc if you use relative depth.
    _ = (image_dir, depth_dir, config)
    raise NotImplementedError("Predict relative depth maps with a pretrained model.")


def build_or_load_frame_semantics(
    image_dir: Path,
    workspace: dict[str, Path],
    config: ApplicantReconstructionConfig,
) -> dict[str, Path]:
    """Create or load 2D semantic label maps for extracted frames."""
    # Where this fits:
    # The challenge says semantic labels are optional, but this exercise includes
    # that extension as a guided path. The labels must stay tied to the geometry.
    #
    # Goal:
    # Return one label-map PNG per extracted frame.
    #
    # Label map meaning:
    # - image shape: `[height, width]`
    # - pixel value `0`: unknown
    # - pixel value `1`: floor
    # - pixel value `2`: wall
    # - pixel value `3`: table
    # - pixel value `4`: chair
    # - pixel value `5`: object
    #
    # First accepted path:
    # If `config.use_pretrained_semantics` is `True`, call
    # `predict_semantic_maps_with_pretrained_model(...)`.
    #
    # Second accepted path:
    # Load precomputed label PNGs from:
    # `workspace["semantics"] / "labels" / frame_name`
    #
    # Fallback teaching path:
    # If no label PNGs exist, create simple starter masks from image layout:
    # - bottom part of the frame -> floor
    # - upper part of the frame -> wall
    # - middle colored or textured regions -> object
    #
    # This fallback is not a strong semantic model. It exists only so the 3D
    # semantic plumbing can be completed and tested when model weights are not
    # available.
    #
    # Output:
    # - dictionary mapping image filename, such as `frame_000000.jpg`, to the
    #   matching label-map path
    #
    # Method:
    # 1. Create `workspace["semantics"] / "labels"`.
    # 2. For each extracted frame, check whether a matching label PNG exists.
    # 3. If it exists, use it.
    # 4. If it does not exist, create the starter mask described above.
    # 5. Save label maps as 8-bit PNG files.
    # 6. Return the mapping.
    #
    # Checks:
    # - Label-map size must match the extracted frame size.
    # - Label ids must be valid indices into `config.semantic_label_names`.
    _ = (image_dir, workspace)
    raise NotImplementedError("Build or load 2D frame semantic label maps.")


def parse_colmap_image_keypoints(sparse_text_dir: Path) -> dict[int, dict[str, object]]:
    """Parse image names and keypoint pixel positions from COLMAP `images.txt`."""
    # Where this fits:
    # To lift 2D labels into 3D, we need to know which image pixel observed each
    # 3D point.
    #
    # COLMAP `images.txt` has two useful lines per registered image:
    #
    # ```text
    # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    # POINTS2D[] as X Y POINT3D_ID triples
    # ```
    #
    # Plain meaning:
    # - the first line names the image
    # - the second line lists keypoints in that image
    # - each keypoint may point to a reconstructed 3D point id
    #
    # Output:
    # A dictionary keyed by image id:
    #
    # ```python
    # {
    #     image_id: {
    #         "name": "frame_000010.jpg",
    #         "keypoints": [(x, y, point3d_id), ...],
    #     }
    # }
    # ```
    #
    # Method:
    # 1. Open `sparse_text_dir / "images.txt"`.
    # 2. Skip blank lines and comment lines starting with `#`.
    # 3. Read one image header line and the next keypoint line together.
    # 4. Parse the image id and image name from the header.
    # 5. Parse keypoint triples from the keypoint line.
    # 6. Return the dictionary.
    raise NotImplementedError("Parse COLMAP image keypoints and point ids.")


def parse_colmap_points3d(sparse_text_dir: Path) -> dict[int, dict[str, object]]:
    """Parse 3D point coordinates and tracks from COLMAP `points3D.txt`."""
    # Where this fits:
    # A semantic 3D point cloud needs each 3D point coordinate and its image
    # observations.
    #
    # COLMAP `points3D.txt` row shape:
    #
    # ```text
    # POINT3D_ID X Y Z R G B ERROR TRACK[]
    # ```
    #
    # Track entries come in pairs:
    #
    # ```text
    # IMAGE_ID POINT2D_IDX IMAGE_ID POINT2D_IDX ...
    # ```
    #
    # Output:
    # A dictionary keyed by 3D point id:
    #
    # ```python
    # {
    #     point3d_id: {
    #         "xyz": [x, y, z],
    #         "rgb": [r, g, b],
    #         "error": error,
    #         "track": [(image_id, point2d_idx), ...],
    #     }
    # }
    # ```
    #
    # Method:
    # 1. Open `sparse_text_dir / "points3D.txt"`.
    # 2. Skip blank lines and comment lines.
    # 3. Parse coordinates, color, reprojection error, and track pairs.
    # 4. Return the dictionary.
    raise NotImplementedError("Parse COLMAP 3D points and tracks.")


def vote_semantic_label_for_point(
    track: list[tuple[int, int]],
    image_keypoints: dict[int, dict[str, object]],
    semantic_maps: dict[str, Path],
    config: ApplicantReconstructionConfig,
) -> tuple[int, int, int]:
    """Vote for one semantic label for a 3D point."""
    # Where this fits:
    # A 3D point can be visible in several frames. This helper gathers the 2D
    # labels at those observed pixels and chooses one 3D label.
    #
    # Input:
    # - `track`: list of `(image_id, point2d_idx)` pairs
    # - `image_keypoints`: output from `parse_colmap_image_keypoints(...)`
    # - `semantic_maps`: output from `build_or_load_frame_semantics(...)`
    #
    # Output:
    # - `label_id`: chosen label id, or `0` for unknown
    # - `num_votes`: number of valid label pixels used
    # - `num_disagreements`: votes that did not match the chosen label
    #
    # Method:
    # 1. For each track observation, find the image name.
    # 2. Find the keypoint pixel `(x, y)` by `point2d_idx`.
    # 3. Load the matching label-map PNG.
    # 4. Round and clamp `(x, y)` to image bounds.
    # 5. Read the label id at that pixel.
    # 6. Ignore label `0` unless it is the only available label.
    # 7. Use majority vote for the final 3D label.
    # 8. Count disagreements.
    #
    # Checks:
    # - Never invent a label when there are no valid votes.
    # - Label ids must be inside `config.semantic_label_names`.
    raise NotImplementedError("Vote 2D semantic labels into one 3D label.")


def save_semantic_point_cloud_ply(
    points3d: dict[int, dict[str, object]],
    point_labels: dict[int, int],
    output_path: Path,
    config: ApplicantReconstructionConfig,
) -> str:
    """Save a semantic-colored PLY using the same COLMAP 3D points."""
    # Where this fits:
    # This is the semantic 3D output a reviewer can open.
    #
    # Goal:
    # Save one row per labeled 3D point.
    #
    # Semantic colors:
    # Use a fixed color table, for example:
    # - unknown: gray
    # - floor: green
    # - wall: blue
    # - table: orange
    # - chair: purple
    # - object: red
    #
    # PLY row format:
    # `x y z red green blue label_id`
    #
    # Checks:
    # - Use the same coordinates from COLMAP `points3D.txt`.
    # - Do not save semantic points that are not in the geometry.
    _ = config
    raise NotImplementedError("Save semantic-colored 3D point cloud.")


def write_semantic_alignment_report(
    semantic_summary: SemanticPointSummary,
    output_path: Path,
) -> str:
    """Save the semantic alignment report as JSON."""
    # Where this fits:
    # The challenge extension asks that labels align with geometry. This report
    # makes that check explicit.
    #
    # Include:
    # - `labeled_points`
    # - `total_points`
    # - `label_coverage`
    # - `disagreement_rate`
    # - `label_names`
    #
    # Plain meaning:
    # - coverage says how many reconstructed points received labels
    # - disagreement says how often different views voted for different labels
    #
    # Method:
    # Use `json.dumps(dict(semantic_summary), indent=2)`.
    _ = json
    raise NotImplementedError("Write semantic alignment report JSON.")


def lift_semantics_to_sparse_points(
    sparse_summary: ColmapSparseSummary,
    semantic_maps: dict[str, Path],
    workspace: dict[str, Path],
    config: ApplicantReconstructionConfig,
) -> SemanticPointSummary:
    """Attach semantic labels to reconstructed 3D points."""
    # Where this fits:
    # This is the semantic extension from the job advert.
    #
    # Plain idea:
    # A 3D point is seen in one or more images. If you know the pixel label in
    # those images, you can vote for a 3D label.
    #
    # Data source:
    # COLMAP `points3D.txt` stores a track for each 3D point. A track lists which
    # images and keypoints observed that point.
    #
    # Method:
    # 1. If `config.enable_semantics` is `False`, return a disabled summary.
    #    In this exercise, the default is `True`.
    # 2. Parse image keypoints with `parse_colmap_image_keypoints(...)`.
    # 3. Parse 3D points with `parse_colmap_points3d(...)`.
    # 4. For each 3D point, call `vote_semantic_label_for_point(...)`.
    # 5. Compute:
    #    `label_coverage = labeled_points / total_points`
    #    `disagreement_rate = total_disagreements / total_votes`
    # 6. Save semantic PLY with `save_semantic_point_cloud_ply(...)`.
    # 7. Save semantic alignment JSON with `write_semantic_alignment_report(...)`.
    #
    # Checks:
    # - Do not assign labels to points that have no matching semantic pixels.
    # - Report how many points received labels.
    # - Report label coverage and disagreement rate.
    # - Keep semantic output tied to the same 3D point coordinates.
    _ = (sparse_summary, semantic_maps, workspace)
    raise NotImplementedError("Lift 2D semantic labels onto 3D points.")


def load_ascii_ply_vertices_for_preview(ply_path: Path, max_points: int) -> dict[str, list[float] | list[int]]:
    """Load point coordinates and colors from an ASCII PLY file."""
    # Where this fits:
    # The visual preview needs actual 3D coordinates from the saved point cloud.
    #
    # Goal:
    # Read a PLY point cloud and return lists that plotting code can use.
    #
    # Input:
    # - `ply_path`: path to a `.ply` file
    # - `max_points`: largest number of points to keep for the preview
    #
    # Output:
    # A dictionary with these keys:
    # - `"x"`: list of x coordinates
    # - `"y"`: list of y coordinates
    # - `"z"`: list of z coordinates
    # - `"red"`: list of red color values from 0 to 255
    # - `"green"`: list of green color values from 0 to 255
    # - `"blue"`: list of blue color values from 0 to 255
    #
    # Method:
    # 1. Read the header until the line `end_header`.
    # 2. Check whether the header contains `format ascii`.
    # 3. Find the vertex property order from lines like:
    #    `property float x`
    #    `property uchar red`
    # 4. Read at most `max_points` vertex rows.
    # 5. Convert x, y, and z to `float`.
    # 6. Convert red, green, and blue to `int` if they exist.
    # 7. If colors are missing, use light gray: red=180, green=180, blue=180.
    #
    # Important:
    # Some tools save binary PLY files. This helper is intentionally simple and
    # reads ASCII PLY files. If your COLMAP output is binary, convert it to ASCII
    # before plotting, or replace this helper with `open3d.io.read_point_cloud`.
    #
    # Checks:
    # - `len(result["x"])` should be greater than 0 for a working reconstruction.
    # - The same number of x, y, z, red, green, and blue values should be returned.
    _ = (ply_path, max_points)
    raise NotImplementedError("Load ASCII PLY vertices for plotting.")


def save_sparse_point_cloud_preview(
    sparse_summary: ColmapSparseSummary,
    output_path: Path,
    config: ApplicantReconstructionConfig,
) -> str:
    """Save a static PNG preview of the sparse point cloud."""
    # Where this fits:
    # This gives you a quick visual check without opening a 3D program.
    #
    # Goal:
    # Save `sparse_point_cloud_preview.png`.
    #
    # Input:
    # - `sparse_summary["sparse_ply_path"]`: path to `reconstruction_sparse.ply`
    # - `output_path`: where to save the PNG
    #
    # Output:
    # Return `str(output_path)`.
    #
    # Method:
    # 1. Use `load_ascii_ply_vertices_for_preview(...)`.
    # 2. Use Matplotlib with:
    #    `import matplotlib`
    #    `matplotlib.use("Agg")`
    #    `import matplotlib.pyplot as plt`
    # 3. Create a figure with two plots:
    #    - top view: x on the horizontal axis and z on the vertical axis
    #    - side view: x on the horizontal axis and y on the vertical axis
    # 4. Draw points with `ax.scatter(...)`.
    # 5. Build colors as floats in the range 0 to 1:
    #    `colors = [(r / 255, g / 255, b / 255) for ...]`
    # 6. Label the axes and add clear titles.
    # 7. Save with `fig.savefig(output_path, dpi=160, bbox_inches="tight")`.
    # 8. Close the figure with `plt.close(fig)`.
    #
    # Checks:
    # - The image should not be blank.
    # - The point cloud should look like a spread of points, not one dot.
    _ = (sparse_summary, output_path, config)
    raise NotImplementedError("Save sparse point cloud preview PNG.")


def save_semantic_point_cloud_preview(
    semantic_summary: SemanticPointSummary,
    output_path: Path,
    config: ApplicantReconstructionConfig,
) -> str:
    """Save a static PNG preview of the semantic point cloud."""
    # Where this fits:
    # This is the fastest way to check whether labels are attached to 3D points.
    #
    # Goal:
    # Save `semantic_point_cloud_preview.png`.
    #
    # Input:
    # - `semantic_summary["semantic_ply_path"]`: path to `semantic_point_cloud.ply`
    # - `semantic_summary["enabled"]`: whether semantics ran
    #
    # Output:
    # Return `str(output_path)`.
    #
    # Method:
    # 1. If semantics are disabled, write a small placeholder plot that says
    #    semantics were disabled. This makes the final folder easy to inspect.
    # 2. Otherwise, use `load_ascii_ply_vertices_for_preview(...)`.
    # 3. Plot top view and side view, like `save_sparse_point_cloud_preview(...)`.
    # 4. Use the point colors from the semantic PLY.
    # 5. Add a title that includes:
    #    - labeled point count
    #    - label coverage
    #    - disagreement rate
    #
    # Checks:
    # - If semantics are enabled, the preview should contain colored 3D points.
    # - A high disagreement rate means views did not agree about labels.
    _ = (semantic_summary, output_path, config)
    raise NotImplementedError("Save semantic point cloud preview PNG.")


def save_plotly_point_cloud_viewer(
    sparse_summary: ColmapSparseSummary,
    semantic_summary: SemanticPointSummary,
    output_path: Path,
    config: ApplicantReconstructionConfig,
) -> str:
    """Save an interactive browser viewer for the point cloud."""
    # Where this fits:
    # This is the main file you open when you want to know whether the system
    # worked. It should let you rotate, pan, and zoom the reconstructed scene.
    #
    # Goal:
    # Save `point_cloud_viewer.html`.
    #
    # Input:
    # - sparse PLY path from `sparse_summary`
    # - semantic PLY path from `semantic_summary`, if semantics are enabled
    #
    # Output:
    # Return `str(output_path)`.
    #
    # Method:
    # 1. Use `load_ascii_ply_vertices_for_preview(...)` for the sparse cloud.
    # 2. Import Plotly:
    #    `import plotly.graph_objects as go`
    # 3. Create a sparse trace with:
    #    `go.Scatter3d(x=x, y=y, z=z, mode="markers", marker={...}, name="sparse")`
    # 4. Use marker settings like:
    #    `marker={"size": 2, "color": color_strings, "opacity": 0.85}`
    # 5. If semantics are enabled, load the semantic PLY and add a second trace
    #    named `"semantic labels"`.
    # 6. Use:
    #    `fig.update_layout(scene={"aspectmode": "data"})`
    #    so the axes use the point cloud shape instead of a stretched cube.
    # 7. Save with:
    #    `fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)`
    #
    # Checks:
    # - Open the HTML file in a browser.
    # - You should be able to rotate the point cloud with the mouse.
    # - If the page is blank, check that the PLY loader returned points.
    _ = (sparse_summary, semantic_summary, output_path, config)
    raise NotImplementedError("Save interactive point cloud viewer HTML.")


def write_visual_index(
    report: ApplicantSystemReport,
    frame_summary: FrameExtractionSummary,
    sparse_summary: ColmapSparseSummary,
    dense_summary: DenseReconstructionSummary,
    semantic_summary: SemanticPointSummary,
    output_path: Path,
) -> str:
    """Write a small HTML page that links to the visual outputs."""
    # Where this fits:
    # A reviewer should not have to guess which file to open first.
    #
    # Goal:
    # Save `VISUAL_INDEX.html`.
    #
    # Input:
    # - the final report dictionary
    # - frame, sparse, dense, and semantic summaries
    # - paths to the preview images and viewer
    #
    # Output:
    # Return `str(output_path)`.
    #
    # Method:
    # 1. Build a small HTML string.
    # 2. Put this near the top:
    #    `Open point_cloud_viewer.html first.`
    # 3. Add links to:
    #    - `point_cloud_viewer.html`
    #    - `sparse_point_cloud_preview.png`
    #    - `semantic_point_cloud_preview.png`
    #    - `frame_contact_sheet.png`
    #    - `quality_report.json`
    #    - `RUN.md`
    #    - `DESIGN_NOTE.md`
    # 4. Add the key metrics in plain text:
    #    - frames extracted
    #    - registered images
    #    - sparse points
    #    - reprojection error
    #    - semantic coverage
    # 5. Use `Path(...).name` for links when files are in the same folder, or
    #    use relative paths with `path.relative_to(output_path.parent)`.
    # 6. Write with `output_path.write_text(html, encoding="utf-8")`.
    #
    # Checks:
    # - Opening `VISUAL_INDEX.html` should show what to inspect first.
    # - The preview images should be visible or linked from the page.
    _ = (report, frame_summary, sparse_summary, dense_summary, semantic_summary, output_path)
    raise NotImplementedError("Write visual index HTML.")


def write_quality_report(
    report: ApplicantSystemReport,
    sparse_summary: ColmapSparseSummary,
    dense_summary: DenseReconstructionSummary,
    semantic_summary: SemanticPointSummary,
    output_path: Path,
) -> str:
    """Save a JSON quality report."""
    # Where this fits:
    # A reviewer should be able to see whether the reconstruction worked.
    #
    # Goal:
    # Write a JSON file with:
    # - final report fields
    # - sparse summary fields
    # - dense summary fields
    # - semantic summary fields
    # - visual output paths
    # - whether semantic alignment passed
    #
    # Method:
    # Use `json.dumps(data, indent=2)`.
    #
    # Checks:
    # - Convert `Path` objects to strings.
    _ = json
    raise NotImplementedError("Write quality report JSON.")


def write_run_instructions(config: ApplicantReconstructionConfig, output_path: Path) -> str:
    """Write applicant-style run instructions."""
    # Include:
    # - required tools: Python dependencies and COLMAP
    # - where to put a phone video
    # - recommended online data:
    #   - TUM RGB-D download page from `config.recommended_tum_rgbd_url`
    #   - recommended sequence from `config.recommended_tum_sequence`
    #   - COLMAP dataset page from `config.colmap_dataset_url`
    # - exact run command
    # - expected output files
    # - how to enable dense reconstruction with `config.run_dense = True`
    # - semantic model id from `config.semantic_model_id`
    # - stronger semantic model id from `config.stronger_semantic_model_id`
    # - depth model id from `config.depth_model_id`
    # - open `VISUAL_INDEX.html` first
    # - then open `point_cloud_viewer.html` to rotate the point cloud
    # - use `sparse_point_cloud_preview.png` as the quick static check
    # - how to inspect `semantic_point_cloud.ply`
    #
    # Suggested command:
    #
    # ```text
    # uv run python exercises/embodied_ai/21_applicant_video_3d_reconstruction_system.py
    # ```
    _ = config
    raise NotImplementedError("Write applicant-system run instructions.")


def write_design_note(report: ApplicantSystemReport, config: ApplicantReconstructionConfig, output_path: Path) -> str:
    """Write a short design note."""
    # Include:
    # - why COLMAP was used
    # - why sparse reconstruction comes before dense reconstruction
    # - which input video or online dataset was used
    # - which pretrained semantic model was used
    # - whether a depth model was used and whether its output is relative depth
    # - what the quality metrics mean
    # - known limits: motion blur, textureless walls, unknown absolute scale
    # - how semantics are aligned with geometry
    # - what semantic coverage and disagreement mean
    # - what the visual preview showed
    # - whether `point_cloud_viewer.html` was useful for judging the result
    # - next improvements
    #
    # Keep the wording direct and concrete.
    _ = (report, config)
    raise NotImplementedError("Write applicant-system design note.")


def passes_basic_geometry_check(summary: ColmapSparseSummary, config: ApplicantReconstructionConfig) -> bool:
    """Return True when the sparse reconstruction meets the basic bar."""
    return (
        summary["registered_images"] >= config.min_registered_images
        and summary["sparse_points"] >= config.min_sparse_points
        and summary["mean_reprojection_error_px"] <= config.max_mean_reprojection_error_px
    )


def passes_semantic_alignment_check(summary: SemanticPointSummary, config: ApplicantReconstructionConfig) -> bool:
    """Return True when semantic labels are usefully aligned with geometry."""
    # Plain meaning:
    # Semantic labels are useful only if enough 3D points receive labels and the
    # views mostly agree about those labels.
    if not summary["enabled"]:
        return False
    return (
        summary["label_coverage"] >= config.min_semantic_label_coverage
        and summary["disagreement_rate"] <= config.max_semantic_disagreement_rate
    )


def run_applicant_reconstruction_system(config: ApplicantReconstructionConfig) -> ApplicantSystemReport:
    """Run the full applicant-style reconstruction system."""
    # Main workflow:
    # real video -> frame extraction -> COLMAP sparse reconstruction -> optional
    # dense output -> semantic labels on 3D points -> visual viewer -> report.
    #
    # Steps:
    # 1. Choose the input video.
    # 2. Prepare the workspace.
    # 3. Extract frames.
    # 4. Run COLMAP sparse reconstruction.
    # 5. Parse sparse stats.
    # 6. Optionally run dense reconstruction.
    # 7. Build or load 2D semantic maps.
    # 8. Lift semantic labels to 3D.
    # 9. Save `sparse_point_cloud_preview.png`.
    # 10. Save `semantic_point_cloud_preview.png`.
    # 11. Save `point_cloud_viewer.html`.
    # 12. Build the final report dictionary.
    # 13. Write `VISUAL_INDEX.html`.
    # 14. Write quality report, run instructions, and design note.
    # 15. Return the report.
    #
    # Checks:
    # - If COLMAP is missing, write a clear report instead of failing silently.
    # - The final output folder should be understandable without reading code.
    # - The final output folder should contain an HTML file to open first.
    # - A reviewer should be able to see a point cloud without extra software.
    # - Semantic labels should be reported with coverage and disagreement rate.
    raise NotImplementedError("Run applicant-style video-to-3D reconstruction system.")


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
    """Run the applicant-style reconstruction scaffold."""
    config = ApplicantReconstructionConfig()
    report, ok = report_task_progress("run_applicant_reconstruction_system", run_applicant_reconstruction_system, config)
    if ok:
        print(report)


if __name__ == "__main__":
    main()
