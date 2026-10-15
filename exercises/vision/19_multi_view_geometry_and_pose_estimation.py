"""
============================================================
EXERCISE 19: MULTI VIEW GEOMETRY AND POSE ESTIMATION
============================================================

WHAT YOU ARE BUILDING
You are building a small two-view pose-estimation system.

Pose means:
- where the camera is
- which way the camera is pointing

Relative pose means:
- how camera view 2 moved compared with camera view 1

In this exercise, the two views are a real stereo image pair.
The cameras are fixed next to each other, so the true movement is known from the
calibration file.

WHY THIS EXERCISE EXISTS
Embodied agents move through the world.
When the camera moves, the image changes.

Two-view pose estimation answers this question:

    "Given the same scene seen from two camera views, how did the camera move?"

This matters for:
- 3D reconstruction
- SLAM
- visual odometry
- navigation
- checking whether an agent's visual memory is consistent

REAL SYSTEM FIRST
In a real two-view pose system:

1. load two real images
2. find visual keypoints in both images
3. match keypoints between the two images
4. reject bad matches
5. estimate the camera movement from the good matches
6. triangulate 3D points from the good matches
7. project those 3D points back into the images
8. report errors and save visual checks

This exercise follows that same shape.

DATASET
Use a real stereo dataset, not random tensors and not hand-written shifted points.

Default dataset:
- Middlebury Stereo 2014
- default scene: Adirondack
- files used:
  - `im0.png`: left image
  - `im1.png`: right image
  - `calib.txt`: camera intrinsics and baseline

The exercise downloads those files from the official Middlebury site when they are not
already in `data/middlebury_2014/Adirondack-perfect/`.

KITTI Odometry is the natural larger dataset after this exercise.
Middlebury is used here because it is smaller and easier to run as a guided exercise.

WHY THIS IS TRAINABLE
Classic pose estimation is mostly geometry.
It does not need a neural network by itself.

To make the exercise trainable without faking the data, this file trains a small match
filter:

    match features -> probability that the match is geometrically consistent

The match filter learns from labels made with the real camera calibration.
Then the pose step uses the filtered real matches.

This is a small version of a common real pattern:

    learned matching or match filtering + geometric pose estimation

Mapping from the real system to this exercise:

    real system                         this exercise
    ---------------------------------------------------------------
    real image pair                     Middlebury left/right images
    camera intrinsics                   Middlebury `calib.txt`
    known camera baseline               Middlebury `baseline`
    keypoint detector                   OpenCV ORB
    raw keypoint matches                ORB descriptor matches
    learned match filtering             small PyTorch MLP
    pose estimation                     OpenCV essential matrix and recoverPose
    3D point recovery                   OpenCV triangulatePoints
    quality report                      inlier accuracy, pose error, reprojection error
    visual checks                       saved match, point-cloud, and curve images

What is included:
- real image loading
- real calibration loading
- real feature matching
- a train/validation/test split for match filtering
- a small trainable PyTorch model
- pose metrics
- reprojection metrics
- saved visuals

What is skipped:
- large learned feature matchers
- bundle adjustment
- multi-frame tracking
- full visual odometry over a long sequence
- scale recovery from monocular video

Those skipped parts matter later, but this exercise is the first real two-view step.

INPUTS AND OUTPUTS
Each raw match becomes one training row.

Input features for one match:
- normalized left x and y
- normalized right x and y
- x shift and y shift between views
- ORB match distance

Feature shape:

    [num_matches, 7]

Target:
- `1` if the match fits the known stereo geometry
- `0` if it does not

Model output:
- one logit per match
- a larger logit means the model thinks the match is more likely to be good

LOSS FUNCTION
Use binary cross entropy with logits:

    loss = nn.BCEWithLogitsLoss()(logits, labels)

Plain meaning:
- this trains the model to score good matches higher than bad matches

METRICS TO TRACK
Report these:
- match-filter validation accuracy, higher is better
- match-filter test accuracy, higher is better
- inlier precision, higher is better
- inlier recall, higher is better
- rotation error in degrees, lower is better
- translation-direction error in degrees, lower is better
- mean reprojection error in pixels, lower is better
- number of pose inliers, higher is usually better if the matches are real

VISUALS TO SAVE
Save simple images to `outputs/two_view_pose_105/`:
- raw ORB matches
- filtered inlier matches
- training curve
- sparse 3D point cloud

STEP-BY-STEP TODO
1. Parse the Middlebury calibration file.
2. Detect ORB keypoints and raw matches in the real images.
3. Build match features and geometry labels.
4. Build the small match-filter model.
5. Train the match filter.
6. Use filtered matches to estimate pose.
7. Triangulate sparse 3D points.
8. Report pose and reprojection metrics.
9. Save visual checks.

DEFINITION OF DONE
You are done when:
- the exercise uses real Middlebury images
- the match-filter model trains on real match data
- validation and test match metrics are printed
- pose and reprojection metrics are printed
- visual outputs are saved
- you can explain how good matches help estimate camera movement
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
import re
from typing import NamedTuple
from urllib.parse import urljoin

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]


# Learning order for this exercise:
# 1. Load the real images and calibration.
# 2. Make raw ORB matches.
# 3. Turn matches into training rows.
# 4. Train the match filter.
# 5. Estimate pose from filtered matches.
# 6. Check pose with reprojection error.
# 7. Save visuals after the numbers make sense.


@dataclass
class TwoViewPoseConfig:
    """Settings for the real two-view pose exercise."""

    scene_name: str = "Adirondack-perfect"
    dataset_root: Path = BOOTCAMP_ROOT / "data" / "middlebury_2014"
    output_dir: Path = BOOTCAMP_ROOT / "outputs" / "two_view_pose_105"
    base_url: str = "https://vision.middlebury.edu/stereo/data/scenes2014/datasets/"

    max_image_width: int = 960
    max_features: int = 4_000
    max_raw_matches: int = 2_000
    epipolar_threshold_px: float = 1.5
    min_pose_matches: int = 20

    batch_size: int = 128
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    random_seed: int = 105


class Calibration(NamedTuple):
    """Camera calibration for the two Middlebury views."""

    k_left: np.ndarray
    k_right: np.ndarray
    baseline_m: float
    true_rotation: np.ndarray
    true_translation_direction: np.ndarray


class TwoViewData(NamedTuple):
    """Loaded real images and their calibration."""

    left_gray: np.ndarray
    right_gray: np.ndarray
    calibration: Calibration


class MatchData(NamedTuple):
    """Raw image matches and model-ready rows."""

    points_left: np.ndarray
    points_right: np.ndarray
    distances: np.ndarray
    features: torch.Tensor
    labels: torch.Tensor


class PoseResult(NamedTuple):
    """Estimated pose and the inlier matches used by OpenCV."""

    rotation: np.ndarray
    translation_direction: np.ndarray
    inlier_mask: np.ndarray
    points_left: np.ndarray
    points_right: np.ndarray


class PoseMetrics(NamedTuple):
    """Final quality numbers for two-view pose."""

    val_accuracy: float
    test_accuracy: float
    inlier_precision: float
    inlier_recall: float
    rotation_error_deg: float
    translation_error_deg: float
    mean_reprojection_error_px: float
    num_pose_inliers: int


class MatchDataset(Dataset):
    """A real match-filter dataset."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class MatchFilterMLP(nn.Module):
    """Small model that predicts whether a real image match is good."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Where this fits:
        # This model filters raw ORB matches before pose estimation.
        #
        # Goal:
        # Build a small binary classifier with `torch.nn`.
        #
        # Input:
        # - `input_dim`: number of match features.
        # - this exercise uses 7 features per match.
        #
        # Output:
        # - `self.net`: an `nn.Sequential` model that maps `[batch, input_dim]` to `[batch, 1]`.
        #
        # Method:
        # Use this exact call shape:
        # `nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))`
        #
        # Steps:
        # 1. Create `self.net`.
        # 2. Use two hidden layers.
        # 3. End with one output logit.
        #
        # Checks:
        # - `model(torch.randn(4, input_dim)).shape` should be `[4]` after `forward`.
        raise NotImplementedError("Build the match-filter model.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one match-quality logit per row."""
        # Where this fits:
        # The training loop calls this method for every match batch.
        #
        # Goal:
        # Run features through `self.net` and return a flat tensor.
        #
        # Input:
        # - `features`: tensor shaped `[batch, input_dim]`
        #
        # Output:
        # - tensor shaped `[batch]`
        #
        # Method:
        # Use:
        # `logits = self.net(features).squeeze(-1)`
        #
        # Steps:
        # 1. Pass `features` into `self.net`.
        # 2. Remove the final size-1 dimension with `.squeeze(-1)`.
        # 3. Return the logits.
        #
        # Checks:
        # - each input match should have one output score.
        raise NotImplementedError("Run the match-filter forward pass.")


def set_seed(seed: int) -> None:
    """Set random seeds so repeated runs are easier to compare."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_middlebury_pair(config: TwoViewPoseConfig) -> dict[str, Path]:
    """Download the needed Middlebury files if they are missing."""
    scene_dir = config.dataset_root / config.scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    filenames = ["im0.png", "im1.png", "calib.txt"]
    paths = {name: scene_dir / name for name in filenames}

    for name, path in paths.items():
        if path.exists():
            continue
        url = urljoin(config.base_url, f"{config.scene_name}/{name}")
        print(f"Downloading {url}")
        response = requests.get(url, timeout=60, headers={"User-Agent": "pytorch-bootcamp"})
        response.raise_for_status()
        path.write_bytes(response.content)

    return paths


def _read_matrix_from_calibration(text: str, name: str) -> np.ndarray:
    """Read one 3 by 3 matrix from Middlebury calibration text."""
    match = re.search(rf"{name}=\[([^\]]+)\]", text)
    if match is None:
        raise ValueError(f"Could not find {name} in calibration text.")

    rows = match.group(1).split(";")
    values = [[float(value) for value in row.strip().split()] for row in rows]
    matrix = np.array(values, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} should be a 3 by 3 matrix.")
    return matrix


def parse_middlebury_calibration(path: Path) -> Calibration:
    """Read camera intrinsics and baseline from `calib.txt`."""
    # Where this fits:
    # Pose estimation needs the camera intrinsics.
    # The true stereo baseline gives a simple target movement direction.
    #
    # Goal:
    # Return a `Calibration` object.
    #
    # Input:
    # - `path`: path to Middlebury `calib.txt`
    #
    # Output:
    # - `Calibration(...)` with:
    #   - `k_left`: left camera matrix shaped `[3, 3]`
    #   - `k_right`: right camera matrix shaped `[3, 3]`
    #   - `baseline_m`: positive baseline in meters
    #   - `true_rotation`: identity matrix shaped `[3, 3]`
    #   - `true_translation_direction`: unit vector shaped `[3]`
    #
    # Method:
    # Use:
    # - `_read_matrix_from_calibration(text, "cam0")`
    # - `_read_matrix_from_calibration(text, "cam1")`
    # - `re.search(r"baseline=([0-9.]+)", text)`
    #
    # Middlebury baseline is in millimeters.
    # Divide it by `1000.0` to get meters.
    #
    # For a rectified stereo pair, the cameras point the same way.
    # So the true rotation is `np.eye(3, dtype=np.float64)`.
    #
    # The right camera is shifted sideways from the left camera.
    # For this exercise, use this direction:
    # `np.array([1.0, 0.0, 0.0], dtype=np.float64)`
    #
    # Steps:
    # 1. Read the text file.
    # 2. Parse `cam0` as `k_left`.
    # 3. Parse `cam1` as `k_right`.
    # 4. Parse `baseline`.
    # 5. Convert baseline from millimeters to meters.
    # 6. Build the identity rotation.
    # 7. Build the x-axis translation direction.
    # 8. Return `Calibration(...)`.
    #
    # Checks:
    # - both camera matrices should be shaped `(3, 3)`.
    # - baseline should be positive.
    raise NotImplementedError("Parse Middlebury camera calibration.")


def resize_if_needed(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Resize a grayscale image and return the scale used."""
    height, width = image.shape
    if width <= max_width:
        return image, 1.0

    scale = max_width / float(width)
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_intrinsics(k: np.ndarray, scale: float) -> np.ndarray:
    """Scale a camera matrix after image resizing."""
    scaled = k.copy()
    scaled[0, 0] *= scale
    scaled[1, 1] *= scale
    scaled[0, 2] *= scale
    scaled[1, 2] *= scale
    return scaled


def load_two_view_data(config: TwoViewPoseConfig) -> TwoViewData:
    """Load the real Middlebury image pair and calibration."""
    paths = download_middlebury_pair(config)

    left = cv2.imread(str(paths["im0.png"]), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(paths["im1.png"]), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError("OpenCV could not read the Middlebury image files.")

    calibration = parse_middlebury_calibration(paths["calib.txt"])

    left, left_scale = resize_if_needed(left, config.max_image_width)
    right, right_scale = resize_if_needed(right, config.max_image_width)
    if not math.isclose(left_scale, right_scale):
        raise ValueError("Left and right images should use the same resize scale.")

    calibration = Calibration(
        k_left=scale_intrinsics(calibration.k_left, left_scale),
        k_right=scale_intrinsics(calibration.k_right, right_scale),
        baseline_m=calibration.baseline_m,
        true_rotation=calibration.true_rotation,
        true_translation_direction=calibration.true_translation_direction,
    )

    return TwoViewData(left_gray=left, right_gray=right, calibration=calibration)


def detect_orb_matches(data: TwoViewData, config: TwoViewPoseConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect ORB keypoints and raw descriptor matches in the two real images."""
    # Where this fits:
    # Pose estimation starts with finding the same visual points in both images.
    #
    # Goal:
    # Return raw matched point coordinates and descriptor distances.
    #
    # Input:
    # - `data.left_gray`: grayscale image array shaped `[height, width]`
    # - `data.right_gray`: grayscale image array shaped `[height, width]`
    #
    # Output:
    # - `points_left`: NumPy array shaped `[num_matches, 2]`
    # - `points_right`: NumPy array shaped `[num_matches, 2]`
    # - `distances`: NumPy array shaped `[num_matches]`
    #
    # Method:
    # Use OpenCV ORB and brute-force matching:
    # - `orb = cv2.ORB_create(nfeatures=config.max_features)`
    # - `keypoints_left, descriptors_left = orb.detectAndCompute(data.left_gray, None)`
    # - `keypoints_right, descriptors_right = orb.detectAndCompute(data.right_gray, None)`
    # - `matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)`
    # - `matches = matcher.match(descriptors_left, descriptors_right)`
    #
    # Each OpenCV match has:
    # - `match.queryIdx`: index into left keypoints
    # - `match.trainIdx`: index into right keypoints
    # - `match.distance`: descriptor distance. Lower usually means more similar.
    #
    # Steps:
    # 1. Create the ORB detector.
    # 2. Detect keypoints and descriptors in the left image.
    # 3. Detect keypoints and descriptors in the right image.
    # 4. Check that descriptors are not `None`.
    # 5. Match descriptors with `BFMatcher`.
    # 6. Sort matches by `match.distance`.
    # 7. Keep at most `config.max_raw_matches`.
    # 8. Build point arrays from `keypoint.pt`.
    # 9. Build the distance array.
    #
    # Checks:
    # - there should be at least `config.min_pose_matches` matches.
    # - points should have two columns: x and y.
    raise NotImplementedError("Detect real ORB matches between the two images.")


def _normalize_points(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convert pixel points to normalized camera coordinates."""
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points.astype(np.float64), ones], axis=1)
    normalized = np.linalg.solve(k, homogeneous.T).T
    return normalized[:, :2]


def _skew(vector: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix used in the essential matrix."""
    x, y, z = vector.tolist()
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def build_match_dataset(
    points_left: np.ndarray,
    points_right: np.ndarray,
    distances: np.ndarray,
    data: TwoViewData,
    config: TwoViewPoseConfig,
) -> MatchData:
    """Turn raw matches into features and geometry labels."""
    # Where this fits:
    # This is the trainable part of the exercise.
    # It turns real raw matches into rows for the match-filter model.
    #
    # Goal:
    # Build a `MatchData` object with feature rows and 0/1 labels.
    #
    # Input:
    # - `points_left`: array shaped `[num_matches, 2]`
    # - `points_right`: array shaped `[num_matches, 2]`
    # - `distances`: array shaped `[num_matches]`
    # - `data.calibration`: real camera intrinsics and true stereo direction
    #
    # Output:
    # - `MatchData(...)`
    # - `features`: tensor shaped `[num_matches, 7]`
    # - `labels`: tensor shaped `[num_matches]`
    #
    # Method:
    # Build these 7 features:
    # 1. left x divided by image width
    # 2. left y divided by image height
    # 3. right x divided by image width
    # 4. right y divided by image height
    # 5. `(left_x - right_x)` divided by image width
    # 6. `(left_y - right_y)` divided by image height
    # 7. ORB distance divided by the largest ORB distance in this match set
    #
    # Label rule:
    # A match is good when it fits the known stereo epipolar geometry.
    #
    # Use the true essential matrix:
    # - `translation = data.calibration.true_translation_direction`
    # - `essential = _skew(translation) @ data.calibration.true_rotation`
    #
    # For normalized points, the epipolar value is:
    # - `value = right_homogeneous @ essential @ left_homogeneous`
    #
    # Use absolute value as a simple residual.
    # Label is 1 when residual is below a threshold.
    #
    # Practical note:
    # The threshold is in normalized camera units, not pixels.
    # Convert the pixel threshold roughly with:
    # `normalized_threshold = config.epipolar_threshold_px / data.calibration.k_left[0, 0]`
    #
    # Steps:
    # 1. Read image height and width.
    # 2. Build the 7 feature columns.
    # 3. Normalize left and right points with `_normalize_points`.
    # 4. Turn normalized points into homogeneous `[x, y, 1]` rows.
    # 5. Build the true essential matrix.
    # 6. Compute one epipolar residual per match.
    # 7. Convert residuals to 0/1 labels.
    # 8. Return `MatchData(...)`.
    #
    # Checks:
    # - there should be both positive and negative labels.
    # - `features.shape[1]` should be 7.
    raise NotImplementedError("Build match features and geometry labels.")


def build_dataloaders(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: TwoViewPoseConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split match rows into train, validation, and test dataloaders."""
    dataset = MatchDataset(features, labels)
    total = len(dataset)
    train_size = int(total * config.train_fraction)
    val_size = int(total * config.val_fraction)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(config.random_seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train the match-filter model for one epoch."""
    # Where this fits:
    # This is the part that changes the learned match filter.
    #
    # Goal:
    # Run one training pass and return average loss.
    #
    # Input:
    # - `model`: predicts one logit per match
    # - `loader`: returns `(features, labels)`
    # - `optimizer`: updates model weights
    # - `loss_fn`: use `nn.BCEWithLogitsLoss()`
    # - `device`: usually `cuda` if available, else `cpu`
    #
    # Output:
    # - one Python float: average training loss
    #
    # Method:
    # Use the normal PyTorch training steps:
    # - `model.train()`
    # - `features = features.to(device)`
    # - `labels = labels.to(device)`
    # - `logits = model(features)`
    # - `loss = loss_fn(logits, labels)`
    # - `optimizer.zero_grad()`
    # - `loss.backward()`
    # - `optimizer.step()`
    #
    # Steps:
    # 1. Put the model in train mode.
    # 2. Loop over batches.
    # 3. Move features and labels to the device.
    # 4. Predict logits.
    # 5. Compute binary cross entropy loss.
    # 6. Clear old gradients.
    # 7. Backprop.
    # 8. Step the optimizer.
    # 9. Return average loss weighted by batch size.
    #
    # Checks:
    # - loss should usually go down over a few epochs.
    raise NotImplementedError("Train the match filter for one epoch.")


@torch.no_grad()
def evaluate_match_filter(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return match-filter accuracy."""
    # Where this fits:
    # This checks whether the learned match filter can separate good and bad matches.
    #
    # Goal:
    # Return classification accuracy.
    #
    # Input:
    # - `model`: trained match filter
    # - `loader`: validation or test dataloader
    #
    # Output:
    # - one float from 0 to 1
    #
    # Method:
    # Use:
    # - `model.eval()`
    # - `torch.sigmoid(logits)`
    # - `predicted = (probabilities >= 0.5).float()`
    # - `(predicted == labels).float().sum()`
    #
    # Steps:
    # 1. Put model in eval mode.
    # 2. Loop over batches without gradients.
    # 3. Predict logits.
    # 4. Convert logits to probabilities with sigmoid.
    # 5. Threshold probabilities at 0.5.
    # 6. Count correct predictions.
    # 7. Divide by total labels.
    #
    # Checks:
    # - accuracy should be between 0 and 1.
    raise NotImplementedError("Evaluate match-filter accuracy.")


@torch.no_grad()
def predict_good_match_mask(model: nn.Module, features: torch.Tensor, device: torch.device) -> np.ndarray:
    """Predict which raw matches should be kept for pose estimation."""
    model.eval()
    logits = model(features.to(device))
    probabilities = torch.sigmoid(logits).cpu().numpy()
    return probabilities >= 0.5


def estimate_pose_from_matches(
    match_data: MatchData,
    keep_mask: np.ndarray,
    calibration: Calibration,
    config: TwoViewPoseConfig,
) -> PoseResult:
    """Estimate relative pose from filtered real matches."""
    # Where this fits:
    # The learned match filter chooses matches.
    # This geometry step estimates camera movement from those matches.
    #
    # Goal:
    # Estimate rotation and translation direction with OpenCV.
    #
    # Input:
    # - `match_data.points_left`: raw left points shaped `[num_matches, 2]`
    # - `match_data.points_right`: raw right points shaped `[num_matches, 2]`
    # - `keep_mask`: boolean array shaped `[num_matches]`
    # - `calibration.k_left`: left camera matrix
    #
    # Output:
    # - `PoseResult(...)`
    #
    # Method:
    # Use OpenCV:
    # - `cv2.findEssentialMat(points_left, points_right, calibration.k_left, method=cv2.RANSAC, prob=0.999, threshold=1.0)`
    # - `cv2.recoverPose(essential, points_left, points_right, calibration.k_left)`
    #
    # `recoverPose` returns a translation vector with unknown scale.
    # That is okay here.
    # We compare only the translation direction.
    #
    # Steps:
    # 1. Keep only rows where `keep_mask` is True.
    # 2. Check that at least `config.min_pose_matches` remain.
    # 3. Convert points to `np.float64`.
    # 4. Call `cv2.findEssentialMat`.
    # 5. Check that an essential matrix was found.
    # 6. Call `cv2.recoverPose`.
    # 7. Flatten and normalize the translation vector.
    # 8. Return `PoseResult(...)`.
    #
    # Checks:
    # - rotation should be shaped `(3, 3)`.
    # - translation direction should be shaped `(3,)`.
    raise NotImplementedError("Estimate camera pose from filtered matches.")


def triangulate_inlier_points(pose: PoseResult, calibration: Calibration) -> np.ndarray:
    """Triangulate sparse 3D points from the pose inliers."""
    # Where this fits:
    # After pose estimation, triangulation checks whether the pose can explain 3D points.
    #
    # Goal:
    # Return sparse 3D points from inlier matches.
    #
    # Input:
    # - `pose.points_left`: filtered left points
    # - `pose.points_right`: filtered right points
    # - `pose.rotation`: estimated rotation
    # - `pose.translation_direction`: estimated translation direction
    #
    # Output:
    # - NumPy array shaped `[num_points, 3]`
    #
    # Method:
    # Use OpenCV:
    # - `p0 = calibration.k_left @ np.hstack([np.eye(3), np.zeros((3, 1))])`
    # - `t = pose.translation_direction.reshape(3, 1) * calibration.baseline_m`
    # - `p1 = calibration.k_left @ np.hstack([pose.rotation, t])`
    # - `points_4d = cv2.triangulatePoints(p0, p1, left_points.T, right_points.T)`
    # - `points_3d = (points_4d[:3] / points_4d[3:4]).T`
    #
    # Steps:
    # 1. Keep only pose inliers from `pose.inlier_mask`.
    # 2. Build projection matrix for camera 1.
    # 3. Build projection matrix for camera 2.
    # 4. Call `cv2.triangulatePoints`.
    # 5. Convert homogeneous 4D points to 3D points.
    # 6. Keep only finite points with positive depth.
    # 7. Return the 3D points.
    #
    # Checks:
    # - returned points should have 3 columns.
    raise NotImplementedError("Triangulate sparse 3D points from inlier matches.")


def _angle_between_degrees(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle between two vectors in degrees."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _rotation_error_degrees(estimated: np.ndarray, target: np.ndarray) -> float:
    """Return rotation difference in degrees."""
    delta = estimated @ target.T
    value = (np.trace(delta) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def compute_reprojection_error(
    points_3d: np.ndarray,
    pose: PoseResult,
    calibration: Calibration,
) -> float:
    """Measure how well triangulated points project back into image 2."""
    # Where this fits:
    # Pose is useful only if it explains the image measurements.
    # Reprojection error checks that.
    #
    # Goal:
    # Return mean pixel error in the second image.
    #
    # Input:
    # - `points_3d`: array shaped `[num_points, 3]`
    # - `pose`: estimated pose and inlier points
    # - `calibration`: camera intrinsics
    #
    # Output:
    # - one float: mean reprojection error in pixels
    #
    # Method:
    # Use:
    # - `cv2.projectPoints(...)`
    # - `np.linalg.norm(projected - target_points, axis=1)`
    #
    # Steps:
    # 1. Keep right-image points that match `pose.inlier_mask`.
    # 2. Convert rotation matrix to rotation vector with `cv2.Rodrigues`.
    # 3. Use translation direction scaled by `calibration.baseline_m`.
    # 4. Project `points_3d` into camera 2 with `cv2.projectPoints`.
    # 5. Compare projected pixels with the target right-image pixels.
    # 6. Return the mean pixel distance.
    #
    # Checks:
    # - lower reprojection error means the pose and 3D points explain the images better.
    raise NotImplementedError("Compute reprojection error in pixels.")


def compute_precision_recall(predicted_mask: np.ndarray, labels: torch.Tensor) -> tuple[float, float]:
    """Compute precision and recall for predicted good matches."""
    target = labels.cpu().numpy().astype(bool)
    predicted = predicted_mask.astype(bool)

    true_positive = float(np.logical_and(predicted, target).sum())
    false_positive = float(np.logical_and(predicted, ~target).sum())
    false_negative = float(np.logical_and(~predicted, target).sum())

    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    return precision, recall


def save_training_curve(train_losses: list[float], output_dir: Path) -> Path:
    """Save a simple training curve."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "match_filter_training_curve.png"

    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, marker="o", label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogits loss")
    plt.title("Match Filter Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_pose_visuals(
    data: TwoViewData,
    match_data: MatchData,
    keep_mask: np.ndarray,
    points_3d: np.ndarray,
    output_dir: Path,
) -> dict[str, Path]:
    """Save match and point-cloud visuals."""
    # Where this fits:
    # Pose metrics are easier to trust when you can inspect the matches and 3D points.
    #
    # Goal:
    # Save visual checks for the exercise.
    #
    # Input:
    # - real left and right images
    # - raw match points
    # - learned keep mask
    # - sparse 3D points
    #
    # Output:
    # - dictionary of saved paths
    #
    # Method:
    # Use:
    # - `cv2.drawMatches(...)` for raw and filtered matches
    # - Matplotlib 3D scatter for the sparse point cloud
    #
    # Steps:
    # 1. Rebuild lightweight OpenCV keypoints from point arrays.
    # 2. Rebuild lightweight `cv2.DMatch` objects with matching indices.
    # 3. Save a raw match image.
    # 4. Save a filtered match image.
    # 5. Save a 3D scatter plot of `points_3d`.
    # 6. Return output paths.
    #
    # Checks:
    # - output files should appear in `outputs/two_view_pose_105/`.
    raise NotImplementedError("Save pose estimation visuals.")


def train_two_view_pose_system(config: TwoViewPoseConfig) -> dict[str, object]:
    """Run the full real-data two-view pose exercise."""
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_two_view_data(config)
    points_left, points_right, distances = detect_orb_matches(data, config)
    match_data = build_match_dataset(points_left, points_right, distances, data, config)
    train_loader, val_loader, test_loader = build_dataloaders(
        match_data.features,
        match_data.labels,
        config,
    )

    model = MatchFilterMLP(input_dim=match_data.features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    train_losses: list[float] = []
    for epoch in range(config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(loss)
        val_accuracy = evaluate_match_filter(model, val_loader, device)
        print(f"epoch={epoch + 1:02d} train_loss={loss:.4f} val_accuracy={val_accuracy:.3f}")

    val_accuracy = evaluate_match_filter(model, val_loader, device)
    test_accuracy = evaluate_match_filter(model, test_loader, device)
    keep_mask = predict_good_match_mask(model, match_data.features, device)

    pose = estimate_pose_from_matches(match_data, keep_mask, data.calibration, config)
    points_3d = triangulate_inlier_points(pose, data.calibration)
    reprojection_error = compute_reprojection_error(points_3d, pose, data.calibration)

    precision, recall = compute_precision_recall(keep_mask, match_data.labels)
    metrics = PoseMetrics(
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
        inlier_precision=precision,
        inlier_recall=recall,
        rotation_error_deg=_rotation_error_degrees(
            pose.rotation,
            data.calibration.true_rotation,
        ),
        translation_error_deg=min(
            _angle_between_degrees(
                pose.translation_direction,
                data.calibration.true_translation_direction,
            ),
            _angle_between_degrees(
                -pose.translation_direction,
                data.calibration.true_translation_direction,
            ),
        ),
        mean_reprojection_error_px=reprojection_error,
        num_pose_inliers=int(pose.inlier_mask.sum()),
    )

    curve_path = save_training_curve(train_losses, config.output_dir)
    visual_paths = save_pose_visuals(data, match_data, keep_mask, points_3d, config.output_dir)

    return {
        "metrics": metrics,
        "training_curve": curve_path,
        "visual_paths": visual_paths,
    }


def explain_pose() -> str:
    """Return a short plain-English pose explanation."""
    return (
        "Camera pose means the camera's position and direction. Relative pose means "
        "how one camera view moved compared with another."
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
    """Run the full real-data two-view pose scaffold."""
    config = TwoViewPoseConfig()
    print(explain_pose())
    result, ok = report_task_progress("train_two_view_pose_system", train_two_view_pose_system, config)
    if not ok:
        return

    metrics: PoseMetrics = result["metrics"]  # type: ignore[index]
    print(
        {
            "val_accuracy": metrics.val_accuracy,
            "test_accuracy": metrics.test_accuracy,
            "inlier_precision": metrics.inlier_precision,
            "inlier_recall": metrics.inlier_recall,
            "rotation_error_deg": metrics.rotation_error_deg,
            "translation_error_deg": metrics.translation_error_deg,
            "mean_reprojection_error_px": metrics.mean_reprojection_error_px,
            "num_pose_inliers": metrics.num_pose_inliers,
            "training_curve": str(result["training_curve"]),
            "visual_paths": {key: str(value) for key, value in result["visual_paths"].items()},
        }
    )


if __name__ == "__main__":
    main()
