"""
============================================================
EXERCISE 18: STEREO DEPTH FROM TWO VIEWS
============================================================

WHAT YOU ARE BUILDING
You are building a small stereo-depth training run.

Stereo depth means:
- you have a left camera image
- you have a right camera image
- the two cameras see the same scene from slightly different places
- nearby objects shift more between the two images
- far objects shift less

The shift is called `disparity`.
Once you know disparity, you can turn it into depth.

Plain-language meanings:
- left image: what the left camera saw
- right image: what the right camera saw
- matching point: the same real-world point in both images
- disparity: how many pixels the point moved sideways
- baseline: the real distance between the two cameras
- focal length: the camera zoom amount, measured in pixels here
- depth: how far the point is from the camera

WHY THIS EXERCISE EXISTS
Embodied agents need distance.

Example:
- "the chair is close" changes what action is safe
- "the door is far away" changes how many steps the agent may need
- "the wall is right in front of me" should stop the agent from moving forward

Stereo is a clean first way to learn depth because it has one important formula:

    depth = focal_length * baseline / disparity

What each part means:
- `focal_length` is the camera zoom in pixels
- `baseline` is the gap between the two cameras in meters
- `disparity` is the left x pixel minus the right x pixel
- `depth` is the distance from the camera in meters

Small example:

    focal_length = 100 pixels
    baseline = 0.2 meters
    disparity = 10 pixels
    depth = 100 * 0.2 / 10 = 2 meters

If disparity gets smaller, depth gets larger.
That means points that barely move between the two cameras are far away.

HOW THIS FITS INTO THE BOOTCAMP
the related exercise teaches the camera model:

    3D point -> camera image pixel

This exercise goes the other way:

    left pixel + right pixel -> disparity -> depth

Later exercises use this idea for pose estimation, mapping, SLAM, and 3D reconstruction.

REAL SYSTEM FIRST
In a real stereo-depth system:

1. two calibrated cameras take images at the same time
2. the system finds matching parts between the left and right image
3. it estimates disparity for many pixels
4. it converts disparity to depth with the camera calibration
5. it checks the predicted depth against a trusted depth or disparity map

This exercise uses the same workflow, but with a small model and a small number of image
patches so it can run on a laptop.

DATASET
Use a real stereo dataset, not random tensors and not a made-up scene.

Default dataset:
- Middlebury Stereo 2014
- default scene: Adirondack
- files used:
  - `im0.png`: left image
  - `im1.png`: right image
  - `disp0.pfm`: ground-truth disparity for the left image
  - `calib.txt`: focal length and baseline

The exercise downloads the files from the official Middlebury site when they are not
already in `data/middlebury_2014/Adirondack-perfect/`.

The Middlebury files are large, so this exercise trains on sampled patches, not every
pixel.

WHY A SMALL PATCH MODEL IS USED
Modern stereo networks are large.
They often compare many possible shifts and use convolutional networks over full images.

This exercise uses a smaller learning problem:

    small left patch + matching small right patch -> disparity at the center pixel

This is not a state-of-the-art stereo model.
It exists so you can learn the full data, training, metric, and visualization loop before
using a larger model.

Mapping from the real system to this exercise:

    real system                         this exercise
    ---------------------------------------------------------------
    full left camera image              real Middlebury left image
    full right camera image             real Middlebury right image
    dense disparity map                 real Middlebury `disp0.pfm`
    stereo calibration                  real Middlebury `calib.txt`
    large stereo network                small patch MLP
    full-image disparity prediction     sampled center-pixel disparity prediction
    depth quality report                MAE, RMSE, bad-pixel rate, and depth preview

What is included:
- real stereo image loading
- real ground-truth disparity loading
- train, validation, and test splits
- a small trainable PyTorch model
- disparity metrics
- depth metrics in meters
- saved visual outputs

What is skipped:
- full-image stereo matching
- occlusion handling
- sub-pixel matching details
- large CNN stereo architectures
- camera rectification

Those skipped parts matter in real projects, but they are too much for the first stereo
training exercise.

INPUTS AND OUTPUTS
Each training item is one sampled pixel from the left image.

Input:
- `left_patch`: grayscale tensor shaped `[patch_size, patch_size]`
- `right_patch`: grayscale tensor shaped `[patch_size, patch_size]`
- flattened together into one vector shaped `[2 * patch_size * patch_size]`

Target:
- `disparity`: one number in pixels

Model output:
- `predicted_disparity`: one number in pixels

Depth conversion:
- `predicted_depth_m = focal_length_px * baseline_m / predicted_disparity`
- `target_depth_m = focal_length_px * baseline_m / target_disparity`

LOSS FUNCTION
Train the model with Smooth L1 loss:

    loss = nn.SmoothL1Loss()(predicted_disparity, target_disparity)

Smooth L1 is less jumpy than plain squared error when a few labels are hard.

METRICS TO TRACK
Report these on validation and test data:
- disparity MAE: average absolute disparity error in pixels, lower is better
- disparity RMSE: squared-error style disparity error in pixels, lower is better
- bad-3px rate: fraction of points with more than 3 pixel disparity error, lower is better
- depth MAE: average absolute depth error in meters, lower is better

VISUALS TO SAVE
Save simple images to `outputs/stereo_depth_104/`:
- a downsampled left image
- the ground-truth disparity map
- a sparse predicted disparity preview
- a sparse predicted depth preview
- a training curve

STEP-BY-STEP TODO
1. Read the Middlebury PFM disparity file.
2. Parse the Middlebury calibration file.
3. Sample valid stereo patch pairs from the real image and disparity map.
4. Build a small PyTorch model that predicts disparity from patch pairs.
5. Train for a few epochs.
6. Evaluate disparity and depth metrics.
7. Save visual previews.

DEFINITION OF DONE
You are done when:
- the exercise uses real Middlebury stereo files
- the model trains without shape errors
- validation and test metrics are printed
- at least one training curve and one disparity/depth preview are saved
- you can explain why small disparity means large depth
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
# 1. Make the dataset loader work first.
# 2. Check one sample before building the model.
# 3. Build the small model.
# 4. Train on a small number of patches.
# 5. Evaluate on held-out patches.
# 6. Save visuals last, after the numbers make sense.


@dataclass
class StereoDepthConfig:
    """Settings for the real stereo-depth exercise."""

    scene_name: str = "Adirondack-perfect"
    dataset_root: Path = BOOTCAMP_ROOT / "data" / "middlebury_2014"
    output_dir: Path = BOOTCAMP_ROOT / "outputs" / "stereo_depth_104"
    base_url: str = "https://vision.middlebury.edu/stereo/data/scenes2014/datasets/"

    patch_size: int = 9
    num_samples: int = 8_000
    batch_size: int = 128
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    random_seed: int = 104

    min_disparity: float = 1.0
    max_disparity: float = 280.0
    max_preview_points: int = 3_000


class StereoScene(NamedTuple):
    """One loaded stereo scene."""

    left_gray: np.ndarray
    right_gray: np.ndarray
    disparity: np.ndarray
    focal_length_px: float
    baseline_m: float


class StereoMetrics(NamedTuple):
    """Evaluation numbers for disparity and depth."""

    disparity_mae_px: float
    disparity_rmse_px: float
    bad_3px_rate: float
    depth_mae_m: float


class StereoPatchDataset(Dataset):
    """A real stereo patch dataset sampled from one Middlebury scene."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class StereoPatchMLP(nn.Module):
    """Small model that predicts one center-pixel disparity from two image patches."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Where this fits:
        # The model receives one flattened left patch and one flattened right patch.
        # It predicts one disparity value for the center pixel.
        #
        # Goal:
        # Build a small neural network with `torch.nn`.
        #
        # Input:
        # - `input_dim`: number of values in one feature vector.
        # - for `patch_size=9`, input_dim is `2 * 9 * 9 = 162`.
        #
        # Output:
        # - `self.net`: an `nn.Sequential` model that maps `[batch, input_dim]` to `[batch, 1]`.
        #
        # Method:
        # Use this exact call shape:
        # `nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))`
        #
        # Steps:
        # 1. Create `self.net`.
        # 2. Use two hidden layers.
        # 3. End with one output number.
        #
        # Checks:
        # - `model(torch.randn(4, input_dim)).shape` should be `[4, 1]`.
        raise NotImplementedError("Build the small patch model.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict disparity for a batch of patch features."""
        # Where this fits:
        # The training loop calls this method for every batch.
        #
        # Goal:
        # Run the feature batch through `self.net`.
        #
        # Input:
        # - `features`: tensor shaped `[batch, input_dim]`
        #
        # Output:
        # - tensor shaped `[batch]`
        #
        # Method:
        # Use:
        # `predictions = self.net(features).squeeze(-1)`
        #
        # Steps:
        # 1. Pass `features` into `self.net`.
        # 2. Remove the last size-1 dimension with `.squeeze(-1)`.
        # 3. Return the result.
        #
        # Checks:
        # - the output should have one number per input row.
        raise NotImplementedError("Run the model forward pass.")


def set_seed(seed: int) -> None:
    """Set random seeds so repeated runs are easier to compare."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_middlebury_scene(config: StereoDepthConfig) -> dict[str, Path]:
    """Download the needed Middlebury files if they are missing."""
    scene_dir = config.dataset_root / config.scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    filenames = ["im0.png", "im1.png", "disp0.pfm", "calib.txt"]
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


def read_pfm(path: Path) -> np.ndarray:
    """Read a Middlebury PFM file into a float32 NumPy array."""
    # Where this fits:
    # Middlebury stores the ground-truth disparity map in a `.pfm` file.
    # The model trains against this real disparity map.
    #
    # Goal:
    # Read the PFM image and return a 2D `float32` array.
    #
    # Input:
    # - `path`: path to `disp0.pfm`
    #
    # Output:
    # - NumPy array shaped `[height, width]`
    # - dtype should be `np.float32`
    #
    # Method:
    # Use normal binary file reading:
    # `with path.open("rb") as file:`
    # The PFM header has:
    # 1. first line: `PF` for color or `Pf` for one channel
    # 2. second line: width and height, for example `2880 1988`
    # 3. third line: scale. A negative scale means little-endian floats.
    # 4. then raw 32-bit float values
    #
    # Useful calls:
    # - `file.readline().decode("ascii").strip()`
    # - `np.fromfile(file, endian + "f")`
    # - `array.reshape((height, width))`
    # - `np.flipud(array)` because PFM stores rows from bottom to top
    #
    # Steps:
    # 1. Read and check the first line.
    # 2. Read width and height from the second line.
    # 3. Read the scale from the third line.
    # 4. Pick endian as `"<"` when scale is negative, else `">"`.
    # 5. Read the rest of the file with `np.fromfile`.
    # 6. Reshape to `[height, width]`.
    # 7. Flip vertically with `np.flipud`.
    # 8. Return `array.astype(np.float32)`.
    #
    # Checks:
    # - `disp.ndim` should be `2`.
    # - Most valid values should be positive.
    raise NotImplementedError("Read the Middlebury PFM disparity file.")


def parse_middlebury_calibration(path: Path) -> tuple[float, float]:
    """Read focal length and baseline from a Middlebury `calib.txt` file."""
    # Where this fits:
    # Disparity is measured in pixels.
    # Depth needs the real camera setup too.
    #
    # Goal:
    # Return focal length in pixels and baseline in meters.
    #
    # Input:
    # - `path`: path to `calib.txt`
    #
    # Output:
    # - `(focal_length_px, baseline_m)`
    #
    # Method:
    # Use:
    # - `text = path.read_text(encoding="utf-8")`
    # - `re.search(...)` from Python's `re` module
    #
    # Middlebury calibration includes text like:
    # `cam0=[4161.221 0 1445.577; 0 4161.221 984.686; 0 0 1]`
    # `baseline=176.252`
    #
    # The first number inside `cam0=[...]` is the focal length in pixels.
    # The baseline value is in millimeters, so divide by `1000.0` to get meters.
    #
    # Steps:
    # 1. Read the file text.
    # 2. Use a regex to get the first number after `cam0=[`.
    # 3. Use a regex to get the number after `baseline=`.
    # 4. Convert both strings to floats.
    # 5. Divide baseline by `1000.0`.
    # 6. Return both numbers.
    #
    # Checks:
    # - focal length should be much larger than 1.
    # - baseline should be a positive meter value, often below 1 meter.
    raise NotImplementedError("Parse focal length and baseline from calibration.")


def load_middlebury_scene(config: StereoDepthConfig) -> StereoScene:
    """Load the real stereo images, disparity map, and camera setup."""
    paths = download_middlebury_scene(config)

    left = cv2.imread(str(paths["im0.png"]), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(paths["im1.png"]), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError("OpenCV could not read the Middlebury image files.")

    disparity = read_pfm(paths["disp0.pfm"])
    focal_length_px, baseline_m = parse_middlebury_calibration(paths["calib.txt"])

    left_gray = left.astype(np.float32) / 255.0
    right_gray = right.astype(np.float32) / 255.0

    if left_gray.shape != right_gray.shape or left_gray.shape != disparity.shape:
        raise ValueError(
            "Left image, right image, and disparity map must have the same height and width."
        )

    return StereoScene(
        left_gray=left_gray,
        right_gray=right_gray,
        disparity=disparity,
        focal_length_px=focal_length_px,
        baseline_m=baseline_m,
    )


def disparity_to_depth_m(
    disparity: torch.Tensor,
    focal_length_px: float,
    baseline_m: float,
    min_disparity: float,
) -> torch.Tensor:
    """Convert disparity in pixels to depth in meters."""
    safe_disparity = disparity.clamp_min(min_disparity)
    return focal_length_px * baseline_m / safe_disparity


def sample_stereo_patches(
    scene: StereoScene,
    config: StereoDepthConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample real left/right patch pairs and disparity targets."""
    # Where this fits:
    # This turns the real Middlebury image pair into training rows.
    # Each row is one small left patch and one small right patch.
    #
    # Goal:
    # Build feature and target tensors from real image pixels.
    #
    # Input:
    # - `scene.left_gray`: NumPy array shaped `[height, width]`, values from 0 to 1
    # - `scene.right_gray`: NumPy array shaped `[height, width]`, values from 0 to 1
    # - `scene.disparity`: NumPy array shaped `[height, width]`, disparity in pixels
    #
    # Output:
    # - `features`: tensor shaped `[num_samples, 2 * patch_size * patch_size]`
    # - `targets`: tensor shaped `[num_samples]`
    # - `coords`: tensor shaped `[num_samples, 2]`, with rows `[y, x]`
    #
    # Method:
    # Use NumPy slicing and PyTorch conversion:
    # - `left_patch = scene.left_gray[y - r : y + r + 1, x - r : x + r + 1]`
    # - `right_x = int(round(x - disparity))`
    # - `right_patch = scene.right_gray[y - r : y + r + 1, right_x - r : right_x + r + 1]`
    # - `np.concatenate([left_patch.reshape(-1), right_patch.reshape(-1)])`
    # - `torch.tensor(features, dtype=torch.float32)`
    #
    # Steps:
    # 1. Let `r = config.patch_size // 2`.
    # 2. Find valid pixels where disparity is finite and between
    #    `config.min_disparity` and `config.max_disparity`.
    # 3. Keep only pixels far enough from the image edge for a full left patch.
    # 4. For each candidate pixel, compute `right_x = round(x - disparity)`.
    # 5. Skip the sample if the right patch would go outside the image.
    # 6. Build one feature vector from the left and right patches.
    # 7. Store the target disparity and the `[y, x]` coordinate.
    # 8. Stop when `config.num_samples` samples are collected.
    #
    # Checks:
    # - `features.shape[0] == targets.shape[0] == coords.shape[0]`
    # - `features.shape[1] == 2 * config.patch_size * config.patch_size`
    # - `targets.min()` should be positive.
    raise NotImplementedError("Sample real stereo patches from Middlebury images.")


def build_dataloaders(
    features: torch.Tensor,
    targets: torch.Tensor,
    config: StereoDepthConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split sampled patches into train, validation, and test dataloaders."""
    dataset = StereoPatchDataset(features, targets)
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
    """Train the stereo model for one epoch."""
    # Where this fits:
    # This is the part that changes the model weights.
    #
    # Goal:
    # Run one training pass over the dataloader and return average loss.
    #
    # Input:
    # - `model`: predicts disparity from patch features
    # - `loader`: returns `(features, targets)`
    # - `optimizer`: updates model weights
    # - `loss_fn`: use `nn.SmoothL1Loss()`
    # - `device`: usually `cuda` if available, else `cpu`
    #
    # Output:
    # - one Python float: average training loss
    #
    # Method:
    # Use the normal PyTorch training steps:
    # - `model.train()`
    # - `features = features.to(device)`
    # - `targets = targets.to(device)`
    # - `predictions = model(features)`
    # - `loss = loss_fn(predictions, targets)`
    # - `optimizer.zero_grad()`
    # - `loss.backward()`
    # - `optimizer.step()`
    #
    # Steps:
    # 1. Put the model in train mode.
    # 2. Loop over batches.
    # 3. Move features and targets to the device.
    # 4. Predict disparity.
    # 5. Compute Smooth L1 loss.
    # 6. Clear old gradients.
    # 7. Backprop with `loss.backward()`.
    # 8. Update with `optimizer.step()`.
    # 9. Track the weighted average loss by number of examples.
    #
    # Checks:
    # - loss should usually go down over a few epochs.
    raise NotImplementedError("Train the stereo model for one epoch.")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    scene: StereoScene,
    config: StereoDepthConfig,
    device: torch.device,
) -> StereoMetrics:
    """Evaluate disparity and depth on held-out patches."""
    # Where this fits:
    # Training loss is not enough.
    # This function checks whether predicted disparity gives useful depth.
    #
    # Goal:
    # Report held-out disparity and depth metrics.
    #
    # Input:
    # - `model`: trained stereo patch model
    # - `loader`: validation or test dataloader
    # - `scene`: has focal length and baseline for depth conversion
    #
    # Output:
    # - `StereoMetrics(...)`
    #
    # Method:
    # Use:
    # - `model.eval()`
    # - collect predictions and targets in lists
    # - `torch.cat(...)`
    # - `torch.mean(torch.abs(pred - target))`
    # - `torch.sqrt(torch.mean((pred - target) ** 2))`
    # - `(torch.abs(pred - target) > 3.0).float().mean()`
    # - `disparity_to_depth_m(...)`
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Predict disparity for every batch.
    # 3. Clamp predicted disparity to at least `config.min_disparity`.
    # 4. Compute disparity MAE.
    # 5. Compute disparity RMSE.
    # 6. Compute bad-3px rate.
    # 7. Convert predicted and target disparities to depth in meters.
    # 8. Compute depth MAE.
    # 9. Return a `StereoMetrics` object.
    #
    # Checks:
    # - all metric values should be finite numbers.
    raise NotImplementedError("Evaluate disparity and depth metrics.")


def save_training_curve(train_losses: list[float], output_dir: Path) -> Path:
    """Save a simple training curve."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_curve.png"

    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, marker="o", label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 loss")
    plt.title("Stereo Patch Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


@torch.no_grad()
def save_prediction_previews(
    model: nn.Module,
    scene: StereoScene,
    coords: torch.Tensor,
    features: torch.Tensor,
    config: StereoDepthConfig,
    device: torch.device,
) -> dict[str, Path]:
    """Save left image, target disparity, predicted disparity, and predicted depth previews."""
    # Where this fits:
    # Numbers are useful, but stereo depth is easier to understand when you can see it.
    #
    # Goal:
    # Save small visual previews of the real image and model predictions.
    #
    # Input:
    # - `coords`: tensor shaped `[num_samples, 2]`, rows are `[y, x]`
    # - `features`: tensor shaped `[num_samples, input_dim]`
    #
    # Output:
    # - dictionary from a short name to a saved image path
    #
    # Method:
    # Use:
    # - `model.eval()`
    # - `model(features[:n].to(device))`
    # - `np.full(scene.disparity.shape, np.nan, dtype=np.float32)`
    # - fill sparse preview maps at sampled coordinates
    # - `plt.imshow(...)`
    # - `plt.colorbar(...)`
    # - `plt.savefig(...)`
    #
    # Steps:
    # 1. Pick at most `config.max_preview_points` samples.
    # 2. Predict disparity for those samples.
    # 3. Convert predicted disparity to depth.
    # 4. Make sparse preview arrays filled with `np.nan`.
    # 5. Fill each sampled `[y, x]` location.
    # 6. Save the left image.
    # 7. Save ground-truth disparity.
    # 8. Save sparse predicted disparity.
    # 9. Save sparse predicted depth.
    #
    # Checks:
    # - output files should appear in `outputs/stereo_depth_104/`.
    raise NotImplementedError("Save stereo prediction preview images.")


def train_stereo_depth_model(config: StereoDepthConfig) -> dict[str, object]:
    """Run the full real-data stereo-depth training project."""
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scene = load_middlebury_scene(config)
    features, targets, coords = sample_stereo_patches(scene, config)
    train_loader, val_loader, test_loader = build_dataloaders(features, targets, config)

    model = StereoPatchMLP(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()

    train_losses: list[float] = []
    for epoch in range(config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(loss)
        val_metrics = evaluate_model(model, val_loader, scene, config, device)
        print(
            f"epoch={epoch + 1:02d} "
            f"train_loss={loss:.4f} "
            f"val_disp_mae_px={val_metrics.disparity_mae_px:.3f} "
            f"val_depth_mae_m={val_metrics.depth_mae_m:.3f}"
        )

    test_metrics = evaluate_model(model, test_loader, scene, config, device)
    curve_path = save_training_curve(train_losses, config.output_dir)
    preview_paths = save_prediction_previews(model, scene, coords, features, config, device)

    return {
        "test_metrics": test_metrics,
        "training_curve": curve_path,
        "preview_paths": preview_paths,
    }


def explain_disparity() -> str:
    """Return the key idea in plain English."""
    return (
        "Small disparity means the point moved only a little between the left and right "
        "camera images, so the point is far away."
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
    """Run the full stereo-depth scaffold."""
    config = StereoDepthConfig()
    print(explain_disparity())
    result, ok = report_task_progress("train_stereo_depth_model", train_stereo_depth_model, config)
    if not ok:
        return

    metrics: StereoMetrics = result["test_metrics"]  # type: ignore[index]
    print(
        {
            "test_disparity_mae_px": metrics.disparity_mae_px,
            "test_disparity_rmse_px": metrics.disparity_rmse_px,
            "test_bad_3px_rate": metrics.bad_3px_rate,
            "test_depth_mae_m": metrics.depth_mae_m,
            "training_curve": str(result["training_curve"]),
            "preview_paths": {key: str(value) for key, value in result["preview_paths"].items()},
        }
    )


if __name__ == "__main__":
    main()
