"""
============================================================
EXERCISE 03: V-JEPA VIDEO REPRESENTATION LEARNING
============================================================

WHAT YOU ARE BUILDING
You are building a small V-JEPA-style video representation learner that stays
close to the real workflow while staying small enough for limited GPU memory.

V-JEPA means Video Joint Embedding Predictive Architecture.

Plain meaning:
- load a real video clip
- split the clip into small space-time tokens
- choose which space-time tokens the online encoder is allowed to use
- let an EMA target encoder see the clean clip
- train a predictor to guess the hidden target token representations
- compare hidden vectors, not raw pixels

WHY THIS IS SPLIT OUT
This should not live in the same file as image JEPA.
Video adds time, clip loading, frame order, target tubes, temporal prediction,
and video-level evaluation.

REAL SYSTEM FIRST
In a larger V-JEPA-style system:

1. load real video clips
2. sample a short ordered clip from each video
3. split each frame into patches
4. choose target patch tokens across time
5. keep those target tokens out of the online encoder attention
6. encode the visible video tokens with an online video encoder
7. encode the clean clip with an EMA target encoder
8. predict the target token representations
9. train with a representation-space loss
10. update the target encoder with EMA
11. check whether the learned video features help a downstream video task

EMA means exponential moving average.
Plain meaning:

```text
The target encoder is a slow-moving copy of the online encoder.
After each optimizer step, the target encoder moves a little closer to the online encoder.
```

WHAT IS SCALED DOWN AND WHY
This exercise should stay as close to the real system as possible, but it must
run on limited GPU resources.

Scaled down:
- UCF101 instead of a huge video pretraining dataset
- short clips instead of long videos
- 64 by 64 or 96 by 96 frames instead of high-resolution frames
- 16 by 16 spatial patches instead of many small ViT patches
- a tiny video-transformer encoder instead of a large video transformer
- a tiny transformer predictor instead of the larger V-JEPA predictor
- a short training run instead of long pretraining
- single-process training instead of distributed training

Not scaled down:
- real video clips
- frame order
- space-time target masks
- token-level context masking
- online encoder
- EMA target encoder
- predictor
- latent prediction loss
- feature usefulness check with an action linear probe

DATASET
Use a real video dataset, not generated clips and not random tensors.

Default dataset:
- UCF101 through `torchvision.datasets.UCF101`

Expected local layout:
- videos under `data/ucf101/videos`
- annotation files under `data/ucf101/annotations`

UCF101 is not silently replaced with fake data.
If the dataset is missing, the exercise should explain which path is missing.

The UCF101 action label is not used for the V-JEPA loss.
It is used later for the linear-probe check.

Mapping from the real system to this exercise:

    real V-JEPA system                  this exercise
    ---------------------------------------------------------------
    large video dataset                 UCF101
    long videos                         short ordered clips
    video tokens                        frame-patch tokens
    target tube sampling                small patch blocks tracked across frames
    context video tokens                target tokens hidden from attention
    online video encoder                tiny video-transformer encoder
    EMA target encoder                  slow copy of the online encoder
    predictor                           tiny transformer with learned target mask tokens
    representation loss                 target-token latent MSE and cosine similarity
    feature usefulness check            action linear probe on frozen video features
    visual checks                       frame strips, masks, curves, retrieval heatmap

INPUTS AND OUTPUTS
Each dataloader batch should contain:

    video: [batch, clip_frames, 3, image_size, image_size]
    label: [batch]

With default settings:

    clip_frames = 4
    image_size = 96
    patch_size = 16
    patch grid = 6 by 6
    spatial patches per frame = 36
    total video tokens = 4 * 36 = 144

During training, you create:

    target_mask:  [batch, clip_frames, patches_per_frame]
    visible_mask: [batch, clip_frames, patches_per_frame]

The encoder outputs:

    video_token_latents: [batch, clip_frames, patches_per_frame, latent_dim]

The predictor outputs:

    predicted_token_latents: [batch, clip_frames, patches_per_frame, latent_dim]

The loss only uses the target tokens:

    predicted_target_latents: [batch, target_tokens_per_clip, latent_dim]
    target_latents:           [batch, target_tokens_per_clip, latent_dim]

LOSS FUNCTION
Use latent-space mean squared error on the selected target tokens:

    loss = nn.MSELoss()(predicted_target_latents, target_latents.detach())

Plain meaning:
- the predictor tries to make its hidden vectors close to the target encoder vectors
- `.detach()` means the target vectors are treated like the answer for this update
- the EMA update moves the target encoder slowly after the optimizer step

METRICS TO TRACK
Track:
- train latent MSE, lower is better
- validation latent MSE, lower is better
- test latent MSE, lower is better
- validation cosine similarity, higher is better
- test cosine similarity, higher is better
- batch retrieval top-1, higher is better
- action linear-probe accuracy, higher is better

Batch retrieval top-1 means:
- average each predicted target token set into one vector per clip
- average each target token set into one vector per clip
- compare every predicted vector with every target vector in the same batch
- count it as correct if row `i` chooses column `i`

VISUALS TO SAVE
Save to `outputs/vjepa_video/`:
- original clip strip
- masked clip strip for visual checks
- target mask strip
- training curve
- retrieval similarity heatmap for one batch

STEP-BY-STEP TODO
1. Load UCF101 clips with `frames_per_clip = config.clip_frames`.
2. Convert clips to `[time, channels, height, width]` float tensors.
3. Create target tube masks across frames and patches.
4. Create visible token masks from target masks.
5. Build the online tiny video-transformer encoder.
6. Build the EMA target video token encoder.
7. Build the tiny transformer predictor.
8. Train on target-token latent MSE.
9. Update the target encoder with EMA after each optimizer step.
10. Report held-out latent metrics and retrieval top-1.
11. Train an action linear probe on frozen video features.
12. Save visual checks.

DEFINITION OF DONE
You are done when:
- the exercise uses real video clips
- frame order is preserved
- the model uses space-time target masks
- the target encoder is updated with EMA
- validation and test latent metrics are printed
- retrieval top-1 is printed
- action linear-probe accuracy is printed
- visuals are saved
- you can explain why hidden video tokens are predicted in representation space
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed


# Learning order:
# 1. Load real UCF101 clips.
# 2. Create target space-time masks.
# 3. Build the visible token mask for the online encoder.
# 4. Build the online encoder, EMA target encoder, and predictor.
# 5. Train on target-token latent MSE.
# 6. Update the target encoder with EMA.
# 7. Evaluate latent metrics, retrieval, and action linear-probe accuracy.
# 8. Save visuals.


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VJEPAConfig:
    """Settings for the small V-JEPA run."""

    video_root: str = str(BOOTCAMP_ROOT / "data" / "ucf101" / "UCF-101")
    annotation_root: str = str(BOOTCAMP_ROOT / "data" / "ucf101" / "ucfTrainTestlist")
    output_dir: str = str(BOOTCAMP_ROOT / "outputs" / "vjepa_video")
    image_size: int = 96
    clip_frames: int = 4
    patch_size: int = 16
    latent_dim: int = 128
    encoder_depth: int = 2
    encoder_num_heads: int = 4
    encoder_mlp_dim: int = 256
    predictor_depth: int = 1
    predictor_num_heads: int = 4
    predictor_mlp_dim: int = 256
    num_target_tubes: int = 4
    target_tube_frames: int = 2
    target_tube_side: int = 2
    batch_size: int = 8
    learning_rate: float = 1e-4
    probe_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    linear_probe_epochs: int = 5
    ema_decay: float = 0.99
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    random_state: int = 302

    def __post_init__(self) -> None:
        """Catch config values that would break the video transformer."""
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must divide evenly by patch_size.")
        if self.latent_dim % self.encoder_num_heads != 0:
            raise ValueError("latent_dim must divide evenly by encoder_num_heads.")
        if self.latent_dim % self.predictor_num_heads != 0:
            raise ValueError("latent_dim must divide evenly by predictor_num_heads.")
        if self.target_tube_frames > self.clip_frames:
            raise ValueError("target_tube_frames must be no larger than clip_frames.")
        if self.target_tube_side > self.patches_per_side:
            raise ValueError("target_tube_side must be no larger than patches_per_side.")
        if self.target_tokens_per_clip >= self.total_tokens:
            raise ValueError("target_tokens_per_clip must be smaller than total_tokens.")
        possible_start_frames = self.clip_frames - self.target_tube_frames + 1
        possible_start_rows = self.patches_per_side - self.target_tube_side + 1
        possible_start_cols = self.patches_per_side - self.target_tube_side + 1
        if possible_start_frames * possible_start_rows * possible_start_cols < self.num_target_tubes:
            raise ValueError("num_target_tubes is too large for the current clip and patch grid.")

    @property
    def patches_per_side(self) -> int:
        """Return how many spatial patches fit along one frame side."""
        return self.image_size // self.patch_size

    @property
    def patches_per_frame(self) -> int:
        """Return the number of spatial patches in one frame."""
        return self.patches_per_side * self.patches_per_side

    @property
    def total_tokens(self) -> int:
        """Return the total number of space-time tokens in one clip."""
        return self.clip_frames * self.patches_per_frame

    @property
    def target_tokens_per_tube(self) -> int:
        """
            Return how many tokens one target tube contains.
            A tube is a small block of video patches that continues across nearby frames.
        """
        return self.target_tube_frames * self.target_tube_side * self.target_tube_side

    @property
    def target_tokens_per_clip(self) -> int:
        """Return how many target tokens each clip should contain."""
        return self.num_target_tubes * self.target_tokens_per_tube


class UCFJEPAVideoDataset(Dataset):
    """Wrap UCF101 so one item has a clear video tensor and label."""

    def __init__(self, base_dataset: Dataset, image_size: int) -> None:
        self.base_dataset = base_dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clip, _, label = self.base_dataset[index]
        # Torchvision UCF101 often returns `[T, H, W, C]`.
        # The training code wants `[T, C, H, W]`.
        if clip.ndim != 4:
            raise ValueError("Expected one video clip shaped [T, H, W, C] or [T, C, H, W].")
        if clip.shape[-1] == 3:
            clip = clip.permute(0, 3, 1, 2)
        clip = clip.float() / 255.0
        clip = F.interpolate(
            clip,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return {
            "video": clip,
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_ucf101_vjepa_dataloaders(config: VJEPAConfig) -> dict[str, DataLoader]:
    """Load real UCF101 clips and return train, validation, and test loaders."""
    # Where this fits:
    # This builds the real video data for the V-JEPA-style task.
    #
    # Goal:
    # Return batches with:
    # - `video`: `[batch, clip_frames, 3, image_size, image_size]`
    # - `label`: `[batch]`
    #
    # Method:
    # Use torchvision:
    # - `from torchvision.datasets import UCF101`
    # - `frames_per_clip = config.clip_frames`
    # - `UCF101(root=config.video_root, annotation_path=config.annotation_root, frames_per_clip=frames_per_clip)`
    #
    # Resize note:
    # The wrapper dataset converts each clip to `[T, C, H, W]` and uses
    # `torch.nn.functional.interpolate(...)` to resize all frames.
    #
    # Steps:
    # 1. Check that `config.video_root` exists.
    # 2. Check that `config.annotation_root` exists.
    # 3. Load UCF101 with `frames_per_clip = config.clip_frames`.
    # 4. Wrap it with `UCFJEPAVideoDataset(base_dataset, config.image_size)`.
    # 5. Split into train, validation, and test.
    # 6. Build DataLoaders.
    #
    # Checks:
    # - one batch should preserve frame order.
    # - `video.shape` should be `[batch, clip_frames, 3, image_size, image_size]`.
    # - if the dataset is missing, raise a clear `FileNotFoundError`.
    video_root = Path(config.video_root)
    annotation_root = Path(config.annotation_root)
    if not video_root.exists():
        raise FileNotFoundError(f"UCF101 video folder not found: {video_root}")
    if not annotation_root.exists():
        raise FileNotFoundError(f"UCF101 annotation folder not found: {annotation_root}")

    from torchvision.datasets import UCF101

    base_dataset = UCF101(
        root=str(video_root),
        annotation_path=str(annotation_root),
        frames_per_clip=config.clip_frames,
    )

    dataset = UCFJEPAVideoDataset(base_dataset, config.image_size)

    total_size = len(dataset)
    train_size = int(total_size * config.train_fraction)
    val_size = int(total_size * config.val_fraction)
    test_size = total_size - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            "The UCF101 split is too small. "
            f"Got train={train_size}, val={val_size}, test={test_size} from {total_size} clips."
        )

    gen = torch.Generator().manual_seed(config.random_state)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, val_size, test_size],
        generator=gen,
    )

    return {
        "train": DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=config.batch_size),
        "test": DataLoader(test_ds, batch_size=config.batch_size),
    }

    # raise NotImplementedError("Load real UCF101 clips for V-JEPA.")


def make_target_token_mask(
    batch_size: int,            # 8
    clip_frames: int,           # 4
    patches_per_frame: int,     # 36
    num_target_tubes: int,      # 4
    target_tube_frames: int,    # 2
    target_tube_side: int,      # 2
    device: str | torch.device,
) -> torch.Tensor:
    """Create a boolean mask for target space-time tubes."""
    # Where this fits:
    # Real V-JEPA systems choose target blocks or tubes across time before
    # the online encoder runs. A tube means the same small spatial patch block
    # over a few neighboring frames.
    #
    # Goal:
    # Return `target_mask` with shape `[batch_size, clip_frames, patches_per_frame]`.
    # `True` means "this video token is inside a target tube to predict".
    #
    # Small concrete example:
    # With `clip_frames=4`, `patches_per_frame=36`, `patches_per_side=6`,
    # `target_tube_frames=2`, and `target_tube_side=2`, one tube might cover:
    #
    # ```text
    # frame 1, patch rows 2..3, patch cols 4..5
    # frame 2, patch rows 2..3, patch cols 4..5
    # ```
    #
    # That one tube contains `2 frames * 2 rows * 2 cols = 8` target tokens.
    #
    # Method:
    # First compute:
    # - `patches_per_side = int(patches_per_frame ** 0.5)`
    # - `target_tokens_per_tube = target_tube_frames * target_tube_side * target_tube_side`
    # - `target_tokens_per_clip = num_target_tubes * target_tokens_per_tube`
    # - `max_attempts = num_target_tubes * 20`
    #   `max_attempts` is a safety limit. If random tube choices keep
    #   overlapping, this stops the loop from trying forever.
    #
    # Check that `patches_per_side * patches_per_side == patches_per_frame`.
    # If not, raise `ValueError` because the mask code cannot make a square
    # patch grid.
    #
    # Steps:
    # 1. Create a boolean tensor full of `False`.
    # 2. For each batch row, keep sampling tubes until it has exactly
    #    `target_tokens_per_clip` target tokens.
    # 3. Sample a start frame from:
    #    `0` to `clip_frames - target_tube_frames`.
    # 4. Sample a start patch row and column from:
    #    `0` to `patches_per_side - target_tube_side`.
    # 5. Convert each row and column inside the tube to a patch index:
    #    `patch_index = patch_row * patches_per_side + patch_col`.
    # 6. Before adding a tube, check whether it overlaps an already selected
    #    target token. If it overlaps, skip it and sample another tube.
    # 7. Set all tokens in the accepted tube to `True`.
    # 8. If `max_attempts` is reached before enough non-overlapping tubes are
    #    found, raise `RuntimeError` with a clear message.
    #
    # Checks:
    # - each row should have exactly `target_tokens_per_clip` true values.
    # - each accepted tube should cover neighboring frames, rows, and columns.
    # - the mask should include both time and spatial dimensions.
    #


    # one clip is 4 frames here, so each dataloader batch should have videos shaped like [batch, 4, 3, image_size, image_size]
    # With the default image size: [batch, 4, 3, 96, 96]

    # clip frames just says how many frames exist in the whole clip. It does not determine how many blocks should be hidden
    # target_tube_frames = how many frames each tube covers
    # num_target_tubes = how many separate tubes to hide

    # So with defaults:
    # target_tube_frames = 2
    # target_tube_side = 2
    # num_target_tubes = 4
    # One tube has: target_tokens_per_tube = 2 * 2 * 2 = 8
    # Four tubes has target_tokens_per_clip = 4 * 8 = 32  i.e. num_target_tubes * target_tokens_per_tube

    # If we were to compute total tokens per clip based on the given defaults for image_size and patch_size, i.e. image_size = 96, patch_size = 16
    # there are 96/16 = 6  patches per side and 6x6 = 36 patches per frame i.e. 36 tokens per frame
    # Then total_tokens = clip_frames * patches_per_frame = 4 * 36 = 144  i.e. total tokens across all frames of a single video.
    # so we are hiding 32 target tokens out of 144 total video tokens
    
    # But we're given patches per frame as a parameter, so we take the square root to find patches per side:
    patches_per_side = int(patches_per_frame ** 0.5)  # i.e sqrt(36) = 6
    target_tokens_per_tube = target_tube_frames * target_tube_side * target_tube_side # target_tube_side * target_tube_side = 2 x 2 tokens per frame  # 8
    target_tokens_per_clip = num_target_tubes * target_tokens_per_tube #32
    max_attempts = num_target_tubes * 20 

    if patches_per_side * patches_per_side != patches_per_frame:
        raise ValueError("Cannot make a square patch grid")

    # Patch size versus tube size:
    # `patch_size` says how large one token is in pixels.
    # For example, `patch_size=16` means one video token is one `16 by 16`
    # patch from one frame.
    #
    # `target_tube_side` says how many neighboring patch tokens are hidden
    # per side inside each frame.
    # For example, `target_tube_side=2` means the hidden block is `2 by 2`
    # patch tokens in each selected frame.
    #
    # Example with `patch_size=16`:
    #
    # ```text
    # target_tube_side = 1  -> hide 1 patch, 16x16 pixels
    # target_tube_side = 2  -> hide 2x2 patches, 32x32 pixels
    # target_tube_side = 3  -> hide 3x3 patches, 48x48 pixels
    # ```
    #
    # `target_tube_frames` says how many nearby frames the hidden block covers.
    # For example, with:
    #
    # ```text
    # patch_size = 16
    # target_tube_side = 2
    # target_tube_frames = 2
    # ```
    #
    # one video token is one `16 by 16` patch in one frame.
    # `target_tube_side=2` means `2 by 2 = 4` video tokens per frame.
    # across `2` frames, that means one tube hides `2 * 4 = 8` video tokens.
    #
    # Plain meaning:
    # - `patch_size` controls token size
    # - `target_tube_side` controls hidden block size in patch units
    # - `target_tube_frames` controls hidden block length through time
    #
    # In the image JEPA exercise, one hidden patch can feel like the whole
    # hidden block because the exercise is very small.
    # Here, a hidden block can contain several patch tokens across space and time.

    target_mask = torch.zeros(size=[batch_size, clip_frames, patches_per_frame], dtype=torch.bool, device=device)
    # True = Target token to hide from the context encoder and predict later in the predictor
    # False = Visible token that the context encoder will be allowed to see

    for batch_index in range(batch_size):
        count = 0
        attempts = 0
        while count < num_target_tubes:
            # size=() means sample 1 scalar value
            # use .item because the python range function below expects normal python ints and not tensor scalars
            sample_start_frame = torch.randint(low=0, high=clip_frames - target_tube_frames + 1, size=(), device=device).item()       # 0 to (4-2) + 1  [0,3) so 0,1,2 with 3 excluded
            sample_start_patch_row = torch.randint(low = 0, high = patches_per_side - target_tube_side + 1, size=(), device=device).item()     # 0 to (6-2) + 1
            sample_start_patch_col = torch.randint(low = 0, high = patches_per_side - target_tube_side + 1, size=(), device=device).item()     # 0 to (6-2) + 1


            tube_positions = []

            for frame_index in range(sample_start_frame, sample_start_frame + target_tube_frames):
                for patch_row in range(sample_start_patch_row, sample_start_patch_row + target_tube_side):
                    for patch_col in range(sample_start_patch_col, sample_start_patch_col + target_tube_side):            
                        
                        patch_index = patch_row * patches_per_side + patch_col                # eg. 0 * 6 + 1
                        tube_positions.append((frame_index, patch_index))
                        # Each item tells you:
                        # which frame
                        # which patch inside that frame

            attempts+=1        
            if attempts == max_attempts:
                raise RuntimeError("Message")

            has_overlap = any(target_mask[batch_index, frame_index, patch_index] for frame_index, patch_index in tube_positions)
            if has_overlap:
                continue
            
            for frame_index, patch_index in tube_positions:
                target_mask[batch_index, frame_index, patch_index] = True

            count+=1
    
    return target_mask

    #raise NotImplementedError("Create target space-time tube masks.")


def apply_video_token_mask(
    videos: torch.Tensor,
    target_mask: torch.Tensor,
    patch_size: int,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """Return a visual copy of videos with target patches hidden."""
    # Where this fits:
    # This helper is mainly for visual checks.
    # The model should use token-level masking in `TinyVideoTransformerEncoder.forward`.
    # The masked clip is still useful because it lets you see which patches
    # were hidden from the online encoder.
    #
    # Goal:
    # Return `masked_videos` with the same shape as `videos`.
    # Every target patch in every target frame should be filled with `mask_value`.
    #
    # Input:
    # - `videos`: `[batch, clip_frames, 3, image_size, image_size]`
    # - `target_mask`: `[batch, clip_frames, patches_per_frame]`
    #
    # Method:
    # Treat each frame as a grid of patches.
    # For `image_size=96` and `patch_size=16`, each frame has a 6 by 6 grid.
    #
    # Steps:
    # 1. Clone `videos` so the original clean videos are kept.
    # 2. Loop over frames, patch rows, and patch columns.
    # 3. Convert each patch position to a patch index.
    # 4. For videos where `target_mask[:, frame_index, patch_index]` is true,
    #    fill that frame patch with `mask_value`.
    #
    # Batch selector example:
    # Suppose the batch has 4 videos.
    # For one frame and one patch, this line:
    #
    # `batch_selector = target_mask[:, frame_index, patch_index]`
    #
    # might give:
    #
    # ```text
    # [False, True, False, True]
    # ```
    #
    # Plain meaning:
    # - video 0: do not hide this patch
    # - video 1: hide this patch
    # - video 2: do not hide this patch
    # - video 3: hide this patch
    #
    # Then this indexing:
    #
    # `masked_videos[batch_selector, frame_index, :, row_start:row_end, col_start:col_end] = mask_value`
    #
    # changes only videos 1 and 3 for this one frame patch.
    #
    # Checks:
    # - output shape should match input shape.
    # - non-target tokens should stay unchanged.

    masked_videos = videos.clone()   # [batch, clip_frames, 3, image_size, image_size]

    batch_size, clip_frames, channels, image_height, image_width = videos.shape

    patches_per_side = image_height // patch_size

    #videos:      [batch, clip_frames, 3, image_size, image_size]
    #target_mask: [batch, clip_frames, patches_per_frame]

    for frame_index in range(clip_frames):
        for patch_row in range(patches_per_side):
            for patch_col in range(patches_per_side):

                patch_index = patch_row * patches_per_side + patch_col

                row_start = patch_row * patch_size
                row_end = row_start + patch_size
                col_start = patch_col * patch_size
                col_end = col_start + patch_size

                batch_selector = target_mask[:, frame_index, patch_index]  # shape here is [batch] and it is a boolean mask.
                masked_videos[batch_selector, frame_index, :, row_start:row_end, col_start:col_end] = mask_value
                # this means for videos where this patch is a target, in this frame, across all 3 color channels, fill this patch area
                # Example: if batch_selector is [False, True, False, True],
                # only videos 1 and 3 get this patch filled with mask_value.
                #
                # Read the indexing from left to right:
                # - `batch_selector`: choose only the videos where this patch is a target
                # - `frame_index`: choose this one frame in those videos
                # - `:`: choose all 3 color channels
                # - `row_start:row_end`: choose the pixel rows for this patch
                # - `col_start:col_end`: choose the pixel columns for this patch
                #
                # So if `batch_selector = [False, True, False, True]`, this one line is like:
                # `masked_videos[1, frame_index, :, row_start:row_end, col_start:col_end] = mask_value`
                # `masked_videos[3, frame_index, :, row_start:row_end, col_start:col_end] = mask_value`
                #
                # Plain meaning:
                # `target_mask` marks hidden tokens.
                # This line turns those hidden token positions into hidden pixel patches for the visual check.
        

    return masked_videos
    #raise NotImplementedError("Create masked clips for V-JEPA visual checks.")


class TinyVideoTransformerEncoder(nn.Module):
    """Tiny video-transformer encoder that returns one latent per frame patch."""

    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()
        # Where this fits:
        # This class is used for both the online encoder and EMA target encoder.
        # In a larger V-JEPA system, this role is played by a video transformer.
        # This is a much smaller version.
        #
        # Goal:
        # Turn `[batch, clip_frames, 3, image_size, image_size]` videos into
        # `[batch, clip_frames, patches_per_frame, latent_dim]` token latents.
        #
        # Method:
        # Build these parts:
        # - `self.clip_frames = config.clip_frames`
        # - `self.patches_per_frame = config.patches_per_frame`
        # - `self.total_tokens = config.total_tokens`
        # - `self.latent_dim = config.latent_dim`
        # - `self.patch_embed = nn.Conv2d(config.channels, config.latent_dim, kernel_size=config.patch_size, stride=config.patch_size)`
        # - `self.pos_embed = nn.Parameter(torch.zeros(1, config.total_tokens, config.latent_dim))`
        # - `self.hidden_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))`
        # - `encoder_layer = nn.TransformerEncoderLayer(...)`
        # - `self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_depth)`
        # - `self.norm = nn.LayerNorm(config.latent_dim)`
        #
        # Use this exact transformer layer shape:
        # `nn.TransformerEncoderLayer(`
        # `    d_model=config.latent_dim,`
        # `    nhead=config.encoder_num_heads,`
        # `    dim_feedforward=config.encoder_mlp_dim,`
        # `    dropout=0.0,`
        # `    activation="gelu",`
        # `    batch_first=True,`
        # `)`
        #
        # What each part means:
        # - `patch_embed` turns each frame patch into one token vector.
        # - `pos_embed` tells the transformer both frame position and patch position.
        # - `hidden_token` fills target positions for the online encoder.
        # - `TransformerEncoder` lets tokens share information across space and time.
        # - `LayerNorm` keeps the output scale steadier.
        #
        # Checks:
        # - `TinyVideoTransformerEncoder(config)(torch.randn(2, 4, 3, 96, 96)).shape`
        #   should be `[2, 4, 36, config.latent_dim]`.

        self.clip_frames = config.clip_frames                   # 4
        self.patches_per_frame = config.patches_per_frame       # 36 (6x6)
        self.total_tokens = config.total_tokens                 # 144 (36 x 4 frames)
        self.latent_dim = config.latent_dim                     # 128

        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=config.latent_dim, kernel_size = config.patch_size, stride=config.patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, config.total_tokens, config.latent_dim))  #[1, 144, 128] i.e. 144 tokens each with latent_dim 128

        self.hidden_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))  # Learned hidden token vector

        encoder_layer = nn.TransformerEncoderLayer(d_model = config.latent_dim, 
                                                        nhead=config.encoder_num_heads, 
                                                        dim_feedforward=config.encoder_mlp_dim,
                                                        dropout=0.0,
                                                        activation='gelu',
                                                        batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = config.encoder_depth)
        
        self.norm = nn.LayerNorm(config.latent_dim)

        #raise NotImplementedError("Build the tiny video-transformer encoder.")

    def forward(self, videos: torch.Tensor, visible_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode videos into space-time token latents."""
        # Goal:
        # Run the tiny video transformer.
        # If `visible_mask` is given, target tokens should be hidden from the
        # online encoder at the token level.
        #
        # Input:
        # - `videos`: `[batch, clip_frames, 3, image_size, image_size]`
        # - `visible_mask`: `[batch, clip_frames, patches_per_frame]`, where
        #   `True` means "the online encoder may use this video token".
        #
        # Steps:
        # 1. Flatten videos from `[batch, time, 3, H, W]` to
        #    `[batch * time, 3, H, W]`.
        # 2. Run `patch_tokens = self.patch_embed(flat_videos)`.
        #    Shape: `[batch * time, latent_dim, 6, 6]` with default settings.
        # 3. Flatten the patch grid with:
        #    `patch_tokens = patch_tokens.flatten(2).transpose(1, 2)`.
        #    Shape: `[batch * time, patches_per_frame, latent_dim]`.
        # 4. Reshape to `[batch, total_tokens, latent_dim]`.
        # 5. If `visible_mask is not None`, flatten it to `[batch, total_tokens]`
        #    and replace hidden target positions with `self.hidden_token`.
        # 6. Add `self.pos_embed`.
        # 7. Run `self.transformer(...)`.
        #    If `visible_mask is not None`, pass:
        #    `src_key_padding_mask=~flat_visible_mask`.
        #    In this PyTorch mask, `True` means "ignore this token as a key".
        # 8. Apply `self.norm`.
        # 9. Reshape back to `[batch, clip_frames, patches_per_frame, latent_dim]`.
        #
        # Why this is closer to V-JEPA:
        # The online encoder is guided by video tokens and a context mask.
        # It is not mainly learning from a pixel-masked clip.


        batch_size, frames_per_clip, channels, height, width = videos.shape  # [batch_size, clip_frames, 3, image_size, image_size]

        flattened_videos = videos.reshape(shape=[batch_size * frames_per_clip, channels, height, width])  # [batch_size * clip_frames, 3, image_size, image_size]

        patch_tokens = self.patch_embed(flattened_videos)       # [batch_size * clip_frames, latent_dim, patches_per_side, patches_per_side]
        patch_tokens = patch_tokens.flatten(start_dim=2)        # [batch_size * clip_frames, latent_dim, patches_per_frame]
        patch_tokens = patch_tokens.transpose(-2, -1)           # [batch_size * clip_frames, patches_per_frame, latent_dim]
        patch_tokens = patch_tokens.reshape(shape=[batch_size, frames_per_clip * self.patches_per_frame, self.latent_dim]) # [batch_size, total_tokens, latent_dim]
        # now patch_tokens is shape [batch, total_tokens, latent_dim]

        flat_visible_mask = None
        if visible_mask is not None: #[batch, clip_frames, patches_per_frame]
            flat_visible_mask = visible_mask.reshape(batch_size, frames_per_clip * self.patches_per_frame)  # [batch_size, total_tokens]
            patch_tokens = torch.where(condition=flat_visible_mask.unsqueeze(dim=-1), input=patch_tokens, other=self.hidden_token)  # [batch_size, total_tokens, latent_dim]
            # `visible_mask` uses True for patches the online encoder may read.

        patch_tokens = patch_tokens + self.pos_embed        # [batch_size, total_tokens, latent_dim] + [1, total_tokens, latent_dim] = [batch_size, total_tokens, latent_dim]

        # if its none, this will be for the target encoder path
        if flat_visible_mask is None:
            patch_tokens = self.transformer(patch_tokens)  # [batch_size, total_tokens, latent_dim] -> [batch_size, total_tokens, latent_dim]
        else:
            # `visible_mask` uses True for patches the online encoder may read.
            # PyTorch's `src_key_padding_mask` uses True for tokens it should ignore.
            # That is why we use `~visible_mask`: it flips visible patches to False
            # and hidden target patches to True.
            # This means hidden target patches do not act as source tokens for attention.
            patch_tokens = self.transformer(patch_tokens, src_key_padding_mask=~flat_visible_mask)  # [batch_size, total_tokens, latent_dim] -> [batch_size, total_tokens, latent_dim]
            #
            # A small picture of what happens inside attention:
            #
            # ```python
            # x = patch_tokens  # [batch_size, total_tokens, latent_dim]
            # query = W_q(x)    # [batch_size, total_tokens, head_dim]
            # key = W_k(x)      # [batch_size, total_tokens, head_dim]
            # value = W_v(x)    # [batch_size, total_tokens, head_dim]
            #
            # scores = query @ key.transpose(-2, -1)
            # # [batch_size, total_tokens, total_tokens]
            # # scores[b, i, j] means: how much token i wants to look at token j
            #
            # key_block_mask = (~flat_visible_mask).unsqueeze(1)
            # # [batch_size, 1, total_tokens]
            # # True means this token is blocked as a key.
            #
            # scores = scores.masked_fill(key_block_mask, float("-inf"))
            # weights = torch.softmax(scores, dim=-1)
            # # Blocked key positions become almost zero after softmax.
            # output = weights @ value
            # # [batch_size, total_tokens, head_dim]
            # ```
            #
            # Plain meaning:
            # hidden target tokens can still have output vectors,
            # but they cannot be used as information sources by other tokens.

        patch_tokens = self.norm(patch_tokens)  # [batch_size, total_tokens, latent_dim] -> [batch_size, total_tokens, latent_dim]

        patch_tokens = patch_tokens.reshape(
            batch_size,
            frames_per_clip,
            self.patches_per_frame,
            self.latent_dim,
        )  # [batch_size, clip_frames, patches_per_frame, latent_dim]

        return patch_tokens
        #raise NotImplementedError("Run the tiny video-transformer encoder forward pass.")


class TinyVJEPAPredictor(nn.Module):
    """Tiny transformer that predicts target video token latents."""

    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()
        # Where this fits:
        # In V-JEPA, the predictor is a transformer-like module. It receives
        # context token representations and learned mask tokens for target
        # positions, then predicts the target token representations.
        #
        # Goal:
        # Build a small transformer predictor.
        #
        # Method:
        # Build these parts:
        # - `self.clip_frames = config.clip_frames`
        # - `self.patches_per_frame = config.patches_per_frame`
        # - `self.total_tokens = config.total_tokens`
        # - `self.latent_dim = config.latent_dim`
        # - `self.mask_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))`
        # - `self.pos_embed = nn.Parameter(torch.zeros(1, config.total_tokens, config.latent_dim))`
        # - `encoder_layer = nn.TransformerEncoderLayer(...)`
        # - `self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.predictor_depth)`
        # - `self.norm = nn.LayerNorm(config.latent_dim)`
        # - `self.proj = nn.Linear(config.latent_dim, config.latent_dim)`
        #
        # Use this exact transformer layer shape:
        # `nn.TransformerEncoderLayer(`
        # `    d_model=config.latent_dim,`
        # `    nhead=config.predictor_num_heads,`
        # `    dim_feedforward=config.predictor_mlp_dim,`
        # `    dropout=0.0,`
        # `    activation="gelu",`
        # `    batch_first=True,`
        # `)`
        #
        # Input:
        # - `context_token_latents`: `[batch, time, patches_per_frame, latent_dim]`
        # - `target_mask`: `[batch, time, patches_per_frame]`
        #
        # Output:
        # - `predicted_token_latents`: `[batch, time, patches_per_frame, latent_dim]`

        self.clip_frames = config.clip_frames                   # 4
        self.patches_per_frame = config.patches_per_frame       # 36  (6x6)
        self.total_tokens = config.total_tokens                 # 144 (36 x 4 frames)
        self.latent_dim = config.latent_dim                     # 128
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))      # [1, 1, 128]

        self.pos_embed = nn.Parameter(torch.zeros(1, config.total_tokens, config.latent_dim))  # [1, 144, 128]

        encoder_layer = nn.TransformerEncoderLayer(d_model = config.latent_dim,
                                                    nhead=config.predictor_num_heads,
                                                    dim_feedforward=config.predictor_mlp_dim,
                                                    dropout=0.0,
                                                    activation='gelu',
                                                    batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = config.predictor_depth)

        self.norm = nn.LayerNorm(config.latent_dim)

        self.proj = nn.Linear(in_features = config.latent_dim, out_features = config.latent_dim)

        #raise NotImplementedError("Build the tiny V-JEPA predictor.")

    def forward(self, context_token_latents: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """Predict video token latents from context token latents."""
        # Goal:
        # Predict one latent vector for every video token. The loss will later
        # keep only the target token positions.
        #
        # Steps:
        # 1. Flatten `context_token_latents` to `[batch, total_tokens, latent_dim]`.
        # 2. Flatten `target_mask` to `[batch, total_tokens]`.
        # 3. Expand `self.mask_token` to `[batch, total_tokens, latent_dim]`.
        # 4. Use `torch.where(...)` so target positions get the mask token and
        #    non-target positions keep their context latents:
        #    `tokens = torch.where(flat_target_mask.unsqueeze(-1), mask_tokens, flat_context_latents)`.
        # 5. Add `self.pos_embed`.
        # 6. Run the predictor transformer.
        # 7. Apply `self.norm` and `self.proj`.
        # 8. Reshape back to `[batch, clip_frames, patches_per_frame, latent_dim]`.
        #
        # Check:
        # - output shape should match `context_token_latents.shape`.


        #context_token_latents: [batch, clip_frames, patches_per_frame, latent_dim]
        batch_size, clip_frames, patches_per_frame, latent_dim = context_token_latents.shape  # [batch_size, clip_frames, patches_per_frame, latent_dim]
        #note: self.total_tokens should be the same as clip_frames*patches_per_frame
        flat_context_token_latents = context_token_latents.reshape(shape=[batch_size, self.total_tokens, latent_dim])  # [batch_size, total_tokens, latent_dim]

        flat_target_mask = target_mask.reshape(shape=[batch_size, self.total_tokens])   # [batch_size, frames_per_clip, patches_per_Frame] -> [batch_size, total_tokens]
        
        mask_tokens = self.mask_token.expand(size=[batch_size, self.total_tokens, self.latent_dim]) # [1, 1, latent_dim] -> [batch_size, total_tokens, latent_dim]

        # `flat_target_mask` starts with shape `[batch_size, total_tokens]`.
        # With the default config, that is `[batch_size, 144]`.
        # `flat_target_mask.unsqueeze(-1)` changes it to `[batch_size, 144, 1]`.
        # The final `1` lets PyTorch copy each True/False value across the full
        # latent vector for that video token.
        # If `flat_target_mask` is True, this uses the learned `mask_token`.
        # If `flat_target_mask` is False, this keeps the real context token latent.
        # `mask_token` means this is a target position the predictor should fill in.
        tokens = torch.where(flat_target_mask.unsqueeze(-1), mask_tokens, flat_context_token_latents)  # [batch_size, total_tokens, latent_dim]

        tokens = tokens + self.pos_embed  # [batch_size, total_tokens, latent_dim] + [1, total_tokens, latent_dim] = [batch_size, total_tokens, latent_dim]

        tokens = self.transformer(tokens)  # [batch_size, total_tokens, latent_dim] -> [batch_size, total_tokens, latent_dim]
        tokens = self.norm(tokens)         # [batch_size, total_tokens, latent_dim] -> [batch_size, total_tokens, latent_dim]
        tokens = self.proj(tokens)         # [batch_size, total_tokens, latent_dim] -> [batch_size, total_tokens, latent_dim]
        tokens = tokens.reshape(batch_size, clip_frames, patches_per_frame, latent_dim)  # [batch_size, clip_frames, patches_per_frame, latent_dim]

        return tokens

        # raise NotImplementedError("Run the tiny V-JEPA predictor forward pass.")


def build_vjepa_modules(config: VJEPAConfig) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Build online encoder, EMA target encoder, and predictor."""
    # Goal:
    # Return three modules:
    # - online encoder: trained by gradients
    # - target encoder: updated by EMA, not by gradients
    # - predictor: trained by gradients
    #
    # Steps:
    # 1. Create `online_encoder = TinyVideoTransformerEncoder(config)`.
    # 2. Create `target_encoder = TinyVideoTransformerEncoder(config)`.
    # 3. Copy online encoder weights into target encoder with:
    #    `target_encoder.load_state_dict(online_encoder.state_dict())`.
    # 4. Set `requires_grad=False` for every target encoder parameter.
    # 5. Create `predictor = TinyVJEPAPredictor(config)`.
    # 6. Move all modules to `device`.
    # 7. Return them.

    online_encoder = TinyVideoTransformerEncoder(config)
    target_encoder = TinyVideoTransformerEncoder(config)

    target_encoder.load_state_dict(online_encoder.state_dict())

    for param in target_encoder.parameters():
        param.requires_grad = False

    predictor = TinyVJEPAPredictor(config)

    online_encoder = online_encoder.to(device)
    target_encoder = target_encoder.to(device)
    predictor = predictor.to(device)

    return online_encoder, target_encoder, predictor

    #raise NotImplementedError("Build the V-JEPA modules.")


@torch.no_grad()
def update_ema_target_encoder(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    ema_decay: float,
) -> None:
    """Move the target encoder a little closer to the online encoder."""
    # Formula:
    # target = ema_decay * target + (1 - ema_decay) * online
    #
    # Steps:
    # 1. Loop over matching parameters from both encoders.
    # 2. Update target parameters in place with the formula above.
    # 3. Do this inside `torch.no_grad()`.

    for online_param, target_param in zip(online_encoder.parameters(), target_encoder.parameters()):
        # Use `.data` here because this is not a normal gradient update.
        # The target encoder is a slow copy of the online encoder, so we change
        # its stored weights directly without asking autograd to track this line.
        target_param.data = ema_decay * target_param.data + (1.0 - ema_decay) * online_param.data

    #raise NotImplementedError("Update the V-JEPA target encoder with EMA.")


def select_target_tokens(token_latents: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Select only the token latents marked by the target mask."""
    # Goal:
    # Convert:
    # - `token_latents`: `[batch, time, patches_per_frame, latent_dim]`
    # - `target_mask`: `[batch, time, patches_per_frame]`
    #
    # Into:
    # - `target_token_latents`: `[batch, target_tokens_per_clip, latent_dim]`
    #
    # Method:
    # First infer the target count from the mask:
    # `target_counts = target_mask.flatten(1).sum(dim=1)`
    #
    # Then check that every row has the same count:
    # `torch.all(target_counts == target_counts[0])`
    #
    # After that, use boolean indexing and reshape:
    # `selected = token_latents[target_mask]`
    # `selected = selected.view(batch_size, int(target_counts[0].item()), latent_dim)`
    #
    # Checks:
    # - if different clips have different target counts, raise `ValueError`.
    # - output shape should be `[batch, target_count, latent_dim]`.


    # time also refers to frames_per_clip
    batch_size, frames_per_clip, patches_per_frame, latent_dim = token_latents.shape  # [batch_size, frames_per_clip, patches_per_frame, latent_dim]

    target_counts = target_mask.flatten(1).sum(dim=1)  # [batch, frames_per_clip, patches_per_frame] -> [batch_size]
    if not torch.all(target_counts == target_counts[0]):
        raise ValueError("Every clip must have the same number of target tokens.")
    target_count = int(target_counts[0].item())  # one clip's True count, usually 32

    # `target_mask` has shape `[batch_size, frames_per_clip, patches_per_frame]`.
    # token_latents has shape [batch, frames_per_clip, patches_per_frame, latent_dim]
    # So the mask lines up with each video token position.
    # Each True in target_mask selects one full latent vector from token_latents.
    # Boolean indexing keeps only the video token vectors where the mask is True.
    # If `batch_size=2` and each clip has 32 target tokens, this gives
    # `[64, latent_dim]`: all selected video token vectors from the whole batch.    
    selected = token_latents[target_mask]  # [batch_size * target_count, latent_dim]

    # Put the flat selected vectors back into one group per video clip.
    # This works because every clip has the same number of True values.
    selected = selected.view(batch_size, target_count, latent_dim)  # [batch_size, target_count, latent_dim] i.e. [batch_size, 32, 128]

    return selected

    # raise NotImplementedError("Select target video token latents.")


def train_one_epoch(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: VJEPAConfig,
) -> dict[str, float]:
    """Train V-JEPA for one epoch."""
    # Steps:
    # 1. Move `batch["video"]` to `device`.
    # 2. Create tube-shaped `target_mask` with:
    #    `make_target_token_mask(`
    #    `    batch_size=videos.shape[0],`
    #    `    clip_frames=config.clip_frames,`
    #    `    patches_per_frame=config.patches_per_frame,`
    #    `    num_target_tubes=config.num_target_tubes,`
    #    `    target_tube_frames=config.target_tube_frames,`
    #    `    target_tube_side=config.target_tube_side,`
    #    `    device=videos.device,`
    #    `)`
    # 3. Create `visible_mask = ~target_mask`.
    # 4. Encode videos with `online_encoder(videos, visible_mask)`.
    # 5. Encode clean `videos` with `target_encoder` inside `torch.no_grad()`.
    # 6. Predict token latents with `predictor(context_token_latents, target_mask)`.
    # 7. Select target tokens from predicted and target latents.
    # 8. Compute latent MSE against `target_latents.detach()`.
    # 9. Backprop and step the optimizer.
    # 10. Update the target encoder with EMA.
    # 11. Track MSE, cosine similarity, and retrieval top-1.    

    online_encoder.train()
    target_encoder.eval()
    predictor.train()

    total_loss = 0
    total_sim = 0
    total_samples = 0

    progress = tqdm(loader, desc="train", leave=False) if tqdm is not None else loader

    for step, data in enumerate(progress, start=1):
        videos = data["video"].to(device)
        #labels = data["label"] # not needed here, it is self-supervised

        batch_size, clip_frames, channels, height, width = videos.shape
       
        target_token_mask = make_target_token_mask(batch_size, clip_frames, config.patches_per_frame, 
                                                    config.num_target_tubes, config.target_tube_frames, config.target_tube_side, videos.device)

        visible_mask = torch.logical_not(target_token_mask) # this is same as ~target_token_mask

        context_token_latents  = online_encoder(videos, visible_mask)       # [batch_size, clip_frames, patches_per_frame, latent_dim]

        with (torch.no_grad()):
            target_token_latents = target_encoder(videos)                   # [batch_size, clip_frames, patches_per_frame, latent_dim]

        predicted_token_latents = predictor(context_token_latents, target_token_mask)   # [batch_size, clip_frames, patches_per_frame, latent_dim]


        selected_target_latents = select_target_tokens(target_token_latents, target_token_mask) # [batch_size, 32, 128]
        selected_pred_latents = select_target_tokens(predicted_token_latents, target_token_mask) # [batch_size, 32, 128]

        optimizer.zero_grad()
        loss = loss_fn(selected_pred_latents, selected_target_latents.detach())
        loss.backward()
        optimizer.step()

        update_ema_target_encoder(online_encoder, target_encoder, config.ema_decay)

        total_loss += loss.item() * batch_size
        total_samples+=batch_size

        # cosine_similarity(..., dim=-1) returns shape: [batch_size, target_count]
        cosine_similarity = F.cosine_similarity(selected_pred_latents, selected_target_latents.detach(), dim=-1).mean()
        total_sim += cosine_similarity.item() * batch_size

        if tqdm is not None:
            progress.set_postfix(
                step=f"{step}/{len(loader)}",
                loss=f"{loss.item():.4f}",
                cos=f"{cosine_similarity.item():.4f}",
            )

    mean_mse = total_loss / total_samples
    mean_sim = total_sim / total_samples

    #print(f"train epoch: latent_mse={mean_mse:.4f}, cosine_similarity={mean_sim:.4f}")

    return {
        "latent_mse": mean_mse,
        "cosine_similarity": mean_sim,
    }
    #raise NotImplementedError("Train V-JEPA for one epoch.")


@torch.no_grad()
def evaluate(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    config: VJEPAConfig,
) -> dict[str, float]:
    """Evaluate latent metrics and batch retrieval top-1."""
    # Steps:
    # 1. Put all modules in eval mode.
    # 2. Create tube-shaped `target_mask` with `make_target_token_mask(...)`
    #    using the same config fields as training, then set
    #    `visible_mask = ~target_mask`.
    # 3. Encode videos with `online_encoder(videos, visible_mask)`.
    # 4. Encode clean videos with `target_encoder(videos)`.
    # 5. Predict token latents with `predictor(context_token_latents, target_mask)`.
    # 6. Select only target tokens.
    # 7. Track average MSE and cosine similarity.
    # 8. Compute retrieval top-1 inside each batch.
    #
    # Retrieval top-1 method:
    # 1. Average target-token latents into one vector per clip.
    # 2. Normalize predicted vectors with `F.normalize(..., dim=-1)`.
    # 3. Normalize target vectors.
    # 4. Compute `similarity = predicted @ target.T`.
    # 5. The correct target for row `i` is column `i`.
    # 6. Top-1 is correct when `similarity.argmax(dim=1) == torch.arange(batch_size)`.

    online_encoder.eval()
    target_encoder.eval()
    predictor.eval()

    total_loss = 0
    total_sim = 0
    total_retrieval_correct = 0
    total_samples = 0

    for data in loader:
        videos = data["video"].to(device)
        # labels are not needed here because this is self-supervised evaluation.

        batch_size, clip_frames, channels, height, width = videos.shape

        target_mask = make_target_token_mask(batch_size, clip_frames, config.patches_per_frame,
                                                config.num_target_tubes, config.target_tube_frames, config.target_tube_side, videos.device)
        visible_mask = torch.logical_not(target_mask)  # same as ~target_mask

        context_token_latents = online_encoder(videos, visible_mask)        # [batch_size, clip_frames, patches_per_frame, latent_dim]

        target_token_latents = target_encoder(videos)                       # [batch_size, clip_frames, patches_per_frame, latent_dim]

        predicted_latents = predictor(context_token_latents, target_mask)   # [batch_size, clip_frames, patches_per_frame, latent_dim]

        selected_target_latents = select_target_tokens(target_token_latents, target_mask)       # [batch_size, target_count, latent_dim]
        selected_pred_latents = select_target_tokens(predicted_latents, target_mask)            # [batch_size, target_count, latent_dim]

        loss = loss_fn(selected_pred_latents, selected_target_latents)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        cosine_similarity = F.cosine_similarity(selected_pred_latents, selected_target_latents, dim=-1).mean()  # scalar
        total_sim += cosine_similarity.item() * batch_size

        predicted_clip_vectors = selected_pred_latents.mean(dim=1)  # [batch_size, latent_dim]
        target_clip_vectors = selected_target_latents.mean(dim=1)   # [batch_size, latent_dim]
        predicted_clip_vectors = F.normalize(predicted_clip_vectors, dim=-1)  # [batch_size, latent_dim]
        target_clip_vectors = F.normalize(target_clip_vectors, dim=-1)        # [batch_size, latent_dim]

        similarity = predicted_clip_vectors @ target_clip_vectors.T  # [batch_size, batch_size]
        correct_indices = torch.arange(batch_size, device=similarity.device)  # [batch_size]
        retrieval_correct = (similarity.argmax(dim=1) == correct_indices).sum().item()
        total_retrieval_correct += retrieval_correct

    mean_mse = total_loss / total_samples
    mean_sim = total_sim / total_samples   
    retrieval_top1 = total_retrieval_correct / total_samples

    # print(
    #     f"eval: latent_mse={mean_mse:.4f}, "
    #     f"cosine_similarity={mean_sim:.4f}, "
    #     f"retrieval_top1={retrieval_top1:.4f}"
    # )

    return {
        "latent_mse": mean_mse,
        "cosine_similarity": mean_sim,
        "retrieval_top1": retrieval_top1,
    }

    # raise NotImplementedError("Evaluate V-JEPA metrics.")


def train_action_linear_probe(
    frozen_encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: VJEPAConfig,
) -> dict[str, float]:
    """Train a linear classifier on frozen V-JEPA video features."""
    # Where this fits:
    # V-JEPA pretraining is useful only if the learned video features help
    # another task. UCF101 action labels give a real downstream check.
    #
    # Goal:
    # Freeze the encoder and train one linear classifier on top of its video features.
    #
    # Method:
    # 1. Put the encoder in eval mode.
    # 2. For each video, run the encoder without gradients.
    # 3. Average token latents across time and patches:
    #    `features = token_latents.mean(dim=(1, 2))`
    # 4. Train `nn.Linear(config.latent_dim, num_classes)` on UCF101 labels.
    # 5. Use `nn.CrossEntropyLoss()` from `torch.nn`.
    # 6. Report validation accuracy.
    #
    # Checks:
    # - `features.shape` should be `[batch, latent_dim]`.
    # - classifier logits should be `[batch, num_classes]`.


    frozen_encoder.eval()
    for param in frozen_encoder.parameters():
        param.requires_grad = False

    num_classes = 101  # UCF101 has 101 action classes.
    classifier = nn.Linear(in_features=config.latent_dim, out_features=num_classes).to(device)
    probe_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.probe_learning_rate)

    final_train_loss = 0.0
    final_train_accuracy = 0.0

    for epoch in range(config.linear_probe_epochs):
        classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data in train_loader:
            videos = data["video"].to(device)  # [batch_size, clip_frames, 3, image_size, image_size]
            labels = data["label"].to(device)  # [batch_size]

            with torch.no_grad():
                token_latents = frozen_encoder(videos)  # [batch_size, clip_frames, patches_per_frame, latent_dim]
                features = token_latents.mean(dim=(1, 2))  # [batch_size, latent_dim]

            logits = classifier(features)  # [batch_size, 101]
            loss = probe_loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = videos.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += batch_size

        final_train_loss = total_loss / total_samples
        final_train_accuracy = total_correct / total_samples
        print(
            f"linear probe epoch {epoch + 1}/{config.linear_probe_epochs}: "
            f"loss={final_train_loss:.4f}, accuracy={final_train_accuracy:.4f}"
        )

    classifier.eval()
    val_correct = 0
    val_samples = 0

    with torch.no_grad():
        for data in val_loader:
            videos = data["video"].to(device)  # [batch_size, clip_frames, 3, image_size, image_size]
            labels = data["label"].to(device)  # [batch_size]

            token_latents = frozen_encoder(videos)  # [batch_size, clip_frames, patches_per_frame, latent_dim]
            features = token_latents.mean(dim=(1, 2))  # [batch_size, latent_dim]
            logits = classifier(features)  # [batch_size, 101]

            val_correct += (logits.argmax(dim=-1) == labels).sum().item()
            val_samples += videos.shape[0]

    val_accuracy = val_correct / val_samples
    print(
        f"linear probe: train_loss={final_train_loss:.4f}, "
        f"train_accuracy={final_train_accuracy:.4f}, "
        f"val_accuracy={val_accuracy:.4f}"
    )

    return {
        "train_loss": final_train_loss,
        "train_accuracy": final_train_accuracy,
        "val_accuracy": val_accuracy,
    }

    # raise NotImplementedError("Train an action linear probe on frozen V-JEPA features.")


def save_vjepa_visuals(
    history: list[dict[str, float]],
    sample_batch: dict[str, torch.Tensor],
    online_encoder: nn.Module,
    config: VJEPAConfig,
) -> dict[str, str]:
    """Save clip strips, mask views, curves, and a retrieval heatmap."""
    # Save:
    # - one strip showing the original clip frames in order
    # - one strip showing the masked clip for visual checks
    # - one target-mask view that shows which patches were hidden
    # - one training curve for train and validation latent MSE
    # - one retrieval heatmap if you compute similarities for a batch
    #
    # Checks:
    # - output paths should point inside `outputs/vjepa_video/`.
    # - at least one visual should show that frame order was preserved.
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_paths: dict[str, str] = {}

    videos = sample_batch["video"].to(device)  # [batch_size, clip_frames, 3, image_size, image_size]
    labels = sample_batch["label"].to(device)  # [batch_size]

    target_mask = make_target_token_mask(
        batch_size=videos.shape[0],
        clip_frames=config.clip_frames,
        patches_per_frame=config.patches_per_frame,
        num_target_tubes=config.num_target_tubes,
        target_tube_frames=config.target_tube_frames,
        target_tube_side=config.target_tube_side,
        device=videos.device,
    )  # [batch_size, clip_frames, patches_per_frame]
    masked_videos = apply_video_token_mask(videos, target_mask, config.patch_size)  # [batch_size, clip_frames, 3, image_size, image_size]

    original_path = output_dir / "original_clip_strip.png"
    fig, axes = plt.subplots(1, config.clip_frames, figsize=(2.4 * config.clip_frames, 2.4))
    if config.clip_frames == 1:
        axes = [axes]

    for frame_index in range(config.clip_frames):
        frame = videos[0, frame_index].detach().cpu().permute(1, 2, 0)  # [image_size, image_size, 3]
        axes[frame_index].imshow(frame)
        axes[frame_index].set_title(f"frame {frame_index}")
        axes[frame_index].axis("off")

    fig.suptitle(f"Original clip, label {int(labels[0].item())}")
    fig.tight_layout()
    fig.savefig(original_path, dpi=160)
    plt.close(fig)
    visual_paths["original_clip_strip"] = str(original_path)

    masked_path = output_dir / "masked_clip_strip.png"
    fig, axes = plt.subplots(1, config.clip_frames, figsize=(2.4 * config.clip_frames, 2.4))
    if config.clip_frames == 1:
        axes = [axes]

    for frame_index in range(config.clip_frames):
        frame = masked_videos[0, frame_index].detach().cpu().permute(1, 2, 0)  # [image_size, image_size, 3]
        axes[frame_index].imshow(frame)
        axes[frame_index].set_title(f"frame {frame_index}")
        axes[frame_index].axis("off")

    fig.suptitle("Masked target patches")
    fig.tight_layout()
    fig.savefig(masked_path, dpi=160)
    plt.close(fig)
    visual_paths["masked_clip_strip"] = str(masked_path)

    mask_path = output_dir / "target_mask_strip.png"
    patches_per_side = config.patches_per_side
    fig, axes = plt.subplots(1, config.clip_frames, figsize=(2.4 * config.clip_frames, 2.4))
    if config.clip_frames == 1:
        axes = [axes]

    for frame_index in range(config.clip_frames):
        mask_grid = target_mask[0, frame_index].detach().cpu().view(patches_per_side, patches_per_side)  # [patches_per_side, patches_per_side]
        axes[frame_index].imshow(mask_grid, cmap="gray_r", vmin=0, vmax=1)
        axes[frame_index].set_title(f"frame {frame_index}")
        axes[frame_index].set_xticks([])
        axes[frame_index].set_yticks([])

    fig.suptitle("Target mask: dark patches are hidden")
    fig.tight_layout()
    fig.savefig(mask_path, dpi=160)
    plt.close(fig)
    visual_paths["target_mask_strip"] = str(mask_path)

    curve_path = output_dir / "latent_metric_curves.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = list(range(1, len(history) + 1))
    train_values = [
        item.get("train_latent_mse", item.get("train_mse", item.get("latent_mse")))
        for item in history
    ]
    val_values = [
        item.get("val_latent_mse", item.get("val_mse"))
        for item in history
    ]

    if any(value is not None for value in train_values):
        ax.plot(epochs, train_values, marker="o", label="train latent MSE")
    if any(value is not None for value in val_values):
        ax.plot(epochs, val_values, marker="o", label="val latent MSE")

    ax.set_title("V-JEPA latent prediction error")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    visual_paths["metric_curves"] = str(curve_path)

    heatmap_path = output_dir / "encoder_feature_similarity_heatmap.png"
    online_encoder.eval()
    with torch.no_grad():
        token_latents = online_encoder(videos)  # [batch_size, clip_frames, patches_per_frame, latent_dim]
        features = token_latents.mean(dim=(1, 2))  # [batch_size, latent_dim]
        features = F.normalize(features, dim=-1)  # [batch_size, latent_dim]
        similarity = features @ features.T  # [batch_size, batch_size]

    fig, ax = plt.subplots(figsize=(5, 4))
    heatmap = ax.imshow(similarity.detach().cpu(), vmin=-1, vmax=1, cmap="viridis")
    ax.set_title("Frozen encoder feature similarity")
    ax.set_xlabel("target clip index")
    ax.set_ylabel("query clip index")
    fig.colorbar(heatmap, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=160)
    plt.close(fig)
    visual_paths["feature_similarity_heatmap"] = str(heatmap_path)

    return visual_paths

    # raise NotImplementedError("Save V-JEPA visuals.")


def run_vjepa_experiment(config: VJEPAConfig) -> dict[str, object]:
    """Run the full V-JEPA-style guided experiment."""
    # Steps:
    # 1. Load dataloaders with `load_ucf101_vjepa_dataloaders(config)`.
    # 2. Build modules with `build_vjepa_modules(config)`.
    # 3. Create an optimizer for the online encoder and predictor only.
    # 4. Train for `config.epochs`.
    # 5. Evaluate on validation and test loaders.
    # 6. Train the action linear probe.
    # 7. Save visuals.
    #
    # Return:
    # - `train_latent_mse`
    # - `val_latent_mse`
    # - `test_latent_mse`
    # - `val_cosine_similarity`
    # - `test_cosine_similarity`
    # - `test_retrieval_top1`
    # - `linear_probe_val_accuracy`
    # - `visual_paths`

    loaders = load_ucf101_vjepa_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    online_encoder, target_encoder, predictor = build_vjepa_modules(config)
    optimizer = torch.optim.AdamW(
        list(online_encoder.parameters()) + list(predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    print(f"device: {device}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}, test batches: {len(test_loader)}")
    print(f"video size: {config.image_size}x{config.image_size}, clip frames: {config.clip_frames}")
    print(f"patch size: {config.patch_size}, patches per frame: {config.patches_per_frame}")
    print(f"total tokens per clip: {config.total_tokens}, target tokens per clip: {config.target_tokens_per_clip}")

    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(
            online_encoder=online_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
        )
        val_metrics = evaluate(
            online_encoder=online_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            loader=val_loader,
            loss_fn=loss_fn,
            config=config,
        )

        epoch_record = {
            "epoch": float(epoch + 1),
            "train_latent_mse": train_metrics["latent_mse"],
            "train_cosine_similarity": train_metrics["cosine_similarity"],
            "val_latent_mse": val_metrics["latent_mse"],
            "val_cosine_similarity": val_metrics["cosine_similarity"],
            "val_retrieval_top1": val_metrics["retrieval_top1"],
        }
        history.append(epoch_record)
        
        print(
            f"epoch {epoch + 1}/{config.epochs}: "
            f"train_mse={train_metrics['latent_mse']:.4f}, "
            f"train_cos={train_metrics['cosine_similarity']:.4f}, "
            f"val_mse={val_metrics['latent_mse']:.4f}, "
            f"val_cos={val_metrics['cosine_similarity']:.4f}, "
            f"val_retrieval_top1={val_metrics['retrieval_top1']:.4f}"
        )

    test_metrics = evaluate(
        online_encoder=online_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        loader=test_loader,
        loss_fn=loss_fn,
        config=config,
    )
    print(
        f"test: latent_mse={test_metrics['latent_mse']:.4f}, "
        f"cosine_similarity={test_metrics['cosine_similarity']:.4f}, "
        f"retrieval_top1={test_metrics['retrieval_top1']:.4f}"
    )

    probe_metrics = train_action_linear_probe(
        frozen_encoder=online_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    sample_batch = next(iter(val_loader))
    visual_paths = save_vjepa_visuals(
        history=history,
        sample_batch=sample_batch,
        online_encoder=online_encoder,
        config=config,
    )
    print("saved visuals:")
    for name, path in visual_paths.items():
        print(f"- {name}: {path}")

    final_train_metrics = history[-1]
    print("final metrics:")
    print(f"train_latent_mse: {final_train_metrics['train_latent_mse']:.4f}")
    print(f"val_latent_mse: {final_train_metrics['val_latent_mse']:.4f}")
    print(f"test_latent_mse: {test_metrics['latent_mse']:.4f}")
    print(f"val_cosine_similarity: {final_train_metrics['val_cosine_similarity']:.4f}")
    print(f"test_cosine_similarity: {test_metrics['cosine_similarity']:.4f}")
    print(f"val_retrieval_top1: {final_train_metrics['val_retrieval_top1']:.4f}")
    print(f"test_retrieval_top1: {test_metrics['retrieval_top1']:.4f}")
    print(f"linear_probe_val_accuracy: {probe_metrics['val_accuracy']:.4f}")

    return {
        "train_latent_mse": final_train_metrics["train_latent_mse"],
        "val_latent_mse": final_train_metrics["val_latent_mse"],
        "test_latent_mse": test_metrics["latent_mse"],
        "val_cosine_similarity": final_train_metrics["val_cosine_similarity"],
        "test_cosine_similarity": test_metrics["cosine_similarity"],
        "val_retrieval_top1": final_train_metrics["val_retrieval_top1"],
        "test_retrieval_top1": test_metrics["retrieval_top1"],
        "linear_probe_val_accuracy": probe_metrics["val_accuracy"],
        "history": history,
        "visual_paths": visual_paths,
    }

    # raise NotImplementedError("Run the full V-JEPA experiment.")


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
    """Run the V-JEPA scaffold."""
    config = VJEPAConfig()
    set_seed(config.random_state)
    print("exercise 03 is V-JEPA with space-time masks and an EMA target encoder.")
    result, ok = report_task_progress("run_vjepa_experiment", run_vjepa_experiment, config)
    if ok:
        print(result)


if __name__ == "__main__":
    main()
