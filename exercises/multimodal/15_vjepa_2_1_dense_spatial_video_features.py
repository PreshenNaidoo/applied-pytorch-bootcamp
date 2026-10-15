"""
============================================================
EXERCISE 15: V-JEPA 2.1 DENSE SPATIAL VIDEO FEATURES
============================================================

WHAT YOU ARE BUILDING
You are building a small V-JEPA 2.1-style model that stays close to the
paper's dense video representation workflow while staying small enough for a
normal bootcamp GPU.

Dense V-JEPA-style learning predicts many video patch features, not one vector
for the whole clip.

Plain meaning:
- one vector for the whole frame can say "a person is moving"
- a grid of vectors can say "the hand is here, the object is there"

This is closer to dense video representation learning.
Dense means the model keeps location information instead of squeezing the whole frame into
one summary vector.

WHY THIS EXERCISE EXISTS
Agents need to know where things are.

Examples:
- a robot needs to know where the cup is
- a game agent needs to know where the door is
- a video model needs to know which region moved

This exercise teaches the next step after basic V-JEPA:

    visible video patch tokens + target mask -> predicted dense patch features

V-JEPA 2.1 adds an important idea on top of basic V-JEPA:

    train dense features at more than one encoder depth

Plain meaning:
- the final transformer layer should learn useful patch features
- one earlier transformer layer should also learn useful patch features
- this helps the model keep local details, not only high-level clip meaning

REAL SYSTEM FIRST
In the larger V-JEPA 2.1 system:

1. load real videos
2. sample short ordered clips
3. split frames into patch tokens
4. choose target patch blocks across space and time
5. keep target patch tokens out of the online encoder attention
6. encode visible video tokens with an online video transformer
7. encode the clean clip with an EMA target video transformer
8. use a predictor to fill dense patch features
9. train with dense representation-space loss over patch tokens
10. also train from an intermediate encoder layer
11. measure dense patch error and location-aware retrieval

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
- 96 by 96 frames instead of high-resolution frames
- 16 by 16 patches instead of many small ViT patches
- a tiny video transformer instead of a large video transformer
- a tiny predictor instead of the larger V-JEPA 2.1 predictor
- one intermediate layer loss instead of many deep losses
- a short training run instead of long pretraining
- single-process training instead of distributed training

Not scaled down:
- real video clips
- frame order
- patch-token video representation
- target patch-block masking
- token-level context masking
- online encoder and EMA target encoder
- dense representation loss on patch features
- a small deep self-supervision loss from one intermediate layer
- dense patch retrieval and error heatmaps

DATASET
Use real UCF101 clips through `torchvision.datasets.UCF101`.

Expected local layout:

    data/ucf101/videos
    data/ucf101/annotations

Do not replace missing UCF101 data with random frames.
If the local dataset is missing, raise a clear error that names the missing path.

Mapping from the real V-JEPA 2.1 system to this exercise:

    real V-JEPA 2.1 system              this exercise
    ---------------------------------------------------------------
    large video dataset                 UCF101 clips
    video patch tokens                  frame-patch tokens
    target block sampling               small patch blocks across frames
    context video tokens                target tokens hidden from attention
    online video encoder                tiny video transformer
    EMA target encoder                  slow copy of online video encoder
    dense predictor                     tiny transformer with learned target mask tokens
    dense predictive loss               all-token patch-feature MSE
    deep self-supervision               final-layer loss + one intermediate-layer loss
    dense quality check                 patch MSE and patch retrieval
    visual check                        patch error heatmap

What is included:
- real video clips
- real train, validation, and test splits
- trainable online video transformer
- EMA target video transformer
- trainable dense transformer predictor
- final-layer dense loss
- one intermediate-layer dense loss
- dense patch metrics
- saved frame strips and heatmaps

What is skipped:
- large ViT/video backbones
- multi-scale feature pyramids
- deep self-supervision at many encoder layers
- image and video tokenizer mixing
- distributed training
- full V-JEPA 2.1 reproduction

Why those parts are skipped:
- Full deep self-supervision would ask the student to collect features from
  several transformer layers and compute several losses. This exercise keeps
  one intermediate layer loss so the paper idea is present without making the
  file too large.
- Image and video tokenizer mixing would add a second data path. This exercise
  stays on video clips so the student can focus on dense video features.

INPUTS AND OUTPUTS
Each batch should contain:

    video: [batch, clip_frames, 3, image_size, image_size]

During training, you create:

    target_mask:  [batch, clip_frames, patches_per_frame]
    visible_mask: [batch, clip_frames, patches_per_frame]

The encoders should produce:

    context_outputs["final_tokens"]:        [batch, clip_frames, patches_per_frame, latent_dim]
    context_outputs["intermediate_tokens"]: [batch, clip_frames, patches_per_frame, latent_dim]
    target_outputs["final_tokens"]:         [batch, clip_frames, patches_per_frame, latent_dim]
    target_outputs["intermediate_tokens"]:  [batch, clip_frames, patches_per_frame, latent_dim]

The predictor should produce:

    predicted_outputs["final_tokens"]:        [batch, clip_frames, patches_per_frame, latent_dim]
    predicted_outputs["intermediate_tokens"]: [batch, clip_frames, patches_per_frame, latent_dim]

The dense loss uses all patch-token positions:

    final_prediction_loss = nn.MSELoss()(
        predicted_outputs["final_tokens"],
        target_outputs["final_tokens"].detach(),
    )

The small deep self-supervision loss uses one earlier layer:

    intermediate_prediction_loss = nn.MSELoss()(
        predicted_outputs["intermediate_tokens"],
        target_outputs["intermediate_tokens"].detach(),
    )

The target mask still matters.
It hides input tokens from the online encoder.
The dense loss then asks the predictor to match the target encoder at every
patch-token position, including both visible and hidden positions.

LOSS FUNCTION
Use mean squared error on dense patch features.

For the final encoder layer:

    final_loss = nn.MSELoss()(predicted_final_tokens, target_final_tokens.detach())

For one intermediate encoder layer:

    intermediate_loss = nn.MSELoss()(
        predicted_intermediate_tokens,
        target_intermediate_tokens.detach(),
    )

Combine them:

    loss = final_loss + intermediate_loss_weight * intermediate_loss

Plain meaning:
- final loss teaches the strongest dense features
- intermediate loss keeps earlier patch features useful too
- `intermediate_loss_weight` controls how much the earlier-layer loss matters

METRICS TO TRACK
Track:
- train dense latent MSE, lower is better
- validation dense latent MSE, lower is better
- test dense latent MSE, lower is better
- patch cosine similarity, higher is better
- all-token retrieval top-1 inside the same clip, higher is better
- intermediate dense latent MSE, lower is better

All-token retrieval top-1 means:
- compare each predicted patch token with all target encoder patch tokens from the same clip
- count it correct when the closest target token is the matching token position

Small example:
- with `4` frames and a `6 by 6` patch grid, there are `144` patch tokens
- for predicted token 10, the correct match is target token 10
- if the most similar target token is also 10, that token is counted correct

VISUALS TO SAVE
Save to `outputs/dense_vjepa/`:
- clean clip strip
- target-masked clip strip
- target patch mask strip
- training curve
- patch error heatmap over a target frame from the clip

STEP-BY-STEP TODO
1. Load real UCF101 clips.
2. Convert clips to `[time, channels, height, width]`.
3. Create target patch-block masks across frames.
4. Create visible token masks from target masks.
5. Build the online tiny video transformer.
6. Build the EMA target tiny video transformer.
7. Build the dense transformer predictor.
8. Train on final-layer and intermediate-layer dense latent MSE.
9. Update the target encoder with EMA after each optimizer step.
10. Evaluate patch MSE, cosine similarity, and patch retrieval.
11. Save visual checks.

Learning order:
1. Read the dataset wrapper first.
2. Build the mask helper next.
3. Build the encoder and predictor.
4. Build the final-layer dense loss path.
5. Add the one-layer deep self-supervision loss.
6. Add evaluation and visuals last.

DEFINITION OF DONE
You are done when:
- the exercise uses real video clips
- the model predicts dense patch features
- the model uses final-layer and intermediate-layer dense losses
- the target encoder is updated with EMA
- validation and test metrics are printed
- patch retrieval top-1 is printed
- patch error heatmap is saved
- you can explain why dense features are more useful than one whole-frame vector
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
import torchvision

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


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DenseVJEPAConfig:
    """Settings for small V-JEPA 2.1-style video learning."""

    video_root: str = str(BOOTCAMP_ROOT / "data" / "ucf101" / "UCF-101")
    annotation_root: str = str(BOOTCAMP_ROOT / "data" / "ucf101" / "ucfTrainTestlist")
    output_dir: Path = PROJECT_ROOT / "outputs" / "vjepa_2_1_dense"
    image_size: int = 96
    clip_frames: int = 4
    patch_size: int = 16
    latent_dim: int = 96
    encoder_depth: int = 2
    encoder_num_heads: int = 4
    encoder_mlp_dim: int = 192
    predictor_depth: int = 1
    predictor_num_heads: int = 4
    predictor_mlp_dim: int = 192
    intermediate_loss_weight: float = 0.5
    num_target_blocks: int = 4
    target_block_frames: int = 2
    target_block_side: int = 2
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.99
    max_clips: int = 512
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    random_state: int = 315

    def __post_init__(self) -> None:
        """Catch settings that would break the small video transformer."""
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must divide evenly by patch_size.")
        if self.latent_dim % self.encoder_num_heads != 0:
            raise ValueError("latent_dim must divide evenly by encoder_num_heads.")
        if self.latent_dim % self.predictor_num_heads != 0:
            raise ValueError("latent_dim must divide evenly by predictor_num_heads.")
        if self.target_block_frames > self.clip_frames:
            raise ValueError("target_block_frames must be no larger than clip_frames.")
        if self.target_block_side > self.patches_per_side:
            raise ValueError("target_block_side must be no larger than patches_per_side.")
        if self.target_tokens_per_clip >= self.total_tokens:
            raise ValueError("target_tokens_per_clip must be smaller than total_tokens.")

        possible_start_frames = self.clip_frames - self.target_block_frames + 1
        possible_start_rows = self.patches_per_side - self.target_block_side + 1
        possible_start_cols = self.patches_per_side - self.target_block_side + 1
        possible_blocks = possible_start_frames * possible_start_rows * possible_start_cols
        if possible_blocks < self.num_target_blocks:
            raise ValueError("num_target_blocks is too large for the current clip and patch grid.")

    @property
    def patches_per_side(self) -> int:
        """Return how many square patches fit along one frame side."""
        return self.image_size // self.patch_size                                           #6

    @property
    def patches_per_frame(self) -> int:
        """Return the number of patch tokens in one frame."""
        return self.patches_per_side * self.patches_per_side                               #36

    @property
    def total_tokens(self) -> int:
        """Return the number of patch tokens in one video clip."""
        return self.clip_frames * self.patches_per_frame                                   #144

    @property
    def target_tokens_per_block(self) -> int:
        """Return how many tokens one target block contains."""
        return self.target_block_frames * self.target_block_side * self.target_block_side  #8

    @property
    def target_tokens_per_clip(self) -> int:
        """Return how many target tokens each clip should contain."""
        return self.num_target_blocks * self.target_tokens_per_block                        #32


class DenseVJEPAVideoDataset(Dataset):
    """Wrap UCF101 so each sample has one ordered video clip."""

    def __init__(self, base_dataset: Dataset, image_size: int) -> None:
        self.base_dataset = base_dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clip, _, label = self.base_dataset[index]
        if clip.ndim != 4:
            raise ValueError("Expected a video clip shaped [T, H, W, C] or [T, C, H, W].")

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


def load_dense_vjepa_dataloaders(config: DenseVJEPAConfig) -> dict[str, DataLoader]:
    """Load real UCF101 clips for dense V-JEPA training."""
    # Where this fits:
    # This is the only data-loading step.
    # Every later model trains on real clips returned by these loaders.
    #
    # Goal:
    # Return train, validation, and test dataloaders.
    #
    # Method:
    # Use `torchvision.datasets.UCF101`.
    # Call shape:
    # `UCF101(root=str(config.video_root), annotation_path=str(config.annotation_root), frames_per_clip=config.clip_frames)`
    #
    # Steps:
    # 1. Check that `config.video_root.exists()`.
    # 2. Check that `config.annotation_root.exists()`.
    # 3. Load UCF101 with `frames_per_clip=config.clip_frames`.
    # 4. If the dataset is large, keep the first `config.max_clips` clips with
    #    `torch.utils.data.Subset(base_dataset, range(config.max_clips))`.
    #    This is still real video data. It is only a smaller slice for laptop training.
    # 5. Wrap it with `DenseVJEPAVideoDataset(...)`.
    # 6. Split into train, validation, and test with `torch.utils.data.random_split`.
    # 7. Return DataLoaders with train shuffle on and validation/test shuffle off.
    #
    # Checks:
    # - one batch has `video` shaped `[batch, clip_frames, 3, image_size, image_size]`.

    if not Path(config.video_root).exists():
        raise ValueError("video_root path does not exist.")
    if not Path(config.annotation_root).exists():
        raise ValueError("annotation_root path does not exist.")

    base_dataset = torchvision.datasets.UCF101(root=config.video_root, annotation_path=config.annotation_root, frames_per_clip=config.clip_frames)
    if len(base_dataset) > config.max_clips:
        data = torch.utils.data.Subset(base_dataset, indices=range(0,config.max_clips))
    else:
        data = base_dataset

    ds = DenseVJEPAVideoDataset(data, config.image_size)

    gen = torch.Generator().manual_seed(config.random_state)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset=ds, lengths=[0.6, 0.2, 0.2], generator = gen)

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size = config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size = config.batch_size)}    
    #raise NotImplementedError("Load real UCF101 clips for dense V-JEPA.")


def make_dense_target_mask(
    batch_size: int,            # 8
    clip_frames: int,           # 4
    patches_per_frame: int,     # 36
    num_target_blocks: int,     # 4
    target_block_frames: int,   # 2
    target_block_side: int,     # 2
    device: str | torch.device,
) -> torch.Tensor:
    """Create target patch-block masks across space and time."""
    # Where this fits:
    # Real V-JEPA-style training chooses target regions before the online
    # encoder runs. The online encoder can use only visible tokens. The
    # predictor must fill the target tokens.
    #
    # Goal:
    # Return `target_mask` shaped `[batch, clip_frames, patches_per_frame]`.
    # `True` means "this patch token is a target to predict".
    #
    # Small example:
    # With a 6 by 6 patch grid, `target_block_frames=2`, and
    # `target_block_side=2`, one target block covers:
    #
    # ```text
    # frame 1, patch rows 2..3, patch cols 4..5
    # frame 2, patch rows 2..3, patch cols 4..5
    # ```
    #
    # That block contains `2 * 2 * 2 = 8` target tokens.
    #
    # Method:
    # 1. Find the width of the square patch grid:
    #    `patches_per_side = int(patches_per_frame ** 0.5)`.
    #    With 36 patches, this gives a 6 by 6 grid.
    # 2. Check that the grid is really square. Raise `ValueError` when
    #    `patches_per_side * patches_per_side != patches_per_frame`.
    # 3. Create `target_mask` with:
    #    `torch.zeros((batch_size, clip_frames, patches_per_frame), dtype=torch.bool, device=device)`.
    #    It starts full of `False` because no token has been chosen as a target yet.
    # 4. Choose a maximum number of sampling attempts, such as
    #    `max_attempts = num_target_blocks * 100`. This prevents an endless loop
    #    when the requested non-overlapping blocks cannot fit.
    # 5. Loop over each clip with `for batch_index in range(batch_size)`.
    #    For that clip, start `placed_blocks = 0` and `attempts = 0`.
    # 6. While `placed_blocks < num_target_blocks`, sample one possible block start:
    #    - frame from `0` through `clip_frames - target_block_frames`
    #    - patch row from `0` through `patches_per_side - target_block_side`
    #    - patch column from `0` through `patches_per_side - target_block_side`
    #
    #    Use `torch.randint(low=0, high=last_start + 1, size=(), device=device).item()`.
    #    `high` is excluded, so the `+ 1` lets the block use the final valid start.
    # 7. Build a list called `block_positions`. Use three nested loops over the
    #    block's frames, patch rows, and patch columns. For each position, convert
    #    the 2D patch location to the stored flat index:
    #    `patch_index = patch_row * patches_per_side + patch_col`.
    #    Append `(frame_index, patch_index)` to the list.
    #    Sample row and column separately; sampling one flat start index could
    #    make a block incorrectly wrap from the end of one row into the next.
    # 8. Increase `attempts`. If it reaches `max_attempts` before all blocks are
    #    placed, raise `RuntimeError` with a message that says non-overlapping
    #    target blocks could not be sampled.
    # 9. Check every `(frame_index, patch_index)` in `block_positions` against
    #    `target_mask[batch_index]`. If any position is already `True`, use
    #    `continue` to reject the whole sampled block and try another start.
    # 10. If there is no overlap, set every position in the block to `True`, then
    #     increase `placed_blocks` by one.
    # 11. After all clips are finished, return `target_mask`.
    #
    # Checks:
    # - `target_mask.shape == (batch_size, clip_frames, patches_per_frame)`.
    # - `target_mask.dtype == torch.bool` and `target_mask.device` matches `device`.
    # - Each clip should contain exactly
    #   `num_target_blocks * target_block_frames * target_block_side ** 2`
    #   true values. Check with `target_mask.sum(dim=(1, 2))`.

    patches_per_side = int(patches_per_frame ** 0.5)
    if patches_per_side * patches_per_side != patches_per_frame:
        raise ValueError("patches_per_frame must form a square patch grid.")

    target_mask = torch.zeros(size=(batch_size, clip_frames, patches_per_frame), dtype=torch.bool, device=device)

    max_attempts = num_target_blocks * 100      # 4*100

    for batch_index in range(batch_size):
        placed_blocks = 0
        attempts = 0

        while placed_blocks < num_target_blocks:
            if attempts >= max_attempts:
                raise RuntimeError("Unable to set number of target blocks within max_attempts.")

            sampled_frame = torch.randint(low=0, high=clip_frames-target_block_frames + 1, size=(), device=device).item() # 0 to 4 - 2 + 1 i.e. 0 to 3(exclusive). thats why +1
            sampled_patch_row = torch.randint(low=0, high=patches_per_side - target_block_side + 1, size=(), device=device).item() # 6 - 2 + 1 i.e. [0, 5)
            sampled_patch_col = torch.randint(low=0, high=patches_per_side - target_block_side + 1, size=(), device=device).item() # 6 - 2 + 1 i.e. [0, 5)

            block_positions=[]

            for frame_index in range(sampled_frame, sampled_frame + target_block_frames):
                for row_index in range(sampled_patch_row, sampled_patch_row + target_block_side):
                    for col_index in range(sampled_patch_col, sampled_patch_col + target_block_side):
                        patch_index = row_index * patches_per_side + col_index

                        block_positions.append((frame_index, patch_index))

            attempts+=1

            reject_block = False
            for frame_index, patch_index in block_positions:
                if target_mask[batch_index, frame_index, patch_index]:
                    reject_block = True
                    break

            if reject_block:
                continue

            # if there is no overlap then set to True:
            for frame_index, patch_index in block_positions:
                target_mask[batch_index, frame_index, patch_index] = True

            placed_blocks+=1

    return target_mask
    #raise NotImplementedError("Create dense V-JEPA target patch-block masks.")


def apply_dense_video_mask(
    videos: torch.Tensor,
    target_mask: torch.Tensor,
    patch_size: int,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """Return a visual copy of videos with target patches hidden."""
    # Where this fits:
    # This helper is for visual checks only.
    # The model itself should use token-level masking inside the transformer.
    #
    # Input:
    # - `videos`: `[batch, clip_frames, 3, image_size, image_size]`
    # - `target_mask`: `[batch, clip_frames, patches_per_frame]`
    #
    # Steps:
    # 1. Clone the input with `masked_videos = videos.clone()`.
    #    Change the clone so this visual helper does not damage the clean videos
    #    that the target encoder needs later.
    # 2. Read `batch_size`, `clip_frames`, `channels`, `image_height`, and
    #    `image_width` from `videos.shape`.
    # 3. Check that both image sides divide evenly by `patch_size`. If either
    #    side does not, raise `ValueError` because the patch slices would not
    #    cover the image cleanly.
    # 4. Compute:
    #    - `patch_rows = image_height // patch_size`
    #    - `patch_cols = image_width // patch_size`
    #    Check that `target_mask.shape` equals
    #    `(batch_size, clip_frames, patch_rows * patch_cols)`.
    # 5. Loop over every `frame_index`, `patch_row`, and `patch_col`.
    #    The batch dimension does not need its own Python loop because a boolean
    #    selector can choose all matching videos at once.
    # 6. Convert the patch row and column to the flat index used by the mask:
    #    `patch_index = patch_row * patch_cols + patch_col`.
    #    For a 6 by 6 grid, row 2 and column 3 become index `2 * 6 + 3 = 15`.
    # 7. Convert the patch location into pixel slice boundaries:
    #    - `row_start = patch_row * patch_size`
    #    - `row_end = row_start + patch_size`
    #    - `col_start = patch_col * patch_size`
    #    - `col_end = col_start + patch_size`
    # 8. Read `batch_selector = target_mask[:, frame_index, patch_index]`.
    #    It has shape `[batch_size]`. For example,
    #    `[False, True, False, True]` selects videos 1 and 3 from the batch.
    # 9. Fill only those videos at this frame and patch location:
    #    `masked_videos[batch_selector, frame_index, :, row_start:row_end, col_start:col_end] = mask_value`.
    #    `:` keeps all colour channels, so the whole image patch is hidden.
    # 10. After all patch positions have been checked, return `masked_videos`.
    #
    # Checks:
    # - `masked_videos.shape == videos.shape`.
    # - Positions where `target_mask` is `True` should contain `mask_value`.
    # - Positions where `target_mask` is `False` should still match `videos`.
    # - `videos` itself should remain unchanged.

    batch, clip_frames, channels, image_height, image_width = videos.shape
    masked_videos = videos.clone()

    patch_rows = image_height // patch_size   # should be 6     i.e. 96 // 16
    patch_cols = image_width // patch_size    # should be 6
    # patch_rows == patch_cols

    #target_mask shape is (batch_size, clip_frames, patch_rows * patch_cols)

    for frame_index in range(clip_frames):
        for row_index in range(patch_rows):
            for col_index in range(patch_cols):

                patch_index = row_index * patch_rows + col_index

                #Convert the patch location into pixel slice boundaries
                row_start = row_index * patch_size
                row_end = row_start + patch_size
                col_start = col_index * patch_size
                col_end = col_start + patch_size

                batch_selector = target_mask[:, frame_index, patch_index]
                masked_videos[batch_selector, frame_index, :, row_start: row_end, col_start: col_end] = mask_value

    return masked_videos
    #raise NotImplementedError("Create masked clips for dense V-JEPA visual checks.")


class TinyDenseVideoTransformerEncoder(nn.Module):
    """Tiny video transformer that returns final and intermediate dense tokens."""

    def __init__(self, config: DenseVJEPAConfig) -> None:
        super().__init__()
        # Where this fits:
        # This class is the small version of the paper's video encoder.
        # It is used twice:
        # - online encoder: sees only visible tokens
        # - target encoder: EMA copy that sees the clean clip
        #
        # Goal:
        # Turn videos shaped `[batch, clip_frames, 3, image_size, image_size]`
        # into two patch-token tensors:
        # - `final_tokens`: `[batch, clip_frames, patches_per_frame, latent_dim]`
        # - `intermediate_tokens`: `[batch, clip_frames, patches_per_frame, latent_dim]`
        #
        # Why return two tensors?
        # V-JEPA 2.1 trains dense features at more than one encoder depth.
        # This small exercise uses only one earlier layer plus the final layer.
        #
        # Build these parts:
        # - `self.clip_frames = config.clip_frames`
        # - `self.patches_per_frame = config.patches_per_frame`
        # - `self.total_tokens = config.total_tokens`
        # - `self.patch_embed = nn.Conv2d(3, config.latent_dim, kernel_size=config.patch_size, stride=config.patch_size)`
        # - `self.pos_embed = nn.Parameter(torch.zeros(1, config.total_tokens, config.latent_dim))`
        # - `self.hidden_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))`
        # - `self.layers = nn.ModuleList([...])`
        # - `self.intermediate_norm = nn.LayerNorm(config.latent_dim)`
        # - `self.norm = nn.LayerNorm(config.latent_dim)`
        #
        # Use this transformer layer:
        # `nn.TransformerEncoderLayer(`
        # `    d_model=config.latent_dim,`
        # `    nhead=config.encoder_num_heads,`
        # `    dim_feedforward=config.encoder_mlp_dim,`
        # `    dropout=0.0,`
        # `    activation="gelu",`
        # `    batch_first=True,`
        # `)`
        #
        # Build `self.layers` from `config.encoder_depth` copies of the
        # transformer layer. Do not use `nn.TransformerEncoder(...)` here,
        # because the exercise needs the output from one middle layer.
        #
        # For the intermediate layer:
        # - use `intermediate_layer_index = max(0, config.encoder_depth // 2 - 1)`
        # - save that layer output during `forward`
        #
        # Checks:
        # - input `[2, 4, 3, 96, 96]` should return a dictionary.
        # - `outputs["final_tokens"].shape` should be `[2, 4, 36, latent_dim]`.
        # - `outputs["intermediate_tokens"].shape` should be `[2, 4, 36, latent_dim]`.

        self.clip_frames = config.clip_frames                   # 4
        self.patches_per_frame = config.patches_per_frame       # 36
        self.total_tokens = config.total_tokens                 # 144
        self.latent_dim = config.latent_dim

        self.patch_embed = nn.Conv2d(in_channels = 3, out_channels = config.latent_dim, kernel_size=config.patch_size, stride=config.patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(size=(1, config.total_tokens, config.latent_dim)))

        self.hidden_token = nn.Parameter(torch.zeros(size=(1, 1, config.latent_dim)))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=config.encoder_num_heads,
                dim_feedforward=config.encoder_mlp_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(config.encoder_depth)
        ])

        self.intermediate_layer_index = max(0, config.encoder_depth // 2 - 1)
        self.intermediate_norm = nn.LayerNorm(config.latent_dim)
        self.norm = nn.LayerNorm(config.latent_dim)

        # raise NotImplementedError("Build the tiny dense video transformer encoder.")

    def forward(self, videos: torch.Tensor, visible_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Encode videos into final and intermediate dense token latents."""
        # Input:
        # - `videos`: `[batch, clip_frames, 3, image_size, image_size]`
        # - `visible_mask`: `[batch, clip_frames, patches_per_frame]`
        #
        # Output:
        # Return a dictionary with:
        # - `final_tokens`: `[batch, clip_frames, patches_per_frame, latent_dim]`
        # - `intermediate_tokens`: `[batch, clip_frames, patches_per_frame, latent_dim]`
        #
        # Steps:
        # 1. Flatten videos to `[batch * clip_frames, 3, height, width]`.
        # 2. Run `self.patch_embed`.
        # 3. Flatten the patch grid with `.flatten(2).transpose(1, 2)`.
        # 4. Reshape to `[batch, total_tokens, latent_dim]`.
        # 5. Start with `flat_visible_mask = None`. Only reshape the mask inside
        #    `if visible_mask is not None`, because the clean target-encoder path
        #    calls this function without a mask.
        #    Reshape `[batch, clip_frames, patches_per_frame]` to
        #    `[batch, total_tokens]` with:
        #    `flat_visible_mask = visible_mask.reshape(batch_size, self.total_tokens)`.
        #    Do not flatten batch and time together. `patches` already has shape
        #    `[batch, total_tokens, latent_dim]`, so its first two dimensions
        #    must line up with the mask.
        #
        #    Add one final dimension to the mask, then replace hidden positions:
        #    `patches = torch.where(`
        #    `    flat_visible_mask.unsqueeze(-1),`
        #    `    patches,`
        #    `    self.hidden_token,`
        #    `)`
        #
        #    Shapes passed to `torch.where`:
        #    - condition: `[batch, total_tokens, 1]`
        #    - visible patch values: `[batch, total_tokens, latent_dim]`
        #    - hidden token: `[1, 1, latent_dim]`
        #    PyTorch copies the size-1 dimensions as needed. `True` keeps the
        #    real patch token; `False` inserts the learned hidden token.
        # 6. Add `self.pos_embed`.
        # 7. Run each layer in `self.layers` one at a time.
        #    Pass `src_key_padding_mask=~flat_visible_mask` when a mask exists.
        # 8. Save the hidden state at `self.intermediate_layer_index`.
        # 9. Apply `self.intermediate_norm` to the saved hidden state.
        # 10. Apply `self.norm` to the final hidden state.
        # 11. Reshape both outputs back to
        #     `[batch, clip_frames, patches_per_frame, latent_dim]`.
        # 12. Return both tensors in a dictionary.
        #
        # Plain meaning:
        # The online encoder should not learn by reading the target token.
        # It should build useful nearby context features and let the predictor
        # fill the hidden target region.

        # videos shape is [batch_size, clip_frames, 3, image_height, image_width]
        batch_size = videos.shape[0]
        flattened_videos = videos.flatten(start_dim=0, end_dim=1)                   # [batch_size * clip_frames, 3, image_height, image_width]

        patches = self.patch_embed(flattened_videos)                                # [batch_size * clip_frames, latent_dim, patches_per_side, patches_per_side]
        patches = patches.flatten(start_dim=2)                                      # [batch_size * clip_frames, latent_dim, 36]
        patches = patches.transpose(1, 2)                                           # [batch_size * clip_frames, 36, latent_dim]
        patches = patches.reshape(batch_size, self.total_tokens, self.latent_dim)   # [batch_size, 144, latent_dim]

        # visible_mask is shape [batch, clip_frames, patches_per_frame]
        flattened_visible_mask = None
        if visible_mask is not None:
            flattened_visible_mask = visible_mask.reshape(batch_size, self.total_tokens)  # [batch, total_tokens]
            # flattened_visible_mask with unsqueeze becomes [batch, total_tokens, 1]
            # hidden token is shape (1, 1, config.latent_dim)
            # patches is [batch_size, 144, latent_dim]
            # so where visible mask is True, then take the patch tensor value, otherwise use the hidden_token tensor value
            # mask is True  -> keep the real patch token
            # mask is False -> insert self.hidden_token
            patches  = torch.where(flattened_visible_mask.unsqueeze(dim=-1), patches, self.hidden_token)
            # Note:
            # PyTorch broadcasts the hidden token as if the same learned vector were available for every video and every patch position
            # It also broadcasts the condition: flat_visible_mask.unsqueeze(-1): [batch, 144, 1]
            # The final 1 is treated as if each True or False were repeated across all latent_dim values of that token.
            # False -> replace the whole token vector with self.hidden_token
            # True  -> keep the whole real patch-token vector


        patches = patches + self.pos_embed                                          # [batch_size, 144, latent_dim] + [1, 144, latent_dim]

        intermediate_features = None
        for layer_index, encoder_layer in enumerate(self.layers):
            if flattened_visible_mask is None:
                patches = encoder_layer(patches)
            else:
                patches = encoder_layer(
                    patches,
                    src_key_padding_mask=~flattened_visible_mask,
                )

            if layer_index == self.intermediate_layer_index:
                intermediate_features = patches

        if intermediate_features is None:
            raise RuntimeError("The intermediate encoder layer was not reached.")

        intermediate_features = self.intermediate_norm(intermediate_features)
        final_features = self.norm(patches)

        output_shape = (
            batch_size,
            self.clip_frames,
            self.patches_per_frame,
            self.latent_dim,
        )
        intermediate_features = intermediate_features.reshape(output_shape)
        final_features = final_features.reshape(output_shape)

        return {
            "final_tokens": final_features,
            "intermediate_tokens": intermediate_features,
        }

        # raise NotImplementedError("Run the dense video transformer encoder.")


class DenseVJEPAPredictor(nn.Module):
    """Tiny transformer predictor for final and intermediate dense patch features."""

    def __init__(self, config: DenseVJEPAConfig) -> None:
        super().__init__()
        # Where this fits:
        # In V-JEPA 2.1-style training, the predictor receives context token
        # features plus learned target mask tokens. It predicts target encoder
        # features for dense patch positions at two depths.
        #
        # Goal:
        # Build a small transformer predictor that keeps one output per
        # space-time patch token for:
        # - final dense features
        # - intermediate dense features
        #
        # Build these parts:
        # - `self.clip_frames = config.clip_frames`
        # - `self.patches_per_frame = config.patches_per_frame`
        # - `self.total_tokens = config.total_tokens`
        # - `self.mask_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))`
        # - `self.pos_embed = nn.Parameter(torch.zeros(1, config.total_tokens, config.latent_dim))`
        # - `self.transformer = nn.TransformerEncoder(...)`
        # - `self.norm = nn.LayerNorm(config.latent_dim)`
        # - `self.proj = nn.Linear(config.latent_dim, config.latent_dim)`
        # - `self.intermediate_proj = nn.Linear(config.latent_dim, config.latent_dim)`
        #
        # Use this transformer layer:
        # `nn.TransformerEncoderLayer(`
        # `    d_model=config.latent_dim,`
        # `    nhead=config.predictor_num_heads,`
        # `    dim_feedforward=config.predictor_mlp_dim,`
        # `    dropout=0.0,`
        # `    activation="gelu",`
        # `    batch_first=True,`
        # `)`

        self.clip_frames = config.clip_frames                   # 4
        self.patches_per_frame = config.patches_per_frame       # 36  i.e 6x6
        self.total_tokens = config.total_tokens                 # 144
        self.latent_dim = config.latent_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, config.total_tokens, config.latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model = config.latent_dim,
                                                    nhead = config.predictor_num_heads,
                                                    dim_feedforward = config.predictor_mlp_dim,
                                                    dropout = 0.0,
                                                    activation = "gelu",
                                                    batch_first = True)

        self.transformer = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = config.predictor_depth)

        self.norm = nn.LayerNorm(config.latent_dim)
        self.proj = nn.Linear(config.latent_dim, config.latent_dim)
        self.intermediate_proj = nn.Linear(config.latent_dim, config.latent_dim)

        # raise NotImplementedError("Build the dense V-JEPA predictor.")

    def forward(self, context_outputs: dict[str, torch.Tensor], target_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict final and intermediate dense token latents for the whole clip."""
        # Input:
        # - `context_outputs["final_tokens"]`: `[batch, clip_frames, patches_per_frame, latent_dim]`
        # - `context_outputs["intermediate_tokens"]`: `[batch, clip_frames, patches_per_frame, latent_dim]`
        # - `target_mask`: `[batch, clip_frames, patches_per_frame]`
        #
        # Output:
        # Return a dictionary with:
        # - `final_tokens`: `[batch, clip_frames, patches_per_frame, latent_dim]`
        # - `intermediate_tokens`: `[batch, clip_frames, patches_per_frame, latent_dim]`
        #
        # Steps:
        # 1. Read `context_tokens = context_outputs["final_tokens"]`.
        #    The predictor uses the final context tokens as its input.
        # 2. Flatten `context_tokens` to `[batch, total_tokens, latent_dim]`.
        # 3. Flatten `target_mask` to `[batch, total_tokens]`.
        # 4. Expand `self.mask_token` to `[batch, total_tokens, latent_dim]`.
        # 5. Use `torch.where(...)` so target positions receive the learned
        #    mask token and visible positions keep their context latents.
        # 6. Add `self.pos_embed`.
        # 7. Run the predictor transformer.
        # 8. Apply `self.norm`.
        # 9. Run `self.proj(...)` to get final token predictions.
        # 10. Run `self.intermediate_proj(...)` to get intermediate token predictions.
        # 11. Reshape both outputs back to
        #     `[batch, clip_frames, patches_per_frame, latent_dim]`.
        # 12. Return both tensors in a dictionary.

        context_tokens = context_outputs["final_tokens"]    # [batch, clip_frames, patches_per_frame, latent_dim]
        batch_size, clip_frames, patches_per_frame, latent_dim = context_tokens.shape

        flat_context_tokens = context_tokens.reshape(batch_size, clip_frames * patches_per_frame, latent_dim)  # [batch, total_tokens, latent_dim]

        flat_target_mask = target_mask.flatten(start_dim=1, end_dim=2)  # [batch, total_tokens]

        # (1, 1, latent_dim)  -->  (batch_size, 144, latent_dim)
        mask_tokens = self.mask_token.expand(size=(batch_size, clip_frames * patches_per_frame, latent_dim))  # [batch, total_tokens, latent_dim]

        tokens = torch.where(flat_target_mask.unsqueeze(dim=-1), mask_tokens, flat_context_tokens)  # [batch, total_tokens, latent_dim]

        tokens = tokens + self.pos_embed  # [batch, total_tokens, latent_dim]

        tokens = self.transformer(tokens)  # [batch, total_tokens, latent_dim]
        tokens = self.norm(tokens)  # [batch, total_tokens, latent_dim]

        final_tokens = self.proj(tokens)  # [batch, total_tokens, latent_dim]
        intermediate_tokens = self.intermediate_proj(tokens)  # [batch, total_tokens, latent_dim]

        output_shape = (batch_size, clip_frames, patches_per_frame, latent_dim)
        final_tokens = final_tokens.reshape(output_shape)  # [batch, clip_frames, patches_per_frame, latent_dim]
        intermediate_tokens = intermediate_tokens.reshape(output_shape)  # [batch, clip_frames, patches_per_frame, latent_dim]

        return {
            "final_tokens": final_tokens,
            "intermediate_tokens": intermediate_tokens,
        }

        # raise NotImplementedError("Run the dense V-JEPA predictor.")


def build_dense_vjepa_modules(config: DenseVJEPAConfig) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Build online encoder, EMA target encoder, and dense predictor."""
    # Goal:
    # Return:
    # - online encoder: trained by gradients
    # - target encoder: slow EMA copy, not trained by gradients
    # - predictor: trained by gradients
    #
    # Steps:
    # 1. Create `online_encoder = TinyDenseVideoTransformerEncoder(config)`.
    # 2. Create `target_encoder = TinyDenseVideoTransformerEncoder(config)`.
    # 3. Copy online weights into target with
    #    `target_encoder.load_state_dict(online_encoder.state_dict())`.
    # 4. Set `requires_grad=False` for every target encoder parameter.
    # 5. Create `predictor = DenseVJEPAPredictor(config)`.
    # 6. Move all modules to `device`.

    online_encoder = TinyDenseVideoTransformerEncoder(config)
    target_encoder = TinyDenseVideoTransformerEncoder(config)
    predictor = DenseVJEPAPredictor(config)

    target_encoder.load_state_dict(online_encoder.state_dict())

    for param in target_encoder.parameters():
        param.requires_grad = False

    online_encoder = online_encoder.to(device)
    target_encoder = target_encoder.to(device)
    predictor = predictor.to(device)

    return online_encoder, target_encoder, predictor
    #raise NotImplementedError("Build V-JEPA 2.1 dense modules.")


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
    # 3. Keep this inside `torch.no_grad()`.

    for target_param, encoder_param in zip(target_encoder.parameters(), online_encoder.parameters()):
        target_param.data = ema_decay * target_param.data + (1.0 - ema_decay) * encoder_param.data

    #raise NotImplementedError("Update the dense V-JEPA target encoder with EMA.")


def flatten_dense_tokens(token_latents: torch.Tensor) -> torch.Tensor:
    """Flatten dense video tokens while keeping batch rows separate."""
    # Goal:
    # Convert:
    # - `token_latents`: `[batch, clip_frames, patches_per_frame, latent_dim]`
    #
    # Into:
    # - `[batch, total_tokens, latent_dim]`
    #
    # Steps:
    # 1. Read `batch_size`, `clip_frames`, `patches_per_frame`, and
    #    `latent_dim` from `token_latents.shape`.
    # 2. Use `.view(batch_size, clip_frames * patches_per_frame, latent_dim)`.
    #
    # Where this fits:
    # V-JEPA 2.1-style dense loss uses all patch-token positions.
    # This helper gives the loss and retrieval code the simple shape they need.

    batch, clip_frames, patches_per_frame, latent_dim = token_latents.shape
    return token_latents.reshape(batch, clip_frames * patches_per_frame, latent_dim)  # [batch, total_tokens, latent_dim]
    #raise NotImplementedError("Flatten dense video tokens.")


def compute_vjepa_2_1_dense_loss(
    predicted_outputs: dict[str, torch.Tensor],
    target_outputs: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    intermediate_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute final-layer loss plus one intermediate-layer loss."""
    # Where this fits:
    # This is the small version of V-JEPA 2.1 deep self-supervision.
    # The full paper can supervise several layers. This exercise supervises:
    # - the final dense tokens
    # - one intermediate dense token layer
    #
    # Goal:
    # Return:
    # - one scalar loss tensor used for backprop
    # - a small metrics dictionary with Python floats
    #
    # Input:
    # - `predicted_outputs["final_tokens"]`: `[batch, clip_frames, patches_per_frame, latent_dim]`
    # - `target_outputs["final_tokens"]`: same shape
    # - `predicted_outputs["intermediate_tokens"]`: same shape
    # - `target_outputs["intermediate_tokens"]`: same shape
    #
    # Method:
    # 1. Compute:
    #    `final_loss = loss_fn(predicted_outputs["final_tokens"], target_outputs["final_tokens"].detach())`
    # 2. Compute:
    #    `intermediate_loss = loss_fn(predicted_outputs["intermediate_tokens"], target_outputs["intermediate_tokens"].detach())`
    # 3. Combine:
    #    `loss = final_loss + intermediate_loss_weight * intermediate_loss`
    # 4. Return the combined loss and metrics.
    #
    # Checks:
    # - both final tensors should have the same shape.
    # - both intermediate tensors should have the same shape.
    # - `intermediate_loss_weight=0.5` means the earlier-layer loss counts half as much as final loss.

    final_loss = loss_fn(predicted_outputs["final_tokens"], target_outputs["final_tokens"].detach())
    intermediate_loss = loss_fn(predicted_outputs["intermediate_tokens"], target_outputs["intermediate_tokens"].detach())
    loss = final_loss + intermediate_loss_weight * intermediate_loss

    return loss, {"final_loss": final_loss.item(), "intermediate_loss": intermediate_loss.item(),}
    #raise NotImplementedError("Compute V-JEPA 2.1 dense final and intermediate losses.")


def make_patch_error_grid(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    config: DenseVJEPAConfig,
) -> torch.Tensor:
    """Compute one MSE error value per patch token."""
    # Goal:
    # Return patch errors shaped `[batch, clip_frames, patches_per_frame]`.
    #
    # Input:
    # - `predicted_tokens`: usually `predicted_outputs["final_tokens"]`
    # - `target_tokens`: usually `target_outputs["final_tokens"]`
    # - both shaped `[batch, clip_frames, patches_per_frame, latent_dim]`
    #
    # Method:
    # Use squared error over the last dimension:
    # `patch_errors = (predicted_tokens - target_tokens.detach()).pow(2).mean(dim=-1)`
    #
    # Visual use:
    # A later helper can reshape one frame of this tensor from `[36]` to
    # `[6, 6]` and draw a heatmap over the frame.

    patch_errors = (predicted_tokens - target_tokens.detach()).pow(2).mean(dim=-1)  # [batch, clip_frames, patches_per_frame]
    return patch_errors

    # raise NotImplementedError("Create dense patch error values.")


def train_one_epoch(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: DenseVJEPAConfig,
) -> dict[str, float]:
    """Train small V-JEPA 2.1 dense features for one epoch."""
    # Where this fits:
    # This is the only step that changes model weights.
    #
    # Steps:
    # 1. Move `batch["video"]` to `device`.
    # 2. Create `target_mask` with `make_dense_target_mask(...)`.
    # 3. Create `visible_mask = ~target_mask`.
    # 4. Encode visible tokens with `context_outputs = online_encoder(videos, visible_mask)`.
    # 5. Encode clean clips with `target_outputs = target_encoder(videos)` inside `torch.no_grad()`.
    # 6. Predict dense token latents with:
    #    `predicted_outputs = predictor(context_outputs, target_mask)`.
    # 7. Compute final and intermediate losses with:
    #    `loss, loss_metrics = compute_vjepa_2_1_dense_loss(...)`.
    # 8. Backprop and step the optimizer.
    # 9. Update the target encoder with `update_ema_target_encoder(...)`.
    # 10. Track:
    #    - final dense MSE
    #    - intermediate dense MSE
    #    - all-token cosine similarity on final dense tokens

    online_encoder.train()
    predictor.train()
    target_encoder.eval()

    total_loss = 0
    final_dense_mse = 0
    intermediate_dense_mse = 0
    total_cosine_similarity = 0
    total_samples = 0

    progress = tqdm(loader, desc="train", leave=False) if tqdm is not None else loader

    for step, data in enumerate(progress, start=1):
        videos = data["video"].to(device)
        batch_size = videos.shape[0]

        target_mask = make_dense_target_mask(batch_size, config.clip_frames, config.patches_per_frame, 
                                            config.num_target_blocks, config.target_block_frames, config.target_block_side,
                                            device)
        visible_mask = ~target_mask

        context_dict = online_encoder(videos, visible_mask)

        with torch.no_grad():
            target_dict = target_encoder(videos)

        predicted_dict = predictor(context_dict, target_mask)

        loss, loss_metrics = compute_vjepa_2_1_dense_loss(predicted_dict, target_dict, loss_fn, config.intermediate_loss_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_ema_target_encoder(online_encoder, target_encoder, config.ema_decay)
        
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        final_dense_mse += loss_metrics["final_loss"] * batch_size
        intermediate_dense_mse += loss_metrics["intermediate_loss"] * batch_size

        #compute cosine sim
        target_latents = target_dict["final_tokens"]       # [batch, clip_frames, patches_per_frame, latent_dim]
        pred_latents = predicted_dict["final_tokens"]         # [batch, clip_frames, patches_per_frame, latent_dim]

        batch, clip_frames, patches_per_frame, latent_dim = target_latents.shape

        target_latents = target_latents.reshape(batch, clip_frames * patches_per_frame, latent_dim) # [batch, 144, latent_dim]
        pred_latents = pred_latents.reshape(batch, clip_frames * patches_per_frame, latent_dim)     # [batch, 144, latent_dim]

        cosine_similarity = F.cosine_similarity(pred_latents.detach(), target_latents.detach(), dim=-1,)  # [batch, total_tokens]

        mean_cosine_similarity = cosine_similarity.mean()  # one scalar
        total_cosine_similarity += mean_cosine_similarity.item() * batch_size

        if tqdm is not None:
            progress.set_postfix(
                step=f"{step}/{len(loader)}",
                loss=f"{loss.item():.4f}",
                dense_mse=f"{loss_metrics['final_loss']:.4f}",
                cos=f"{mean_cosine_similarity.item():.4f}",
            )

    return {
        "loss": total_loss / total_samples,
        "dense_latent_mse": final_dense_mse / total_samples,
        "intermediate_dense_latent_mse": intermediate_dense_mse / total_samples,
        "patch_cosine_similarity": total_cosine_similarity / total_samples,
    }

    # raise NotImplementedError("Train V-JEPA 2.1 dense features for one epoch.")


@torch.no_grad()
def evaluate_dense_vjepa(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    config: DenseVJEPAConfig,
) -> dict[str, float]:
    """Evaluate V-JEPA 2.1 dense latent metrics."""
    # Return:
    # - `dense_latent_mse`
    # - `intermediate_dense_latent_mse`
    # - `patch_cosine_similarity`
    # - `patch_retrieval_top1`
    #
    # Patch retrieval method:
    # 1. Create a fresh target mask.
    # 2. Run the same online encoder, target encoder, and predictor path used
    #    in training.
    # 3. Compute final and intermediate MSE with `compute_vjepa_2_1_dense_loss(...)`.
    # 4. Flatten final predicted and target tokens with `flatten_dense_tokens(...)`.
    # 5. Normalize predicted and target tokens with `F.normalize(..., dim=-1)`.
    # 6. For each clip, compute a similarity matrix:
    #    `[total_tokens, total_tokens]`.
    # 7. The correct target for row `i` is column `i`.
    #
    # Patch error method:
    # - use `make_patch_error_grid(...)`
    # - average it for dense MSE
    # - keep one batch of errors for the heatmap visual

    online_encoder.eval()
    target_encoder.eval()
    predictor.eval()

    total_dense_mse = 0.0
    total_intermediate_dense_mse = 0.0
    total_cosine_similarity = 0.0
    total_samples = 0
    correct_retrievals = 0
    total_retrievals = 0

    for data in loader:
        videos = data["video"].to(device)
        batch_size = videos.shape[0]

        target_mask = make_dense_target_mask(
            batch_size,
            config.clip_frames,
            config.patches_per_frame,
            config.num_target_blocks,
            config.target_block_frames,
            config.target_block_side,
            videos.device,
        )
        visible_mask = ~target_mask

        context_outputs = online_encoder(videos, visible_mask)
        target_outputs = target_encoder(videos)
        predicted_outputs = predictor(context_outputs, target_mask)

        _, loss_metrics = compute_vjepa_2_1_dense_loss(
            predicted_outputs,
            target_outputs,
            loss_fn,
            config.intermediate_loss_weight,
        )

        patch_errors = make_patch_error_grid(
            predicted_outputs["final_tokens"],
            target_outputs["final_tokens"],
            config,
        )  # [batch, clip_frames, patches_per_frame]
        dense_mse = patch_errors.mean()

        predicted_tokens = flatten_dense_tokens(predicted_outputs["final_tokens"])  # [batch, total_tokens, latent_dim]
        target_tokens = flatten_dense_tokens(target_outputs["final_tokens"])  # [batch, total_tokens, latent_dim]

        cosine_similarity = F.cosine_similarity(
            predicted_tokens,
            target_tokens,
            dim=-1,
        ).mean()

        predicted_features = F.normalize(predicted_tokens, dim=-1)  # [batch, total_tokens, latent_dim]
        target_features = F.normalize(target_tokens, dim=-1)  # [batch, total_tokens, latent_dim]
        similarity = predicted_features @ target_features.transpose(1, 2)  # [batch, total_tokens, total_tokens]

        retrieved_indices = similarity.argmax(dim=-1)  # [batch, total_tokens]
        correct_indices = torch.arange(
            predicted_tokens.shape[1],
            device=videos.device,
        ).unsqueeze(0)  # [1, total_tokens]

        correct_retrievals += (retrieved_indices == correct_indices).sum().item()
        total_retrievals += retrieved_indices.numel()

        total_dense_mse += dense_mse.item() * batch_size
        total_intermediate_dense_mse += loss_metrics["intermediate_loss"] * batch_size
        total_cosine_similarity += cosine_similarity.item() * batch_size
        total_samples += batch_size

    return {
        "dense_latent_mse": total_dense_mse / total_samples,
        "intermediate_dense_latent_mse": total_intermediate_dense_mse / total_samples,
        "patch_cosine_similarity": total_cosine_similarity / total_samples,
        "patch_retrieval_top1": correct_retrievals / total_retrievals,
    }

    # raise NotImplementedError("Evaluate V-JEPA 2.1 dense metrics.")


def save_dense_vjepa_visuals(
    history: list[dict[str, float]],
    sample_batch: dict[str, torch.Tensor],
    patch_errors: torch.Tensor,
    config: DenseVJEPAConfig,
) -> dict[str, str]:
    """Save frame strips, loss curves, and a patch error heatmap."""
    # Save:
    # - `clean_clip_strip.png`
    # - `target_mask_strip.png`
    # - `training_curve.png`
    # - `patch_error_heatmap.png`

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_paths: dict[str, str] = {}
    videos = sample_batch["video"].to(device)  # [batch, clip_frames, 3, image_size, image_size]

    clean_clip_path = output_dir / "clean_clip_strip.png"
    fig, axes = plt.subplots(
        1,
        config.clip_frames,
        figsize=(2.4 * config.clip_frames, 2.4),
    )
    if config.clip_frames == 1:
        axes = [axes]

    for frame_index in range(config.clip_frames):
        frame = videos[0, frame_index].detach().cpu().permute(1, 2, 0)  # [image_size, image_size, 3]
        axes[frame_index].imshow(frame)
        axes[frame_index].set_title(f"frame {frame_index}")
        axes[frame_index].axis("off")

    fig.suptitle("Clean video clip")
    fig.tight_layout()
    fig.savefig(clean_clip_path, dpi=160)
    plt.close(fig)
    visual_paths["clean_clip_strip"] = str(clean_clip_path)

    target_mask = make_dense_target_mask(
        batch_size=videos.shape[0],
        clip_frames=config.clip_frames,
        patches_per_frame=config.patches_per_frame,
        num_target_blocks=config.num_target_blocks,
        target_block_frames=config.target_block_frames,
        target_block_side=config.target_block_side,
        device=videos.device,
    )  # [batch, clip_frames, patches_per_frame]

    target_mask_path = output_dir / "target_mask_strip.png"
    fig, axes = plt.subplots(
        1,
        config.clip_frames,
        figsize=(2.4 * config.clip_frames, 2.4),
    )
    if config.clip_frames == 1:
        axes = [axes]

    for frame_index in range(config.clip_frames):
        mask_grid = target_mask[0, frame_index].detach().cpu().reshape(
            config.patches_per_side,
            config.patches_per_side,
        )  # [patches_per_side, patches_per_side]
        axes[frame_index].imshow(mask_grid, cmap="gray_r", vmin=0, vmax=1)
        axes[frame_index].set_title(f"frame {frame_index}")
        axes[frame_index].set_xticks([])
        axes[frame_index].set_yticks([])

    fig.suptitle("Target mask: dark patches are hidden")
    fig.tight_layout()
    fig.savefig(target_mask_path, dpi=160)
    plt.close(fig)
    visual_paths["target_mask_strip"] = str(target_mask_path)

    training_curve_path = output_dir / "training_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = list(range(1, len(history) + 1))
    train_values = [
        item.get("train_dense_latent_mse", item.get("dense_latent_mse"))
        for item in history
    ]
    val_values = [item.get("val_dense_latent_mse") for item in history]

    if any(value is not None for value in train_values):
        ax.plot(epochs, train_values, marker="o", label="train dense MSE")
    if any(value is not None for value in val_values):
        ax.plot(epochs, val_values, marker="o", label="validation dense MSE")

    ax.set_title("Dense patch prediction error")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(training_curve_path, dpi=160)
    plt.close(fig)
    visual_paths["training_curve"] = str(training_curve_path)

    patch_error_path = output_dir / "patch_error_heatmap.png"
    error_grids = patch_errors[0].detach().cpu().reshape(
        config.clip_frames,
        config.patches_per_side,
        config.patches_per_side,
    )  # [clip_frames, patches_per_side, patches_per_side]
    max_error = max(float(error_grids.max().item()), 1e-8)

    fig, axes = plt.subplots(
        1,
        config.clip_frames,
        figsize=(2.8 * config.clip_frames, 2.6),
    )
    if config.clip_frames == 1:
        axes = [axes]

    heatmap = None
    for frame_index in range(config.clip_frames):
        heatmap = axes[frame_index].imshow(
            error_grids[frame_index],
            cmap="magma",
            vmin=0,
            vmax=max_error,
        )
        axes[frame_index].set_title(f"frame {frame_index}")
        axes[frame_index].set_xlabel("patch column")
        axes[frame_index].set_ylabel("patch row")

    fig.suptitle("Dense prediction MSE by patch")
    if heatmap is not None:
        fig.colorbar(heatmap, ax=axes, label="MSE")
    fig.subplots_adjust(top=0.80, right=0.88, wspace=0.35)
    fig.savefig(patch_error_path, dpi=160)
    plt.close(fig)
    visual_paths["patch_error_heatmap"] = str(patch_error_path)

    return visual_paths

    # raise NotImplementedError("Save V-JEPA 2.1 dense visuals.")


def run_dense_vjepa_experiment(config: DenseVJEPAConfig) -> dict[str, object]:
    """Run the full small V-JEPA 2.1 dense exercise."""
    # Return a dictionary with:
    # - `train_dense_latent_mse`
    # - `train_intermediate_dense_latent_mse`
    # - `val_dense_latent_mse`
    # - `val_intermediate_dense_latent_mse`
    # - `test_dense_latent_mse`
    # - `test_intermediate_dense_latent_mse`
    # - `test_patch_cosine_similarity`
    # - `test_patch_retrieval_top1`
    # - `visual_paths`
    #
    # Main steps:
    # 1. Load dataloaders with `load_dense_vjepa_dataloaders(config)`.
    # 2. Build modules with `build_dense_vjepa_modules(config)`.
    # 3. Create an optimizer over online encoder and predictor parameters only.
    # 4. Train for `config.epochs`.
    # 5. Evaluate validation and test metrics.
    # 6. Save visuals.

    set_seed(config.random_state)
    loaders = load_dense_vjepa_dataloaders(config)
    online_encoder, target_encoder, predictor = build_dense_vjepa_modules(config)

    optimizer = torch.optim.AdamW(
        list(online_encoder.parameters()) + list(predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    final_train_metrics: dict[str, float] = {}
    final_val_metrics: dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        final_train_metrics = train_one_epoch(
            online_encoder,
            target_encoder,
            predictor,
            loaders["train"],
            optimizer,
            loss_fn,
            config,
        )
        final_val_metrics = evaluate_dense_vjepa(
            online_encoder,
            target_encoder,
            predictor,
            loaders["val"],
            loss_fn,
            config,
        )

        history.append(
            {
                "train_dense_latent_mse": final_train_metrics["dense_latent_mse"],
                "train_intermediate_dense_latent_mse": final_train_metrics["intermediate_dense_latent_mse"],
                "train_patch_cosine_similarity": final_train_metrics["patch_cosine_similarity"],
                "val_dense_latent_mse": final_val_metrics["dense_latent_mse"],
                "val_intermediate_dense_latent_mse": final_val_metrics["intermediate_dense_latent_mse"],
                "val_patch_cosine_similarity": final_val_metrics["patch_cosine_similarity"],
                "val_patch_retrieval_top1": final_val_metrics["patch_retrieval_top1"],
            }
        )

        print(f"\nepoch {epoch}/{config.epochs}")
        print(
            f"  train | combined loss {final_train_metrics['loss']:.4f} | "
            f"dense MSE {final_train_metrics['dense_latent_mse']:.4f} | "
            f"intermediate MSE {final_train_metrics['intermediate_dense_latent_mse']:.4f} | "
            f"patch cosine {final_train_metrics['patch_cosine_similarity']:.4f}"
        )
        print(
            f"  val   | dense MSE {final_val_metrics['dense_latent_mse']:.4f} | "
            f"intermediate MSE {final_val_metrics['intermediate_dense_latent_mse']:.4f} | "
            f"patch cosine {final_val_metrics['patch_cosine_similarity']:.4f} | "
            f"retrieval top-1 {100.0 * final_val_metrics['patch_retrieval_top1']:.2f}%"
        )

    test_metrics = evaluate_dense_vjepa(
        online_encoder,
        target_encoder,
        predictor,
        loaders["test"],
        loss_fn,
        config,
    )

    sample_batch = next(iter(loaders["test"]))
    sample_videos = sample_batch["video"].to(device)
    sample_target_mask = make_dense_target_mask(
        batch_size=sample_videos.shape[0],
        clip_frames=config.clip_frames,
        patches_per_frame=config.patches_per_frame,
        num_target_blocks=config.num_target_blocks,
        target_block_frames=config.target_block_frames,
        target_block_side=config.target_block_side,
        device=sample_videos.device,
    )
    sample_visible_mask = ~sample_target_mask

    online_encoder.eval()
    target_encoder.eval()
    predictor.eval()
    with torch.no_grad():
        sample_context_outputs = online_encoder(sample_videos, sample_visible_mask)
        sample_target_outputs = target_encoder(sample_videos)
        sample_predicted_outputs = predictor(sample_context_outputs, sample_target_mask)
        patch_errors = make_patch_error_grid(
            sample_predicted_outputs["final_tokens"],
            sample_target_outputs["final_tokens"],
            config,
        )

    visual_paths = save_dense_vjepa_visuals(
        history,
        sample_batch,
        patch_errors,
        config,
    )

    random_retrieval_top1 = 1.0 / config.total_tokens
    print("\nFinal V-JEPA 2.1-style metrics")
    print(f"train dense MSE: {final_train_metrics['dense_latent_mse']:.4f} (lower is better)")
    print(f"validation dense MSE: {final_val_metrics['dense_latent_mse']:.4f} (lower is better)")
    print(f"test dense MSE: {test_metrics['dense_latent_mse']:.4f} (lower is better)")
    print(
        f"test intermediate dense MSE: "
        f"{test_metrics['intermediate_dense_latent_mse']:.4f} (lower is better)"
    )
    print(f"test patch cosine similarity: {test_metrics['patch_cosine_similarity']:.4f} (higher is better)")
    print(
        f"test patch retrieval top-1: {100.0 * test_metrics['patch_retrieval_top1']:.2f}% "
        f"(higher is better; random is about {100.0 * random_retrieval_top1:.2f}%)"
    )
    print(f"visuals saved in: {config.output_dir}")

    return {
        "train_dense_latent_mse": final_train_metrics["dense_latent_mse"],
        "train_intermediate_dense_latent_mse": final_train_metrics["intermediate_dense_latent_mse"],
        "val_dense_latent_mse": final_val_metrics["dense_latent_mse"],
        "val_intermediate_dense_latent_mse": final_val_metrics["intermediate_dense_latent_mse"],
        "test_dense_latent_mse": test_metrics["dense_latent_mse"],
        "test_intermediate_dense_latent_mse": test_metrics["intermediate_dense_latent_mse"],
        "test_patch_cosine_similarity": test_metrics["patch_cosine_similarity"],
        "test_patch_retrieval_top1": test_metrics["patch_retrieval_top1"],
        "visual_paths": visual_paths,
    }

    # raise NotImplementedError("Run V-JEPA 2.1 dense training, evaluation, and visuals.")


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
    """Run the V-JEPA 2.1 dense scaffold."""
    config = DenseVJEPAConfig()
    set_seed(config.random_state)
    result, ok = report_task_progress("run_dense_vjepa_experiment", run_dense_vjepa_experiment, config)
    if ok:
        print(result)


if __name__ == "__main__":
    main()
