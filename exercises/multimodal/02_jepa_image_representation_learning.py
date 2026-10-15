"""
============================================================
EXERCISE 02: JEPA IMAGE REPRESENTATION LEARNING
============================================================

WHAT YOU ARE BUILDING
You are building a small image JEPA learner that stays close to the real
training pattern while still being small enough for a normal bootcamp GPU.

JEPA means Joint Embedding Predictive Architecture.

Plain meaning:
- take a real image
- choose which image patches the context encoder is allowed to use
- let the target encoder see the original image
- train a predictor to guess the hidden target patch representations
- compare hidden vectors, not raw pixels

This file is only about image JEPA.
Video JEPA and vision-language JEPA are split into:
- `exercises/multimodal/03_vjepa_video_representation_learning.py`
- `exercises/multimodal/04_vljepa_image_text_representation_learning.py`

WHY THIS EXERCISE EXISTS
Earlier vision exercises mostly train a model to predict a class label.
JEPA-style learning trains a model to predict a hidden summary of another view.

Simple example:
- image: a real CIFAR10 image
- target patches: a few hidden 8 by 8 patches
- visible patches: the patches the online encoder is allowed to read
- target encoder: reads the clean image and creates target patch vectors
- context encoder: reads only the visible patch tokens and creates context vectors
- predictor: uses the context vectors to predict the target vectors

The model is not asked to repaint missing pixels.
It is asked to predict useful hidden vectors for the missing patches.

REAL SYSTEM FIRST
In a larger JEPA-style image system:

1. take real images
2. split each image into patches
3. choose target patches to predict
4. keep those target patches out of the context encoder attention
5. encode the visible patch tokens with an online encoder (context encoder)
6. encode the clean image with a target encoder
7. predict the target patch representations from the context representations
8. train with a representation-space loss
9. update the target encoder with EMA
10. check whether the learned features are useful

EMA means exponential moving average.
Plain meaning:

```text
The target encoder is a slow-moving copy of the online encoder ( also called context encoder).
After each optimizer step, the target encoder moves a little closer to the online encoder.
```

This exercise uses the same workflow on a small real image dataset.

WHAT IS SCALED DOWN AND WHY
This exercise should stay as close to the real system as possible, but it must
run on limited GPU resources.

Scaled down:
- CIFAR10 instead of a large image pretraining dataset
- 32 by 32 images instead of high-resolution images
- 8 by 8 patches instead of many small ViT patches
- a tiny ViT-style encoder instead of a large ViT
- a tiny transformer predictor instead of the larger I-JEPA predictor
- a short training run instead of long pretraining
- single-process training instead of distributed training

Not scaled down:
- real images
- patch-token masking
- online encoder
- EMA target encoder
- predictor
- latent prediction loss
- feature usefulness check with a linear probe

DATASET
Use a real dataset, not random tensors and not generated images.

Default dataset:
- `torchvision.datasets.CIFAR10`
- real 32 by 32 color images
- 10 object classes

The class labels are not used for the JEPA loss.
They are used later for the linear-probe check.

Mapping from the real system to this exercise:

    real JEPA system                    this exercise
    ---------------------------------------------------------------
    large image dataset                 CIFAR10
    image patches                       8 by 8 patches
    target block sampling               a few target patch indices
    context patch tokens                target patches hidden from attention
    online encoder (context encoder)    tiny ViT-style patch transformer
    EMA target encoder                  slow copy of the online encoder
    predictor                           tiny transformer with learned target mask tokens
    representation loss                 target-patch latent MSE and cosine similarity
    feature usefulness check            linear probe on frozen image features
    visual checks                       masked images, target masks, curves, PCA scatter

INPUTS AND OUTPUTS
Each dataloader batch should contain:

    image: [batch, 3, 32, 32]
    label: [batch]

During training, you create:

    target_mask:   [batch, num_patches]
    visible_mask:  [batch, num_patches]

With the default settings:

    patch_size = 8
    image_size = 32
    patch grid = 4 by 4
    num_patches = 16

The encoder outputs:

    patch_latents: [batch, 16, latent_dim]

The predictor outputs:

    predicted_patch_latents: [batch, 16, latent_dim]

The loss only uses the target patches:

    predicted_target_latents: [batch, num_target_patches, latent_dim]
    target_latents:           [batch, num_target_patches, latent_dim]

LOSS FUNCTION
Use latent-space mean squared error on the selected target patches:

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
- linear-probe accuracy, higher is better

VISUALS TO SAVE
Save to `outputs/jepa_30/`:
- original and masked image examples
- target patch mask examples
- training curve
- latent PCA scatter, colored by CIFAR10 class

STEP-BY-STEP TODO
1. Load CIFAR10 and make one shared train/validation/test split.
2. Create random target patch masks.
3. Create visible patch masks from target masks.
4. Build the online/context tiny ViT encoder.
5. Build the EMA target patch encoder.
6. Build the tiny transformer predictor.
7. Train with target-patch latent MSE.
8. Update the target encoder with EMA after each optimizer step.
9. Evaluate latent MSE and cosine similarity.
10. Train a linear probe on frozen features.
11. Save visual checks.

DEFINITION OF DONE
You are done when:
- the exercise uses real CIFAR10 images
- the JEPA model uses patch masks, not only fixed image halves
- the target encoder is updated with EMA
- validation and test latent metrics are printed
- linear-probe accuracy is printed
- visual outputs are saved
- you can explain why the model predicts hidden patch vectors instead of pixels
"""

from __future__ import annotations

from dataclasses import dataclass
from json import encoder
from pathlib import Path
import select
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed


# Learning order:
# 1. Load real CIFAR10 images.
# 2. Make target patch masks.
# 3. Build the visible patch mask for the online encoder.
# 4. Build the online encoder, EMA target encoder, and predictor.
# 5. Train on target-patch latent MSE.
# 6. Update the target encoder with EMA.
# 7. Evaluate latent metrics and linear-probe accuracy.
# 8. Save visuals.


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ImageJEPAConfig:
    """Settings for the small image JEPA run."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    output_dir: str = str(BOOTCAMP_ROOT / "outputs" / "jepa_30")
    image_size: int = 32
    channels: int = 3
    patch_size: int = 8
    latent_dim: int = 96
    encoder_depth: int = 2
    encoder_num_heads: int = 4
    encoder_mlp_dim: int = 192
    predictor_depth: int = 1
    predictor_num_heads: int = 4
    predictor_mlp_dim: int = 192
    num_target_patches: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    probe_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 16
    linear_probe_epochs: int = 10
    ema_decay: float = 0.99
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    random_state: int = 30

    def __post_init__(self) -> None:
        """Catch config values that would break the patch transformer."""
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must divide evenly by patch_size.")
        if self.latent_dim % self.encoder_num_heads != 0:
            raise ValueError("latent_dim must divide evenly by encoder_num_heads.")
        if self.latent_dim % self.predictor_num_heads != 0:
            raise ValueError("latent_dim must divide evenly by predictor_num_heads.")
        if self.num_target_patches >= self.num_patches:
            raise ValueError("num_target_patches must be smaller than num_patches.")

    @property
    def patches_per_side(self) -> int:
        """Return how many patches fit along one image side."""
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Return the total number of image patches."""
        return self.patches_per_side * self.patches_per_side


class CIFARImageDataset(Dataset):
    """Wrap CIFAR10 so each item has clear dictionary keys."""

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.base_dataset[index]
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_cifar_image_dataloaders(config: ImageJEPAConfig) -> dict[str, DataLoader]:
    """Load real CIFAR10 images and return train, validation, and test loaders."""
    # Where this fits:
    # This builds the real image data for the JEPA task.
    #
    # Goal:
    # Return dataloaders with batches containing:
    # - `image`: `[batch, 3, 32, 32]`
    # - `label`: `[batch]`
    #
    # Method:
    # Use plain CIFAR10 images with `torchvision.transforms.ToTensor()`.
    # This keeps the exercise closer to the original I-JEPA idea, where the
    # learning signal comes from predicting hidden representations rather than
    # from hand-made image augmentations.
    #
    # Steps:
    # 1. Load CIFAR10 train with a plain tensor transform.
    # 2. Split CIFAR10 train indices into train and validation indices.
    # 3. Use the official CIFAR10 test split as the test dataset.
    # 4. Wrap each dataset with `CIFARImageDataset`.
    # 5. Build DataLoaders.
    # 6. Use `shuffle=True` only for the train loader.
    #
    # Checks:
    # - one batch should have `image.shape == [batch, 3, 32, 32]`.
    # - one batch should have `label.shape == [batch]`.
    transform = torchvision.transforms.ToTensor()

    train_base = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=True,
        transform=transform,
        download=True,
    )
    test_base = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=False,
        transform=transform,
        download=True,
    )

    train_pool_size = len(train_base)
    train_size = int(train_pool_size * config.train_fraction / (config.train_fraction + config.val_fraction))
    val_size = train_pool_size - train_size
    generator = torch.Generator().manual_seed(config.random_state)
    train_indices, val_indices = torch.utils.data.random_split(
        range(train_pool_size),
        [train_size, val_size],
        generator=generator,
    )

    train_ds = CIFARImageDataset(torch.utils.data.Subset(train_base, train_indices.indices))
    val_ds = CIFARImageDataset(torch.utils.data.Subset(train_base, val_indices.indices))
    test_ds = CIFARImageDataset(test_base)

    return {
        "train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
        "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
        "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size),
    }
    
    #raise NotImplementedError("Load CIFAR10 and return JEPA image dataloaders.")


def make_target_patch_mask(
    batch_size: int,
    num_patches: int,
    num_target_patches: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Create a boolean mask that marks which patches should be predicted."""
    # Where this fits:
    # Real I-JEPA systems choose target blocks before the online encoder runs.
    # This exercise uses random target patch indices because CIFAR10 has only
    # a 4 by 4 patch grid.
    #
    # Goal:
    # Return `target_mask` with shape `[batch_size, num_patches]`.
    # `True` means "this patch is a target patch to predict".
    #
    # Method:
    # Use `torch.randperm(num_patches, device=device)` for each image.
    #
    # Steps:
    # 1. Create a boolean tensor full of `False`.
    # 2. For each batch row, sample `num_target_patches` patch indices.
    # 3. Set those positions to `True`.
    #
    # Checks:
    # - `target_mask.shape` should be `[batch_size, num_patches]`.
    # - each row should have exactly `num_target_patches` true values.


    target_mask = torch.zeros(size=[batch_size, num_patches], dtype=torch.bool, device=device)

    # for i in range(batch_size):
    #     rand_ints = torch.randperm(num_patches, device = device)
    #     for j in range(num_target_patches):
    #         index = rand_ints[j]
    #         target_mask[i][index] = torch.tensor(1, dtype=torch.bool, device = device)

    # alternate and better way:
    for batch_index in range(batch_size):
        patch_ids = torch.randperm(num_patches, device=device)      # num_patches is the total number of patches in the image i.e 16 in this case
        target_patch_ids = patch_ids[:num_target_patches]           # num_target_patches is the number of target patches to predict i.e 4 in this case
        target_mask[batch_index, target_patch_ids] = True

    return target_mask  # shape [batch_size, num_patches]

    #raise NotImplementedError("Create random target patch masks.")


def apply_patch_mask(
    images: torch.Tensor,
    target_mask: torch.Tensor,
    patch_size: int,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """Return a visual copy of images with target patches hidden."""
    # Where this fits:
    # This helper is mainly for visual checks.
    # The model should use token-level masking in `TinyViTImageEncoder.forward`.
    # The masked image is still useful because it lets you see which patches
    # were hidden from the online encoder.
    #
    # Goal:
    # Return `masked_images` with the same shape as `images`.
    # Every target patch should be filled with `mask_value`.
    #
    # Input:
    # - `images`: `[batch, 3, 32, 32]`
    # - `target_mask`: `[batch, 16]` when `patch_size=8`
    #
    # Method:
    # Treat the 32 by 32 image as a 4 by 4 grid of 8 by 8 patches.
    # Patch index `0` is row 0, column 0.
    # Patch index `1` is row 0, column 1.
    #
    # Steps:
    # 1. Clone `images` so the original clean image is kept.
    # 2. Loop over patch rows and patch columns.
    # 3. Convert each patch position to a patch index.
    # 4. For images where `target_mask[:, patch_index]` is true, fill that patch.
    #
    # Checks:
    # - output shape should match input shape.
    # - non-target patches should stay unchanged.

    masked_images = images.clone()

    batch_size, channels, image_height, image_width = images.shape

    patches_per_side = image_height // patch_size

    # target_mask has shape [batch_size, num_patches] of type bool
    # `True` means "this patch is a target patch to predict"
    # so we need to fill that with mask_value

    # image is 32 x 32
    # patch grid is 4 x 4
    # num patches is 16

    # patch ids:
    # 0  1  2  3
    # 4  5  6  7
    # 8  9  10 11
    # 12 13 14 15

    for patch_row in range(patches_per_side):
        for patch_col in range(patches_per_side):
            patch_index = patch_row * patches_per_side + patch_col

            row_start = patch_row * patch_size
            row_end = row_start + patch_size
            col_start = patch_col * patch_size
            col_end = col_start + patch_size

            images_to_mask = target_mask[:, patch_index]
            masked_images[images_to_mask, :, row_start:row_end, col_start:col_end,] = mask_value
            # `images_to_mask` has shape `[batch_size]`.
            # It is one True/False value for each image in the batch.
            # PyTorch lets a boolean tensor select rows from a tensor.
            # Example:
            # - if `images_to_mask == [True, False, True]`
            # - this line selects image 0 and image 2 from `masked_images`
            # - it leaves image 1 unchanged
            # The row and column slices choose the same patch location for the
            # selected images, but each image is selected independently by its
            # own True/False value in `target_mask[:, patch_index]`.

            #the above line is equivalent to:
            # for image_index in range(batch_size):
            # if target_mask[image_index, patch_index]:
            #     masked_images[
            #         image_index,
            #         :,
            #         row_start:row_end,
            #         col_start:col_end,
            #     ] = mask_value

    return masked_images
    #raise NotImplementedError("Create masked images for JEPA visual checks.")


class TinyViTImageEncoder(nn.Module):
    """Tiny ViT-style encoder that returns one latent vector per image patch."""

    def __init__(self, config: ImageJEPAConfig) -> None:
        super().__init__()
        # Where this fits:
        # This class is used for both the online encoder and EMA target encoder.
        # In the real I-JEPA paper, these encoders are Vision Transformers.
        # This is a much smaller ViT-style version.
        #
        # Goal:
        # Turn `[batch, 3, 32, 32]` images into `[batch, 16, latent_dim]`
        # patch latents.
        #
        # Method:
        # Build these parts:
        # - `self.num_patches = config.num_patches`
        # - `self.latent_dim = config.latent_dim`
        # - `self.patch_embed = nn.Conv2d(config.channels, config.latent_dim, kernel_size=config.patch_size, stride=config.patch_size)`
        # - `self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.latent_dim))`
        # - `self.hidden_patch_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))`
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
        # - `patch_embed` turns each 8 by 8 patch into one token vector.
        # - `pos_embed` tells the transformer where each patch came from.
        # - `hidden_patch_token` fills target positions for the online encoder.
        # - `TransformerEncoder` lets patches pass information to each other.
        # - `LayerNorm` keeps the output scale steadier.
        #
        # Checks:
        # - `TinyViTImageEncoder(config)(torch.randn(4, 3, 32, 32)).shape`
        #   should be `[4, 16, config.latent_dim]`.

        self.num_patches = config.num_patches
        self.latent_dim = config.latent_dim

        # This is the usual ViT patch embedding trick.
        # A Conv2d with `kernel_size=patch_size` looks at one whole patch at a time.
        # A Conv2d with `stride=patch_size` jumps by one full patch, so the patches do not overlap.
        # For a `[batch, 3, 32, 32]` image and `patch_size=8`, this gives `[batch, latent_dim, 4, 4]`.
        # Each of the 4 by 4 positions is one learned vector for one image patch.
        self.patch_embed = nn.Conv2d(in_channels = config.channels, out_channels=config.latent_dim, kernel_size=config.patch_size, stride=config.patch_size)

        # This is a learned position embedding table.
        # Shape `[1, num_patches, latent_dim]` means there is one learned vector
        # for each patch position. For this exercise, that is one vector for
        # patch 0, one for patch 1, and so on up to patch 15.
        # There is no `torch.arange(...)` here because the position id is
        # implied by the row in this tensor:
        # - `self.pos_embed[:, 0, :]` is added to patch token 0
        # - `self.pos_embed[:, 1, :]` is added to patch token 1
        # - `self.pos_embed[:, 15, :]` is added to patch token 15
        # An `arange` is used more often with `nn.Embedding`, where integer
        # position ids are looked up in a table.
        # Example of that other style:
        # - `pos_embed = nn.Embedding(num_patches, latent_dim)`
        # - `position_ids = torch.arange(num_patches)`
        # - `position_vectors = pos_embed(position_ids)`
        # Here, `position_ids` would be `[0, 1, 2, ..., 15]`, and `nn.Embedding`
        # would return one vector for each id.
        # This file uses the simpler ViT style instead: the position vector table
        # is already shaped like `[1, num_patches, latent_dim]`, so it can be
        # added directly to `patch_tokens` without a lookup step.
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.latent_dim))        # [1, 16, 96] position embedding table for the online encoder

        self.hidden_patch_token = nn.Parameter(torch.zeros(size=[1, 1, config.latent_dim]))        # [1, 1, 96] hidden patch token for the online encoder

        encoder_layer = nn.TransformerEncoderLayer(d_model = config.latent_dim, 
                                                    nhead=config.encoder_num_heads, 
                                                    dim_feedforward=config.encoder_mlp_dim, 
                                                    dropout=0.0,
                                                    activation='gelu',
                                                    batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers = config.encoder_depth)

        self.norm = nn.LayerNorm(config.latent_dim)

        #raise NotImplementedError("Build the tiny ViT image encoder.")

    def forward(self, images: torch.Tensor, visible_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode images into patch latent vectors."""
        # Goal:
        # Run the tiny ViT encoder.
        # If `visible_mask` is given, target patches should be hidden from the
        # online encoder at the token level.
        # visible_mask means:
        # True = this patch is visible and the online encoder may use it
        # False = this patch is hidden target
        #
        # Input:
        # - `images`: `[batch, 3, 32, 32]`
        # - `visible_mask`: `[batch, 16]`, where `True` means "the online
        #   encoder may use this patch".
        #
        # Steps:
        # 1. Run `patch_tokens = self.patch_embed(images)`.
        #    Shape: `[batch, latent_dim, 4, 4]`.
        # 2. Flatten the patch grid with:
        #    `patch_tokens = patch_tokens.flatten(2).transpose(1, 2)`.
        #    Shape: `[batch, 16, latent_dim]`.
        # 3. If `visible_mask is not None`, replace hidden target positions
        #    with `self.hidden_patch_token`.
        #    Use:
        #    `hidden_tokens = self.hidden_patch_token.expand(batch_size, self.num_patches, self.latent_dim)`.
        #    Then:
        #    `patch_tokens = torch.where(visible_mask.unsqueeze(-1), patch_tokens, hidden_tokens)`.
        # 4. Add `self.pos_embed`.
        # 5. Run `self.transformer(...)`.
        #    If `visible_mask is not None`, pass:
        #    `src_key_padding_mask=~visible_mask`.
        #    In this PyTorch mask, `True` means "ignore this token as a key".
        # 6. Return `self.norm(patch_tokens)`.
        #
        # Why this is closer to I-JEPA:
        # The online encoder is guided by patch tokens and a context mask.
        # It is not mainly learning from a pixel-masked image.

        batch_size = images.shape[0]

        patch_tokens = self.patch_embed(images)                         # [batch_size, latent_dim, num_patches, num_patches] = [batch_size, 96, 4, 4]

        patch_tokens = torch.flatten(patch_tokens, start_dim=2)         # [batch_size, 96, 16]   num_patches * num_patches = 4 * 4 = 16 patches per image
        patch_tokens = torch.transpose(patch_tokens, dim0=1, dim1=2)    # [batch_size, 16, 96]  each patch has a feature vector of size latent_dim

        # if the visible mask is not None, then we need to hide the target patches from the online encoder (context encoder)
        # self.hidden_patch_token is learnable because it stands in for a missing patch.
        # In the online encoder, hidden target patches are not allowed to show their real image content. 
        # But the transformer still needs some token at that position so it knows:
        # there is a patch here, but its content is hidden
        # Why learnable?
        # Because the model can learn the best “missing patch” placeholder during training.
        # If it stayed fixed as all zeros, every hidden patch would always start as a plain zero vector. 
        # That can work, but it gives the model a weaker placeholder.
        # A learnable hidden token can become something like:
        # a general vector meaning "unknown image patch here" instead of just a plain zero vector.
        # This is more useful for the transformer to work with.
        # Then position embeddings tell it where the hidden patch is:
        # hidden_patch_token + position embedding for patch 7
        # Patch 7 is hidden.
        # The online encoder receives:
        # learned hidden token + patch 7 position embedding
        # This tells the transformer:
        # "patch 7 exists, but you cannot see its real pixels."
        if visible_mask is not None:
            # expand the hidden patch token to the batch size
            hidden_tokens = self.hidden_patch_token.expand(size=[batch_size, self.num_patches, self.latent_dim]) # [1, 1, 96] -> [batch_size, 16, 96]
            mask = visible_mask.unsqueeze(-1)                           # [batch, 16]  -> [batch, 16, 1]
            # `torch.where` chooses one value for every patch token.
            # If `mask` is True, keep the real patch token from `patch_tokens`.
            # If `mask` is False, replace that patch with `hidden_tokens`.
            # The final `1` in `mask` is copied across the whole latent vector,
            # so one True/False value controls the full vector for that patch.
            patch_tokens = torch.where(condition=mask, input=patch_tokens, other=hidden_tokens) # [batch_size, 16, 96]

        # PyTorch broadcasts the 1 in the batch dimension, so the same position embeddings are added to every image in the batch.
        patch_tokens = patch_tokens + self.pos_embed                    # [batch_size, 16, latent_dim] + [1, 16, latent_dim]

        # Each patch token gets extra information about where it came from in the image.
        # Patch 0 gets position vector 0.
        # Patch 1 gets position vector 1.
        # ...
        # Patch 15 gets position vector 15.
        #
        # So the transformer can tell the difference between, for example, “top-left patch” and “bottom-right patch.”

        # if its not none, this will be for the target encoder path
        if visible_mask is None:
            patch_tokens = self.transformer(patch_tokens)  # [batch_size, 16, latent_dim] -> [batch_size, 16, latent_dim]
        else:
            # `visible_mask` uses True for patches the online encoder may read.
            # PyTorch's `src_key_padding_mask` uses True for tokens it should ignore.
            # That is why we use `~visible_mask`: it flips visible patches to False
            # and hidden target patches to True.
            # This means hidden target patches do not act as source tokens for attention.
            patch_tokens = self.transformer(patch_tokens, src_key_padding_mask=~visible_mask)  # [batch_size, 16, latent_dim] -> [batch_size, 16, latent_dim]

        # In this encoder:
        # visible_mask is used to decide which real patch tokens the online encoder can read.
        # target_patch_mask is used later to select the target patch latents we want the predictor to match.

        patch_tokens = self.norm(patch_tokens)  # [batch_size, 16, latent_dim] -> [batch_size, 16, latent_dim]

        return patch_tokens  # [batch_size, 16, latent_dim]
        #raise NotImplementedError("Run the tiny ViT image encoder forward pass.")


class TinyJEPAPredictor(nn.Module):
    """Tiny transformer that predicts target patch latents from context patch latents."""

    def __init__(self, config: ImageJEPAConfig) -> None:
        super().__init__()
        # Where this fits:
        # In I-JEPA, the predictor is a transformer-like module. It receives
        # context patch representations and learned mask tokens for the target
        # patch positions, then predicts the target patch representations.
        #
        # Goal:
        # Build a small transformer predictor.
        #
        # Method:
        # Build these parts:
        # - `self.mask_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))`
        # - `self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.latent_dim))`
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
        # - `context_patch_latents`: `[batch, 16, latent_dim]`
        # - `target_mask`: `[batch, 16]`
        #
        # Output:
        # - `predicted_patch_latents`: `[batch, 16, latent_dim]`

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))                        # [1, 1, 96]
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.latent_dim))        # [1, 16, 96]
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.latent_dim,
                                                    nhead=config.predictor_num_heads,
                                                    dim_feedforward=config.predictor_mlp_dim,
                                                    dropout=0.0,
                                                    activation="gelu",
                                                    batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers=config.predictor_depth)

        self.norm = nn.LayerNorm(config.latent_dim)

        self.proj = nn.Linear(in_features = config.latent_dim, out_features=config.latent_dim)

        #raise NotImplementedError("Build the tiny JEPA predictor.")

    def forward(self, context_patch_latents: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """Predict patch latents from context patch latents."""
        # Goal:
        # Predict one latent vector for every patch position. The loss will
        # later keep only the target patch positions.
        #
        # Steps:
        # 1. Expand `self.mask_token` to `[batch, 16, latent_dim]`.
        # 2. Use `torch.where(...)` so target positions get the mask token and
        #    non-target positions keep their context latents:
        #    `tokens = torch.where(target_mask.unsqueeze(-1), mask_tokens, context_patch_latents)`.
        # 3. Add `self.pos_embed` so the predictor knows each patch position.
        # 4. Run the predictor transformer.
        # 5. Apply `self.norm` and `self.proj`.
        #
        # Check:
        # - output shape should be `[batch, 16, latent_dim]`.

        # target_mask is [batch_size, num_patches]
        # target_mask is the opposite of visible mask and shows which positions the predictor should predict.

        batch_size, num_patches, latent_dim = context_patch_latents.shape                       # [batch_size, 16, latent_dim]

        mask_token = self.mask_token.expand(size=[batch_size, num_patches, latent_dim])         # [1, 1, latent_dim] -> [batch_size, 16, latent_dim]

        # `target_mask` starts with shape `[batch_size, num_patches]`  i.e [batch_size, 16].
        # In this exercise that is usually `[batch_size, 16]`.
        # `target_mask.unsqueeze(-1)` changes it to `[batch_size, 16, 1]`.
        # The final `1` lets PyTorch copy each True/False value across the full
        # latent vector for that patch.
        # If `target_mask` is True, this uses the learned `mask_token`.
        # If `target_mask` is False, this keeps the real context patch latent.
        # mask_token this means that this is a target position the predictor should fill in
        tokens = torch.where(condition=target_mask.unsqueeze(-1), input=mask_token, other=context_patch_latents)  # [batch_size, 16, latent_dim]

        tokens = tokens + self.pos_embed                # [batch_size, 16, latent_dim] + [1, 16, latent_dim] -> [batch_size, 16, latent_dim]

        tokens = self.transformer(tokens)               # [batch_size, 16, latent_dim] -> [batch_size, 16, latent_dim]
        tokens = self.norm(tokens)                      # [batch_size, 16, latent_dim] -> [batch_size, 16, latent_dim]
        predicted_patch_latents = self.proj(tokens)     # [batch_size, 16, latent_dim] -> [batch_size, 16, latent_dim]

        return predicted_patch_latents  # [batch_size, 16, latent_dim]
        #raise NotImplementedError("Run the tiny JEPA predictor forward pass.")


def build_jepa_modules(config: ImageJEPAConfig) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Build online encoder, EMA target encoder, and predictor."""
    # Goal:
    # Return three modules:
    # - online encoder: trained by gradients
    # - target encoder: updated by EMA, not by gradients
    # - predictor: trained by gradients
    #
    # Steps:
    # 1. Create `online_encoder = TinyViTImageEncoder(config)`.
    # 2. Create `target_encoder = TinyViTImageEncoder(config)`.
    # 3. Copy online encoder weights into target encoder with:
    #    `target_encoder.load_state_dict(online_encoder.state_dict())`.
    # 4. Set `requires_grad=False` for every target encoder parameter.
    # 5. Create `predictor = TinyJEPAPredictor(config)`.
    # 6. Move all modules to `device`.
    # 7. Return them.

    online_encoder = TinyViTImageEncoder(config)
    target_encoder = TinyViTImageEncoder(config)

    target_encoder.load_state_dict(state_dict=online_encoder.state_dict())

    for param in target_encoder.parameters():
        param.requires_grad = False

    predictor = TinyJEPAPredictor(config)

    online_encoder = online_encoder.to(device)
    target_encoder = target_encoder.to(device)
    predictor = predictor.to(device)


    # exercise name      paper name
    # -----------------------------------------
    # online encoder     context encoder, or x-encoder
    # target encoder     target encoder, or y-encoder
    # predictor          predictor

    # online_encoder / context encoder
    # - sees only the visible patches
    # - does not see the target patches
    # - gets updated by the optimizer

    # target_encoder / target encoder
    # - sees the full clean image
    # - makes the "answer" vectors for the target patches
    # - does not get updated by the optimizer
    # - is updated slowly from the online encoder with EMA

    # predictor / predictor
    # - takes the online encoder's visible-patch information
    # - gets mask tokens at the target positions
    # - tries to predict the target encoder's "answer" vectors

    # Small example:
    # Image has 16 patches.
    #
    # target_mask says:
    # patches 3, 6, 10, 12 are hidden target patches
    #
    # visible_mask says:
    # all other patches are visible
    # Then:
    # context_patch_latents = online_encoder(images, visible_mask)
    # Use the visible patches to build context.
    # Do not let the online encoder use patches 3, 6, 10, 12.
    #
    # target_patch_latents = target_encoder(images)
    # Look at the clean image and produce the correct target vectors.
    # These are the answers the predictor should match.
    #
    # predicted_patch_latents = predictor(context_patch_latents, target_mask)
    # Use the visible context to guess the vectors for patches 3, 6, 10, 12.
    #
    # The key idea:
    # online_encoder = the student
    # target_encoder = the answer maker
    # predictor = the part that makes the guess
    #
    # The loss compares:
    # predictor's guessed target vectors vs target_encoder's target vectors

    return online_encoder, target_encoder, predictor

    #raise NotImplementedError("Build the JEPA modules.")


@torch.no_grad()
def update_ema_target_encoder(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    ema_decay: float,
) -> None:
    """Move the target encoder a little closer to the online encoder."""
    # Where this fits:
    # Real JEPA-style systems often use a target encoder that is updated by EMA.
    #
    # Formula:
    # target = ema_decay * target + (1 - ema_decay) * online
    #
    # Plain meaning:
    # If `ema_decay = 0.99`, the target keeps 99% of its old value and takes
    # 1% from the online encoder after each update.
    #
    # Steps:
    # 1. Loop over matching parameters from both encoders.
    # 2. Update target parameters in place with the formula above.
    # 3. Do this inside `torch.no_grad()`.
    for online_param, target_param in zip(online_encoder.parameters(), target_encoder.parameters()):
        # target_param.data.mul_(ema_decay)
        # target_param.data.add_(online_param.data, alpha=1.0 - ema_decay)

        # The above is same as: 
        target_param.data = (ema_decay * target_param.data + (1.0 - ema_decay) * online_param.data)

    #raise NotImplementedError("Update the target encoder with EMA.")


def select_target_patches(patch_latents: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Select only the patch latents marked by the target mask."""
    # Goal:
    # Convert:
    # - `patch_latents`: `[batch, 16, latent_dim]`
    # - `target_mask`: `[batch, 16]`
    #
    # Into:
    # - `target_patch_latents`: `[batch, num_target_patches, latent_dim]`
    #
    # Method:
    # Because each row has the same number of target patches, you can use
    # boolean indexing and then reshape:
    # `selected = patch_latents[target_mask]`
    # `selected = selected.view(batch_size, num_target_patches, latent_dim)`
    #
    # Checks:
    # - the output batch size should match the input batch size.
    # - the middle dimension should equal the number of true values per row.

    batch_size, num_patches, latent_dim = patch_latents.shape  # [batch_size, 16, 96]
    num_target_patches = target_mask.sum(dim=1)[0].item()  # one row's True count, usually 4

    # `target_mask` has shape `[batch_size, 16]`.
    # Boolean indexing keeps only the patch vectors where the mask is True.
    # If `batch_size=2` and each image has 4 target patches, this gives
    # `[8, latent_dim]`: all selected patch vectors from the whole batch.
    selected = patch_latents[target_mask]  # [batch_size * num_target_patches, latent_dim]

    # Put the flat selected vectors back into one group per image.
    # This works because every image has the same number of True values.
    selected = selected.view(batch_size, num_target_patches, latent_dim)  # [batch_size, num_target_patches, latent_dim] = [batch_size, 4, 96]

    return selected  # [batch_size, num_target_patches, latent_dim] = [batch_size, 4, 96]
    #raise NotImplementedError("Select the target patch latents.")


def train_one_epoch(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: ImageJEPAConfig,
) -> dict[str, float]:
    """Train image JEPA for one epoch."""
    # Goal:
    # Train on target-patch latent prediction and return metrics.
    #
    # Method:
    # For each batch:
    # 1. Move `batch["image"]` to `device`.
    # 2. Create `target_mask` with `make_target_patch_mask(...)`.
    # 3. Create `visible_mask = ~target_mask`.
    # 4. Encode images with `online_encoder(images, visible_mask)`.
    # 5. Encode clean `images` with `target_encoder` inside `torch.no_grad()`.
    # 6. Predict patch latents with `predictor(context_patch_latents, target_mask)`.
    # 7. Select target patches from predicted and target latents.
    # 8. Compute `loss = loss_fn(predicted_target_latents, target_latents.detach())`.
    # 9. Call `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.
    # 10. Call `update_ema_target_encoder(...)`.
    # 11. Track latent MSE and cosine similarity.

    online_encoder.train()
    target_encoder.eval()
    predictor.train()

    total_loss = 0.0
    total_cosine_similarity = 0.0
    total_images = 0

    for data in loader:
        images = data["image"].to(device)  # [batch_size, 3, 32, 32]

        batch_size = images.shape[0]
        
        # Get a mask containing true/false value for 16 positions for every sample in the batch, randomly putting True for positions that
        # need to be masked out. Four random positions will be masked out for each sample in the batch.
        # Positions here refer to which patch positions.
        target_mask = make_target_patch_mask(batch_size, config.num_patches, config.num_target_patches, device) # shape [batch_size, num_patches] = [batch_size, 16]

        # context block = visible patches used by the context encoder
        # target blocks = hidden patches whose representations must be predicted
        # In the I-JEPA paper, the context encoder reads a context block and
        # the predictor tries to predict target blocks from that context.
        # In this small exercise:
        # - `target_mask == True` marks the target patches to predict.
        # - `visible_mask == True` marks the context patches the encoder may read.
        # So `visible_mask = ~target_mask` means every non-target patch becomes
        # part of the context, and every target patch is hidden from the encoder.
        visible_mask = ~target_mask

        context_patch_latents = online_encoder(images, visible_mask)                # [batch_size, 16, latent_dim]

        with (torch.no_grad()):
            target_patch_latents = target_encoder(images)

        predicted_patch_latents = predictor(context_patch_latents, target_mask)     # Target mask is passed here so the predictor knows which positions to produce guesses for target patches.

        # predicted_target_latents = the predictor's guesses for hidden target patches
        # target_latents = the target encoder's answer vectors for those same patches
        # We select only the 4 target patches/blocks to compute the loss on because this is what the predictor needed to guess.
        target_latents = select_target_patches(target_patch_latents, target_mask)  # [batch_size, num_target_patches, latent_dim]
        predicted_target_latents = select_target_patches(predicted_patch_latents, target_mask)  # [batch_size, num_target_patches, latent_dim]
        
        loss = loss_fn(predicted_target_latents, target_latents.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_ema_target_encoder(online_encoder=online_encoder, target_encoder=target_encoder, ema_decay=config.ema_decay)

        cosine_similarity = F.cosine_similarity(predicted_target_latents, target_latents.detach(), dim=-1).mean()
        total_loss += loss.item() * batch_size
        total_cosine_similarity += cosine_similarity.item() * batch_size
        total_images += batch_size

    #
    # Cosine similarity:
    # `F.cosine_similarity(predicted_target_latents, target_latents.detach(), dim=-1).mean()`
    return {
        "latent_mse": total_loss / total_images,
        "cosine_similarity": total_cosine_similarity / total_images,
    }
    #raise NotImplementedError("Train image JEPA for one epoch.")


@torch.no_grad()
def evaluate(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    config: ImageJEPAConfig,
) -> dict[str, float]:
    """Evaluate latent MSE and cosine similarity."""
    # Goal:
    # Return:
    # - `latent_mse`
    # - `cosine_similarity`
    #
    # Steps:
    # 1. Put all modules in eval mode.
    # 2. Loop over the loader without gradients.
    # 3. Create `target_mask` and `visible_mask = ~target_mask`.
    # 4. Encode images with `online_encoder(images, visible_mask)`.
    # 5. Encode clean images with `target_encoder(images)`.
    # 6. Predict patch latents with `predictor(context_patch_latents, target_mask)`.
    # 7. Select only target patches.
    # 8. Track average MSE and average cosine similarity.
    online_encoder.eval()
    target_encoder.eval()
    predictor.eval()

    total_loss = 0.0
    total_cosine_similarity = 0.0
    total_images = 0

    for data in loader:
        images = data["image"].to(device)  # [batch_size, 3, 32, 32]
        batch_size = images.shape[0]

        target_mask = make_target_patch_mask(batch_size, config.num_patches, config.num_target_patches, device)  # [batch_size, 16]
        visible_mask = ~target_mask  # [batch_size, 16]

        context_patch_latents = online_encoder(images, visible_mask)  # [batch_size, 16, latent_dim]
        target_patch_latents = target_encoder(images)  # [batch_size, 16, latent_dim]
        predicted_patch_latents = predictor(context_patch_latents, target_mask)  # [batch_size, 16, latent_dim]

        predicted_target_latents = select_target_patches(predicted_patch_latents, target_mask)  # [batch_size, num_target_patches, latent_dim]
        target_latents = select_target_patches(target_patch_latents, target_mask)  # [batch_size, num_target_patches, latent_dim]

        loss = loss_fn(predicted_target_latents, target_latents)  # scalar
        cosine_similarity = F.cosine_similarity(predicted_target_latents, target_latents, dim=-1).mean()  # scalar

        total_loss += loss.item() * batch_size
        total_cosine_similarity += cosine_similarity.item() * batch_size
        total_images += batch_size

    return {
        "latent_mse": total_loss / total_images,
        "cosine_similarity": total_cosine_similarity / total_images,
    }
    #raise NotImplementedError("Evaluate image JEPA latent metrics.")


def train_linear_probe(
    frozen_encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ImageJEPAConfig,
) -> dict[str, float]:
    """Train a linear classifier on frozen JEPA features."""
    # Where this fits:
    # JEPA pretraining is useful only if the learned features help another task.
    # A linear probe is a simple way to test that.
    #
    # Goal:
    # Freeze the encoder and train one linear classifier on top of its image features.
    #
    # Input:
    # - `frozen_encoder`: the trained online encoder
    # - batches with `image: [batch, 3, 32, 32]` and `label: [batch]`
    #
    # Method:
    # 1. Put the encoder in eval mode.
    # 2. For each image, run the encoder without gradients.
    # 3. Average patch latents across the patch dimension:
    #    `features = patch_latents.mean(dim=1)`
    # 4. Train `nn.Linear(config.latent_dim, 10)` on CIFAR10 labels.
    # 5. Use `nn.CrossEntropyLoss()` from `torch.nn`.
    # 6. Report validation accuracy.
    #
    # Checks:
    # - `features.shape` should be `[batch, latent_dim]`.
    # - classifier logits should be `[batch, 10]`.

    frozen_encoder.eval()
    for param in frozen_encoder.parameters():
        param.requires_grad = False

    linear_probe = nn.Linear(config.latent_dim, 10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=config.probe_learning_rate)

    mean_training_loss = 0.0
    mean_training_accuracy = 0.0

    for epoch in range(config.linear_probe_epochs):
        linear_probe.train()

        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        for data in train_loader:
            images = data["image"].to(device)  # [batch_size, 3, 32, 32]
            labels = data["label"].to(device)  # [batch_size]
            batch_size = images.shape[0]

            with torch.no_grad():
                patch_latents = frozen_encoder(images)  # [batch_size, 16, latent_dim]
                features = patch_latents.mean(dim=1)  # [batch_size, latent_dim]

            logits = linear_probe(features)  # [batch_size, 10]
            loss = loss_fn(logits, labels)  # scalar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)  # [batch_size]            
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        mean_training_loss = total_loss / total_samples
        mean_training_accuracy = total_correct / total_samples
        print(
            f"linear probe epoch {epoch + 1}/{config.linear_probe_epochs}: "
            f"loss={mean_training_loss:.4f}, accuracy={mean_training_accuracy:.4f}"
        )

    linear_probe.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in val_loader:
            images = data["image"].to(device)  # [batch_size, 3, 32, 32]
            labels = data["label"].to(device)  # [batch_size]
            batch_size = images.shape[0]

            patch_latents = frozen_encoder(images)  # [batch_size, 16, latent_dim]
            features = patch_latents.mean(dim=1)  # [batch_size, latent_dim]
            logits = linear_probe(features)  # [batch_size, 10]

            preds = logits.argmax(dim=-1)  # [batch_size]
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    mean_val_accuracy = total_correct / total_samples

    return {
        "train_loss": mean_training_loss,
        "train_accuracy": mean_training_accuracy,
        "val_accuracy": mean_val_accuracy,
    }
    #raise NotImplementedError("Train a linear probe on frozen JEPA features.")


def save_jepa_visuals(
    history: list[dict[str, float]],
    sample_batch: dict[str, torch.Tensor],
    online_encoder: nn.Module,
    config: ImageJEPAConfig,
) -> dict[str, str]:
    """Save masked-image examples, curves, and a latent PCA scatter."""
    # Goal:
    # Save visual checks that make the exercise easy to inspect.
    #
    # Method:
    # Use Matplotlib:
    # - save original and masked image examples side by side
    # - show target patches by drawing a simple patch grid or using masked images
    # - save train and validation latent MSE over epochs
    # - save a PCA scatter of frozen encoder features colored by CIFAR10 label
    #
    # PCA note:
    # A simple CPU-friendly method is:
    # 1. collect a small number of frozen features
    # 2. center the feature matrix
    # 3. use `torch.pca_lowrank(features, q=2)`
    # 4. plot the first two PCA coordinates
    #
    # Checks:
    # - output paths should point inside `outputs/jepa_30/`.
    # - at least one visual should show which target patches were hidden.
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_paths: dict[str, str] = {}

    images = sample_batch["image"].to(device)  # [batch_size, 3, 32, 32]
    labels = sample_batch["label"].to(device)  # [batch_size]
    num_examples = min(6, images.shape[0])

    target_mask = make_target_patch_mask(
        batch_size=images.shape[0],
        num_patches=config.num_patches,
        num_target_patches=config.num_target_patches,
        device=device,
    )  # [batch_size, 16]
    masked_images = apply_patch_mask(images, target_mask, config.patch_size)  # [batch_size, 3, 32, 32]

    image_path = output_dir / "masked_image_examples.png"
    fig, axes = plt.subplots(num_examples, 2, figsize=(5, 2.2 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, 2)

    for index in range(num_examples):
        clean_image = images[index].detach().cpu().permute(1, 2, 0)  # [32, 32, 3]
        masked_image = masked_images[index].detach().cpu().permute(1, 2, 0)  # [32, 32, 3]

        axes[index, 0].imshow(clean_image)
        axes[index, 0].set_title(f"original label {int(labels[index].item())}")
        axes[index, 0].axis("off")

        axes[index, 1].imshow(masked_image)
        axes[index, 1].set_title("target patches hidden")
        axes[index, 1].axis("off")

    fig.tight_layout()
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    visual_paths["masked_images"] = str(image_path)

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

    ax.set_title("JEPA latent prediction error")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    visual_paths["metric_curves"] = str(curve_path)

    pca_path = output_dir / "encoder_feature_pca.png"
    online_encoder.eval()
    with torch.no_grad():
        patch_latents = online_encoder(images)  # [batch_size, 16, latent_dim]
        features = patch_latents.mean(dim=1).detach().cpu()  # [batch_size, latent_dim]

    centered_features = features - features.mean(dim=0, keepdim=True)  # [batch_size, latent_dim]
    q = min(2, centered_features.shape[0], centered_features.shape[1])

    fig, ax = plt.subplots(figsize=(5, 4))
    if q >= 2:
        _, _, v = torch.pca_lowrank(centered_features, q=2)
        points = centered_features @ v[:, :2]  # [batch_size, 2]
        scatter = ax.scatter(points[:, 0], points[:, 1], c=labels.detach().cpu(), cmap="tab10", s=35)
        fig.colorbar(scatter, ax=ax, label="CIFAR10 label")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    else:
        ax.text(0.5, 0.5, "Need at least 2 samples for PCA", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title("Frozen encoder features")
    fig.tight_layout()
    fig.savefig(pca_path, dpi=160)
    plt.close(fig)
    visual_paths["feature_pca"] = str(pca_path)

    return visual_paths
    #raise NotImplementedError("Save image JEPA visuals.")


def run_image_jepa_experiment(config: ImageJEPAConfig) -> dict[str, object]:
    """Run the full image JEPA guided experiment."""
    # Goal:
    # Connect the dataloaders, model, training loop, EMA update, metrics,
    # linear probe, and visuals.
    #
    # Steps:
    # 1. Load dataloaders with `load_cifar_image_dataloaders(config)`.
    # 2. Build modules with `build_jepa_modules(config)`.
    # 3. Create an optimizer for the online encoder and predictor only.
    # 4. Train for `config.epochs`.
    # 5. Evaluate on validation and test loaders.
    # 6. Train the linear probe.
    # 7. Save visuals.
    #
    # Output:
    # Return a dictionary with:
    # - `train_latent_mse`
    # - `val_latent_mse`
    # - `test_latent_mse`
    # - `val_cosine_similarity`
    # - `test_cosine_similarity`
    # - `linear_probe_val_accuracy`
    # - `visual_paths`
    dataloaders = load_cifar_image_dataloaders(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    online_encoder, target_encoder, predictor = build_jepa_modules(config)

    optimizer = torch.optim.AdamW(
        list(online_encoder.parameters()) + list(predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    print(f"device: {device}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}, test batches: {len(test_loader)}")
    print(f"image size: {config.image_size}x{config.image_size}, patch size: {config.patch_size}")
    print(f"patches per image: {config.num_patches}, target patches per image: {config.num_target_patches}")

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
        }
        history.append(epoch_record)

        print(
            f"epoch {epoch + 1}/{config.epochs}: "
            f"train_mse={train_metrics['latent_mse']:.4f}, "
            f"train_cos={train_metrics['cosine_similarity']:.4f}, "
            f"val_mse={val_metrics['latent_mse']:.4f}, "
            f"val_cos={val_metrics['cosine_similarity']:.4f}"
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
        f"cosine_similarity={test_metrics['cosine_similarity']:.4f}"
    )

    probe_metrics = train_linear_probe(
        frozen_encoder=online_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    print(
        f"linear probe: train_loss={probe_metrics['train_loss']:.4f}, "
        f"train_accuracy={probe_metrics['train_accuracy']:.4f}, "
        f"val_accuracy={probe_metrics['val_accuracy']:.4f}"
    )

    sample_batch = next(iter(val_loader))
    visual_paths = save_jepa_visuals(
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
    print(f"linear_probe_val_accuracy: {probe_metrics['val_accuracy']:.4f}")

    return {
        "train_latent_mse": final_train_metrics["train_latent_mse"],
        "val_latent_mse": final_train_metrics["val_latent_mse"],
        "test_latent_mse": test_metrics["latent_mse"],
        "val_cosine_similarity": final_train_metrics["val_cosine_similarity"],
        "test_cosine_similarity": test_metrics["cosine_similarity"],
        "linear_probe_val_accuracy": probe_metrics["val_accuracy"],
        "history": history,
        "visual_paths": visual_paths,
    }
    #raise NotImplementedError("Run the full image JEPA experiment.")


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
    """Run the image JEPA scaffold."""
    config = ImageJEPAConfig()
    set_seed(config.random_state)
    print("exercise 02 is image JEPA with patch masks and an EMA target encoder.")
    print("exercise 03 covers V-JEPA, and exercise 04 covers vision-language JEPA.")
    result, ok = report_task_progress("run_image_jepa_experiment", run_image_jepa_experiment, config)
    #if ok:
    #    print(result)


if __name__ == "__main__":
    main()
