"""
============================================================
EXERCISE 14: LATENT DIFFUSION AUTOENCODER BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny autoencoder that shows the idea behind latent diffusion.

Plain-language meanings:
- an encoder compresses data into a smaller hidden form
- a decoder rebuilds data from that hidden form
- latent means hidden representation
- latent diffusion adds and removes noise in the smaller hidden space

WHY THIS EXERCISE EXISTS
Large image diffusion models often do not denoise full images directly.
They first compress images into a smaller latent space.

This exercise teaches:
- how an autoencoder compresses input data
- how to reconstruct data from latents
- why denoising a smaller latent can be cheaper
- how to report the compression ratio

HOW THIS FITS INTO THE BOOTCAMP
The earlier diffusion files explain noising, denoising, and guidance.
This file adds the common trick used by many larger systems:
- work in a smaller hidden space instead of the full input space

Beginner intuition:
- a full image has many numbers
- a latent has fewer numbers
- if the latent still keeps the important information, it can be cheaper to model

Small concrete example:
- a flattened `8 x 8` digit image has `64` numbers
- an encoder might compress it to `16` numbers
- a decoder tries to rebuild the original `64` numbers from those `16`
- latent diffusion would later add noise to the smaller `16`-number latent

DATASET
Use `sklearn.datasets.load_digits` as the recommended small dataset.

Why this dataset:
- it contains real small images
- each image is only `8 x 8`
- flattened images are short vectors
- it is a good beginner bridge between vectors and images

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load `load_digits`
- flatten images into vectors
- scale pixel values to `[0, 1]`
- convert vectors to `torch.float32`
- wrap them in a `TensorDataset`
- build a `DataLoader`

INPUTS AND OUTPUTS
Input batch:
- `x`
  - shape: `[batch_size, input_dim]`
  - for `load_digits`, `input_dim` is usually `64`

Latent batch:
- `z`
  - shape: `[batch_size, latent_dim]`
  - this is the compressed hidden representation

Reconstruction:
- `reconstruction`
  - shape: `[batch_size, input_dim]`
  - this should match the input shape

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small autoencoder.

Encoder:
- linear layer
- activation
- linear layer to `latent_dim`

Decoder:
- linear layer
- activation
- linear layer back to `input_dim`

LOSS FUNCTION
Use `nn.MSELoss`.

Plain-language meaning:
- MSE compares the reconstructed vector with the original vector
- lower MSE means the reconstruction is closer to the input

METRICS TO TRACK
Track at least:
- reconstruction loss
- latent shape check
- reconstruction shape check
- compression ratio

COMMON MISTAKES
- making the decoder output `latent_dim` instead of `input_dim`
- forgetting to scale image values before training
- using labels even though this exercise does not need class labels
- thinking the autoencoder is already a full diffusion model
- using a latent size that is not smaller than the input size

STEP-BY-STEP TODO
1. Build a small encoder.
2. Build a small decoder.
3. Train the autoencoder on simple vector data.
4. Measure reconstruction loss.
5. Report how much smaller the latent space is.

DEFINITION OF DONE
You are done when:
- input vectors can be encoded into smaller latent vectors
- latent vectors can be decoded back to input shape
- reconstruction loss is reported
- compression ratio is greater than 1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


AutoencoderMetrics = dict[str, float]
LatentAutoencoderSummary = dict[str, float | bool]

# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.


# Store settings for the latent autoencoder.
# Expected return:
# - a config object with data, model, and training settings
@dataclass
class LatentAutoencoderConfig:
    """Settings for the tiny latent autoencoder."""

    num_samples: int = 512
    input_dim: int = 64
    latent_dim: int = 16
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    random_state: int = 42


# Build the training DataLoader.
# Recommended data:
# - flattened `load_digits` images
# Return:
# - DataLoader that yields one tensor batch at a time
def build_vector_loader(config: LatentAutoencoderConfig) -> DataLoader:
    """Return a simple vector dataset for autoencoder practice."""
    # Goal:
    # - Load small real digit images and return batches of flattened vectors.
    # - The DataLoader should yield input vectors only. Labels are not needed.
    #
    # Input:
    # - `config.num_samples`: maximum number of digit examples to keep.
    # - `config.input_dim`: expected flattened size. For `load_digits`, this should be 64.
    # - `config.batch_size`: number of vectors per batch.
    #
    # Output:
    # - Return a PyTorch `DataLoader`.
    # - Each batch comes from `TensorDataset(x)`, so it is a one-item tuple: `(x_batch,)`.
    # - `x_batch` shape: `[batch_size, config.input_dim]`.
    # - `x_batch` dtype: `torch.float32`.
    # - Small example shape:
    #   `[32, 64]` means 32 digit images, each flattened to 64 numbers.
    #
    # Method:
    # - Use scikit-learn and PyTorch:
    #   `from sklearn.datasets import load_digits`
    #   `from torch.utils.data import TensorDataset, DataLoader`
    # - Load data with:
    #   `digits = load_digits()`
    # - `digits.data` already has shape `[num_examples, 64]`.
    # - Scale pixel values from `[0, 16]` to `[0, 1]`:
    #   `x = torch.tensor(digits.data[: config.num_samples], dtype=torch.float32) / 16.0`
    # - Wrap the tensor with:
    #   `dataset = TensorDataset(x)`
    # - Build the loader with:
    #   `DataLoader(dataset, batch_size=config.batch_size, shuffle=True)`
    #
    # Steps:
    # 1. Load the digits dataset.
    # 2. Take at most `config.num_samples` rows from `digits.data`.
    # 3. Convert the data to `torch.float32`.
    # 4. Divide by `16.0` so values are in `[0, 1]`.
    # 5. Check that `x.shape[1] == config.input_dim`.
    # 6. Create `TensorDataset(x)`.
    # 7. Return the DataLoader.
    #
    # Checks:
    # - `batch = next(iter(loader))[0]`
    # - `batch.shape[1] == config.input_dim`
    # - `batch.dtype == torch.float32`
    # - `batch.min() >= 0.0` and `batch.max() <= 1.0`
    raise NotImplementedError("Create a simple DataLoader of vector data.")



# Define a tiny autoencoder.
# Expected behavior:
# - `encode` compresses input vectors into smaller latent vectors
# - `decode` rebuilds input-shaped vectors from latent vectors
class TinyAutoencoder(nn.Module):
    """Small autoencoder for latent diffusion practice."""

    # Build encoder and decoder modules.
    # Expected modules:
    # - encoder maps `input_dim -> latent_dim`
    # - decoder maps `latent_dim -> input_dim`
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        # Goal:
        # - Build an encoder that compresses input vectors.
        # - Build a decoder that rebuilds input-shaped vectors from latents.
        #
        # Input:
        # - `input_dim`: number of values in the input vector, 64 for flattened digits.
        # - `hidden_dim`: hidden size used inside the MLP.
        # - `latent_dim`: compressed size. This should be smaller than `input_dim`.
        #
        # Output:
        # - Return `None`, but create:
        #   - `self.encoder`
        #   - `self.decoder`
        #
        # Method:
        # - Use these exact modules:
        #   `self.encoder = nn.Sequential(`
        #   `    nn.Linear(input_dim, hidden_dim),`
        #   `    nn.ReLU(),`
        #   `    nn.Linear(hidden_dim, latent_dim),`
        #   `)`
        #   `self.decoder = nn.Sequential(`
        #   `    nn.Linear(latent_dim, hidden_dim),`
        #   `    nn.ReLU(),`
        #   `    nn.Linear(hidden_dim, input_dim),`
        #   `    nn.Sigmoid(),`
        #   `)`
        # - `nn.Sigmoid()` keeps reconstruction values between 0 and 1.
        # - That matches the scaled digit input values.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Create the encoder.
        # 3. Create the decoder.
        # 4. Do not put a softmax on the decoder output.
        #
        # Checks:
        # - `self.encoder(torch.randn(4, input_dim)).shape == (4, latent_dim)`
        # - `self.decoder(torch.randn(4, latent_dim)).shape == (4, input_dim)`
        # - `latent_dim < input_dim`
        super().__init__()
        raise NotImplementedError("Build the encoder and decoder modules.")

    # Encode one batch.
    # Input:
    # - `x`: `[batch_size, input_dim]`
    # Return:
    # - latent tensor with shape `[batch_size, latent_dim]`
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return a smaller latent vector."""
        # Goal:
        # - Compress one input batch into latent vectors.
        #
        # Input:
        # - `x`: `torch.float32` tensor with shape `[batch_size, input_dim]`.
        # - Small example:
        #   if `x.shape == (32, 64)` and `latent_dim == 16`,
        #   the output shape should be `(32, 16)`.
        #
        # Output:
        # - Return `z`, the latent tensor.
        # - Shape: `[batch_size, latent_dim]`.
        #
        # Method:
        # - Use the encoder module created in `__init__`:
        #   `z = self.encoder(x)`
        #
        # Steps:
        # 1. Check that `x.ndim == 2`.
        # 2. Pass `x` through `self.encoder`.
        # 3. Return the latent tensor.
        #
        # Checks:
        # - `z.shape[0] == x.shape[0]`
        # - `z.shape[1]` should equal the configured `latent_dim`.
        raise NotImplementedError("Encode the input into a latent vector.")

    # Decode one latent batch.
    # Input:
    # - `z`: `[batch_size, latent_dim]`
    # Return:
    # - reconstruction tensor with shape `[batch_size, input_dim]`
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return a reconstruction from the latent vector."""
        # Goal:
        # - Rebuild input-shaped vectors from latent vectors.
        #
        # Input:
        # - `z`: `torch.float32` tensor with shape `[batch_size, latent_dim]`.
        # - Small example:
        #   if `z.shape == (32, 16)` and `input_dim == 64`,
        #   the output shape should be `(32, 64)`.
        #
        # Output:
        # - Return `reconstruction`.
        # - Shape: `[batch_size, input_dim]`.
        # - Values should be in `[0, 1]` if the decoder ends with `nn.Sigmoid()`.
        #
        # Method:
        # - Use the decoder module created in `__init__`:
        #   `reconstruction = self.decoder(z)`
        #
        # Steps:
        # 1. Check that `z.ndim == 2`.
        # 2. Pass `z` through `self.decoder`.
        # 3. Return the reconstruction.
        #
        # Checks:
        # - `reconstruction.shape[0] == z.shape[0]`
        # - `reconstruction.shape[1]` should equal the configured `input_dim`.
        raise NotImplementedError("Decode the latent vector back to input shape.")

    # Run encode and decode together.
    # Input:
    # - `x`: `[batch_size, input_dim]`
    # Return:
    # - reconstruction with shape `[batch_size, input_dim]`
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and then decode the input."""
        return self.decode(self.encode(x))


# Train the autoencoder.
# Input:
# - model
# - DataLoader of input vectors
# - config with epochs and learning rate
# Return:
# - dictionary with `reconstruction_loss`
def train_autoencoder(
    model: TinyAutoencoder,
    loader: DataLoader,
    config: LatentAutoencoderConfig,
) -> AutoencoderMetrics:
    """Train the autoencoder and return reconstruction loss."""
    # Goal:
    # - Train the autoencoder to reconstruct its input vectors.
    # - Return the mean reconstruction loss from the final epoch.
    #
    # Input:
    # - `model`: `TinyAutoencoder`.
    # - `loader`: DataLoader from `build_vector_loader`.
    # - `config.learning_rate`: optimizer learning rate.
    # - `config.epochs`: number of passes over the data.
    #
    # Output:
    # - Return `AutoencoderMetrics`.
    # - Exact key:
    #   - `"reconstruction_loss"`: average final-epoch MSE over all examples
    #
    # Method:
    # - Use these exact training objects:
    #   `loss_fn = nn.MSELoss()`
    #   `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    # - Each batch from `TensorDataset(x)` is a one-item tuple, so use:
    #   `x = batch[0]`
    # - Forward and loss:
    #   `reconstruction = model(x)`
    #   `loss = loss_fn(reconstruction, x)`
    # - The target is the original input `x`, not a class label.
    #
    # Steps:
    # 1. Create `nn.MSELoss()`.
    # 2. Create `torch.optim.AdamW(...)`.
    # 3. Put the model in train mode.
    # 4. For each epoch, loop over the DataLoader.
    # 5. Get `x` from the batch tuple.
    # 6. Compute `reconstruction = model(x)`.
    # 7. Compute MSE loss against `x`.
    # 8. Clear gradients, backpropagate, and step the optimizer.
    # 9. Track `loss.item() * x.shape[0]` so the final average is weighted by batch size.
    # 10. Return `{"reconstruction_loss": average_loss}` from the final epoch.
    #
    # Checks:
    # - `reconstruction.shape == x.shape`
    # - `metrics["reconstruction_loss"] >= 0.0`
    # - The returned dictionary must contain `"reconstruction_loss"`.
    raise NotImplementedError("Train the autoencoder with mean squared error.")


# Run the full latent-autoencoder workflow.
# Return:
# - reconstruction loss
# - latent shape check
# - reconstruction shape check
# - compression ratio
def run_latent_autoencoder(config: LatentAutoencoderConfig) -> LatentAutoencoderSummary:
    """Run the latent autoencoder workflow."""
    loader = build_vector_loader(config)
    model = TinyAutoencoder(config.input_dim, config.hidden_dim, config.latent_dim)
    metrics = train_autoencoder(model, loader, config)

    sample_batch = next(iter(loader))[0]
    latent = model.encode(sample_batch)
    reconstruction = model.decode(latent)
    compression_ratio = config.input_dim / config.latent_dim

    return {
        "reconstruction_loss": float(metrics["reconstruction_loss"]),
        "latent_shape_ok": bool(latent.shape[-1] == config.latent_dim),
        "reconstruction_shape_ok": bool(reconstruction.shape == sample_batch.shape),
        "compression_ratio": float(compression_ratio),
    }


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the latent diffusion scaffold."""
    config = LatentAutoencoderConfig()
    set_seed(config.random_state)

    try:
        summary = run_latent_autoencoder(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    passed = (
        bool(summary["latent_shape_ok"])
        and bool(summary["reconstruction_shape_ok"])
        and float(summary["compression_ratio"]) > 1.0
    )
    print(summary)
    print_pass_or_try_again(passed, "latent_autoencoder_checks", int(passed), "1")


if __name__ == "__main__":
    main()
