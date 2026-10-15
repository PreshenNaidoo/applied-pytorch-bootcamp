"""
============================================================
EXERCISE 16: TINY U-NET FOR DIFFUSION
============================================================

WHAT YOU ARE BUILDING
You are building a very small U-Net-style denoiser for diffusion.

Plain-language meanings:
- a U-Net goes down to smaller features and then back up
- a skip connection copies useful early features into a later layer
- a denoiser predicts the noise that was added to an image
- timestep embedding tells the model how noisy the image is

WHY THIS EXERCISE EXISTS
Earlier diffusion files can use simple denoisers.
Real image diffusion models usually use U-Net-style denoisers.
This file builds a tiny version so the shape flow is clear.

INPUTS AND OUTPUTS
Input noisy image:
- shape: `[batch_size, channels, height, width]`

Timesteps:
- shape: `[batch_size]`

Predicted noise:
- shape: `[batch_size, channels, height, width]`

STEP-BY-STEP TODO
1. Build a tiny timestep embedding.
2. Build a down block.
3. Build an up block.
4. Add a skip connection.
5. Build the tiny U-Net denoiser.
6. Check that output shape matches input shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import torch
import torch.nn as nn


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42) -> None:
    """Set the main random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a short result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")


@dataclass
class TinyUNetDiffusionConfig:
    """Settings for the tiny U-Net denoiser."""

    batch_size: int = 4
    channels: int = 1
    height: int = 8
    width: int = 8
    base_channels: int = 16
    timestep_dim: int = 16
    num_steps: int = 100
    random_state: int = 42


class TimestepEmbedding(nn.Module):
    """Small learned timestep embedding."""

    def __init__(self, num_steps: int, timestep_dim: int) -> None:
        # Goal:
        # Build an embedding table for diffusion timesteps.
        #
        # Method:
        # Use `nn.Embedding(num_steps, timestep_dim)`.
        super().__init__()
        # Method:
        # - Use the formula, layer, or PyTorch function named above.
        # - Keep the code small and readable; this is a teaching exercise.
        #
        # Steps:
        # 1. Check the input shapes and dtypes from the comments above.
        # 2. Do the calculation one clear step at a time.
        # 3. Return the exact tensor, object, or dictionary that the next function expects.
        #
        # Checks:
        # - Try one tiny example before using the full demo.
        # - Check at least one output shape, key, or value before moving on.
        raise NotImplementedError("Build the timestep embedding table.")

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Return timestep vectors."""
        # Input:
        # - `timesteps`: `[batch_size]`
        #
        # Output:
        # - embeddings: `[batch_size, timestep_dim]`
        # Method:
        # - Use the formula, layer, or PyTorch function named above.
        # - Keep the code small and readable; this is a teaching exercise.
        #
        # Steps:
        # 1. Check the input shapes and dtypes from the comments above.
        # 2. Do the calculation one clear step at a time.
        # 3. Return the exact tensor, object, or dictionary that the next function expects.
        #
        # Checks:
        # - Try one tiny example before using the full demo.
        # - Check at least one output shape, key, or value before moving on.
        raise NotImplementedError("Return timestep embeddings.")


class TinyUNetDenoiser(nn.Module):
    """Very small U-Net-style noise predictor."""

    def __init__(self, config: TinyUNetDiffusionConfig) -> None:
        # Goal:
        # Build a tiny U-Net-style denoiser.
        #
        # Suggested layers:
        # - timestep embedding
        # - input convolution: channels -> base_channels
        # - down convolution with stride 2
        # - bottleneck convolution
        # - upsample back to original height and width
        # - output convolution: base_channels -> channels
        #
        # Skip connection:
        # - save the early feature map before downsampling
        # - add it back after upsampling
        super().__init__()
        # Method:
        # - Use the formula, layer, or PyTorch function named above.
        # - Keep the code small and readable; this is a teaching exercise.
        #
        # Steps:
        # 1. Check the input shapes and dtypes from the comments above.
        # 2. Do the calculation one clear step at a time.
        # 3. Return the exact tensor, object, or dictionary that the next function expects.
        #
        # Checks:
        # - Try one tiny example before using the full demo.
        # - Check at least one output shape, key, or value before moving on.
        raise NotImplementedError("Build the tiny U-Net denoiser layers.")

    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return predicted noise with the same shape as the input."""
        # Goal:
        # Predict the noise inside `noisy_images`.
        #
        # Input:
        # - `noisy_images`: `[batch_size, channels, height, width]`
        # - `timesteps`: `[batch_size]`
        #
        # Output:
        # - predicted noise: `[batch_size, channels, height, width]`
        #
        # Steps:
        # 1. Embed timesteps.
        # 2. Reshape timestep features so they can be added to image features.
        # 3. Run the down path.
        # 4. Run the bottleneck.
        # 5. Upsample.
        # 6. Add the skip connection.
        # 7. Return predicted noise.
        # Method:
        # - Use the formula, layer, or PyTorch function named above.
        # - Keep the code small and readable; this is a teaching exercise.
        #
        # Steps:
        # 1. Check the input shapes and dtypes from the comments above.
        # 2. Do the calculation one clear step at a time.
        # 3. Return the exact tensor, object, or dictionary that the next function expects.
        #
        # Checks:
        # - Try one tiny example before using the full demo.
        # - Check at least one output shape, key, or value before moving on.
        raise NotImplementedError("Return predicted noise.")


def make_toy_noisy_images(config: TinyUNetDiffusionConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return toy noisy images and timesteps."""
    # Goal:
    # Build a tiny batch for shape checks.
    #
    # Output:
    # - noisy images: `[batch_size, channels, height, width]`
    # - timesteps: `[batch_size]`
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Create toy noisy images and timesteps.")


def run_unet_shape_demo(config: TinyUNetDiffusionConfig) -> dict[str, object]:
    """Run a forward pass and return shape checks."""
    # Output keys:
    # - `input_shape`
    # - `output_shape`
    # - `shape_ok`
    # - `unet_note`
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Run tiny U-Net shape demo.")


def check_metrics(summary: dict[str, object]) -> bool:
    """Return True when the U-Net output shape matches the input shape."""
    return bool(summary.get("shape_ok", False))


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    print(f"[RUNNING] {task_name}")
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    except Exception as error:
        print(f"[ERROR] {task_name}: {type(error).__name__}: {error}")
        raise
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the tiny U-Net diffusion scaffold."""
    config = TinyUNetDiffusionConfig()
    set_seed(config.random_state)

    _, ok = report_task_progress("make_toy_noisy_images", make_toy_noisy_images, config)
    if not ok:
        return
    summary, ok = report_task_progress("run_unet_shape_demo", run_unet_shape_demo, config)
    if ok and summary is not None:
        passed = check_metrics(summary)
        print_pass_or_try_again(passed, "unet_shape", float(int(passed)), "output shape matches input shape")


if __name__ == "__main__":
    main()
