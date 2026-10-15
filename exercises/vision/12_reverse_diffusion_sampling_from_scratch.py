"""
============================================================
EXERCISE 12: REVERSE DIFFUSION SAMPLING FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building the reverse sampling step for a tiny DDPM-style diffusion model.

Plain-language meanings:
- DDPM is a common diffusion sampling method
- reverse sampling means walking from noisy data back toward clean data
- `x_t` is the sample at noise step `t`
- `x_0` is the clean sample we are trying to recover
- predicted noise is the model's guess for the noise inside `x_t`

WHY THIS EXERCISE EXISTS
the related exercise teaches the forward noise schedule.
This part teaches how that schedule is used when sampling backward.

Core formula:

predicted_x0 = (x_t - sqrt(1 - alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)

What each part means:
- `x_t`: noisy sample at step `t`, shape `[batch_size, data_dim]`
- `predicted_noise`: model output, same shape as `x_t`
- `alpha_bar_t`: how much original signal is left at step `t`
- `predicted_x0`: the model's estimate of the clean sample

STEP-BY-STEP TODO
1. Build a beta schedule.
2. Compute alpha and alpha_bar values.
3. Estimate clean `x_0` from noisy `x_t`.
4. Compute one reverse mean for `x_{t-1}`.
5. Add optional noise when `t > 0`.
6. Run a tiny full sampling loop.
7. Explain what the reverse step is doing.
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
class ReverseDiffusionSamplingConfig:
    """Settings for the tiny reverse diffusion sampler."""

    data_dim: int = 2
    batch_size: int = 8
    num_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    random_state: int = 42


class ZeroNoisePredictor(nn.Module):
    """Tiny placeholder denoiser used to test sampler shapes."""

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return a zero noise prediction with the same shape as `x_t`."""
        # Goal:
        # Return a tensor that lets you test sampling code before training a denoiser.
        #
        # Input:
        # - `x_t`: `[batch_size, data_dim]`
        # - `timesteps`: `[batch_size]`
        #
        # Output:
        # - predicted noise: `[batch_size, data_dim]`
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
        raise NotImplementedError("Return zero predicted noise with the same shape as x_t.")


def build_beta_schedule(config: ReverseDiffusionSamplingConfig) -> torch.Tensor:
    """Return a linear beta schedule."""
    # Goal:
    # Build beta values from `beta_start` to `beta_end`.
    #
    # Output:
    # - tensor with shape `[num_steps]`
    #
    # Method:
    # Use `torch.linspace(config.beta_start, config.beta_end, config.num_steps)`.
    #
    # Checks:
    # - first value should be close to `config.beta_start`
    # - last value should be close to `config.beta_end`
    raise NotImplementedError("Build a linear beta schedule.")


def compute_schedule_terms(betas: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return alpha and alpha_bar values."""
    # Goal:
    # Convert beta values into the terms used by the reverse sampler.
    #
    # Input:
    # - `betas`: `[num_steps]`
    #
    # Output keys:
    # - `betas`
    # - `alphas`
    # - `alpha_bars`
    #
    # Formula:
    # - `alpha_t = 1 - beta_t`
    # - `alpha_bar_t = alpha_0 * alpha_1 * ... * alpha_t`
    #
    # Checks:
    # - `alpha_bars` should go down as `t` gets larger.
    raise NotImplementedError("Compute alpha and alpha_bar values.")


def gather_by_timestep(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Gather schedule values for each sample and reshape for broadcasting."""
    # Goal:
    # Pick one schedule value per batch item.
    #
    # Input:
    # - `values`: `[num_steps]`
    # - `timesteps`: `[batch_size]`
    # - `target_shape`: shape of a sample tensor, such as `[batch_size, data_dim]`
    #
    # Output:
    # - gathered values with shape `[batch_size, 1]` for vector data
    #
    # Steps:
    # 1. Use `values[timesteps]`.
    # 2. Reshape so it can multiply a tensor with `target_shape`.
    #
    # Check:
    # - for vector data, output shape should be `[batch_size, 1]`.
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
    raise NotImplementedError("Gather schedule values for each timestep.")


def estimate_x0_from_noise(
    x_t: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Estimate clean data from noisy data and predicted noise."""
    # Goal:
    # Compute the model's estimate of the clean sample.
    #
    # Formula:
    # `predicted_x0 = (x_t - sqrt(1 - alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)`
    #
    # Input shapes:
    # - `x_t`: `[batch_size, data_dim]`
    # - `predicted_noise`: `[batch_size, data_dim]`
    # - `timesteps`: `[batch_size]`
    #
    # Output:
    # - predicted clean sample with shape `[batch_size, data_dim]`
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
    raise NotImplementedError("Estimate x0 from x_t and predicted noise.")


def reverse_step(
    x_t: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return one sample from the previous timestep."""
    # Goal:
    # Move one step backward from `x_t` toward `x_{t-1}`.
    #
    # Simple teaching version:
    # 1. Estimate `x_0`.
    # 2. Use `predicted_x0` and `predicted_noise` to build a less noisy sample.
    # 3. Add small random noise only when `t > 0`.
    #
    # Output:
    # - tensor with the same shape as `x_t`
    #
    # Checks:
    # - if all timesteps are 0, do not add new random noise.
    raise NotImplementedError("Run one reverse diffusion step.")


@torch.no_grad()
def sample_loop(
    model: nn.Module,
    schedule: dict[str, torch.Tensor],
    config: ReverseDiffusionSamplingConfig,
) -> torch.Tensor:
    """Start from random noise and run all reverse steps."""
    # Goal:
    # Generate samples by walking backward from the last timestep to zero.
    #
    # Steps:
    # 1. Start with `x_t = torch.randn(batch_size, data_dim)`.
    # 2. Loop from `num_steps - 1` down to `0`.
    # 3. Build a timestep tensor filled with the current step.
    # 4. Ask the model for predicted noise.
    # 5. Call `reverse_step`.
    # 6. Return the final sample.
    #
    # Check:
    # - final shape should be `[batch_size, data_dim]`.
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
    raise NotImplementedError("Run the full reverse sampling loop.")


def run_reverse_sampling_demo(config: ReverseDiffusionSamplingConfig) -> dict[str, object]:
    """Run the reverse sampler shape checks."""
    # Output keys:
    # - `sample_shape`
    # - `num_steps`
    # - `reverse_note`
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
    raise NotImplementedError("Run reverse diffusion sampling demo.")


def check_metrics(summary: dict[str, object], config: ReverseDiffusionSamplingConfig) -> bool:
    """Return True when the sample shape is correct."""
    return tuple(summary.get("sample_shape", ())) == (config.batch_size, config.data_dim)


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
    """Run the reverse diffusion sampler scaffold."""
    config = ReverseDiffusionSamplingConfig()
    set_seed(config.random_state)

    betas, ok = report_task_progress("build_beta_schedule", build_beta_schedule, config)
    if not ok:
        return
    schedule, ok = report_task_progress("compute_schedule_terms", compute_schedule_terms, betas)
    if not ok:
        return
    model = ZeroNoisePredictor()
    summary, ok = report_task_progress("run_reverse_sampling_demo", run_reverse_sampling_demo, config)
    if ok and summary is not None:
        passed = check_metrics(summary, config)
        print_pass_or_try_again(passed, "sample_shape", float(int(passed)), "expected shape")
    _ = model, schedule


if __name__ == "__main__":
    main()
