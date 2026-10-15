"""
============================================================
EXERCISE 11: DIFFUSION NOISE SCHEDULES
============================================================

WHAT YOU ARE BUILDING
You are building small tools that show how a diffusion noise schedule works.

Plain-language meanings:
- a noise schedule says how much noise to add at each step
- beta is the small amount of new noise added at one step
- alpha is the amount of signal kept at one step
- alpha_bar is the amount of original signal still visible after many steps

WHY THIS EXERCISE EXISTS
Diffusion models depend on the noise schedule.
Before training a denoising model, you should understand what the schedule does.

This exercise teaches:
- how to create a linear beta schedule
- how to compute alpha and alpha_bar
- how to add noise to clean data at a chosen timestep
- how to check that later timesteps look noisier

HOW THIS FITS INTO THE BOOTCAMP
The previous diffusion exercise introduced the basic idea.
This file focuses on one smaller part:
- the forward noising process
- the numbers that control it

Beginner intuition:
- early timesteps keep most of the original data
- late timesteps hide the original data under more noise
- the model later learns to undo this damage

Small concrete example:
- clean vector: `[1.0, -1.0]`
- early timestep:
  - the noisy version may still look close to `[1.0, -1.0]`
- late timestep:
  - the noisy version may look much more random

DATASET
Use `sklearn.datasets.load_digits` as the recommended small dataset.

Why this dataset:
- it is small
- it is real image data
- each digit image can be flattened into one vector
- it is much easier than starting with large image files

Simple fallback:
- if you only want to test the schedule math first, use random vectors
- shape: `[batch_size, data_dim]`

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load `load_digits`
- take the image array
- flatten each image into one vector
- scale the values to a simple range, such as `[0, 1]` or `[-1, 1]`
- convert the data to `torch.float32`

INPUTS AND OUTPUTS
Clean batch:
- `clean`
  - shape: `[batch_size, data_dim]`
  - dtype: `torch.float32`

Timesteps:
- `timesteps`
  - shape: `[batch_size]`
  - dtype: `torch.long`
  - meaning: one noise step for each sample

Noisy batch:
- `noisy`
  - shape: `[batch_size, data_dim]`
  - dtype: `torch.float32`

True noise:
- `noise`
  - shape: `[batch_size, data_dim]`
  - this is the exact random noise that was added

Schedule tensors:
- `betas`: `[num_steps]`
- `alphas`: `[num_steps]`
- `alpha_bars`: `[num_steps]`

What those names mean:
- `beta` controls new noise at one step
- `alpha` controls signal kept at one step
- `alpha_bar` controls signal kept after many steps

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
No model is required in this exercise.
This file is about the forward noising process only.

LOSS FUNCTION
No training loss is required.

METRICS TO TRACK
Track at least:
- first beta
- last beta
- first alpha_bar
- last alpha_bar
- whether late timesteps change the clean input more than early timesteps

COMMON MISTAKES
- making betas too large
- forgetting that alpha_bar should usually go down over time
- using one timestep for the whole batch when the function expects one timestep per sample
- mixing CPU tensors and GPU tensors
- returning noisy data without also returning the exact sampled noise

STEP-BY-STEP TODO
1. Build a beta schedule.
2. Convert beta values into alpha values.
3. Compute alpha_bar values.
4. Add noise to a clean batch at chosen timesteps.
5. Return checks that prove later timesteps add more noise.

DEFINITION OF DONE
You are done when:
- the schedule has the requested number of steps
- all values have valid shapes
- later timesteps keep less of the clean input
- noisy samples can be created without shape errors
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


ScheduleTerms = dict[str, torch.Tensor]
ScheduleSummary = dict[str, float | int | bool]

# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.


# Store settings for the diffusion schedule exercise.
# Expected return:
# - a config object with schedule length, data shape, and seed values
@dataclass
class NoiseScheduleConfig:
    """Settings for the toy diffusion schedule."""

    num_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    batch_size: int = 8
    data_dim: int = 16
    random_state: int = 42


# Build a simple linear beta schedule.
# Return:
# - type: `torch.Tensor`
# - shape: `[num_steps]`
# - first value should be near `config.beta_start`
# - last value should be near `config.beta_end`
# Beginner intuition:
# - early steps add a small amount of noise
# - later steps add more noise
# - this list controls how quickly clean data gets corrupted
def build_linear_beta_schedule(config: NoiseScheduleConfig) -> torch.Tensor:
    """Return beta values from small noise to larger noise."""
    # Goal:
    # - Create one beta value for each diffusion step.
    # - Early beta values should be small. Later beta values should be larger.
    #
    # Input:
    # - `config.num_steps`: number of schedule values to create.
    # - `config.beta_start`: first beta value, usually a tiny value like `1e-4`.
    # - `config.beta_end`: last beta value, usually a larger value like `2e-2`.
    #
    # Output:
    # - Return a `torch.float32` tensor.
    # - Shape: `[config.num_steps]`.
    # - Small example:
    #   if `num_steps=4`, `beta_start=0.1`, and `beta_end=0.4`,
    #   the result should be close to `tensor([0.1, 0.2, 0.3, 0.4])`.
    #
    # Method:
    # - Use `torch.linspace`.
    # - Exact call:
    #   `torch.linspace(config.beta_start, config.beta_end, config.num_steps, dtype=torch.float32)`
    # - `torch.linspace(start, end, steps)` returns evenly spaced numbers from `start` to `end`.
    #
    # Steps:
    # 1. Call `torch.linspace(...)` with the exact arguments above.
    # 2. Store the result in `betas`.
    # 3. Return `betas`.
    #
    # Checks:
    # - `betas.shape == (config.num_steps,)`
    # - `betas.dtype == torch.float32`
    # - `betas[0]` should be close to `config.beta_start`.
    # - `betas[-1]` should be close to `config.beta_end`.
    raise NotImplementedError("Create a 1D tensor of beta values.")


# Convert beta values into alpha values and alpha_bar values.
# Return:
# - dictionary with:
#   - `betas`
#   - `alphas`
#   - `alpha_bars`
# Shape:
# - each tensor should have shape `[num_steps]`
# Plain-language idea:
# - `alpha` is how much signal survives one step
# - `alpha_bar` is how much original signal survives up to that step
def compute_schedule_terms(betas: torch.Tensor) -> ScheduleTerms:
    """Return alpha and alpha_bar values for the schedule."""
    # Goal:
    # - Convert beta values into the two values used by the noising formula.
    # - Keep the original `betas` in the returned dictionary too.
    #
    # Input:
    # - `betas`: `torch.float32` tensor with shape `[num_steps]`.
    # - Example:
    #   `betas = torch.tensor([0.1, 0.2, 0.3])`
    #
    # Output:
    # - Return `ScheduleTerms`.
    # - Exact keys:
    #   - `"betas"`: same tensor passed in, shape `[num_steps]`
    #   - `"alphas"`: tensor with shape `[num_steps]`
    #   - `"alpha_bars"`: tensor with shape `[num_steps]`
    #
    # Method:
    # - Formula:
    #   `alphas = 1.0 - betas`
    #   `alpha_bars = torch.cumprod(alphas, dim=0)`
    # - `torch.cumprod(alphas, dim=0)` multiplies values as it moves along the tensor.
    # - Small example:
    #   if `alphas = tensor([0.9, 0.8, 0.7])`,
    #   then `alpha_bars = tensor([0.9, 0.72, 0.504])`.
    #
    # Steps:
    # 1. Check that `betas.ndim == 1`.
    # 2. Compute `alphas = 1.0 - betas`.
    # 3. Compute `alpha_bars = torch.cumprod(alphas, dim=0)`.
    # 4. Return `{"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}`.
    #
    # Checks:
    # - `set(terms) == {"betas", "alphas", "alpha_bars"}`
    # - `terms["alphas"].shape == betas.shape`
    # - `terms["alpha_bars"].shape == betas.shape`
    # - `terms["alpha_bars"][0] > terms["alpha_bars"][-1]`
    raise NotImplementedError("Compute alphas and cumulative alpha_bar values.")


# Add noise to clean data at chosen timesteps.
# Input:
# - `clean`: `[batch_size, data_dim]`
# - `timesteps`: `[batch_size]`
# - `schedule_terms`: dictionary from `compute_schedule_terms`
# Return:
# - `noisy`: `[batch_size, data_dim]`
# - `noise`: `[batch_size, data_dim]`
# Formula idea:
# - noisy = sqrt(alpha_bar_t) * clean + sqrt(1 - alpha_bar_t) * noise
# Beginner intuition:
# - early timestep: mostly clean data, little noise
# - late timestep: less clean data, more noise
def add_noise(
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    schedule_terms: ScheduleTerms,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return noisy data and the exact noise that was added."""
    # Goal:
    # - Add the right amount of noise to each clean row.
    # - Return both the noisy data and the exact random noise used.
    #
    # Input:
    # - `clean`: `torch.float32` tensor with shape `[batch_size, data_dim]`.
    # - `timesteps`: `torch.long` tensor with shape `[batch_size]`.
    # - `schedule_terms`: dictionary from `compute_schedule_terms`.
    # - `schedule_terms["alpha_bars"]`: tensor with shape `[num_steps]`.
    # - Example:
    #   `clean.shape == (8, 16)` and `timesteps.shape == (8,)`.
    #
    # Output:
    # - Return a tuple:
    #   `(noisy, noise)`
    # - `noisy` shape: `[batch_size, data_dim]`.
    # - `noise` shape: `[batch_size, data_dim]`.
    # - `noise` is the target a denoising model would learn to predict later.
    #
    # Method:
    # - Sample random noise with:
    #   `noise = torch.randn_like(clean)`
    # - Move schedule values to the same device as `clean`:
    #   `alpha_bars = schedule_terms["alpha_bars"].to(clean.device)`
    # - Pick the alpha_bar value for each row:
    #   `alpha_bar_t = alpha_bars[timesteps].unsqueeze(1)`
    # - Formula:
    #   `noisy = torch.sqrt(alpha_bar_t) * clean + torch.sqrt(1.0 - alpha_bar_t) * noise`
    # - The `.unsqueeze(1)` changes shape from `[batch_size]` to `[batch_size, 1]`.
    # - That lets PyTorch apply each row's timestep value across all `data_dim` columns.
    #
    # Steps:
    # 1. Check that `clean.ndim == 2`.
    # 2. Check that `timesteps.shape == (clean.shape[0],)`.
    # 3. Sample `noise` with the same shape as `clean`.
    # 4. Gather one `alpha_bar_t` value for each row.
    # 5. Apply the formula above.
    # 6. Return `(noisy, noise)`.
    #
    # Checks:
    # - `noisy.shape == clean.shape`
    # - `noise.shape == clean.shape`
    # - `alpha_bar_t.shape == (clean.shape[0], 1)`
    # - A late timestep should usually change `clean` more than timestep 0.
    raise NotImplementedError("Add timestep-dependent noise to the clean batch.")


# Run the schedule checks.
# Return:
# - summary dictionary with schedule values and a late-noise check
# The check should prove:
# - late timesteps change the clean input more than early timesteps
def summarize_schedule(config: NoiseScheduleConfig) -> ScheduleSummary:
    """Return simple facts that show the schedule is wired correctly."""
    betas = build_linear_beta_schedule(config)
    terms = compute_schedule_terms(betas)

    clean = torch.randn(config.batch_size, config.data_dim)
    early_t = torch.zeros(config.batch_size, dtype=torch.long)
    late_t = torch.full((config.batch_size,), config.num_steps - 1, dtype=torch.long)

    early_noisy, _ = add_noise(clean, early_t, terms)
    late_noisy, _ = add_noise(clean, late_t, terms)

    early_change = (early_noisy - clean).abs().mean().item()
    late_change = (late_noisy - clean).abs().mean().item()

    return {
        "num_steps": int(betas.numel()),
        "first_beta": float(betas[0]),
        "last_beta": float(betas[-1]),
        "first_alpha_bar": float(terms["alpha_bars"][0]),
        "last_alpha_bar": float(terms["alpha_bars"][-1]),
        "late_change_larger": bool(late_change > early_change),
    }


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the diffusion schedule scaffold."""
    config = NoiseScheduleConfig()
    set_seed(config.random_state)

    try:
        summary = summarize_schedule(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    passed = (
        summary["num_steps"] == config.num_steps
        and summary["first_beta"] < summary["last_beta"]
        and summary["first_alpha_bar"] > summary["last_alpha_bar"]
        and bool(summary["late_change_larger"])
    )
    print(summary)
    print_pass_or_try_again(passed, "schedule_checks", int(passed), "1")


if __name__ == "__main__":
    main()
