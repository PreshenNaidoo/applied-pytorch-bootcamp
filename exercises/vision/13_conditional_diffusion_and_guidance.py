"""
============================================================
EXERCISE 13: CONDITIONAL DIFFUSION AND GUIDANCE
============================================================

WHAT YOU ARE BUILDING
You are building a tiny example of conditional diffusion guidance.

Plain-language meanings:
- conditional means the model also receives a label or prompt
- guidance means pushing samples more strongly toward the condition
- unconditional means the model does not receive the condition
- guidance scale controls how strong the push is

WHY THIS EXERCISE EXISTS
Many useful diffusion models are conditional.
For example, they may use a text prompt or a class label.

This exercise teaches:
- how to represent a condition
- how to compare conditional and unconditional predictions
- how classifier-free guidance combines both predictions
- why larger guidance scale changes the denoising direction

HOW THIS FITS INTO THE BOOTCAMP
The previous diffusion schedule exercise focused on adding noise.
This file focuses on controlling what the model tries to generate.

Beginner intuition:
- unconditional prediction says: "make this less noisy"
- conditional prediction says: "make this less noisy and more like the label"
- guidance mixes those two directions

Small concrete example:
- label `0` might mean "digit zero"
- label `1` might mean "digit one"
- the unconditional path only tries to denoise
- the conditional path tries to denoise in a way that matches the label
- guidance makes the conditional direction stronger

DATASET
Use `sklearn.datasets.load_digits` as the recommended small dataset.

Why this dataset:
- it has real digit images
- it has class labels from `0` to `9`
- labels make it a good fit for conditional diffusion
- images are tiny, so the exercise stays fast

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load `load_digits`
- flatten each `8 x 8` image into a vector
- scale pixel values to a simple range
- convert images to `torch.float32`
- convert labels to `torch.long`
- use a tiny batch first so you can inspect shapes

INPUTS AND OUTPUTS
Noisy batch:
- `noisy`
  - shape: `[batch_size, data_dim]`
  - dtype: `torch.float32`

Labels:
- `labels`
  - shape: `[batch_size]`
  - dtype: `torch.long`
  - each value is a class id

Unconditional prediction:
- shape: `[batch_size, data_dim]`
- produced without labels

Conditional prediction:
- shape: `[batch_size, data_dim]`
- produced with labels

Guided prediction:
- shape: `[batch_size, data_dim]`
- combines both predictions

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a tiny MLP denoiser.

Default path:
- noisy vector input
- optional class embedding
- one or two hidden linear layers
- output vector with the same shape as the noise target

LOSS FUNCTION
For a full training version, use `nn.MSELoss`.
Compare predicted noise with the true noise.

This scaffold only checks the guidance math first.

METRICS TO TRACK
Track at least:
- whether unconditional prediction runs
- whether conditional prediction runs
- whether guided prediction has the right shape
- whether changing guidance scale changes the output

COMMON MISTAKES
- mixing up labels and timesteps
- making the model output class logits instead of predicted noise
- using guidance before both conditional and unconditional predictions exist
- expecting guidance scale `0` to behave like strong conditioning
- returning a guided tensor with the wrong shape

STEP-BY-STEP TODO
1. Build a tiny denoiser that can receive an optional class label.
2. Produce an unconditional prediction.
3. Produce a conditional prediction.
4. Combine both predictions with guidance.
5. Show that guidance scale changes the final prediction.

DEFINITION OF DONE
You are done when:
- unconditional and conditional paths both run
- guided prediction has the same shape as the noise prediction
- changing the guidance scale changes the output
- the summary explains the result in simple words
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


GuidanceSummary = dict[str, float | bool | str]

# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.


# Store settings for conditional diffusion guidance.
# Expected return:
# - a config object with batch shape, class count, and guidance scale
@dataclass
class GuidanceConfig:
    """Settings for the toy guidance exercise."""

    batch_size: int = 8
    data_dim: int = 16
    hidden_dim: int = 32
    num_classes: int = 4
    guidance_scale: float = 3.0
    random_state: int = 42


# Define a tiny denoiser that can run with or without labels.
# Expected behavior:
# - if labels are provided, use them as class condition information
# - if labels are `None`, run the unconditional path
class TinyConditionalDenoiser(nn.Module):
    """Small denoiser with an optional class condition."""

    # Build the model layers.
    # Expected modules:
    # - input projection for the noisy vector
    # - optional class embedding
    # - hidden layers
    # - output layer with size `data_dim`
    def __init__(self, data_dim: int, hidden_dim: int, num_classes: int) -> None:
        # Goal:
        # - Build a tiny denoiser that can run with or without labels.
        # - The model should always return a predicted noise vector.
        #
        # Input:
        # - `data_dim`: number of values in each noisy vector.
        # - `hidden_dim`: hidden feature size used inside the MLP.
        # - `num_classes`: number of class labels, such as digit labels.
        #
        # Output:
        # - Return `None`, but create these modules:
        #   - `self.noisy_projection`
        #   - `self.label_embedding`
        #   - `self.net`
        #
        # Method:
        # - Use these exact layers:
        #   `self.noisy_projection = nn.Linear(data_dim, hidden_dim)`
        #   `self.label_embedding = nn.Embedding(num_classes, hidden_dim)`
        #   `self.net = nn.Sequential(`
        #   `    nn.SiLU(),`
        #   `    nn.Linear(hidden_dim, hidden_dim),`
        #   `    nn.SiLU(),`
        #   `    nn.Linear(hidden_dim, data_dim),`
        #   `)`
        # - `nn.Embedding(num_classes, hidden_dim)` turns each label id into one learned vector.
        # - Small example:
        #   labels with shape `[8]` become label features with shape `[8, hidden_dim]`.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Save `data_dim`, `hidden_dim`, and `num_classes` if useful.
        # 3. Create the noisy-vector projection.
        # 4. Create the label embedding.
        # 5. Create the MLP that maps hidden features back to `data_dim`.
        #
        # Checks:
        # - `self.noisy_projection(torch.randn(8, data_dim)).shape == (8, hidden_dim)`
        # - `self.label_embedding(torch.zeros(8, dtype=torch.long)).shape == (8, hidden_dim)`
        # - The final model output should have shape `[batch_size, data_dim]`.
        super().__init__()
        raise NotImplementedError("Build a tiny conditional denoiser.")

    # Run the denoiser.
    # Input:
    # - `noisy`: `[batch_size, data_dim]`
    # - `labels`: `[batch_size]` or `None`
    # Return:
    # - predicted noise with shape `[batch_size, data_dim]`
    def forward(
        self,
        noisy: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return a noise prediction."""
        # Goal:
        # - Return a noise prediction with or without a class label.
        # - If labels are `None`, use only the noisy vector.
        # - If labels are provided, add label information before the MLP.
        #
        # Input:
        # - `noisy`: `torch.float32` tensor with shape `[batch_size, data_dim]`.
        # - `labels`: either:
        #   - `None` for the unconditional path
        #   - `torch.long` tensor with shape `[batch_size]` for the conditional path
        #
        # Output:
        # - Return `predicted_noise` with shape `[batch_size, data_dim]`.
        # - Do not return class logits. This is still a denoising model.
        #
        # Method:
        # - Project noisy data:
        #   `features = self.noisy_projection(noisy)`
        # - If labels are provided, add label features:
        #   `label_features = self.label_embedding(labels)`
        #   `features = features + label_features`
        # - Then predict noise:
        #   `predicted_noise = self.net(features)`
        # - For `labels=None`, skip the embedding and use only noisy-vector features.
        #
        # Steps:
        # 1. Check that `noisy.ndim == 2`.
        # 2. Project `noisy` to hidden features.
        # 3. If `labels is not None`, check `labels.shape == (noisy.shape[0],)`.
        # 4. Add the label embedding only on the conditional path.
        # 5. Pass the features through `self.net`.
        # 6. Return the predicted noise.
        #
        # Checks:
        # - `model(noisy, None).shape == noisy.shape`
        # - `model(noisy, labels).shape == noisy.shape`
        # - Do not apply softmax. Noise values are real numbers, not class probabilities.
        raise NotImplementedError("Return a conditional or unconditional noise prediction.")


# Apply classifier-free guidance.
# Input:
# - unconditional prediction: `[batch_size, data_dim]`
# - conditional prediction: `[batch_size, data_dim]`
# - guidance scale: Python float
# Return:
# - guided prediction with shape `[batch_size, data_dim]`
# Formula:
# - unconditional + scale * (conditional - unconditional)
# Plain-language idea:
# - start from the general denoising direction
# - add extra push toward the conditional denoising direction
def apply_classifier_free_guidance(
    unconditional_prediction: torch.Tensor,
    conditional_prediction: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Combine unconditional and conditional predictions."""
    # Goal:
    # - Combine the unconditional and conditional noise predictions.
    # - Return a guided prediction with the same shape.
    #
    # Input:
    # - `unconditional_prediction`: tensor with shape `[batch_size, data_dim]`.
    # - `conditional_prediction`: tensor with shape `[batch_size, data_dim]`.
    # - `guidance_scale`: Python float.
    # - Small example:
    #   if unconditional is `1.0`, conditional is `3.0`, and scale is `2.0`,
    #   guided value is `1.0 + 2.0 * (3.0 - 1.0) = 5.0`.
    #
    # Output:
    # - Return `guided_prediction`.
    # - Shape: `[batch_size, data_dim]`.
    #
    # Method:
    # - Use the classifier-free guidance formula:
    #   `guided_prediction = unconditional_prediction + guidance_scale * (conditional_prediction - unconditional_prediction)`
    # - Meaning:
    #   - `conditional_prediction - unconditional_prediction` is the label-specific direction.
    #   - `guidance_scale` controls how much of that direction to add.
    #   - scale `0.0` returns the unconditional prediction.
    #   - scale `1.0` returns the conditional prediction.
    #
    # Steps:
    # 1. Check that both prediction tensors have the same shape.
    # 2. Compute the label-specific difference.
    # 3. Multiply that difference by `guidance_scale`.
    # 4. Add the scaled difference to the unconditional prediction.
    # 5. Return the guided prediction.
    #
    # Checks:
    # - `guided_prediction.shape == unconditional_prediction.shape`
    # - With scale `0.0`, output should equal `unconditional_prediction`.
    # - With scale `1.0`, output should equal `conditional_prediction`.
    raise NotImplementedError("Apply the classifier-free guidance formula.")


# Run one guidance demo.
# Return:
# - dictionary with shape checks and guidance-change checks
# The output should show:
# - guided predictions keep the same shape
# - different guidance scales produce different values
def run_guidance_demo(config: GuidanceConfig) -> GuidanceSummary:
    """Run one small guidance demo and return checks."""
    model = TinyConditionalDenoiser(config.data_dim, config.hidden_dim, config.num_classes)
    noisy = torch.randn(config.batch_size, config.data_dim)
    labels = torch.randint(0, config.num_classes, (config.batch_size,))

    unconditional = model(noisy, None)
    conditional = model(noisy, labels)
    guided = apply_classifier_free_guidance(unconditional, conditional, config.guidance_scale)
    low_guided = apply_classifier_free_guidance(unconditional, conditional, 0.5)

    changed = not torch.allclose(guided, low_guided)
    shape_ok = guided.shape == noisy.shape

    return {
        "shape_ok": bool(shape_ok),
        "guidance_changes_output": bool(changed),
        "mean_absolute_guidance_shift": float((guided - unconditional).abs().mean()),
        "note": "Guidance moves the prediction toward the conditional path.",
    }


# Run this file as a script.
# Expected behavior:
# - print TODO guidance until the student fills in the scaffold
def main() -> None:
    """Run the conditional diffusion scaffold."""
    config = GuidanceConfig()
    set_seed(config.random_state)

    try:
        summary = run_guidance_demo(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return

    passed = bool(summary["shape_ok"]) and bool(summary["guidance_changes_output"])
    print(summary)
    print_pass_or_try_again(passed, "guidance_checks", int(passed), "1")


if __name__ == "__main__":
    main()
