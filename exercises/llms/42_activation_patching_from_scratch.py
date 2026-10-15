"""
============================================================
EXERCISE 42: ACTIVATION PATCHING FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building tiny activation capture and patching tools from scratch.

Plain-language meanings:
- an activation is a hidden tensor inside a model
- a hook is a PyTorch tool that lets you save or change a tensor during a forward pass
- activation patching means replace one run's hidden tensor with another run's hidden tensor
- clean input means the input where the model behaves correctly
- corrupted input means the input where something has been changed

WHY THIS EXERCISE EXISTS
the related exercise introduces interpretability tools.
This part makes one of those tools concrete.

Activation patching asks:
- if I copy this hidden value from the clean run into the corrupted run,
- does the model's answer move back toward the clean answer?

This is not perfect proof.
It is a careful test that can give a useful clue.

INPUTS AND OUTPUTS
Input token ids:
- shape: `[batch_size, sequence_length]`

Hidden states:
- shape: `[batch_size, sequence_length, hidden_dim]`

Logits:
- shape: `[batch_size, num_classes]`

Patch result:
- clean probability for the target class
- corrupted probability for the target class
- patched probability for the target class
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    """Set the main random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a short result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")


@dataclass
class ActivationPatchingConfig:
    """Settings for the activation patching exercise."""

    vocab_size: int = 20
    sequence_length: int = 6
    hidden_dim: int = 32
    num_classes: int = 2
    target_class: int = 0
    random_state: int = 42


class TinySequenceClassifier(nn.Module):
    """Small model with a named hidden layer we can patch."""

    def __init__(self, config: ActivationPatchingConfig) -> None:
        # Goal:
        # Build a small classifier with an easy-to-hook hidden layer.
        #
        # Layers:
        # - `nn.Embedding(vocab_size, hidden_dim)`
        # - `nn.Linear(hidden_dim, hidden_dim)` named `mixer`
        # - `nn.Linear(hidden_dim, num_classes)` named `classifier`
        #
        # Forward shape:
        # - token ids: `[batch_size, sequence_length]`
        # - embeddings: `[batch_size, sequence_length, hidden_dim]`
        # - mixed hidden states: `[batch_size, sequence_length, hidden_dim]`
        # - pooled hidden state: `[batch_size, hidden_dim]`
        # - logits: `[batch_size, num_classes]`
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
        raise NotImplementedError("Build the tiny sequence classifier.")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        # Goal:
        # Run the model on token ids.
        #
        # Steps:
        # 1. Embed token ids.
        # 2. Pass embeddings through `self.mixer`.
        # 3. Apply `torch.tanh`.
        # 4. Mean-pool over sequence positions.
        # 5. Return classifier logits.
        #
        # Check:
        # - Do not apply softmax here.
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
        raise NotImplementedError("Return class logits.")


def build_clean_and_corrupted_inputs(config: ActivationPatchingConfig) -> dict[str, torch.Tensor]:
    """Return clean and corrupted token batches."""
    # Goal:
    # Build one clean input and one corrupted input.
    #
    # Output:
    # Return a dictionary with:
    # - `clean_tokens`: `[1, sequence_length]`
    # - `corrupted_tokens`: `[1, sequence_length]`
    # - `target_class`: scalar tensor
    #
    # Small example:
    # - clean: `[1, 2, 3, 4, 5, 6]`
    # - corrupted: `[1, 2, 9, 4, 5, 6]`
    #
    # Steps:
    # 1. Make a clean sequence.
    # 2. Copy it.
    # 3. Change one middle token in the copy.
    # 4. Return both with batch dimension.
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
    raise NotImplementedError("Build clean and corrupted token batches.")


def save_activation(
    model: TinySequenceClassifier,
    module_name: str,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model and save one module output."""
    # Goal:
    # Save the output activation from a named module during a forward pass.
    #
    # Method:
    # Use `module.register_forward_hook(hook_fn)`.
    #
    # Steps:
    # 1. Find the module with `dict(model.named_modules())[module_name]`.
    # 2. Define a hook that stores the module output.
    # 3. Register the hook.
    # 4. Run the model.
    # 5. Remove the hook with `handle.remove()`.
    # 6. Return logits and the saved activation.
    #
    # Check:
    # - Always remove the hook after the forward pass.
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
    raise NotImplementedError("Save one activation with a forward hook.")


def run_with_patched_activation(
    model: TinySequenceClassifier,
    module_name: str,
    token_ids: torch.Tensor,
    replacement_activation: torch.Tensor,
) -> torch.Tensor:
    """Run the model while replacing one module output."""
    # Goal:
    # Replace a module output during the corrupted forward pass.
    #
    # Method:
    # Use a forward hook that returns `replacement_activation`.
    # In PyTorch, a forward hook may replace the output by returning a new tensor.
    #
    # Steps:
    # 1. Find the named module.
    # 2. Register a hook that returns `replacement_activation`.
    # 3. Run the model on corrupted tokens.
    # 4. Remove the hook.
    # 5. Return patched logits.
    #
    # Check:
    # - Replacement activation shape must match the original module output shape.
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
    raise NotImplementedError("Run model with a patched activation.")


def target_probability(logits: torch.Tensor, target_class: int) -> float:
    """Return the probability for one target class."""
    # Goal:
    # Convert logits to probabilities and read one class.
    #
    # Input:
    # - `logits`: `[batch_size, num_classes]`
    # - `target_class`: class id to inspect
    #
    # Method:
    # Use `F.softmax(logits, dim=-1)`.
    #
    # Output:
    # - Python float for the first item in the batch
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
    raise NotImplementedError("Return target class probability.")


def run_activation_patching(config: ActivationPatchingConfig) -> dict[str, float | str]:
    """Run the full clean, corrupted, and patched comparison."""
    # Goal:
    # Compare target-class probability across three runs.
    #
    # Output keys:
    # - `clean_probability`
    # - `corrupted_probability`
    # - `patched_probability`
    # - `patch_delta`
    # - `patch_note`
    #
    # Steps:
    # 1. Build the model.
    # 2. Build clean and corrupted inputs.
    # 3. Save clean activation from `mixer`.
    # 4. Run corrupted input normally.
    # 5. Run corrupted input with clean activation patched into `mixer`.
    # 6. Convert logits to target-class probabilities.
    # 7. Compute `patch_delta = patched_probability - corrupted_probability`.
    # 8. Write a careful note.
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
    raise NotImplementedError("Run activation patching and return probabilities.")


def check_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when the expected patch fields are present."""
    required = {"clean_probability", "corrupted_probability", "patched_probability", "patch_delta", "patch_note"}
    return required.issubset(metrics)


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
    """Run the activation-patching scaffold."""
    config = ActivationPatchingConfig()
    set_seed(config.random_state)

    model, ok = report_task_progress("TinySequenceClassifier", TinySequenceClassifier, config)
    if not ok:
        return
    batch, ok = report_task_progress("build_clean_and_corrupted_inputs", build_clean_and_corrupted_inputs, config)
    if not ok:
        return
    _, ok = report_task_progress("save_activation", save_activation, model, "mixer", batch["clean_tokens"])
    if not ok:
        return
    metrics, ok = report_task_progress("run_activation_patching", run_activation_patching, config)
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "activation_patching_fields", float(int(passed)), "all expected fields")


if __name__ == "__main__":
    main()
