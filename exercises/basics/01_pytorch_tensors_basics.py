"""
============================================================
EXERCISE 01: PYTORCH TENSORS BASICS
============================================================

WHAT YOU ARE BUILDING
You are building comfort with the most basic PyTorch object: the tensor.

In this file, you will write small pieces of code that create tensors, inspect shapes, slice values, reshape data, use broadcasting, reduce values, do matrix multiplication, and check simple gradients.

WHY THIS EXERCISE EXISTS
Before you train a model, you need to feel calm around tensors.

If tensor shapes, dtypes, and gradients feel confusing, later training code will also feel confusing.

This file gives you a small safe place to practice those basics.

DATASET
There is no external dataset in this exercise.

You will create tiny example tensors directly in code.

HOW TO LOAD OR DOWNLOAD THE DATASET
You do not need to download anything.

Just run the file.

INPUTS AND OUTPUTS
Inputs:
- tiny tensors you create yourself

Outputs:
- new tensors with expected shapes and values
- a simple gradient result for one small autograd example

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
There is no real model here.

You will only use very small tensor operations.

LOSS FUNCTION
There is no training loss in this exercise.

For autograd practice, you will build one simple scalar expression so PyTorch can compute gradients.

METRICS TO TRACK
Track these checks:
- shape correctness
- dtype correctness
- simple value correctness
- gradient correctness

TARGET RESULT RANGE
Your tensor checks should all pass exactly.

COMMON MISTAKES
- mixing up rows and columns
- using the wrong dtype
- forgetting that PyTorch indexing starts at 0
- reshaping to a shape with the wrong number of elements
- expecting broadcasting to work when dimensions do not line up
- forgetting to call `backward()` before reading a gradient

STEP-BY-STEP TODO
1. Create a few basic tensors.
2. Return their shapes and dtypes.
3. Slice out a small piece from a tensor.
4. Reshape a flat tensor into a matrix.
5. Add tensors with broadcasting.
6. Compute sums and means.
7. Multiply two matrices.
8. Build one tiny autograd example and check the gradient.

STRETCH GOALS
- try `torch.zeros`, `torch.ones`, and `torch.arange`
- compare `view` and `reshape`
- print shapes after every operation

DEFINITION OF DONE
You are done when:
- your helper checks pass
- you can explain each tensor operation in simple words
- your gradient check matches the expected answer
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

from common.training_checks import print_pass_or_try_again


@dataclass
class TensorBasicsConfig:
    """Simple settings for this exercise."""

    seed: int = 42
    # This is the target shape for run_reshape_task.
    expected_matrix_shape: tuple[int, int] = (2, 3)
    # This is the target value for run_autograd_task.
    expected_gradient: float = 6.0



# TODO: Create one or more small tensors and return them in a dictionary.
# Beginner hint:
# - include at least one flat tensor with exactly 6 values
# - that 6-value tensor will be useful for the reshape task below
def load_practice_tensors() -> dict[str, torch.Tensor]:
    raise NotImplementedError("Create and return the practice tensors.")



class AutogradPracticeModule(nn.Module):
    """Optional tiny module for gradient practice.

    Important:
    - You do not have to use this class in run_autograd_task.
    - A plain tensor with requires_grad=True is enough for this exercise.

    This class is only here as a small example of a module with one trainable value.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a simple scaled tensor."""
        return x * self.weight



# TODO: Slice a small part of the given tensor and return it.
# Beginner hint:
# - pick one small slice and stay consistent
# - for example, you might return the first 2 values or the last 2 values
# - the main goal is to practice indexing and slicing, not to guess a hidden trick
def run_indexing_and_slicing_task(base_tensor: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Return a small slice of the input tensor.")


# TODO: Reshape the flat tensor into the expected 2D shape.
# Beginner hint:
# - the expected shape for this exercise is stored in TensorBasicsConfig.expected_matrix_shape
# - right now that shape is (2, 3)
# - so the input tensor must have exactly 6 total values
def run_reshape_task(flat_tensor: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Reshape the flat tensor to the expected matrix shape.")


# TODO: Use broadcasting to add these tensors and return the result.
# Beginner hint:
# - choose shapes that can broadcast cleanly
# - a common simple example is a matrix plus a length-matching row vector
def run_broadcasting_task(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Add the tensors using broadcasting.")


# TODO: Return reduction results such as sum and mean.
# Beginner hint:
# - start with all-values sum and all-values mean
# - you can add dimension-based reductions later if you want
def run_reduction_task(values: torch.Tensor) -> dict[str, torch.Tensor]:
    raise NotImplementedError("Return the sum and mean of the tensor.")


# TODO: Multiply two matrices and return the result.
# Beginner hint:
# - choose shapes where the inner dimensions match
# - example: (2, 3) @ (3, 2) works, but (2, 3) @ (2, 3) does not
def run_matrix_multiplication_task(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Return the matrix multiplication result.")


def run_autograd_task() -> float:
    raise NotImplementedError("Compute and return the gradient for the autograd example.")

# TODO: Use the functions above and return a small summary dictionary.
# Beginner hint:
# - this function is the organizer
# - call your helper tasks
# - use check_shape(...) with config.expected_matrix_shape
# - use check_gradient(...) with config.expected_gradient
# - return a small dictionary such as {"shape_ok": ..., "gradient_ok": ...}
def run_tensor_task_suite(config: TensorBasicsConfig) -> dict[str, float | bool]:
    raise NotImplementedError("Run the tensor tasks and return the summary checks.")








# TODO: Use this if you want one place to print extra details later.
# Beginner hint:
# - this can read the summary from run_tensor_task_suite
# - you can keep it simple and just return the same dictionary at first
def evaluate_tensor_task_suite(summary: dict[str, float | bool]) -> dict[str, float | bool]:
    raise NotImplementedError("Return the evaluation summary.")


def check_shape(actual_shape: tuple[int, ...], expected_shape: tuple[int, ...]) -> bool:
    """Return True when a tensor shape matches the expected shape."""
    return actual_shape == expected_shape



def check_gradient(actual_gradient: float, expected_gradient: float, tolerance: float = 1e-6) -> bool:
    """Return True when a gradient is close to the expected value."""
    return abs(actual_gradient - expected_gradient) <= tolerance



def check_metrics(metrics: dict[str, float | bool]) -> bool:
    """Return True when the key tensor checks pass."""
    shape_ok = bool(metrics.get("shape_ok", False))
    gradient_ok = bool(metrics.get("gradient_ok", False))
    return shape_ok and gradient_ok



def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
        print(f"[DONE] {task_name}")
        return result, True
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False



def main() -> None:
    """Run a small starter check for the tensor exercise."""
    config = TensorBasicsConfig()
    print("Running exercise 01 starter scaffold...")
    print("Current progress:")

    completed_steps = 0
    total_steps = 8

    practice_tensors, ok = report_task_progress("load_practice_tensors", load_practice_tensors)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    base_tensor = practice_tensors.get("tensor 2")
    if base_tensor is None:
        print('[TODO] load_practice_tensors: include a tensor with key "tensor 2" so the scaffold can run the checks.')
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("run_indexing_and_slicing_task", run_indexing_and_slicing_task, base_tensor)
    if ok:
        completed_steps += 1

    _, ok = report_task_progress("run_reshape_task", run_reshape_task, base_tensor)
    if ok:
        completed_steps += 1

    left_for_broadcast = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right_for_broadcast = torch.tensor([10.0, 20.0, 30.0])
    _, ok = report_task_progress("run_broadcasting_task", run_broadcasting_task, left_for_broadcast, right_for_broadcast)
    if ok:
        completed_steps += 1

    _, ok = report_task_progress("run_reduction_task", run_reduction_task, base_tensor.float())
    if ok:
        completed_steps += 1

    left_for_matmul = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right_for_matmul = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _, ok = report_task_progress("run_matrix_multiplication_task", run_matrix_multiplication_task, left_for_matmul, right_for_matmul)
    if ok:
        completed_steps += 1

    _, ok = report_task_progress("run_autograd_task", run_autograd_task)
    if ok:
        completed_steps += 1

    metrics, ok = report_task_progress("run_tensor_task_suite", run_tensor_task_suite, config)
    if ok:
        completed_steps += 1

    evaluated_metrics = None
    if metrics is not None:
        evaluated_metrics, ok = report_task_progress("evaluate_tensor_task_suite", evaluate_tensor_task_suite, metrics)
        if ok:
            completed_steps += 1
    else:
        print("[TODO] evaluate_tensor_task_suite: waiting for run_tensor_task_suite to finish first.")

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if evaluated_metrics is not None:
        passed = check_metrics(evaluated_metrics)
        print_pass_or_try_again(passed, "tensor checks", float(int(passed)), "shape_ok=True and gradient_ok=True")


if __name__ == "__main__":
    main()

