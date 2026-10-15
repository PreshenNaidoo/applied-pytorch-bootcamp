"""
============================================================
EXERCISE 08: DEBUGGING TRAINING FAILURES
============================================================

WHAT YOU ARE BUILDING
You are building a debugging habit.

This file gives you a controlled training setup for synthetic vector data, but something is wrong. Your job is to inspect the setup, 
find the problem or problems, and fix them one by one.

WHY THIS EXERCISE EXISTS
A big part of ML work is not writing perfect code on the first try.

A big part is noticing when training is broken and then debugging it in a calm, structured way.

DATASET
Use synthetic vector classification data.

HOW TO LOAD OR DOWNLOAD THE DATASET
You do not need to download anything.

The data should be created in memory with `sklearn.datasets.make_classification` and split 60% train / 20% val / 20% test.

INPUTS AND OUTPUTS
Inputs:
- vector features
- class labels
- a starter setup with at least one failure in it

Outputs:
- a fixed version of the training setup
- a short note about what was wrong
- a basic metric report that shows learning is happening again

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a very small vector classifier.

Keep the model simple so debugging stays focused on training mistakes.

LOSS FUNCTION
Use one exact default setup in this exercise so the debugging target is clear.

Use this default path:
- binary classification
- model returns one raw logit per example
- labels are float tensors with values like `0.0` and `1.0`
- loss is `BCEWithLogitsLoss`

Part of the debugging work is checking whether the chosen loss really matches the output shape and label type.

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, keep the model, inputs, and labels on the same device.

METRICS TO TRACK
Track at least:
- training loss
- validation loss
- validation accuracy

TARGET RESULT RANGE
There is no one exact number.

The target is that the broken setup becomes a learning setup again.

COMMON MISTAKES
Possible failure modes to inspect:
- wrong output shape
- wrong loss usage
- labels with the wrong dtype
- forgetting `model.train()` or `model.eval()`
- not zeroing gradients
- learning rate too high

DEBUGGING CHECKLIST
1. Print one batch shape.
2. Print model output shape.
3. Check label dtype.
4. Check which loss function is being used.
5. Check whether gradients are cleared each step.
6. Check whether training and validation modes are switched correctly.
7. Check whether the learning rate is reasonable.
8. Change one thing at a time.

STEP-BY-STEP TODO
1. Build the broken starter setup.
2. Run a small inspection pass.
3. Write down what looks suspicious.
4. Fix one issue.
5. Run again.
6. Repeat until training behaves sensibly.
7. Summarize what you fixed.

STRETCH GOALS
- create two different broken versions and compare them
- write a helper that prints a short debug summary
- keep a small list of the bugs you found

DEFINITION OF DONE
You are done when:
- you can explain what was broken
- you can explain why the fix matters
- the training setup shows clear learning after your fixes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.training_checks import print_pass_or_try_again


@dataclass
class DebuggingConfig:
    """Settings for the debugging exercise."""

    num_samples: int = 900
    num_features: int = 12
    num_classes: int = 2
    hidden_dim: int = 32
    batch_size: int = 32
    learning_rate: float = 1e-1
    epochs: int = 12
    random_state: int = 42



# Build one intentionally broken training setup.
# Still use a normal 60% train / 20% val / 20% test split for the data.
# Keep the base task itself simple:
# - binary classification
# - one raw logit per example
# - BCEWithLogitsLoss
# Keep it low-friction:
# - use one or two simple bugs, not many bugs at once
# Return a dict with exactly these keys:
# - "model"
# - "loaders"
# - "loss_fn"
# - "optimizer"
# - "bug_note"
def build_broken_setup(config: DebuggingConfig) -> dict[str, object]:
    raise NotImplementedError("Return one small broken setup with the objects needed for training and debugging.")



class DebugMLP(nn.Module):
    """Simple model stub for the debugging exercise."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        raise NotImplementedError("Build a very small classifier so the bug is easy to spot.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for the task.

        Easiest shape:
        - return shape `[batch_size]`
        - or return `[batch_size, 1]` and squeeze dim 1 before you return
        """
        raise NotImplementedError("Return one raw logit per example. Do not apply sigmoid here.")



# Inspect the broken setup before training.
# Return a small summary dict with plain keys such as:
# - "batch_shape"
# - "label_dtype"
# - "output_shape"
# - "loss_name"
# - "learning_rate"
# - "suspected_issues"
# Good checks here:
# - label dtype matches the loss
# - output shape matches the loss
# - model, data, and labels are on the same device if you use a GPU
def inspect_setup(broken_setup: dict[str, object]) -> dict[str, object]:
    raise NotImplementedError("Return a small debug summary that makes the bug easier to see.")


# Fix the bug or bugs you found.
# Return the same setup dictionary keys as build_broken_setup(...), but corrected.
# Also keep a short list or note of what you changed, for example in a key like "fixed_issues".
def fix_setup(broken_setup: dict[str, object]) -> dict[str, object]:
    raise NotImplementedError("Return the fixed setup and record what you changed.")


# Run the full debugging flow.
# Smallest correct path:
# 1. build the broken setup
# 2. inspect it
# 3. fix it
# 4. train again
# Return final summary numbers, not per-sample values.
# Here, "val_accuracy" means the mean validation accuracy after the setup is fixed.
# Return a dict with at least:
# - "val_accuracy"
# - "fixed_issues"
# You can also include a short plain-language note.
def run_training(config: DebuggingConfig) -> dict[str, float | str]:
    raise NotImplementedError("Return the final validation accuracy and a short list of the issues you fixed.")



def check_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when the debugging summary includes a learning signal."""
    return "val_accuracy" in metrics and "fixed_issues" in metrics



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
    """Run the debugging scaffold one step at a time."""
    config = DebuggingConfig()
    print("Running exercise 08 starter scaffold...")
    print("Main goal: debug training failures one issue at a time.")

    completed_steps = 0
    total_steps = 6

    model, ok = report_task_progress("DebugMLP.__init__", DebugMLP, config.num_features, config.hidden_dim, config.num_classes)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("DebugMLP.forward", model, sample_x)
    if ok:
        completed_steps += 1

    broken_setup, ok = report_task_progress("build_broken_setup", build_broken_setup, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("inspect_setup", inspect_setup, broken_setup)
    if ok:
        completed_steps += 1

    _, ok = report_task_progress("fix_setup", fix_setup, broken_setup)
    if ok:
        completed_steps += 1

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "debugging summary", float(int(passed)), "val_accuracy and fixed_issues recorded")


if __name__ == "__main__":
    main()











