"""
============================================================
EXERCISE 09: RESEARCH ENGINEER CV MINI PROJECT
============================================================

WHAT YOU ARE BUILDING
You are building a small research-engineering-style comparison project.

This file is more open-ended than the other exercises, but the minimum required path is still clear.

Default comparison in this scaffold:
- approach A: simple CNN baseline
- approach B: transfer learning baseline

Use one shared dataset and compare them fairly.

WHY THIS EXERCISE EXISTS
Research engineering is not only about training one model.

It is also about:
- making a clean train / val / test split
- tracking metrics clearly
- comparing approaches fairly
- writing down what changed
- summarizing what you learned

DATASET
Use a small `CIFAR10` subset to keep the project short.

Default class subset in this scaffold:
- airplane
- automobile
- bird
- ship

That gives a 4-class problem.

Use one shared `60% / 20% / 20%` train / val / test split for every approach in the project.

HOW TO LOAD OR DOWNLOAD THE DATASET
Use `torchvision.datasets.CIFAR10(...)`.

Simple path:
- load train and test
- combine them
- keep only the 4 chosen classes
- remap labels to `0, 1, 2, 3`
- make one shared `60/20/20` split
- reuse the exact same split for every experiment

Use `download=True` the first time.

INPUTS AND OUTPUTS
Input image batch:
- type: `torch.Tensor`
- shape: `[batch_size, 3, 32, 32]` for the CNN baseline
- if you use transfer learning with a pretrained backbone, you may resize to `224 x 224`

Label batch:
- type: `torch.Tensor`
- dtype: `torch.long`
- shape: `[batch_size]`
- values: `0, 1, 2, 3`

Project output:
- a metrics dict for each approach
- one comparison summary
- one short markdown report

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build at least two approaches.

Simple default path:
- baseline CNN from scratch
- pretrained transfer-learning model

LOSS FUNCTION
Use `nn.CrossEntropyLoss` for both approaches.

Important:
- both models should return raw logits
- do not apply softmax before the loss

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, move each model, each image batch, and each label batch to the same device.
- Device handling belongs inside each training and evaluation loop, right before the forward pass.

METRICS TO TRACK
Track at least for each approach:
- `train_loss`
  - final epoch mean over the whole training epoch
- `val_accuracy`
  - mean over the whole validation loader
- `test_accuracy`
  - mean over the whole test loader

Also include:
- one small ablation note
  - for example, image size, augmentation choice, or frozen-vs-unfrozen backbone

TARGET RESULT RANGE
There is no one exact target number here.

Success means:
- both approaches run correctly
- the comparison is fair
- the report is clear
- you can explain which approach worked better and why you think that happened

COMMON MISTAKES
- changing the split between experiments
- comparing models with different preprocessing but not writing that down
- reporting validation accuracy as test accuracy
- writing a report with no concrete numbers
- doing an ablation but not saying exactly what changed

STEP-BY-STEP TODO
1. Create one shared train / val / test split.
2. Build the simple CNN baseline.
3. Build the transfer-learning baseline.
4. Train both approaches.
5. Evaluate both on val and test.
6. Write one short comparison.
7. Add one small ablation.
8. Write the markdown report.

STRETCH GOALS
- compare a third approach
- add a simple confusion matrix
- track training time per epoch

DEFINITION OF DONE
You are done when:
- at least two approaches are compared fairly
- the report includes real numbers
- train / val / test metrics are clearly labeled
- you can explain one lesson from the comparison in plain language
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass


# Build the shared train, validation, and test loaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Use one shared 60% train / 20% val / 20% test split for every project experiment.
# Beginner-friendly fairness rule:
# - build the split one time only
# - save or reuse those exact split indices for every approach
# - do not let the CNN use one split and the transfer model use a different split
# - if preprocessing differs between approaches, write that down in the report
def load_dataloaders(config: VisionMiniProjectConfig) -> dict[str, DataLoader]:
    raise NotImplementedError("Return the shared train, val, and test loaders for the mini project.")

class VisionMiniProjectConfig:
    """Settings for the CV mini project."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    download: bool = False
    class_ids: tuple[int, int, int, int] = (0, 1, 2, 8)
    class_names: tuple[str, str, str, str] = ("airplane", "automobile", "bird", "ship")
    num_classes: int = 4
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 8
    random_state: int = 42
    transfer_image_size: int = 224
    report_path: str = str(BOOTCAMP_ROOT / "reports" / "16_research_engineer_cv_mini_project.md")


def describe_project_scope(config: VisionMiniProjectConfig) -> dict[str, object]:
    """Return the exact comparison scope.

    Return a dict with exactly these keys:
    - `class_names`
    - `num_classes`
    - `required_approaches`
    - `required_report_path`
    """
    return {
        "class_names": list(config.class_names),
        "num_classes": config.num_classes,
        "required_approaches": ["cnn_baseline", "transfer_learning"],
        "required_report_path": config.report_path,
    }


class ProjectBaselineCNN(nn.Module):
    """Simple CNN baseline stub for the mini project."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Beginner-friendly build idea:
        # - make a small image classifier from scratch
        # - simple path:
        #   - conv layer
        #   - activation
        #   - pooling
        #   - maybe repeat once or twice
        #   - flatten
        #   - linear classifier head
        raise NotImplementedError("Build the small CNN baseline for the mini project.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`."""
        raise NotImplementedError("Return raw logits from the CNN baseline.")


class ProjectTransferModel(nn.Module):
    """Transfer-learning baseline stub for the mini project."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Beginner-friendly build idea:
        # - start from a pretrained vision backbone from `torchvision.models`
        # - replace its final classifier layer so it predicts `num_classes`
        # - if you freeze most of the backbone, only the new head learns
        # - if you unfreeze more layers, more of the model learns
        raise NotImplementedError("Build the transfer-learning baseline for the mini project.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`."""
        raise NotImplementedError("Return raw logits from the transfer-learning baseline.")



# Run one full experiment for one named approach.
# Return a dict with exactly these keys:
# - `approach_name`
# - `train_loss`
# - `val_accuracy`
# - `test_accuracy`
# - `ablation_note`
# Metric meaning:
# - `train_loss` means the final epoch mean train loss
# - `val_accuracy` means mean validation accuracy over the whole validation loader
# - `test_accuracy` means mean test accuracy over the whole test loader
# Beginner-friendly experiment rule:
# - this function should handle one named approach from start to finish
# - build the right model for that approach
# - train it
# - evaluate it on the shared validation and test loaders
# - return one final summary dict for that approach
def run_one_experiment(config: VisionMiniProjectConfig, approach_name: str) -> dict[str, float | str]:
    raise NotImplementedError("Run one approach end to end and return the final metrics.")


# Compare at least two finished experiment summaries.
# Return a dict with exactly these keys:
# - `best_approach`
# - `comparison_note`
# - `results`
# `results` should be the same `list[dict[str, float | str]]` you want to report.
# Each dict inside `results` should keep these exact keys:
# - `approach_name`
# - `train_loss`
# - `val_accuracy`
# - `test_accuracy`
# - `ablation_note`
def compare_experiments(experiment_results: list[dict[str, float | str]]) -> dict[str, object]:
    # Beginner-friendly comparison rule:
    # - compare the approaches using real numbers from the same split
    # - a simple default is to choose the best approach by validation accuracy
    # - then write one short note explaining:
    #   - which approach won
    #   - by how much
    #   - what changed between the approaches
    raise NotImplementedError("Compare the finished experiments and return a short summary.")


# Write the short markdown report.
# Return:
# - type: `str`
# - meaning: the markdown text you wrote
# Also save the same text to `config.report_path`.
# Keep one clear default report shape with these markdown headings:
# - `# Goal`
# - `## Dataset`
# - `## Approaches`
# - `## Results`
# - `## Ablation`
# - `## Conclusion`
def write_report(config: VisionMiniProjectConfig, comparison: dict[str, object]) -> str:
    # Beginner-friendly report idea:
    # - `# Goal`: say what you compared
    # - `## Dataset`: say which classes and which split you used
    # - `## Approaches`: say what model A and model B were
    # - `## Results`: include the real metrics
    # - `## Ablation`: say one small thing you changed
    # - `## Conclusion`: say what you learned in plain words
    raise NotImplementedError("Write the short markdown report and save it to the project report path.")


# Run the whole mini project.
# Return a final summary dict with exactly these keys:
# - `best_approach`
# - `report_path`
# - `num_approaches_compared`
def run_project(config: VisionMiniProjectConfig) -> dict[str, object]:
    raise NotImplementedError("Run the full mini project and return the final summary.")


def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when the final project summary contains the expected keys."""
    return "best_approach" in metrics and "report_path" in metrics and "num_approaches_compared" in metrics


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
    """Run the CV mini project scaffold one step at a time."""
    config = VisionMiniProjectConfig()
    set_seed(config.random_state)
    print("Running exercise 09 starter scaffold...")
    print("Main goal: compare at least two vision approaches fairly and write a short report.")

    completed_steps = 0
    total_steps = 11


    _, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_project_scope", describe_project_scope, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    baseline_model, ok = report_task_progress("ProjectBaselineCNN.__init__", ProjectBaselineCNN, config.num_classes)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, 3, 32, 32)
    _, ok = report_task_progress("ProjectBaselineCNN.forward", baseline_model, sample_x)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    transfer_model, ok = report_task_progress("ProjectTransferModel.__init__", ProjectTransferModel, config.num_classes)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    transfer_sample_x = torch.randn(config.batch_size, 3, config.transfer_image_size, config.transfer_image_size)
    _, ok = report_task_progress("ProjectTransferModel.forward", transfer_model, transfer_sample_x)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    cnn_result, ok = report_task_progress("run_one_experiment(cnn_baseline)", run_one_experiment, config, "cnn_baseline")
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    transfer_result, ok = report_task_progress("run_one_experiment(transfer_learning)", run_one_experiment, config, "transfer_learning")
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    comparison, ok = report_task_progress("compare_experiments", compare_experiments, [cnn_result, transfer_result])
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("write_report", write_report, config, comparison)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("run_project", run_project, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")


if __name__ == "__main__":
    main()

