"""
============================================================
EXERCISE 02: IMAGE CLASSIFICATION MULTICLASS
============================================================

WHAT YOU ARE BUILDING
You are building a small CNN for multiclass image classification.

This time the model must choose one class from 10 classes.

Use `FashionMNIST`.

WHY THIS EXERCISE EXISTS
This is the image version of your earlier multiclass vector work.

It teaches:
- conv layers
- pooling
- flatten
- classifier head
- raw logits for many classes
- simple normalization
- light augmentation on the train set only

DATASET
Use `torchvision.datasets.FashionMNIST`.

Each image has shape `[1, 28, 28]`.
That means:
- 1 channel
- height 28
- width 28

HOW TO LOAD OR DOWNLOAD THE DATASET
Use `torchvision.datasets.FashionMNIST(...)`.

Simple path:
- load the official train split
- load the official test split
- combine them
- make your own `60% / 20% / 20%` train / val / test split
- apply train transforms only to the train split
- apply plain tensor + normalize transforms to val and test

Important library note:
- one `torchvision` dataset object only stores one transform
- if you want different train and val/test transforms, use separate dataset objects or a small wrapper after splitting

Use `download=True` the first time.

INPUTS AND OUTPUTS
Input image batch:
- type: `torch.Tensor`
- shape: `[batch_size, 1, 28, 28]`

Label batch:
- type: `torch.Tensor`
- dtype: `torch.long`
- shape: `[batch_size]`
- values: integers from `0` to `9`

Model output:
- type: `torch.Tensor`
- shape: `[batch_size, 10]`
- meaning: one raw logit per class for each image

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build a small CNN with:
- conv layer
- activation
- pooling
- another conv layer
- flatten
- classifier head

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

Important:
- return raw logits
- do not apply softmax before the loss

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, move the model, image batch, and label batch to the same device.
- Device handling belongs inside your training and evaluation code, right before the forward pass.

METRICS TO TRACK
Track at least:
- `train_loss`
  - mean over the whole training epoch
- `train_accuracy`
  - mean over the whole training epoch
- `val_loss`
  - mean over the whole validation loader
- `val_accuracy`
  - mean over the whole validation loader

If you pass the test loader to `evaluate(...)`, return test loss and test accuracy for that loader.

TARGET RESULT RANGE
For FashionMNIST, a simple correct CNN often reaches about `0.80` to `0.90` validation accuracy.

This target is stronger than a simple CIFAR10 CNN target because FashionMNIST is usually easier.

COMMON MISTAKES
- forgetting that FashionMNIST uses 1-channel images
- applying softmax before `CrossEntropyLoss`
- using float labels instead of `torch.long`
- flattening the wrong dimension
- using augmentation on validation or test by mistake

STEP-BY-STEP TODO
1. Load FashionMNIST.
2. Make a `60/20/20` split.
3. Add normalization.
4. Add one small train-only augmentation, like random horizontal flip.
5. Build the CNN.
6. Train the model.
7. Evaluate on the validation loader.
8. Check whether the validation accuracy is in the target range.

STRETCH GOALS
- compare no augmentation vs simple augmentation
- try one more conv block
- print the shape after each block once

DEFINITION OF DONE
You are done when:
- the file trains end to end
- the model returns raw logits with shape `[batch_size, 10]`
- the validation accuracy is in a healthy range
- you can explain the full shape path from image batch to logits
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

import torch
from torch._inductor.ir import NoneLayout
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.metrics import top1_accuracy_from_logits
from common.reproducibility import set_seed
from common.training_checks import check_threshold, print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"





# Build the train, validation, and test loaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Use a 60% train / 20% val / 20% test split.
# Labels must stay as integer class ids with dtype `torch.long`.
def load_dataloaders(config: MulticlassImageConfig) -> dict[str, DataLoader]:

    raise NotImplementedError("Return train, val, and test loaders for FashionMNIST.")










@dataclass
class MulticlassImageConfig:
    """Settings for the multiclass image classification exercise."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    download: bool = False
    num_classes: int = 10
    image_channels: int = 1
    image_height: int = 28
    image_width: int = 28
    hidden_channels: int = 32
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    random_state: int = 42
    target_val_accuracy: float = 0.82


def describe_multiclass_problem(config: MulticlassImageConfig) -> dict[str, object]:
    """Return the expected shapes for the multiclass image task.

    Return a dict with exactly these keys:
    - `input_shape`
    - `label_shape`
    - `num_classes`
    - `output_shape`
    """
    return {
        "input_shape": (config.image_channels, config.image_height, config.image_width),
        "label_shape": (config.batch_size,),
        "num_classes": config.num_classes,
        "output_shape": (config.batch_size, config.num_classes),
    }


class MulticlassImageCNN(nn.Module):
    """Small CNN stub for FashionMNIST classification."""

    def __init__(self, input_channels: int, hidden_channels: int, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("Build a small multiclass CNN with conv layers, pooling, flatten, and a classifier head.")



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits.

        Return:
        - type: `torch.Tensor`
        - shape: `[batch_size, num_classes]`
        """

        raise NotImplementedError("Return raw logits with shape [batch_size, num_classes].")








# Build the model, loss, and optimizer.
# Return:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.CrossEntropyLoss`.
def build_training_objects(config: MulticlassImageConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    raise NotImplementedError("Return the CNN, CrossEntropyLoss, and optimizer.")




# Run one full training epoch.
# Return a dict with exactly these keys:
# - `loss`
# - `accuracy`
# Both values should be Python floats.
# Here:
# - `loss` means mean train loss over the whole epoch
# - `accuracy` means mean train accuracy over the whole epoch
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:

    raise NotImplementedError("Return the mean train loss and mean train accuracy for one full epoch.")








# Run one full evaluation pass.
# Return a dict with exactly these keys:
# - `loss`
# - `accuracy`
# Both values should be Python floats.
# If you pass the validation loader:
# - `loss` means mean validation loss over the whole validation loader
# - `accuracy` means mean validation accuracy over the whole validation loader
# If you pass the test loader:
# - `loss` means mean test loss over the whole test loader
# - `accuracy` means mean test accuracy over the whole test loader
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    raise NotImplementedError("Return the mean loss and mean accuracy for the loader you evaluate.")






# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `train_accuracy`
# - `val_loss`
# - `val_accuracy`
def run_training(config: MulticlassImageConfig) -> dict[str, float]:

    raise NotImplementedError("Run the full training flow and return the final train and val metrics.")






def check_metrics(metrics: dict[str, float], target: float) -> bool:
    """Return True when validation accuracy reaches the target."""
    return check_threshold(metrics.get("val_accuracy", 0.0), target)


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
    """Run the multiclass image scaffold one step at a time."""
    config = MulticlassImageConfig()
    set_seed(config.random_state)
    print("Running exercise 02 starter scaffold...")
    print("Multiclass task: FashionMNIST with 10 classes.")

    completed_steps = 0
    total_steps = 8

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_multiclass_problem", describe_multiclass_problem, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress(
        "MulticlassImageCNN.__init__",
        MulticlassImageCNN,
        config.image_channels,
        config.hidden_channels,
        config.num_classes,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.image_channels, config.image_height, config.image_width)
    _, ok = report_task_progress("MulticlassImageCNN.forward", model, sample_x)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    training_model, loss_fn, optimizer = training_objects

    _, ok = report_task_progress(
        "train_one_epoch",
        train_one_epoch,
        training_model,
        loaders["train"],
        loss_fn,
        optimizer,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("evaluate", evaluate, training_model, loaders["val"], loss_fn)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics, config.target_val_accuracy)
        print_pass_or_try_again(
            passed,
            "val_accuracy",
            float(metrics["val_accuracy"]),
            f">= {config.target_val_accuracy:.2f}",
        )


if __name__ == "__main__":
    main()

