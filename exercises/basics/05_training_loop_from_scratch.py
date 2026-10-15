"""
============================================================
EXERCISE 05: TRAINING LOOP FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building the training and validation loop yourself.

The main goal is not the dataset and not the model. The main goal is loop correctness.

You will write:
- `train_one_epoch`
- `validate_one_epoch`
- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`
- `model.train()`
- `model.eval()`
- `torch.no_grad()`

WHY THIS EXERCISE EXISTS
A lot of PyTorch work becomes much easier once you can write a correct training loop from memory.

This exercise is where that habit starts.

DATASET
Reuse simple synthetic vector classification data with a 60/20/20 train/val/test split.

HOW TO LOAD OR DOWNLOAD THE DATASET
You do not need to download anything.

Use `sklearn.datasets.make_classification` to create data in memory.

INPUTS AND OUTPUTS
Inputs:
- feature vectors
- class labels
- a small model
- loss function and optimizer

Outputs:
- one epoch of training metrics
- one epoch of validation metrics
- a history of metrics across epochs

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small vector classifier.

Keep the model simple.

The important part here is not the model design. The important part is writing the loop correctly.

LOSS FUNCTION
Use one exact default setup in this exercise so you do not have to guess.

Use this default path:
- multiclass classification
- model returns raw logits with shape `[batch_size, num_classes]`
- labels are integer class ids with dtype `torch.long`
- loss is `CrossEntropyLoss`

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, keep the model, inputs, and labels on the same device.

METRICS TO TRACK
Track at least:
- train loss
- validation loss
- validation accuracy

TARGET RESULT RANGE
Do not chase a perfect number here.

The main target is that the loop behaves correctly and the metrics improve in a sensible way.

A simple model on easy synthetic data should usually show clear learning.

COMMON MISTAKES
- forgetting `optimizer.zero_grad()`
- calling `backward()` during validation
- forgetting `model.train()` before training
- forgetting `model.eval()` before validation
- forgetting `torch.no_grad()` during validation
- averaging losses incorrectly across batches

STEP-BY-STEP TODO
1. Build a small dataset and loaders.
2. Build a small model.
3. Write `train_one_epoch`.
4. Write `validate_one_epoch`.
5. Run several epochs.
6. Save metrics in a history dictionary.
7. Check whether loss goes down and accuracy improves.

STRETCH GOALS
- print batch-level loss for the first epoch
- add a progress bar with `tqdm`
- compare a correct loop and a broken loop

DEFINITION OF DONE
You are done when:
- your training loop runs end to end
- train and validation phases are clearly separated
- your metrics show sensible learning
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from scipy import optimize
from sympy import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.metrics import multiclass_accuracy_from_logits

from common.data_utils import (
    numpy_features_to_tensor,
    numpy_labels_to_long_tensor,
    train_val_test_split,
)
from common.training_checks import print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"





# Build the synthetic dataset and loaders.
# Use a 60% train / 20% val / 20% test split.
# Return a dict with exactly these keys:
# - "train"
# - "val"
# - "test"
# Features should be float tensors.
# Labels should be integer class tensors with dtype `torch.long`.
def load_dataloaders(config: TrainingLoopConfig) -> dict[str, DataLoader]:

    raise NotImplementedError("Return train, val, and test loaders for the synthetic dataset.")





@dataclass
class TrainingLoopConfig:
    """Settings for the training loop exercise."""

    num_samples: int = 1000
    num_features: int = 16
    num_classes: int = 3
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 15
    random_state: int = 42


class VectorClassifier(nn.Module):
    """Simple model stub used by the training loop exercise."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("Build a small classifier, for example Linear -> ReLU -> Linear.")



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits with shape [batch_size, num_classes]."""

        raise NotImplementedError("Return raw class logits from the model.")





# Build the three training objects.
# Return:
# - model
# - loss_fn
# - optimizer
# For this exercise, use CrossEntropyLoss.
# Important:
# - the model returns raw logits
# - the labels are integer class ids with dtype `torch.long`
def build_training_objects(config: TrainingLoopConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    raise NotImplementedError("Return the model, loss function, and optimizer.")






# Run one full training epoch.
# Small checklist for each batch:
# 1. model.train()
# 2. optimizer.zero_grad()
# 3. forward pass
# 4. loss.backward()
# 5. optimizer.step()
# Return one summary for the whole training epoch.
# Do not return one value per sample.
# Use the whole train loader, then return the mean loss and mean accuracy.
# Return a dict with at least:
# - "loss"
# - "accuracy"
def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    

    raise NotImplementedError("Return one epoch of training metrics with loss and accuracy.")




    


# Run one full validation epoch.
# Small checklist:
# 1. model.eval()
# 2. use torch.no_grad()
# 3. do not call backward()
# Return one summary for the whole loader.
# If you pass the validation loader, return the mean validation loss and mean validation accuracy.
# If you pass the test loader, return the mean test loss and mean test accuracy.
# Return a dict with at least:
# - "loss"
# - "accuracy"
@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    raise NotImplementedError("Return one epoch of validation metrics with loss and accuracy.")








# Run several epochs and keep the history.
# Each item in these lists should be one epoch-level mean.
# Example:
# - train_losses[i] is the mean training loss for epoch i
# - val_accuracies[i] is the mean validation accuracy for epoch i
# Return a dict with these list keys:
# - "train_losses"
# - "train_accuracies"
# - "val_losses"
# - "val_accuracies"
def run_training(config: TrainingLoopConfig) -> dict[str, list[float] | float]:

    raise NotImplementedError("Run the full experiment and return metric history lists for train and val.")






def check_metrics(metrics: dict[str, list[float] | float]) -> bool:
    """Return True when the metric history looks usable."""
    train_losses = metrics.get("train_losses", [])
    val_losses = metrics.get("val_losses", [])
    val_accuracies = metrics.get("val_accuracies", [])
    return bool(train_losses) and bool(val_losses) and bool(val_accuracies)



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
    """Run the training loop scaffold one step at a time."""
    config = TrainingLoopConfig()
    print("Running exercise 05 starter scaffold...")
    print("Main goal: write a correct training loop from scratch.")

    completed_steps = 0
    total_steps = 7


    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("VectorClassifier.__init__", VectorClassifier, config.num_features, config.hidden_dim, config.num_classes)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("VectorClassifier.forward", model, sample_x)
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

    train_model, loss_fn, optimizer = training_objects

    _, ok = report_task_progress("train_one_epoch", train_one_epoch, train_model, loaders["train"], loss_fn, optimizer)
    if ok:
        completed_steps += 1

    _, ok = report_task_progress("validate_one_epoch", validate_one_epoch, train_model, loaders["val"], loss_fn)
    if ok:
        completed_steps += 1

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "loop history", float(int(passed)), "non-empty train/val metric history")


if __name__ == "__main__":
    main()














