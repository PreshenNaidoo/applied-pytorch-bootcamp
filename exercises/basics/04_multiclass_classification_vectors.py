"""
============================================================
EXERCISE 04: MULTICLASS CLASSIFICATION VECTORS
============================================================

WHAT YOU ARE BUILDING
You are building a small MLP for multiclass classification on numeric vector data.

This is similar to the binary task, but now the model must choose one class from several classes.

WHY THIS EXERCISE EXISTS
This exercise helps you move from binary thinking to multiclass thinking.

You will practice:
- logits for several classes
- softmax intuition
- `CrossEntropyLoss`
- `argmax` predictions

DATASET
Use `sklearn.datasets.make_classification` with more than two classes.

HOW TO LOAD OR DOWNLOAD THE DATASET
You do not need to download anything.

The dataset is generated in memory.

INPUTS AND OUTPUTS
Inputs:
- feature vectors of shape `[batch_size, num_features]`
- integer class labels such as 0, 1, 2, 3

Outputs:
- one logit per class for each example
- predicted class index from `argmax`
- train and validation metrics

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build a small MLP.

At a high level:
- input layer for vector features
- 1 or 2 hidden layers
- ReLU activations
- final layer with `num_classes` outputs

LOSS FUNCTION
Use `CrossEntropyLoss`.

Simple idea:
- the model returns raw logits with shape `[batch_size, num_classes]`
- the labels should be integer class ids with dtype `torch.long`
- the loss compares those logits to the true class labels
- you do not need to apply softmax before `CrossEntropyLoss`
- in this case nn.CrossEntropyLoss already does the softmax step inside the loss,
    it combines log_softmax and negative log-likelihood in one function

Softmax intuition:
- softmax turns raw scores into class probabilities that add up to 1
- for prediction, you can often just use `argmax` on the logits because the largest logit and the largest softmax value point to the same class

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, keep the model, inputs, and labels on the same device.

METRICS TO TRACK
Track at least:
- training loss
- validation loss
- validation accuracy

TARGET RESULT RANGE
A reasonable implementation often reaches validation accuracy around 0.70 to 0.90, depending on the data settings.

COMMON MISTAKES
- applying softmax before `CrossEntropyLoss`
- using one-hot labels with `CrossEntropyLoss` when integer class labels are expected
- returning the wrong final output shape
- using float labels instead of integer class labels

STEP-BY-STEP TODO
1. Generate a multiclass dataset.
2. Split it into train, validation, and test sets with a 60/20/20 split.
3. Build dataloaders.
4. Write the multiclass MLP.
5. Use `CrossEntropyLoss`.
6. Train the model.
7. Use `argmax` for predictions.
8. Track metrics and compare with the target range.

STRETCH GOALS
- try 3 classes and then 4 classes
- compare a small hidden size and a larger hidden size
- print a few logits and predicted class indices

DEFINITION OF DONE
You are done when:
- the model trains end to end
- validation accuracy is in a healthy range
- you can explain logits, softmax intuition, and `argmax` in simple words
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, Dataset, TensorDataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data_utils import (
    numpy_features_to_tensor,
    numpy_labels_to_long_tensor,
    train_val_test_split,
)
from common.metrics import multiclass_accuracy_from_logits
from common.reproducibility import set_seed
from common.training_checks import check_threshold, print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"





# Build the multiclass dataset and loaders.
# Use a 60% train / 20% val / 20% test split.
# Return a dict with exactly these keys:
# - "train"
# - "val"
# - "test"
# Features should be float tensors.
# Labels should be integer class tensors with dtype `torch.long`.
def load_dataloaders(config: MulticlassClassificationConfig) -> dict[str, DataLoader]:

    raise NotImplementedError("Return train, val, and test loaders for the multiclass dataset.")
    


@dataclass
class MulticlassClassificationConfig:
    """Settings for the multiclass exercise."""

    num_samples: int = 1500
    num_features: int = 24
    num_classes: int = 4
    hidden_dim: int = 96
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 800
    random_state: int = 42
    target_val_accuracy: float = 0.75


class MulticlassMLP(nn.Module):
    """Small MLP stub for multiclass classification."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("Build a small MLP, for example Linear -> ReLU -> Linear with num_classes outputs.")



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits with shape [batch_size, num_classes].

        Do not apply softmax here if you use CrossEntropyLoss.
        """

        raise NotImplementedError("Return raw class logits with shape [batch_size, num_classes].")


    



# Build the three training objects.
# Return:
# - model
# - loss_fn
# - optimizer
# Use CrossEntropyLoss for the loss.
# Important:
# - the model should return raw logits
# - the labels should be integer class ids, not one-hot vectors
def build_training_objects(config: MulticlassClassificationConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    raise NotImplementedError("Return the model, CrossEntropyLoss, and optimizer.")


# Run one full training epoch.
# Return one summary for the whole training epoch.
# Do not return one value per sample.
# Use the whole train loader, then return the mean loss and mean accuracy.
# Return a dict with at least:
# - "loss"
# - "accuracy"
# Use multiclass_accuracy_from_logits(...) for the accuracy.
def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:

    raise NotImplementedError("Return one epoch of training metrics with loss and accuracy.")





# Run one full validation epoch.
# Return one summary for the whole loader.
# If you pass the validation loader, return the mean validation loss and mean validation accuracy.
# If you pass the test loader, return the mean test loss and mean test accuracy.
# Do not return one value per sample.
# Return a dict with the same keys as train_one_epoch(...):
# - "loss"
# - "accuracy"
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:

    raise NotImplementedError("Return validation loss and validation accuracy.")








# Train for several epochs and return the final summary.
# Return final summary numbers, not a history list.
# In this function:
# - "train_loss" means the mean training loss from the final training epoch
# - "train_accuracy" means the mean training accuracy from the final training epoch
# - "val_loss" means the mean validation loss from the final validation pass
# - "val_accuracy" means the mean validation accuracy from the final validation pass
# Return a dict with at least these keys:
# - "train_loss"
# - "train_accuracy"
# - "val_loss"
# - "val_accuracy"
def run_training(config: MulticlassClassificationConfig) -> dict[str, float]:
    
    raise NotImplementedError("Run the full multiclass training flow and return the final train and val metrics.")







def check_metrics(metrics: dict[str, float], target: float = 0.75) -> bool:
    """Return True when validation accuracy meets the target."""
    val_accuracy = float(metrics.get("val_accuracy", 0.0))
    return check_threshold(val_accuracy, target)



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
    """Run the multiclass classification scaffold one step at a time."""
    config = MulticlassClassificationConfig()
    set_seed(config.random_state)
    print("Running exercise 04 starter scaffold...")
    print("Target: validation accuracy is often around 0.70 to 0.90. A good first target is 0.75.")

    completed_steps = 0
    total_steps = 7


    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("MulticlassMLP.__init__", MulticlassMLP, config.num_features, config.hidden_dim, config.num_classes)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("MulticlassMLP.forward", model, sample_x)
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

    _, ok = report_task_progress("evaluate", evaluate, train_model, loaders["val"], loss_fn)
    if ok:
        completed_steps += 1

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics, config.target_val_accuracy)
        print_pass_or_try_again(passed, "val_accuracy", float(metrics.get("val_accuracy", 0.0)), ">= 0.75")


if __name__ == "__main__":
    main()













