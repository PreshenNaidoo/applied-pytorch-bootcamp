"""
============================================================
EXERCISE 03: BINARY CLASSIFICATION VECTORS
============================================================

WHAT YOU ARE BUILDING
You are building a small MLP for binary classification on numeric vector data.

You will create a synthetic dataset, split it with a 60/20/20 train/val/test split, build a neural network with 1 or 2 hidden layers, train it, evaluate it, and track simple metrics.

Bias and variance, in simple words:
- bias means the model is too simple and misses the real pattern
- variance means the model learns the training data too closely and does worse on new data
- high bias often looks like weak results on both train and validation data
- high variance often looks like strong train results but worse validation results
- the goal is to find a good balance between the two
- in this exercise, the model size and the train/validation metrics help you notice that balance

Another simple way to think about it:
bias is usually linked to underfitting, and variance is usually linked to overfitting.
Underfitting means the model did not learn enough.
Overfitting means the model learned the training data too narrowly and does worse on new data.

High bias can happen when:
- the model is too small
- you train for too few epochs
- the features are weak
- the optimization setup is poor

High variance can happen when:
- the model is too large for the dataset
- you train too long
- the dataset is too small
- regularization is weak

How to reduce high bias:
- use a stronger model
- train longer
- improve the features or input representation
- reduce overly strong regularization

How to reduce high variance:
- use more data
- use regularization such as weight decay or dropout
- use a smaller model
- use early stopping
- add data augmentation when it makes sense

WHY THIS EXERCISE EXISTS
This is your first full supervised learning exercise in this phase.

It connects the earlier pieces:
- data loading
- model definition
- loss choice
- predictions
- metrics

DATASET
Use `sklearn.datasets.make_classification`.

This gives you vector inputs and binary labels.

HOW TO LOAD OR DOWNLOAD THE DATASET
You do not need to download anything.

The dataset is generated in memory.

INPUTS AND OUTPUTS
Inputs:
- feature vectors of shape `[batch_size, num_features]`
- binary labels such as 0 and 1

Outputs:
- one raw model score per example
- predicted class labels
- train and validation metrics

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build a small MLP.

At a high level:
- input layer that matches the number of features
- 1 or 2 hidden layers
- ReLU activations
- one binary output head

Keep it small and readable.

LOSS FUNCTION
Use a binary classification loss.

High-level note:
- `BCEWithLogitsLoss` usually works well when your model returns raw logits
- if you use raw logits, do not apply sigmoid before giving values to `BCEWithLogitsLoss`

Sigmoid vs logits, simple idea:
- sigmoid turns a raw score into a number between 0 and 1
- a logit is the raw score before sigmoid
- for training, it is often cleaner to keep logits and let the loss handle the sigmoid part internally

METRICS TO TRACK
Track at least:
- training loss
- validation loss
- validation accuracy

You can also track:
- training accuracy
- precision and recall later if you want

TARGET RESULT RANGE
With a reasonable implementation, validation accuracy should usually reach about 0.85 or better.

COMMON MISTAKES
- applying sigmoid in the model and then also using `BCEWithLogitsLoss`
- using labels with the wrong shape
- using labels with the wrong dtype
- forgetting to switch between train and eval mode
- using a learning rate that is much too high

STEP-BY-STEP TODO
1. Generate the binary dataset.
2. Split it into train, validation, and test sets with a 60/20/20 split.
3. Build dataloaders.
4. Write the MLP class.
5. Choose the loss.
6. Write the training loop pieces.
7. Turn logits into predictions.
8. Track metrics.
9. Check whether you reached the target range.

STRETCH GOALS
- compare 1 hidden layer vs 2 hidden layers
- try a different hidden size
- print a few logits and predicted labels

DEFINITION OF DONE
You are done when:
- the file trains end to end
- validation accuracy is usually near or above the target
- you can explain logits, sigmoid, and binary predictions in simple words
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data_utils import (
    numpy_features_to_tensor,
    numpy_labels_to_float_tensor,
    train_val_test_split,
)
from common.metrics import binary_accuracy_from_logits
from common.reproducibility import set_seed
from common.training_checks import check_threshold, print_pass_or_try_again

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass

        #"Return raw logits. Do not apply sigmoid here if you use BCEWithLogitsLoss.")


# Build the binary dataset and loaders.
# Return a dict with exactly these keys:
# - "train"
# - "val"
# - "test"
# Use make_classification(...), split the arrays, convert them to tensors, then wrap them in DataLoader.
# Use a 60% train / 20% val / 20% test split.
def load_dataloaders(config: BinaryClassificationConfig) -> dict[str, DataLoader]:

    x, y = make_classification(n_samples = config.num_samples, n_features = config.num_features, n_classes = 2)
    num_train, num_val = int( config.num_samples * 0.6), int( config.num_samples * 0.2)
    
    x_train, y_train = torch.tensor(x[:num_train,:], dtype=torch.float), torch.tensor(y[:num_train], dtype=torch.float)
    x_val, y_val = torch.tensor(x[num_train:num_train+num_val,:], dtype=torch.float), torch.tensor(y[num_train:num_train+num_val], dtype=torch.float)
    x_test, y_test = torch.tensor(x[num_train+num_val:,:], dtype=torch.float), torch.tensor(y[num_train+num_val:], dtype=torch.float)

    train_ds, val_ds, test_ds = TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), TensorDataset(x_test, y_test)

    dict1 = {"train": DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
             "val": DataLoader(dataset=val_ds, batch_size=config.batch_size),
             "test": DataLoader(dataset=test_ds, batch_size=config.batch_size)}

    return dict1

class BinaryClassificationConfig:
    """Settings for the binary classification exercise."""

    num_samples: int = 1200
    num_features: int = 20
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 800
    random_state: int = 42
    target_val_accuracy: float = 0.85


class BinaryMLP(nn.Module):
    """Small MLP stub for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_features=1)

        #Build a small MLP, for example Linear -> ReLU -> Linear with one output logit.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for binary classification.

        Easiest shape:
        - return shape [batch_size]

        If your last layer gives shape [batch_size, 1], squeeze dim 1 before you return it.
        """

        x1 = self.fc1(x)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        x1 = F.relu(x1)
        x1 = self.fc3(x1)

        return x1



# Build the three training objects.
# Return:
# - model
# - loss_fn
# - optimizer
# Use BCEWithLogitsLoss for the loss.
def build_training_objects(config: BinaryClassificationConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    model = BinaryMLP(input_dim=config.num_features, hidden_dim=config.hidden_dim)
    model.to(device=device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(lr = config.learning_rate, params=model.parameters())
    return model, loss_fn, optimizer


# Run one full training epoch.
# Return one summary for the whole training epoch.
# Do not return one value per sample.
# Do not stop after the first batch.
# Use the whole train loader, then return the mean loss and mean accuracy.
# Return a dict with at least:
# - "loss"
# - "accuracy"
# Use binary_accuracy_from_logits(...) for the accuracy.
def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:

    # We call optimizer.zero_grad() to clear out old gradients before computing new ones. Why this matters: 
    # when you call loss.backward(), PyTorch adds gradients into each parameter’s .grad 
    # it does not replace them automatically so if you do not clear them, gradients from 
    # earlier batches get mixed into the new batch
    losses, accuracies = [], []
    for x_data, targets in loader:
        x_data = x_data.to(device=device)
        targets = targets.to(device=device)

        optimizer.zero_grad()
        output = model(x_data)  
        output = output.squeeze()      
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        acc = binary_accuracy_from_logits(output, targets)

        losses.append(loss.item())
        accuracies.append(acc)

    losses = torch.tensor(losses)
    accuracies = torch.tensor(accuracies)

    return {"loss": losses.mean(), "accuracy": accuracies.mean()}

    
    #raise NotImplementedError("Return one epoch of training metrics with loss and accuracy.")


# Run one full validation epoch.
# Return one summary for the whole loader.
# If you pass the validation loader, return the mean validation loss and mean validation accuracy.
# Do not return one value per sample.
# Return a dict with the same keys as train_one_epoch(...):
# - "loss"
# - "accuracy"
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:
    model.eval()


    losses, accuracies = [], []
    for x_data, targets in loader:
        x_data = x_data.to(device=device)
        targets = targets.to(device=device)

        output = model(x_data)
        output = output.squeeze()
        loss = loss_fn(output, targets)
        acc = binary_accuracy_from_logits(output, targets)

        losses.append(loss.item())
        accuracies.append(acc)

    losses = torch.tensor(losses)
    accuracies = torch.tensor(accuracies)

    model.train()
    
    return {"loss": losses.mean(), "accuracy": accuracies.mean()}


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
# The checker only needs val_accuracy, but the other keys make the result easier to read.
def run_training(config: BinaryClassificationConfig) -> dict[str, float]:
    config = BinaryClassificationConfig()

    loaders_dict = load_dataloaders(config)
    train_loader = loaders_dict["train"]
    val_loader = loaders_dict["val"]
    test_loader = loaders_dict["test"]

    model, loss_fn, optimizer = build_training_objects(config)

    train_dict, val_dict = None, None
    for i in range(config.epochs):

        train_dict = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_dict = evaluate(model, val_loader, loss_fn)

        if i %10 == 0:
            print(f"epoch:{i}, loss: {train_dict["loss"]:.4f}, acc: {train_dict["accuracy"]:.4f}, val_loss: {val_dict["loss"]:.4f}, val_acc: {val_dict["accuracy"]:.4f}")

    return {"train_loss": train_dict["loss"],
            "train_accuracy": train_dict["accuracy"],
            "val_loss": val_dict["loss"],
            "val_accuracy": val_dict["accuracy"]}

    #raise NotImplementedError("Run the full binary training flow and return the final train and val metrics.")



def check_metrics(metrics: dict[str, float], target: float = 0.85) -> bool:
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
    """Run the binary classification scaffold one step at a time."""
    config = BinaryClassificationConfig()
    set_seed(config.random_state)
    print("Running exercise 03 starter scaffold...")
    print("Target: validation accuracy should usually be about 0.85 or better.")

    completed_steps = 0
    total_steps = 7


    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("BinaryMLP.__init__", BinaryMLP, config.num_features, config.hidden_dim)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("BinaryMLP.forward", model, sample_x)
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
        print_pass_or_try_again(passed, "val_accuracy", float(metrics.get("val_accuracy", 0.0)), ">= 0.85")


if __name__ == "__main__":
    main()



