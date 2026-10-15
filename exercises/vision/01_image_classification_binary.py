"""
============================================================
EXERCISE 01: IMAGE CLASSIFICATION BINARY
============================================================

WHAT YOU ARE BUILDING
You are building your first small CNN for images.

The task is binary classification.
That means the model must choose between exactly two classes.

In this exercise, use `CIFAR10`, but do not keep all 10 classes.
Keep only:
- class `0`: airplane
- class `1`: automobile

Then remap them into the binary problem:
- airplane -> `0`
- automobile -> `1`

WHY THIS EXERCISE EXISTS
So far, you have worked with vector data.

Images are different.
They have:
- channels
- height
- width

This exercise teaches the basic CNN path:
- convolution blocks
- pooling
- flatten
- classifier head

DATASET
Use `torchvision.datasets.CIFAR10`.

Each image has shape `[3, 32, 32]`.
That means:
- 3 color channels
- height 32
- width 32

HOW TO LOAD OR DOWNLOAD THE DATASET
Use `torchvision.datasets.CIFAR10(...)`.

Simple first-time path:
- load the official train split
- load the official test split
- combine them
- keep only class ids `0` and `1`
- remap them to binary labels `0` and `1`
- make your own `60% / 20% / 20%` train / val / test split

Use `download=True` the first time.
After the data is on disk, `download=False` is fine.

INPUTS AND OUTPUTS
Input image batch:
- type: `torch.Tensor`
- shape: `[batch_size, 3, 32, 32]`

Label batch:
- type: `torch.Tensor`
- dtype: `torch.float32`
- shape: `[batch_size]`
- values: `0.0` or `1.0`

Model output:
- type: `torch.Tensor`
- shape: `[batch_size]` or `[batch_size, 1]`
- meaning: one raw logit per image

If your last layer returns `[batch_size, 1]`, squeeze dim 1 before you return it.

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Build a small CNN with:
- a first conv block
- a second conv block
- pooling to shrink the spatial size
- flatten
- a small classifier head

Keep it small.
You do not need a deep network here.

LOSS FUNCTION
Use `nn.BCEWithLogitsLoss`.

Important:
- the model should return raw logits
- do not apply sigmoid before this loss

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
A small correct CNN should usually reach about `0.90` to `0.98` validation accuracy on this two-class version.

COMMON MISTAKES
- forgetting to remap the two classes to `0` and `1`
- using integer labels with `BCEWithLogitsLoss` instead of float labels
- returning probabilities instead of raw logits
- forgetting normalization
- flattening the wrong dimension before the linear layer
- moving the model to GPU but leaving images or labels on CPU

STEP-BY-STEP TODO
1. Load both CIFAR10 splits.
2. Keep only class ids `0` and `1`.
3. Remap labels to binary `0` and `1`.
4. Make a `60/20/20` split.
5. Add simple normalization.
6. Build the CNN.
7. Train for a few epochs.
8. Evaluate on the validation loader.
9. Check whether your validation accuracy is in the target range.

STRETCH GOALS
- try one more conv block
- compare max pooling and average pooling
- print one batch shape before training

DEFINITION OF DONE
You are done when:
- the file trains end to end
- the model returns one raw logit per image
- the validation accuracy is in a healthy range
- you can explain what the conv blocks and pooling are doing in simple words
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

from scipy import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import torchvision


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.metrics import binary_accuracy_from_logits
from common.reproducibility import set_seed
from common.training_checks import check_threshold, print_pass_or_try_again


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass

        #raise NotImplementedError("Return one raw logit per image. Do not apply sigmoid in forward.")


# Build the train, validation, and test loaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Use a 60% train / 20% val / 20% test split after you filter to the two classes.
# Labels must be float tensors with shape `[batch_size]` for BCEWithLogitsLoss.
def load_dataloaders(config: BinaryImageConfig) -> dict[str, DataLoader]:

    cifar10_train = torchvision.datasets.CIFAR10(root=config.data_root, train=True, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root=config.data_root, train=False, download=True)
    print(type(cifar10_train.data))  #50000,32,32,3
    print(cifar10_train.data.shape)
    print(type(cifar10_train.targets))
    print(cifar10_test.data.shape)   #10000, 32,32,3

    combined_data = np.concat([cifar10_train.data, cifar10_test.data], axis=0)
    combined_labels = np.concat([cifar10_train.targets, cifar10_test.targets], axis=0)
    
    mask = np.isin(combined_labels, [0, 1])
    data, targets = combined_data[mask], combined_labels[mask]

    num_train, num_val = int(data.shape[0] * 0.6), int(data.shape[0] * 0.2)
    x_train, y_train = torch.tensor(data[:num_train], dtype=torch.float), torch.tensor(targets[:num_train], dtype=torch.float)
    x_val, y_val = torch.tensor(data[num_train:num_train+num_val], dtype=torch.float), torch.tensor(targets[num_train:num_train+num_val], dtype=torch.float)
    x_test, y_test = torch.tensor(data[num_train+num_val:], dtype=torch.float), torch.tensor(targets[num_train+num_val:], dtype=torch.float)

    #pytorch expects channel first i.e,  3,32,32  but the data is in 32, 32, 3.
    #Easiest way is to use permute
    x_train = x_train.permute(0, 3, 1, 2)
    x_val = x_val.permute(0, 3, 1, 2)
    x_test = x_test.permute(0, 3, 1, 2)

    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0

    #Alternate way is to use transpose like so:
    # Input shape: (Batch, Height, Width, Channels) -> [1, 32, 32, 3]
    # N refers to batch_size in this example
    # x = torch.randn(1, 32, 32, 3) 
    # 1. Swap Height (dim 1) and Channels (dim 3)
    # x = x.transpose(1, 3) # Shape is now [1, 3, 32, 32] but the last two are (Width, Height)
    # 2. Swap the new dim 2 (Width) and dim 3 (Height) to get NCHW
    # x = x.transpose(2, 3) # Final shape: [1, 3, 32, 32] (NCHW)
    # or you could swap 2 and 3 and then 1 and 2

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    test_ds = TensorDataset(x_test, y_test)

    return {"train": DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": DataLoader(dataset=test_ds, batch_size=config.batch_size)}

class BinaryImageConfig:
    """Settings for the binary image classification exercise."""

    data_root: str = str(BOOTCAMP_ROOT / "data")
    download: bool = False
    negative_class_id: int = 0
    positive_class_id: int = 1
    negative_class_name: str = "airplane"
    positive_class_name: str = "automobile"
    image_channels: int = 3
    image_size: int = 32
    hidden_channels: int = 32
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    random_state: int = 42
    target_val_accuracy: float = 0.90


def describe_binary_problem(config: BinaryImageConfig) -> dict[str, object]:
    """Return the exact binary problem setup.

    Return a dict with exactly these keys:
    - `negative_class_id`
    - `negative_class_name`
    - `positive_class_id`
    - `positive_class_name`
    - `input_shape`
    - `label_shape`
    - `output_shape`
    """
    return {
        "negative_class_id": config.negative_class_id,
        "negative_class_name": config.negative_class_name,
        "positive_class_id": config.positive_class_id,
        "positive_class_name": config.positive_class_name,
        "input_shape": (config.image_channels, config.image_size, config.image_size),
        "label_shape": (config.batch_size,),
        "output_shape": (config.batch_size,),
    }


class BinaryImageCNN(nn.Module):
    """Small CNN stub for binary image classification."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)    
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels*2, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels*2, out_channels=hidden_channels*3, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=4*4*(hidden_channels*3), out_features=1)
        # Instead of hardcoding 4*4 in the linear layer, a clean way is to use nn.AdaptiveAvgPool2d((1, 1))
        # It turns each 4x4 feature map into one number by averaging all 16 values in that 4x4
        # If your tensor is: [batch, channels, 4, 4]
        # then after AdaptiveAvgPool2d, it becomes: [batch, channels, 1, 1]
        # Then the linear layer definition will be self.fc1 = nn.Linear(in_features=hidden_channels * 3, out_features=1)

        # raise NotImplementedError(
        #     "Build a small CNN with conv blocks, pooling, flatten, and one output logit."
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for binary classification.

        Return:
        - type: `torch.Tensor`
        - shape: `[batch_size]`

        If your last layer gives `[batch_size, 1]`, squeeze dim 1 before you return.
        Do not apply sigmoid here.
        """

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.fc1(x)

        return x.squeeze()


    # raise NotImplementedError("Return train, val, and test loaders for the two-class CIFAR10 problem.")


# Build the model, loss function, and optimizer.
# Return this tuple in this exact order:
# - model: `nn.Module`
# - loss_fn: `nn.Module`
# - optimizer: `torch.optim.Optimizer`
# Use `nn.BCEWithLogitsLoss` for the loss.
def build_training_objects(config: BinaryImageConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:

    model = BinaryImageCNN(input_channels = config.image_channels, hidden_channels = config.hidden_channels)
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(lr=config.learning_rate, params = model.parameters())

    return  model, loss_fn, optimizer

    #raise NotImplementedError("Return the CNN, BCEWithLogitsLoss, and optimizer.")


# Run one full training epoch.
# Return one epoch summary dict with exactly these keys:
# - `loss`
# - `accuracy`
# Both values should be Python floats.
# Here:
# - `loss` means mean train loss over the whole epoch
# - `accuracy` means mean train accuracy over the whole epoch
# Use `binary_accuracy_from_logits(...)` for accuracy.
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:

    model.train()

    temp_losses, temp_accs = [], []        
    for x_data, y_targets in loader:

        x_data = x_data.to(device)
        y_targets = y_targets.to(device)
        
        optimizer.zero_grad()

        logits = model(x_data)
        loss = loss_fn(logits, y_targets)
        loss.backward()
        optimizer.step()

        temp_losses.append(loss.item())

        preds = (logits>=0).float()
        correct = (preds == y_targets).float()
        temp_accs.append(float(correct.mean().item()))        

    mean_train_loss = torch.tensor(temp_losses).mean().item()
    mean_train_accs = torch.tensor(temp_accs).mean().item()

    return {"loss": mean_train_loss, "accuracy": mean_train_accs}

    #raise NotImplementedError("Return the mean train loss and mean train accuracy for one full epoch.")


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

    model.eval()

    losses, accuracies = [], []

    for x_data, y_targets in loader:
        x_data = x_data.to(device)
        y_targets = y_targets.to(device)

        logits = model(x_data)

        loss = loss_fn(logits, y_targets)
        losses.append(loss.item())

        preds = (logits>=0).float()
        correct = (preds==y_targets).float()
        accuracies.append(float(correct.mean().item()))

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accuracies).mean().item()

    return {"loss": mean_loss, "accuracy": mean_acc}

    #raise NotImplementedError("Return the mean loss and mean accuracy for the loader you evaluate.")


# Run the full training flow.
# Return a final summary dict with exactly these keys:
# - `train_loss`
# - `train_accuracy`
# - `val_loss`
# - `val_accuracy`
# Each value should be a Python float.
# This return is a final epoch summary, not a history across epochs.
def run_training(config: BinaryImageConfig) -> dict[str, float]:

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    losses, accuracies = [], []
    val_dict = {}

    for i in range(config.epochs):
        
        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        if i % 10 == 0:
            print(f"Epoch {i} -> loss: {train_dict['loss']}, acc {train_dict['accuracy']}, val_loss: {val_dict['loss']}, val_acc: {val_dict['accuracy']}")

    return {"train_loss": train_dict['loss'], "train_accuracy": train_dict['accuracy'], "val_loss":val_dict['loss'], "val_accuracy":val_dict['accuracy']}
    #raise NotImplementedError("Run the full training flow and return the final train and val metrics.")


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
    """Run the binary image scaffold one step at a time."""
    config = BinaryImageConfig()
    set_seed(config.random_state)
    print("Running exercise 01 starter scaffold...")
    print("Binary task: airplane vs automobile on CIFAR10.")

    completed_steps = 0
    total_steps = 8

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("describe_binary_problem", describe_binary_problem, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress(
        "BinaryImageCNN.__init__",
        BinaryImageCNN,
        config.image_channels,
        config.hidden_channels,
    )
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.image_channels, config.image_size, config.image_size)
    _, ok = report_task_progress("BinaryImageCNN.forward", model, sample_x)
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

