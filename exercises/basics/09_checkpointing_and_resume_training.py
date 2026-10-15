"""
============================================================
EXERCISE 09: CHECKPOINTING AND RESUME TRAINING
============================================================

WHAT YOU ARE BUILDING
You are building a small training run that can be saved and continued later.

Plain-language meaning:
- a checkpoint is a saved copy of training progress
- resume training means load that saved progress and keep going
- the best checkpoint is the saved model that did best on validation data

WHY THIS EXERCISE EXISTS
Research runs can take a long time.
If a machine stops, or if you want to continue tomorrow, you need to save your work.

This exercise teaches:
- how to save model weights
- how to save optimizer state
- how to remember which epoch you reached
- how to save the best validation model
- how to continue training from a checkpoint

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises taught how to train a model correctly.
This exercise adds one of the first real research-engineering habits:
- save your work
- restore your work
- continue without losing progress

Beginner intuition:
- think of a checkpoint like a save point in a game
- if training stops, you do not want to restart from the beginning
- you want to come back to the exact same place and keep going

Small example:
- suppose training finished epoch `3`
- you save:
  - model weights
  - optimizer state
  - best validation accuracy so far
  - current epoch number
- later, you load that checkpoint
- now training should continue from epoch `4`, not epoch `1`

DATASET
Use a small real multiclass classification dataset.

Keep the default split:
- 60% train
- 20% validation
- 20% test

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load `sklearn.datasets.load_wine`
- turn the features into `torch.float32`
- turn the labels into `torch.long`
- split once into train, validation, and test
- wrap each split in a `DataLoader`

Why this is a good default:
- it is a real dataset
- it is small
- it works well for a simple MLP
- the checkpointing idea stays the main lesson

INPUTS AND OUTPUTS
Input batch:
- `features`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, num_features]`
- `labels`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`

Checkpoint dictionary:
- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `best_val_accuracy`
- `config`

Plain-language meaning:
- `model_state_dict` stores the model weights
- `optimizer_state_dict` stores optimizer memory, such as Adam running averages
- `epoch` tells you where to continue

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Default path:
- small MLP classifier
- one or two hidden layers
- final linear layer for class logits

Simple mental picture:
- the model is the student
- the checkpoint is the saved notebook
- the optimizer state is the memory of how learning has been moving

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

METRICS TO TRACK
Track at least:
- `train_loss`
- `val_loss`
- `val_accuracy`
- `start_epoch_after_resume`
- `best_val_accuracy`

TARGET RESULT RANGE
Focus on behavior, not a magic score.

Useful signs:
- validation accuracy improves at least a bit during training
- the loaded checkpoint restores the same best score you saved
- resumed training starts at the next epoch, not back at epoch 1

COMMON MISTAKES
- saving only the model, then forgetting the optimizer
- loading the model but starting again from epoch 0 by mistake
- overwriting the best checkpoint with a worse model
- saving a checkpoint but never testing that it can load

What each step is doing in simple words:
1. build one small training setup
2. train long enough that saving progress matters
3. create the latest save point after each epoch
4. keep a separate best save point for the best validation model
5. prove loading works by using a fresh model and optimizer
6. continue from the next epoch instead of starting over
7. return evidence that resume really worked

STEP-BY-STEP TODO
1. Build the real dataset and dataloaders.
2. Build a small MLP classifier.
3. Train for a few epochs.
4. Save a latest checkpoint after each epoch.
5. Save a best checkpoint only when validation accuracy improves.
6. Load the latest checkpoint into a fresh model and optimizer.
7. Resume training from the next epoch.
8. Return a final summary proving resume worked.

DEFINITION OF DONE
You are done when:
- a latest checkpoint file is created
- a best checkpoint file is created
- loading restores model and optimizer state
- resumed training starts from the correct next epoch
- you can explain why optimizer state matters
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from torch.utils.data import DataLoader

import sklearn.datasets as datasets


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import check_threshold, print_pass_or_try_again
# Learning order for this exercise:
# 1. Start with the small shape, data, or config helpers.
# 2. Finish pure helper functions before model classes or training loops.
# 3. Check tensor shapes after each helper.
# 4. Connect helpers in the workflow only after the pieces work.
# 5. If a later TODO depends on an earlier helper, finish the helper first.
#
# The goal is to follow a path, not guess the missing step.
# Do not start with the hardest model or training function unless the earlier helpers already work.


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CheckpointingConfig:
    """Settings for the checkpointing exercise."""

    num_samples: int = 1200
    num_features: int = 20
    num_classes: int = 3
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    first_train_epochs: int = 3
    resume_train_epochs: int = 2
    random_state: int = 42
    target_val_accuracy: float = 0.70
    checkpoint_dir: str = str(BOOTCAMP_ROOT / "outputs" / "checkpoints")
    latest_checkpoint_name: str = "exercise_31_latest.pt"
    best_checkpoint_name: str = "exercise_31_best.pt"
        #raise NotImplementedError("Return raw logits from the classifier.")


def describe_checkpoint_plan(config: CheckpointingConfig) -> dict[str, object]:
    """Return the main checkpoint paths and training stages."""
    checkpoint_dir = Path(config.checkpoint_dir)
    return {
        "first_train_epochs": config.first_train_epochs,
        "resume_train_epochs": config.resume_train_epochs,
        "latest_checkpoint_path": str(checkpoint_dir / config.latest_checkpoint_name),
        "best_checkpoint_path": str(checkpoint_dir / config.best_checkpoint_name),
    }



# Build the train, validation, and test dataloaders.
# Return a dict with exactly these keys:
# - `train`
# - `val`
# - `test`
# Each batch should contain:
# - `features`: `[batch_size, num_features]`
# - `labels`: `[batch_size]`
# In plain words:
# - this gives you one fixed split for the whole checkpoint exercise
def load_dataloaders(config: CheckpointingConfig) -> dict[str, DataLoader]:
    """Return train, val, and test loaders for a small real multiclass task."""
    # Beginner-friendly path:
    # - import `load_wine` from `sklearn.datasets`
    # - read `data.data` and `data.target`
    # - convert features to a float tensor
    # - convert labels to a long tensor
    # - make one `TensorDataset(features, labels)`
    # - split it once into train, val, and test with `random_split`
    # - build three dataloaders from those splits
    # Exact simple split rule:
    # - `train_size = int(total_examples * 0.60)`
    # - `val_size = int(total_examples * 0.20)`
    # - `test_size = total_examples - train_size - val_size`
    # Practical note:
    # - `load_wine` has `13` input features and `3` classes
    # - it is fine to update `config.num_features` and `config.num_classes`
    #   from the dataset after loading it
    # - split it once into train, val, and test
    # - use the same split for the full exercise
    # - the checkpointing idea is the main lesson here, not fancy data handling

    data_wine = datasets.load_wine()
    features = torch.tensor(data_wine.data, dtype=torch.float)
    labels = torch.tensor(data_wine.target, dtype=torch.long)

    data_ds = torch.utils.data.TensorDataset(features, labels)

    gen = torch.Generator()
    gen.manual_seed(config.random_state)

    train_ds, val_ds, test_ds = torch.utils.data.random_split(data_ds, lengths=(0.6, 0.2, 0.2), generator=gen)

    config.num_features = features.shape[1]
    config.num_classes = int(labels.max().item()) + 1


    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}



# Beginner hints for this file:
# - keep the first version very small: one model, one optimizer, one checkpoint dictionary
# - a checkpoint is not only model weights; it should also remember optimizer state and epoch
# - test resume with a fresh model object, not the same one already in memory
# - "best checkpoint" means save only when validation improves
# - if resume starts from the wrong epoch, training logs become confusing very quickly

class CheckpointMLP(nn.Module):
    """Small classifier used for checkpoint practice."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()

        self.input = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.classification_head = nn.Linear(in_features=hidden_dim, out_features = num_classes)

        #raise NotImplementedError("Build a small MLP classifier.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`."""

        x = self.input(x)
        x = F.relu(x)
        logits = self.classification_head(x)
        
        return logits
    #raise NotImplementedError("Return train, val, and test dataloaders.")


# Build the simple model, loss, and optimizer.
# Return:
# - model
# - `nn.CrossEntropyLoss`
# - optimizer
# Keep the model boring so the save/load behavior stays easy to follow.
def build_training_objects(config: CheckpointingConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, loss function, and optimizer."""

    model = CheckpointMLP(input_dim=config.num_features, hidden_dim=config.hidden_dim, num_classes=config.num_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())

    return model, loss_fn, optimizer
    #raise NotImplementedError("Return the model, CrossEntropyLoss, and optimizer.")


# Run one full training epoch.
# Return:
# - `loss`
# - `accuracy`
# Both values should be Python floats.
# This is one epoch summary, not one value per batch.
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run one full training epoch and return mean loss and accuracy."""
    # Small mental model:
    # - one epoch means one full pass through the train loader
    # - keep batch losses in a list or running average
    # - return one final summary, not one value per batch

    losses, accs = [], []

    model.train()

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = logits.argmax(dim=-1).long()
        correct = (preds == labels).float()
        accs.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()

    return {"loss": mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Return one full epoch of train metrics.")


# Run one full evaluation pass.
# Return:
# - `loss`
# - `accuracy`
# Evaluation measures the current model without updating it.
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:
    """Run one full evaluation pass and return mean loss and accuracy."""

    model.eval()

    losses, accs = [], []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = loss_fn(logits, labels)
        losses.append(loss.item())

        preds = logits.argmax(dim=-1).long()
        correct = (preds == labels).float()
        accs.append(correct.mean().item())

    mean_loss = torch.tensor(losses).mean().item()
    mean_acc = torch.tensor(accs).mean().item()
    
    return {"loss": mean_loss, "accuracy": mean_acc}
    #raise NotImplementedError("Return one full evaluation summary.")


# Save one checkpoint file to disk.
# The saved dictionary should include at least:
# - current epoch
# - model weights
# - optimizer state
# - best validation accuracy so far
# Why optimizer state matters:
# - optimizers like Adam remember past gradients
# - without that memory, resume training is not truly resuming
def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_accuracy: float,
    config: CheckpointingConfig,
) -> str:
    """Save training progress to disk.

    Beginner-friendly rule:
    - put everything needed to resume into one dictionary
    - use `torch.save(...)`
    - return the saved path as a string
    """    

    # "epoch" says which epoch was just finished
    # "model_state_dict" stores the model weights
    # "optimizer_state_dict" stores optimizer memory
    # "best_val_accuracy" remembers the best validation score so far
    # "config" stores the settings used for this run

    checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_accuracy": best_val_accuracy,
                "config": asdict(config),
                }

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, save_path)
    
    return str(save_path)
    #raise NotImplementedError("Save a checkpoint dictionary and return the path.")


# Load a checkpoint into a fresh model and optimizer.
# Return enough information to resume cleanly:
# - loaded epoch
# - loaded best validation accuracy
# - loaded checkpoint path
def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    """Load training progress into the given model and optimizer."""
    # Why load into a fresh model and optimizer:
    # - this proves the checkpoint really contains enough information
    # - if it only works with the old in-memory objects, resume is not truly tested

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
                "epoch": checkpoint["epoch"],
                "best_val_accuracy": checkpoint["best_val_accuracy"],
                "checkpoint_path": str(path),
            }   
    #raise NotImplementedError("Load model state, optimizer state, epoch, and best validation accuracy.")


def run_training(config: CheckpointingConfig) -> dict[str, float | int | str]:
    """Run initial training, load the checkpoint, then resume training."""
    # Clear workflow:
    # 1. train a few epochs
    # 2. save checkpoints
    # 3. make a fresh model and optimizer
    # 4. load the saved state
    # 5. continue from the next epoch number
    #
    # Return a summary dictionary with at least these keys:
    # - `best_val_accuracy`
    #   - best validation accuracy seen during the whole run
    # - `start_epoch_after_resume`
    #   - the first epoch number used after loading the latest checkpoint
    # - `latest_checkpoint_path`
    #   - path to the latest checkpoint file
    # - `best_checkpoint_path`
    #   - path to the best checkpoint file
    #
    # `check_metrics(...)` below uses these keys to decide whether the task passed.

    best_val_accuracy = 0.0
    latest_checkpoint_path = os.path.join(config.checkpoint_dir, config.latest_checkpoint_name)
    best_checkpoint_path = os.path.join(config.checkpoint_dir, config.best_checkpoint_name)

    torch.manual_seed(config.random_state)

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    for i in range(config.first_train_epochs):

        train_dict = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_dict = evaluate(model, loaders["val"], loss_fn)

        print(f"Epoch {i+1} -> loss: {train_dict['loss']:.5f}, acc: {train_dict['accuracy']:.5f}, val_loss: {val_dict['loss']:.5f}, val_acc: {val_dict['accuracy']:.4f}")

        if val_dict["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_dict["accuracy"]
            save_checkpoint(path = best_checkpoint_path,
                        epoch = i+1,
                        model = model,
                        optimizer = optimizer,
                        best_val_accuracy=best_val_accuracy,
                        config = config)

        save_checkpoint(path = latest_checkpoint_path,
                        epoch = i+1,
                        model = model,
                        optimizer = optimizer,
                        best_val_accuracy=best_val_accuracy,
                        config = config)

    # Resume trianing step:
    resumed_model, resumed_loss_fn, resumed_optimizer = build_training_objects(config)
    
    checkpoint_dict = load_checkpoint(path = latest_checkpoint_path, model = resumed_model, optimizer=resumed_optimizer)
    start_epoch_after_resume = int(checkpoint_dict["epoch"]) + 1
    best_val_accuracy = float(checkpoint_dict["best_val_accuracy"])

    for i in range(start_epoch_after_resume, start_epoch_after_resume + config.resume_train_epochs):
        train_dict = train_one_epoch(resumed_model, loaders["train"], resumed_loss_fn, resumed_optimizer)
        val_dict = evaluate(resumed_model, loaders["val"], resumed_loss_fn)

        print(f"Epoch {i} -> loss: {train_dict['loss']:.5f}, acc: {train_dict['accuracy']:.5f}, val_loss: {val_dict['loss']:.5f}, val_acc: {val_dict['accuracy']:.4f}")


        if val_dict["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_dict["accuracy"]
            save_checkpoint(path = best_checkpoint_path,
                        epoch = i,
                        model = resumed_model,
                        optimizer = resumed_optimizer,
                        best_val_accuracy=best_val_accuracy,
                        config = config)

        save_checkpoint(path = latest_checkpoint_path,
                        epoch = i,
                        model = resumed_model,
                        optimizer = resumed_optimizer,
                        best_val_accuracy=best_val_accuracy,
                        config = config)

    return {
            "best_val_accuracy": best_val_accuracy,
            "start_epoch_after_resume": start_epoch_after_resume,
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
        }

    #raise NotImplementedError("Run the checkpoint and resume workflow.")


def check_metrics(metrics: dict[str, float | int | str], target: float) -> bool:
    """Return True when the resume summary contains the main expected values."""
    required = {
        "best_val_accuracy",
        "start_epoch_after_resume",
        "latest_checkpoint_path",
        "best_checkpoint_path",
    }
    return required.issubset(metrics) and float(metrics.get("best_val_accuracy", 0.0)) >= target


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the checkpointing scaffold one step at a time."""
    config = CheckpointingConfig()
    set_seed(config.random_state)

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if not ok:
        return


    _, ok = report_task_progress("describe_checkpoint_plan", describe_checkpoint_plan, config)
    if not ok:
        return

    model, ok = report_task_progress(
        "CheckpointMLP.__init__",
        CheckpointMLP,
        config.num_features,
        config.hidden_dim,
        config.num_classes,
    )
    if not ok:
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("CheckpointMLP.forward", model, sample_x)
    if not ok:
        return    

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not ok:
        return

    training_model, loss_fn, optimizer = training_objects
    _, ok = report_task_progress("train_one_epoch", train_one_epoch, training_model, loaders["train"], loss_fn, optimizer)
    if not ok:
        return

    _, ok = report_task_progress("evaluate", evaluate, training_model, loaders["val"], loss_fn)
    if not ok:
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok and metrics is not None:
        passed = check_metrics(metrics, config.target_val_accuracy)
        print_pass_or_try_again(passed, "best_val_accuracy", float(metrics.get("best_val_accuracy", 0.0)), f">= {config.target_val_accuracy}")


if __name__ == "__main__":
    main()
