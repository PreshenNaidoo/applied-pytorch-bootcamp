"""
============================================================
EXERCISE 10: EXPERIMENT TRACKING
============================================================

WHAT YOU ARE BUILDING
You are building a tiny experiment tracker.

Plain-language meaning:
- an experiment is one training run
- tracking means saving the settings and results for that run
- a run name helps you compare results later

WHY THIS EXERCISE EXISTS
Research engineering is not only writing model code.
You also need to know what changed and what happened.

This exercise teaches:
- how to save config values
- how to save metrics for every epoch
- how to compare two runs
- how to write a short result note with real numbers

HOW THIS FITS INTO THE BOOTCAMP
After you can train a model and save checkpoints,
the next habit is to record what changed across runs.
That is how experiments become comparable instead of forgettable.

What each step is doing in simple words:
1. define one run clearly
2. train while writing down what happened each epoch
3. save both settings and results
4. run a second setup with one clear change
5. compare the runs like a small experiment notebook

Beginner intuition:
- experiment tracking is like keeping a lab notebook
- if a run improves, you should know exactly what changed
- if a run gets worse, you should also know exactly what changed

Small example:
- run A uses learning rate `1e-3`
- run B uses learning rate `1e-4`
- if run B does better, the saved log should make that obvious

DATASET
Use one real multiclass vector dataset for every tracked run.

Clear default:
- `sklearn.datasets.load_wine`
- features: 13 numeric wine measurements
- labels: 3 wine classes

Keep the default split:
- 60% train
- 20% validation
- 20% test

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load `sklearn.datasets.load_wine`
- convert features and labels to tensors
- make one fixed train, validation, and test split
- reuse that split for every tracked run
- change only the run settings you want to compare

TRACKING OUTPUTS
Save at least:
- run name
- seed
- learning rate
- batch size
- hidden size
- epoch number
- train loss
- validation loss
- validation accuracy

Simple default format:
- JSON is easiest for nested data
- CSV is also fine for one row per epoch

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small MLP classifier.

The point of this exercise is not model design.
The point is to learn how to record what happened during a run.

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

METRICS TO TRACK
Track at least:
- `train_loss`
- `val_loss`
- `val_accuracy`
- `run_name`
- `seed`
- one short comparison note

TARGET RESULT RANGE
Focus on clear tracking, not a specific score.

Useful signs:
- each run writes a separate file
- the saved file makes it obvious what changed between runs
- you can compare two runs without reading the code again

COMMON MISTAKES
- changing a setting but not writing it down
- saving only the final metric and losing the learning curve
- comparing two runs that used different data splits
- not saving the random seed

STEP-BY-STEP TODO
1. Build a small dataset and model.
2. Create one config object for a run.
3. Train and collect metrics every epoch.
4. Save config and metrics to disk.
5. Run two configs with one clear difference.
6. Load or read the saved results.
7. Compare the runs in plain language.

DEFINITION OF DONE
You are done when:
- each run has a saved metrics file
- the config is saved with the metrics
- two runs can be compared
- the comparison says what changed and which run did better
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import json

from sklearn.datasets import load_wine


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again
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
class ExperimentTrackingConfig:
    """Settings for one experiment run."""

    run_name: str = "baseline_lr_1e_3"
    num_samples: int = 178
    num_features: int = 13
    num_classes: int = 3
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    random_state: int = 42
    output_dir: str = str(BOOTCAMP_ROOT / "outputs" / "experiment_tracking")
        #raise NotImplementedError("Return raw logits from the classifier.")


def describe_tracking_plan(config: ExperimentTrackingConfig) -> dict[str, object]:
    """Return the main values that should be tracked."""
    return {
        "run_name": config.run_name,
        "tracked_config": asdict(config),
        "output_path": str(Path(config.output_dir) / f"{config.run_name}.json"),
    }



# Build the dataloaders used by each run.
# Return:
# - `train`
# - `val`
# - `test`
# Important:
# - keep the split fixed so the run comparison is fair
def load_dataloaders(config: ExperimentTrackingConfig) -> dict[str, DataLoader]:
    """Return train, val, and test dataloaders."""
    # Keep the data path simple here.
    # The important lesson is not the dataset itself.
    # The important lesson is recording what happened during each run.
    # One clean implementation plan:
    # - load `sklearn.datasets.load_wine`
    # - keep at most `config.num_samples` examples
    # - use the real feature table and real labels from the dataset
    # - convert the features to `torch.float32`
    # - convert the labels to `torch.long`
    # - wrap them in one `TensorDataset`
    # - make one fixed `60 / 20 / 20` split with the exercise seed
    # - return three dataloaders
    # Important:
    # - every tracked run must reuse the same split
    # - otherwise the comparison is not fair
    # Practical note:
    # - this is not an image dataset
    # - each example is just one row of numbers
    # - this keeps the exercise focused on experiment tracking

    data = load_wine()
    features = torch.tensor(data.data[: config.num_samples], dtype=torch.float32)
    labels = torch.tensor(data.target[: config.num_samples], dtype=torch.long)

    data_ds = torch.utils.data.TensorDataset(features, labels)

    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset=data_ds, lengths=(0.6, 0.2, 0.2))

    return {"train": torch.utils.data.DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True),
            "val": torch.utils.data.DataLoader(dataset=val_ds, batch_size=config.batch_size),
            "test": torch.utils.data.DataLoader(dataset=test_ds, batch_size=config.batch_size)}



# Beginner hints for this file:
# - think of one experiment as one named training run
# - save both settings and results, otherwise the numbers are hard to trust later
# - one easy pattern is: one row or one dict per epoch, plus one final summary
# - two runs are only fairly comparable if they use the same split and same seed policy
# - start with JSON if CSV feels too limited for nested information

class TrackingMLP(nn.Module):
    """Small classifier used for experiment tracking."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()

        self.input = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.hidden = nn.Linear(in_features=hidden_dim, out_features=hidden_dim//2)
        self.classification_head = nn.Linear(in_features=hidden_dim//2, out_features=num_classes)

        #raise NotImplementedError("Build a small MLP classifier.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits."""

        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        logits = self.classification_head(x)

        return logits
    #raise NotImplementedError("Return dataloaders for the tracked experiment.")


# Build the simple training objects for one run.
# Return:
# - model
# - loss function
# - optimizer
def build_training_objects(config: ExperimentTrackingConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, loss function, and optimizer."""
    model = TrackingMLP(input_dim=config.num_features, hidden_dim=config.hidden_dim, num_classes=config.num_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())

    return model, loss_fn, optimizer
    #raise NotImplementedError("Return model, CrossEntropyLoss, and optimizer.")


# Train for one epoch and return the key numbers.
# Return:
# - mean train loss
# - mean train accuracy
# Think of this as one row in the experiment notebook.
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run one full training epoch."""

    model.train()

    losses, accs = [], []

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
    #raise NotImplementedError("Return mean train loss and accuracy.")


# Measure validation or test performance.
# Return:
# - mean loss
# - mean accuracy
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:
    """Run one full evaluation pass."""
    model.eval()

    losses, accs = [], []

    for feature, labels in loader:
        features = feature.to(device)
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
    #raise NotImplementedError("Return mean evaluation loss and accuracy.")


# Save one run log to disk.
# The saved file should clearly answer:
# - what settings did I use?
# - how did metrics change each epoch?
# - what was the final result?
def save_run_log(
    config: ExperimentTrackingConfig,
    epoch_metrics: list[dict[str, float | int]],
    final_summary: dict[str, float],
) -> str:
    """Save config, epoch metrics, and final summary to disk."""
    # Think of the saved log as the answer to:
    # - what settings did I use?
    # - how did training move each epoch?
    # - what was the final result?

    save_path = Path(config.output_dir) / f"{config.run_name}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    log_data = {
        "config": asdict(config),
        "epoch_metrics": epoch_metrics,
        "final_summary": final_summary,
    }

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    return str(save_path)

    #raise NotImplementedError("Save the run log as JSON or CSV and return the path.")


# Run one full tracked experiment from start to finish.
# Return one result dictionary containing:
# - config
# - saved log path
# - final metrics
def run_one_experiment(config: ExperimentTrackingConfig) -> dict[str, object]:
    """Train one run and save its tracked results."""

    loaders = load_dataloaders(config)
    model, loss_fn, optimizer = build_training_objects(config)

    epoch_metrics = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], loss_fn, optimizer)
        val_metrics = evaluate(model, loaders["val"], loss_fn)

        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        })

        print(f"Epoch {epoch} -> loss: {train_metrics['loss']:.5f}, acc: {train_metrics['accuracy']:.5f}, val_loss: {val_metrics['loss']:.5f}, val_acc: {val_metrics['accuracy']:.4f}")
    

    final_summary = {
        "final_train_loss": epoch_metrics[-1]["train_loss"],
        "final_train_accuracy": epoch_metrics[-1]["train_accuracy"],
        "final_val_loss": epoch_metrics[-1]["val_loss"],
        "final_val_accuracy": epoch_metrics[-1]["val_accuracy"],
        "best_val_accuracy": max(row["val_accuracy"] for row in epoch_metrics),
    }

    log_path = save_run_log(config, epoch_metrics, final_summary)

    return {
        "config": asdict(config),
        "log_path": log_path,
        "final_metrics": final_summary,
    }
    #raise NotImplementedError("Run one tracked experiment from start to finish.")


def compare_runs(run_results: list[dict[str, object]]) -> dict[str, object]:
    """Compare at least two saved run summaries."""
    # A useful comparison note should say:
    # - what changed
    # - which run did better
    # - by how much
    # - one short guess about why

    run_a = run_results[0]
    run_b = run_results[1]

    acc_a = float(run_a["final_metrics"]["best_val_accuracy"])
    acc_b = float(run_b["final_metrics"]["best_val_accuracy"])

    name_a = str(run_a["config"]["run_name"])
    name_b = str(run_b["config"]["run_name"])

    lr_a = run_a["config"]["learning_rate"]
    lr_b = run_b["config"]["learning_rate"]

    if acc_a >= acc_b:
        best_run_name = name_a
        winner_acc = acc_a
        loser_name = name_b
        loser_acc = acc_b
    else:
        best_run_name = name_b
        winner_acc = acc_b
        loser_name = name_a
        loser_acc = acc_a

    comparison_note = (
        f"{best_run_name} did better. "
        f"The changed setting was learning_rate ({lr_a} vs {lr_b}). "
        f"Best validation accuracy was {winner_acc:.4f} vs {loser_acc:.4f}, "
        f"for a gap of {winner_acc - loser_acc:.4f}. "
        f"One simple guess is that the better learning rate matched the dataset better."
    )

    return {
        "best_run_name": best_run_name,
        "comparison_note": comparison_note,
        "run_results": run_results,
    }
    #raise NotImplementedError("Compare runs using validation accuracy and a plain-language note.")


def run_training(config: ExperimentTrackingConfig) -> dict[str, object]:
    """Run two small experiments and compare them."""

    run_a_config = ExperimentTrackingConfig(
        run_name="baseline_lr_1e_3",
        num_samples=config.num_samples,
        num_features=config.num_features,
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        learning_rate=1e-3,
        epochs=config.epochs,
        random_state=config.random_state,
        output_dir=config.output_dir,
    )

    run_b_config = ExperimentTrackingConfig(
        run_name="compare_lr_1e_4",
        num_samples=config.num_samples,
        num_features=config.num_features,
        num_classes=config.num_classes,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        learning_rate=1e-4,
        epochs=config.epochs,
        random_state=config.random_state,
        output_dir=config.output_dir,
    )

    run_a_result = run_one_experiment(run_a_config)
    run_b_result = run_one_experiment(run_b_config)

    run_results = [run_a_result, run_b_result]
    return compare_runs(run_results)

    #raise NotImplementedError("Run the experiment tracking workflow.")


def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when a comparison summary has the expected keys."""
    required = {"best_run_name", "comparison_note", "run_results"}
    return required.issubset(metrics)


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
    """Run the experiment tracking scaffold one step at a time."""
    config = ExperimentTrackingConfig()
    set_seed(config.random_state)

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if not ok:
        return

    _, ok = report_task_progress("describe_tracking_plan", describe_tracking_plan, config)
    if not ok:
        return

    model, ok = report_task_progress("TrackingMLP.__init__", TrackingMLP, config.num_features, config.hidden_dim, config.num_classes)
    if not ok:
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("TrackingMLP.forward", model, sample_x)
    if not ok:
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "tracking summary", float(int(passed)), "comparison keys present")


if __name__ == "__main__":
    main()
