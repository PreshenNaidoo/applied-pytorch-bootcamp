"""
============================================================
EXERCISE 12: BETTER METRICS AND FAILURE ANALYSIS
============================================================

WHAT YOU ARE BUILDING
You are building a model evaluation report that goes beyond accuracy.

Plain-language meanings:
- a confusion matrix shows which classes get mixed up
- precision asks: when the model predicts a class, how often is it right?
- recall asks: out of all real examples of a class, how many did the model find?
- F1 is one number that balances precision and recall
- failure analysis means reading mistakes and explaining what went wrong

WHY THIS EXERCISE EXISTS
Accuracy can hide problems.
A model can look good overall and still fail badly on one class.

This exercise teaches:
- per-class metrics
- confusion matrix reading
- inspecting wrong predictions
- writing a short failure report

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises mostly used one main score such as accuracy.
This file teaches the next evaluation habit:
- look deeper
- see which classes fail
- read actual mistakes instead of trusting one average number

What each step is doing in simple words:
1. collect predictions and true labels
2. build a confusion matrix so class mix-ups are visible
3. compute per-class metrics instead of only one global score
4. read wrong predictions directly
5. write a short note that names a real failure pattern

Beginner intuition:
- one score can hide a lot
- a model can look good overall and still fail badly on one class
- reading mistakes is often how you discover the next improvement idea

Small example:
- imagine `90%` overall accuracy
- that sounds strong
- but maybe one class is almost always confused with another
- the confusion matrix and per-class recall would show that problem quickly

DATASET
Use a small real multiclass text dataset.

Clear default:
- Hugging Face dataset: `ag_news`
- text field: `text`
- label field: `label`
- labels: 4 news topics
- use a tiny subset first, such as 2,000 train examples and 500 validation examples

HOW TO LOAD OR CREATE THE DATASET
Recommended beginner path:
- load `ag_news`
- build a tiny vocabulary from the training text
- turn each example into a simple bag-of-words vector
- train one simple classifier
- run the full metrics pipeline on validation or test data
- keep the text pipeline simple so the metrics stay easy to inspect

METRICS TO TRACK
Track at least:
- overall accuracy
- confusion matrix
- per-class precision
- per-class recall
- per-class F1
- five wrong examples
- one plain-language failure note

INPUTS AND OUTPUTS
Prediction inputs:
- model predictions
- true labels

Useful output objects:
- confusion matrix
- per-class metric dictionary
- list of wrong examples

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a simple classifier.
This exercise is about evaluation quality, not model complexity.

LOSS FUNCTION
Use the same loss you used to train the classifier, usually `nn.CrossEntropyLoss`.

TARGET RESULT RANGE
Focus on whether the report exposes real weaknesses.

Useful signs:
- one class has meaningfully different precision or recall from another
- the confusion matrix tells a clear story
- the failure note names a real error pattern instead of a vague complaint

COMMON MISTAKES
- trusting accuracy alone
- mixing up rows and columns in the confusion matrix
- computing metrics on training data instead of validation or test data
- writing a failure report without looking at actual mistakes

STEP-BY-STEP TODO
1. Train or load a small multiclass model.
2. Collect all predictions and labels on validation data.
3. Build a confusion matrix.
4. Compute precision, recall, and F1 for each class.
5. Find at least five wrong predictions.
6. Write a short failure analysis with concrete examples.

DEFINITION OF DONE
You are done when:
- per-class metrics are returned
- a confusion matrix is returned
- wrong examples are inspected
- the final note explains one real failure pattern
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
class BetterMetricsConfig:
    """Settings for better metrics and failure analysis."""

    num_samples: int = 1500
    num_features: int = 20
    num_classes: int = 4
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    random_state: int = 42
    num_wrong_examples: int = 5
    class_names: tuple[str, ...] = ("class_0", "class_1", "class_2", "class_3")


def describe_metrics_plan(config: BetterMetricsConfig) -> dict[str, object]:
    """Return the metrics this exercise should produce."""
    return {
        "class_names": list(config.class_names),
        "metrics": ["accuracy", "confusion_matrix", "precision", "recall", "f1"],
        "num_wrong_examples": config.num_wrong_examples,
    }



# Build the dataloaders for the multiclass task.
# Return:
# - `train`
# - `val`
# - `test`
# Keep the model simple. The main lesson is how to read mistakes.
def load_dataloaders(config: BetterMetricsConfig) -> dict[str, DataLoader]:
    """Return train, val, and test dataloaders."""
    # One clean implementation plan:
    # - load `ag_news` with `datasets.load_dataset`
    # - combine the original train and test splits if you want one shared split
    # - make one `60 / 20 / 20` split
    # - build a tiny vocabulary from the training text only
    # - turn each text into a fixed-size bag-of-words vector
    # - return batches with:
    #   - `features`: `[batch_size, vocab_size]`
    #   - `labels`: `[batch_size]`
    # Why bag-of-words is a good fit here:
    # - the model can stay a simple MLP or linear classifier
    # - the metric and failure-analysis code stays the main lesson
    # Simple bag-of-words recipe:
    # - lowercase the text
    # - split on spaces
    # - count known words
    # - store the counts in one fixed-length float vector
    raise NotImplementedError("Return dataloaders for a multiclass task.")




# Beginner hints for this file:
# - accuracy is one summary number, but it can hide class-specific failures
# - the confusion matrix is often the fastest way to see which classes are getting mixed up
# - precision asks "when the model says class X, how often is that right?"
# - recall asks "out of all true class X examples, how many did the model find?"
# - failure analysis should use real wrong examples, not only summary numbers

class MetricsMLP(nn.Module):
    """Small classifier used for metrics practice."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("Build a small multiclass classifier.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits."""
        raise NotImplementedError("Return raw logits from the classifier.")
# Build the simple model, loss, and optimizer.
# Return:
# - model
# - loss function
# - optimizer
def build_training_objects(config: BetterMetricsConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, loss function, and optimizer."""
    raise NotImplementedError("Return model, CrossEntropyLoss, and optimizer.")


# Train for one epoch and return the training summary.
# This part is basic on purpose.
# The real work of this exercise happens after training, when you inspect errors.
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train for one epoch."""
    raise NotImplementedError("Return train loss and accuracy.")


# Collect everything needed for analysis from one loader.
# Return at least:
# - predictions
# - labels
# - raw examples if available
# This is the bridge from model output to failure analysis.
def collect_predictions(model: nn.Module, loader: DataLoader) -> dict[str, torch.Tensor | list[object]]:
    """Collect predictions, labels, and optional raw examples from a loader."""
    # This function is the bridge from training to analysis.
    # Once you have all predictions and labels in one place,
    # confusion matrix and per-class metrics become much easier to compute.
    raise NotImplementedError("Return predictions, labels, and examples from validation or test data.")


# Build the confusion matrix.
# Shape:
# - `[num_classes, num_classes]`
# Reading rule:
# - rows = true class
# - columns = predicted class
def compute_confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return a confusion matrix with shape `[num_classes, num_classes]`."""
    # One common reading rule:
    # - rows = true class
    # - columns = predicted class
    # Then each row shows what that true class got turned into.
    raise NotImplementedError("Compute the confusion matrix.")


def compute_per_class_metrics(confusion_matrix: torch.Tensor) -> dict[str, list[float]]:
    """Return per-class precision, recall, and F1."""
    raise NotImplementedError("Compute precision, recall, and F1 for each class.")


def find_wrong_examples(
    prediction_artifacts: dict[str, torch.Tensor | list[object]],
    num_examples: int,
) -> list[dict[str, object]]:
    """Return a few wrong predictions for inspection."""
    raise NotImplementedError("Return wrong examples with true and predicted labels.")


def write_failure_analysis(
    class_names: tuple[str, ...],
    confusion_matrix: torch.Tensor,
    wrong_examples: list[dict[str, object]],
) -> str:
    """Write a short note about one visible failure pattern."""
    # Good failure analysis is concrete.
    # It should point to one class mix-up or one repeated mistake,
    # not only say "the model needs improvement."
    raise NotImplementedError("Write a plain-language failure analysis.")


def run_training(config: BetterMetricsConfig) -> dict[str, object]:
    """Train, evaluate, compute metrics, and write a failure note."""
    raise NotImplementedError("Run the better-metrics workflow.")


def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when the final summary has the expected metric outputs."""
    required = {"accuracy", "confusion_matrix", "per_class_metrics", "wrong_examples", "failure_note"}
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
    """Run the better metrics scaffold one step at a time."""
    config = BetterMetricsConfig()
    set_seed(config.random_state)

    completed_steps = 0
    total_steps = 12

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config)
    if not count_or_stop(ok):
        return


    _, ok = report_task_progress("describe_metrics_plan", describe_metrics_plan, config)
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("MetricsMLP.__init__", MetricsMLP, config.num_features, config.hidden_dim, config.num_classes)
    if not count_or_stop(ok):
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("MetricsMLP.forward", model, sample_x)
    if not count_or_stop(ok):
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return

    training_model, loss_fn, optimizer = training_objects
    _, ok = report_task_progress("train_one_epoch", train_one_epoch, training_model, loaders["train"], loss_fn, optimizer)
    if not count_or_stop(ok):
        return

    prediction_artifacts, ok = report_task_progress("collect_predictions", collect_predictions, training_model, loaders["val"])
    if not count_or_stop(ok):
        return

    predictions = prediction_artifacts["predictions"]
    labels = prediction_artifacts["labels"]
    confusion, ok = report_task_progress("compute_confusion_matrix", compute_confusion_matrix, predictions, labels, config.num_classes)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("compute_per_class_metrics", compute_per_class_metrics, confusion)
    if not count_or_stop(ok):
        return

    wrong_examples, ok = report_task_progress("find_wrong_examples", find_wrong_examples, prediction_artifacts, config.num_wrong_examples)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("write_failure_analysis", write_failure_analysis, config.class_names, confusion, wrong_examples)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "better metrics summary", float(int(passed)), "all metric keys present")

if __name__ == "__main__":
    main()
