"""
============================================================
EXERCISE 45: ROBUSTNESS AND DISTRIBUTION SHIFT
============================================================

WHAT YOU ARE BUILDING
You are building a small robustness test.

Plain-language meaning:
- distribution shift means test data differs from training data
- robustness means the model still works when inputs change
- shortcut learning means the model uses an easy clue that may fail later

WHY THIS EXERCISE EXISTS
Models can look good on clean validation data and fail on changed data.
Research engineers need to test those changes.

This exercise teaches:
- clean vs shifted evaluation
- input corruptions
- spurious correlations
- performance drops
- failure notes

HOW THIS FITS INTO THE BOOTCAMP
After standard evaluation and failure analysis,
this file asks a harder question:
- does the model still work when the data changes a bit?

Beginner intuition:
- a model may learn the real task
- or it may learn an easy shortcut that only works on familiar data
- robustness testing helps you tell the difference

Small example:
- if a model learned "bright images mean class A"
- then a simple brightness change may hurt it badly
- that shows it relied on a shortcut, not a deeper pattern

DATASET
Use a small real classification dataset.

Clear default:
- `sklearn.datasets.load_wine`
- clean input features: 13 numeric wine measurements
- labels: 3 wine classes
- shifted validation data: copy the validation features, then change one feature pattern in a controlled way

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load clean training data from `sklearn.datasets.load_wine`
- create clean validation data from the same dataset
- create shifted validation data by changing one feature pattern
- compare results on clean and shifted data

INPUTS AND OUTPUTS
Input:
- clean examples
- shifted examples

Output:
- clean accuracy
- shifted accuracy
- performance drop
- failure note

METRICS TO TRACK
Track at least:
- clean accuracy
- shifted accuracy
- accuracy drop
- one robustness note
- a few shifted failures

TARGET RESULT RANGE
Focus on the gap between clean and shifted performance.

Useful signs:
- clean accuracy is clearly higher than random
- shifted accuracy is lower than clean accuracy when the shift matters
- the failure note explains what changed and why the model may have struggled

COMMON MISTAKES
- changing too many things at once in the shifted data
- comparing clean training data to shifted test data without keeping track of the exact shift
- reporting only the shifted score without the clean baseline
- making the shift so extreme that it becomes a totally different task

STEP-BY-STEP TODO
1. Train a small classifier on clean data.
2. Create shifted validation data.
3. Add noise or corruption.
4. Add a spurious feature in training.
5. Evaluate clean and shifted data.
6. Measure the performance drop.
7. Inspect failed examples.
8. Write a robustness note.

DEFINITION OF DONE
You are done when:
- clean accuracy is reported
- shifted accuracy is reported
- performance drop is reported
- failures are inspected
- you can explain distribution shift in plain words

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small classifier.
Keep the model simple so the data shift stays easy to reason about.

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

What each step is doing in simple words:
1. train on normal data
2. make a changed version of that data
3. test the same model on both
4. measure how much performance drops
5. read some failures and explain them
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


ShiftPlan = dict[str, str | float]
CleanLoaders = dict[str, DataLoader]
MetricRow = dict[str, float]
FailureExample = dict[str, int | float | list[float]]
RobustnessSummary = dict[str, float | str | list[FailureExample]]
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
class RobustnessConfig:
    """Settings for robustness and distribution shift."""

    num_samples: int = 178
    num_features: int = 13
    num_classes: int = 3
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    noise_strength: float = 0.5
    random_state: int = 42


def describe_shift_plan(config: RobustnessConfig) -> ShiftPlan:
    """Return the shifts to test."""
    return {
        "clean_eval": "same style as training data",
        "shifted_eval": "noise, corruption, or changed feature pattern",
        "noise_strength": config.noise_strength,
    }



# Build the clean train, validation, and test dataloaders.
# These are the baseline data the model sees before any shift is applied.
def load_clean_dataloaders(config: RobustnessConfig) -> CleanLoaders:
    """Return clean train, val, and test dataloaders."""
    # Goal:
    # - Load the clean Wine dataset and return train, validation, and test dataloaders.
    # - These clean loaders are the baseline before any distribution shift is added.
    #
    # Input:
    # - `config.batch_size`: number of examples per batch.
    # - `config.random_state`: seed used for repeatable train/val/test splits.
    # - Wine has 13 numeric features and 3 classes, so the config defaults use `num_features=13` and `num_classes=3`.
    #
    # Output:
    # - Return `CleanLoaders`.
    # - Exact keys:
    #   - `"train"`: training dataloader
    #   - `"val"`: clean validation dataloader
    #   - `"test"`: clean test dataloader
    # - Each batch should be a tuple:
    #   `(features, labels)`
    # - `features`: `torch.float32` tensor with shape `[batch_size, 13]`
    # - `labels`: `torch.long` tensor with shape `[batch_size]`
    #
    # Method:
    # - Use scikit-learn and PyTorch:
    #   `from sklearn.datasets import load_wine`
    #   `from sklearn.model_selection import train_test_split`
    #   `from torch.utils.data import TensorDataset, DataLoader`
    # - Load data with:
    #   `data = load_wine()`
    # - Split with:
    #   `train_test_split(features, labels, test_size=0.4, random_state=config.random_state, stratify=labels)`
    # - Split the held-out part again into validation and test.
    #
    # Steps:
    # 1. Load Wine features and labels.
    # 2. Convert features to `torch.float32`.
    # 3. Standardize features using the training mean and standard deviation.
    #    Use `train_mean = x_train.mean(dim=0, keepdim=True)` and `train_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)`.
    # 4. Convert labels to `torch.long`.
    # 5. Wrap each split with `TensorDataset(features, labels)`.
    # 6. Build dataloaders with `DataLoader(dataset, batch_size=config.batch_size, shuffle=...)`.
    # 7. Shuffle only the training loader.
    # 8. Return the three exact keys listed above.
    #
    # Checks:
    # - `set(loaders) == {"train", "val", "test"}`
    # - `features, labels = next(iter(loaders["train"]))`
    # - `features.shape[1] == config.num_features`
    # - `features.dtype == torch.float32`
    # - `labels.dtype == torch.long`
    raise NotImplementedError("Return clean dataloaders.")
# Create the shifted evaluation loader.
# Start with one simple shift only, such as added noise or changed feature scale.
def make_shifted_loader(config: RobustnessConfig, clean_loader: DataLoader) -> DataLoader:
    """Return a shifted version of a clean loader."""
    # Goal:
    # - Return a new dataloader where the features have one controlled shift.
    # - Keep labels unchanged so clean and shifted scores are comparable.
    #
    # Input:
    # - `config.noise_strength`: how much Gaussian noise to add.
    # - `clean_loader`: validation or test dataloader with `(features, labels)` batches.
    #
    # Output:
    # - Return one `DataLoader`.
    # - Batches should still be `(features, labels)`.
    # - Shifted `features` should have the same shape and dtype as the clean features.
    #
    # Method:
    # - Read all clean batches, concatenate them, and clone the features:
    #   `features = torch.cat(feature_batches, dim=0).clone()`
    #   `labels = torch.cat(label_batches, dim=0).clone()`
    # - Add noise with:
    #   `shifted_features = features + config.noise_strength * torch.randn_like(features)`
    # - Wrap the shifted tensors with:
    #   `TensorDataset(shifted_features, labels)`
    #   `DataLoader(dataset, batch_size=clean_loader.batch_size, shuffle=False)`
    #
    # Steps:
    # 1. Loop over `clean_loader` and collect feature and label batches.
    # 2. Concatenate the feature batches and label batches.
    # 3. Clone features so the clean loader is not changed in place.
    # 4. Add Gaussian noise to the features.
    # 5. Keep labels unchanged.
    # 6. Build and return the shifted dataloader.
    #
    # Checks:
    # - `shifted_x, shifted_y = next(iter(shifted_loader))`
    # - `clean_x, clean_y = next(iter(clean_loader))`
    # - `shifted_x.shape[1] == clean_x.shape[1]`
    # - `shifted_y.dtype == clean_y.dtype`
    # - The shifted features should not be exactly equal to the clean features.
    raise NotImplementedError("Return shifted or corrupted evaluation data.")





# Beginner hints for this file:
# - clean accuracy can look good even when robustness is weak
# - distribution shift means the new data is not quite like the training data
# - start with one simple shift, such as added noise, before testing many shifts
# - the important number is often the performance drop from clean to shifted data
# - read a few failures so the drop has a real story behind it

class RobustnessMLP(nn.Module):
    """Small classifier for robustness tests."""

    def __init__(self, config: RobustnessConfig) -> None:
        # Goal:
        # - Build a small MLP classifier for the Wine features.
        #
        # Input:
        # - `config.num_features`: input feature count, 13 for Wine.
        # - `config.hidden_dim`: hidden layer size.
        # - `config.num_classes`: output class count, 3 for Wine.
        #
        # Output:
        # - Return `None`, but create the layers used by `forward`.
        #
        # Method:
        # - Use this exact module shape:
        #   `self.net = nn.Sequential(`
        #   `    nn.Linear(config.num_features, config.hidden_dim),`
        #   `    nn.ReLU(),`
        #   `    nn.Linear(config.hidden_dim, config.num_classes),`
        #   `)`
        # - The final layer returns raw class scores called logits.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Save `self.config = config` if useful.
        # 3. Create `self.net` with the layers above.
        #
        # Checks:
        # - `model(torch.randn(4, config.num_features)).shape == (4, config.num_classes)`
        # - Do not add softmax. `nn.CrossEntropyLoss` expects raw logits.
        #
        super().__init__()
        raise NotImplementedError("Build a small classifier.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits."""
        # Goal:
        # - Convert a batch of Wine features into raw class logits.
        #
        # Input:
        # - `x`: `torch.float32` tensor with shape `[batch_size, num_features]`.
        # - For Wine, `num_features` is 13.
        #
        # Output:
        # - Return logits with shape `[batch_size, num_classes]`.
        # - For Wine, that is `[batch_size, 3]`.
        #
        # Method:
        # - Call the sequential model:
        #   `return self.net(x)`
        #
        # Steps:
        # 1. Pass `x` through `self.net`.
        # 2. Return the logits directly.
        #
        # Checks:
        # - `logits.shape[0] == x.shape[0]`
        # - `logits.shape[1] == self.config.num_classes` if you saved the config.
        # - Do not apply softmax here.
        #
        raise NotImplementedError("Return raw logits.")


# Build the classifier, loss, and optimizer.
# Keep the model simple so the effect of the data shift stays easy to interpret.
def build_training_objects(config: RobustnessConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, loss function, and optimizer."""
    # Goal:
    # - Build the classifier, loss function, and optimizer used for clean training.
    #
    # Input:
    # - `config.learning_rate`: optimizer learning rate.
    # - `config.num_features`, `config.hidden_dim`, and `config.num_classes`: model sizes.
    #
    # Output:
    # - Return type: tuple[nn.Module, nn.Module, torch.optim.Optimizer]
    # - Tuple order must be:
    #   `(model, loss_fn, optimizer)`
    #
    # Method:
    # - Use these exact calls:
    #   `model = RobustnessMLP(config).to(device)`
    #   `loss_fn = nn.CrossEntropyLoss()`
    #   `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    # - `CrossEntropyLoss` expects logits with shape `[batch_size, num_classes]` and labels with shape `[batch_size]`.
    #
    # Steps:
    # 1. Build the model and move it to `device`.
    # 2. Create the cross-entropy loss.
    # 3. Create the AdamW optimizer.
    # 4. Return the tuple in the exact order listed above.
    #
    # Checks:
    # - `next(model.parameters()).device.type` should match `device`.
    # - `isinstance(loss_fn, nn.CrossEntropyLoss)`
    #
    raise NotImplementedError("Return model, loss, and optimizer.")


# Train the classifier for one epoch on clean data.
# The goal is to compare what this clean-trained model does on clean vs shifted inputs.
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> MetricRow:
    """Train one epoch."""
    # Goal:
    # - Train the model for one pass over the clean training loader.
    # - Return average loss and accuracy for that epoch.
    #
    # Input:
    # - `model`: classifier that returns logits.
    # - `loader`: dataloader yielding `(features, labels)`.
    # - `loss_fn`: usually `nn.CrossEntropyLoss`.
    # - `optimizer`: usually `torch.optim.AdamW`.
    #
    # Output:
    # - Return `MetricRow`.
    # - Exact keys:
    #   - `"loss"`: average training loss
    #   - `"accuracy"`: training accuracy from 0.0 to 1.0
    #
    # Method:
    # - Use the standard PyTorch training order:
    #   `model.train()`
    #   `optimizer.zero_grad()`
    #   `logits = model(features)`
    #   `loss = loss_fn(logits, labels)`
    #   `loss.backward()`
    #   `optimizer.step()`
    # - Predictions:
    #   `predictions = logits.argmax(dim=1)`
    # - Accuracy formula:
    #   `correct / total_examples`
    #
    # Steps:
    # 1. Put the model in training mode.
    # 2. For each batch, move features and labels to `device`.
    # 3. Run the forward pass.
    # 4. Compute loss.
    # 5. Clear gradients, backpropagate, and step the optimizer.
    # 6. Accumulate `loss.item() * batch_size`, number correct, and total examples.
    # 7. Return average loss and accuracy.
    #
    # Checks:
    # - `0.0 <= metrics["accuracy"] <= 1.0`
    # - Average loss should use all examples, not only the final batch.
    #
    raise NotImplementedError("Return train loss and accuracy.")


# Evaluate one loader and return:
# - loss
# - accuracy
# Use this for both clean and shifted data so the comparison is fair.
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> MetricRow:
    """Evaluate one loader."""
    # Goal:
    # - Evaluate the same model on one loader and return loss and accuracy.
    # - Use this for both clean and shifted loaders so the comparison is fair.
    #
    # Input:
    # - `model`: classifier that returns logits.
    # - `loader`: dataloader yielding `(features, labels)`.
    # - `loss_fn`: usually `nn.CrossEntropyLoss`.
    #
    # Output:
    # - Return `MetricRow`.
    # - Exact keys:
    #   - `"loss"`: average loss
    #   - `"accuracy"`: accuracy from 0.0 to 1.0
    #
    # Method:
    # - Use eval mode and no gradient tracking:
    #   `model.eval()`
    #   `with torch.no_grad():`
    # - Predictions:
    #   `predictions = logits.argmax(dim=1)`
    # - Accuracy formula:
    #   `correct / total_examples`
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Turn off gradients with `torch.no_grad()`.
    # 3. Move each batch to `device`.
    # 4. Run logits and loss.
    # 5. Count correct predictions.
    # 6. Return average loss and accuracy over all examples.
    #
    # Checks:
    # - `0.0 <= metrics["accuracy"] <= 1.0`
    # - This function should not call `optimizer.step()`.
    #
    raise NotImplementedError("Return loss and accuracy.")


# Return a few concrete failures from the shifted setting.
# These examples help explain the size of the performance drop.
def inspect_shift_failures(model: nn.Module, clean_loader: DataLoader, shifted_loader: DataLoader) -> list[FailureExample]:
    """Return examples that pass clean eval but fail shifted eval."""
    # Goal:
    # - Return a few examples where the model is correct on clean data but wrong on shifted data.
    # - These examples help explain the performance drop.
    #
    # Input:
    # - `model`: trained classifier.
    # - `clean_loader`: clean validation or test loader.
    # - `shifted_loader`: shifted loader made from the same examples in the same order.
    #
    # Output:
    # - Return `list[FailureExample]`.
    # - Each failure dictionary should have exactly these keys:
    #   - `"index"`: int position in the inspected data
    #   - `"label"`: true class id
    #   - `"clean_prediction"`: model prediction on clean features
    #   - `"shifted_prediction"`: model prediction on shifted features
    #   - `"clean_features"`: small list of clean feature values
    #   - `"shifted_features"`: small list of shifted feature values
    # - Keep only the first few failures, such as 5.
    #
    # Method:
    # - Iterate through clean and shifted loaders together:
    #   `for clean_batch, shifted_batch in zip(clean_loader, shifted_loader):`
    # - Run both batches in eval mode with `torch.no_grad()`.
    # - Use:
    #   `clean_pred = clean_logits.argmax(dim=1)`
    #   `shifted_pred = shifted_logits.argmax(dim=1)`
    # - A robustness failure is:
    #   `clean_pred == label and shifted_pred != label`
    #
    # Steps:
    # 1. Put the model in eval mode.
    # 2. Loop through clean and shifted batches together.
    # 3. Move features and labels to `device`.
    # 4. Compute clean and shifted predictions.
    # 5. For each example, check the failure condition above.
    # 6. Store up to 5 failure dictionaries.
    # 7. Convert tensor values to Python ints, floats, and lists before storing them.
    #
    # Checks:
    # - `len(failures) <= 5`
    # - If failures exist, `set(failures[0])` should match the keys listed above.
    #
    raise NotImplementedError("Return a few robustness failure examples.")


# Write a short robustness note.
# A good note should say:
# - clean accuracy
# - shifted accuracy
# - the size of the drop
# - one plain guess about what shortcut or weakness may be present
def write_robustness_note(clean_metrics: MetricRow, shifted_metrics: MetricRow) -> str:
    """Write a short note about the performance drop."""
    # Goal:
    # - Explain the clean-vs-shifted result in plain language.
    # - The note should say what changed and how much accuracy dropped.
    #
    # Input:
    # - `clean_metrics`: dictionary from `evaluate` on clean data.
    # - `shifted_metrics`: dictionary from `evaluate` on shifted data.
    # - Both must include `"accuracy"`.
    #
    # Output:
    # - Return one short string.
    # - Include clean accuracy, shifted accuracy, and accuracy drop.
    #
    # Method:
    # - Formula:
    #   `accuracy_drop = clean_metrics["accuracy"] - shifted_metrics["accuracy"]`
    # - Use an f-string:
    #   `f"Clean accuracy was {clean_acc:.3f}, shifted accuracy was {shifted_acc:.3f}, so the drop was {drop:.3f}."`
    #
    # Steps:
    # 1. Read clean and shifted accuracy.
    # 2. Compute the drop.
    # 3. Write one or two direct sentences.
    # 4. Mention that the shift was added noise or feature corruption.
    # 5. Return the note.
    #
    # Checks:
    # - `isinstance(note, str)`
    # - The note should include the word `"drop"` or `"lower"`.
    # - Do not claim the model is robust if shifted accuracy is much lower.
    raise NotImplementedError("Explain the shift result in plain language.")


def run_robustness_experiment(config: RobustnessConfig) -> RobustnessSummary:
    """Run the full robustness workflow."""
    # Goal:
    # - Train on clean data, evaluate on clean and shifted data, then return a robustness summary.
    #
    # Input:
    # - `config`: dataset, model, training, and shift settings.
    #
    # Output:
    # - Return `RobustnessSummary`.
    # - Exact keys:
    #   - `"clean_accuracy"`: clean validation or test accuracy
    #   - `"shifted_accuracy"`: shifted validation or test accuracy
    #   - `"accuracy_drop"`: `clean_accuracy - shifted_accuracy`
    #   - `"clean_loss"`: clean loss
    #   - `"shifted_loss"`: shifted loss
    #   - `"shift_failures"`: list from `inspect_shift_failures`
    #   - `"robustness_note"`: note from `write_robustness_note`
    #
    # Method:
    # - Call the helpers in this order:
    #   `load_clean_dataloaders(config)`
    #   `make_shifted_loader(config, loaders["val"])`
    #   `build_training_objects(config)`
    #   `train_one_epoch(model, loaders["train"], loss_fn, optimizer)` for each epoch
    #   `evaluate(model, loaders["val"], loss_fn)`
    #   `evaluate(model, shifted_loader, loss_fn)`
    #   `inspect_shift_failures(model, loaders["val"], shifted_loader)`
    #   `write_robustness_note(clean_metrics, shifted_metrics)`
    #
    # Steps:
    # 1. Seed the run with `set_seed(config.random_state)`.
    # 2. Load clean dataloaders.
    # 3. Make a shifted validation loader.
    # 4. Build model, loss, and optimizer.
    # 5. Train for `config.epochs` epochs on the clean training loader.
    # 6. Evaluate clean validation data.
    # 7. Evaluate shifted validation data.
    # 8. Compute `accuracy_drop`.
    # 9. Inspect a few shifted failures.
    # 10. Write the note.
    # 11. Return the summary dictionary.
    #
    # Checks:
    # - `summary["accuracy_drop"] == summary["clean_accuracy"] - summary["shifted_accuracy"]`
    # - `0.0 <= summary["clean_accuracy"] <= 1.0`
    # - `0.0 <= summary["shifted_accuracy"] <= 1.0`
    #
    raise NotImplementedError("Train, evaluate clean and shifted data, and return a summary.")


def check_metrics(metrics: RobustnessSummary) -> bool:
    """Return True when robustness summary values are present."""
    required = {"clean_accuracy", "shifted_accuracy", "accuracy_drop", "robustness_note"}
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
    """Run the robustness scaffold."""
    config = RobustnessConfig()
    set_seed(config.random_state)

    completed_steps = 0
    total_steps = 8

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    loaders, ok = report_task_progress("load_clean_dataloaders", load_clean_dataloaders, config)
    if not count_or_stop(ok):
        return


    _, ok = report_task_progress("describe_shift_plan", describe_shift_plan, config)
    if not count_or_stop(ok):
        return

    shifted_loader, ok = report_task_progress("make_shifted_loader", make_shifted_loader, config, loaders["val"])
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("RobustnessMLP.__init__", RobustnessMLP, config)
    if not count_or_stop(ok):
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("RobustnessMLP.forward", model, sample_x)
    if not count_or_stop(ok):
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return

    training_model, loss_fn, optimizer = training_objects
    _, ok = report_task_progress("train_one_epoch", train_one_epoch, training_model, loaders["train"], loss_fn, optimizer)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_robustness_experiment", run_robustness_experiment, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "robustness summary", float(int(passed)), "robustness keys present")

if __name__ == "__main__":
    main()
