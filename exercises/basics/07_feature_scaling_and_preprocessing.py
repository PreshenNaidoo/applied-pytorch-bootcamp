"""
============================================================
EXERCISE 07: FEATURE SCALING AND PREPROCESSING
============================================================

WHAT YOU ARE BUILDING
You are building a simple preprocessing pipeline for vector data.

You will compare training with no scaling and training with standardization.

WHY THIS EXERCISE EXISTS
Many models train better when numeric features are on similar scales.

This exercise teaches two very important ideas:
- why scaling can help
- why preprocessing must be fit on training data only

DATASET
Reuse simple synthetic vector classification data with a 60/20/20 train/val/test split.

HOW TO LOAD OR DOWNLOAD THE DATASET
You do not need to download anything.

Generate the data with `sklearn.datasets.make_classification` and split it 60% train / 20% val / 20% test.

INPUTS AND OUTPUTS
Inputs:
- raw numeric feature vectors
- labels
- one preprocessing choice

Outputs:
- a no-scaling experiment result
- a scaling experiment result
- a short comparison of the two

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small vector classifier.

The model can stay simple because the main idea here is preprocessing.

LOSS FUNCTION
Use one exact default setup in this exercise so you do not have to guess.

Use this default path:
- binary classification
- model returns one raw logit per example
- labels are float tensors with values like `0.0` and `1.0`
- loss is `BCEWithLogitsLoss`

DEVICE NOTE
- CPU is fine for this exercise.
- If you use a GPU, keep the model, inputs, and labels on the same device.

METRICS TO TRACK
Track at least:
- train loss
- validation loss
- validation accuracy

Also compare:
- no scaling vs scaling

TARGET RESULT RANGE
There is no single exact target.

Expected note:
- scaling often makes optimization smoother or faster
- sometimes the final accuracy improves
- sometimes the main difference is stability, not the final number

COMMON MISTAKES
- fitting the scaler on train + validation + test together
- fitting a different scaler on each split
- forgetting to reuse the train-fitted scaler on validation and test data
- mixing NumPy preprocessing and tensor conversion in a confusing order

STEP-BY-STEP TODO
1. Build the raw dataset and splits with a 60/20/20 split.
2. Train one baseline with no scaling.
3. Fit standardization values on the training features only.
4. Apply the same transform to validation and test features.
5. Train the same model again with scaled features.
6. Compare the results in simple words.
7. Write one note about data leakage and why it matters.

STRETCH GOALS
- print the training-feature means and standard deviations before and after scaling
- try a dataset with more uneven feature scales
- save your comparison notes in a small dictionary

DEFINITION OF DONE
You are done when:
- both experiments run end to end
- you can explain why train-only fitting matters
- you can describe how scaling changed training behavior
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset, dataloader


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.training_checks import print_pass_or_try_again

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class PreprocessingConfig:
    """Settings for the preprocessing comparison exercise."""

    num_samples: int = 1200
    num_features: int = 20
    num_classes: int = 2
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 200
    random_state: int = 42




# Build the raw dataset splits.
# Use a 60% train / 20% val / 20% test split.
# Return a dict with these six NumPy keys:
# - "X_train", "X_val", "X_test"
# - "y_train", "y_val", "y_test"
def load_raw_splits(config: PreprocessingConfig) -> dict[str, object]:
    raise NotImplementedError("Return NumPy train, val, and test arrays with the six split keys.")






@dataclass
class Standardizer:
    """Simple container for mean and standard deviation."""

    mean: np.ndarray
    std: np.ndarray




# Fit the standardizer on training features only.
# Return a Standardizer(mean=..., std=...).
# If any std value is 0, replace it with 1.0 so division stays safe.
def fit_standardizer(train_features: np.ndarray) -> Standardizer:

    raise NotImplementedError("Return the training-set mean and std only.")

    #std[std == 0] = 1





# Apply the saved train-only standardizer.
# Return a new NumPy array with the same shape as features.
# Formula: (features - mean) / std
def apply_standardizer(features: np.ndarray, standardizer: Standardizer) -> np.ndarray:

    raise NotImplementedError("Return the standardized features using the saved train mean and std.")





# Build loaders from ready-made arrays.
# Return a dict with keys:
# - "train"
# - "val"
# - "test"
# Features should become float tensors.
# Labels should become float tensors if you use BCEWithLogitsLoss.
def build_dataloaders_from_arrays(feature_arrays: dict[str, np.ndarray], label_arrays: dict[str, np.ndarray], batch_size: int) -> dict[str, DataLoader]:
    
    raise NotImplementedError("Return train, val, and test loaders built from the given arrays.")
    



class SimpleVectorClassifier(nn.Module):
    """Model stub used for the preprocessing comparison."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("Build one small classifier and reuse the exact same model shape in both experiments.")



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for the classification task.

        Easiest shape:
        - return shape `[batch_size]`
        - or return `[batch_size, 1]` and squeeze dim 1 before you return
        """

        raise NotImplementedError("Return one raw logit per example. Do not apply sigmoid here.")




# Run one full experiment.
# use_scaling=False means raw features.
# use_scaling=True means fit on train only, then apply to train, val, and test.
# Return final summary numbers for that experiment, not per-sample values.
# In this function:
# - "train_loss" means the mean training loss from the final training epoch
# - "val_loss" means the mean validation loss from the final validation pass
# - "val_accuracy" means the mean validation accuracy from the final validation pass
# Return a dict with at least:
# - "train_loss"
# - "val_loss"
# - "val_accuracy"
def run_single_experiment(config: PreprocessingConfig, use_scaling: bool) -> dict[str, float]:

    raise NotImplementedError("Return one experiment result with train_loss, val_loss, and val_accuracy.")






    # nn.BCEWithLogitsLoss() expects raw logits and applies the sigmoid part internally.
    # your model should return raw scores
    # do not apply sigmoid in the model before this loss
    # Use threshold 0.0 on the raw logits.
    # sigmoid of 0.0 is exactly 0.5
    # So these are equivalent:
    # logit >= 0.0 and sigmoid(logit) >= 0.5







# Compare the two experiment settings.
# These accuracy values should be final validation means from each full experiment.
# Return a dict with at least:
# - "no_scaling_val_accuracy"
# - "scaled_val_accuracy"
# - "comparison_note"
# Write comparison_note in plain language.
def run_training(config: PreprocessingConfig) -> dict[str, float | str]:    

    raise NotImplementedError("Return both validation accuracies and one plain-language comparison note.")


    # Results after training for 200 epochs:
    # No Scaling: 
    # train_loss: 0.45248574018478394, train_acc: 0.8600543737411499, val_loss: 0.48533564805984497, val_acc: 0.81640625 
    # Scaling: 
    # train_loss: 0.5997560620307922, train_acc: 0.7663043737411499, val_loss: 0.5989771485328674, val_acc: 0.76953125
    # Explanation:
    # Scaling often helps, but it does not always give better results right away.
    # when you scale the inputs, the model now sees different-sized numbers than before. That means the learning rate, 
    # number of epochs, and other settings that worked well before may no longer be the best settings after scaling. 
    # So scaling can be better in principle, but your current training setup may fit the unscaled version better.
    # Also, if scaling made things clearly worse, there may be a bug in the preprocessing. 
    # For example: 
    # fitting the standardizer on the wrong split 
    # not applying the same standardizer to validation data 
    # mixing up train, val, or test arrays 
    # standardizing twice by mistake

    # I found my mistake.
    # My mistake was that I set torch.manual_seed once in this function in the beginning but the called run_single_experiment twice, without
    # setting the seed again. so the second run may start from different random weights than the first. That means some of the difference 
    # may come from different initialization, not just scaling.
    # So I corrected this by setting torch.manual_seed inside the run_single_experiment function.
    # Now I get these results and they look better, scaling is helping:
    # No Scaling: train_loss: 0.4609763026237488, train_acc: 0.85597825050354, val_loss: 0.4751186668872833, val_acc: 0.828125
    # Scaling:
    # train_loss: 0.5341066122055054, train_acc: 0.8301630616188049, val_loss: 0.5197018384933472, val_acc: 0.86328125 





def check_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when both experiment results are present."""
    return "no_scaling_val_accuracy" in metrics and "scaled_val_accuracy" in metrics



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
    """Run the preprocessing scaffold one step at a time."""
    config = PreprocessingConfig()
    print("Running exercise 07 starter scaffold...")
    print("Main goal: compare no scaling and train-only standardization.")

    completed_steps = 0
    total_steps = 8


    raw_splits, ok = report_task_progress("load_raw_splits", load_raw_splits, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    model, ok = report_task_progress("SimpleVectorClassifier.__init__", SimpleVectorClassifier, config.num_features, config.hidden_dim, config.num_classes)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("SimpleVectorClassifier.forward", model, sample_x)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    standardizer, ok = report_task_progress("fit_standardizer", fit_standardizer, raw_splits["X_train"])
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("apply_standardizer", apply_standardizer, raw_splits["X_train"], standardizer)
    if ok:
        completed_steps += 1

    feature_arrays = {
        "train": raw_splits["X_train"],
        "val": raw_splits["X_val"],
        "test": raw_splits["X_test"],
    }
    label_arrays = {
        "train": raw_splits["y_train"],
        "val": raw_splits["y_val"],
        "test": raw_splits["y_test"],
    }
    _, ok = report_task_progress("build_dataloaders_from_arrays", build_dataloaders_from_arrays, feature_arrays, label_arrays, config.batch_size)
    if ok:
        completed_steps += 1

    _, ok = report_task_progress("run_single_experiment", run_single_experiment, config, False)
    if ok:
        completed_steps += 1

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "preprocessing comparison", float(int(passed)), "both experiment metrics available")


if __name__ == "__main__":
    main()











