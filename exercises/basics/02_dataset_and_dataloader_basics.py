"""
============================================================
EXERCISE 02: DATASET AND DATALOADER BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a small data pipeline for vector classification.

You will create synthetic data with scikit-learn, split it into train, validation, and test sets using a 60/20/20 split, convert arrays into tensors, wrap them in datasets, and iterate through batches with dataloaders.

WHY THIS EXERCISE EXISTS
A model is only one part of training.

You also need a reliable way to move data into batches.

This exercise helps you understand the path from NumPy arrays to PyTorch batches.

DATASET
Use `sklearn.datasets.make_classification`.

This creates synthetic numeric data for classification.

HOW TO LOAD OR DOWNLOAD THE DATASET
You do not need to download anything.

The dataset is created in memory when the file runs.

INPUTS AND OUTPUTS
Inputs:
- numeric feature vectors
- integer labels

Outputs:
- train, validation, and test splits
- `TensorDataset` objects
- one custom `Dataset`
- `DataLoader` batches with expected shapes

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
There is no real model in this exercise.

The focus is the data pipeline.

LOSS FUNCTION
There is no training loss in this exercise.

METRICS TO TRACK
Track these checks:
- split sizes make sense
- feature tensors have shape `[batch_size, num_features]`
- label tensors have shape `[batch_size]`
- train loader uses shuffle
- validation and test loaders usually do not shuffle

TARGET RESULT RANGE
Expected batch shapes depend on your config.

A common example is:
- features: `[32, num_features]`
- labels: `[32]`

COMMON MISTAKES
- forgetting to convert NumPy arrays to tensors
- using float labels for multiclass classification
- making train, validation, and test splits overlap
- shuffling validation or test data without a reason
- returning one sample with the wrong shape from a custom dataset

STEP-BY-STEP TODO
1. Make a synthetic dataset.
2. Split it into train, validation, and test parts with a 60/20/20 split.
3. Convert arrays into PyTorch tensors.
4. Build a `TensorDataset`.
5. Build a custom `Dataset` class.
6. Create dataloaders.
7. Inspect one batch and check its shapes.
8. Compare shuffle and no-shuffle behavior.

STRETCH GOALS
- print class counts in each split
- try a different batch size
- compare `drop_last=True` and `drop_last=False`

DEFINITION OF DONE
You are done when:
- your loaders produce the shapes you expect
- you can explain what a dataset does
- you can explain what a dataloader does
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data_utils import (
    numpy_features_to_tensor,
    numpy_labels_to_long_tensor,
    train_val_test_split,
)

@dataclass
class DataLoaderBasicsConfig:
    """Settings for the synthetic dataset and loaders."""

    num_samples: int = 512
    num_features: int = 12
    num_classes: int = 2
    batch_size: int = 32
    random_state: int = 42



# Build the NumPy data first.
# Return a dict with these keys:
# - "X_train", "X_val", "X_test"
# - "y_train", "y_val", "y_test"
# Use make_classification(...) and then train_val_test_split(...).
# Use a 60% train / 20% val / 20% test split.
# Keep this step in NumPy only.
def load_numpy_splits(config: DataLoaderBasicsConfig) -> dict[str, object]:
    raise NotImplementedError("Return the NumPy train, validation, and test splits.")








# Convert the split arrays into PyTorch tensors.
# Return the same six keys as load_numpy_splits(...), but now each value is a tensor.
# Features should be float tensors. Labels should be integer class tensors.
def convert_splits_to_tensors(split_dict: dict[str, object]) -> dict[str, torch.Tensor]:
    raise NotImplementedError("Convert every split array to the correct PyTorch tensor type.")



# Wrap each tensor split in TensorDataset.
# Return a dict with exactly these keys:
# - "train"
# - "val"
# - "test"
def build_tensor_datasets(tensor_dict: dict[str, torch.Tensor]) -> dict[str, TensorDataset]:
    raise NotImplementedError("Return train, validation, and test TensorDataset objects.")



class VectorClassificationDataset(Dataset):
    """Custom dataset stub for vector classification data."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        raise NotImplementedError("Store the feature and label tensors.")

    def __len__(self) -> int:
        """Return the number of examples."""
        raise NotImplementedError("Return the number of examples.")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one example.

        Return:
        - the feature row at this index
        - the label from the same row

        Smallest correct answer:
        return self.features[index], self.labels[index]
        """
        raise NotImplementedError("Return one feature row and its matching label.")



# Wrap each tensor split in your custom dataset class.
# Return the same three keys: "train", "val", and "test".
def build_custom_datasets(tensor_dict: dict[str, torch.Tensor]) -> dict[str, VectorClassificationDataset]:
    raise NotImplementedError("Return train, validation, and test custom datasets.")





# Build the DataLoader objects.
# Return a dict with keys "train", "val", and "test".
# Use shuffle=True for train.
# Use shuffle=False for val and test.
def build_dataloaders(config: DataLoaderBasicsConfig, datasets: dict[str, Dataset]) -> dict[str, DataLoader]:
    raise NotImplementedError("Return the train, validation, and test dataloaders.")




# Look at one batch from the train loader.
# Return a small summary dict with at least these keys:
# - "feature_batch_shape"
# - "label_batch_shape"
# Store the shapes as tuples.
def inspect_one_batch(loaders: dict[str, DataLoader]) -> dict[str, object]:
    raise NotImplementedError("Inspect one training batch and return its shapes.")





# Connect the whole data pipeline.
# Smallest correct path:
# 1. load NumPy splits
# 2. convert them to tensors
# 3. build datasets
# 4. build loaders
# 5. inspect one train batch
# Return the summary from inspect_one_batch(...).
def run_training_stub(config: DataLoaderBasicsConfig) -> dict[str, object]:    
    raise NotImplementedError("Run the full data pipeline and return the batch summary.")


# Keep this simple.
# Smallest correct answer: return summary unchanged.
def evaluate_stub(summary: dict[str, object]) -> dict[str, object]:
    raise NotImplementedError("Return the evaluation summary.")



def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when the batch shapes look correct."""
    feature_shape = metrics.get("feature_batch_shape")
    label_shape = metrics.get("label_batch_shape")
    if feature_shape is None or label_shape is None:
        return False
    return len(feature_shape) == 2 and len(label_shape) == 1



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
    """Run the dataloader scaffold one step at a time."""
    config = DataLoaderBasicsConfig()
    print("Running exercise 02 starter scaffold...")
    print(f"Expected feature batch shape example: [{config.batch_size}, {config.num_features}]")
    print(f"Expected label batch shape example: [{config.batch_size}]")

    completed_steps = 0
    total_steps = 8

    numpy_splits, ok = report_task_progress("load_numpy_splits", load_numpy_splits, config)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    tensor_splits, ok = report_task_progress("convert_splits_to_tensors", convert_splits_to_tensors, numpy_splits)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    _, ok = report_task_progress("build_tensor_datasets", build_tensor_datasets, tensor_splits)
    if ok:
        completed_steps += 1

    custom_datasets, ok = report_task_progress("build_custom_datasets", build_custom_datasets, tensor_splits)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    loaders, ok = report_task_progress("build_dataloaders", build_dataloaders, config, custom_datasets)
    if ok:
        completed_steps += 1
    else:
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return

    summary, ok = report_task_progress("inspect_one_batch", inspect_one_batch, loaders)
    if ok:
        completed_steps += 1

    pipeline_summary, ok = report_task_progress("run_training_stub", run_training_stub, config)
    if ok:
        completed_steps += 1

    if pipeline_summary is not None:
        final_summary, ok = report_task_progress("evaluate_stub", evaluate_stub, pipeline_summary)
        if ok:
            completed_steps += 1
        else:
            final_summary = pipeline_summary
    else:
        final_summary = summary

    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if final_summary is not None:
        passed = check_metrics(final_summary)
        print(f"Batch shape check: {'PASS' if passed else 'TRY_AGAIN'}")


if __name__ == "__main__":
    main()



