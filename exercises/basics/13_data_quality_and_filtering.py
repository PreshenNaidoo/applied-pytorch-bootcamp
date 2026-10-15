"""
============================================================
EXERCISE 13: DATA QUALITY AND FILTERING
============================================================

WHAT YOU ARE BUILDING
You are building a small dataset quality checker.

Plain-language meaning:
- data quality means how useful and trustworthy the data is
- duplicate examples can make validation look better than it really is
- bad labels teach the model the wrong thing
- leakage means training data appears in validation or test data

WHY THIS EXERCISE EXISTS
Models learn from the data you give them.
Bad data can make a good model look bad.
Bad data can also make a weak model look better than it really is.
This exercise teaches:
- class counts
- duplicate checks
- train/val/test leakage checks
- empty or very short text checks
- suspicious label checks
- dataset reports
- before-and-after filtering comparison

HOW THIS FITS INTO THE BOOTCAMP
Many earlier files focused on models and training loops.
This file teaches a different research-engineering habit:
- inspect the data first
- because a clean model cannot fully rescue broken data

What each step is doing in simple words:
1. create a tiny dataset with real problems on purpose
2. count what is in each split and class
3. find duplicates and leakage
4. find empty, short, or suspicious examples
5. clean the dataset carefully
6. compare before and after so the filtering is explainable

Beginner intuition:
- models only learn from the data you give them
- if the data is messy, the model may learn messy patterns too
- if validation data leaks into training, evaluation stops being trustworthy

Small example:
- if the same text appears in both train and validation,
- the model may look better than it really is
- that is not real generalization; it is just repeated data

DATASET
Use a tiny local text classification dataset written inside this file.

Clear default:
- task: support-message classification
- valid labels: `billing`, `technical`, `account`
- examples are short customer-support messages

The dataset should intentionally include:
- exact duplicate texts
- empty or very short texts
- suspicious labels
- overlap between train and validation or test

HOW TO LOAD OR CREATE THE DATASET

Simple path:
- write a tiny local dataset inside the file
- insert a few clear problems on purpose
- split the dataset
- then run the quality checks and filtering code

COMMON MISTAKES
- checking model results before checking data
- deduplicating each split separately but missing train/val overlap
- deleting data without writing down what changed
- using validation data while cleaning training data in a way that leaks information
- trusting a metric without reading examples

INPUTS AND OUTPUTS
Input:
- text
- label
- split name
Output:
- duplicate list
- leakage list
- suspicious-example list
- filtered dataset
- dataset quality report

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
No complex model is required.
The main goal is to inspect the data itself first.

LOSS FUNCTION
If you compare model results before and after cleaning, use a simple classification loss such as `nn.CrossEntropyLoss`.

METRICS TO TRACK
Track at least:
- examples per split
- examples per class
- duplicate count
- leakage count
- filtered example count

TARGET RESULT RANGE
Focus on whether the checks catch the planted problems.

Useful signs:
- duplicates are found
- overlap across splits is found
- the report explains why these problems would mislead evaluation

STEP-BY-STEP TODO
1. Build a small noisy dataset.
2. Count examples per split.
3. Count examples per class.
4. Find exact duplicate texts.
5. Find train/val/test overlap.
6. Flag empty or very short examples.
7. Flag suspicious labels.
8. Filter the bad examples.
9. Compare dataset size before and after filtering.
10. Write a dataset quality report.

DEFINITION OF DONE
You are done when:
- duplicate texts are found
- split leakage is found
- class counts are reported
- bad examples are filtered
- the report explains why bad data can mislead evaluation
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys

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

@dataclass
class DataQualityConfig:
    """Settings for data quality and filtering."""
    min_text_length: int = 5
    valid_labels: tuple[str, ...] = ("world", "sports", "business", "tech")
    random_state: int = 42
    report_path: str = str(BOOTCAMP_ROOT / "reports" / "35_data_quality_and_filtering.md")

# Beginner hints for this file:
# - bad data can make a model look better or worse for the wrong reason
# - duplicates matter because the model may effectively see the same example many times
# - leakage matters because validation should feel new to the model
# - start with exact string matches for duplicates and leakage before trying anything more clever
# - write down what you removed, otherwise the cleaning step becomes hard to trust later
Example = dict[str, str]
DatasetSplits = dict[str, list[Example]]

def describe_quality_checks(config: DataQualityConfig) -> dict[str, object]:
    """Return the checks this exercise should run."""
    return {
        "checks": [
            "split sizes",
            "class counts",
            "exact duplicates",
            "train/val/test leakage",
            "short or empty text",
            "invalid labels",
        ],
        "min_text_length": config.min_text_length,
        "valid_labels": list(config.valid_labels),
        "report_path": config.report_path,
    }

# Return a tiny dataset with intentional problems.
# Include examples of:
# - duplicates
# - leakage across splits
# - short or empty text
# - invalid labels
def build_noisy_dataset() -> DatasetSplits:
    """Return train, val, and test splits with intentional data problems."""
    raise NotImplementedError("Return a small noisy dataset with duplicates, leakage, and bad labels.")

def count_split_sizes(dataset: DatasetSplits) -> dict[str, int]:
    """Return number of examples in each split."""
    raise NotImplementedError("Count examples in train, val, and test.")

def count_labels(dataset: DatasetSplits, valid_labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Return label counts for each split."""
    raise NotImplementedError("Count labels per split.")

# Find exact duplicate texts inside each split.
# Start with exact string matches first.
# That is the simplest and clearest beginner check.
def find_duplicate_texts(dataset: DatasetSplits) -> dict[str, list[str]]:
    """Find exact duplicate texts inside each split."""
    raise NotImplementedError("Return duplicate texts found inside each split.")

# Find exact text overlap across train, validation, and test.
# This is one of the most important trust checks in the file.
def find_split_leakage(dataset: DatasetSplits) -> dict[str, list[str]]:
    """Find exact text overlap across train, val, and test."""
    # Leakage is especially important because it can make validation look falsely good.
    # If the model sees the same text in training and validation,
    # validation no longer feels like truly new data.
    raise NotImplementedError("Return texts that appear in more than one split.")

def find_short_or_empty_examples(dataset: DatasetSplits, min_text_length: int) -> dict[str, list[Example]]:
    """Find examples whose text is too short."""
    raise NotImplementedError("Return empty or very short examples.")

def find_invalid_label_examples(dataset: DatasetSplits, valid_labels: tuple[str, ...]) -> dict[str, list[Example]]:
    """Find examples with labels outside the valid label set."""
    raise NotImplementedError("Return examples with invalid labels.")

# Return a cleaned copy of the dataset.
# Keep the filtering rules simple and explicit.
# A beginner should be able to say exactly why each removed example was removed.
def filter_dataset(dataset: DatasetSplits, config: DataQualityConfig) -> DatasetSplits:
    """Return a cleaned copy of the dataset."""
    # Keep the filtering rules simple and explicit.
    # A good beginner report should be able to say exactly why each removed example was removed.
    raise NotImplementedError("Remove or fix duplicates, leakage, short texts, and invalid labels.")

def compare_before_after_filtering(before: DatasetSplits, after: DatasetSplits) -> dict[str, object]:
    """Return a summary of what changed after filtering."""
    raise NotImplementedError("Compare dataset sizes and removed examples before and after filtering.")

def write_quality_report(config: DataQualityConfig, summary: dict[str, object]) -> str:
    """Write a markdown dataset quality report and save it."""
    # A useful report should answer:
    # - what problems were found?
    # - how many examples were affected?
    # - what was removed or changed?
    # - why does that matter for trustworthy evaluation?
    raise NotImplementedError("Write and save a plain-language data quality report.")

def run_data_quality_workflow(config: DataQualityConfig) -> dict[str, object]:
    """Run all checks, filter the dataset, and write a report."""
    raise NotImplementedError("Run the full data quality workflow.")

def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when the data quality workflow returns the main outputs."""
    required = {"duplicates_found", "leakage_found", "num_removed", "report_path"}
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
    """Run the data quality scaffold one step at a time."""
    config = DataQualityConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 7
    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    dataset, ok = report_task_progress("build_noisy_dataset", build_noisy_dataset)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("describe_quality_checks", describe_quality_checks, config)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("count_split_sizes", count_split_sizes, dataset)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("count_labels", count_labels, dataset, config.valid_labels)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("find_duplicate_texts", find_duplicate_texts, dataset)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("find_split_leakage", find_split_leakage, dataset)
    if not count_or_stop(ok):
        return
    metrics, ok = report_task_progress("run_data_quality_workflow", run_data_quality_workflow, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "data quality summary", float(int(passed)), "main report keys present")
if __name__ == "__main__":
    main()
