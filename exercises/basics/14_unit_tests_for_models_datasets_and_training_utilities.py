"""
============================================================
EXERCISE 14: UNIT TESTS FOR MODELS DATASETS AND TRAINING UTILITIES
============================================================

WHAT YOU ARE BUILDING
You are building a small set of unit tests for ML code.

Plain-language meaning:
- a unit test is one small check with one clear expected answer
- instead of waiting for a full training run to fail,
  you test the small pieces early

WHY THIS EXERCISE EXISTS
Research code often changes quickly.
Small mistakes in shapes, labels, metrics, or save-load code can waste a lot of time.
Unit tests catch many of those mistakes before you launch a long run.
This exercise teaches:
- how to test dataset length and sample format
- how to test model output shapes
- how to test one training step
- how to test a metric helper on known values
- how to test checkpoint save-load behavior
- how to write small clear assertions

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises asked you to write training code, data code, and helper functions.
This exercise adds the next engineering habit:
- check those small pieces automatically
- before a long run wastes time
That matters because research engineering is not only about getting code to run once.
It is also about making code easier to trust when it changes.

IMPORTANT SCOPE NOTE
This exercise is mainly about testing.
Because of that, the tiny dataset, tiny model, and helper functions are already provided.
Your main job is to write tests around them.

Beginner intuition:
- think of unit tests like checking bike parts before a long ride
- if the wheel is loose, you want to catch that before the trip
- in ML code, a "loose wheel" might be:
  - wrong tensor shape
  - wrong label dtype
  - broken metric math
  - checkpoint load that does not restore the model

Small example:
- if your classifier should return shape `[8, 3]`
- a unit test can check that exact shape in a few milliseconds
- that is much faster than finding the mistake halfway through training

DATASET
Use the tiny local dataset already provided in this file.

HOW TO LOAD OR CREATE THE DATASET
The starter code below already creates a small dataset class.
You do not need downloads here.

MODEL
Use the tiny classifier already included in this file.
The learning goal is testing, not architecture design.

TEST TOOL NOTE
Real projects often use `pytest`.
For this exercise, start with plain `assert` statements inside test functions.
If you already know `pytest`, you can later move these tests into `tests/`.

INPUTS AND OUTPUTS
Input:
- a tiny dataset object
- a tiny model
- tiny batches
- tiny helper functions
Output:
- passing or failing test functions
- clear assertion messages
- a short summary of what was tested

TARGET RESULT RANGE
Focus on test clarity, not on model accuracy.

Useful signs:
- each test checks one clear thing
- expected shapes and dtypes are written directly in the test
- at least one test would fail if the code were broken
- the final summary says exactly which parts were checked

STEP-BY-STEP TODO
1. Read the provided tiny dataset, model, and helpers.
2. Write a test for dataset length.
3. Write a test for one sample's keys, shapes, and dtypes.
4. Write a test for the model output shape.
5. Write a test that loss is a scalar.
6. Write a test that one optimizer step changes parameters.
7. Write a test for the accuracy helper on a known toy example.
8. Write a test for checkpoint save-load round trip.
9. Run all tests and summarize what was checked.

DEFINITION OF DONE
You are done when:
- at least several useful tests are written
- each test checks one clear thing
- tests use simple `assert` statements
- the final summary says which parts were tested
- you can explain why tests save time in research work

COMMON MISTAKES
- writing one huge test that checks everything at once
- using random values without a fixed seed
- checking only that code runs, not that outputs make sense
- writing tests that depend on downloads or long training runs
- hiding the expected answer instead of stating it clearly

What each step is doing in simple words:
1. understand the small pieces before you test them
2. start with the easiest possible test so you learn the pattern
3. make sure one sample really looks the way later code expects
4. check the model contract: input shape in, output shape out
5. check that the loss is one number, because training expects one scalar loss
6. check that training actually changes parameters, otherwise learning cannot happen
7. test a helper on known numbers so the expected answer is obvious
8. test save and load because broken checkpoints waste real training time
9. return a short report so you can explain what your tests protect you from
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
class UnitTestingConfig:
    """Settings for the unit testing exercise."""
    num_samples: int = 24
    input_dim: int = 6
    hidden_dim: int = 16
    num_classes: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-2
    random_state: int = 42
    checkpoint_path: str = str(BOOTCAMP_ROOT / "outputs" / "checkpoints" / "exercise_36_test_model.pt")

def describe_testing_scope(config: UnitTestingConfig) -> dict[str, object]:
    """Return the main parts this exercise should test."""
    return {
        "objects_under_test": [
            "TinyClassificationDataset",
            "TinyClassifier",
            "accuracy_from_logits",
            "save_checkpoint/load_checkpoint",
        ],
        "suggested_test_file": "tests/test_exercise_36_ml_basics.py",
        "minimum_useful_tests": 6,
        "batch_size": config.batch_size,
    }

# Build one tiny batch used by several tests.
# Return:
# - `features` with shape `[batch_size, input_dim]`
# - `labels` with shape `[batch_size]`
def build_toy_batch(config: UnitTestingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one tiny deterministic batch for testing."""
    raise NotImplementedError("Return one small batch of features and labels for the tests.")


# Beginner hints for this file:
# - a good unit test should have one simple question
# - write the expected answer directly in the test
# - fixed seeds make tests easier to trust
# - tiny local data is usually better for tests than real big data
# - when a test fails, the failure message should tell you what went wrong quickly
class TinyClassificationDataset(Dataset):
    """Tiny dataset used as the object under test."""
    def __init__(self, config: UnitTestingConfig) -> None:
        generator = torch.Generator().manual_seed(config.random_state)
        self.features = torch.randn(config.num_samples, config.input_dim, generator=generator)
        self.labels = torch.randint(
            low=0,
            high=config.num_classes,
            size=(config.num_samples,),
            generator=generator,
        ).long()
    def __len__(self) -> int:
        return int(self.features.shape[0])
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "label": self.labels[index],
        }

class TinyClassifier(nn.Module):
    """Tiny classifier used as the object under test."""
    def __init__(self, config: UnitTestingConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return multiclass accuracy from raw logits."""
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())

def save_checkpoint(model: nn.Module, path: str) -> str:
    """Save the model state dict and return the path."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    return str(save_path)

def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    """Load a saved state dict into the given model and return it."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

# Test dataset length.
# A good first test is very small and very clear:
# - if you asked for `num_samples`, does the dataset really have that many items?
def test_dataset_length(config: UnitTestingConfig) -> None:
    """Assert that dataset length matches the config."""
    raise NotImplementedError("Write an assert that checks dataset length.")

# Test one dataset sample.
# Check at least:
# - keys are present
# - feature shape is `[input_dim]`
# - feature dtype is float
# - label dtype is `torch.long`
def test_dataset_sample_structure(config: UnitTestingConfig) -> None:
    """Assert that one dataset sample has the right structure."""
    raise NotImplementedError("Write asserts for sample keys, shapes, and dtypes.")

# Test the model forward pass.
# The most important check here is output shape:
# - input: `[batch_size, input_dim]`
# - output: `[batch_size, num_classes]`
def test_model_forward_shape(config: UnitTestingConfig) -> None:
    """Assert that model output shape is correct."""
    raise NotImplementedError("Write an assert that checks the model output shape.")

# Test that the loss is a scalar.
# In plain words:
# - one training step should produce one loss number, not a whole vector of losses
def test_loss_is_scalar(config: UnitTestingConfig) -> None:
    """Assert that classification loss is a scalar tensor."""
    raise NotImplementedError("Write an assert that checks the loss is scalar.")

# Test that one optimizer step really changes model parameters.
# This is a very useful sanity check.
# If parameters never change, training cannot learn.
def test_optimizer_step_changes_parameters(config: UnitTestingConfig) -> None:
    """Assert that one optimizer step updates at least one parameter tensor."""
    raise NotImplementedError("Write a test that checks at least one parameter changed after optimizer.step().")

# Test the metric helper on a tiny example with a known answer.
# Example idea:
# - choose logits where the best class is obvious
# - choose labels so you already know the exact accuracy
def test_accuracy_helper_on_known_values() -> None:
    """Assert that the accuracy helper returns the expected value."""
    raise NotImplementedError("Write a test using tiny known logits and labels.")

# Test save-load round trip.
# In plain words:
# - save a model
# - load into a fresh model
# - check that parameters now match
def test_checkpoint_round_trip(config: UnitTestingConfig) -> None:
    """Assert that save and load restore the same parameter values."""
    raise NotImplementedError("Write a checkpoint save-load test.")

# Run all test functions and return a short summary.
# A simple first version can:
# - call each test function
# - collect names of tests that passed
# - return counts and a short note
def run_all_tests(config: UnitTestingConfig) -> dict[str, object]:
    """Run all exercise tests and return a summary."""
    raise NotImplementedError("Call the test functions, collect pass results, and return a summary.")

def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when the test summary has the expected keys."""
    required = {"tests_written", "tests_passed", "tested_parts", "testing_note"}
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
    """Run the unit testing scaffold one step at a time."""
    config = UnitTestingConfig()
    set_seed(config.random_state)
    completed_steps = 0
    total_steps = 11

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False
    _, ok = report_task_progress("describe_testing_scope", describe_testing_scope, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("build_toy_batch", build_toy_batch, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("test_dataset_length", test_dataset_length, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("test_dataset_sample_structure", test_dataset_sample_structure, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("test_model_forward_shape", test_model_forward_shape, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("test_loss_is_scalar", test_loss_is_scalar, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("test_optimizer_step_changes_parameters", test_optimizer_step_changes_parameters, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("test_accuracy_helper_on_known_values", test_accuracy_helper_on_known_values)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("test_checkpoint_round_trip", test_checkpoint_round_trip, config)
    if not count_or_stop(ok):
        return

    sample_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    sample_labels = torch.tensor([0, 1])
    _, ok = report_task_progress("accuracy_from_logits", accuracy_from_logits, sample_logits, sample_labels)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_all_tests", run_all_tests, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(
            passed,
            "unit testing summary",
            float(int(passed)),
            "testing keys present",
        )
if __name__ == "__main__":
    main()
