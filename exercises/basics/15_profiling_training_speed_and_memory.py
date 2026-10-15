"""
============================================================
EXERCISE 15: PROFILING TRAINING SPEED AND MEMORY
============================================================

WHAT YOU ARE BUILDING
You are building a small profiling report for a training loop.

Plain-language meaning:
- profiling means measuring where time and memory go
- throughput means how much work you finish per second
- a bottleneck is the slow part that limits the whole run

WHY THIS EXERCISE EXISTS
Research engineers often need to make training faster and cheaper.
Before changing code, you should measure the problem.

This exercise teaches:
- batch time
- examples per second
- tokens or images per second
- dataloader time vs model time
- GPU memory if CUDA is available
- simple profiling notes

HOW THIS FITS INTO THE BOOTCAMP
Once you can train models correctly,
the next question is often:
- where is the time going?
- where is the memory going?

This file teaches that measurement habit.

What each step is doing in simple words:
1. build one simple training setup
2. time how long batches take to arrive
3. time how long train steps take
4. measure throughput and memory
5. compare two settings and explain the likely bottleneck

Beginner intuition:
- if training feels slow, do not guess first
- measure what is slow
- then decide whether the bottleneck is data loading, model compute, or memory

Small example:
- if batch fetch is slow but the model step is fast,
- the dataloader may be the bottleneck
- if batch fetch is fast but the model step is slow,
- the model compute may be the bottleneck

DATASET
Use a small real classification dataset.

Clear default:
- `sklearn.datasets.load_digits`
- input features: flattened `8 x 8` digit images
- input shape after flattening: `[batch_size, 64]`
- labels: digit class from `0` to `9`

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load `sklearn.datasets.load_digits`
- convert features and labels to tensors
- build a dataloader
- measure fetch time and train-step time separately
- compare two batch sizes

STEP-BY-STEP TODO
1. Build a small dataset and model.
2. Time dataloader batch fetching.
3. Time forward and backward pass.
4. Measure examples per second.
5. Measure peak GPU memory if available.
6. Compare two batch sizes.
7. Write a short bottleneck note.

DEFINITION OF DONE
You are done when:
- batch time is measured
- throughput is measured
- memory is measured or has a CPU fallback
- two settings are compared
- you can explain the main bottleneck

INPUTS AND OUTPUTS
Input:
- dataloader batches
- model

Output:
- fetch-time summary
- train-step summary
- throughput summary
- memory summary

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small MLP classifier.
The main goal is learning how to measure training behavior.

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
class ProfilingConfig:
    """Settings for speed and memory profiling."""

    num_samples: int = 2048
    num_features: int = 64
    num_classes: int = 10
    hidden_dim: int = 256
    batch_size: int = 32
    bigger_batch_size: int = 128
    num_profile_batches: int = 10
    learning_rate: float = 1e-3
    random_state: int = 42


def describe_profiling_plan(config: ProfilingConfig) -> dict[str, object]:
    """Return the profiling measurements this exercise should make."""
    return {
        "measurements": ["batch_fetch_time", "train_step_time", "examples_per_second", "peak_memory_mb"],
        "batch_sizes": [config.batch_size, config.bigger_batch_size],
        "device": device,
    }



# Build a dataloader for one chosen batch size.
# This lets you compare small and larger batches under the same setup.
def load_dataloader(config: ProfilingConfig, batch_size: int) -> DataLoader:
    """Return a dataloader for one batch size."""
    # Goal:
    # Return a dataloader with the requested batch size.
    #
    # Input:
    # - `config`: ProfilingConfig
    # - `batch_size`: number of examples in each batch
    #
    # Output:
    # - `DataLoader`
    # - Each batch returns `(features, labels)`.
    # - `features`: float tensor shaped `[batch_size, 64]`
    # - `labels`: long tensor shaped `[batch_size]`
    #
    # Small batch example:
    # - if `batch_size == 32`, `features.shape` should be `(32, 64)`
    # - `labels[0]` is one digit class from `0` to `9`
    #
    # Method:
    # Import the real dataset inside this function:
    # ```python
    # from sklearn.datasets import load_digits
    # ```
    #
    # Use:
    # ```python
    # digits = load_digits()
    # features = torch.tensor(digits.data, dtype=torch.float32)
    # labels = torch.tensor(digits.target, dtype=torch.long)
    # dataset = TensorDataset(features, labels)
    # return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # ```
    #
    # Meaning:
    # - `digits.data` has shape `[num_examples, 64]` because each digit image is `8 * 8`.
    # - `digits.target` has labels from `0` to `9`.
    # - `torch.long` is needed because `nn.CrossEntropyLoss` expects class ids.
    # - The `batch_size` argument is used so this exercise can compare two batch sizes.
    #
    # Steps:
    # 1. Load the digits dataset.
    # 2. Convert features to `torch.float32`.
    # 3. Convert labels to `torch.long`.
    # 4. If you use `config.num_samples`, keep the first `config.num_samples` examples.
    # 5. Build `TensorDataset(features, labels)`.
    # 6. Return a shuffled `DataLoader`.
    #
    # Checks:
    # - Do not use random tensors for the dataset.
    # - One batch should unpack as `features, labels = next(iter(loader))`.
    # - `features.shape[1]` should equal `config.num_features`.
    # - `labels.dtype` should be `torch.long`.
    raise NotImplementedError("Return a dataloader with the requested batch size.")




# Beginner hints for this file:
# - profiling starts with measurement, not guessing
# - split the timing into parts so you can see whether data loading or model compute is slower
# - throughput is often easier to compare than raw seconds alone
# - larger batch size may improve throughput, but it can also raise memory use
# - take a few batches, not only one, so the timing is less noisy

class ProfiledMLP(nn.Module):
    """Small model used for profiling."""

    def __init__(self, config: ProfilingConfig) -> None:
        # Goal:
        # Build a small MLP for profiling.
        #
        # Input:
        # - `config`: ProfilingConfig
        #
        # Output:
        # - no return value
        # - creates `self.net`
        #
        # Shape plan:
        # - input batch: `[batch_size, config.num_features]`
        # - hidden batch: `[batch_size, config.hidden_dim]`
        # - logits: `[batch_size, config.num_classes]`
        #
        # Method:
        # Use:
        # ```python
        # self.net = nn.Sequential(
        #     nn.Linear(config.num_features, config.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_dim, config.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_dim, config.num_classes),
        # )
        # ```
        #
        # Meaning:
        # - The first `Linear` reads the 64 digit features.
        # - The last `Linear` returns one logit for each digit class.
        # - Logits are raw scores. Do not apply softmax in the model.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Build `self.net`.
        # 3. Leave device movement to `build_training_objects`.
        #
        # Checks:
        # - Input shape `[8, 64]` should return logits shaped `[8, 10]`.
        # - Do not apply softmax before `nn.CrossEntropyLoss`.
        super().__init__()
        raise NotImplementedError("Build a small MLP for profiling.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits."""
        # Goal:
        # Return raw logits for each digit class.
        #
        # Input:
        # - `x`: float tensor shaped `[batch_size, config.num_features]`
        #
        # Output:
        # - tensor shaped `[batch_size, config.num_classes]`
        #
        # Method:
        # Use:
        # `return self.net(x)`
        #
        # Steps:
        # 1. Pass `x` through `self.net`.
        # 2. Return the logits.
        #
        # Checks:
        # - Return logits, not softmax probabilities.
        # - Batch size should stay the same.
        raise NotImplementedError("Return raw logits.")
# Build the model, loss function, and optimizer used for profiling.
# Keep this part simple so the measurements are easier to read.
def build_training_objects(config: ProfilingConfig) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, loss function, and optimizer."""
    # Goal:
    # Return the model, loss function, and optimizer used for profiling.
    #
    # Input:
    # - `config`: ProfilingConfig
    #
    # Output:
    # - `(model, loss_fn, optimizer)`
    # - `model`: ProfiledMLP moved to `device`
    # - `loss_fn`: `nn.CrossEntropyLoss()`
    # - `optimizer`: `torch.optim.AdamW`
    #
    # Method:
    # Use:
    # ```python
    # model = ProfiledMLP(config).to(device)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # ```
    #
    # Meaning:
    # - `CrossEntropyLoss` expects logits shaped `[batch_size, num_classes]`.
    # - It expects labels shaped `[batch_size]` with dtype `torch.long`.
    # - AdamW updates model parameters during training.
    #
    # Steps:
    # 1. Build the model.
    # 2. Move it to `device`.
    # 3. Build the loss function.
    # 4. Build the optimizer with `model.parameters()`.
    # 5. Return all three.
    #
    # Checks:
    # - `next(model.parameters()).device.type` should match `device`.
    # - The optimizer should receive the model parameters.
    raise NotImplementedError("Return model, CrossEntropyLoss, and optimizer.")


# Measure how long it takes to fetch batches from the dataloader.
# This is the data side only, before the model does any work.
def profile_batch_fetch(loader: DataLoader, num_batches: int) -> dict[str, float]:
    """Measure how long it takes to fetch batches."""
    # Goal:
    # Return mean dataloader fetch time.
    # This measures the data side only, before the model does any work.
    #
    # Input:
    # - `loader`: DataLoader
    # - `num_batches`: number of batches to time
    #
    # Output:
    # Return this exact dictionary:
    # - `"mean_fetch_seconds"`: average seconds to fetch one batch
    # - `"total_fetch_seconds"`: total seconds for all fetched batches
    # - `"batches_timed"`: number of batches actually timed, as a float
    #
    # Method:
    # Use:
    # ```python
    # start = time.perf_counter()
    # for batch_index, batch in enumerate(loader):
    #     if batch_index >= num_batches:
    #         break
    # elapsed = time.perf_counter() - start
    # ```
    #
    # Formula:
    # ```text
    # mean_fetch_seconds = total_fetch_seconds / batches_timed
    # ```
    #
    # Meaning:
    # - This timing includes Python/DataLoader work.
    # - It should not include forward pass, backward pass, or optimizer work.
    #
    # Steps:
    # 1. Start a timer.
    # 2. Iterate over the loader.
    # 3. Stop after `num_batches` batches.
    # 4. Compute total time.
    # 5. Compute mean time.
    # 6. Return the dictionary.
    #
    # Checks:
    # - `batches_timed` should be greater than `0`.
    # - If the loader has fewer batches than `num_batches`, use the number actually fetched.
    _ = time.perf_counter
    # This is only the data side of the pipeline.
    # If this part is slow, the model may spend time waiting for data.
    raise NotImplementedError("Return mean dataloader fetch time.")


def measure_peak_memory() -> float:
    """Return peak CUDA memory in MB, or 0.0 on CPU."""
    # Goal:
    # Measure peak GPU memory with a safe CPU fallback.
    #
    # Output:
    # - Python float in MB.
    # - Return `0.0` when CUDA is not available.
    #
    # Method:
    # Use:
    # ```python
    # if not torch.cuda.is_available():
    #     return 0.0
    # bytes_used = torch.cuda.max_memory_allocated()
    # return bytes_used / (1024 ** 2)
    # ```
    #
    # Meaning:
    # - CUDA reports memory in bytes.
    # - Dividing by `1024 ** 2` converts bytes to MB.
    #
    # Checks:
    # - CPU should return exactly `0.0`.
    # - GPU should return a non-negative float.
    raise NotImplementedError("Measure peak GPU memory, with a safe CPU fallback.")


# Measure the model side of training.
# Return values such as:
# - mean step time
# - throughput
# - peak memory
def profile_training_steps(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_batches: int,
) -> dict[str, float]:
    """Measure training step speed and memory."""
    # Goal:
    # Return mean step time, throughput, peak memory, and mean loss.
    #
    # Input:
    # - `model`: nn.Module
    # - `loader`: DataLoader
    # - `loss_fn`: nn.Module
    # - `optimizer`: torch.optim.Optimizer
    # - `num_batches`: number of training batches to profile
    #
    # Output:
    # Return this exact dictionary:
    # - `"mean_step_seconds"`: average seconds per training step
    # - `"examples_per_second"`: examples trained per second
    # - `"peak_memory_mb"`: peak CUDA memory in MB, or `0.0` on CPU
    # - `"mean_loss"`: average loss over profiled batches
    # - `"batches_timed"`: number of batches timed, as a float
    #
    # Method:
    # Use this training step:
    # ```python
    # features = features.to(device)
    # labels = labels.to(device)
    # logits = model(features)
    # loss = loss_fn(logits, labels)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # ```
    #
    # Use CUDA timing safety:
    # ```python
    # if torch.cuda.is_available():
    #     torch.cuda.reset_peak_memory_stats()
    # start = time.perf_counter()
    # ...
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    # elapsed = time.perf_counter() - start
    # ```
    #
    # Throughput formula:
    # ```text
    # examples_per_second = total_examples / total_step_seconds
    # ```
    #
    # Meaning:
    # - Step time includes forward pass, loss, backward pass, and optimizer step.
    # - It does not include the setup before the loop.
    #
    # Steps:
    # 1. Put the model in train mode.
    # 2. Reset CUDA peak memory stats if CUDA is available.
    # 3. Loop over up to `num_batches` batches.
    # 4. Move features and labels to `device`.
    # 5. Time each full training step.
    # 6. Add up elapsed seconds, examples, and loss values.
    # 7. Compute mean step time, throughput, memory, and mean loss.
    # 8. Return the dictionary.
    #
    # Checks:
    # - `batches_timed` should be greater than `0`.
    # - `examples_per_second` should be positive.
    # - `mean_loss` should be finite, not NaN.
    # - Return averaged metrics, not only the last batch's numbers.
    raise NotImplementedError("Return mean step time, throughput, and peak memory.")


# Compare profiling results for two batch sizes.
# This helps show the tradeoff between throughput and memory use.
def compare_batch_sizes(config: ProfilingConfig) -> dict[str, object]:
    """Profile two batch sizes and compare them."""
    # Goal:
    # Return a profiling comparison for two batch sizes.
    #
    # Input:
    # - `config`: ProfilingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"small_batch"`: profiling dictionary for `config.batch_size`
    # - `"large_batch"`: profiling dictionary for `config.bigger_batch_size`
    # - `"throughput_change"`: large throughput minus small throughput
    # - `"memory_change_mb"`: large peak memory minus small peak memory
    #
    # Method:
    # For each batch size:
    # ```python
    # loader = load_dataloader(config, batch_size)
    # model, loss_fn, optimizer = build_training_objects(config)
    # fetch = profile_batch_fetch(loader, config.num_profile_batches)
    # train = profile_training_steps(model, loader, loss_fn, optimizer, config.num_profile_batches)
    # ```
    #
    # Store each result like:
    # ```python
    # result = {
    #     "batch_size": float(batch_size),
    #     "fetch": fetch,
    #     "train": train,
    # }
    # ```
    #
    # Comparison formulas:
    # ```text
    # throughput_change = large_train_examples_per_second - small_train_examples_per_second
    # memory_change_mb = large_train_peak_memory_mb - small_train_peak_memory_mb
    # ```
    #
    # Steps:
    # 1. Profile `config.batch_size`.
    # 2. Profile `config.bigger_batch_size`.
    # 3. Compute throughput change.
    # 4. Compute memory change.
    # 5. Return the dictionary above.
    #
    # Checks:
    # - Both nested train dictionaries should include `"examples_per_second"`.
    # - Memory values are MB.
    # - Time values are seconds.
    raise NotImplementedError("Return a profiling comparison for two batch sizes.")


def write_bottleneck_note(profile_summary: dict[str, object]) -> str:
    """Write a short note about the likely bottleneck."""
    # Goal:
    # Explain the slowest part in plain words.
    #
    # Input:
    # - `profile_summary`: dict[str, object]
    # - This should be the output from `compare_batch_sizes(config)`.
    #
    # Output:
    # - short string written in plain English
    #
    # Method:
    # Read:
    # - small batch fetch time from `profile_summary["small_batch"]["fetch"]["mean_fetch_seconds"]`
    # - small batch step time from `profile_summary["small_batch"]["train"]["mean_step_seconds"]`
    # - large batch throughput from `profile_summary["large_batch"]["train"]["examples_per_second"]`
    #
    # Simple rule:
    # - If fetch time is larger than train step time, say data loading looks slower.
    # - Otherwise, say model training looks slower.
    #
    # Include one concrete comparison number when possible.
    #
    # Steps:
    # 1. Read the nested fetch and train metrics.
    # 2. Compare fetch time with train step time.
    # 3. Mention whether larger batch size improved throughput.
    # 4. Return one short note.
    #
    # Checks:
    # - Keep the note short and plain.
    # - Say "looks like" rather than making a hard claim from a tiny profile.
    raise NotImplementedError("Explain the slowest part in plain words.")


def run_profiling(config: ProfilingConfig) -> dict[str, object]:
    """Run the full profiling workflow."""
    # Goal:
    # Run profiling and return a summary.
    #
    # Input:
    # - `config`: ProfilingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"batch_size_results"`: output from `compare_batch_sizes(config)`
    # - `"bottleneck_note"`: output from `write_bottleneck_note(...)`
    # - `"device"`: `"cuda"` or `"cpu"`
    #
    # Method:
    # Use:
    # ```python
    # set_seed(config.random_state)
    # batch_size_results = compare_batch_sizes(config)
    # bottleneck_note = write_bottleneck_note(batch_size_results)
    # return {
    #     "batch_size_results": batch_size_results,
    #     "bottleneck_note": bottleneck_note,
    #     "device": device,
    # }
    # ```
    #
    # Steps:
    # 1. Set the seed.
    # 2. Compare the two batch sizes.
    # 3. Write the bottleneck note.
    # 4. Return the summary dictionary.
    #
    # Checks:
    # - The returned dictionary must include `"batch_size_results"` and `"bottleneck_note"`.
    # - The note should be a string.
    raise NotImplementedError("Run profiling and return a summary.")


def check_metrics(metrics: dict[str, object]) -> bool:
    """Return True when profiling summary keys are present."""
    required = {"batch_size_results", "bottleneck_note"}
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
    """Run the profiling scaffold."""
    config = ProfilingConfig()
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

    loader, ok = report_task_progress("load_dataloader", load_dataloader, config, config.batch_size)
    if not count_or_stop(ok):
        return


    _, ok = report_task_progress("describe_profiling_plan", describe_profiling_plan, config)
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("ProfiledMLP.__init__", ProfiledMLP, config)
    if not count_or_stop(ok):
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("ProfiledMLP.forward", model, sample_x)
    if not count_or_stop(ok):
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return

    training_model, loss_fn, optimizer = training_objects
    _, ok = report_task_progress("profile_batch_fetch", profile_batch_fetch, loader, config.num_profile_batches)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress(
        "profile_training_steps",
        profile_training_steps,
        training_model,
        loader,
        loss_fn,
        optimizer,
        config.num_profile_batches,
    )
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_profiling", run_profiling, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "profiling summary", float(int(passed)), "profiling keys present")

if __name__ == "__main__":
    main()
