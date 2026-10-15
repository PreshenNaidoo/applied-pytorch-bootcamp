"""
============================================================
EXERCISE 23: GRADIENT CHECKPOINTING MEMORY TRADEOFFS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny experiment that compares normal training with gradient checkpointing.

Plain-language meaning:
- normal training saves many hidden values so backward pass can use them later
- gradient checkpointing saves fewer hidden values
- during backward pass, it recomputes some values
- this can save memory, but it usually makes training slower

WHY THIS EXERCISE EXISTS
Large models often do not fit in memory with normal training.
Gradient checkpointing is one common way to train bigger models on limited GPU memory.

This exercise teaches:
- what activation memory means
- how to turn checkpointing on for part of a model
- how to compare memory and speed
- why saving memory can cost extra compute

HOW THIS FITS INTO THE BOOTCAMP
After training stability and larger-model ideas,
this file teaches one of the main memory tradeoffs used when models become deeper or batches become larger.

What each step is doing in simple words:
1. build a deeper model than earlier toy examples
2. measure memory without checkpointing
3. run the same kind of work with checkpointing
4. compare memory saved against extra compute time
5. explain the tradeoff in plain words

Beginner intuition:
- normal training tries to remember many intermediate values from the forward pass
- checkpointing remembers fewer of them
- later, backward pass recomputes some missing pieces when needed

Small example:
- without checkpointing:
  - save many forward-pass activations
  - backward pass can reuse them directly
- with checkpointing:
  - save fewer activations
  - backward pass recomputes some of the missing work

DATASET
Use a small real dataset.

Clear default:
- `sklearn.datasets.load_digits`
- input features: flattened `8 x 8` digit images
- input shape after flattening: `[batch_size, 64]`
- labels: digit class from `0` to `9`

HOW TO CREATE THE INPUTS
Simple path:
- build a deeper toy network
- load one small real batch from `sklearn.datasets.load_digits`
- run one training step without checkpointing
- run one training step with checkpointing
- compare memory and time

STEP-BY-STEP TODO
1. Build a deep MLP or tiny transformer-like stack.
2. Train one batch without checkpointing.
3. Train one batch with checkpointing.
4. Measure peak memory if CUDA is available.
5. Measure time for both paths.
6. Check that both paths produce a valid loss.
7. Write a short note about the tradeoff.

DEFINITION OF DONE
You are done when:
- normal and checkpointed training both run
- memory or a clear CPU fallback estimate is reported
- speed is reported
- you can explain memory saved vs extra compute

INPUTS AND OUTPUTS
Input:
- real digit feature batch from `sklearn.datasets.load_digits`
- real digit labels from `0` to `9`

Output:
- normal loss
- checkpointed loss
- memory summary
- timing summary

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a deeper MLP or transformer-like stack.
The model should be deep enough that saved activations matter.

LOSS FUNCTION
Use a simple supervised loss such as `nn.CrossEntropyLoss`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
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
class GradientCheckpointingConfig:
    """Settings for gradient checkpointing practice."""

    num_samples: int = 512
    input_dim: int = 64
    hidden_dim: int = 256
    num_classes: int = 10
    num_blocks: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    random_state: int = 42


def describe_checkpointing_tradeoff(config: GradientCheckpointingConfig) -> dict[str, object]:
    """Return the main comparison this exercise should make."""
    return {
        "normal_training": "saves more activations",
        "checkpointed_training": "saves fewer activations and recomputes some work",
        "num_blocks": config.num_blocks,
        "device": device,
    }



# Build one simple dataloader for the comparison.
# Keep the task small.
# The main lesson is memory vs compute, not dataset complexity.
def load_dataloader(config: GradientCheckpointingConfig) -> DataLoader:
    """Return one simple dataloader."""
    # Goal:
    # Return a dataloader for a small real classification task.
    #
    # Input:
    # - `config`: GradientCheckpointingConfig
    #
    # Output:
    # - `DataLoader`
    # - Each batch returns `(features, labels)`.
    # - `features`: float tensor shaped `[batch_size, 64]`
    # - `labels`: long tensor shaped `[batch_size]`
    #
    # Small batch example:
    # - if `batch_size == 32`, `features.shape` should be `(32, 64)`
    # - `labels[0]` is one digit class such as `0`, `1`, ..., `9`
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
    # return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # ```
    #
    # Meaning:
    # - `digits.data` has shape `[num_examples, 64]` because each digit image is `8 * 8`.
    # - `digits.target` has labels from `0` to `9`.
    # - `torch.long` is needed because `nn.CrossEntropyLoss` expects class ids.
    #
    # Steps:
    # 1. Load the digits dataset.
    # 2. Convert features to `torch.float32`.
    # 3. Convert labels to `torch.long`.
    # 4. Build a `TensorDataset(features, labels)`.
    # 5. Return a shuffled `DataLoader`.
    #
    # Checks:
    # - Do not use random tensors for the dataset.
    # - One batch should unpack as `features, labels = next(iter(loader))`.
    # - `features.shape[1]` should equal `config.input_dim`.
    # - `labels.dtype` should be `torch.long`.
    raise NotImplementedError("Return a dataloader for a small real classification task.")




# Beginner hints for this file:
# - do not confuse gradient checkpointing with saving model checkpoints to disk
# - here, "checkpointing" means save less intermediate memory during forward pass
# - then backward pass recomputes some work instead of reading all saved activations
# - this is mainly a memory-saving trick, not a quality-improving trick
# - compare one clean small batch first, then scale up if needed

class DeepToyModel(nn.Module):
    """Deep toy model where checkpointing can be tested."""

    def __init__(self, config: GradientCheckpointingConfig, use_checkpointing: bool) -> None:
        # Goal:
        # Build a deep stack of simple blocks and a classifier head.
        # `use_checkpointing` only changes the forward pass, not the layers.
        #
        # Input:
        # - `config`: GradientCheckpointingConfig
        # - `use_checkpointing`: bool
        #
        # Output:
        # - no return value
        # - creates:
        #   - `self.input_layer`
        #   - `self.blocks`
        #   - `self.classifier`
        #
        # Shape plan:
        # - input batch: `[batch_size, input_dim]`
        # - hidden batch after `input_layer`: `[batch_size, hidden_dim]`
        # - hidden batch after each block: `[batch_size, hidden_dim]`
        # - logits: `[batch_size, num_classes]`
        #
        # Method:
        # Use:
        # ```python
        # self.input_layer = nn.Sequential(
        #     nn.Linear(config.input_dim, config.hidden_dim),
        #     nn.ReLU(),
        # )
        # self.blocks = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(config.hidden_dim, config.hidden_dim),
        #             nn.ReLU(),
        #         )
        #         for _ in range(config.num_blocks)
        #     ]
        # )
        # self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        # ```
        #
        # Meaning:
        # - `nn.ModuleList` tells PyTorch that the blocks contain trainable parameters.
        # - `num_blocks` controls how deep the model is.
        # - A deeper model saves more activations during normal training.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Store `config`.
        # 3. Store `use_checkpointing`.
        # 4. Build `self.input_layer`.
        # 5. Build `self.blocks`.
        # 6. Build `self.classifier`.
        #
        # Checks:
        # - `len(self.blocks)` should equal `config.num_blocks`.
        # - Do not apply softmax in the model. Cross-entropy expects raw logits.
        #
        super().__init__()
        self.config = config
        self.use_checkpointing = use_checkpointing
        raise NotImplementedError("Build a deep stack of simple blocks and a classifier head.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits."""
        # Goal:
        # Run the deep model.
        # If checkpointing is enabled, run each block through PyTorch checkpointing.
        #
        # Input:
        # - `x`: float tensor shaped `[batch_size, config.input_dim]`
        #
        # Output:
        # - raw class logits shaped `[batch_size, config.num_classes]`
        #
        # Method:
        # Use normal block call when checkpointing is off:
        # ```python
        # x = block(x)
        # ```
        #
        # Use this exact checkpoint call when checkpointing is on:
        # ```python
        # x = checkpoint(block, x, use_reentrant=False)
        # ```
        #
        # Meaning:
        # - `checkpoint` is imported from `torch.utils.checkpoint`.
        # - `use_reentrant=False` is the current recommended PyTorch path for most new code.
        # - Checkpointing saves fewer forward activations.
        # - During backward, PyTorch recomputes the block output.
        #
        # Steps:
        # 1. Pass `x` through `self.input_layer`.
        # 2. Loop over each block in `self.blocks`.
        # 3. If `self.use_checkpointing` is true, call `checkpoint(block, x, use_reentrant=False)`.
        # 4. Otherwise, call `block(x)`.
        # 5. Pass the final hidden tensor through `self.classifier`.
        # 6. Return logits.
        #
        # Checks:
        # - For input shape `[8, 64]`, output shape should be `[8, 10]`.
        # - Return logits, not softmax probabilities.
        raise NotImplementedError("Run the deep model, with checkpointing if enabled.")
# Build the model, loss function, and optimizer for one comparison path.
# `use_checkpointing` decides whether the model should save less activation memory.
def build_training_objects(
    config: GradientCheckpointingConfig,
    use_checkpointing: bool,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, loss function, and optimizer."""
    # Goal:
    # Return the model, loss function, and optimizer for one comparison path.
    #
    # Input:
    # - `config`: GradientCheckpointingConfig
    # - `use_checkpointing`: bool
    #
    # Output:
    # - `(model, loss_fn, optimizer)`
    # - `model`: DeepToyModel moved to `device`
    # - `loss_fn`: `nn.CrossEntropyLoss()`
    # - `optimizer`: `torch.optim.AdamW`
    #
    # Method:
    # Use:
    # ```python
    # model = DeepToyModel(config, use_checkpointing=use_checkpointing).to(device)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # ```
    #
    # Meaning:
    # - `CrossEntropyLoss` expects logits shaped `[batch_size, num_classes]`.
    # - It also expects labels shaped `[batch_size]` with dtype `torch.long`.
    #
    # Steps:
    # 1. Build the model.
    # 2. Move it to `device`.
    # 3. Build the loss.
    # 4. Build the optimizer.
    # 5. Return all three objects.
    #
    # Checks:
    # - `next(model.parameters()).device.type` should match `device`.
    # - The optimizer should receive `model.parameters()`.
    raise NotImplementedError("Return model, CrossEntropyLoss, and optimizer.")


# Return peak memory in MB.
# On GPU this should use CUDA memory stats.
# On CPU you can return a simple safe fallback such as `0.0`.
def measure_peak_memory() -> float:
    """Return peak CUDA memory in MB, or 0.0 on CPU."""
    # Goal:
    # Measure peak GPU memory, with a safe CPU fallback.
    #
    # Input:
    # - no direct arguments
    #
    # Output:
    # - Python float in megabytes, also called MB
    # - return `0.0` on CPU
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
    # Steps:
    # 1. If CUDA is not available, return `0.0`.
    # 2. Read `torch.cuda.max_memory_allocated()`.
    # 3. Convert bytes to MB.
    # 4. Return the float.
    #
    # Checks:
    # - CPU should return exactly `0.0`.
    # - GPU should return a non-negative float.
    raise NotImplementedError("Measure peak GPU memory, with a safe CPU fallback.")


# Train on exactly one batch and measure the result.
# Return:
# - loss
# - time
# - memory
# One clean batch is enough to show the checkpointing tradeoff.
def train_one_batch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Train on one batch and return loss, time, and memory."""
    # Goal:
    # Train one batch and return loss, seconds, and peak memory.
    #
    # Input:
    # - `model`: nn.Module
    # - `loader`: DataLoader
    # - `loss_fn`: nn.Module
    # - `optimizer`: torch.optim.Optimizer
    #
    # Output:
    # Return this exact dictionary:
    # - `"loss"`: Python float
    # - `"seconds"`: Python float, time for one train step
    # - `"peak_memory_mb"`: Python float, peak CUDA memory in MB or `0.0` on CPU
    #
    # Method:
    # Use:
    # ```python
    # model.train()
    # features, labels = next(iter(loader))
    # features = features.to(device)
    # labels = labels.to(device)
    # if torch.cuda.is_available():
    #     torch.cuda.reset_peak_memory_stats()
    # start = time.perf_counter()
    # logits = model(features)
    # loss = loss_fn(logits, labels)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    # seconds = time.perf_counter() - start
    # ```
    #
    # Meaning:
    # - `reset_peak_memory_stats` starts the memory measurement from this step.
    # - `torch.cuda.synchronize()` makes timing wait for GPU work to finish.
    # - On CPU, there is no CUDA memory to report, so memory is `0.0`.
    #
    # Steps:
    # 1. Put the model in train mode.
    # 2. Get one batch from the loader.
    # 3. Move features and labels to `device`.
    # 4. Reset CUDA peak memory stats if CUDA is available.
    # 5. Time one forward pass, loss, backward pass, and optimizer step.
    # 6. Measure peak memory with `measure_peak_memory()`.
    # 7. Return the dictionary.
    #
    # Checks:
    # - `logits.shape` should be `[batch_size, config.num_classes]`.
    # - `loss` should be finite, not NaN.
    # - Returned keys should be exactly the keys listed above.
    _ = time.perf_counter
    # Keep this measurement simple:
    # - one forward pass
    # - one backward pass
    # - one optimizer step
    # Then record time and memory right around that work.
    raise NotImplementedError("Train one batch and return loss, seconds, and peak memory.")


# Run one normal step and one checkpointed step, then compare them.
# The goal is to see the direction of the tradeoff:
# - checkpointing usually saves memory
# - checkpointing usually costs extra compute time
def compare_normal_and_checkpointed(config: GradientCheckpointingConfig) -> dict[str, float | str]:
    """Run one normal step and one checkpointed step."""
    # Goal:
    # Compare memory and speed for normal vs checkpointed training.
    #
    # Input:
    # - `config`: GradientCheckpointingConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"normal_loss"`: Python float
    # - `"checkpointed_loss"`: Python float
    # - `"normal_seconds"`: Python float
    # - `"checkpointed_seconds"`: Python float
    # - `"normal_peak_memory_mb"`: Python float
    # - `"checkpointed_peak_memory_mb"`: Python float
    # - `"memory_saved_mb"`: Python float
    # - `"extra_seconds"`: Python float
    # - `"tradeoff_note"`: short plain-English string
    #
    # Method:
    # Use:
    # ```python
    # loader = load_dataloader(config)
    # normal_model, normal_loss_fn, normal_optimizer = build_training_objects(config, use_checkpointing=False)
    # checkpointed_model, checkpointed_loss_fn, checkpointed_optimizer = build_training_objects(config, use_checkpointing=True)
    # normal = train_one_batch(normal_model, loader, normal_loss_fn, normal_optimizer)
    # checkpointed = train_one_batch(checkpointed_model, loader, checkpointed_loss_fn, checkpointed_optimizer)
    # ```
    #
    # Comparison formulas:
    # ```text
    # memory_saved_mb = normal_peak_memory_mb - checkpointed_peak_memory_mb
    # extra_seconds = checkpointed_seconds - normal_seconds
    # ```
    #
    # CPU note:
    # - On CPU, both memory values may be `0.0`.
    # - That is okay. Still compare the timing and explain that CUDA memory was not available.
    #
    # Steps:
    # 1. Set the random seed for a repeatable comparison.
    # 2. Build one dataloader.
    # 3. Build normal training objects.
    # 4. Build checkpointed training objects.
    # 5. Train one batch with the normal model.
    # 6. Train one batch with the checkpointed model.
    # 7. Compute memory saved and extra seconds.
    # 8. Write a short tradeoff note.
    # 9. Return the dictionary above.
    #
    # Checks:
    # The goal is not perfect benchmarking.
    # The goal is to see the direction of the tradeoff:
    # - checkpointing usually lowers memory
    # - checkpointing usually raises compute time
    # - checkpointing does not make the model better by itself
    raise NotImplementedError("Compare memory and speed for normal vs checkpointed training.")


def check_metrics(metrics: dict[str, float | str]) -> bool:
    """Return True when the comparison has the expected values."""
    required = {"normal_loss", "checkpointed_loss", "normal_seconds", "checkpointed_seconds", "tradeoff_note"}
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
    """Run the gradient checkpointing scaffold."""
    config = GradientCheckpointingConfig()
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

    loader, ok = report_task_progress("load_dataloader", load_dataloader, config)
    if not count_or_stop(ok):
        return


    _, ok = report_task_progress("describe_checkpointing_tradeoff", describe_checkpointing_tradeoff, config)
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("DeepToyModel.__init__", DeepToyModel, config, False)
    if not count_or_stop(ok):
        return

    sample_x, _ = next(iter(loader))
    _, ok = report_task_progress("DeepToyModel.forward", model, sample_x)
    if not count_or_stop(ok):
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config, False)
    if not count_or_stop(ok):
        return

    training_model, loss_fn, optimizer = training_objects
    _, ok = report_task_progress("train_one_batch", train_one_batch, training_model, loader, loss_fn, optimizer)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("compare_normal_and_checkpointed", compare_normal_and_checkpointed, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(passed, "checkpointing comparison", float(int(passed)), "all comparison keys present")

if __name__ == "__main__":
    main()
