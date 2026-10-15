"""
============================================================
EXERCISE 11: TRAINING STABILITY TOOLS
============================================================

WHAT YOU ARE BUILDING
You are building one training loop with common stability tools.

Plain-language meanings:
- a scheduler changes the learning rate during training
- gradient clipping stops very large gradients from causing huge updates
- gradient accumulation lets several small batches act like one bigger batch
- mixed precision can make GPU training faster and use less memory

WHY THIS EXERCISE EXISTS
Large model training often depends on these tools.
They are also useful for debugging smaller models.

This exercise teaches:
- how to read the current learning rate
- how to step a scheduler
- how to measure and clip gradient norm
- how to delay optimizer steps for accumulation
- how to use mixed precision only when the device supports it

HOW THIS FITS INTO THE BOOTCAMP
After basic training loops, checkpointing, and tracking,
this file teaches common tools that make larger training runs easier to control.
These are the kinds of details that often separate a toy loop from a useful training loop.

What each step is doing in simple words:
1. start from a normal training loop
2. make several small batches behave like one bigger batch
3. measure whether gradients are getting too large
4. clip them before a harmful update
5. change the learning rate over time
6. use mixed precision only when the device can handle it
7. return proof that each tool was actually wired in

Beginner intuition:
- these tools do not change the task itself
- they change how training behaves while trying to solve the task
- some help stability
- some help memory
- some help speed

Small example:
- if your batch is too large for memory, accumulation can let several small batches act like one bigger batch
- if gradients become huge, clipping can stop one bad update from pushing training far off course

DATASET
Use one real multiclass vector dataset.

Clear default:
- `sklearn.datasets.load_wine`
- features: 13 numeric wine measurements
- labels: 3 wine classes

Why this fits the exercise:
- it is small enough to debug quickly
- it is real data, not generated points
- it still has enough examples to show learning-rate and gradient behavior

HOW TO LOAD OR CREATE THE DATASET
Simple path:
- load `sklearn.datasets.load_wine`
- convert features and labels to tensors
- split into train, validation, and test
- keep the model small so you can debug the training loop easily

INPUTS AND OUTPUTS
Input batch:
- `features`
  - shape: `[batch_size, num_features]`
- `labels`
  - shape: `[batch_size]`

Model output:
- raw class logits with shape `[batch_size, num_classes]`

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small MLP classifier.
Keep the model simple so the training tools stay the main focus.

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

Another simple way to think about the tools:
- scheduler: "how fast should learning move right now?"
- clipping: "is this update too large?"
- accumulation: "can I build a bigger effective batch from smaller ones?"
- mixed precision: "can I do some math more cheaply without breaking training?"

METRICS TO TRACK
Track at least:
- `train_loss`
- `val_accuracy`
- `final_learning_rate`
- `max_gradient_norm_before_clip`
- `optimizer_steps`
- `used_mixed_precision`

COMMON MISTAKES
- calling `optimizer.step()` on every batch when using accumulation
- forgetting to divide the loss by `accumulation_steps`
- clipping gradients before `loss.backward()`
- using CUDA-only mixed precision code on CPU
- stepping the scheduler at the wrong time

TARGET RESULT RANGE
Focus on proving that the tools are wired correctly.

Useful signs:
- the learning rate changes over time
- the optimizer steps fewer times when accumulation is on
- gradient norm is measured before clipping
- mixed precision is only marked as used when the device supports it

STEP-BY-STEP TODO
1. Build a small baseline model and dataloaders.
2. Add gradient accumulation to the training loop.
3. Add gradient norm measurement.
4. Add gradient clipping.
5. Add a learning-rate scheduler.
6. Add optional mixed precision for CUDA.
7. Return a summary that proves each tool was used.

DEFINITION OF DONE
You are done when:
- accumulation changes how often the optimizer steps
- gradient norm is measured before clipping
- scheduler changes the learning rate
- CPU users can still run the exercise
- you can explain each tool in plain words
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

#from scipy import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class StabilityToolsConfig:
    """Settings for training stability tools."""

    num_samples: int = 178
    num_features: int = 13
    num_classes: int = 3
    hidden_dim: int = 96
    batch_size: int = 16
    learning_rate: float = 2e-3
    epochs: int = 15
    accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    random_state: int = 42
    target_val_accuracy: float = 0.70


def describe_stability_tools(config: StabilityToolsConfig) -> dict[str, object]:
    """Return the tools this exercise should use."""
    return {
        "scheduler": "cosine or step scheduler",
        "gradient_clipping": config.max_grad_norm,
        "gradient_accumulation_steps": config.accumulation_steps,
        "mixed_precision_requested": config.use_mixed_precision,
        "mixed_precision_available": torch.cuda.is_available(),
    }



# Beginner hints for this file:
# - add one tool at a time so you can see what changed
# - gradient accumulation means "delay optimizer.step()" until several mini-batches are combined
# - clipping should happen after backward pass and before optimizer step
# - a scheduler changes learning rate over time; print the value so you can see it move
# - mixed precision should be optional because CPU users still need the file to run

# Build the train, validation, and test dataloaders.
# Keep the dataset simple.
# The real lesson in this file is the training loop behavior.
def load_dataloaders(config: StabilityToolsConfig) -> dict[str, DataLoader]:
    """Return train, val, and test dataloaders."""
    # One clean implementation plan:
    # - load `sklearn.datasets.load_wine`
    # - keep at most `config.num_samples` examples
    # - use the real feature table and real labels from the dataset
    # - convert them to:
    #   - `features`: `torch.float32`
    #   - `labels`: `torch.long`
    # - build one `TensorDataset`
    # - make one `60 / 20 / 20` split with a fixed generator seed
    # - return three dataloaders
    # Why this is enough:
    # - the training loop tools are the main point
    # - the dataset only needs to be simple, easy to batch, and large enough
    #   to make accumulation and scheduling feel meaningful
    # Practical note:
    # - this is not an image dataset
    # - each example is just one row of numbers
    # - this keeps the exercise focused on training-loop stability tools

    raise NotImplementedError("Return dataloaders for the stability exercise.")





class StabilityMLP(nn.Module):
    """Small classifier for stability-tool practice."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()

        raise NotImplementedError("Build a small MLP classifier.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits."""

        raise NotImplementedError("Return raw logits from the classifier.")



# Build:
# - model
# - loss function
# - optimizer
# - scheduler
# These are the pieces the training loop will use.
def build_training_objects(
    config: StabilityToolsConfig,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer, object]:
    """Return model, loss function, optimizer, and scheduler."""
    # Keep the first version boring and clear:
    # - one model
    # - one optimizer
    # - one scheduler
    # Then add clipping, accumulation, and mixed precision in the training loop.

    raise NotImplementedError("Return model, loss, optimizer, and scheduler.")



# Return the current learning rate as one Python float.
# This is the easiest way to prove the scheduler changed it.
def get_current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate from the first optimizer parameter group."""
    
    raise NotImplementedError("Return the current learning rate.")



# Measure the full gradient norm for the current model state.
# This gives you one number that says how large the gradient signal is.
def compute_gradient_norm(model: nn.Module) -> float:
    """Return the total gradient norm for model parameters."""
    # Plain idea:
    # - each parameter has a gradient after backward pass
    # - gradient norm is one number that says how large the overall update signal is
    # - clipping is a way to stop that signal from becoming too large
    #
    # Why use `detach()`?
    # - here we only want to measure the gradients
    # - we do not want PyTorch to track this measurement as part of training
    # - `param.grad.detach()` means:
    #   "use these gradient values as plain numbers"
    #
    # Important:
    # - `.norm(2)` measures length
    # - it does not turn the vector length into 1
    #
    # Small example:
    # - if `x = torch.tensor([3.0, 4.0])`
    # - then `x.norm(2)` is 5.0
    # - because `sqrt(3^2 + 4^2) = 5`
    # - `x` is still `[3.0, 4.0]`
    # - to make its length 1, you would divide by the norm:
    #   `x / x.norm(2)`
    #
    # Why square each parameter norm, add them, then square root?
    # - the model has many separate parameter tensors
    # - we want one number for the whole model
    # - this is like flattening every gradient into one giant vector and finding the norm
    #
    # Small model-wide example:
    # - layer 1 weight gradient norm is 3
    # - layer 1 bias gradient norm is 4
    # - layer 2 weight gradient norm is 12
    # - total gradient norm is:
    #   `sqrt(3^2 + 4^2 + 12^2) = 13`
    #
    # Simple intuition:
    # - small norm means a small total push on the model
    # - large norm means a large total push on the model
    # - clipping shrinks that push if it is too large
    #
    # This is not the average norm across parameters.
    # It is the total norm across all gradients in the model.
    # If you had many parameters, this number can grow because it includes all of them.
    #
    # Average norm vs total norm:
    # - suppose three parameter tensors have gradient norms:
    #   3, 4, and 12
    # - the average norm is:
    #   `(3 + 4 + 12) / 3 = 6.33`
    # - the total norm is:
    #   `sqrt(3^2 + 4^2 + 12^2) = 13`
    #
    # Why clipping uses total norm instead of average norm:
    # - clipping is trying to limit the whole model update
    # - the optimizer updates all parameters, not just one average parameter
    # - total norm measures the size of the full update signal
    # - average norm can hide the full size of the update
    #
    # Simple example of why average can hide the problem:
    # - imagine 100 parameter tensors
    # - 99 have tiny gradient norm 0.1
    # - 1 has huge gradient norm 100
    # - the average looks much smaller than 100
    # - but the model still has one very large update signal
    # - total norm makes that large signal visible
    #
    # Simple clipping example:
    # - suppose the total gradient norm is 10
    # - suppose `max_grad_norm` is 2
    # - clipping scales all gradients by:
    #   `2 / 10 = 0.2`
    # - a gradient value of 5 becomes:
    #   `5 * 0.2 = 1`
    # - a gradient value of 1 becomes:
    #   `1 * 0.2 = 0.2`
    # - the direction stays the same
    # - the whole update just becomes smaller
    #
    # Tiny code example:
    # ```python
    # total_norm_before = compute_gradient_norm(model)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    # total_norm_after = compute_gradient_norm(model)
    # print(total_norm_before, total_norm_after)
    # ```
    #
    # If the norm before clipping was 10,
    # the norm after clipping should be close to 2.

    raise NotImplementedError("Measure the gradient norm before clipping.")




#This is a unused function for understanding the code without mized precision complexities
def train_one_epoch_no_mixed_precision(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    config: StabilityToolsConfig,
) -> dict[str, float | int | bool]:
    """Train one epoch with accumulation, clipping, and scheduler."""
    model.train()
    optimizer.zero_grad()

    losses = []
    accuracies = []
    max_gradient_norm_before_clip = 0.0
    optimizer_steps = 0

    for batch_index, (features, labels) in enumerate(loader):
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        raw_loss = loss_fn(logits, labels)
        loss = raw_loss / config.accumulation_steps

        loss.backward()

        is_accumulation_step = (batch_index + 1) % config.accumulation_steps == 0
        is_last_batch = (batch_index + 1) == len(loader)

        if is_accumulation_step or is_last_batch:
            gradient_norm_before_clip = compute_gradient_norm(model)
            max_gradient_norm_before_clip = max(
                max_gradient_norm_before_clip,
                gradient_norm_before_clip,
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            optimizer_steps += 1

        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()

        losses.append(raw_loss.item())
        accuracies.append(accuracy)

    scheduler.step()

    mean_loss = torch.tensor(losses).mean().item()
    mean_accuracy = torch.tensor(accuracies).mean().item()

    return {
        "loss": mean_loss,
        "accuracy": mean_accuracy,
        "max_gradient_norm_before_clip": max_gradient_norm_before_clip,
        "optimizer_steps": optimizer_steps,
        "final_learning_rate": get_current_learning_rate(optimizer),
        "used_mixed_precision": False,
    }

# Run one full training epoch with accumulation, clipping, scheduler, and optional mixed precision.
# Return a summary dict with values such as:
# - loss
# - max gradient norm before clip
# - optimizer step count
# - final learning rate
# - whether mixed precision was used
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    config: StabilityToolsConfig,
) -> dict[str, float | int | bool]:
    """Train one epoch with accumulation, clipping, scheduler, and optional mixed precision."""
    # Suggested order inside one batch:
    # 1. forward pass
    # 2. compute loss
    # 3. divide loss if using accumulation
    # 4. backward pass
    # 5. clip gradients before optimizer step
    # 6. only step optimizer every `accumulation_steps` batches
    # 7. step the scheduler in the right place for your chosen scheduler type

    # normal precision usually uses float32, mixed precision lets some GPU math use smaller numbers, 
    # often float16, this can be faster and use less memory on CUDA GPUs.
    # You normally use two tools:
    # torch.amp.autocast and torch.amp.GradScaler
    # autocast: runs some operations in lower precision when safe
    # GradScaler: protects gradients from becoming too tiny when using lower precision

    # Simple order:
    # forward + loss inside autocast
    # scale loss
    # backward
    # unscale gradients
    # measure gradient norm
    # clip gradients
    # optimizer step through scaler
    # scaler update
    # zero gradients    

    # Why divide the loss by `accumulation_steps` on every batch?
    # - your intuition may be:
    #   "add several losses, then divide by accumulation_steps"
    # - that idea is right
    # - but in practice, we usually divide each mini-batch loss before `backward()`
    # - this gives almost the same result while using less memory
    #
    # Example:
    # - batch 1 gradient is 8
    # - batch 2 gradient is 4
    # - `accumulation_steps = 2`
    #
    # Option A:
    # - add first, then average:
    #   `(8 + 4) / 2 = 6`
    #
    # Option B:
    # - divide each batch first, then let PyTorch add gradients:
    #   `8 / 2 = 4`
    #   `4 / 2 = 2`
    #   `4 + 2 = 6`
    #
    # Same final gradient.
    #
    # Why use Option B?
    # - it avoids storing all loss tensors
    # - it lets PyTorch free each batch's computation graph after `backward()`
    # - it uses less memory
    #
    # What does `loss.backward()` do?
    # - `loss.backward()` does backpropagation
    # - it computes gradients
    # - it stores those gradients in `param.grad`
    # - it does not update the weights
    #
    # What does `optimizer.step()` do?
    # - it reads the gradients stored in `param.grad`
    # - it updates the model weights
    #
    # Small weight update example:
    # - weight is 10
    # - gradient is 3
    # - learning rate is 0.1
    # - after `loss.backward()`, PyTorch has:
    #   `weight.grad = 3`
    # - after `optimizer.step()`, the optimizer does roughly:
    #   `weight = 10 - 0.1 * 3 = 9.7`
    #
    # Short version:
    # - `backward()` works out which direction to move
    # - `optimizer.step()` actually moves the weights
    #
    # Why scale the loss with mixed precision?
    # - mixed precision may use `float16`
    # - `float16` has trouble representing very tiny numbers
    # - gradients can be tiny, such as `0.00000003`
    # - in `float16`, a tiny gradient can become 0
    # - 0 gradient means no learning signal
    #
    # Loss scaling example:
    # - real gradient is `0.00000003`
    # - scale is `1000000`
    # - scaled gradient is `0.03`
    # - `0.03` is much easier for `float16` to represent
    #
    # Why unscale before clipping or stepping?
    # - after scaling, the gradients are too large on purpose
    # - before clipping, we need the real gradient size again
    # - `scaler.unscale_(optimizer)` puts gradients back to their real size
    #
    # Clipping example with scaling:
    # - real gradient norm is 2
    # - scale is 1000
    # - scaled gradient norm looks like 2000
    # - clipping 2000 would clip the wrong number
    # - so we unscale first, then measure and clip the real norm


    raise NotImplementedError("Return train metrics and stability-tool tracking values.")





    

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict[str, float]:
    """Run one full evaluation pass.
    
        Main differences from train_one_epoch:

        no loss.backward()
        no optimizer.step()
        no scheduler step
        no gradient clipping
        no mixed precision needed, unless you want it
    """

    raise NotImplementedError("Return validation loss and accuracy.")






def run_training(config: StabilityToolsConfig) -> dict[str, float | int | bool]:
    """Run the full stability-tool training workflow."""

    raise NotImplementedError("Train with stability tools and return final summary metrics.")






def check_metrics(metrics: dict[str, float | int | bool], target: float) -> bool:
    """Return True when the summary shows the main tools were tracked."""
    required = {
        "val_accuracy",
        "final_learning_rate",
        "max_gradient_norm_before_clip",
        "optimizer_steps",
        "used_mixed_precision",
    }
    return required.issubset(metrics) and float(metrics.get("val_accuracy", 0.0)) >= target


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    print(f"[DONE] {task_name}")
    return result, True


def show_learning_rate_schedule(
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    num_steps: int,
) -> None:
    """Plot how the scheduler changes the learning rate."""
    if not hasattr(scheduler, "step") or not hasattr(scheduler, "state_dict"):
        print("[SKIP] learning-rate plot: scheduler does not look like a PyTorch scheduler.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] learning-rate plot: matplotlib is not installed.")
        return

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    learning_rates = [float(optimizer.param_groups[0]["lr"])]

    for _ in range(num_steps):
        scheduler.step()
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))

    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)

    steps = list(range(len(learning_rates)))
    plt.figure(figsize=(7, 4))
    plt.plot(steps, learning_rates, marker="o")
    plt.title("Learning rate schedule")
    plt.xlabel("Scheduler step")
    plt.ylabel("Learning rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the training stability scaffold one step at a time."""
    config = StabilityToolsConfig()
    set_seed(config.random_state)

    completed_steps = 0
    total_steps = 10

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


    _, ok = report_task_progress("describe_stability_tools", describe_stability_tools, config)
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("StabilityMLP.__init__", StabilityMLP, config.num_features, config.hidden_dim, config.num_classes)
    if not count_or_stop(ok):
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("StabilityMLP.forward", model, sample_x)
    if not count_or_stop(ok):
        return

    training_objects, ok = report_task_progress("build_training_objects", build_training_objects, config)
    if not count_or_stop(ok):
        return

    training_model, loss_fn, optimizer, scheduler = training_objects
    show_learning_rate_schedule(optimizer, scheduler, config.epochs)

    _, ok = report_task_progress("get_current_learning_rate", get_current_learning_rate, optimizer)
    if not count_or_stop(ok):
        return

    train_loader = loaders["train"]
    sample_features, sample_labels = next(iter(train_loader))
    sample_features = sample_features.to(device)
    sample_labels = sample_labels.to(device)
    optimizer.zero_grad()
    sample_logits = training_model(sample_features)
    sample_loss = loss_fn(sample_logits, sample_labels)
    sample_loss.backward()

    _, ok = report_task_progress("compute_gradient_norm", compute_gradient_norm, training_model)
    optimizer.zero_grad()
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("train_one_epoch", train_one_epoch, training_model, train_loader, loss_fn, optimizer, scheduler, config)
    if not count_or_stop(ok):
        return

    _, ok = report_task_progress("evaluate", evaluate, training_model, loaders["val"], loss_fn)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("run_training", run_training, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if ok and metrics is not None:
        passed = check_metrics(metrics, config.target_val_accuracy)
        print_pass_or_try_again(passed, "val_accuracy", float(metrics.get("val_accuracy", 0.0)), f">= {config.target_val_accuracy}")


if __name__ == "__main__":
    main()
