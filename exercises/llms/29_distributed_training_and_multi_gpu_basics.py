"""
============================================================
EXERCISE 29: DISTRIBUTED TRAINING AND MULTI GPU BASICS
============================================================

WHAT YOU ARE BUILDING
You are building a tiny training script that can run in:
- normal single-process mode
- distributed mode with one process per device

Plain-language meanings:
- a process is one running copy of your Python program
- rank means which process number this is
- world size means how many processes are working together
- distributed training means several processes train the same model together
- DDP means `DistributedDataParallel`, PyTorch's main wrapper for this

WHY THIS EXERCISE EXISTS
Many research jobs train models on more than one GPU.
A beginner does not need large-scale cluster skills on day one,
but they should understand the basic shape of distributed training.

This exercise teaches:
- rank and world size
- process setup and cleanup
- one model copy per device
- `DistributedSampler`
- `DistributedDataParallel`
- metric averaging across processes
- saving checkpoints from rank 0 only

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises taught:
- one model
- one process
- one training loop

This exercise adds the next idea:
- several copies of the same training job can work together

That matters because many real training jobs are too slow or too large for one device.

Beginner intuition:
- imagine one student copying the same worksheet onto several desks
- each desk works on a different part of the homework
- after each step, they compare notes so every copy stays in sync
- that note-sharing step is like gradient synchronization

Small example:
- world size is `2`
- rank `0` uses GPU `0`
- rank `1` uses GPU `1`
- each rank sees different batches from the same dataset
- after backward pass, gradients are shared
- both model copies stay matched

DATASET
Use a small real multiclass classification dataset.

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
- make one fixed split
- wrap the training split with `DistributedSampler`
- use normal evaluation loaders first, then improve later if you want

INPUTS AND OUTPUTS
Input batch:
- `features`
  - type: `torch.Tensor`
  - dtype: `torch.float32`
  - shape: `[batch_size, num_features]`
- `labels`
  - type: `torch.Tensor`
  - dtype: `torch.long`
  - shape: `[batch_size]`

Training summary:
- backend
- world size
- rank count
- train loss
- validation accuracy
- checkpoint path

DEVICE NOTE
- if you have several GPUs, use one GPU per rank
- if you do not have several GPUs, you can still learn the setup on CPU with the `gloo` backend
- the main learning goal is understanding the structure, not chasing speed

MODEL ARCHITECTURE (HIGH LEVEL ONLY)
Use a small MLP classifier.

LOSS FUNCTION
Use `nn.CrossEntropyLoss`.

METRICS TO TRACK
Track at least:
- `train_loss`
- `val_loss`
- `val_accuracy`
- `world_size`
- `backend`
- `used_distributed_sampler`
- `checkpoint_path`

TARGET RESULT RANGE
Focus on correct distributed behavior, not a magic accuracy number.

Useful signs:
- each rank gets different training data
- training runs without rank mismatch errors
- metrics are averaged across ranks
- checkpoint saving happens only once

COMMON MISTAKES
- forgetting `DistributedSampler`
- forgetting `sampler.set_epoch(epoch)`
- saving checkpoints from every rank
- printing every message from every rank
- averaging metrics on one rank only and pretending they are global
- forgetting that each process needs its own device

What each step is doing in simple words:
1. build one small problem so the distributed setup is the hard part, not the dataset
2. connect all the processes so they can talk to each other
3. make sure cleanup happens so later runs do not get stuck
4. build the same model on each rank
5. tell PyTorch that this model should synchronize gradients across ranks
6. give each rank a different slice of the training data
7. run the usual training loop on each rank
8. combine the numbers so you get one honest global summary
9. save only once from rank 0 so files do not collide
10. return a summary that proves the setup worked the way you intended

STEP-BY-STEP TODO
1. Build the real dataset and loaders.
2. Write the distributed setup function.
3. Write the cleanup function.
4. Build the small classifier.
5. Wrap the model with `DistributedDataParallel`.
6. Use `DistributedSampler` for the training loader.
7. Train for a few epochs.
8. Average loss and accuracy across ranks.
9. Save one checkpoint from rank 0 only.
10. Return a summary that proves the distributed path worked.

DEFINITION OF DONE
You are done when:
- rank and world size are used correctly
- the train loader uses `DistributedSampler`
- metrics are combined across ranks
- only rank 0 saves the checkpoint
- you can explain, in plain words, why DDP needs one process per device

LAUNCH NOTE
Typical launch shape:
- `torchrun --nproc_per_node=2 exercises/llms/29_distributed_training_and_multi_gpu_basics.py`

You do not need to memorize the command yet.
The main goal is to understand what the script is doing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset, random_split


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.reproducibility import set_seed
from common.training_checks import print_pass_or_try_again


MetricDict = dict[str, float]
LoaderBundle = dict[str, DataLoader | DistributedSampler]
DistributedSummary = dict[str, str | int | float | bool | None]
DistributedPlan = dict[str, str | list[str]]


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
class DistributedTrainingConfig:
    """Settings for distributed training basics."""

    num_samples: int = 1200
    num_features: int = 20
    num_classes: int = 3
    hidden_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 3
    random_state: int = 42
    backend: str = "nccl"
    cpu_backend: str = "gloo"
    init_method: str = "env://"
    checkpoint_path: str = str(BOOTCAMP_ROOT / "outputs" / "checkpoints" / "exercise_51_ddp.pt")


def describe_distributed_plan(config: DistributedTrainingConfig) -> DistributedPlan:
    """Return the main distributed training pieces to build."""
    return {
        "single_process_device": device,
        "distributed_backend_if_gpu": config.backend,
        "distributed_backend_if_cpu": config.cpu_backend,
        "launch_shape": "torchrun --nproc_per_node=<num_devices> exercises/llms/29_distributed_training_and_multi_gpu_basics.py",
        "required_summary_keys": [
            "backend",
            "world_size",
            "used_distributed_sampler",
            "final_val_accuracy",
            "checkpoint_path",
        ],
    }



# Build the train, validation, and test dataloaders for one rank.
# Return a dict with at least:
# - `train`
# - `val`
# - `test`
# - `train_sampler`
# Important:
# - use `DistributedSampler` for the train loader
# - each rank should see a different training shard
def load_dataloaders(
    config: DistributedTrainingConfig,
    rank: int,
    world_size: int,
) -> LoaderBundle:
    """Return distributed-aware dataloaders and the training sampler."""
    # Goal:
    # - Load a real small classification dataset.
    # - Return dataloaders where the train loader gives each rank a different slice of data.
    #
    # Inputs:
    # - `config.batch_size`: number of rows per batch.
    # - `rank`: this process number. For two ranks, valid ranks are `0` and `1`.
    # - `world_size`: total number of ranks. In single-process mode this is `1`.
    # - `load_wine()` returns:
    #   - features with shape `[178, 13]`
    #   - labels with shape `[178]`
    #   - 3 label classes
    #
    # Output:
    # - Return a dictionary with exactly these keys:
    #   - `"train"`: `DataLoader` yielding `(features, labels)`
    #   - `"val"`: `DataLoader` yielding `(features, labels)`
    #   - `"test"`: `DataLoader` yielding `(features, labels)`
    #   - `"train_sampler"`: `DistributedSampler` used by the train loader
    # - One batch should look like:
    #   - `features`: float tensor with shape `[batch_size, 13]`
    #   - `labels`: long tensor with shape `[batch_size]`
    #
    # Method:
    # - Import the dataset inside this function:
    #   `from sklearn.datasets import load_wine`
    # - Build tensors:
    #   `features = torch.tensor(wine.data, dtype=torch.float32)`
    #   `labels = torch.tensor(wine.target, dtype=torch.long)`
    # - Store them together with:
    #   `dataset = TensorDataset(features, labels)`
    # - Split with:
    #   `train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len], generator=generator)`
    # - Build the train sampler with:
    #   `train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)`
    # - Build the train loader with:
    #   `DataLoader(train_data, batch_size=config.batch_size, sampler=train_sampler)`
    # - Do not pass `shuffle=True` when a sampler is already passed. The sampler controls shuffling.
    #
    # Steps:
    # 1. Assert `0 <= rank < world_size`.
    # 2. Load wine data.
    # 3. Convert features to `torch.float32` and labels to `torch.long`.
    # 4. Set `config.num_features = 13` and `config.num_classes = 3` only if your design allows mutation.
    #    Otherwise, make sure the model config already matches those values before training.
    # 5. Compute split sizes:
    #    - `train_len = int(0.6 * len(dataset))`
    #    - `val_len = int(0.2 * len(dataset))`
    #    - `test_len = len(dataset) - train_len - val_len`
    # 6. Use `torch.Generator().manual_seed(config.random_state)` for a repeatable split.
    # 7. Create a `DistributedSampler` for `train_data`.
    # 8. Create train, validation, and test dataloaders.
    # 9. Return the dictionary with the four exact keys above.
    #
    # Checks:
    # - `set(loaders)` should be `{"train", "val", "test", "train_sampler"}`.
    # - `next(iter(loaders["train"]))[0].shape[1]` should be `13`.
    # - `isinstance(loaders["train_sampler"], DistributedSampler)` should be True.
    raise NotImplementedError("Return train, val, test loaders and the distributed train sampler.")




# Beginner hints for this file:
# - DDP does not mean "one giant model floating across all GPUs"
# - it usually means one full model copy per process and per device
# - `DistributedSampler` is how each rank gets a different slice of training data
# - rank 0 is often the main rank used for logging and saving
# - metric averaging matters because each rank only sees part of the total data

class DistributedMLP(nn.Module):
    """Small classifier used for distributed training practice."""

    def __init__(self, config: DistributedTrainingConfig) -> None:
        # Goal:
        # - Build a small classifier for wine features.
        # - The model should output raw class scores for `nn.CrossEntropyLoss`.
        #
        # Inputs:
        # - `config.num_features`: number of input columns. Wine data uses 13.
        # - `config.hidden_dim`: hidden layer width, such as 64.
        # - `config.num_classes`: number of output classes. Wine data uses 3.
        #
        # Output:
        # - Set `self.net` to an `nn.Sequential`.
        # - The model receives `x` with shape `[batch_size, config.num_features]`.
        # - The model returns logits with shape `[batch_size, config.num_classes]`.
        #
        # Method:
        # - Use this exact module shape:
        #   `self.net = nn.Sequential(`
        #   `    nn.Linear(config.num_features, config.hidden_dim),`
        #   `    nn.ReLU(),`
        #   `    nn.Linear(config.hidden_dim, config.num_classes),`
        #   `)`
        # - `nn.Linear(config.num_features, config.hidden_dim)` reads one feature row.
        # - `nn.ReLU()` adds a simple non-linearity.
        # - `nn.Linear(config.hidden_dim, config.num_classes)` returns one score per class.
        #
        # Steps:
        # 1. Call `super().__init__()`.
        # 2. Assert `config.num_features > 0`, `config.hidden_dim > 0`, and `config.num_classes > 1`.
        # 3. Create `self.net` with the exact `nn.Sequential(...)` shown above.
        #
        # Checks:
        # - `model(torch.randn(4, config.num_features)).shape` should be `[4, config.num_classes]`.
        # - Do not add `Softmax`; `nn.CrossEntropyLoss` expects raw logits.
        #
        super().__init__()
        raise NotImplementedError("Build a small MLP classifier for distributed training.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape `[batch_size, num_classes]`."""
        # Goal:
        # - Run the feature batch through `self.net`.
        #
        # Inputs:
        # - `x`: float tensor with shape `[batch_size, config.num_features]`.
        # - Small example: if batch size is 4 and wine has 13 features, shape is `[4, 13]`.
        #
        # Output:
        # - Return a float tensor of logits with shape `[batch_size, config.num_classes]`.
        # - Example for 3 classes: each row has 3 scores, such as `[0.2, -0.1, 0.7]`.
        #
        # Method:
        # - Use `return self.net(x)`.
        #
        # Steps:
        # 1. Check that `x.ndim == 2`.
        # 2. Pass `x` to `self.net`.
        # 3. Return the logits.
        #
        # Checks:
        # - If `x.shape` is `[4, 13]`, logits should have shape `[4, 3]`.
        #
        raise NotImplementedError("Return raw logits from the classifier.")


# Set up one distributed process.
# In plain words:
# - choose the backend
# - connect this rank to the other ranks
# - prepare the right device for this process
def setup_distributed_process(rank: int, world_size: int, config: DistributedTrainingConfig) -> str:
    """Set up distributed training and return the backend name."""
    # Goal:
    # - Connect this process to the other training processes.
    # - Return the backend name that was used.
    #
    # Inputs:
    # - `rank`: this process number.
    # - `world_size`: total number of processes.
    # - `config.backend`: usually `"nccl"` for CUDA GPUs.
    # - `config.cpu_backend`: usually `"gloo"` for CPU.
    # - `config.init_method`: `"env://"` means PyTorch reads launch info from environment variables.
    # - `torchrun` sets the needed environment variables for you.
    # - If you call this function directly, you must set `MASTER_ADDR` and `MASTER_PORT`
    #   before `dist.init_process_group(...)`, or use a different `init_method`.
    #
    # Output:
    # - Return `"nccl"` when CUDA is available.
    # - Return `"gloo"` when running on CPU.
    #
    # Method:
    # - Pick the backend:
    #   `backend_name = config.backend if torch.cuda.is_available() else config.cpu_backend`
    # - Initialize the process group with:
    #   `dist.init_process_group(backend=backend_name, init_method=config.init_method, rank=rank, world_size=world_size)`
    # - If using CUDA, set the current GPU for this process with:
    #   `torch.cuda.set_device(rank)`
    # - For a real multi-node job, rank-to-device mapping can be more complex. For this exercise,
    #   one rank per local GPU is enough.
    #
    # Steps:
    # 1. Assert `0 <= rank < world_size`.
    # 2. Choose `"nccl"` for CUDA, otherwise `"gloo"`.
    # 3. If `world_size > 1` or you want to practice DDP setup, call `dist.init_process_group(...)`.
    # 4. If using CUDA, call `torch.cuda.set_device(rank)`.
    # 5. Return the backend name.
    #
    # Checks:
    # - After setup, `dist.is_initialized()` should be True when distributed setup was run.
    # - `dist.get_world_size()` should match `world_size`.
    raise NotImplementedError("Initialize the process group and return the backend in use.")


def cleanup_distributed_process() -> None:
    """Clean up the distributed process group."""
    # Goal:
    # - Close the distributed process group after training.
    # - This prevents later runs from getting stuck on an old process group.
    #
    # Inputs:
    # - No direct arguments.
    #
    # Output:
    # - Return `None`.
    #
    # Method:
    # - Check first with `dist.is_available()` and `dist.is_initialized()`.
    # - If both are True, call:
    #   `dist.destroy_process_group()`
    #
    # Steps:
    # 1. If distributed is not available, do nothing.
    # 2. If no process group is initialized, do nothing.
    # 3. Otherwise destroy the process group.
    #
    # Checks:
    # - Calling cleanup twice should not crash.
    #
    raise NotImplementedError("Destroy the process group when distributed work is finished.")
# Build:
# - model
# - loss function
# - optimizer
# The model should be ready to wrap with DDP.
def build_training_objects(
    config: DistributedTrainingConfig,
    local_device: torch.device,
) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """Return model, loss function, and optimizer."""
    # Goal:
    # - Build the model, loss function, and optimizer for this rank.
    # - Move the model to this rank's device before DDP wrapping.
    #
    # Inputs:
    # - `config`: model and optimizer settings.
    # - `local_device`: usually `torch.device("cuda", rank)` or `torch.device("cpu")`.
    #
    # Output:
    # - Return `(model, loss_fn, optimizer)`.
    # - `model`: `DistributedMLP` on `local_device`.
    # - `loss_fn`: `nn.CrossEntropyLoss()`.
    # - `optimizer`: `torch.optim.AdamW`.
    #
    # Method:
    # - Use:
    #   `model = DistributedMLP(config).to(local_device)`
    #   `loss_fn = nn.CrossEntropyLoss()`
    #   `optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`
    # - `nn.CrossEntropyLoss` expects logits with shape `[batch_size, num_classes]`
    #   and labels with shape `[batch_size]`.
    #
    # Steps:
    # 1. Build `DistributedMLP(config)`.
    # 2. Move it to `local_device`.
    # 3. Create `nn.CrossEntropyLoss()`.
    # 4. Create AdamW with `model.parameters()` and `config.learning_rate`.
    # 5. Return the three objects.
    #
    # Checks:
    # - `next(model.parameters()).device` should equal `local_device`.
    # - `type(loss_fn).__name__` should be `"CrossEntropyLoss"`.
    #
    raise NotImplementedError("Return model, CrossEntropyLoss, and optimizer on the local device.")


# Run one distributed training epoch.
# Return:
# - mean train loss for this rank
# - mean train accuracy for this rank
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    local_device: torch.device,
) -> MetricDict:
    """Run one full distributed training epoch for one rank."""
    # Goal:
    # - Train for one full pass over this rank's dataloader.
    # - Return this rank's mean loss and accuracy.
    #
    # Inputs:
    # - `model`: normal model or DDP-wrapped model.
    # - `loader`: yields `(features, labels)`.
    # - `features`: float tensor with shape `[batch_size, 13]`.
    # - `labels`: long tensor with shape `[batch_size]`.
    # - `loss_fn`: `nn.CrossEntropyLoss()`.
    # - `optimizer`: for example `torch.optim.AdamW(...)`.
    # - `local_device`: device for this rank.
    #
    # Output:
    # - Return `{"train_loss": ..., "train_accuracy": ...}`.
    # - `train_loss` is the mean loss per example over this rank.
    # - `train_accuracy` is correct predictions divided by examples seen by this rank.
    #
    # Method:
    # - Put the model in train mode with `model.train()`.
    # - Move tensors with:
    #   `features = features.to(local_device)`
    #   `labels = labels.to(local_device)`
    # - Compute logits and loss:
    #   `logits = model(features)`
    #   `loss = loss_fn(logits, labels)`
    # - Update weights:
    #   `optimizer.zero_grad()`
    #   `loss.backward()`
    #   `optimizer.step()`
    # - Predictions:
    #   `predictions = logits.argmax(dim=1)`
    #
    # Steps:
    # 1. Set `model.train()`.
    # 2. Create counters: `total_loss = 0.0`, `total_correct = 0`, `total_examples = 0`.
    # 3. Loop over `(features, labels)` from `loader`.
    # 4. Move both tensors to `local_device`.
    # 5. Run forward pass, loss, backward pass, and optimizer step.
    # 6. Add `loss.item() * labels.numel()` to `total_loss`.
    # 7. Add `(predictions == labels).sum().item()` to `total_correct`.
    # 8. Add `labels.numel()` to `total_examples`.
    # 9. Return mean loss and accuracy.
    #
    # Checks:
    # - `logits.shape[0] == labels.shape[0]` should be True.
    # - Returned keys should be `{"train_loss", "train_accuracy"}`.
    #
    raise NotImplementedError("Return mean train loss and accuracy for this rank.")


# Evaluate on one loader.
# Return:
# - mean loss for this rank
# - mean accuracy for this rank
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    local_device: torch.device,
) -> MetricDict:
    """Run one evaluation pass for one rank."""
    # Goal:
    # - Measure loss and accuracy without changing model weights.
    #
    # Inputs:
    # - Same batch shape as training:
    #   - `features`: `[batch_size, 13]`, dtype `torch.float32`
    #   - `labels`: `[batch_size]`, dtype `torch.long`
    # - `loss_fn`: `nn.CrossEntropyLoss()`.
    #
    # Output:
    # - Return `{"loss": ..., "accuracy": ...}`.
    # - `loss` is the mean loss per example.
    # - `accuracy` is correct predictions divided by total examples.
    #
    # Method:
    # - Use:
    #   `model.eval()`
    #   `with torch.no_grad():`
    # - Inside the loop, use the same forward pass as training:
    #   `logits = model(features)`
    #   `loss = loss_fn(logits, labels)`
    #   `predictions = logits.argmax(dim=1)`
    #
    # Steps:
    # 1. Set eval mode.
    # 2. Create the same three counters as training.
    # 3. Enter `with torch.no_grad():`.
    # 4. Loop over the loader.
    # 5. Move tensors to `local_device`.
    # 6. Compute logits, loss, predictions, and counters.
    # 7. Return mean loss and accuracy.
    #
    # Checks:
    # - No optimizer step should happen in this function.
    # - Returned keys should be `{"loss", "accuracy"}`.
    #
    raise NotImplementedError("Return mean loss and accuracy for this rank.")


# Average one metrics dictionary across ranks.
# In plain words:
# - each rank has only part of the full picture
# - this function combines the numbers so the summary is global
def average_metrics_across_ranks(metrics: MetricDict, local_device: torch.device) -> MetricDict:
    """Average float metrics across all ranks."""
    # Goal:
    # - Combine this rank's metric values with the same values from every other rank.
    # - Return one global average dictionary.
    #
    # Inputs:
    # - `metrics`: dictionary of float values, such as `{"loss": 0.8, "accuracy": 0.7}`.
    # - All ranks must pass the same keys in the same order.
    # - `local_device`: the device where the reduction tensor should live.
    #
    # Output:
    # - Return a dictionary with the same keys as `metrics`.
    # - Each value is averaged across ranks.
    #
    # Method:
    # - If `dist.is_available()` or `dist.is_initialized()` is False, return `metrics` unchanged.
    # - Otherwise use:
    #   `keys = list(metrics.keys())`
    #   `values = torch.tensor([metrics[key] for key in keys], device=local_device, dtype=torch.float32)`
    #   `dist.all_reduce(values, op=dist.ReduceOp.SUM)`
    #   `values = values / dist.get_world_size()`
    # - `all_reduce` sends the values from every rank, sums them, and gives the same sum back
    #   to every rank.
    #
    # Steps:
    # 1. If distributed is not initialized, return `metrics`.
    # 2. Save metric keys in a list.
    # 3. Turn metric values into a float tensor on `local_device`.
    # 4. Call `dist.all_reduce(..., op=dist.ReduceOp.SUM)`.
    # 5. Divide by `dist.get_world_size()`.
    # 6. Build and return a new dictionary from the keys and averaged values.
    #
    # Checks:
    # - If `world_size == 1`, output values should match input values.
    # - Output keys should match input keys.
    #
    raise NotImplementedError("Average the metric values across all ranks and return the combined result.")


# Save one checkpoint only on rank 0.
# Save at least:
# - epoch
# - model state
# - optimizer state
# - backend
# - world size
def save_checkpoint_on_rank_zero(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: DistributedTrainingConfig,
    rank: int,
    world_size: int,
    backend_name: str,
) -> str | None:
    """Save a checkpoint from rank 0 only."""
    # Goal:
    # - Save one checkpoint file, but only from rank 0.
    # - Other ranks should return `None` so they do not write the same file at the same time.
    #
    # Inputs:
    # - `model`: normal model or DDP-wrapped model.
    # - `optimizer`: optimizer whose state should be saved.
    # - `epoch`: current epoch number.
    # - `config.checkpoint_path`: output path.
    # - `rank`: process number.
    # - `world_size`: total number of processes.
    # - `backend_name`: backend used, such as `"gloo"` or `"nccl"`.
    #
    # Output:
    # - If `rank != 0`, return `None`.
    # - If `rank == 0`, save the file and return `config.checkpoint_path`.
    #
    # Method:
    # - DDP stores the real model at `model.module`.
    # - Use this safe pattern:
    #   `model_to_save = model.module if hasattr(model, "module") else model`
    # - Make the folder:
    #   `Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)`
    # - Save:
    #   `torch.save(checkpoint, config.checkpoint_path)`
    # - Checkpoint dictionary keys:
    #   - `"epoch"`: int
    #   - `"model_state_dict"`: `model_to_save.state_dict()`
    #   - `"optimizer_state_dict"`: `optimizer.state_dict()`
    #   - `"backend"`: backend name
    #   - `"world_size"`: world size
    #
    # Steps:
    # 1. If `rank != 0`, return `None`.
    # 2. Pick `model.module` when the model is DDP-wrapped.
    # 3. Build the checkpoint dictionary with the keys above.
    # 4. Create the parent folder.
    # 5. Save with `torch.save(...)`.
    # 6. Return the checkpoint path string.
    #
    # Checks:
    # - Rank 0 should return a string path.
    # - Rank 1 and higher should return `None`.
    raise NotImplementedError("Save one checkpoint from rank 0 and return its path.")


# Run one worker process.
# Return a summary from rank 0.
# Non-zero ranks may return an empty dict or a tiny status dict.
def run_distributed_worker(rank: int, world_size: int, config: DistributedTrainingConfig) -> DistributedSummary:
    """Run one distributed worker from setup to cleanup."""
    # Goal:
    # - Run all work for one rank from setup to cleanup.
    # - Return the main summary from rank 0.
    #
    # Inputs:
    # - `rank`: process number.
    # - `world_size`: total process count.
    # - `config`: training settings.
    #
    # Output:
    # - Rank 0 should return a summary with these keys:
    #   - `"backend"`: backend string
    #   - `"world_size"`: int
    #   - `"used_distributed_sampler"`: bool
    #   - `"final_train_loss"`: float
    #   - `"final_val_loss"`: float
    #   - `"final_val_accuracy"`: float
    #   - `"checkpoint_path"`: str or None
    # - Non-zero ranks can return `{}` because rank 0 owns the user-facing summary.
    #
    # Method:
    # - Choose the local device:
    #   `local_device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")`
    # - Wrap with DDP only after moving the model to the device:
    #   `model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])`
    #   when using CUDA.
    # - On CPU, use:
    #   `model = torch.nn.parallel.DistributedDataParallel(model)`
    # - Before each epoch, call:
    #   `loaders["train_sampler"].set_epoch(epoch)`
    # - `set_epoch` matters because it changes the shuffled order in the same way across ranks.
    #
    # Steps:
    # 1. Call `setup_distributed_process(...)`.
    # 2. Create `local_device`.
    # 3. Build dataloaders.
    # 4. Build model, loss, and optimizer.
    # 5. Wrap the model in `DistributedDataParallel`.
    # 6. For each epoch:
    #    - call `train_sampler.set_epoch(epoch)`
    #    - train one epoch
    #    - evaluate on validation data
    #    - average metrics across ranks
    # 7. Save a checkpoint with `save_checkpoint_on_rank_zero(...)`.
    # 8. Build the summary only on rank 0.
    # 9. Always call `cleanup_distributed_process()` in a `finally` block.
    #
    # Checks:
    # - `used_distributed_sampler` should be True.
    # - Only rank 0 should return a non-empty summary.
    # - Cleanup should run even if training raises an error.
    #
    raise NotImplementedError("Run setup, training, metric averaging, saving, and cleanup for one rank.")


# Launch the distributed training demo.
# Beginner-friendly first version:
# - choose `world_size` from available GPUs
# - if only CPU is available, let `world_size` be 1
def launch_distributed_demo(config: DistributedTrainingConfig) -> DistributedSummary:
    """Launch the distributed training demo and return the main summary."""
    # Goal:
    # - Start the distributed demo and return the rank-0 summary.
    # - Keep CPU-only machines usable by running world size 1.
    #
    # Inputs:
    # - `config`: training and launch settings.
    #
    # Output:
    # - Return the same rank-0 summary described in `run_distributed_worker(...)`.
    #
    # Method:
    # - Choose world size with:
    #   `world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1`
    # - For the beginner version, call the worker directly when `world_size == 1`:
    #   `return run_distributed_worker(rank=0, world_size=1, config=config)`
    # - If that direct path still calls `dist.init_process_group(init_method="env://")`,
    #   set `MASTER_ADDR="127.0.0.1"` and `MASTER_PORT` to a free port before setup.
    # - For `world_size > 1`, launch one process per GPU with:
    #   `torch.multiprocessing.spawn(run_distributed_worker, args=(world_size, config), nprocs=world_size, join=True)`
    # - `spawn` does not directly return rank 0's dictionary, so a full multi-process version
    #   usually writes the summary to a file or uses a multiprocessing queue. For this exercise,
    #   make the single-process path work first, then add the multi-process result handoff.
    #
    # Steps:
    # 1. Count CUDA devices.
    # 2. If there are no CUDA devices, set `world_size = 1`.
    # 3. If `world_size == 1`, call `run_distributed_worker(0, 1, config)` directly.
    # 4. If `world_size > 1`, use `torch.multiprocessing.spawn(...)`.
    # 5. Return the rank-0 summary.
    #
    # Checks:
    # - On CPU, returned `"world_size"` should be 1.
    # - Returned summary should pass `check_metrics(...)`.
    #
    raise NotImplementedError("Launch the distributed demo and return the rank-0 summary.")


def check_metrics(metrics: DistributedSummary) -> bool:
    """Return True when the distributed summary has the expected keys."""
    required = {
        "backend",
        "world_size",
        "used_distributed_sampler",
        "final_val_accuracy",
        "checkpoint_path",
    }
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
    """Run the distributed training scaffold one step at a time."""
    config = DistributedTrainingConfig()
    set_seed(config.random_state)

    completed_steps = 0
    total_steps = 5

    def count_or_stop(ok: bool) -> bool:
        """Update progress and return whether the scaffold should keep running."""
        nonlocal completed_steps
        if ok:
            completed_steps += 1
            print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
            return True
        print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")
        return False

    loaders, ok = report_task_progress("load_dataloaders", load_dataloaders, config, 0, 1)
    if not count_or_stop(ok):
        return
    _, ok = report_task_progress("describe_distributed_plan", describe_distributed_plan, config)
    if not count_or_stop(ok):
        return

    model, ok = report_task_progress("DistributedMLP.__init__", DistributedMLP, config)
    if not count_or_stop(ok):
        return

    sample_x = torch.randn(config.batch_size, config.num_features)
    _, ok = report_task_progress("DistributedMLP.forward", model, sample_x)
    if not count_or_stop(ok):
        return

    metrics, ok = report_task_progress("launch_distributed_demo", launch_distributed_demo, config)
    if ok:
        completed_steps += 1
    print(f"Progress: {completed_steps}/{total_steps} core tasks finished.")

    if ok and metrics is not None:
        passed = check_metrics(metrics)
        print_pass_or_try_again(
            passed,
            "distributed training summary",
            float(int(passed)),
            "distributed keys present",
        )

if __name__ == "__main__":
    main()
