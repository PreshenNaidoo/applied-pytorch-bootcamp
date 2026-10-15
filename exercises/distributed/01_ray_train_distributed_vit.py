"""
============================================================
EXERCISE 1: RAY TRAIN DISTRIBUTED VIT
============================================================

WHAT YOU ARE BUILDING
You are training one real Vision Transformer with two Ray Train workers.
Each worker owns one GPU and receives a different part of CIFAR-100.

This first exercise teaches the part Ray owns:
- starting workers
- giving each worker a rank and GPU
- setting up PyTorch distributed communication
- splitting the data between workers
- returning one set of metrics to the driver process

There is no single-GPU baseline. The first run uses two workers.

REAL SYSTEM AND THIS EXERCISE
real large run                    this exercise
---------------------------------------------------------------
large image collection           real CIFAR-100 images
large ViT                        torchvision VisionTransformer
many cluster workers             2 Ray workers by default
long training                    a small subset and 2 epochs

The model in this file is copied on both GPUs. Exercise 2 replaces this model
copy with FSDP2 parameter sharding.

INPUTS AND OUTPUTS
One image batch:
- images: `[local_batch, 3, 224, 224]`
- labels: `[local_batch]`
- logits: `[local_batch, 100]`

With 2 workers and `batch_size_per_worker=16`, one optimizer step sees up to
32 images in total. Each worker still holds only its own 16 images.

LOSS AND METRIC
Cross-entropy loss checks whether the correct class has a high score.

global accuracy = correct predictions on all workers / images on all workers

Do not average worker accuracies directly. One worker may see a shorter final
batch. Add the raw counts first, then divide.

FULL LOOP
1. Ray starts two worker processes.
2. Every worker builds the same ViT.
3. Ray wraps it with DistributedDataParallel (DDP).
4. Ray gives each loader a distributed sampler.
5. Each worker runs forward, loss, backward, and optimizer step.
6. DDP averages gradients during backward.
7. Workers add their metric counts together.
8. Ray reports one global result.

COMMON MISTAKES
- calling `dist.all_reduce` only on rank 0; every worker must join a collective
- calling `train.report` a different number of times on different workers
- averaging worker accuracies instead of adding correct and image counts
- forgetting `prepare_data_loader`, so every worker reads the same images
- downloading the dataset separately in every worker at the same time
- treating local batch size as global batch size

STEP-BY-STEP TODO
Complete the stubs in this order:
1. `reduce_classification_counts`
2. `train_loop_per_worker`
3. `build_trainer`

DEFINITION OF DONE
- the logs show world ranks 0 and 1
- both workers use CUDA
- global validation count is close to the configured validation size
- Ray returns loss and accuracy for each epoch
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_data_loader, prepare_model
from torch import nn

from _shared_vit import (
    DistributedViTConfig,
    build_data_loaders,
    build_vit,
    count_parameters,
    seed_everything,
)


def reduce_classification_counts(
    loss_sum: float,
    correct: int,
    example_count: int,
    device: torch.device,
) -> dict[str, float]:
    """Add validation counts from every worker and return global metrics."""

    # Where this fits:
    # Every worker evaluates only its own data. This helper combines those
    # separate pieces before rank 0 reports the result.
    #
    # Goal:
    # Return one loss and accuracy calculated from all workers together.
    #
    # Input example from one worker:
    # loss_sum=41.5, correct=73, example_count=100
    # `loss_sum` must already be the sum of per-image losses, not a batch mean.
    #
    # Output:
    # {"validation_loss": 0.415, "validation_accuracy": 0.73,
    #  "validation_examples": 200.0}
    # after counts from every worker have been added.
    #
    # Method:
    # 1. Create one float64 tensor on `device` containing
    #    `[loss_sum, correct, example_count]`.
    # 2. Call `dist.all_reduce(counts, op=dist.ReduceOp.SUM)` from
    #    `torch.distributed`. Every worker must call it.
    # 3. Divide the global loss sum and correct count by the global image count.
    # 4. Include the global image count as `validation_examples`.
    # 5. Use `.item()` when turning a one-value tensor into a Python number.
    #
    # Check:
    # `0.0 <= validation_accuracy <= 1.0` and the denominator is not zero.
    raise NotImplementedError("Combine the three counts with dist.all_reduce.")


def train_loop_per_worker(train_loop_config: dict[str, object]) -> None:
    """Run inside every Ray worker."""

    # Where this fits:
    # `TorchTrainer` starts one copy of this function per worker. Code outside
    # this function runs only in the driver process.
    #
    # Goal:
    # Train and validate the ViT, then report global metrics after each epoch.
    #
    # Input:
    # `train_loop_config` is `DistributedViTConfig.to_dict()`. Rebuild it with
    # `DistributedViTConfig.from_dict(train_loop_config)`.
    #
    # Exact Ray calls:
    # - `context = train.get_context()`
    # - `rank = context.get_world_rank()`
    # - `world_size = context.get_world_size()`
    # - `device = get_device()`
    # - `model = prepare_model(model)`
    # - `train_loader = prepare_data_loader(train_loader)`
    # - `train.report({"epoch": ..., "validation_loss": ..., ...})`
    #
    # Steps:
    # 1. Rebuild the config and seed with `config.seed`.
    # 2. Get rank, world size, and device. Print them once.
    # 3. Build the loaders with `build_data_loaders(config, download=False)`.
    #    Download CIFAR-100 once in `main` before workers start.
    # 4. Pass both loaders through `prepare_data_loader`. Ray adds a sampler so
    #    workers do not train on the same examples.
    # 5. Build the ViT and call `prepare_model(model)`. Ray moves it to the GPU
    #    and wraps it in PyTorch DDP.
    # 6. Create `torch.optim.AdamW(model.parameters(),
    #    lr=config.learning_rate, weight_decay=config.weight_decay)`.
    # 7. Create `nn.CrossEntropyLoss()` for training and
    #    `nn.CrossEntropyLoss(reduction="sum")` for validation counts.
    # 8. For each epoch, call `train_loader.sampler.set_epoch(epoch)` when the
    #    sampler has that method. This changes the shuffle order on every epoch.
    # 9. Run the normal PyTorch train loop: move batch to device, clear
    #    gradients, forward, loss, backward, optimizer step.
    # 10. Validate under `torch.no_grad()`. Add summed loss, correct predictions,
    #     and image count for this worker.
    # 11. Call `reduce_classification_counts(...)`.
    # 12. Call `train.report(...)` from every worker with exactly these keys:
    #     - `epoch`
    #     - `world_size`
    #     - `model_parameters`
    #     - `validation_loss`
    #     - `validation_accuracy`
    #     - `validation_examples`
    #
    # Check:
    # Print `count_parameters(model)` before training. Both ranks should print
    # the same value, because DDP keeps a full model copy on each GPU.
    raise NotImplementedError("Write the Ray worker training and validation loop.")


def build_trainer(config: DistributedViTConfig) -> TorchTrainer:
    """Create the two-worker Ray Train job."""

    # Where this fits:
    # This runs in the driver and tells Ray how much hardware the job needs.
    #
    # Goal:
    # Return a `TorchTrainer` that launches two GPU workers.
    #
    # Method and full call:
    # `TorchTrainer(
    #     train_loop_per_worker=train_loop_per_worker,
    #     train_loop_config=config.to_dict(),
    #     scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    # )`
    #
    # `num_workers=2` means two Python worker processes.
    # `use_gpu=True` asks Ray for one GPU per worker by default.
    #
    # Check:
    # `trainer.fit()` should wait until Ray can reserve two GPUs.
    raise NotImplementedError("Create TorchTrainer with two GPU workers.")


def check_final_result(metrics: dict[str, object]) -> None:
    """Provided final check for the Ray worker and metric behavior."""

    required = {
        "epoch",
        "world_size",
        "model_parameters",
        "validation_loss",
        "validation_accuracy",
        "validation_examples",
    }
    missing = required.difference(metrics)
    if missing:
        raise AssertionError(f"Ray metrics are missing keys: {sorted(missing)}")
    if int(metrics["world_size"]) != 2:
        raise AssertionError("Exercise 1 must run with two Ray workers.")
    if int(metrics["model_parameters"]) <= 0:
        raise AssertionError("The reported model parameter count must be positive.")
    accuracy = float(metrics["validation_accuracy"])
    if not 0.0 <= accuracy <= 1.0:
        raise AssertionError("Validation accuracy must be between 0 and 1.")
    if int(metrics["validation_examples"]) <= 0:
        raise AssertionError("No validation images were counted.")


def main() -> None:
    config = DistributedViTConfig()

    # The driver downloads once. This is safe when workers can read the same
    # `data_root`. For a multi-node cluster, stage this folder on every node or
    # use shared storage before calling `trainer.fit()`.
    build_data_loaders(config, download=True)

    trainer = build_trainer(config)
    result = trainer.fit()
    check_final_result(result.metrics)
    print(result.metrics)


if __name__ == "__main__":
    main()
