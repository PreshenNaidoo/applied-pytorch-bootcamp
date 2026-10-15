"""
============================================================
EXERCISE 3: FSDP2 VIT MEMORY CONTROLS
============================================================

WHAT YOU ARE BUILDING
You are running the same automatically sharded ViT with four memory settings:

1. FSDP2 only
2. FSDP2 + bfloat16 mixed precision
3. FSDP2 + activation checkpointing
4. FSDP2 + both controls

CPU offload is an optional fifth run. It can fit more model state by moving
local parameter shards to CPU memory, but GPU-to-CPU transfers often slow the
step down.

WHY THIS EXERCISE EXISTS
FSDP2 removes full parameter copies between workers, but activations can still
use a lot of memory. Activations are intermediate tensors saved during forward
so backward can calculate gradients.

- mixed precision stores many tensors with fewer bytes
- activation checkpointing saves fewer activations and calculates them again
  during backward
- CPU offload trades speed for more available GPU memory

These controls solve different memory problems, so they can be combined.

REAL WORKFLOW
The dataset, ViT, loss, optimizer, and FSDP2 layout stay fixed. Only one memory
setting changes at a time. This makes the comparison meaningful.

MEMORY EXAMPLE
A tensor with 1,000,000 values needs about:
- float32: 4,000,000 bytes
- bfloat16: 2,000,000 bytes

Activation checkpointing does not change this byte size. It reduces how many
activation tensors stay saved between forward and backward.

METRICS TO TRACK
- `peak_memory_gib`: highest allocated CUDA memory during measured steps; lower
  is better
- `mean_step_seconds`: average train step time; lower is better
- `images_per_second`: global images processed per second; higher is better
- `loss`: should stay finite and in a similar range between settings

Do not time the first few warm-up steps. CUDA kernel setup makes them unusually
slow.

COMMON MISTAKES
- stopping the timer before queued CUDA work finishes
- comparing runs with different local batches or measured step counts
- assuming bfloat16 halves every part of memory; optimizer shards stay in their
  original precision by default
- expecting activation checkpointing to improve speed; it repeats computation
- assuming CPU offload is free; it adds CPU memory use and data transfers
- changing several controls at once before measuring each one separately

STEP-BY-STEP TODO
1. `build_mixed_precision_policy`
2. `enable_activation_checkpointing`
3. `build_offload_policy`
4. `measure_train_steps`
5. `train_loop_per_worker`
6. `run_memory_experiments`

DEFINITION OF DONE
- each run still discovers and shards all ViT blocks automatically
- all four required settings complete the same number of measured steps
- a printed table compares memory, time, throughput, and loss
- you can explain why the lowest-memory setting may not be the fastest
"""

from __future__ import annotations

import time
from functools import partial

import torch
import torch.distributed as dist
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_data_loader
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torchvision.models.vision_transformer import EncoderBlock

from _shared_vit import DistributedViTConfig, build_data_loaders, build_vit, get_vit_blocks


EXPERIMENTS = (
    "fsdp2",
    "bf16",
    "activation_checkpointing",
    "bf16_and_activation_checkpointing",
)


def build_mixed_precision_policy(experiment: str) -> MixedPrecisionPolicy:
    """Choose full precision or bfloat16 from the experiment name."""

    # Where this fits:
    # The policy is passed into every `fully_shard` call. It tells FSDP2 which
    # dtype to use during forward and gradient reduction.
    #
    # Goal:
    # Return a bfloat16 policy for the two experiment names containing `bf16`.
    # Return the default full-precision policy for the other names.
    #
    # Method and full calls:
    # - bfloat16:
    #   `MixedPrecisionPolicy(
    #       param_dtype=torch.bfloat16,
    #       reduce_dtype=torch.float32,
    #   )`
    # - full precision: `MixedPrecisionPolicy()`
    #
    # `param_dtype` is the dtype used for parameter computation.
    # `reduce_dtype` is the dtype used when workers combine gradients. Keeping
    # reductions in float32 is a simple stable starting point.
    #
    # Check:
    # `build_mixed_precision_policy("bf16").param_dtype == torch.bfloat16`.
    raise NotImplementedError("Return the mixed-precision policy for this run.")


def enable_activation_checkpointing(model: nn.Module) -> None:
    """Checkpoint every ViT transformer block."""

    # Where this fits:
    # Call this after the child blocks are sharded and before the root is
    # sharded. It changes how each EncoderBlock saves activations, but it does
    # not change the block's mathematical result.
    #
    # Goal:
    # Wrap every `EncoderBlock` with non-reentrant activation checkpointing.
    #
    # Method and full call:
    # `wrapper = partial(
    #     checkpoint_wrapper,
    #     checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    # )`
    # `apply_activation_checkpointing(
    #     model,
    #     checkpoint_wrapper_fn=wrapper,
    #     check_fn=lambda module: isinstance(module, EncoderBlock),
    # )`
    #
    # `check_fn` decides which child modules are wrapped. It finds blocks by
    # type, so there is no hand-written layer list.
    #
    # Check:
    # Run forward and backward once. Both must work and produce a finite loss.
    raise NotImplementedError("Apply checkpointing to every EncoderBlock.")


def build_offload_policy(use_cpu_offload: bool) -> OffloadPolicy:
    """Return CPU offload only for the optional offload experiment."""

    # Goal:
    # Return `CPUOffloadPolicy()` when use_cpu_offload is true. Return
    # `OffloadPolicy()` for the normal GPU-resident path. Both classes come from
    # `torch.distributed.fsdp`.
    raise NotImplementedError("Choose whether local parameter shards live on CPU.")


def shard_vit_for_experiment(
    model: nn.Module,
    mesh: DeviceMesh,
    experiment: str,
    use_cpu_offload: bool,
) -> nn.Module:
    """Provided FSDP2 layout from exercise 2, now with memory policies."""

    mixed_precision = build_mixed_precision_policy(experiment)
    offload = build_offload_policy(use_cpu_offload)
    for block in get_vit_blocks(model):
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mixed_precision,
            offload_policy=offload,
        )
    if "activation_checkpointing" in experiment:
        enable_activation_checkpointing(model)
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mixed_precision,
        offload_policy=offload,
    )
    return model


def measure_train_steps(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, float]:
    """Measure stable train-step speed and peak allocated CUDA memory."""

    # Goal:
    # Run warm-up and measured training steps, then return local raw values.
    #
    # Output keys:
    # - `elapsed_seconds`: total time for measured steps
    # - `local_images`: images processed during measured steps on this worker
    # - `peak_memory_bytes`: this worker's peak allocated CUDA bytes
    # - `loss_sum`: sum of per-image losses across measured batches
    # - `measured_steps`: actual number of measured steps
    #
    # Method:
    # - `torch.cuda.reset_peak_memory_stats(device)` clears the old peak.
    # - `torch.cuda.synchronize(device)` waits for queued CUDA work. Call it just
    #   before starting and just after stopping the timer.
    # - `torch.cuda.max_memory_allocated(device)` returns bytes.
    # - `time.perf_counter()` is a high-resolution wall-clock timer.
    #
    # Steps:
    # 1. Iterate over at most `warmup_steps + measured_steps` batches.
    # 2. For every batch, do a real optimizer step.
    # 3. After warm-up, reset memory, synchronize, and start the timer once.
    # 4. Count images, loss, and steps only in the measured section. If the loss
    #    is a batch mean, add `loss.item() * images.shape[0]` to loss_sum.
    # 5. Synchronize after the final measured step, then stop the timer.
    #
    # Check:
    # elapsed time and peak bytes must both be greater than zero.
    raise NotImplementedError("Measure real FSDP2 train steps after warm-up.")


def train_loop_per_worker(train_loop_config: dict[str, object]) -> None:
    """Run one selected memory experiment on every Ray worker."""

    # Goal:
    # Build the selected model, measure it, combine worker values, and report.
    #
    # Steps:
    # 1. Read `experiment`, `use_cpu_offload`, `warmup_steps`, and
    #    `measured_steps` from train_loop_config. Rebuild the ViT config.
    # 2. Build and prepare the train loader.
    # 3. Get Ray's device, call `torch.cuda.set_device(device)`, then build the
    #    CUDA mesh with
    #    `init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))`.
    # 4. Build the model and call `shard_vit_for_experiment(...)`.
    # 5. Create AdamW after sharding.
    # 6. Call `measure_train_steps(...)`.
    # 7. Use `dist.all_reduce` to add image and loss counts. Use
    #    `dist.all_reduce(..., op=dist.ReduceOp.MAX)` for elapsed time and peak
    #    memory because the slowest and largest worker limit the whole step.
    # 8. Report peak GiB (`bytes / 1024**3`), mean step seconds, global images
    #    per second, mean loss (`global_loss_sum / global_image_count`),
    #    experiment name, and world size.
    raise NotImplementedError("Run and report one distributed memory experiment.")


def run_memory_experiments(config: DistributedViTConfig) -> list[dict[str, object]]:
    """Launch the four fair comparison jobs from the Ray driver."""

    # Goal:
    # Run every name in `EXPERIMENTS` with the same model and measurement budget.
    #
    # Method:
    # For each name, create `TorchTrainer(...,
    # scaling_config=ScalingConfig(num_workers=2, use_gpu=True))`, call `.fit()`,
    # and append `result.metrics` to a list.
    #
    # Use `warmup_steps=3` and `measured_steps=10` to start. Keep these values
    # identical across experiments.
    #
    # Output:
    # One list of metric dictionaries that can be printed as a table.
    raise NotImplementedError("Run the same measurement for each memory setting.")


def check_experiment_rows(rows: list[dict[str, object]]) -> None:
    """Provided final check that the comparison is complete and meaningful."""

    names = {str(row.get("experiment")) for row in rows}
    if names != set(EXPERIMENTS):
        raise AssertionError(f"Expected {list(EXPERIMENTS)}, but received {sorted(names)}")
    required = {
        "experiment",
        "world_size",
        "peak_memory_gib",
        "mean_step_seconds",
        "global_images_per_second",
        "mean_loss",
        "measured_steps",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise AssertionError(f"Experiment row is missing keys: {sorted(missing)}")
        if int(row["world_size"]) != 2:
            raise AssertionError("Every memory experiment must use two workers.")
        for key in ("peak_memory_gib", "mean_step_seconds", "global_images_per_second"):
            if float(row[key]) <= 0.0:
                raise AssertionError(f"{key} must be positive.")


def main() -> None:
    config = DistributedViTConfig(epochs=1)
    build_data_loaders(config, download=True)
    rows = run_memory_experiments(config)
    check_experiment_rows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
