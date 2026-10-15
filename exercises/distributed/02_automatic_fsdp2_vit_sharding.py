"""
============================================================
EXERCISE 2: AUTOMATIC FSDP2 VIT SHARDING
============================================================

WHAT YOU ARE BUILDING
You are training a ViT whose parameters, gradients, and optimizer state are
automatically split across Ray workers by PyTorch FSDP2.

You will not assign layers to GPUs. Your code finds every ViT transformer block
by its module type. FSDP2 then works out the tensor slices for the current worker
count.

WHY THIS EXERCISE EXISTS
DDP in exercise 1 stores a full model copy on every GPU. That becomes the main
problem when the model is too large for one GPU. FSDP2 stores only a shard of
most model state on each GPU and gathers parameters when a layer needs them.

PLAIN MEANINGS
- shard: one slice of a larger tensor
- device mesh: the group of GPUs that share model state
- bottom-up: shard child transformer blocks first, then shard the whole model
- reshard: free the gathered full parameter after its computation finishes

AUTOMATIC BOUNDARY
You choose the useful unit to wrap: each transformer block. You do this once by
searching the model. FSDP2 automatically chooses every parameter slice. Adding
more identical blocks does not require new GPU placement code.

INPUTS AND OUTPUTS
- input images: `[local_batch, 3, 224, 224]`
- patch grid: `14 x 14` for 224-pixel images and 16-pixel patches
- token count: 196 patch tokens + 1 class token = 197
- logits: `[local_batch, 100]`

MEMORY PICTURE FOR N WORKERS
A rough parameter-state picture is:

local sharded bytes ~= full sharded bytes / N

Temporary full parameters are still gathered for computation, so measured peak
memory will not be exactly `1/N` of the original value.

FULL LOOP
1. Ray starts workers and initializes PyTorch distributed.
2. Every worker builds the same ViT on CPU.
3. Code finds all `EncoderBlock` objects automatically.
4. FSDP2 shards each block, then the root model.
5. The optimizer is created after sharding.
6. Forward gathers one block's parameters when needed.
7. Backward reduces and re-shards gradients.
8. The optimizer updates local shards.

COMMON MISTAKES
- creating the optimizer before `fully_shard`; it must receive DTensor parameters
- sharding the root before child blocks; FSDP2 grouping is built bottom-up
- calling Ray's `prepare_model`, which would add DDP around the FSDP2 model
- calling `model.forward(images)` directly; call `model(images)` so hooks run
- thinking a DTensor's global shape is the amount stored on one GPU
- writing rank-specific layer placement even though FSDP2 owns tensor slices

STEP-BY-STEP TODO
1. `find_vit_blocks`
2. `build_data_parallel_mesh`
3. `automatically_shard_vit`
4. `assert_model_is_sharded`
5. `describe_largest_parameter_shard`
6. `train_loop_per_worker`
7. `build_trainer`

DEFINITION OF DONE
- all 12 ViT blocks are found without a hand-written block list
- every block and the root model are FSDP modules
- at least one model parameter is a DTensor
- one forward, backward, and optimizer step succeeds on two GPUs
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_data_loader
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor
from torchvision.models.vision_transformer import EncoderBlock

from _shared_vit import (
    DistributedViTConfig,
    build_data_loaders,
    build_vit,
    seed_everything,
)


def find_vit_blocks(model: nn.Module) -> list[EncoderBlock]:
    """Find every transformer block from the model structure."""

    # Where this fits:
    # FSDP2 works best when large repeated blocks are sharded separately. This
    # lets it gather the next block while the current block is computing.
    #
    # Goal:
    # Return all torchvision `EncoderBlock` objects in model order.
    #
    # Input:
    # A torchvision `VisionTransformer` stored as the general type `nn.Module`.
    #
    # Output:
    # A list with `config.num_layers` entries. For the default model, length 12.
    #
    # Method:
    # Use `model.modules()` from `torch.nn.Module`. Keep a module when
    # `isinstance(module, EncoderBlock)` is true.
    #
    # Steps:
    # 1. Walk `model.modules()` once.
    # 2. Collect matching blocks.
    # 3. Raise `ValueError` when no blocks were found. Silent success would leave
    #    the model with a poor sharding layout.
    #
    # Check:
    # `len(find_vit_blocks(model)) == 12` for the default config.
    raise NotImplementedError("Find all EncoderBlock objects automatically.")


def build_data_parallel_mesh(world_size: int) -> DeviceMesh:
    """Create the one-dimensional GPU group used for FSDP2."""

    # Where this fits:
    # The mesh tells FSDP2 which workers share one model. It does not tell FSDP2
    # where individual layers or tensor pieces go.
    #
    # Goal:
    # Return one CUDA mesh with `world_size` workers and the dimension name
    # `"fsdp"`.
    #
    # Method and full call:
    # `init_device_mesh(
    #     device_type="cuda",
    #     mesh_shape=(world_size,),
    #     mesh_dim_names=("fsdp",),
    # )`
    #
    # `(world_size,)` is a one-dimensional shape. The comma matters in Python.
    #
    # Check:
    # `mesh.size() == world_size`.
    raise NotImplementedError("Create a one-dimensional CUDA device mesh.")


def automatically_shard_vit(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """Apply FSDP2 to every ViT block and then to the root model."""

    # Where this fits:
    # This is the one function that changes a normal ViT into a sharded ViT.
    #
    # Goal:
    # Use automatic block discovery and FSDP2. Do not name a GPU, rank, tensor,
    # parameter, or block number.
    #
    # Method:
    # `fully_shard(module, mesh=mesh)` comes from `torch.distributed.fsdp`.
    # It changes the module in place and returns that module.
    #
    # Steps:
    # 1. Call `find_vit_blocks(model)`.
    # 2. Loop over the returned blocks and call `fully_shard(block, mesh=mesh)`.
    # 3. After every child block is sharded, call
    #    `fully_shard(model, mesh=mesh)` on the root model.
    # 4. Return `model`.
    #
    # Why bottom-up:
    # If the root is sharded first, FSDP2 cannot build the intended nested block
    # groups. Child modules must be handled before their parent.
    #
    # Check:
    # `isinstance(model, FSDPModule)` should be true after this function.
    raise NotImplementedError("Shard discovered blocks first, then the root model.")


def assert_model_is_sharded(model: nn.Module, expected_blocks: int) -> None:
    """Fail early when the automatic sharding step missed part of the ViT."""

    # Goal:
    # Check the root, every block, and the parameter type.
    #
    # Steps:
    # 1. Assert `isinstance(model, FSDPModule)`.
    # 2. Call `find_vit_blocks(model)` and assert its length is expected_blocks.
    # 3. Assert every found block is an `FSDPModule`.
    # 4. Assert `any(isinstance(p, DTensor) for p in model.parameters())`.
    #
    # A DTensor stores both a local tensor slice and information about how all
    # slices form the full tensor.
    raise NotImplementedError("Check that FSDP2 changed the root and every block.")


def describe_largest_parameter_shard(model: nn.Module) -> dict[str, object]:
    """Show the global parameter and the local slice stored by this worker."""

    # Where this fits:
    # The type check above proves FSDP2 ran. This helper makes the memory split
    # concrete by looking at one real ViT parameter.
    #
    # Goal:
    # Find the DTensor parameter with the largest global number of values and
    # return its global and local shapes.
    #
    # Input:
    # An already-sharded ViT whose parameters are DTensors outside forward and
    # backward computation.
    #
    # Output keys:
    # - `parameter_name`: string from `model.named_parameters()`
    # - `global_shape`: list built from `parameter.shape`
    # - `local_shape`: list built from `parameter.to_local().shape`
    # - `global_values`: `parameter.numel()`
    # - `local_values`: `parameter.to_local().numel()`
    # - `placements`: list of strings from `parameter.placements`
    #
    # Method:
    # 1. Keep `(name, parameter)` pairs where `isinstance(parameter, DTensor)`.
    # 2. Use `max(pairs, key=lambda pair: pair[1].numel())`.
    # 3. Call `.to_local()` only to inspect the local slice. Do not replace the
    #    model parameter with that local tensor.
    #
    # Concrete expectation with two workers:
    # A large matrix with global shape `[3072, 768]` will normally have a local
    # shape close to `[1536, 768]` on each worker.
    #
    # Check:
    # For world size 2, `local_values < global_values` for this large parameter.
    raise NotImplementedError("Describe one real global DTensor and its local shard.")


def train_loop_per_worker(train_loop_config: dict[str, object]) -> None:
    """Train one automatically sharded ViT inside each Ray worker."""

    # Goal:
    # Run one real FSDP2 train and validation loop.
    #
    # Steps:
    # 1. Rebuild and seed the config.
    # 2. Get `world_size = train.get_context().get_world_size()` and
    #    `device = get_device()`. Call `torch.cuda.set_device(device)` before
    #    creating the CUDA mesh.
    # 3. Build loaders with `download=False`, then pass each through
    #    `prepare_data_loader(loader)` so Ray splits the data.
    # 4. Build the ViT on CPU.
    # 5. Build the mesh and call `automatically_shard_vit(model, mesh)`. FSDP2
    #    moves local shards to the mesh device, so do not call `prepare_model`.
    # 6. Call `assert_model_is_sharded(model, config.num_layers)`.
    # 7. Call `describe_largest_parameter_shard(model)` and print its dictionary
    #    with the current rank. This is the direct evidence of automatic slicing.
    # 8. Create AdamW after sharding. This order matters because the optimizer
    #    must see the sharded parameters.
    # 9. Train with the normal forward, cross-entropy, backward, and step calls.
    # 10. Validate and combine `[loss_sum, correct, count]` with
    #    `dist.all_reduce(..., op=dist.ReduceOp.SUM)`.
    # 11. Every worker calls `train.report(...)` with exactly these keys:
    #     - `epoch`, `world_size`, and `sharded_blocks`
    #     - `has_dtensor`
    #     - `global_parameter_values` and `local_parameter_values` from the
    #       shard description
    #     - `validation_loss`, `validation_accuracy`, and `validation_examples`
    #
    # Checks:
    # - logits have shape `[local_batch, config.num_classes]`
    # - loss is finite
    # - validation accuracy is between 0 and 1
    raise NotImplementedError("Write the Ray Train and FSDP2 training loop.")


def build_trainer(config: DistributedViTConfig) -> TorchTrainer:
    """Create a two-GPU FSDP2 Ray job."""

    # Goal:
    # Return the Ray job object. Do not call `.fit()` inside this helper.
    #
    # Input:
    # `config` contains the model, data, and optimizer settings that every worker
    # receives through `train_loop_config`.
    #
    # Output:
    # One `TorchTrainer`. Its `.fit()` method returns the Ray Result.
    #
    # Method and full call:
    # `TorchTrainer(
    #     train_loop_per_worker=train_loop_per_worker,
    #     train_loop_config=config.to_dict(),
    #     scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    # )`
    raise NotImplementedError("Create the two-worker TorchTrainer.")


def check_final_result(metrics: dict[str, object], expected_blocks: int) -> None:
    """Provided final check that real sharding happened."""

    required = {
        "world_size",
        "sharded_blocks",
        "has_dtensor",
        "global_parameter_values",
        "local_parameter_values",
        "validation_loss",
        "validation_accuracy",
        "validation_examples",
    }
    missing = required.difference(metrics)
    if missing:
        raise AssertionError(f"Ray metrics are missing keys: {sorted(missing)}")
    if int(metrics["world_size"]) != 2:
        raise AssertionError("Exercise 2 must run with two workers.")
    if int(metrics["sharded_blocks"]) != expected_blocks:
        raise AssertionError("FSDP2 did not shard every ViT block.")
    if not bool(metrics["has_dtensor"]):
        raise AssertionError("No FSDP2 DTensor parameter was reported.")
    if int(metrics["local_parameter_values"]) >= int(metrics["global_parameter_values"]):
        raise AssertionError("The inspected local parameter was not smaller than its global form.")
    accuracy = float(metrics["validation_accuracy"])
    if not 0.0 <= accuracy <= 1.0:
        raise AssertionError("Validation accuracy must be between 0 and 1.")


def main() -> None:
    config = DistributedViTConfig()
    build_data_loaders(config, download=True)
    result = build_trainer(config).fit()
    check_final_result(result.metrics, config.num_layers)
    print(result.metrics)


if __name__ == "__main__":
    main()
