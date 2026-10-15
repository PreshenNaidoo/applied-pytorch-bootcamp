"""
============================================================
EXERCISE 6: FSDP2 WORKER-COUNT RESHARDING
============================================================

WHAT YOU ARE BUILDING
You are saving a sharded ViT with two workers, then restoring it with four
workers and continuing training.

No model code, transformer-block list, or tensor split rule changes between the
two runs. Distributed Checkpoint reads the saved state and FSDP2 creates the new
local shards for the four-worker mesh.

WHY THIS EXERCISE EXISTS
Available cluster size can change. A useful large-model checkpoint should not be
tied to the exact number of workers that wrote it.

SMALL SHARD EXAMPLE
Imagine a parameter with 12 rows:
- with 2 workers, each worker may own about 6 rows
- with 4 workers, each worker may own about 3 rows

The checkpoint stores enough information to rebuild the parameter. You do not
write code to cut 6-row files into 3-row files.

FULL WORKFLOW
1. Stage A builds a two-worker mesh and trains one epoch.
2. Every Stage A worker writes its DCP shard.
3. Stage B builds a new four-worker mesh and the same ViT architecture.
4. `train.get_checkpoint()` gives Stage B the Stage A checkpoint.
5. Every Stage B worker calls DCP load into its live four-worker shards.
6. A fixed real CIFAR-100 image checks that the restored model is unchanged.
7. Stage B continues with the restored optimizer and step count.

WHAT MUST STAY THE SAME
- model architecture and class count
- checkpoint state keys
- optimizer type and parameter groups

WHAT MAY CHANGE
- worker count
- local shard shapes
- local batch ownership

COMMON MISTAKES
- changing model dimensions between Stage A and Stage B
- loading before the new four-worker FSDP2 model and optimizer exist
- writing separate two-worker and four-worker tensor slicing code
- comparing checksums from different validation images
- resetting the optimizer or global step after loading
- assuming a normal single-file `torch.save` checkpoint will reshard itself

STEP-BY-STEP TODO
1. `validate_worker_plan`
2. `train_or_resume_worker`
3. `run_stage`
4. `verify_resharding_result`
5. `run_resharding_demo`

DEFINITION OF DONE
- Stage A reports world size 2 and returns a DCP checkpoint
- Stage B reports world size 4 and says it restored that checkpoint
- Stage B's starting global step equals Stage A's saved step
- the fixed-image checksum matches before new Stage B updates
- training continues without manual rank, layer, or tensor placement rules
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from ray import train
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_data_loader
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from _shared_vit import DistributedViTConfig, build_data_loaders, build_vit, get_vit_blocks


@dataclass
class ReshardPosition:
    next_epoch: int = 0
    global_step: int = 0
    saved_checksum: float = 0.0


class ReshardState(Stateful):
    """Provided sharded model, optimizer, and comparison state."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        position: ReshardPosition,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.position = position

    def state_dict(self) -> dict[str, Any]:
        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state,
            "optimizer": optimizer_state,
            "next_epoch": torch.tensor(self.position.next_epoch, dtype=torch.int64),
            "global_step": torch.tensor(self.position.global_step, dtype=torch.int64),
            "saved_checksum": torch.tensor(
                self.position.saved_checksum,
                dtype=torch.float64,
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )
        self.position.next_epoch = int(state_dict["next_epoch"].item())
        self.position.global_step = int(state_dict["global_step"].item())
        self.position.saved_checksum = float(state_dict["saved_checksum"].item())


def shard_vit(model: nn.Module, world_size: int) -> nn.Module:
    """Use the same automatic layout for any worker count."""

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    for block in get_vit_blocks(model):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def validate_worker_plan(save_workers: int, resume_workers: int) -> None:
    """Check that this run really tests a different shard layout."""

    # Goal:
    # Raise ValueError unless both counts are positive and the counts differ.
    # The default exercise uses `save_workers=2` and `resume_workers=4`.
    #
    # Check:
    # `(2, 4)` passes, `(2, 2)` raises, and `(0, 4)` raises.
    raise NotImplementedError("Validate two positive, different worker counts.")


def fixed_image_checksum(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> float:
    """Measure model output on one fixed real image without learning."""

    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
    model.train()
    return float(logits.float().sum().item())


def train_or_resume_worker(train_loop_config: dict[str, object]) -> None:
    """Run either Stage A or Stage B inside each Ray worker."""

    # Goal:
    # Save with one mesh size or load into another mesh size with the same code.
    #
    # Steps:
    # 1. Read `stage_name` and rebuild the ViT config.
    # 2. Build loaders. Before giving any loader to Ray, read
    #    `fixed_image, _ = validation_loader.dataset[0]`. This uses the same
    #    deterministic real validation image on every rank. Prepare only the
    #    train loader with Ray.
    # 3. Get Ray world size and device, then call
    #    `torch.cuda.set_device(device)`. Build the ViT and call
    #    `shard_vit(model, world_size)`.
    # 4. Create AdamW, `ReshardPosition`, and `ReshardState` in that order.
    # 5. If `train.get_checkpoint()` returns a checkpoint, call
    #    `dcp.load(state_dict={"reshard_state": state},
    #    checkpoint_id=checkpoint_dir)` inside `checkpoint.as_directory()`.
    # 6. Immediately after load and before any update, calculate the fixed-image
    #    checksum. Save `starting_step = position.global_step`.
    # 7. On Stage B, compare this checksum with `position.saved_checksum` using
    #    `abs(current - saved) < 1e-4`. Raise an AssertionError if it differs.
    # 8. Train the requested epochs and update the counters.
    # 9. Calculate the checksum again. Store it in `position.saved_checksum`
    #    before saving.
    # 10. Every rank calls DCP save in one temporary directory and reports a Ray
    #     checkpoint with stage, world size, restored flag, starting step, final
    #     step, and checksum-match result.
    #
    # Important:
    # Do not branch on world size when selecting blocks or parameters. Only the
    # mesh shape changes.
    raise NotImplementedError("Train or resume with the current Ray world size.")


def run_stage(
    config: DistributedViTConfig,
    *,
    worker_count: int,
    stage_name: str,
    checkpoint: Checkpoint | None,
) -> train.Result:
    """Launch one worker-count stage from the driver."""

    # Goal:
    # Build `TorchTrainer` with the requested `worker_count`, `use_gpu=True`, and
    # `resume_from_checkpoint=checkpoint`. Put `stage_name` in loop config.
    # Call `.fit()` and return its Result.
    #
    # Check:
    # The reported `world_size` equals worker_count.
    raise NotImplementedError("Launch one save or resume stage.")


def verify_resharding_result(first: train.Result, second: train.Result) -> None:
    """Check the facts that prove the checkpoint changed worker counts."""

    # Goal:
    # Assert all of these:
    # - first world size is 2
    # - second world size is 4
    # - second restored flag is true
    # - second checksum-match flag is true
    # - second starting step equals first final step
    #
    # Use `first.metrics[...]` and `second.metrics[...]`.
    raise NotImplementedError("Check world sizes, counters, and model checksum.")


def run_resharding_demo(config: DistributedViTConfig) -> tuple[train.Result, train.Result]:
    """Run two-worker save followed by four-worker restore."""

    # Steps:
    # 1. Call `validate_worker_plan(2, 4)`.
    # 2. Run Stage A with 2 workers and checkpoint=None.
    # 3. Assert Stage A returned a checkpoint.
    # 4. Run Stage B with 4 workers and Stage A's checkpoint.
    # 5. Call `verify_resharding_result(first, second)`.
    # 6. Return both results.
    raise NotImplementedError("Run the two stages and verify automatic resharding.")


def main() -> None:
    config = DistributedViTConfig(epochs=1)
    build_data_loaders(config, download=True)
    first, second = run_resharding_demo(config)
    print("saved with 2 workers:", first.metrics)
    print("restored with 4 workers:", second.metrics)


if __name__ == "__main__":
    main()
