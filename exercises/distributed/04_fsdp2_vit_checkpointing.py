"""
============================================================
EXERCISE 4: FSDP2 VIT CHECKPOINTING
============================================================

WHAT YOU ARE BUILDING
You are saving and restoring an automatically sharded ViT with:
- PyTorch Distributed Checkpoint (DCP), which knows how to save DTensor shards
- a Ray Checkpoint, which makes the saved directory part of the Ray Train result

WHY THIS EXERCISE EXISTS
A normal `torch.save(model.state_dict())` path often gathers a full model or
assumes one process owns the whole state. That is the wrong mental model for a
model that cannot fit on one GPU.

DCP lets every worker write its local pieces. On restore, every worker reads the
pieces it needs. No worker has to build one full parameter dictionary in GPU
memory.

STATE THAT MUST BE SAVED
- model parameters: what the ViT has learned
- optimizer state: AdamW's moving averages for each parameter
- scheduler state: the learning-rate position
- epoch and step: where the loop should continue
- random-number state: helps a resumed run follow the same random sequence

FULL WORKFLOW
1. Build and automatically shard the ViT.
2. Create the optimizer and scheduler.
3. If Ray supplies a checkpoint, restore it before training continues.
4. Train one epoch.
5. Every worker calls DCP save for its local shards.
6. Every worker calls `train.report` with the Ray Checkpoint.
7. Start a new trainer from the checkpoint and continue.

SMALL CONCRETE CHECK
If the first run saves `epoch=0` and `global_step=20`, the next run should start
at `epoch=1` and `global_step=20`. It must not restart at epoch 0.

COMMON MISTAKES
- saving only on rank 0; every rank owns checkpoint shards
- gathering one full state dictionary onto a GPU before saving
- loading before the model is sharded and the optimizer exists
- restoring weights but losing AdamW moving averages
- saving `epoch=0` and then running epoch 0 again after restore
- checking only that files exist instead of checking restored model output

STEP-BY-STEP TODO
1. `TrainingState.state_dict`
2. `TrainingState.load_state_dict`
3. `save_and_report_checkpoint`
4. `restore_if_available`
5. `train_loop_per_worker`
6. `build_trainer`

DEFINITION OF DONE
- checkpoint files are created without gathering the full model on one GPU
- a second two-worker job restores model and optimizer state
- epoch and global step continue from the saved values
- fixed validation images give matching logits immediately before save and
  immediately after restore, within floating-point tolerance
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from ray import train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_data_loader
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from _shared_vit import DistributedViTConfig, build_data_loaders, build_vit, get_vit_blocks


@dataclass
class LoopPosition:
    next_epoch: int = 0
    global_step: int = 0
    saved_checksum: float = 0.0


class TrainingState(Stateful):
    """Connect the live training objects to Distributed Checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        position: LoopPosition,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.position = position

    def state_dict(self) -> dict[str, Any]:
        # Where this fits:
        # DCP calls this method when it needs a saveable view of live state.
        #
        # Goal:
        # Return model, optimizer, scheduler, loop position, and RNG state.
        #
        # Method:
        # `model_state, optimizer_state = get_state_dict(
        #     self.model,
        #     self.optimizer,
        # )`
        # This helper understands FSDP2 DTensors and creates canonical state
        # dictionaries. Do not call the model's normal `state_dict()` here.
        #
        # Output keys:
        # - `model`
        # - `optimizer`
        # - `scheduler`
        # - `next_epoch` as a scalar int64 tensor
        # - `global_step` as a scalar int64 tensor
        # - `saved_checksum` as a scalar float64 tensor
        # - `torch_rng_state`
        # - `cuda_rng_state`
        # - `python_rng_state`
        #
        # Use `torch.tensor(value, dtype=torch.int64)` for the two counters,
        # `torch.tensor(value, dtype=torch.float64)` for the checksum,
        # `self.scheduler.state_dict()`, `torch.get_rng_state()`,
        # `torch.cuda.get_rng_state()`, and `random.getstate()`.
        raise NotImplementedError("Return every piece needed for an exact resume.")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Where this fits:
        # DCP fills `state_dict` from checkpoint files, then calls this method to
        # put those values back into the live model and optimizer.
        #
        # Method and full call:
        # `set_state_dict(
        #     self.model,
        #     self.optimizer,
        #     model_state_dict=state_dict["model"],
        #     optim_state_dict=state_dict["optimizer"],
        # )`
        #
        # Steps:
        # 1. Call `set_state_dict` with the exact keyword arguments above.
        # 2. Load the scheduler with `self.scheduler.load_state_dict(...)`.
        # 3. Copy scalar counters with `.item()` and `int(...)`. Copy the saved
        #    checksum with `.item()` and `float(...)`.
        # 4. Restore CPU PyTorch, CUDA, and Python RNG states with
        #    `torch.set_rng_state(...)`, `torch.cuda.set_rng_state(...)`, and
        #    `random.setstate(...)`.
        #
        # Check:
        # The optimizer still refers to the live sharded model parameters. Do not
        # replace the optimizer object during load.
        raise NotImplementedError("Restore all live objects and loop counters.")


def shard_vit(model: nn.Module, world_size: int) -> nn.Module:
    """Provided automatic FSDP2 block layout from exercise 2."""

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    for block in get_vit_blocks(model):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def save_and_report_checkpoint(
    state: TrainingState,
    metrics: dict[str, float | int],
) -> None:
    """Save local DCP shards and hand the directory to Ray Train."""

    # Goal:
    # Make every worker join the distributed save and report its checkpoint.
    #
    # Method and full calls:
    # `with tempfile.TemporaryDirectory() as checkpoint_dir:`
    # `dcp.save(
    #     state_dict={"training_state": state},
    #     checkpoint_id=checkpoint_dir,
    # )`
    # `checkpoint = Checkpoint.from_directory(checkpoint_dir)`
    # `train.report(metrics, checkpoint=checkpoint)`
    #
    # Important:
    # Do not put DCP save behind `if rank == 0`. Every worker owns different
    # shards and must join the call.
    #
    # Check:
    # The returned Ray result should have `result.checkpoint is not None`.
    raise NotImplementedError("Save all shards with DCP and report through Ray.")


def restore_if_available(state: TrainingState) -> bool:
    """Load Ray's latest checkpoint into already-created live objects."""

    # Goal:
    # Return False for a new run. Restore and return True for a resumed run.
    #
    # Method and full calls:
    # `checkpoint = train.get_checkpoint()`
    # `with checkpoint.as_directory() as checkpoint_dir:`
    # `dcp.load(
    #     state_dict={"training_state": state},
    #     checkpoint_id=checkpoint_dir,
    # )`
    #
    # Build and shard the model and create the optimizer before this function is
    # called. DCP needs those live objects so it knows the new local shard shapes.
    raise NotImplementedError("Restore the latest checkpoint when Ray provides one.")


def train_loop_per_worker(train_loop_config: dict[str, object]) -> None:
    """Train, checkpoint, and resume one sharded ViT."""

    # Goal:
    # Connect the complete FSDP2 loop to `TrainingState` and DCP.
    #
    # Steps:
    # 1. Rebuild config and loaders. Before passing the validation loader to Ray,
    #    read `fixed_image, _ = validation_loader.dataset[0]`. Validation
    #    transforms are deterministic, so every rank gets the same real image.
    #    Then prepare the distributed samplers. Get Ray's device and call
    #    `torch.cuda.set_device(device)` before building the mesh.
    # 2. Build and shard the model, then create AdamW.
    # 3. Create `torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, T_max=config.epochs)`.
    # 4. Create `LoopPosition()` and `TrainingState(...)`.
    # 5. Call `restore_if_available(state)` before the first train step. Save the
    #    returned boolean as `restored`.
    # 6. Start the epoch loop at `position.next_epoch`.
    # 7. After every optimizer step, add 1 to `position.global_step`.
    # 8. After an epoch, call `scheduler.step()` and set
    #    `position.next_epoch = epoch + 1` before saving.
    # 9. Use `fixed_image` to calculate
    #    `logit_checksum = float(model(fixed_image.unsqueeze(0).to(device))
    #    .float().sum().item())` under `torch.no_grad()`. Every rank should use
    #    that same real image rather than a distributed validation batch.
    # 10. Immediately after restore and before any optimizer step, compare this
    #     value with `position.saved_checksum` using an absolute tolerance such
    #     as `1e-4`. Report a `restore_checksum_matches` boolean.
    # 11. At the end of the epoch, calculate the checksum again and store it in
    #     `position.saved_checksum` before saving.
    # 12. Call `save_and_report_checkpoint(state, metrics)` from every worker.
    #     Report exactly these keys:
    #     - `next_epoch`, `global_step`, and `starting_step`
    #     - `restored`
    #     - `saved_checksum`
    #     - `restore_checksum_matches`
    #
    # Check:
    # A resumed job's starting step must equal the saved step. Its first checksum
    # before any new optimizer step should match the saved checksum within the
    # stated tolerance.
    raise NotImplementedError("Write the checkpoint-aware distributed train loop.")


def build_trainer(
    config: DistributedViTConfig,
    *,
    resume_from: Checkpoint | None = None,
    storage_path: str | None = None,
) -> TorchTrainer:
    """Build a two-worker trainer for a new or resumed run."""

    # Goal:
    # Return a two-GPU `TorchTrainer`.
    #
    # Method:
    # Use the same train function, config dictionary, and
    # `ScalingConfig(num_workers=2, use_gpu=True)` as earlier exercises. Pass
    # `RunConfig(name="fsdp2_vit_checkpointing", storage_path=storage_path)` so
    # Ray knows where durable checkpoint files belong.
    # Pass `resume_from_checkpoint=resume_from` to `TorchTrainer`. The worker
    # function can then read it with `train.get_checkpoint()`.
    #
    # Do not change worker count in this exercise. Exercise 6 tests that.
    raise NotImplementedError("Create the new or resumed TorchTrainer.")


def main() -> None:
    config = DistributedViTConfig(epochs=2)
    build_data_loaders(config, download=True)
    storage_path = str(Path(config.data_root).parent / "ray_results")

    first_result = build_trainer(config, storage_path=storage_path).fit()
    if first_result.checkpoint is None:
        raise AssertionError("The first job did not return a checkpoint.")
    print("first run:", first_result.metrics)

    resumed_config = DistributedViTConfig(epochs=3)
    resumed_result = build_trainer(
        resumed_config,
        resume_from=first_result.checkpoint,
        storage_path=storage_path,
    ).fit()
    if not bool(resumed_result.metrics.get("restored", False)):
        raise AssertionError("The second job did not report a checkpoint restore.")
    if not bool(resumed_result.metrics.get("restore_checksum_matches", False)):
        raise AssertionError("Model output changed while saving and restoring.")
    if int(resumed_result.metrics["starting_step"]) != int(
        first_result.metrics["global_step"]
    ):
        raise AssertionError("The resumed global step does not match the saved step.")
    print("resumed run:", resumed_result.metrics)


if __name__ == "__main__":
    main()
