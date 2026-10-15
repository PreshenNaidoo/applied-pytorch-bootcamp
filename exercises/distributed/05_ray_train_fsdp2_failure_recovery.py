"""
============================================================
EXERCISE 5: RAY TRAIN FSDP2 FAILURE RECOVERY
============================================================

WHAT YOU ARE BUILDING
You are running a sharded ViT job that deliberately crashes one worker once.
Ray restarts the worker group, and every worker restores the latest distributed
checkpoint before training continues.

WHY THIS EXERCISE EXISTS
Long jobs eventually meet a failed process, lost machine, or cloud preemption.
Automatic worker restart is useful only when the training function knows how to
restore its state. Otherwise the job starts from the beginning and loses work.

THE SAFE FAILURE TEST
The planned failure happens only after a checkpoint has been saved. That
checkpoint contains `failure_already_injected=True`. The restarted job reads
that value, so it does not enter an endless crash-and-restart loop.

FULL EVENT ORDER
1. Train and save a normal checkpoint.
2. Set the durable failure flag to true.
3. Save another checkpoint containing the flag.
4. Rank 0 raises one planned error.
5. Ray sees the worker failure and restarts the whole worker group.
6. `train.get_checkpoint()` returns the latest checkpoint.
7. DCP restores model, optimizer, counters, and the failure flag.
8. Training continues from the next unsaved step.

WHAT RAY AND DCP EACH DO
- Ray restarts processes and supplies the latest checkpoint.
- DCP restores the correct local model and optimizer shards.
- your loop restores epoch and step counters and avoids repeating saved work.

METRICS TO TRACK
- `global_step`: must continue rather than return to zero
- `restart_count`: should finish at 1 in this controlled test
- `failure_already_injected`: should be true after recovery
- loss: should remain finite after the restart

COMMON MISTAKES
- keeping the failure flag in a normal Python variable that disappears on restart
- raising the planned error before saving the true failure flag
- assuming Ray restart also restores model state without code in the worker loop
- setting `max_failures=0`, which disables worker-group recovery
- letting different ranks make different checkpoint calls
- treating a retry as successful without checking the resumed global step

STEP-BY-STEP TODO
1. `should_inject_failure`
2. `train_loop_per_worker`
3. `build_fault_tolerant_trainer`

DEFINITION OF DONE
- the first worker group fails once after a durable checkpoint
- Ray starts a replacement worker group
- the same planned failure does not repeat
- the final global step is beyond the step saved before the crash
- all ViT blocks remain automatically sharded after restore
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from ray import train
from ray.train import Checkpoint, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, prepare_data_loader
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from _shared_vit import DistributedViTConfig, build_data_loaders, build_vit, get_vit_blocks


@dataclass
class RecoveryPosition:
    next_epoch: int = 0
    global_step: int = 0
    failure_already_injected: bool = False


class RecoveryState(Stateful):
    """Provided DCP wrapper so this exercise can focus on failure recovery."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        position: RecoveryPosition,
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
            "failure_already_injected": torch.tensor(
                self.position.failure_already_injected,
                dtype=torch.bool,
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
        self.position.failure_already_injected = bool(
            state_dict["failure_already_injected"].item()
        )


def shard_vit(model: nn.Module, world_size: int) -> nn.Module:
    """Provided automatic block discovery and FSDP2 layout."""

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    for block in get_vit_blocks(model):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def save_recovery_checkpoint(
    state: RecoveryState,
    metrics: dict[str, float | int | bool],
) -> None:
    """Provided all-rank DCP save and Ray report from exercise 4."""

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        dcp.save(
            state_dict={"recovery_state": state},
            checkpoint_id=checkpoint_dir,
        )
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(metrics, checkpoint=checkpoint)


def restore_recovery_checkpoint(state: RecoveryState) -> bool:
    """Provided restore path from exercise 4."""

    checkpoint = train.get_checkpoint()
    if checkpoint is None:
        return False
    with checkpoint.as_directory() as checkpoint_dir:
        dcp.load(
            state_dict={"recovery_state": state},
            checkpoint_id=checkpoint_dir,
        )
    return True


def should_inject_failure(
    position: RecoveryPosition,
    *,
    current_epoch: int,
    failure_epoch: int,
) -> bool:
    """Decide whether this is the one safe point for the planned failure."""

    # Where this fits:
    # Every restarted worker runs the Python function from the top. This check
    # must depend on checkpointed state, not a normal local variable.
    #
    # Goal:
    # Return true only when:
    # - current_epoch equals failure_epoch
    # - `position.failure_already_injected` is false
    #
    # Input example:
    # current_epoch=1, failure_epoch=1, flag=False -> True
    # current_epoch=1, failure_epoch=1, flag=True  -> False after restart
    #
    # Check:
    # Calling this again after setting the flag to true must return false.
    raise NotImplementedError("Use the durable flag to allow one failure only.")


def train_loop_per_worker(train_loop_config: dict[str, object]) -> None:
    """Train the sharded ViT and recover from one planned rank-0 error."""

    # Goal:
    # Join automatic FSDP2 sharding, DCP restore, and Ray worker restart.
    #
    # Steps:
    # 1. Rebuild the config and read `failure_epoch`.
    # 2. Build and prepare the loader. Get Ray's CUDA device and call
    #    `torch.cuda.set_device(device)`. Build the ViT and shard it before
    #    making AdamW.
    # 3. Create `RecoveryPosition` and `RecoveryState`, then call
    #    `restore_recovery_checkpoint(state)`.
    # 4. Set `restart_count = 1 if restored else 0`. This controlled exercise has
    #    at most one restart.
    # 5. Start at `position.next_epoch` and train normal optimizer steps. Add one
    #    to `position.global_step` after each completed step.
    # 6. After an epoch, set `position.next_epoch = epoch + 1`.
    # 7. If `should_inject_failure(...)` is true, first set the durable flag to
    #    true and call `save_recovery_checkpoint(...)` from every rank.
    # 8. After that report returns, rank 0 raises
    #    `RuntimeError("planned worker failure")`. Other ranks must not raise it.
    # 9. After recovery, continue training and checkpoint normally each epoch.
    #    Report exactly these keys: `epoch`, `global_step`, `restart_count`,
    #    `failure_already_injected`, and `loss`.
    #
    # Important:
    # Never raise before the checkpoint with the true flag is safely reported.
    #
    # Check:
    # Final metrics show `restart_count=1` and `failure_already_injected=True`.
    raise NotImplementedError("Write the one-failure recovery training loop.")


def build_fault_tolerant_trainer(
    config: DistributedViTConfig,
    storage_path: str,
) -> TorchTrainer:
    """Allow Ray to replace the worker group once."""

    # Goal:
    # Return a two-GPU trainer that tolerates exactly one worker-group failure.
    #
    # Method and full objects:
    # `run_config = RunConfig(
    #     name="fsdp2_failure_recovery",
    #     storage_path=storage_path,
    #     failure_config=FailureConfig(max_failures=1),
    # )`
    # Then pass this run_config to `TorchTrainer(...)` with
    # `ScalingConfig(num_workers=2, use_gpu=True)` and a loop config containing
    # `failure_epoch=0` plus the ViT config values.
    #
    # `max_failures=1` means Ray may restart the worker group once. It does not
    # mean one worker is allowed to stay missing.
    #
    # Check:
    # `max_failures=0` should make the planned error end the job. Return it to 1
    # for the completed exercise.
    raise NotImplementedError("Create a trainer with one allowed recovery.")


def main() -> None:
    config = DistributedViTConfig(epochs=2)
    build_data_loaders(config, download=True)
    storage_path = str(Path(config.data_root).parent / "ray_results")
    result = build_fault_tolerant_trainer(config, storage_path).fit()
    if int(result.metrics.get("restart_count", 0)) != 1:
        raise AssertionError("The controlled run should recover exactly once.")
    if not bool(result.metrics.get("failure_already_injected", False)):
        raise AssertionError("The restored checkpoint lost the durable failure flag.")
    if int(result.metrics.get("global_step", 0)) <= 0:
        raise AssertionError("Training did not continue after recovery.")
    print(result.metrics)


if __name__ == "__main__":
    main()
