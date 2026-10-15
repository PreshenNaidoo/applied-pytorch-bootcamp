"""
============================================================
EXERCISE 04: SENSOR FUSION MINI PROJECT
============================================================

WHAT YOU ARE BUILDING
You are building a tiny sensor-fusion pipeline.

Plain-language meanings:
- sensor fusion means combining several sensor signals into one better state estimate
- camera-like features give visual hints
- range features give distance hints
- odometry gives movement hints
- uncertainty means how much you should trust a signal

WHY THIS EXERCISE EXISTS
Embodied agents often use more than one sensor.
One sensor may be noisy, blocked, or missing.
Fusion helps the agent use the best parts of each signal.

The default data should come from a logged motion or grid-world rollout, not
random features with random labels. A small version is still useful if the
target state comes from the real simulated state.

DATASET OPTIONS
Default: use logged rollouts from `common.embodied_gridworld`.

Real dataset option after the default works:
- KITTI Odometry: combine image-derived features, pose changes, and optional LiDAR-style range features.

MAIN LOOP IN PLAIN WORDS

```text
read camera-like features
read range features
read odometry features
normalize each feature group
combine them into one vector
predict the state estimate
test what happens when one sensor gets noisy
```

INPUTS AND OUTPUTS
- camera features: `[batch_size, camera_dim]`
- range features: `[batch_size, range_dim]`
- odometry features: `[batch_size, odometry_dim]`
- fused state estimate: `[batch_size, state_dim]`

DEFINITION OF DONE
You are done when:
- three sensor feature groups are created from logged rollout data
- the feature groups are combined
- one state estimate is returned
- noisy-sensor behavior is tested
- held-out state error is reported
- you can explain why fusion can beat one sensor alone
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.embodied_gridworld import (
    VisualGridWorldConfig,
    collect_visual_gridworld_episodes,
    flatten_logged_episodes,
)


class SensorBatch(TypedDict):
    """One batch of simple sensor features."""

    camera: torch.Tensor
    range: torch.Tensor
    odometry: torch.Tensor
    target_state: torch.Tensor


@dataclass
class SensorFusionConfig:
    """Settings for the sensor fusion exercise."""

    batch_size: int = 32
    camera_dim: int = 8
    range_dim: int = 6
    odometry_dim: int = 4
    state_dim: int = 3
    hidden_dim: int = 64
    num_episodes: int = 64
    max_steps: int = 18
    learning_rate: float = 1e-3
    epochs: int = 5
    random_state: int = 42


def build_tiny_sensor_batch(config: SensorFusionConfig) -> SensorBatch:
    """Return one sensor batch from logged rollout data."""
    # Goal:
    # Build a small batch with three sensor groups and one target state.
    #
    # Input:
    # - `config`: feature sizes and batch size.
    #
    # Output:
    # - dictionary with:
    #   - `camera`: `[batch_size, camera_dim]`
    #   - `range`: `[batch_size, range_dim]`
    #   - `odometry`: `[batch_size, odometry_dim]`
    #   - `target_state`: `[batch_size, state_dim]`
    #
    # Method:
    # Use the shared visual grid-world log:
    # - camera-like features can be simple image summaries, such as channel means and target color strength
    # - range features can come from distances to goal, hazard, or walls
    # - odometry features can come from previous action and step index
    # - target_state should come from the real logged state
    #
    # Steps:
    # 1. Collect episodes with `collect_visual_gridworld_episodes(...)`.
    # 2. Flatten the logged steps.
    # 3. Build each feature group from real step fields.
    # 4. Use the first `config.state_dim` values from `step["state"]` as the target state.
    # 5. Return the dictionary.
    #
    # Checks:
    # - Every tensor should have the same batch size.
    # - Do not create random target states.
    _ = (VisualGridWorldConfig, collect_visual_gridworld_episodes, flatten_logged_episodes)
    raise NotImplementedError("Build a tiny sensor feature batch.")


class SensorFusionNet(nn.Module):
    """Fuse camera, range, and odometry features."""

    def __init__(self, config: SensorFusionConfig) -> None:
        # Goal:
        # Build a small network that combines all sensor features.
        #
        # Input to the network:
        # - camera: `[batch_size, camera_dim]`
        # - range: `[batch_size, range_dim]`
        # - odometry: `[batch_size, odometry_dim]`
        #
        # Output:
        # - state estimate shaped `[batch_size, state_dim]`
        #
        # Method:
        # Concatenate the three feature groups, then pass them through:
        # `Linear -> ReLU -> Linear`.
        #
        # Use this input size:
        # `config.camera_dim + config.range_dim + config.odometry_dim`
        super().__init__()
        raise NotImplementedError("Build the sensor fusion network.")

    def forward(self, camera: torch.Tensor, range_features: torch.Tensor, odometry: torch.Tensor) -> torch.Tensor:
        """Return the fused state estimate."""
        # Goal:
        # Combine the three sensor groups and predict one state vector.
        #
        # Method:
        # 1. Concatenate with:
        #    `features = torch.cat([camera, range_features, odometry], dim=1)`.
        # 2. Pass `features` through the network.
        # 3. Return the state estimate.
        #
        # Checks:
        # - Output shape should be `[batch_size, state_dim]`.
        raise NotImplementedError("Run the sensor fusion network.")


def add_sensor_noise(batch: SensorBatch, sensor_name: str, noise_std: float) -> SensorBatch:
    """Return a copy of the batch with noise added to one sensor."""
    # Goal:
    # Test what happens when one sensor becomes noisy.
    #
    # Input:
    # - `batch`: SensorBatch
    # - `sensor_name`: one of `"camera"`, `"range"`, or `"odometry"`
    # - `noise_std`: noise size, such as `0.5`
    #
    # Output:
    # - new SensorBatch with the same keys.
    #
    # Method:
    # Copy every tensor.
    # Add `torch.randn_like(tensor) * noise_std` only to the named sensor.
    #
    # Checks:
    # - The original batch should not be changed.
    raise NotImplementedError("Add noise to one sensor group.")


def train_sensor_fusion_model(config: SensorFusionConfig) -> dict[str, float]:
    """Train the fusion model and test noisy-sensor behavior."""
    # Goal:
    # Train the model to estimate state from logged sensor features.
    #
    # Output:
    # - dictionary with:
    #   - `"val_state_mse"`
    #   - `"noisy_camera_mse"`
    #   - `"noisy_range_mse"`
    #   - `"noisy_odometry_mse"`
    #
    # Method:
    # 1. Build a sensor batch from logged rollouts.
    # 2. Split it into train and validation tensors.
    # 3. Train `SensorFusionNet(config)` with `torch.optim.AdamW(...)`.
    # 4. Use `nn.functional.mse_loss(predicted_state, target_state)`.
    # 5. Add noise to one sensor group at a time.
    # 6. Report how validation error changes.
    #
    # Checks:
    # - A fused model should be more robust than depending on one sensor only.
    _ = (DataLoader, TensorDataset)
    raise NotImplementedError("Train and evaluate sensor fusion.")


def explain_sensor_fusion() -> str:
    """Return a short sensor-fusion explanation."""
    return "Sensor fusion combines several imperfect signals so the agent can make a better state estimate."


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
    """Run the full sensor-fusion training scaffold."""
    config = SensorFusionConfig()
    metrics, ok = report_task_progress("train_sensor_fusion_model", train_sensor_fusion_model, config)
    if ok:
        print(metrics)


if __name__ == "__main__":
    main()
