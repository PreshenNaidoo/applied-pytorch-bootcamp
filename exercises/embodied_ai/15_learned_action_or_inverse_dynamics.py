"""
============================================================
EXERCISE 15: LEARNED ACTION OR INVERSE DYNAMICS
============================================================

WHAT YOU ARE BUILDING
You are training a model that guesses the action between two frames.

The model learns:

```text
frame_t + frame_t+1 -> action_t
```

This is called inverse dynamics.
Normal dynamics predicts the next state from the current state and action.
Inverse dynamics does the reverse: it looks at two states and guesses the action
that likely caused the change.

WHY THIS EXERCISE EXISTS
Some interactive world-model systems learn from video where actions are missing
or weakly labeled. A small inverse-dynamics model can infer likely actions from
visual change.

DATASET AND DOWNLOADS
Default dataset:
- no download
- use frame pairs and action labels from `common.embodied_gridworld`
- later use 3D rollouts from `outputs/mini3d_vla_dataset.pt`

INPUTS AND OUTPUTS
- current frames: `[batch_size, 3, image_size, image_size]`
- next frames: `[batch_size, 3, image_size, image_size]`
- action labels: `[batch_size]`
- action logits: `[batch_size, num_actions]`

LOSS FUNCTION
Use cross-entropy:

```text
loss = nn.functional.cross_entropy(action_logits, action_labels)
```

METRICS TO TRACK
- inverse action accuracy: higher is better
- random action accuracy: about `1 / num_actions`
- confusion matrix: shows which actions are mixed up

VISUALS TO CREATE
Save `outputs/inverse_dynamics/confusion_matrix.png`.
Rows should be true actions.
Columns should be predicted actions.

DEFINITION OF DONE
You are done when:
- frame pairs come from real rollouts
- inverse action accuracy is reported
- a confusion matrix is saved or printed
- ambiguous examples are shown
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn


class InverseDynamicsMetrics(TypedDict):
    """Metrics for inverse dynamics."""

    action_accuracy: float
    random_action_accuracy: float
    confusion_matrix_path: str


@dataclass
class InverseDynamicsConfig:
    """Settings for inverse dynamics."""

    image_size: int = 64
    channels: int = 3
    num_actions: int = 6
    hidden_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5


class InverseDynamicsModel(nn.Module):
    """Predict which action moved one frame to the next."""

    def __init__(self, config: InverseDynamicsConfig) -> None:
        # Where this fits:
        # This model can create action labels when video has frames but no
        # action column. In this exercise, actions are known, so you can measure
        # whether the guesses are correct.
        #
        # Suggested model:
        # - concatenate current and next frames on the channel dimension
        # - input shape becomes `[batch_size, 2 * channels, image_size, image_size]`
        # - use a small CNN and a final `nn.Linear(..., num_actions)`
        super().__init__()
        _ = config
        raise NotImplementedError("Build the inverse-dynamics model.")

    def forward(self, current_frames: torch.Tensor, next_frames: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        # Steps:
        # 1. Concatenate frames with `torch.cat([current_frames, next_frames], dim=1)`.
        # 2. Run the CNN.
        # 3. Return logits shaped `[batch_size, num_actions]`.
        #
        # Check:
        # - Do not apply softmax before cross-entropy.
        _ = (current_frames, next_frames)
        raise NotImplementedError("Predict inverse-dynamics action logits.")


def train_inverse_dynamics_model(config: InverseDynamicsConfig) -> InverseDynamicsMetrics:
    """Train and evaluate inverse dynamics from rollout frame pairs."""
    # Check:
    # Some actions may be visually ambiguous, such as turning in a plain hallway.
    #
    # Method:
    # 1. Build frame pairs from real logged rollouts.
    # 2. Split into train and validation sets.
    # 3. Train with `nn.functional.cross_entropy(...)`.
    # 4. Compute validation action accuracy.
    # 5. Build a confusion matrix shaped `[num_actions, num_actions]`.
    # 6. Save the confusion matrix as a plot.
    raise NotImplementedError("Train and evaluate inverse dynamics.")


def main() -> None:
    """Run the inverse-dynamics scaffold."""
    config = InverseDynamicsConfig()
    try:
        print(train_inverse_dynamics_model(config))
    except NotImplementedError as error:
        print(f"[TODO] train_inverse_dynamics_model: {error}")


if __name__ == "__main__":
    main()
