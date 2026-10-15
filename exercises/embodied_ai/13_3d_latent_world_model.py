"""
============================================================
EXERCISE 13: 3D LATENT WORLD MODEL
============================================================

WHAT YOU ARE BUILDING
You are training a world model for first-person 3D navigation.

The first version predicts a latent state instead of full future images:

```text
image -> latent
latent + action + instruction -> next latent
next latent -> reward, cost, done
```

In this file, `latent` means a smaller vector made from an image.
For example, a full image might be `[3, 64, 64]`, but the latent might be `[64]`.
The latent is easier for a model to predict than every pixel.

WHY THIS EXERCISE EXISTS
Full 3D image prediction is hard. A latent world model is a smaller first step:
the model predicts a useful hidden vector and task signals.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
large video/world model             small CNN latent model
first-person game frames            local 3D frames from the related exercise
long text goals                     tokenized short instruction
many reward and safety signals      reward, cost, done heads
long planning rollouts              short latent rollout drift check

DATASET AND DOWNLOADS
Default dataset:
- no download
- load `outputs/mini3d_vla_dataset.pt` from the related exercise
- use `image`, `action`, `instruction`, `next_image`, `reward`, `cost`, and `done`

Optional later dataset:
- after the local file works, adapt Matterport3D / Room-to-Room or MiniWorld
  rollouts to the same VLA fields

INPUTS AND OUTPUTS
- images: `[batch_size, 3, image_size, image_size]`
- actions: `[batch_size]`
- instruction ids: `[batch_size, text_length]`
- predicted next latent: `[batch_size, latent_dim]`
- predicted reward: `[batch_size, 1]`
- predicted cost: `[batch_size, 1]`
- predicted done logit: `[batch_size, 1]`

LOSS FUNCTION
Use a sum of small losses:

```text
total_loss =
    next_latent_mse
  + reward_mse
  + cost_mse
  + done_bce
```

Plain meaning:
- `next_latent_mse` teaches what visual state comes next
- `reward_mse` teaches task progress
- `cost_mse` teaches risk
- `done_bce` teaches when the episode ends

METRICS TO TRACK
- next latent MSE: lower is better
- reward MSE: lower is better
- cost MSE: lower is better
- done accuracy: higher is better
- latent rollout drift: lower is better

VISUALS TO CREATE
Save `outputs/mini3d_world_model/latent_rollout_error.png`.
The x-axis should be rollout step.
The y-axis should be latent prediction error.

DEFINITION OF DONE
You are done when:
- saved 3D rollouts are loaded
- images are encoded into latents
- next-latent prediction trains
- reward, cost, and done errors are reported
- rollout drift is measured in latent space
- a rollout-drift plot is saved
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn


class LatentWorldModelMetrics(TypedDict):
    """Metrics for the 3D latent world model."""

    next_latent_mse: float
    reward_mse: float
    cost_mse: float
    done_accuracy: float
    latent_rollout_drift: float


@dataclass
class LatentWorldModel3DConfig:
    """Settings for the 3D latent world model."""

    image_size: int = 64
    channels: int = 3
    num_actions: int = 6
    latent_dim: int = 64
    text_dim: int = 32
    hidden_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    rollout_horizon: int = 8
    random_state: int = 42


class LatentWorldModel3D(nn.Module):
    """Predict next latent state and task signals."""

    def __init__(self, config: LatentWorldModel3DConfig) -> None:
        # Where this fits:
        # This model is the part the planner will use in the related exercise to
        # imagine futures before acting.
        #
        # Goal:
        # Build:
        # - image encoder
        # - action embedding
        # - text or instruction embedding
        # - next-latent head
        # - reward, cost, and done heads
        #
        # Suggested structure:
        # - `self.image_encoder`: CNN ending in `nn.Linear(..., config.latent_dim)`
        # - `self.action_embedding = nn.Embedding(config.num_actions, config.text_dim)`
        # - `self.text_embedding = nn.Embedding(vocab_size, config.text_dim)`
        # - concatenate image latent, action embedding, and text vector
        # - shared MLP:
        #   `nn.Linear(..., config.hidden_dim)`, `nn.ReLU()`
        # - heads:
        #   `next_latent_head`, `reward_head`, `cost_head`, `done_head`
        super().__init__()
        _ = config
        raise NotImplementedError("Build the 3D latent world model.")

    def forward(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        instruction_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return next latent, reward, cost, and done logits."""
        # Goal:
        # Predict what happens after taking `actions` from `images`.
        #
        # Steps:
        # 1. Encode `images` to current latents.
        # 2. Embed `actions`.
        # 3. Embed and average `instruction_ids`.
        # 4. Concatenate all three vectors.
        # 5. Run the shared MLP.
        # 6. Return next latent, reward, cost, and done logit.
        #
        # Checks:
        # - Do not apply sigmoid to done inside `forward`.
        #   Use `nn.functional.binary_cross_entropy_with_logits(...)` in the loss.
        _ = (images, actions, instruction_ids)
        raise NotImplementedError("Run the 3D latent world model.")


def train_3d_latent_world_model(config: LatentWorldModel3DConfig) -> LatentWorldModelMetrics:
    """Train and evaluate the 3D latent world model."""
    # Goal:
    # Use saved 3D VLA rollouts from the related exercise.
    #
    # Method:
    # 1. Load and flatten `outputs/mini3d_vla_dataset.pt`.
    # 2. Build tensors for image, next image, action, instruction ids, reward, cost, and done.
    # 3. Use the model's image encoder on `next_image` to create the target next latent.
    # 4. Train the model with the total loss shown at the top of the file.
    # 5. Evaluate on held-out episodes.
    # 6. Run a multi-step latent rollout using logged actions.
    # 7. Save a rollout-drift plot.
    #
    # Checks:
    # - Report reward, cost, and done separately.
    # - Measure multi-step latent rollout drift, not only one-step loss.
    raise NotImplementedError("Train and evaluate the 3D latent world model.")


def main() -> None:
    """Run the 3D latent world-model scaffold."""
    config = LatentWorldModel3DConfig()
    try:
        print(train_3d_latent_world_model(config))
    except NotImplementedError as error:
        print(f"[TODO] train_3d_latent_world_model: {error}")


if __name__ == "__main__":
    main()
