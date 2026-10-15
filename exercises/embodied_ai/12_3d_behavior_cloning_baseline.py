"""
============================================================
EXERCISE 12: 3D BEHAVIOR CLONING BASELINE
============================================================

WHAT YOU ARE BUILDING
You are training a first-person policy that copies logged 3D actions.

The model learns:

```text
camera image + instruction -> keyboard-like action
```

In this file, `policy` means the model that chooses an action.
Behavior cloning means the policy learns by copying saved example actions.
It does not learn from reward directly.

WHY THIS EXERCISE EXISTS
Before adding a world model and planner, you need a simple reactive baseline.
This tells you whether vision and language alone are enough for the task.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
large multi-world action model      small CNN plus text embedding
large 3D game data                  local 3D rollouts from the related exercise
long natural language goals         short instruction strings
keyboard and mouse controls         6 small action ids
online task evaluation              rollout success in the related exercise's environment

DATASET AND DOWNLOADS
Default dataset:
- no download
- load `outputs/mini3d_vla_dataset.pt` created by the related exercise
- each row should already have image, instruction, action, episode id, step id, and split

Optional later dataset:
- swap in Matterport3D / Room-to-Room or MiniWorld rollouts after the local
  dataset works
- keep the same output tensors before changing model code

INPUTS AND OUTPUTS
- images: `[batch_size, 3, image_size, image_size]`
- instruction ids: `[batch_size, max_instruction_length]`
- action labels: `[batch_size]`
- model logits: `[batch_size, num_actions]`

LOSS FUNCTION
Use cross-entropy:

```text
loss = nn.functional.cross_entropy(logits, actions)
```

Plain meaning:
- `logits` are the model's raw scores for each action
- `actions` are the saved action ids from the dataset
- lower loss means the model is better at copying the saved actions

METRICS TO TRACK
- training loss: lower is better
- held-out action accuracy: higher is better
- random action accuracy: about `1 / num_actions`
- rollout success rate: higher is better
- unseen rollout success rate: higher is better

VISUALS TO CREATE
Save a small bar chart to `outputs/mini3d_bc/action_accuracy.png` comparing:
- random action accuracy
- behavior-cloning validation accuracy

DEFINITION OF DONE
You are done when:
- the model trains on saved 3D VLA rollouts
- held-out action accuracy is reported
- rollout success rate is reported
- seen-room and unseen-room results are separated
- random-action accuracy is used as a baseline
- a simple action-accuracy chart is saved
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Policy3DMetrics(TypedDict):
    """Metrics for a 3D behavior-cloning policy."""

    train_loss: float
    held_out_action_accuracy: float
    random_action_accuracy: float
    rollout_success_rate: float
    unseen_rollout_success_rate: float


@dataclass
class BehaviorCloning3DConfig:
    """Settings for the 3D behavior-cloning baseline."""

    image_size: int = 64
    channels: int = 3
    num_actions: int = 6
    max_instruction_length: int = 12
    vocab_size: int = 128
    hidden_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    random_state: int = 42


class FirstPersonBCPolicy(nn.Module):
    """Predict a 3D navigation action from image and instruction."""

    def __init__(self, config: BehaviorCloning3DConfig) -> None:
        # Where this fits:
        # This is the reactive baseline. It chooses from the current image and
        # instruction only. It does not imagine future steps.
        #
        # Goal:
        # Build an image encoder, a text encoder, and an action head.
        #
        # Suggested model:
        # - image path:
        #   `nn.Conv2d(3, 32, kernel_size=3, padding=1)`,
        #   `nn.ReLU()`, `nn.AdaptiveAvgPool2d((1, 1))`, `nn.Flatten()`
        # - text path:
        #   `nn.Embedding(config.vocab_size, config.hidden_dim)`
        #   then average token embeddings over the sequence
        # - action head:
        #   concatenate image and text vectors, then use
        #   `nn.Linear(..., config.num_actions)`
        #
        # Output:
        # - action logits shaped `[batch_size, num_actions]`
        super().__init__()
        _ = config
        raise NotImplementedError("Build the 3D behavior-cloning policy.")

    def forward(self, images: torch.Tensor, instruction_ids: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        # Goal:
        # Run the image and text through the model and return raw action scores.
        #
        # Steps:
        # 1. Encode `images` with the CNN.
        # 2. Embed `instruction_ids` with `nn.Embedding`.
        # 3. Average the token embeddings over the text length.
        # 4. Concatenate image and text features with `torch.cat(..., dim=-1)`.
        # 5. Return logits shaped `[batch_size, num_actions]`.
        #
        # Check:
        # - Do not apply softmax here. `cross_entropy` expects raw logits.
        _ = (images, instruction_ids)
        raise NotImplementedError("Run the 3D behavior-cloning policy.")


def build_3d_behavior_cloning_dataset(config: BehaviorCloning3DConfig) -> TensorDataset:
    """Load saved 3D VLA rollouts and return image, text, and action tensors."""
    # Goal:
    # Turn the related exercise data into training tensors.
    #
    # Method:
    # 1. Load `outputs/mini3d_vla_dataset.pt` with `torch.load(..., weights_only=False)`.
    # 2. Flatten episodes into step rows.
    # 3. Stack `step["image"]` into `[num_steps, 3, image_size, image_size]`.
    # 4. Build a word vocabulary from `step["instruction"]`.
    #    Use `"<pad>" = 0` and `"<unk>" = 1`.
    # 5. Tokenize each instruction to `[max_instruction_length]`.
    # 6. Turn `step["action"]` into a `torch.long` tensor shaped `[num_steps]`.
    # 7. Return `TensorDataset(images, instruction_ids, actions)`.
    #
    # Checks:
    # - Do not shuffle away episode ids before creating seen and unseen splits.
    # - Actions must come from logged rollouts, not random labels.
    _ = config
    raise NotImplementedError("Build 3D behavior-cloning tensors.")


def train_3d_behavior_cloning_policy(config: BehaviorCloning3DConfig) -> Policy3DMetrics:
    """Train and evaluate the 3D behavior-cloning baseline."""
    # Goal:
    # Train the reactive baseline and measure whether it beats random actions.
    #
    # Method:
    # 1. Build the dataset with `build_3d_behavior_cloning_dataset(config)`.
    # 2. Split into train and validation sets.
    # 3. Use `DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)`.
    # 4. Build `FirstPersonBCPolicy(config)`.
    # 5. Use `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 6. Use `nn.functional.cross_entropy(logits, actions)`.
    # 7. Report validation action accuracy.
    # 8. Then run the policy in the real 3D environment from the related exercise.
    #
    # Checks:
    # - Validation accuracy should beat random accuracy before the policy is useful.
    # - Rollout success can still be weak even when action accuracy is okay.
    _ = (DataLoader,)
    raise NotImplementedError("Train and evaluate the 3D behavior-cloning baseline.")


def main() -> None:
    """Run the 3D behavior-cloning scaffold."""
    config = BehaviorCloning3DConfig()
    try:
        print(train_3d_behavior_cloning_policy(config))
    except NotImplementedError as error:
        print(f"[TODO] train_3d_behavior_cloning_policy: {error}")


if __name__ == "__main__":
    main()
