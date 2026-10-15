"""
============================================================
EXERCISE 07: MULTIMODAL ROBOT POLICY
============================================================

WHAT YOU ARE BUILDING
You are training a policy that uses image observations and language instructions.

Plain-language meanings:
- multimodal means the model uses more than one input type
- image encoder turns pixels into a vector
- text encoder turns an instruction into a vector
- action head predicts the action to take
- behavior cloning means learning to copy example actions

WHY THIS EXERCISE EXISTS
Robots and embodied agents often need to follow natural language goals using visual input.
This file connects images, instructions, and actions into one policy.

The policy should train on real logged rollouts from the shared visual grid-world.
This is behavior cloning: the model learns to copy useful example actions.
For the flagship project, this becomes the main reactive baseline:

```text
image + instruction -> action
```

Later, the world-model planner should beat or improve on this baseline.

DATASET OPTIONS
Default: use logged rollouts from `common.embodied_gridworld`.

Exact local data method:
1. Build `VisualGridWorldConfig(num_episodes=config.num_episodes, max_steps=config.max_steps, random_state=config.random_state)`.
2. Call `collect_visual_gridworld_episodes(grid_config)`.
3. Call `flatten_logged_episodes(episodes)`.
4. Use `step["image"]`, `step["instruction"]`, and `step["action"]`.

Real dataset options after the default works:
- `lerobot/pusht` on Hugging Face for image or video observations and actions.
- `lerobot/aloha_sim_transfer_cube_human` on Hugging Face for simulated robot demonstrations.
- `lerobot/aloha_sim_insertion_human` on Hugging Face to test a second task.
- Matterport3D Simulator / Room-to-Room later for first-person navigation with natural-language instructions.

If a robot dataset has no instruction text column, add one fixed instruction for
the task before tokenization. Example: `"push the T block to the target"`.

MAIN LOOP IN PLAIN WORDS

```text
image -> image encoder -> image vector
instruction -> text encoder -> text vector
combine vectors
action head predicts the action
compare predicted action to example action
train with cross-entropy loss
```

INPUTS AND OUTPUTS
- image: `[batch_size, channels, image_size, image_size]`
- instruction ids: `[batch_size, max_instruction_length]`
- action labels: `[batch_size]`
- action logits: `[batch_size, num_actions]`

LOSS FUNCTION
Use cross-entropy:

```text
loss = nn.functional.cross_entropy(logits, actions)
```

METRICS TO TRACK
- training loss: lower is better
- held-out action accuracy: higher is better
- random-action accuracy: about `1 / num_actions`
- rollout success rate: higher is better after you add online evaluation

VISUALS TO CREATE
Save `outputs/multimodal_policy/action_accuracy.png`.
It should compare random-action accuracy and validation action accuracy.

DEFINITION OF DONE
You are done when:
- images are encoded
- instructions are tokenized and encoded
- image and text features are combined
- the policy predicts actions
- behavior-cloning training runs on logged rollout data
- held-out scenes or instructions are tested
- accuracy is compared with a random-action baseline
- an action-accuracy chart is saved
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
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


@dataclass
class MultimodalPolicyConfig:
    """Settings for the tiny multimodal policy."""

    image_size: int = 28
    channels: int = 3
    vocab_size: int = 64
    max_instruction_length: int = 8
    text_dim: int = 32
    image_dim: int = 64
    hidden_dim: int = 64
    num_actions: int = 4
    num_episodes: int = 96
    max_steps: int = 18
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    random_state: int = 42


def build_instruction_vocab(instructions: list[str]) -> dict[str, int]:
    """Build a tiny word vocabulary."""
    # Goal:
    # Map instruction words to integer ids.
    #
    # Input:
    # - list of instruction strings.
    #
    # Output:
    # - dictionary mapping word to id.
    # - include `"<pad>"` with id `0`.
    # - include `"<unk>"` for unknown words.
    #
    # Method:
    # Lowercase each instruction and split on spaces.
    #
    # Checks:
    # - `vocab["<pad>"]` should be `0`.
    raise NotImplementedError("Build instruction vocabulary.")


def tokenize_instructions(
    instructions: list[str],
    vocab: dict[str, int],
    config: MultimodalPolicyConfig,
) -> torch.Tensor:
    """Turn instructions into token id tensors."""
    # Goal:
    # Convert a list of strings into a padded integer tensor.
    #
    # Input:
    # - `instructions`: list[str]
    # - `vocab`: word-to-id dictionary
    #
    # Output:
    # - tensor shaped `[batch_size, max_instruction_length]`
    #
    # Method:
    # 1. Lowercase and split each instruction.
    # 2. Convert each word to an id.
    # 3. Use `vocab["<unk>"]` for missing words.
    # 4. Pad or cut to `config.max_instruction_length`.
    #
    # Checks:
    # - Output dtype should be `torch.long`.
    raise NotImplementedError("Tokenize instruction strings.")


class MultimodalRobotPolicy(nn.Module):
    """Predict actions from images and text instructions."""

    def __init__(self, config: MultimodalPolicyConfig) -> None:
        # Goal:
        # Build image, text, and action parts.
        #
        # Image path:
        # - small CNN or flatten plus Linear
        #
        # Text path:
        # - `nn.Embedding(config.vocab_size, config.text_dim)`
        # - mean over token embeddings, ignoring padding in a later improvement
        #
        # Combined path:
        # - concatenate image vector and text vector
        # - `Linear -> ReLU -> Linear(num_actions)`
        super().__init__()
        raise NotImplementedError("Build multimodal robot policy.")

    def forward(self, images: torch.Tensor, instruction_ids: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        # Goal:
        # Predict raw action scores from image and instruction.
        #
        # Input:
        # - `images`: `[batch_size, channels, height, width]`
        # - `instruction_ids`: `[batch_size, max_instruction_length]`
        #
        # Output:
        # - logits shaped `[batch_size, num_actions]`
        #
        # Method:
        # 1. Encode images.
        # 2. Embed instruction ids.
        # 3. Average token embeddings over the sequence.
        # 4. Concatenate image and text vectors.
        # 5. Return action logits.
        #
        # Checks:
        # - Do not apply softmax before cross-entropy.
        raise NotImplementedError("Run multimodal robot policy.")


def behavior_cloning_loss(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return action imitation loss."""
    # Goal:
    # Train the policy to copy example actions.
    #
    # Input:
    # - `logits`: `[batch_size, num_actions]`
    # - `actions`: `[batch_size]`
    #
    # Method:
    # Use `nn.functional.cross_entropy(logits, actions)`.
    raise NotImplementedError("Compute behavior cloning loss.")


def build_behavior_cloning_data(
    config: MultimodalPolicyConfig,
) -> tuple[TensorDataset, list[str], dict[str, int]]:
    """Build image, instruction, and action data from logged rollouts."""
    # Goal:
    # Turn logged visual grid-world rollouts into tensors for behavior cloning.
    #
    # Output:
    # - `dataset`: TensorDataset with:
    #   - images: `[num_steps, channels, image_size, image_size]`
    #   - instruction ids: `[num_steps, max_instruction_length]`
    #   - actions: `[num_steps]`
    # - `instructions`: the raw instruction string for each step
    # - `vocab`: word-to-id dictionary
    #
    # Method:
    # 1. Build `grid_config = VisualGridWorldConfig(...)`.
    # 2. Collect episodes with `collect_visual_gridworld_episodes(grid_config)`.
    # 3. Flatten episodes with `flatten_logged_episodes(episodes)`.
    # 4. Stack `step["image"]`.
    # 5. Build a vocab from all `step["instruction"]` strings.
    # 6. Tokenize the instructions with `tokenize_instructions(...)`.
    # 7. Turn `step["action"]` into a `torch.long` tensor.
    # 8. Return `TensorDataset(images, instruction_ids, actions)`, raw instructions, and vocab.
    #
    # Checks:
    # - Do not use random labels.
    # - Actions should be the actions chosen during environment data collection.
    _ = (VisualGridWorldConfig, collect_visual_gridworld_episodes, flatten_logged_episodes)
    raise NotImplementedError("Build behavior-cloning data from logged rollouts.")


def train_multimodal_policy(config: MultimodalPolicyConfig) -> dict[str, float]:
    """Train and evaluate the image-instruction policy."""
    # Goal:
    # Train the policy to copy logged actions, then evaluate it on held-out data.
    #
    # Output:
    # - dictionary with:
    #   - `"train_loss"`
    #   - `"val_action_accuracy"`
    #   - `"random_action_accuracy"`
    #
    # Method:
    # 1. Call `build_behavior_cloning_data(config)`.
    # 2. Split the dataset into train and validation parts.
    # 3. Use `DataLoader(..., batch_size=config.batch_size, shuffle=True)` for training.
    # 4. Build `MultimodalRobotPolicy(config)`.
    # 5. Train with `torch.optim.AdamW(...)` and `behavior_cloning_loss(...)`.
    # 6. On validation data, compute action accuracy:
    #    `predicted_actions = logits.argmax(dim=-1)`.
    # 7. Compare with random guessing, which is about `1 / config.num_actions`.
    #
    # Checks:
    # - Do not apply softmax before `cross_entropy`.
    # - The policy should beat random guessing before it is useful as a baseline.
    _ = (DataLoader,)
    raise NotImplementedError("Train and evaluate multimodal behavior cloning.")


def explain_multimodal_policy() -> str:
    """Return a short multimodal policy explanation."""
    return "A multimodal policy uses both what the agent sees and what the instruction says before choosing an action."


def main() -> None:
    """Run the full multimodal behavior-cloning scaffold."""
    config = MultimodalPolicyConfig()
    try:
        metrics = train_multimodal_policy(config)
    except NotImplementedError as error:
        print(f"[TODO] train_multimodal_policy: {error}")
        return
    print(metrics)


if __name__ == "__main__":
    main()
