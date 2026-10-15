"""
============================================================
EXERCISE 06: VISION LANGUAGE ACTION DATASET FORMAT
============================================================

WHAT YOU ARE BUILDING
You are building a vision-language-action dataset from logged agent rollouts.

Plain-language meanings:
- vision means what the agent saw, usually an image
- language means the instruction or goal
- action means what the agent did
- trajectory means a sequence of steps from one episode

WHY THIS EXERCISE EXISTS
Many robotics and embodied AI datasets store what the agent saw, what it was told, and what it did.
This file teaches that data shape before you train a multimodal policy.

For this path, the default data comes from the shared visual grid-world.
The world is small, but each row is a real step collected from an environment:

```text
image + instruction + action -> reward, cost, next image, done
```

This is the same data shape used by generalist visual-action agents such as
SIMA, except the real systems use richer 3D images, longer instructions, and
more actions.

DATASET OPTIONS
Default: use logged rollouts from `common.embodied_gridworld`.

Exact local data method:
1. Build `VisualGridWorldConfig(image_size=config.image_size, num_episodes=config.num_trajectories, max_steps=config.steps_per_trajectory, random_state=config.random_state)`.
2. Call `collect_visual_gridworld_episodes(grid_config)`.
3. Convert each logged step into a `VLAStep`.
4. Add `episode_id`, `step_id`, and `split`.

Real dataset options after the default works:
- `lerobot/pusht` on Hugging Face for image or video, state, action, reward, done, and success fields.
- `lerobot/aloha_sim_transfer_cube_human` on Hugging Face for simulated robot video, state, and action demonstrations.
- `lerobot/aloha_sim_insertion_human` on Hugging Face as a second task to test whether the same data code generalizes.
- Matterport3D Simulator / Room-to-Room later for real indoor views and natural-language navigation instructions.

Some robot datasets do not contain a natural-language instruction per row.
For those, attach a fixed task string such as `"push the block to the goal"` or
`"transfer the cube"` before training the text path.

ONE STEP LOOKS LIKE THIS

```text
image observation
instruction = "go to the red cube"
action = move forward
reward = 0.0
next image observation
done = False
```

INPUTS AND OUTPUTS
- image: `[channels, height, width]`
- instruction: string
- action: int
- reward: float
- next image: `[channels, height, width]`
- done: bool
- episode_id: int
- step_id: int
- split: string such as `"train"` or `"val"`

METRICS TO TRACK
- number of trajectories
- number of total steps
- mean trajectory length
- percent of steps with `done=True`
- number of train, validation, and test rows

VISUALS TO CREATE
Save `outputs/vla_dataset/trajectory_preview.png`.
It should show a few ordered frames from one trajectory with action ids.

DEFINITION OF DONE
You are done when:
- trajectories are collected from a stepped environment
- steps are flattened into examples
- a dataloader can return image, text, action, reward, cost, next image, done, episode id, step id, and split
- a trajectory preview is saved
- you can explain why trajectory order matters
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict
import torch
from torch.utils.data import DataLoader, Dataset

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.embodied_gridworld import (
    VisualGridWorldConfig,
    collect_visual_gridworld_episodes,
)


class VLAStep(TypedDict):
    """One vision-language-action step."""

    image: torch.Tensor
    instruction: str
    action: int
    reward: float
    cost: float
    next_image: torch.Tensor
    done: bool
    episode_id: int
    step_id: int
    split: str


class VLABatch(TypedDict):
    """One batch of vision-language-action steps."""

    images: torch.Tensor
    instructions: list[str]
    actions: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    next_images: torch.Tensor
    dones: torch.Tensor
    episode_ids: torch.Tensor
    step_ids: torch.Tensor
    splits: list[str]


@dataclass
class VLADatasetConfig:
    """Settings for the tiny VLA dataset."""

    image_size: int = 28
    channels: int = 3
    num_actions: int = 4
    num_trajectories: int = 64
    steps_per_trajectory: int = 18
    batch_size: int = 4
    random_state: int = 42


def build_tiny_vla_trajectories(config: VLADatasetConfig) -> list[list[VLAStep]]:
    """Return VLA trajectories collected from the visual grid-world."""
    # Goal:
    # Create trajectories with images, instructions, actions, rewards, costs, and next images.
    #
    # Input:
    # - `config`: image shape and trajectory counts.
    #
    # Output:
    # - list of trajectories.
    # - Each trajectory is a list of VLAStep dictionaries.
    #
    # Method:
    # Use the shared project environment:
    # `VisualGridWorldConfig(...)`
    # `collect_visual_gridworld_episodes(grid_config)`
    #
    # Each logged step already contains:
    # - `image`
    # - `instruction`
    # - `action`
    # - `reward`
    # - `cost`
    # - `next_image`
    # - `done`
    #
    # Add these bookkeeping fields while converting:
    # - `episode_id`: which trajectory this step came from
    # - `step_id`: where this step sits inside that trajectory
    # - `split`: `"train"`, `"val"`, or `"test"`
    #
    # These fields matter because later exercises rebuild ordered rollouts.
    #
    # Steps:
    # 1. Build `grid_config = VisualGridWorldConfig(...)`.
    # 2. Use `num_trajectories` as `num_episodes`.
    # 3. Use `steps_per_trajectory` as `max_steps`.
    # 4. Collect episodes.
    # 5. Convert each logged step into a `VLAStep`.
    #
    # Checks:
    # - `done` should be true on the final step of each trajectory.
    # - The same instruction string should stay attached to each step in an episode.
    _ = (VisualGridWorldConfig, collect_visual_gridworld_episodes)
    raise NotImplementedError("Build tiny VLA trajectories.")


def flatten_trajectories(trajectories: list[list[VLAStep]]) -> list[VLAStep]:
    """Flatten trajectories into individual step examples."""
    # Goal:
    # Turn a list of episodes into a list of training examples.
    #
    # Input:
    # - `trajectories`: list of trajectory lists.
    #
    # Output:
    # - flat list of VLAStep dictionaries.
    #
    # Method:
    # Use a normal nested loop:
    # `for trajectory in trajectories:`
    # `    for step in trajectory:`
    #
    # Checks:
    # - The number of flat examples should equal the sum of trajectory lengths.
    raise NotImplementedError("Flatten VLA trajectories.")


class VLADataset(Dataset):
    """Small Dataset wrapper for VLA steps."""

    def __init__(self, steps: list[VLAStep]) -> None:
        # Goal:
        # Store the list of step examples.
        #
        # Method:
        # Set `self.steps = steps`.
        self.steps = steps

    def __len__(self) -> int:
        """Return the number of step examples."""
        return len(self.steps)

    def __getitem__(self, index: int) -> VLAStep:
        """Return one VLA step."""
        return self.steps[index]


def collate_vla_steps(steps: list[VLAStep]) -> VLABatch:
    """Collate VLA steps into one batch."""
    # Goal:
    # Convert a list of step dictionaries into batch tensors and text lists.
    #
    # Input:
    # - list of VLAStep dictionaries.
    #
    # Output:
    # - dictionary with:
    #   - `images`: `[batch_size, channels, height, width]`
    #   - `instructions`: list[str]
    #   - `actions`: `[batch_size]`
    #   - `rewards`: `[batch_size]`
    #   - `costs`: `[batch_size]`
    #   - `next_images`: `[batch_size, channels, height, width]`
    #   - `dones`: `[batch_size]`
    #   - `episode_ids`: `[batch_size]`
    #   - `step_ids`: `[batch_size]`
    #   - `splits`: list[str]
    #
    # Method:
    # Use `torch.stack(...)` for image tensors.
    # Use `torch.tensor(...)` for actions, rewards, costs, and dones.
    #
    # Checks:
    # - Keep instructions as strings. Do not tokenize them in this dataset file.
    raise NotImplementedError("Collate VLA steps into a batch.")


def build_vla_dataloader(config: VLADatasetConfig) -> DataLoader:
    """Return a DataLoader for tiny VLA data."""
    # Goal:
    # Build trajectories, flatten them, wrap them in a Dataset, and return a DataLoader.
    #
    # Method:
    # Use:
    # - `build_tiny_vla_trajectories(config)`
    # - `flatten_trajectories(trajectories)`
    # - `VLADataset(steps)`
    # - `DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_vla_steps)`
    raise NotImplementedError("Build tiny VLA dataloader.")


def explain_vla_format() -> str:
    """Return a short VLA dataset explanation."""
    return "A VLA dataset connects what the agent saw, what it was told, what it did, and what happened next."


def main() -> None:
    """Run the full VLA dataloader scaffold."""
    config = VLADatasetConfig()
    try:
        loader = build_vla_dataloader(config)
    except NotImplementedError as error:
        print(f"[TODO] build_vla_dataloader: {error}")
        return
    batch = next(iter(loader))
    print(
        {
            "images": tuple(batch["images"].shape),
            "instructions": len(batch["instructions"]),
            "actions": tuple(batch["actions"].shape),
            "rewards": tuple(batch["rewards"].shape),
            "costs": tuple(batch["costs"].shape),
            "next_images": tuple(batch["next_images"].shape),
            "dones": tuple(batch["dones"].shape),
            "episode_ids": tuple(batch["episode_ids"].shape),
            "step_ids": tuple(batch["step_ids"].shape),
            "splits": len(batch["splits"]),
        }
    )


if __name__ == "__main__":
    main()
