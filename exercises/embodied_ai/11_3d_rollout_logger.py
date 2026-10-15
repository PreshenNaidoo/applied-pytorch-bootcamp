"""
============================================================
EXERCISE 11: 3D ROLLOUT LOGGER
============================================================

WHAT YOU ARE BUILDING
You are saving first-person 3D rollouts in the same vision-language-action
format used by the 2D flagship exercises.

The important idea is:

```text
camera image + instruction + action -> reward, cost, next image, done
```

FULL WORKFLOW

```text
embodied_ai/10 creates the environment
embodied_ai/11 records episodes from it
embodied_ai/12 trains a behavior-cloning policy from the records
embodied_ai/13 trains a world model from the records
embodied_ai/14 tests planning in the environment
embodied_ai/22 trains and tests a policy with temporal memory
```

This exercise covers the recording step.
It does not train a model.

WHY THIS EXERCISE EXISTS
Generalist visual-action agents and robot policies train from logged
trajectories.
If the logging format is messy, every later model becomes harder to train and
evaluate.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
many games or simulators            one small 3D room
large trajectory dataset            small local saved rollouts
rich metadata                       episode id, step id, split, pose
offline training store              simple torch or json-like records

DATASET AND DOWNLOADS
Default dataset:
- no download
- call the environment from `embodied_ai/10_miniworld_first_person_navigation.py`
- save the result to `outputs/mini3d_vla_dataset.pt`

Saved file format:
- use `torch.save(episodes, config.dataset_path)`
- `episodes` is `list[list[Logged3DStep]]`
- the outer list is episodes
- the inner list is ordered steps inside one episode

Optional later dataset:
- after the local logger works, adapt Matterport3D / Room-to-Room or a
  MiniWorld dataset into the same field names
- do not start there; first prove the local format works

WHY ORDER MATTERS
Training a one-step policy can use shuffled rows.
World-model rollout tests cannot.
They need step `0`, then step `1`, then step `2`, so the model can be rolled
forward through the same actions the real environment used.

INPUTS AND OUTPUTS
- `image`: `[3, image_size, image_size]`
- `instruction`: string
- `action`: int
- `reward`: float
- `cost`: float
- `next_image`: `[3, image_size, image_size]`
- `done`: bool
- `agent_pose`: `[3]`
- `episode_id`: int
- `step_id`: int
- `split`: `"train"`, `"val"`, or `"test"`
- `layout_seed`: seed used to build the room geometry
- `task_variant`: `"navigation"` or `"memory_corridor"`
- `cue_id`: red cue `0`, blue cue `1`, or `-1` when no cue is used
- `cue_visible`: whether this frame still shows the cue
- `memory_required`: whether the current action needs the earlier cue

METRICS TO TRACK
- number of episodes saved
- number of total steps saved
- mean episode length
- mean reward
- mean cost
- percent of episodes that ended with `done=True`

VISUALS TO CREATE
Save one replay strip to `outputs/mini3d_logger/replay_episode_0.png`.
It should show frames from one saved episode in order.

TARGET RESULT RANGE
This exercise is mostly a data plumbing check.

Useful signs:
- `num_episodes` saved is the number you requested
- most episodes have at least a few steps
- `step_id` starts at `0` inside each episode
- total reward and total cost are finite numbers
- the replay strip shows ordered frames, not shuffled frames

COMMON MISTAKES
- flattening episodes too early and losing order
- saving only `image` and forgetting `next_image`
- forgetting `episode_id` and `step_id`
- using one split for every row and then pretending there is validation data
- saving tensors with the wrong shape

STEP-BY-STEP TODO
1. Import or move the environment helpers from the related exercise.
2. Collect ordered episodes.
3. Add `episode_id`, `step_id`, and `split` to every step.
4. Keep cue and memory metadata from the environment.
5. Save the nested episode list with `torch.save`.
6. Load the same file back.
7. Replay one episode summary.
8. Save a frame strip from the first episode.

DEFINITION OF DONE
You are done when:
- 3D rollouts are saved in the shared VLA shape
- saved data can be loaded again
- one trajectory can be replayed in order
- actions line up with the next frames
- memory episodes keep the cue id and hidden-cue decision point
- a replay strip is saved for one episode
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import torch


class Logged3DStep(TypedDict):
    """One saved 3D VLA step."""

    image: torch.Tensor
    instruction: str
    action: int
    reward: float
    cost: float
    next_image: torch.Tensor
    done: bool
    agent_pose: torch.Tensor
    episode_id: int
    step_id: int
    split: str
    layout_seed: int
    task_variant: str
    cue_id: int
    cue_visible: bool
    memory_required: bool


@dataclass
class RolloutLogger3DConfig:
    """Settings for 3D rollout logging."""

    num_episodes: int = 64
    max_steps: int = 40
    image_size: int = 64
    random_state: int = 42
    collection_policy: Literal["scripted", "random"] = "scripted"
    dataset_path: Path = Path("outputs/mini3d_vla_dataset.pt")


def collect_3d_rollouts(config: RolloutLogger3DConfig) -> list[list[Logged3DStep]]:
    """Collect ordered 3D rollouts."""
    # Where this fits:
    # This creates the exact dataset used by embodied exercises 12, 13, 14, and 22.
    #
    # Goal:
    # Use the environment from the related exercise and save ordered episodes.
    #
    # Method:
    # Import these helpers from the related environment exercise or move
    # shared environment code into `common/` later:
    # - `Mini3DNavigationConfig`
    # - `reset_mini_3d_environment(...)`
    # - `step_mini_3d_environment(...)`
    # - `choose_scripted_3d_action(...)`
    #
    # Steps:
    # 1. Build a matching `Mini3DNavigationConfig(...)`.
    # 2. Shuffle layout seeds once with a generator seeded by
    #    `config.random_state`, then assign about 70% train, 15% validation,
    #    and 15% test. Every episode using the same layout seed must receive
    #    the same split.
    # 3. For normal navigation, call reset with a different layout seed for
    #    each episode.
    # 4. For memory data, collect paired episodes:
    #    - same layout seed, forced `cue_id=0`
    #    - same layout seed, forced `cue_id=1`
    #    This makes the hidden-cue decision frames directly comparable.
    # 5. Pick actions:
    #    - use random actions only for the first logger smoke test
    #    - call `choose_scripted_3d_action(state, environment_config)` for the
    #      saved training dataset
    #    - at the memory junction, red must choose left and blue must choose
    #      right
    # 6. Call `step_mini_3d_environment(...)`.
    # 7. Add `episode_id`, `step_id`, and the preassigned `split` to every step.
    # 8. Copy `layout_seed`, `task_variant`, `cue_id`, `cue_visible`, and `memory_required`
    #    from the environment step.
    # 9. Set `state = transition["next_state"]` before the next action.
    # 10. Stop the episode when `done` is true or `max_steps` is reached.
    #
    # Output:
    # - list of episodes
    # - each episode is a list of Logged3DStep dictionaries
    #
    # Checks:
    # - `episode_id` should be stable inside one episode.
    # - `step_id` should count up from 0.
    # - At least one memory episode should contain a hidden step where
    #   `memory_required=True` and `cue_visible=False`.
    # - Paired red and blue episodes should share a layout seed.
    # - One layout seed must never occur in more than one split.
    # - Before behavior cloning, confirm memory decision labels are correct.
    #   Random action labels cannot teach the cue rule.
    # - Measure the scripted expert's rollout success before saving a large
    #   dataset. Bad expert labels produce a bad behavior-cloning policy.
    _ = config
    raise NotImplementedError("Collect 3D VLA rollouts.")


def save_3d_dataset(episodes: list[list[Logged3DStep]], config: RolloutLogger3DConfig) -> str:
    """Save the 3D rollout dataset."""
    # Goal:
    # Save the collected episodes to `config.dataset_path`.
    #
    # Method:
    # Use `torch.save(...)` for the first version.
    # Exact call:
    # `torch.save(episodes, config.dataset_path)`
    #
    # Check:
    # - Create the parent folder before saving.
    # - Return `str(config.dataset_path)`.
    _ = (episodes, config)
    raise NotImplementedError("Save the 3D rollout dataset.")


def load_3d_dataset(config: RolloutLogger3DConfig) -> list[list[Logged3DStep]]:
    """Load saved 3D rollouts."""
    # Goal:
    # Load the same structure saved by `save_3d_dataset`.
    #
    # Method:
    # Use `torch.load(config.dataset_path, weights_only=False)`.
    #
    # Check:
    # - The returned value should be a list of episodes.
    # - The first step should have keys like `"image"`, `"action"`, and `"next_image"`.
    _ = config
    raise NotImplementedError("Load the 3D rollout dataset.")


def replay_trajectory_summary(episode: list[Logged3DStep]) -> dict[str, float]:
    """Return simple checks for one saved trajectory."""
    # Goal:
    # Check that saved steps are ordered and usable.
    #
    # Output:
    # - `"num_steps"`
    # - `"total_reward"`
    # - `"total_cost"`
    # - `"ended"`
    #
    # Method:
    # - `num_steps = len(episode)`
    # - total reward is the sum of `step["reward"]`
    # - total cost is the sum of `step["cost"]`
    # - ended is `1.0` if the last step has `done=True`, else `0.0`
    #
    # Visual:
    # After this summary works, save a replay strip by plotting several
    # `step["image"]` tensors in step order.
    _ = episode
    raise NotImplementedError("Replay one saved 3D trajectory.")


def save_replay_strip(episode: list[Logged3DStep], config: RolloutLogger3DConfig) -> str:
    """Save ordered frames from one logged episode."""
    # Where this fits:
    # This catches a common logging bug: saved rows can look fine as tensors,
    # but the episode order can be wrong.
    #
    # Goal:
    # Save a frame strip from the first few `step["image"]` tensors.
    #
    # Output:
    # - path to `outputs/mini3d_logger/replay_episode_0.png`.
    #
    # Method:
    # 1. Use `matplotlib.use("Agg")`.
    # 2. Plot the first 6 frames in `step_id` order.
    # 3. Put each action id in the subplot title.
    # 4. Save and return the path.
    #
    # Check:
    # - The subplot titles should show action ids in the same order as the data.
    _ = (episode, config)
    raise NotImplementedError("Save a replay strip from one logged episode.")


def main() -> None:
    """Run the 3D rollout logger scaffold."""
    config = RolloutLogger3DConfig()
    try:
        episodes = collect_3d_rollouts(config)
        print(save_3d_dataset(episodes, config))
    except NotImplementedError as error:
        print(f"[TODO] collect_3d_rollouts: {error}")


if __name__ == "__main__":
    main()
