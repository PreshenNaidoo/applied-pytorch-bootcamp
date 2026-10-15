"""
============================================================
EXERCISE 05: REWARD AND COST MODEL FOR EMBODIED AGENTS
============================================================

WHAT YOU ARE BUILDING
You are training small models that predict reward and cost from logged transitions.

Plain-language meanings:
- reward means how useful an outcome is
- cost means how unsafe or unwanted an outcome is
- logged trajectories are saved steps from earlier agent runs
- reward minus cost is a simple safety-aware score

WHY THIS EXERCISE EXISTS
Embodied agents need to know not only what helps them succeed, but also what may be unsafe.
Reward alone can teach risky behavior if cost is ignored.

The logged data should come from the same visual grid-world used by the video
prediction and VLA exercises. Each row is a real environment step:

```text
state + action -> next_state, reward, cost, done
```

This is the reward-and-risk part of the flagship 2D world-model agent.
Later, the 3D version uses the same idea with first-person frames and
keyboard-like actions.

DATASET OPTIONS
Default: use logged rollouts from `common.embodied_gridworld`.

Exact local data method:
1. Build `VisualGridWorldConfig(num_episodes=config.num_episodes, max_steps=config.max_steps, random_state=config.random_state)`.
2. Call `collect_visual_gridworld_episodes(grid_config)`.
3. Call `flatten_logged_episodes(episodes)`.
4. Use `step["state"]`, `step["action"]`, `step["reward"]`, and `step["cost"]`.

Real dataset options after the default works:
- `lerobot/pusht` on Hugging Face because it includes reward, done, success, state, and action fields.
- LeRobot ALOHA simulation datasets for action and state prediction. Add a task-specific cost rule if the dataset does not provide one.

MAIN LOOP IN PLAIN WORDS

```text
load logged transitions
train a model to predict reward
train a model to predict cost
score possible actions with reward minus cost
choose safer actions
```

INPUTS AND OUTPUTS
- state: `[state_dim]`
- one-hot action: `[num_actions]`
- model input: `[state_dim + num_actions]`
- predicted reward: `[1]`
- predicted cost: `[1]`

LOSS FUNCTION
Use two mean squared errors:

```text
loss = reward_mse + cost_mse
```

METRICS TO TRACK
- validation reward MSE: lower is better
- validation cost MSE: lower is better
- cost event accuracy: higher is better if you binarize `cost > 0`

VISUALS TO CREATE
Save `outputs/reward_cost_model/reward_cost_predictions.png`.
Use a small scatter plot or bar chart comparing true and predicted reward/cost.

DEFINITION OF DONE
You are done when:
- logged transitions are collected from a stepped environment
- reward prediction trains
- cost prediction trains
- actions can be scored with reward minus cost
- held-out reward and cost errors are reported
- a reward/cost prediction chart is saved
- you can explain why reward alone can be unsafe
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


class LoggedTransition(TypedDict):
    """One saved embodied-agent transition."""

    state: torch.Tensor
    action: int
    next_state: torch.Tensor
    reward: float
    cost: float


@dataclass
class RewardCostConfig:
    """Settings for reward and cost prediction."""

    state_dim: int = 6
    num_actions: int = 4
    num_episodes: int = 96
    max_steps: int = 18
    hidden_dim: int = 64
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 5
    val_fraction: float = 0.2
    random_state: int = 42


def build_logged_transitions(config: RewardCostConfig) -> list[LoggedTransition]:
    """Return logged transitions from real grid-world rollouts."""
    # Goal:
    # Build saved transitions with reward and cost labels from the project environment.
    #
    # Input:
    # - `config`: state size, action count, and rollout settings.
    #
    # Output:
    # - list of transition dictionaries.
    # - Each transition has:
    #   - `state`: tensor shaped `[state_dim]`
    #   - `action`: int
    #   - `next_state`: tensor shaped `[state_dim]`
    #   - `reward`: float
    #   - `cost`: float
    #
    # Method:
    # Use `common.embodied_gridworld`:
    # - `VisualGridWorldConfig(num_episodes=..., max_steps=..., random_state=...)`
    # - `collect_visual_gridworld_episodes(grid_config)`
    # - `flatten_logged_episodes(episodes)`
    #
    # The reward and cost should be the values returned by `step_grid_world`.
    # Do not invent a separate random rule here.
    #
    # Real meaning:
    # - reward goes up when the agent reaches the green goal
    # - cost goes up when the agent hits a wall or red hazard
    #
    # Checks:
    # - Do not make reward and cost always the same.
    # - Some examples should have high reward and low cost.
    # - Some examples should have high cost.
    _ = (VisualGridWorldConfig, collect_visual_gridworld_episodes, flatten_logged_episodes)
    raise NotImplementedError("Build logged reward and cost transitions.")


def build_reward_cost_loader(config: RewardCostConfig) -> DataLoader:
    """Build a dataloader for reward and cost prediction."""
    # Goal:
    # Turn logged transitions into tensors for training.
    #
    # Output batch:
    # - inputs: `[batch_size, state_dim + num_actions]`
    # - rewards: `[batch_size, 1]`
    # - costs: `[batch_size, 1]`
    #
    # Method:
    # 1. Call `build_logged_transitions(config)`.
    # 2. One-hot encode each action.
    # 3. Concatenate state and one-hot action with `torch.cat(..., dim=-1)`.
    # 4. Build a TensorDataset and DataLoader.
    #
    # Real project note:
    # - In the 2D flagship project, `state` can be the six-number grid state.
    # - In the 3D flagship project, `state` will usually be a learned visual or memory vector.
    _ = TensorDataset
    raise NotImplementedError("Build reward and cost dataloader.")


class RewardCostModel(nn.Module):
    """Predict reward and cost from state plus action."""

    def __init__(self, config: RewardCostConfig) -> None:
        # Goal:
        # Build a shared trunk with two heads.
        #
        # Input:
        # - `[batch_size, state_dim + num_actions]`
        #
        # Output:
        # - predicted reward: `[batch_size, 1]`
        # - predicted cost: `[batch_size, 1]`
        #
        # Method:
        # Use:
        # - shared `Linear -> ReLU -> Linear -> ReLU`
        # - `reward_head = nn.Linear(hidden_dim, 1)`
        # - `cost_head = nn.Linear(hidden_dim, 1)`
        super().__init__()
        raise NotImplementedError("Build reward and cost model.")

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted reward and cost."""
        # Goal:
        # Predict both useful reward and unsafe cost.
        #
    # Checks:
    # - Both outputs should be shaped `[batch_size, 1]`.
    raise NotImplementedError("Run reward and cost model.")


def train_reward_cost_model(config: RewardCostConfig) -> dict[str, float]:
    """Train and evaluate reward and cost prediction."""
    # Goal:
    # Train the model on logged environment steps, then report held-out errors.
    #
    # Output:
    # - dictionary with:
    #   - `"train_loss"`
    #   - `"val_reward_mse"`
    #   - `"val_cost_mse"`
    #
    # Method:
    # 1. Build logged transitions.
    # 2. Turn each state and action into the model input.
    # 3. Split into train and validation sets.
    # 4. Train `RewardCostModel(config)` with `torch.optim.AdamW(...)`.
    # 5. Use `nn.functional.mse_loss(...)` for reward and cost.
    # 6. Evaluate on held-out steps that were not used for training.
    #
    # Checks:
    # - Report reward and cost errors separately.
    # - Cost may be harder because high-cost events are rarer.
    raise NotImplementedError("Train and evaluate reward and cost model.")


def choose_action_with_reward_minus_cost(
    model: RewardCostModel,
    state: torch.Tensor,
    config: RewardCostConfig,
) -> int:
    """Choose the action with the best predicted reward minus cost."""
    # Goal:
    # Use the trained model to choose one action safely.
    #
    # Input:
    # - `state`: tensor shaped `[state_dim]`
    #
    # Output:
    # - action id as an int.
    #
    # Method:
    # For every possible action:
    # 1. Build state plus one-hot action.
    # 2. Predict reward and cost.
    # 3. Compute `score = reward - cost`.
    # 4. Return the action with the largest score.
    #
    # Checks:
    # - Do not use reward alone.
    raise NotImplementedError("Choose an action using reward minus cost.")


def explain_reward_vs_cost() -> str:
    """Return a short safety explanation."""
    return "Reward says what helps the task. Cost says what is risky or unsafe. A good agent should care about both."


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
    """Run the full reward-and-cost training scaffold."""
    config = RewardCostConfig()
    metrics, ok = report_task_progress("train_reward_cost_model", train_reward_cost_model, config)
    if ok:
        print(metrics)


if __name__ == "__main__":
    main()
