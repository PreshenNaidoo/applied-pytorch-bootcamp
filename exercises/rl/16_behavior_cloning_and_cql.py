"""
============================================================
EXERCISE 16: BEHAVIOR CLONING AND CQL
============================================================

WHAT YOU ARE BUILDING
You are building two tiny offline RL ideas from scratch:
- behavior cloning
- conservative Q-learning, often called CQL

Plain-language meanings:
- offline RL learns from a fixed dataset, not live environment interaction
- behavior cloning imitates the dataset actions
- CQL penalizes high Q-values for actions not well supported by the dataset
- support means actions that appear enough in the dataset

WHY THIS EXERCISE EXISTS
the related exercise introduces offline RL and action support.
This part adds two common learning objectives.

STEP-BY-STEP TODO
1. Build a tiny offline dataset.
2. Train a behavior cloning policy.
3. Build a small Q-network.
4. Compute a Bellman target.
5. Add a conservative Q penalty.
6. Compare behavior cloning and CQL in plain words.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42) -> None:
    """Set the main random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a short result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")


@dataclass
class OfflineObjectiveConfig:
    """Settings for behavior cloning and CQL."""

    num_states: int = 6
    num_actions: int = 3
    hidden_dim: int = 32
    num_examples: int = 128
    batch_size: int = 32
    gamma: float = 0.95
    cql_alpha: float = 0.2
    learning_rate: float = 1e-3
    random_state: int = 42


class BehaviorPolicy(nn.Module):
    """Small policy that predicts dataset actions."""

    def __init__(self, config: OfflineObjectiveConfig) -> None:
        # Goal:
        # Build a small model that maps state ids to action logits.
        #
        # Layers:
        # - `nn.Embedding(num_states, hidden_dim)`
        # - `nn.Linear(hidden_dim, num_actions)`
        super().__init__()
        # Method:
        # - Use the formula, layer, or PyTorch function named above.
        # - Keep the code small and readable; this is a teaching exercise.
        #
        # Steps:
        # 1. Check the input shapes and dtypes from the comments above.
        # 2. Do the calculation one clear step at a time.
        # 3. Return the exact tensor, object, or dictionary that the next function expects.
        #
        # Checks:
        # - Try one tiny example before using the full demo.
        # - Check at least one output shape, key, or value before moving on.
        raise NotImplementedError("Build the behavior cloning policy.")

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        # Method:
        # - Use the formula, layer, or PyTorch function named above.
        # - Keep the code small and readable; this is a teaching exercise.
        #
        # Steps:
        # 1. Check the input shapes and dtypes from the comments above.
        # 2. Do the calculation one clear step at a time.
        # 3. Return the exact tensor, object, or dictionary that the next function expects.
        #
        # Checks:
        # - Try one tiny example before using the full demo.
        # - Check at least one output shape, key, or value before moving on.
        raise NotImplementedError("Return action logits.")


class QNetwork(nn.Module):
    """Small Q-network for discrete states and actions."""

    def __init__(self, config: OfflineObjectiveConfig) -> None:
        # Goal:
        # Build a model that returns one Q-value per action.
        #
        # Output:
        # - Q-values: `[batch_size, num_actions]`
        super().__init__()
        # Method:
        # - Use the formula, layer, or PyTorch function named above.
        # - Keep the code small and readable; this is a teaching exercise.
        #
        # Steps:
        # 1. Check the input shapes and dtypes from the comments above.
        # 2. Do the calculation one clear step at a time.
        # 3. Return the exact tensor, object, or dictionary that the next function expects.
        #
        # Checks:
        # - Try one tiny example before using the full demo.
        # - Check at least one output shape, key, or value before moving on.
        raise NotImplementedError("Build the Q-network.")

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return Q-values for every action."""
        # Method:
        # - Use the formula, layer, or PyTorch function named above.
        # - Keep the code small and readable; this is a teaching exercise.
        #
        # Steps:
        # 1. Check the input shapes and dtypes from the comments above.
        # 2. Do the calculation one clear step at a time.
        # 3. Return the exact tensor, object, or dictionary that the next function expects.
        #
        # Checks:
        # - Try one tiny example before using the full demo.
        # - Check at least one output shape, key, or value before moving on.
        raise NotImplementedError("Return Q-values.")


def build_offline_transition_dataset(config: OfflineObjectiveConfig) -> DataLoader:
    """Return a dataloader of offline transitions."""
    # Goal:
    # Build a fixed dataset of transitions.
    #
    # Batch tensors:
    # - `states`: `[batch_size]`
    # - `actions`: `[batch_size]`
    # - `rewards`: `[batch_size]`
    # - `next_states`: `[batch_size]`
    # - `dones`: `[batch_size]`
    #
    # Method:
    # Use `TensorDataset` and `DataLoader`.
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Build offline transition dataloader.")


def behavior_cloning_loss(policy: nn.Module, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return cross entropy imitation loss."""
    # Goal:
    # Train the policy to imitate dataset actions.
    #
    # Formula:
    # `loss = cross_entropy(policy(states), actions)`
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Compute behavior cloning loss.")


def bellman_q_loss(
    q_network: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Return a one-step Bellman Q-learning loss."""
    # Goal:
    # Match Q(s, a) to a one-step target.
    #
    # Target:
    # `target = reward + gamma * max_next_q * (1 - done)`
    #
    # Loss:
    # mean squared error between chosen Q-values and targets
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Compute Bellman Q loss.")


def cql_penalty(q_values: torch.Tensor, dataset_actions: torch.Tensor) -> torch.Tensor:
    """Return a simple conservative Q penalty."""
    # Goal:
    # Penalize the model for giving too much value to all actions,
    # compared with the dataset action.
    #
    # Formula:
    # `penalty = logsumexp(q_values over actions) - q_value_for_dataset_action`
    #
    # Output:
    # - scalar mean penalty
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Compute CQL penalty.")


def run_offline_objective_demo(config: OfflineObjectiveConfig) -> dict[str, float | str]:
    """Run one tiny behavior cloning and CQL demo."""
    # Output keys:
    # - `behavior_cloning_loss`
    # - `bellman_loss`
    # - `cql_penalty`
    # - `offline_note`
    # Method:
    # - Use the formula, layer, or PyTorch function named above.
    # - Keep the code small and readable; this is a teaching exercise.
    #
    # Steps:
    # 1. Check the input shapes and dtypes from the comments above.
    # 2. Do the calculation one clear step at a time.
    # 3. Return the exact tensor, object, or dictionary that the next function expects.
    #
    # Checks:
    # - Try one tiny example before using the full demo.
    # - Check at least one output shape, key, or value before moving on.
    raise NotImplementedError("Run behavior cloning and CQL demo.")


def check_metrics(summary: dict[str, float | str]) -> bool:
    """Return True when expected objective values are present."""
    required = {"behavior_cloning_loss", "bellman_loss", "cql_penalty", "offline_note"}
    return required.issubset(summary)


def report_task_progress(task_name: str, task_fn, *args):
    """Run one task and print whether it is done yet."""
    print(f"[RUNNING] {task_name}")
    try:
        result = task_fn(*args)
    except NotImplementedError as error:
        print(f"[TODO] {task_name}: {error}")
        return None, False
    except Exception as error:
        print(f"[ERROR] {task_name}: {type(error).__name__}: {error}")
        raise
    print(f"[DONE] {task_name}")
    return result, True


def main() -> None:
    """Run the offline objective scaffold."""
    config = OfflineObjectiveConfig()
    set_seed(config.random_state)

    _, ok = report_task_progress("build_offline_transition_dataset", build_offline_transition_dataset, config)
    if not ok:
        return
    summary, ok = report_task_progress("run_offline_objective_demo", run_offline_objective_demo, config)
    if ok and summary is not None:
        passed = check_metrics(summary)
        print_pass_or_try_again(passed, "offline_objective_fields", float(int(passed)), "all expected fields")


if __name__ == "__main__":
    main()
