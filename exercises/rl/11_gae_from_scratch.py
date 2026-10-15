"""
============================================================
EXERCISE 11: GAE FROM SCRATCH
============================================================

WHAT YOU ARE BUILDING
You are building generalized advantage estimation, usually called GAE.

Plain-language meanings:
- value means the critic's guess for future reward
- advantage means how much better an action was than expected
- delta means one-step surprise
- lambda controls how much future surprise is mixed in

WHY THIS EXERCISE EXISTS
PPO depends heavily on good advantage estimates.
the related exercise uses advantages.
This part teaches how to compute them.

Formula:

delta_t = reward_t + gamma * value_{t+1} * (1 - done_t) - value_t
advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_{t+1}

What each symbol means:
- `reward_t`: reward at step `t`
- `gamma`: discount for future reward
- `lambda`: smoothing value between 0 and 1
- `done_t`: 1 if the episode ended at step `t`, else 0
- `value_t`: critic value estimate at step `t`
- `advantage_t`: better-or-worse-than-expected score
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import TypedDict

import torch


BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42) -> None:
    """Set the main random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)


def print_pass_or_try_again(passed: bool, label: str, actual: float, target: str) -> None:
    """Print a short result message."""
    status = "PASS" if passed else "TRY_AGAIN"
    print(f"[{status}] {label}: actual={actual:.4f} | target={target}")


@dataclass
class GAEConfig:
    """Settings for the GAE exercise."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    rollout_length: int = 6
    random_state: int = 42


class GAERollout(TypedDict):
    """Small rollout tensors used to practice GAE."""

    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


class GAESummary(TypedDict):
    """Summary returned by the GAE demo."""

    advantages_shape: tuple[int, ...]
    returns_shape: tuple[int, ...]
    normalized_mean: float
    gae_note: str


def build_toy_rollout(config: GAEConfig) -> GAERollout:
    """Return a tiny rollout with rewards, dones, and values."""
    # Goal:
    # Build small tensors so GAE can be checked by hand.
    # This function does not use an environment. It just returns a tiny fixed rollout.
    #
    # Input:
    # - `config`: GAEConfig
    # - Use `config.rollout_length` as the number of steps, called `T` below.
    #
    # Output:
    # Return this exact dictionary:
    # - `"rewards"`: float tensor with shape `[T]`
    # - `"dones"`: float tensor with shape `[T]`
    # - `"values"`: float tensor with shape `[T + 1]`
    #
    # Small example:
    # - `rewards = tensor([0.0, 0.0, 1.0])`
    # - `dones = tensor([0.0, 0.0, 1.0])`
    # - `values = tensor([0.2, 0.3, 0.4, 0.0])`
    #
    # Method:
    # Use:
    # - `torch.zeros(config.rollout_length, dtype=torch.float32)` for rewards and dones
    # - `torch.linspace(0.2, 0.7, steps=config.rollout_length + 1)` for simple value guesses
    #
    # Then set:
    # - the final reward to `1.0`
    # - the final done flag to `1.0`
    # - the final value to `0.0` because there is no future value after a terminal step
    #
    # `values` has one extra item because TD delta needs `value_{t+1}`.
    #
    # Steps:
    # 1. Create `rewards` with shape `[T]`.
    # 2. Create `dones` with shape `[T]`.
    # 3. Create `values` with shape `[T + 1]`.
    # 4. Set the last reward and done flag.
    # 5. Set the last value to `0.0`.
    # 6. Return the dictionary.
    #
    # Checks:
    # - `rewards.shape` should be `(config.rollout_length,)`.
    # - `dones.shape` should be `(config.rollout_length,)`.
    # - `values.shape` should be `(config.rollout_length + 1,)`.
    # - `dones[-1]` should be `1.0`.
    raise NotImplementedError("Build a toy rollout for GAE.")


def compute_td_deltas(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute one-step TD deltas."""
    # Goal:
    # Compute the one-step surprise at each rollout step.
    # A positive delta means the reward plus next value was better than the critic expected.
    #
    # Formula:
    # ```text
    # delta_t = reward_t + gamma * value_(t+1) * (1 - done_t) - value_t
    # ```
    #
    # Meaning:
    # - `reward_t` is the reward at step `t`.
    # - `gamma` controls how much future value counts.
    # - `value_(t+1)` is the critic value for the next step.
    # - `done_t` is `1.0` if the episode ended at this step, else `0.0`.
    # - `(1 - done_t)` removes future value after a terminal step.
    # - `value_t` is the critic value for the current step.
    #
    # Input:
    # - `rewards`: float tensor with shape `[T]`
    # - `dones`: float tensor with shape `[T]`
    # - `values`: float tensor with shape `[T + 1]`
    # - `gamma`: float, such as `0.99`
    #
    # Output:
    # - `deltas`: float tensor with shape `[T]`
    #
    # Tiny example:
    # If `reward_t=1.0`, `done_t=1.0`, and `value_t=0.4`, then:
    # `delta_t = 1.0 + gamma * value_next * 0.0 - 0.4 = 0.6`.
    #
    # Method:
    # Use vector slices:
    # - `current_values = values[:-1]`
    # - `next_values = values[1:]`
    # - `deltas = rewards + gamma * next_values * (1.0 - dones) - current_values`
    #
    # Steps:
    # 1. Read `current_values` from all values except the last one.
    # 2. Read `next_values` from all values except the first one.
    # 3. Apply the formula above.
    # 4. Return `deltas`.
    #
    # Checks:
    # - `deltas.shape` should match `rewards.shape`.
    # - If `dones[t] == 1.0`, changing `values[t + 1]` should not change `deltas[t]`.
    raise NotImplementedError("Compute TD deltas.")


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Compute generalized advantage estimates."""
    # Goal:
    # Walk backward through the rollout and compute advantages.
    # GAE mixes one-step surprises over time.
    #
    # Formula:
    # ```text
    # advantage_t = delta_t + gamma * gae_lambda * (1 - done_t) * advantage_(t+1)
    # ```
    #
    # Meaning:
    # - `delta_t` is the one-step surprise from `compute_td_deltas(...)`.
    # - `gae_lambda` controls how much later surprise is mixed into this step.
    # - `(1 - done_t)` stops the carry when the episode ended.
    #
    # Input:
    # - `rewards`: float tensor with shape `[T]`
    # - `dones`: float tensor with shape `[T]`
    # - `values`: float tensor with shape `[T + 1]`
    # - `gamma`: float
    # - `gae_lambda`: float between `0.0` and `1.0`
    #
    # Output:
    # - `advantages`: float tensor with shape `[T]`
    #
    # Method:
    # Use:
    # - `deltas = compute_td_deltas(rewards, dones, values, gamma)`
    # - `advantages = torch.zeros_like(rewards)`
    # - `next_advantage = 0.0`
    #
    # Steps:
    # 1. Compute deltas.
    # 2. Create `advantages` with shape `[T]`.
    # 3. Start `next_advantage = 0.0`.
    # 4. Loop backward from `T - 1` down to `0`.
    # 5. Compute:
    #    `next_advantage = deltas[t] + gamma * gae_lambda * (1.0 - dones[t]) * next_advantage`
    # 6. Store `next_advantage` in `advantages[t]`.
    # 7. Return `advantages`.
    #
    # Checks:
    # - `advantages.shape` should match `rewards.shape`.
    # - If all `dones` are `1.0`, each advantage should equal its matching delta.
    # - If `gae_lambda=0.0`, each advantage should also equal its matching delta.
    raise NotImplementedError("Compute GAE advantages.")


def compute_returns_from_advantages(values: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Return value targets from values and advantages."""
    # Goal:
    # Convert advantages into value targets for the critic.
    #
    # Formula:
    # ```text
    # return_t = advantage_t + value_t
    # ```
    #
    # Meaning:
    # - `value_t` is the critic's old guess.
    # - `advantage_t` is how much better or worse the outcome was than that guess.
    # - adding them gives the target the critic should move toward.
    #
    # Input:
    # - `values`: float tensor with shape `[T + 1]`
    # - `advantages`: float tensor with shape `[T]`
    #
    # Output:
    # - `returns`: float tensor with shape `[T]`
    #
    # Method:
    # Use only the first `T` values:
    # - `current_values = values[:-1]`
    # - `returns = advantages + current_values`
    #
    # Steps:
    # 1. Slice current values with `values[:-1]`.
    # 2. Add advantages to current values.
    # 3. Return the result.
    #
    # Checks:
    # - `returns.shape` should match `advantages.shape`.
    # - If `advantages` are all zero, returns should equal `values[:-1]`.
    raise NotImplementedError("Compute value targets from advantages.")


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalize advantages to mean 0 and standard deviation 1."""
    # Goal:
    # Make advantages easier for PPO to optimize.
    # This keeps very large advantage numbers from making policy updates too large.
    #
    # Formula:
    # ```text
    # normalized = (advantages - mean) / (std + 1e-8)
    # ```
    #
    # Meaning:
    # - `mean` is the average advantage in this rollout.
    # - `std` is the standard deviation, which measures how spread out the values are.
    # - `1e-8` is a tiny number that prevents division by zero.
    #
    # Input:
    # - `advantages`: float tensor with shape `[T]`
    #
    # Output:
    # - `normalized`: float tensor with shape `[T]`
    #
    # Method:
    # Use:
    # - `mean = advantages.mean()`
    # - `std = advantages.std(unbiased=False)`
    # - `normalized = (advantages - mean) / (std + 1e-8)`
    #
    # `unbiased=False` keeps this simple and uses the batch's own standard deviation.
    #
    # Steps:
    # 1. Compute the mean.
    # 2. Compute the standard deviation.
    # 3. Apply the formula above.
    # 4. Return the normalized tensor.
    #
    # Checks:
    # - `normalized.shape` should match `advantages.shape`.
    # - `normalized.mean()` should be close to `0.0`.
    # - If all advantages are the same, the output should stay finite, not become NaN.
    raise NotImplementedError("Normalize advantages.")


def run_gae_demo(config: GAEConfig) -> GAESummary:
    """Run the full GAE demo."""
    # Goal:
    # Run every helper in order and return a small summary.
    #
    # Input:
    # - `config`: GAEConfig
    #
    # Output:
    # Return this exact dictionary:
    # - `"advantages_shape"`: tuple[int, ...]
    # - `"returns_shape"`: tuple[int, ...]
    # - `"normalized_mean"`: float
    # - `"gae_note"`: str
    #
    # Method:
    # Use the helper functions in this file:
    # - `rollout = build_toy_rollout(config)`
    # - `advantages = compute_gae(...)`
    # - `returns = compute_returns_from_advantages(...)`
    # - `normalized = normalize_advantages(advantages)`
    #
    # Steps:
    # 1. Build the toy rollout.
    # 2. Read `rewards`, `dones`, and `values` from the rollout dictionary.
    # 3. Compute GAE advantages.
    # 4. Compute returns from values and advantages.
    # 5. Normalize the advantages.
    # 6. Build a short note that says GAE mixes one-step surprises over time.
    # 7. Return the summary dictionary.
    #
    # Checks:
    # - `advantages_shape` should equal `(config.rollout_length,)`.
    # - `returns_shape` should equal `(config.rollout_length,)`.
    # - `normalized_mean` should be close to `0.0`.
    raise NotImplementedError("Run GAE demo.")


def check_metrics(summary: GAESummary, config: GAEConfig) -> bool:
    """Return True when advantage shape is correct."""
    return tuple(summary.get("advantages_shape", ())) == (config.rollout_length,)


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
    """Run the GAE scaffold."""
    config = GAEConfig()
    set_seed(config.random_state)

    rollout, ok = report_task_progress("build_toy_rollout", build_toy_rollout, config)
    if not ok:
        return
    _, ok = report_task_progress("compute_td_deltas", compute_td_deltas, rollout["rewards"], rollout["dones"], rollout["values"], config.gamma)
    if not ok:
        return
    summary, ok = report_task_progress("run_gae_demo", run_gae_demo, config)
    if ok and summary is not None:
        passed = check_metrics(summary, config)
        print_pass_or_try_again(passed, "advantages_shape", float(int(passed)), "expected shape")


if __name__ == "__main__":
    main()
