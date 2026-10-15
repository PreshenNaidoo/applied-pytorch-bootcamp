"""
============================================================
EXERCISE 27: POET STYLE OPEN ENDED LEARNING
============================================================

WHAT YOU ARE BUILDING
You are building a tiny POET-style open-ended learning loop.

POET stands for:
- Paired Open-Ended Trailblazer

In plain language, POET trains pairs:

```text
environment + agent
```

The system does three things:
- improves agents on their own environments
- creates new environments from old environments
- transfers agents between environments when another agent works better

WHY THIS EXERCISE EXISTS
Open-ended learning is not only "train longer".
The system also needs new tasks.

This exercise shows the basic loop without using a large simulator.

Environment:
- a gridworld with a start, a goal, and wall cells

Agent:
- a simple action-probability policy

Environment descriptor:
- grid size
- wall count
- shortest path length from start to goal

Important:
- Generated environments must still be solvable.
- If there is no path from start to goal, reject the environment.

HOW THIS FITS INTO THE BOOTCAMP
Curriculum learning changes the task order.
Novelty search and quality-diversity keep different behaviors.
POET-style learning adds one more idea: generate new tasks while agents improve.

Beginner intuition:
- an agent gets better on one environment
- the environment mutates into a new challenge
- agents can transfer between environments when they help each other

COMMON MISTAKES
- keeping environments that cannot be solved
- mutating the parent environment in place by accident
- training agents but never creating new tasks
- creating new tasks but never improving agents
- forgetting to try transfers between environments

STEP-BY-STEP TODO
1. Build the first easy environment.
2. Normalize and sample policies.
3. Implement environment stepping.
4. Check solvability with breadth-first search.
5. Mutate environments while keeping them solvable.
6. Roll out policies in environments.
7. Improve policies with mutation and selection.
8. Try policy transfers between environments.
9. Run the POET-style outer loop.

DEFINITION OF DONE
You are done when:
- every kept environment is solvable
- each environment has a paired policy
- policies improve on their own environments
- new environments are generated from old ones
- transfers can replace weaker paired policies
- you can explain how tasks and agents grow together
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random
from typing import TypedDict

import torch


# Learning order for this exercise:
# 1. Make environment stepping and solvability checks correct.
# 2. Mutate environments only after solvability works.
# 3. Make one policy rollout work in one environment.
# 4. Improve one policy before running many pairs.
# 5. Add transfers after pairs have scores.
#
# The goal is to combine new tasks with improving agents.

State = tuple[int, int]
Policy = torch.Tensor


@dataclass
class PoetEnvironment:
    """One generated gridworld task."""

    rows: int
    cols: int
    start: State
    goal: State
    walls: set[State]
    name: str


class PoetPair(TypedDict):
    """One environment and its paired agent."""

    env: PoetEnvironment
    policy: Policy
    score: float


class PoetSummary(TypedDict):
    """Final numbers returned by the POET-style demo."""

    pair_count: int
    best_score: float
    generated_envs: int
    successful_transfers: int
    note: str


@dataclass
class PoetConfig:
    """Settings for POET-style open-ended learning."""

    action_count: int = 4
    initial_rows: int = 5
    initial_cols: int = 5
    max_pairs: int = 8
    outer_iterations: int = 12
    mutations_per_iteration: int = 3
    rollout_steps: int = 30
    agent_mutations: int = 12
    mutation_scale: float = 0.35
    transfer_interval: int = 3
    random_state: int = 42


def build_initial_environment(config: PoetConfig) -> PoetEnvironment:
    """Return the first easy environment."""
    # Goal:
    # Create a small solvable gridworld.
    #
    # Output:
    # - `PoetEnvironment`.
    #
    # Suggested environment:
    # - rows and cols from config
    # - start `(0, 0)`
    # - goal `(rows - 1, cols - 1)`
    # - no walls
    # - name `"seed_env"`
    #
    # Checks:
    # - `is_solvable(env)` should be true.
    raise NotImplementedError("Build initial POET environment.")


def normalize_policy(raw_policy: torch.Tensor) -> Policy:
    """Return action probabilities."""
    # Goal:
    # Convert raw action values into a valid policy.
    #
    # Input:
    # - tensor shaped `[4]`.
    #
    # Method:
    # Use `torch.softmax(raw_policy.float(), dim=0)`.
    #
    # Checks:
    # - The output should sum to `1.0`.
    raise NotImplementedError("Normalize POET policy.")


def sample_policy(config: PoetConfig) -> Policy:
    """Create one random policy."""
    # Goal:
    # Return a valid random policy.
    #
    # Method:
    # Use `torch.randn(config.action_count)` and `normalize_policy(...)`.
    #
    # Checks:
    # - Shape should be `(config.action_count,)`.
    raise NotImplementedError("Sample POET policy.")


def step_environment(state: State, action: int, env: PoetEnvironment) -> State:
    """Move one step in one generated environment."""
    # Goal:
    # Apply an action while respecting walls and grid edges.
    #
    # Actions:
    # - `0`: up
    # - `1`: right
    # - `2`: down
    # - `3`: left
    #
    # Rule:
    # If the move hits a wall or leaves the grid, stay in the same state.
    #
    # Checks:
    # - The returned state should never be a wall.
    # - The returned state should stay inside the grid.
    raise NotImplementedError("Step one POET environment.")


def shortest_path_length(env: PoetEnvironment) -> int | None:
    """Return shortest path length from start to goal."""
    # Goal:
    # Check whether an environment is solvable.
    #
    # Output:
    # - number of steps in the shortest path.
    # - `None` if no path exists.
    #
    # Method:
    # Use breadth-first search.
    # Breadth-first search means checking all states 1 step away,
    # then all states 2 steps away, and so on.
    #
    # Use:
    # - `deque([(env.start, 0)])`
    # - a `set` for visited states
    #
    # Steps:
    # 1. Start from `env.start`.
    # 2. Pop one state and distance from the queue.
    # 3. If it is the goal, return the distance.
    # 4. Add valid neighbor states that are not walls.
    # 5. Return `None` if the queue becomes empty.
    #
    # Checks:
    # - A no-wall 5 by 5 grid should have path length `8`.
    raise NotImplementedError("Compute shortest path length.")


def is_solvable(env: PoetEnvironment) -> bool:
    """Return whether the environment has a path to the goal."""
    # Goal:
    # Make the solvable check easy to reuse.
    #
    # Method:
    # Return `shortest_path_length(env) is not None`.
    #
    # Checks:
    # - Environments with blocked start or goal should be false.
    raise NotImplementedError("Check whether POET environment is solvable.")


def mutate_environment(parent: PoetEnvironment, config: PoetConfig, index: int) -> PoetEnvironment | None:
    """Create a new environment from an old one."""
    # Goal:
    # Make a slightly harder or different environment.
    #
    # Mutation ideas:
    # - add one wall
    # - remove one wall
    # - keep start and goal fixed
    #
    # Output:
    # - a new `PoetEnvironment` if it is solvable.
    # - `None` if the mutation creates an unsolvable environment.
    #
    # Steps:
    # 1. Copy the parent's wall set.
    # 2. Randomly add or remove one wall.
    # 3. Never place a wall on start or goal.
    # 4. Build a new environment with a new name.
    # 5. Return it only if `is_solvable(new_env)` is true.
    #
    # Checks:
    # - The parent should not be changed in place.
    # - The new environment should be solvable.
    raise NotImplementedError("Mutate one POET environment.")


def rollout_policy(policy: Policy, env: PoetEnvironment, config: PoetConfig) -> tuple[float, bool]:
    """Run one policy in one environment."""
    # Goal:
    # Return a score and whether the goal was reached.
    #
    # Reward rule:
    # - `1.0` for reaching the goal.
    # - `-0.01` for each step.
    #
    # Method:
    # Start at `env.start`.
    # Sample actions with `torch.multinomial(policy, 1).item()`.
    #
    # Output:
    # - total reward
    # - reached_goal flag
    #
    # Checks:
    # - Reaching the goal should stop the rollout early.
    raise NotImplementedError("Roll out a POET policy.")


def improve_policy(policy: Policy, env: PoetEnvironment, config: PoetConfig) -> tuple[Policy, float]:
    """Improve a policy by mutation and selection."""
    # Goal:
    # Try small policy changes and keep the best one for this environment.
    #
    # Method:
    # 1. Score the current policy.
    # 2. For `config.agent_mutations`, create mutated children.
    # 3. Score each child.
    # 4. Keep the policy with the highest score.
    #
    # Mutation:
    # Use log probabilities plus Gaussian noise, then normalize.
    #
    # Checks:
    # - The returned policy should be valid.
    # - The returned score should be the score of that policy.
    raise NotImplementedError("Improve one POET policy.")


def try_transfers(pairs: list[PoetPair], config: PoetConfig) -> int:
    """Try agents from other environments."""
    # Goal:
    # Let an environment borrow a better agent from another pair.
    #
    # Method:
    # For each target pair:
    # - score every other pair's policy in the target environment
    # - if one scores better than the target's current policy, copy it
    #
    # Output:
    # - number of successful transfers.
    #
    # Checks:
    # - Do not delete pairs.
    # - Update the target score when a transfer succeeds.
    raise NotImplementedError("Try POET policy transfers.")


def run_poet_style_loop(config: PoetConfig) -> PoetSummary:
    """Run the full POET-style open-ended loop."""
    # Goal:
    # Improve agents, generate environments, and try transfers.
    #
    # Output keys:
    # - `pair_count`
    # - `best_score`
    # - `generated_envs`
    # - `successful_transfers`
    # - `note`
    #
    # Steps:
    # 1. Set random seeds.
    # 2. Build the first environment and policy.
    # 3. Improve every pair's policy each iteration.
    # 4. Mutate environments to create new solvable tasks.
    # 5. Add new pairs until `config.max_pairs`.
    # 6. Every few iterations, try transfers between pairs.
    # 7. Return the summary.
    #
    # Checks:
    # - Generated environments should be solvable.
    # - The note should explain that POET grows tasks and agents together.
    raise NotImplementedError("Run POET-style open-ended learning.")


def main() -> None:
    """Run the POET-style scaffold."""
    config = PoetConfig()
    try:
        summary = run_poet_style_loop(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return
    print(summary)


if __name__ == "__main__":
    main()
