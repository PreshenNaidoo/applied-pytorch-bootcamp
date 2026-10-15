"""
============================================================
EXERCISE 25: NOVELTY SEARCH
============================================================

WHAT YOU ARE BUILDING
You are building a tiny novelty search loop.

Novelty search rewards new behavior.
It does not only reward task score.

Small example:
- Agent A ends near the start.
- Agent B reaches the top-right corner.
- Agent C reaches the bottom-left corner.

Agent B and C may be useful because they behave differently.
Novelty search keeps behavior from collapsing into one narrow pattern.

WHY THIS EXERCISE EXISTS
Open-ended systems need ways to discover new skills.
If the reward only says "reach the goal", the agent may ignore other useful behaviors.

This exercise uses a simple gridworld and hand-sized policies.

Policy shape:
- `[4]`
- one value per action: up, right, down, left

Behavior descriptor:
- `[2]`
- normalized final row and final column after one rollout

Novelty score:
- average distance to the nearest behavior descriptors in the archive

HOW THIS FITS INTO THE BOOTCAMP
Earlier exercises reward task success.
This exercise rewards behavior that is different.

Beginner intuition:
- normal reward asks "did it win?"
- novelty asks "did it do something new?"
- new behavior can later become useful even if it is not best right now

COMMON MISTAKES
- using task reward instead of novelty score
- forgetting to store old behaviors in the archive
- comparing raw states with different scales
- adding every behavior to the archive without a threshold
- selecting only one best policy and losing diversity

STEP-BY-STEP TODO
1. Implement the gridworld step.
2. Normalize policy tensors.
3. Build a starting population.
4. Roll out each policy.
5. Convert each rollout into a behavior descriptor.
6. Compute novelty from nearest archive behaviors.
7. Add novel behaviors to the archive.
8. Build the next generation from novel parents.
9. Report archive size and unique cells reached.

DEFINITION OF DONE
You are done when:
- each rollout has a behavior descriptor
- novelty comes from distance to old behaviors
- the archive grows when new behaviors appear
- the next generation favors novelty
- you can explain why novelty is different from reward
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TypedDict

import torch


# Learning order for this exercise:
# 1. Make the gridworld movement correct.
# 2. Make policy rollout produce a clear final state.
# 3. Turn final states into normalized descriptors.
# 4. Compute novelty before changing the population.
# 5. Add selection and mutation after the archive works.
#
# The goal is to reward new behavior in a concrete way.

State = tuple[int, int]
Policy = torch.Tensor


class NoveltySummary(TypedDict):
    """Final numbers returned by novelty search."""

    archive_size: int
    best_novelty: float
    unique_cells_reached: int
    note: str


@dataclass
class NoveltySearchConfig:
    """Settings for novelty search."""

    rows: int = 7
    cols: int = 7
    action_count: int = 4
    population_size: int = 24
    generations: int = 15
    rollout_steps: int = 12
    archive_add_threshold: float = 0.18
    nearest_k: int = 5
    mutation_scale: float = 0.35
    random_state: int = 42


def step_gridworld(state: State, action: int, config: NoveltySearchConfig) -> State:
    """Move one step inside the grid."""
    # Goal:
    # Return the next grid state.
    #
    # Actions:
    # - `0`: up
    # - `1`: right
    # - `2`: down
    # - `3`: left
    #
    # Steps:
    # 1. Check that the state is inside the grid.
    # 2. Check that the action is valid.
    # 3. Move the row or column.
    # 4. Clamp inside the grid.
    # 5. Return the next state.
    #
    # Checks:
    # - The returned state should always be inside the grid.
    raise NotImplementedError("Step the novelty gridworld.")


def normalize_policy(raw_policy: torch.Tensor) -> Policy:
    """Convert raw values into action probabilities."""
    # Goal:
    # Return a valid action policy.
    #
    # Input:
    # - tensor shaped `[4]`.
    #
    # Output:
    # - tensor shaped `[4]`.
    # - probabilities sum to `1.0`.
    #
    # Method:
    # Use `torch.softmax(raw_policy.float(), dim=0)`.
    #
    # Checks:
    # - No probability should be negative.
    # - The output should sum to `1.0`.
    raise NotImplementedError("Normalize one policy.")


def build_initial_population(config: NoveltySearchConfig) -> list[Policy]:
    """Build the first policy population."""
    # Goal:
    # Create a group of different starting policies.
    #
    # Output:
    # - list of policy tensors, each shaped `[4]`.
    #
    # Method:
    # Use random raw tensors and `normalize_policy(...)`.
    #
    # Steps:
    # 1. Set `torch.manual_seed(config.random_state)`.
    # 2. Create `config.population_size` raw tensors with `torch.randn(config.action_count)`.
    # 3. Normalize each raw tensor.
    # 4. Return the list.
    #
    # Checks:
    # - List length should equal `config.population_size`.
    raise NotImplementedError("Build novelty search population.")


def rollout_policy(policy: Policy, config: NoveltySearchConfig) -> list[State]:
    """Run one policy and return visited states."""
    # Goal:
    # Use a policy to move through the grid.
    #
    # Input:
    # - `policy`: action probabilities shaped `[4]`.
    #
    # Output:
    # - list of states, including the start state.
    #
    # Method:
    # Start from the middle of the grid:
    # `(config.rows // 2, config.cols // 2)`.
    #
    # At each step:
    # - sample an action with `torch.multinomial(policy, 1).item()`
    # - call `step_gridworld(...)`
    #
    # Checks:
    # - The list should contain `rollout_steps + 1` states.
    raise NotImplementedError("Roll out one policy.")


def behavior_descriptor(states: list[State], config: NoveltySearchConfig) -> torch.Tensor:
    """Return the final-position behavior descriptor."""
    # Goal:
    # Summarize one rollout with a small behavior vector.
    #
    # Input:
    # - `states`: list returned by `rollout_policy(...)`.
    #
    # Output:
    # - tensor shaped `[2]`.
    #
    # Descriptor:
    # - final row normalized by `config.rows - 1`
    # - final col normalized by `config.cols - 1`
    #
    # Example:
    # - final state `(6, 0)` on a 7 by 7 grid becomes `[1.0, 0.0]`.
    #
    # Checks:
    # - Both descriptor values should be between `0.0` and `1.0`.
    raise NotImplementedError("Compute behavior descriptor.")


def compute_novelty(descriptor: torch.Tensor, archive: list[torch.Tensor], config: NoveltySearchConfig) -> float:
    """Compute how different one behavior is from the archive."""
    # Goal:
    # Return a novelty score for one descriptor.
    #
    # Input:
    # - `descriptor`: `[2]`.
    # - `archive`: list of old descriptors.
    #
    # Method:
    # If the archive is empty, return a large value such as `1.0`.
    # Otherwise:
    # 1. Stack archive descriptors into `[archive_size, 2]`.
    # 2. Compute Euclidean distances with `torch.norm(stacked - descriptor, dim=1)`.
    # 3. Take the smallest `k` distances with `torch.topk(..., largest=False)`.
    # 4. Return their mean as a Python float.
    #
    # Checks:
    # - Novelty should be non-negative.
    raise NotImplementedError("Compute novelty score.")


def mutate_policy(policy: Policy, config: NoveltySearchConfig) -> Policy:
    """Create a changed copy of one policy."""
    # Goal:
    # Make a child policy that explores a slightly different behavior.
    #
    # Method:
    # Use:
    # ```python
    # logits = torch.log(policy.clamp_min(1e-6))
    # child = normalize_policy(logits + torch.randn_like(logits) * config.mutation_scale)
    # ```
    #
    # Checks:
    # - The child must still sum to `1.0`.
    raise NotImplementedError("Mutate one novelty-search policy.")


def update_archive(descriptors: list[torch.Tensor], scores: list[float], archive: list[torch.Tensor], config: NoveltySearchConfig) -> None:
    """Add novel descriptors to the archive."""
    # Goal:
    # Keep a memory of behaviors that were different enough.
    #
    # Input:
    # - `descriptors`: descriptors from the current generation.
    # - `scores`: novelty scores for those descriptors.
    # - `archive`: list updated in place.
    #
    # Rule:
    # Add a descriptor when its novelty score is at least
    # `config.archive_add_threshold`.
    #
    # Checks:
    # - This function should update `archive` in place.
    raise NotImplementedError("Update the novelty archive.")


def next_generation(population: list[Policy], novelty_scores: list[float], config: NoveltySearchConfig) -> list[Policy]:
    """Build a new population from novel parents."""
    # Goal:
    # Keep the most novel policies and mutate them.
    #
    # Method:
    # Sort policy indexes by novelty score from high to low.
    # Keep the top half as parents.
    # Mutate parents until the new population reaches `population_size`.
    #
    # Checks:
    # - The returned population length should equal `config.population_size`.
    raise NotImplementedError("Build next novelty-search generation.")


def run_novelty_search(config: NoveltySearchConfig) -> NoveltySummary:
    """Run the novelty search loop."""
    # Goal:
    # Search for many different gridworld behaviors.
    #
    # Output keys:
    # - `archive_size`
    # - `best_novelty`
    # - `unique_cells_reached`
    # - `note`
    #
    # Steps:
    # 1. Build the first population.
    # 2. Create an empty archive.
    # 3. For each generation, roll out every policy.
    # 4. Turn rollouts into descriptors.
    # 5. Compute novelty scores.
    # 6. Update the archive.
    # 7. Build the next generation from novel policies.
    # 8. Return the summary.
    #
    # Checks:
    # - The note should explain that novelty is about being different, not only winning.
    raise NotImplementedError("Run novelty search.")


def main() -> None:
    """Run the novelty search scaffold."""
    config = NoveltySearchConfig()
    try:
        summary = run_novelty_search(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return
    print(summary)


if __name__ == "__main__":
    main()
