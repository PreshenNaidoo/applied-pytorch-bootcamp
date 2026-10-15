"""
============================================================
EXERCISE 26: QUALITY DIVERSITY
============================================================

WHAT YOU ARE BUILDING
You are building a tiny quality-diversity system.

Quality-diversity means:
- keep solutions that are good
- also keep many different kinds of solutions

This is different from normal training.
Normal training often keeps only the single best solution.

WHY THIS EXERCISE EXISTS
Open-ended systems need a library of useful behaviors.
One best policy is often not enough.

Small example:
- one policy reaches the top-left area
- one policy reaches the top-right area
- one policy reaches the bottom-left area

Each policy may be useful later, even if only one has the highest score.

This exercise uses a small MAP-Elites style grid.
MAP-Elites means:
- split behavior space into bins
- keep the best policy found for each bin

Behavior descriptor:
- final row and final column after a rollout

Archive:
- dictionary from `(row_bin, col_bin)` to an elite policy and its score

HOW THIS FITS INTO THE BOOTCAMP
Novelty search keeps new behavior.
Quality-diversity keeps behavior that is both different and useful.

Beginner intuition:
- novelty asks for "different"
- quality asks for "good"
- quality-diversity asks for "many different good things"

COMMON MISTAKES
- keeping only the global best policy
- replacing an elite with a worse policy in the same bin
- using descriptor values without clamping bin indexes
- confusing behavior score with behavior descriptor
- making bins too vague to show diversity

STEP-BY-STEP TODO
1. Implement gridworld stepping.
2. Normalize and sample policies.
3. Mutate policies.
4. Roll out a policy.
5. Compute its quality score.
6. Compute its behavior descriptor.
7. Map the descriptor to an archive bin.
8. Store the policy only if it is best for that bin.
9. Run the MAP-Elites style loop.

DEFINITION OF DONE
You are done when:
- the archive stores one elite per bin
- better policies replace worse policies in the same bin
- different bins can hold different behaviors
- the summary reports filled bins and elite quality
- you can explain quality vs diversity
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TypedDict

import torch


# Learning order for this exercise:
# 1. Make one rollout work.
# 2. Separate the quality score from the descriptor.
# 3. Make descriptor-to-bin mapping correct.
# 4. Update the archive for one policy.
# 5. Run the loop only after archive updates are clear.
#
# The goal is to keep many useful behaviors, not only one winner.

State = tuple[int, int]
Policy = torch.Tensor
Bin = tuple[int, int]


class Elite(TypedDict):
    """One stored solution in the archive."""

    policy: Policy
    score: float
    descriptor: torch.Tensor


class QualityDiversitySummary(TypedDict):
    """Final numbers returned by quality-diversity."""

    filled_bins: int
    best_score: float
    mean_elite_score: float
    note: str


@dataclass
class QualityDiversityConfig:
    """Settings for quality-diversity."""

    rows: int = 7
    cols: int = 7
    action_count: int = 4
    bin_count: int = 5
    iterations: int = 300
    rollout_steps: int = 14
    mutation_scale: float = 0.35
    random_state: int = 42


def step_gridworld(state: State, action: int, config: QualityDiversityConfig) -> State:
    """Move one step in the grid."""
    # Goal:
    # Return the next state after one action.
    #
    # Actions:
    # - `0`: up
    # - `1`: right
    # - `2`: down
    # - `3`: left
    #
    # Steps:
    # 1. Check state and action.
    # 2. Move row or column.
    # 3. Clamp inside the grid.
    # 4. Return the next state.
    raise NotImplementedError("Step the quality-diversity gridworld.")


def normalize_policy(raw_policy: torch.Tensor) -> Policy:
    """Return action probabilities."""
    # Goal:
    # Convert raw policy values into a valid probability vector.
    #
    # Input:
    # - tensor shaped `[4]`.
    #
    # Method:
    # Use `torch.softmax(raw_policy.float(), dim=0)`.
    #
    # Checks:
    # - The output should sum to `1.0`.
    raise NotImplementedError("Normalize quality-diversity policy.")


def sample_random_policy(config: QualityDiversityConfig) -> Policy:
    """Create one random policy."""
    # Goal:
    # Return a new random policy.
    #
    # Method:
    # Use:
    # ```python
    # raw = torch.randn(config.action_count)
    # return normalize_policy(raw)
    # ```
    #
    # Checks:
    # - Shape should be `(config.action_count,)`.
    raise NotImplementedError("Sample random policy.")


def mutate_policy(policy: Policy, config: QualityDiversityConfig) -> Policy:
    """Create one child policy from one parent policy."""
    # Goal:
    # Slightly change an existing policy.
    #
    # Method:
    # Use log probabilities plus Gaussian noise:
    # ```python
    # logits = torch.log(policy.clamp_min(1e-6))
    # noise = torch.randn_like(logits) * config.mutation_scale
    # child = normalize_policy(logits + noise)
    # ```
    #
    # Checks:
    # - The child should be a valid probability vector.
    raise NotImplementedError("Mutate quality-diversity policy.")


def rollout_policy(policy: Policy, config: QualityDiversityConfig) -> list[State]:
    """Run a policy and return visited states."""
    # Goal:
    # Produce one behavior from one policy.
    #
    # Method:
    # Start in the middle of the grid.
    # Sample actions with `torch.multinomial(policy, 1).item()`.
    #
    # Output:
    # - list of visited states.
    #
    # Checks:
    # - The first state should be the start state.
    # - The list length should be `rollout_steps + 1`.
    raise NotImplementedError("Roll out one quality-diversity policy.")


def score_rollout(states: list[State], config: QualityDiversityConfig) -> float:
    """Return a quality score for one rollout."""
    # Goal:
    # Score how useful the behavior is.
    #
    # Teaching score:
    # Reward getting far from the start.
    #
    # Formula:
    # ```text
    # score = abs(final_row - start_row) + abs(final_col - start_col)
    # ```
    #
    # This distance is called Manhattan distance.
    # It counts grid steps, not diagonal distance.
    #
    # Checks:
    # - Staying at the start should get score `0.0`.
    raise NotImplementedError("Score one rollout.")


def behavior_descriptor(states: list[State], config: QualityDiversityConfig) -> torch.Tensor:
    """Return normalized final position."""
    # Goal:
    # Describe where the policy ended.
    #
    # Output:
    # - tensor shaped `[2]`.
    # - values are final row and col normalized to `0.0` through `1.0`.
    #
    # Checks:
    # - Both values should be between `0.0` and `1.0`.
    raise NotImplementedError("Compute quality-diversity descriptor.")


def descriptor_to_bin(descriptor: torch.Tensor, config: QualityDiversityConfig) -> Bin:
    """Map a descriptor to one archive bin."""
    # Goal:
    # Convert a continuous descriptor into a discrete archive cell.
    #
    # Input:
    # - descriptor shaped `[2]`, with values from `0.0` to `1.0`.
    #
    # Output:
    # - `(row_bin, col_bin)`.
    #
    # Formula:
    # ```text
    # bin_index = int(value * bin_count)
    # ```
    #
    # Clamp the result so the largest value maps to `bin_count - 1`.
    #
    # Example:
    # - with `bin_count = 5`, value `0.0` maps to bin `0`
    # - value `0.99` maps to bin `4`
    #
    # Checks:
    # - Each bin index should be between `0` and `bin_count - 1`.
    raise NotImplementedError("Map descriptor to archive bin.")


def update_archive(archive: dict[Bin, Elite], policy: Policy, score: float, descriptor: torch.Tensor, config: QualityDiversityConfig) -> None:
    """Store the policy if it is the best for its bin."""
    # Goal:
    # Keep the best policy found for one behavior bin.
    #
    # Rule:
    # - If the bin is empty, store the policy.
    # - If the bin already has an elite, replace it only when `score` is higher.
    #
    # Steps:
    # 1. Convert descriptor to a bin.
    # 2. Check whether the bin is missing from `archive`.
    # 3. If missing, store the new elite.
    # 4. If present, compare scores and keep the better one.
    #
    # Checks:
    # - This function should update `archive` in place.
    raise NotImplementedError("Update quality-diversity archive.")


def choose_parent_from_archive(archive: dict[Bin, Elite], config: QualityDiversityConfig) -> Policy:
    """Choose one stored elite to mutate."""
    # Goal:
    # Pick an existing elite as a parent.
    #
    # Method:
    # Convert `archive.values()` to a list.
    # Use `random.choice(elites)` to choose one.
    #
    # If the archive is empty, return `sample_random_policy(config)`.
    #
    # Checks:
    # - The returned policy should be valid.
    raise NotImplementedError("Choose parent from archive.")


def run_quality_diversity(config: QualityDiversityConfig) -> QualityDiversitySummary:
    """Run the MAP-Elites style loop."""
    # Goal:
    # Fill an archive with many different useful policies.
    #
    # Output keys:
    # - `filled_bins`
    # - `best_score`
    # - `mean_elite_score`
    # - `note`
    #
    # Steps:
    # 1. Set random seeds.
    # 2. Create an empty archive.
    # 3. For each iteration, choose a parent or random policy.
    # 4. Mutate it.
    # 5. Roll it out.
    # 6. Score the rollout.
    # 7. Compute its descriptor.
    # 8. Update the archive.
    # 9. Return final archive stats.
    #
    # Checks:
    # - The note should explain the difference between quality and diversity.
    raise NotImplementedError("Run quality-diversity search.")


def main() -> None:
    """Run the quality-diversity scaffold."""
    config = QualityDiversityConfig()
    try:
        summary = run_quality_diversity(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return
    print(summary)


if __name__ == "__main__":
    main()
