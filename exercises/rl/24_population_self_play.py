"""
============================================================
EXERCISE 24: POPULATION SELF PLAY
============================================================

WHAT YOU ARE BUILDING
You are building a tiny population self-play system.

Self-play means an agent improves by playing against other agents.
Population self-play means there are many agents, not just one.

WHY THIS EXERCISE EXISTS
Training against only the newest agent can be unstable.
The agent may forget how to beat older styles.

A population helps because it keeps many opponents:
- old versions
- current versions
- mutated versions
- strong and weak versions

This exercise uses Rock-Paper-Scissors so the game rules stay simple.
The learning idea is the important part.

Action ids:
- `0`: rock
- `1`: paper
- `2`: scissors

Policy shape:
- one policy is a tensor shaped `[3]`
- values are action probabilities and must sum to `1.0`

Payoff:
- win: `1.0`
- draw: `0.0`
- loss: `-1.0`

HOW THIS FITS INTO THE BOOTCAMP
Earlier self-play exercises train in fixed games.
This exercise adds a population of opponents.

Beginner intuition:
- one opponent can make the agent overfit
- many opponents expose more styles
- older opponents can show when the agent forgot a useful skill

COMMON MISTAKES
- keeping only the newest best policy
- forgetting to normalize mutated policies
- scoring a policy only against one opponent
- using one sampled game as if it were a stable score
- not keeping older policies around

STEP-BY-STEP TODO
1. Normalize policy tensors.
2. Build several starting policies.
3. Implement Rock-Paper-Scissors payoff.
4. Sample actions from a policy.
5. Play repeated matches.
6. Score each policy against the population.
7. Mutate policies.
8. Keep strong parents and add children.
9. Run several generations.

DEFINITION OF DONE
You are done when:
- policies are valid probability vectors
- each policy is scored against many opponents
- mutation creates valid child policies
- the population changes over generations
- you can explain why a population helps reduce forgetting
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TypedDict

import torch


# Learning order for this exercise:
# 1. Make action probabilities valid.
# 2. Make the game payoff correct.
# 3. Score policies through repeated matches.
# 4. Add mutation only after scoring works.
# 5. Run the generation loop after selection and reproduction work.
#
# The goal is to see why many opponents are better than one.

Policy = torch.Tensor


class PopulationSummary(TypedDict):
    """Final numbers returned by population self-play."""

    population_size: int
    best_score: float
    average_score: float
    note: str


@dataclass
class PopulationSelfPlayConfig:
    """Settings for population self-play."""

    action_count: int = 3
    initial_population_size: int = 4
    generations: int = 12
    matches_per_pair: int = 20
    keep_top_k: int = 4
    children_per_generation: int = 4
    mutation_scale: float = 0.25
    random_state: int = 42


def normalize_policy(logits_or_weights: torch.Tensor) -> Policy:
    """Return a valid action-probability vector."""
    # Goal:
    # Convert any length-3 tensor into probabilities.
    #
    # Input:
    # - tensor shaped `[3]`.
    #
    # Output:
    # - tensor shaped `[3]`.
    # - all values are non-negative.
    # - values sum to `1.0`.
    #
    # Method:
    # Use `torch.softmax(logits_or_weights.float(), dim=0)`.
    #
    # Checks:
    # - `policy.shape == (3,)`.
    # - `torch.isclose(policy.sum(), torch.tensor(1.0))` should be true.
    raise NotImplementedError("Normalize one policy.")


def build_initial_population(config: PopulationSelfPlayConfig) -> list[Policy]:
    """Build the starting policy population."""
    # Goal:
    # Return several different policies.
    #
    # Output:
    # - list of tensors, each shaped `[3]`.
    #
    # Suggested starting policies:
    # - mostly rock
    # - mostly paper
    # - mostly scissors
    # - almost uniform
    #
    # Method:
    # Use `normalize_policy(...)` on hand-written tensors.
    #
    # Steps:
    # 1. Create a list of raw tensors.
    # 2. Normalize each one.
    # 3. Return the list.
    #
    # Checks:
    # - The list length should be at least `config.initial_population_size`.
    # - Every policy should sum to `1.0`.
    raise NotImplementedError("Build initial policy population.")


def rps_payoff(action_a: int, action_b: int) -> float:
    """Return player A's payoff against player B."""
    # Goal:
    # Implement Rock-Paper-Scissors payoff.
    #
    # Rules:
    # - rock beats scissors
    # - paper beats rock
    # - scissors beats paper
    #
    # Input:
    # - `action_a`: action for player A.
    # - `action_b`: action for player B.
    #
    # Output:
    # - `1.0` if A wins.
    # - `0.0` if draw.
    # - `-1.0` if A loses.
    #
    # Checks:
    # - `rps_payoff(0, 2) == 1.0`.
    # - `rps_payoff(0, 1) == -1.0`.
    # - `rps_payoff(1, 1) == 0.0`.
    raise NotImplementedError("Return Rock-Paper-Scissors payoff.")


def sample_action(policy: Policy) -> int:
    """Sample one action from one policy."""
    # Goal:
    # Draw one action according to the policy probabilities.
    #
    # Method:
    # Use:
    # `torch.multinomial(policy, num_samples=1).item()`
    #
    # Checks:
    # - The returned action should be `0`, `1`, or `2`.
    raise NotImplementedError("Sample one policy action.")


def play_match(policy_a: Policy, policy_b: Policy, config: PopulationSelfPlayConfig) -> float:
    """Return player A's average score against player B."""
    # Goal:
    # Play repeated sampled games between two policies.
    #
    # Output:
    # - average payoff for player A.
    #
    # Steps:
    # 1. For `config.matches_per_pair` rounds, sample one action from each policy.
    # 2. Compute `rps_payoff(action_a, action_b)`.
    # 3. Average the payoff values.
    # 4. Return the average.
    #
    # Checks:
    # - A policy that always plays paper should beat one that always plays rock.
    raise NotImplementedError("Play one repeated match.")


def score_population(population: list[Policy], config: PopulationSelfPlayConfig) -> torch.Tensor:
    """Score each policy against all other policies."""
    # Goal:
    # Return one average score per policy.
    #
    # Output:
    # - tensor shaped `[population_size]`.
    #
    # Method:
    # For each policy, play it against every other policy.
    # Do not score a policy against itself unless you handle the draw clearly.
    #
    # Steps:
    # 1. Create a zero score tensor.
    # 2. Loop over every pair `(i, j)`.
    # 3. Skip when `i == j`.
    # 4. Add match score for policy `i`.
    # 5. Divide by the number of opponents.
    #
    # Checks:
    # - Output length should equal `len(population)`.
    raise NotImplementedError("Score the policy population.")


def mutate_policy(parent: Policy, config: PopulationSelfPlayConfig) -> Policy:
    """Create one child policy from a parent policy."""
    # Goal:
    # Make a slightly changed copy of one policy.
    #
    # Method:
    # Convert the parent into logits with:
    # `torch.log(parent.clamp_min(1e-6))`
    #
    # Then add noise:
    # `noise = torch.randn_like(logits) * config.mutation_scale`
    #
    # Finally normalize with `normalize_policy(logits + noise)`.
    #
    # Checks:
    # - The child should be a valid probability vector.
    # - The child should not share storage with the parent.
    raise NotImplementedError("Mutate one policy.")


def select_and_reproduce(population: list[Policy], scores: torch.Tensor, config: PopulationSelfPlayConfig) -> list[Policy]:
    """Keep strong policies and add mutated children."""
    # Goal:
    # Build the next generation.
    #
    # Method:
    # Use `torch.topk(scores, k=config.keep_top_k).indices`
    # to choose parents.
    #
    # Steps:
    # 1. Select the top `keep_top_k` policies.
    # 2. Copy those policies into the next population.
    # 3. Create `children_per_generation` mutated children from selected parents.
    # 4. Return the new population.
    #
    # Checks:
    # - The new population should include kept parents and new children.
    # - Keep the population size manageable.
    raise NotImplementedError("Select parents and make children.")


def run_population_self_play(config: PopulationSelfPlayConfig) -> PopulationSummary:
    """Run the population self-play loop."""
    # Goal:
    # Let a population improve over generations.
    #
    # Output keys:
    # - `population_size`
    # - `best_score`
    # - `average_score`
    # - `note`
    #
    # Steps:
    # 1. Set random seeds.
    # 2. Build the initial population.
    # 3. For each generation, score the population.
    # 4. Select strong policies.
    # 5. Add mutated children.
    # 6. Score the final population.
    # 7. Return the summary.
    #
    # Checks:
    # - The note should explain why a population helps avoid forgetting.
    raise NotImplementedError("Run population self-play.")


def main() -> None:
    """Run the population self-play scaffold."""
    config = PopulationSelfPlayConfig()
    try:
        summary = run_population_self_play(config)
    except NotImplementedError as error:
        print(f"[TODO] {error}")
        return
    print(summary)


if __name__ == "__main__":
    main()
