"""
============================================================
EXERCISE 14: 3D WORLD MODEL PLANNER
============================================================

WHAT YOU ARE BUILDING
You are using a learned 3D latent world model to plan actions.

The planner tries possible future action sequences inside the model:

```text
sample actions -> imagine latents -> score reward - cost - step penalty
```

Then it takes only the first action in the real environment and plans again.

In this file, `planner` means search logic.
It is usually not a neural network.
It uses the world model from the related exercise to test possible futures.

WHY THIS EXERCISE EXISTS
This is the small generalist visual-action project loop:

```text
see -> remember or encode -> imagine -> choose action -> act -> evaluate
```

DATASET AND DOWNLOADS
Default setup:
- no new dataset download
- use the environment from the related exercise
- use the learned world model from the related exercise
- optionally compare against the behavior-cloning policy from the related exercise

The planner is evaluated online:
it acts in the real local 3D environment and gets real reward, cost, and done.

WHAT IS HIDDEN FROM THE PLANNER
- the full map
- the true target and hazard positions, unless they are visible in the input
- the real next state while it is imagining

The planner may use only:
- current latent or encoded observation
- instruction ids
- the learned world model
- the scoring rule

SCORING RULE

```text
score = total_predicted_reward - total_predicted_cost - step_penalty * number_of_steps
```

Plain meaning:
- reward says the imagined path helps the task
- cost says the imagined path is risky
- step penalty stops the agent from wasting time

METRICS TO TRACK
- success rate: higher is better
- hazard rate: lower is better
- mean path length: lower is better after success
- mean cost: lower is better
- behavior-cloning success rate: comparison baseline

VISUALS TO CREATE
Save `outputs/mini3d_planner/planner_vs_bc.png`.
It should compare behavior cloning and world-model planning on success, cost,
and path length.

TARGET RESULT RANGE
Do not expect the first planner to be strong.
The important first result is that it runs without peeking at the hidden map and
produces real environment metrics.

Useful signs:
- planner success is better than random, or cost is lower than random
- the planner does not always choose the same action
- increasing `num_sequences` changes behavior or improves score
- failures can be inspected with saved frame strips

COMMON MISTAKES
- executing the whole imagined action sequence instead of only the first action
- using true future rewards while scoring imagined plans
- forgetting the cost term, so the planner chooses risky paths
- comparing planner and behavior cloning on different episode seeds
- reporting model loss instead of real rollout success

STEP-BY-STEP TODO
1. Sample candidate action sequences.
2. Score one sequence inside the learned model.
3. Pick the first action from the best sequence.
4. Run the planner in the real 3D environment.
5. Run the behavior-cloning baseline on the same episodes.
6. Compare success, cost, and path length.
7. Save planner-vs-baseline charts and failure frame strips.

DEFINITION OF DONE
You are done when:
- action sequences are sampled
- imagined futures are scored
- the planner runs in the real 3D environment
- behavior cloning and world-model planning are compared
- failure-case videos or frame strips are saved
- planner-vs-baseline chart is saved
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

import torch


class LatentPlannerModel(Protocol):
    """Interface needed by the 3D planner."""

    def __call__(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        instruction_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return next latent, reward, cost, and done logit."""


class Planner3DMetrics(TypedDict):
    """3D planner evaluation metrics."""

    success_rate: float
    hazard_rate: float
    mean_path_length: float
    mean_cost: float
    bc_success_rate: float


@dataclass
class WorldModelPlanner3DConfig:
    """Settings for 3D world-model planning."""

    num_actions: int = 6
    planning_horizon: int = 6
    num_sequences: int = 128
    eval_episodes: int = 32
    max_steps: int = 40
    step_penalty: float = 0.01
    random_state: int = 42


def sample_3d_action_sequences(config: WorldModelPlanner3DConfig) -> torch.Tensor:
    """Return candidate future actions."""
    # Where this fits:
    # The planner needs possible futures before it can choose one.
    #
    # Goal:
    # Sample many possible action sequences for the model to imagine.
    #
    # Output:
    # - tensor shaped `[num_sequences, planning_horizon]`
    # - each value is an action id from `0` to `num_actions - 1`
    #
    # Method:
    # Use this exact PyTorch call:
    # `torch.randint(low=0, high=config.num_actions, size=(config.num_sequences, config.planning_horizon))`
    #
    # Check:
    # - `actions.min() >= 0`
    # - `actions.max() < config.num_actions`
    raise NotImplementedError("Sample 3D planner action sequences.")


def choose_first_action_from_best_sequence(scores: torch.Tensor, action_sequences: torch.Tensor) -> int:
    """Return the first action from the highest-scoring sequence."""
    # Where this fits:
    # After the model imagines many futures, the real agent should take only one
    # action, then observe the world and plan again.
    #
    # Input:
    # - `scores`: `[num_sequences]`, one score per imagined sequence.
    # - `action_sequences`: `[num_sequences, planning_horizon]`.
    #
    # Output:
    # - one Python int action id.
    #
    # Method:
    # 1. Find `best_index = scores.argmax().item()`.
    # 2. Read `first_action = action_sequences[best_index, 0]`.
    # 3. Return `int(first_action.item())`.
    #
    # Check:
    # - Do not return the full sequence.
    _ = (scores, action_sequences)
    raise NotImplementedError("Choose the first action from the best sequence.")


def score_3d_imagined_rollout(
    model: LatentPlannerModel,
    start_latent: torch.Tensor,
    actions: torch.Tensor,
    instruction_ids: torch.Tensor,
    config: WorldModelPlanner3DConfig,
) -> float:
    """Score one imagined 3D rollout."""
    # Where this fits:
    # This function never touches the real environment. It only asks the learned
    # model what might happen.
    #
    # Input:
    # - `start_latent`: `[latent_dim]`
    # - `actions`: `[planning_horizon]`
    # - `instruction_ids`: token ids for the goal
    #
    # Score:
    # `sum(predicted_reward) - sum(predicted_cost) - step_penalty * horizon`
    #
    # Steps:
    # 1. Set `latent = start_latent`.
    # 2. For each action in `actions`:
    #    - call `model(latent, action, instruction_ids)`
    #    - read predicted reward and cost
    #    - add reward and cost to running totals
    #    - set `latent = next_latent`
    # 3. Return the score formula above as a Python float.
    #
    # Check:
    # - Do not use real future frames or true rewards here.
    _ = (model, start_latent, actions, instruction_ids, config)
    raise NotImplementedError("Score one imagined 3D rollout.")


def evaluate_3d_world_model_planner(
    model: LatentPlannerModel,
    config: WorldModelPlanner3DConfig,
) -> Planner3DMetrics:
    """Evaluate world-model planning in the real 3D environment."""
    # Goal:
    # Compare the planner with the behavior-cloning baseline from the related exercise.
    #
    # Method:
    # 1. Reset the 3D environment from the related exercise.
    # 2. Encode the current image into a latent.
    # 3. Sample candidate action sequences.
    # 4. Score each sequence with `score_3d_imagined_rollout(...)`.
    # 5. Take only the first action from the best sequence.
    # 6. Step the real environment once.
    # 7. Repeat until done or timeout.
    # 8. Report success, hazard rate, cost, and path length.
    #
    # Checks:
    # - Report task behavior, not only model loss.
    # - Do not let the planner see the hidden full map during imagined scoring.
    # - Compare against behavior cloning on the same number of episodes.
    _ = (model, config)
    raise NotImplementedError("Evaluate the 3D world-model planner.")


def main() -> None:
    """Run the 3D world-model planner scaffold."""
    print("TODO: train the related exercise's model, then pass it to evaluate_3d_world_model_planner.")


if __name__ == "__main__":
    main()
