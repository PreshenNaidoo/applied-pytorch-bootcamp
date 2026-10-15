"""
============================================================
EXERCISE 08: SIM TO REAL TRANSFER TOY STUDY
============================================================

WHAT YOU ARE BUILDING
You are building a transfer study for a visual embodied policy.

Plain-language meanings:
- sim-to-real gap means the simulator differs from the real world
- domain randomization means changing simulator details during training
- visual shift means colors, lighting, or textures change
- dynamics shift means movement rules change

WHY THIS EXERCISE EXISTS
Robots trained in simulation often fail when the real world looks or moves differently.
This exercise teaches the basic evaluation shape.

For this bootcamp path, use the shared visual grid-world as the source
simulator. The target world should change things the policy can actually feel,
such as colors, observation noise, wall count, hazard count, or movement cost.
The point is to test whether a trained policy still works when the world is not
exactly like the training world.

DATASET OPTIONS
Default: train and test on changed versions of `common.embodied_gridworld`.

Exact local data method:
- source world: normal `VisualGridWorldConfig`
- target world: changed colors, noise, wall count, hazard count, or movement scale
- domain-randomized source: sample small random changes during training or scripted data collection

Real dataset options after the default works:
- `lerobot/aloha_sim_transfer_cube_human` and `lerobot/aloha_sim_insertion_human` on Hugging Face as two related simulated robot tasks.
- `lerobot/pusht` on Hugging Face for a different action and observation setup.

Use these to test whether the same data loader, policy code, and metrics still
work when the task or observation style changes.

MAIN LOOP IN PLAIN WORDS

```text
train or script an agent in one world
test it in a changed world
add random changes during training
test again
compare transfer
```

METRICS TO TRACK
- source success rate
- target success rate
- transfer gap: `source_success_rate - target_success_rate`
- randomized target success rate
- cost rate in each world

VISUALS TO CREATE
Save `outputs/sim_to_real_transfer/transfer_gap.png`.
It should compare source, target, and domain-randomized target results.

DEFINITION OF DONE
You are done when:
- a source world and changed target world are created
- a trained or scripted baseline policy is evaluated in both
- domain randomization is added
- transfer metrics are compared
- a transfer-gap chart is saved
- you can explain what changed between sim and target
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
from common.embodied_gridworld import VisualGridWorldConfig


class ToyWorld(TypedDict):
    """Small world settings for transfer tests."""

    goal_color: torch.Tensor
    hazard_color: torch.Tensor
    wall_count: int
    hazard_count: int
    movement_scale: float
    observation_noise: float


@dataclass
class SimToRealConfig:
    """Settings for the toy transfer study."""

    num_episodes: int = 20
    max_steps: int = 18
    domain_randomization_strength: float = 0.2
    random_state: int = 42


def build_source_and_target_worlds() -> tuple[ToyWorld, ToyWorld]:
    """Return one training world and one changed test world."""
    # Goal:
    # Create a source world and a target world that differ in clear ways.
    #
    # Output:
    # - `(source_world, target_world)`
    #
    # Method:
    # Source world:
    # - normal colors
    # - normal wall and hazard count
    # - movement scale 1.0
    #
    # Target world:
    # - shifted colors
    # - changed wall or hazard count
    # - different movement scale
    # - more observation noise
    #
    # Checks:
    # - The target world should not be identical to the source world.
    raise NotImplementedError("Build source and target toy worlds.")


def apply_domain_randomization(world: ToyWorld, strength: float) -> ToyWorld:
    """Return a randomized copy of one world."""
    # Goal:
    # Change simulator details during training so the policy sees variation.
    #
    # Input:
    # - `world`: ToyWorld
    # - `strength`: size of the random changes.
    #
    # Output:
    # - randomized ToyWorld.
    #
    # Method:
    # Randomly shift colors, movement scale, noise, wall count, and hazard count.
    #
    # Use `random.Random(...)` or a passed-in random generator in your solution
    # so the study can be repeated with the same seed.
    #
    # Checks:
    # - Do not change the original dictionary in place.
    _ = random
    raise NotImplementedError("Apply domain randomization.")


def evaluate_policy_in_world(policy_name: str, world: ToyWorld, config: SimToRealConfig) -> dict[str, float]:
    """Evaluate a policy in one changed visual grid-world."""
    # Goal:
    # Return transfer metrics for a policy in a world.
    #
    # Input:
    # - `policy_name`: for example `"baseline"` or `"randomized_training"`
    # - `world`: ToyWorld
    #
    # Output:
    # - dictionary with:
    #   - `"success_rate"`
    #   - `"collision_rate"`
    #   - `"mean_return"`
    #
    # Method:
    # 1. Convert `world` into `VisualGridWorldConfig(...)`.
    # 2. Run `config.num_episodes` real episodes.
    # 3. Use one of these policies:
    #    - `"random"`: sample random actions
    #    - `"scripted"`: use the shortest-path helper from the shared environment
    #    - `"randomized_training"`: evaluate the policy trained with randomized data
    # 4. Record success, collisions or hazards, total reward, and total cost.
    #
    # Later project version:
    # - replace the scripted policy with the behavior-cloning policy from the related exercise
    # - compare normal training data with domain-randomized training data
    #
    # Checks:
    # - Metrics should be Python floats.
    _ = VisualGridWorldConfig
    raise NotImplementedError("Evaluate policy in one toy world.")


def train_with_domain_randomization(config: SimToRealConfig) -> dict[str, float]:
    """Train or collect randomized data, then test transfer."""
    # Goal:
    # Run the whole transfer study end to end.
    #
    # Output:
    # - dictionary with:
    #   - `"source_success_rate"`
    #   - `"target_success_rate"`
    #   - `"target_success_rate_with_randomization"`
    #   - `"transfer_gap"`
    #
    # Method:
    # 1. Build source and target worlds.
    # 2. Evaluate the baseline policy in the source world.
    # 3. Evaluate the same baseline in the target world.
    # 4. Build several randomized source worlds with `apply_domain_randomization(...)`.
    # 5. Train or script using those randomized worlds.
    # 6. Evaluate again in the target world.
    # 7. Compare target performance before and after randomization.
    #
    # Checks:
    # - The target world should not be used for training.
    # - A useful randomization setting should reduce the transfer gap.
    raise NotImplementedError("Run sim-to-target transfer study.")


def compare_transfer(source_metrics: dict[str, float], target_metrics: dict[str, float]) -> dict[str, float]:
    """Compare source and target performance."""
    # Goal:
    # Measure how much performance dropped after the world changed.
    #
    # Output:
    # - dictionary with:
    #   - `"success_rate_drop"`
    #   - `"return_drop"`
    #
    # Method:
    # `drop = source_value - target_value`
    raise NotImplementedError("Compare source and target metrics.")


def explain_sim_to_real() -> str:
    """Return a short sim-to-real explanation."""
    return "Sim-to-real transfer asks whether a policy trained in simulation still works when the world changes."


def main() -> None:
    """Run the full sim-to-target transfer scaffold."""
    config = SimToRealConfig()
    try:
        metrics = train_with_domain_randomization(config)
    except NotImplementedError as error:
        print(f"[TODO] train_with_domain_randomization: {error}")
        return
    print(metrics)


if __name__ == "__main__":
    main()
