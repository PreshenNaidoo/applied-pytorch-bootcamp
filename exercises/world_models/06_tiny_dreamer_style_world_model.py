"""
============================================================
EXERCISE 06: TINY DREAMER STYLE WORLD MODEL
============================================================

WHAT YOU ARE BUILDING
You are building a small Dreamer-style training loop.

Dreamer-style means:
- train a world model from real environment data
- start imagined rollouts from real states
- train a policy through imagined futures
- check the final policy in the real environment

The small version here uses the full grid-world state as the latent state.
The full state keeps the agent, goal, all hazards, and all walls.
It is flattened before it goes into the world model or policy.

```text
real logged transition:
flattened full_state_t + action_t -> flattened full_state_t+1, reward_t, cost_t, done_t

imagined policy training:
real latent_t -> policy -> soft action -> world model -> imagined reward/cost
```

Plain meaning:
- latent means the flattened full grid state used by the world model
- policy means the model that chooses actions
- imagined rollout means using the learned world model instead of the real environment
- real evaluation means stepping the real environment rules after policy training

WHY THIS EXERCISE EXISTS
the related exercise trained a visual latent world model.
This exercise teaches the Dreamer idea: use a learned model to train behavior
from imagined futures.

The important split is:

```text
real data trains the world model
imagined data trains the policy
real environment checks the final policy
```

HOW THIS CONNECTS TO DREAMERV3
DreamerV3 keeps the same main loop, but makes it much more stable across very
different tasks.

Important DreamerV3 ideas to recognize:

- a recurrent state-space model carries deterministic memory through time
- a categorical stochastic latent represents uncertain parts of the state
- KL loss keeps predicted and encoded latent distributions close
- free bits stop the KL term from crushing useful information too early
- `symlog` compresses very large positive and negative reward or value targets
- two-hot targets place probability on two nearby value bins
- return normalization keeps actor and critic updates on a useful scale

This exercise does not implement all of those pieces.
It first teaches the shared real-data -> imagination -> real-evaluation loop.

After the base version works, add these two small DreamerV3 upgrades:

```text
upgrade 1:
replace raw reward regression with symlog reward regression

upgrade 2:
add a stochastic categorical latent and a balanced KL loss
```

Report each upgrade as an ablation.
Do not call the small model DreamerV3 unless those modern parts are present.

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
robot/game environment              shared visual grid-world
recurrent stochastic latent         flattened full grid state
motor/keyboard action               action id 0, 1, 2, or 3
learned dynamics model              small PyTorch world model
actor/policy network                small latent policy
imagined rollout                    multi-step rollout inside the world model
real evaluation                     held-out grid-world scenes

WHAT IS SKIPPED FROM DREAMERV3
- recurrent state-space memory
- categorical stochastic latents
- KL balancing and free bits
- symlog and two-hot value targets
- a separately trained critic

Why:
- those parts matter, but adding them all at once would hide the basic
  imagination loop
- the first version should be small enough to debug one tensor at a time

WHY THIS IS NOT RANDOM SYNTHETIC DATA
The world-model dataset comes from `common.embodied_gridworld`.

Every training row is created by stepping the real project environment:

```text
full_state, action -> next_full_state, reward, cost, done
```

Do not create random latent vectors.
The policy may imagine futures only after the world model has trained on real
logged transitions.

WHAT IS PROVIDED
`common.embodied_gridworld` provides:
- scenes with walls, hazards, a goal, and an agent
- real step rules
- logged transitions
- encoded full grid states
- rendered images

WHAT THE MODEL KNOWS
The world model receives:
- flattened full grid state shaped `[4 * grid_size * grid_size]`
- action vector shaped `[4]`

The policy receives:
- flattened full grid state shaped `[4 * grid_size * grid_size]`

WHAT THE WORLD MODEL PREDICTS
- next flattened full grid state shaped `[4 * grid_size * grid_size]`
- reward shaped `[1]`
- cost shaped `[1]`
- done logit shaped `[1]`

WHAT CHANGES DURING TRAINING
- world-model weights change during world-model training
- policy weights change during imagined policy training
- the real environment rules do not change

FULL WORKFLOW

```text
1. collect real logged transitions
2. train a world model on those transitions
3. freeze the world model for policy training
4. start imagined rollouts from real training latents
5. train the policy to maximize imagined reward minus cost
6. evaluate the policy in real held-out grid-world scenes
7. save metrics and visuals
```

DATASET
Default dataset:
- no download
- use `common.embodied_gridworld`
- call `collect_visual_gridworld_episodes(...)`
- call `flatten_logged_episodes(...)`

One logged row contains:

```text
full_state: [4, grid_size, grid_size]
action: int
next_full_state: [4, grid_size, grid_size]
reward: float
cost: float
done: bool
```

LOSS FUNCTIONS
World-model loss:

```text
next_agent_loss = cross_entropy(pred_next_agent_logits, true_next_agent_cell)
reward_loss = mse(pred_reward, true_reward)
cost_loss = binary_cross_entropy_with_logits(cost_logit, cost_event_target)
done_loss = binary_cross_entropy_with_logits(done_logit, done_target)

world_model_loss = next_agent_loss + reward_loss + cost_loss + done_loss
```

Policy imagination score:

```text
score = sum(imagined_reward - cost_weight * predicted_cost_probability)
policy_loss = -mean(score)
```

Why the policy uses soft actions during training:
- `argmax` chooses one hard action, but it does not pass a useful gradient back
  into the policy.
- During imagined policy training, use `torch.softmax(action_logits, dim=-1)`.
- During real evaluation, use `argmax` to choose one real action.

METRICS TO TRACK
- validation next-agent loss: lower is better
- validation next-agent accuracy: higher is better
- validation reward MSE: lower is better
- validation cost MSE: lower is better
- validation done accuracy: higher is better
- imagined policy return: higher is better
- real policy success rate: higher is better
- real mean cost: lower is better

VISUALS TO CREATE
Save these files:
- `outputs/tiny_dreamer/training_curves.png`
- `outputs/tiny_dreamer/imagined_return_curve.png`
- `outputs/tiny_dreamer/real_policy_rollout.png`

COMMON MISTAKES
- training the world model on random latents
- using imagined rollouts before training the world model
- training the policy with `argmax` and expecting gradients to flow
- reporting imagined reward but never testing the policy in the real environment
- forgetting cost, which teaches unsafe plans
- letting a negative predicted cost become a fake reward during imagination
- not saving visuals
- saying the first version is a full DreamerV3 implementation

STEP-BY-STEP TODO
1. Collect complete logged grid-world episodes.
2. Split episodes, then build train and validation tensors.
3. Train the world model.
4. Evaluate world-model validation metrics.
5. Freeze the world model.
6. Train the policy through imagined rollouts with soft actions.
7. Evaluate the policy in real held-out scenes.
8. Save training curves, imagined return curve, and a real rollout plot.
9. Explain which steps used real data and which used imagined data.

DEFINITION OF DONE
You are done when:
- world-model data comes from `common.embodied_gridworld`
- no episode appears in both training and validation
- policy training uses training latents, never validation latents
- the world model trains after you complete the stubs
- validation world-model metrics are reported
- imagined policy training runs after world-model training
- real policy evaluation metrics are reported
- all three visual files are saved
- every completed stub keeps its original `raise NotImplementedError(...)` line commented out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
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
    LoggedGridStep,
    VisualGridWorldConfig,
    collect_visual_gridworld_episodes,
    encode_full_grid_state,
    flatten_logged_episodes,
    make_grid_scene,
    render_grid_image,
    step_grid_world,
)
from common.reproducibility import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"


# Learning order for this exercise:
# 1. Build the real transition dataset.
# 2. Build and train the world model.
# 3. Train the policy through imagined rollouts.
# 4. Evaluate the policy in the real grid-world.
# 5. Save the three visuals.


Position = tuple[int, int]
Scene = tuple[Position, Position, set[Position], set[Position]]


class DreamerDatasetSplits(TypedDict):
    """Train and validation tensors for the Dreamer-style world model."""

    train_latents: torch.Tensor
    train_actions: torch.Tensor
    train_next_agent_cells: torch.Tensor
    train_rewards: torch.Tensor
    train_costs: torch.Tensor
    train_dones: torch.Tensor
    val_latents: torch.Tensor
    val_actions: torch.Tensor
    val_next_agent_cells: torch.Tensor
    val_rewards: torch.Tensor
    val_costs: torch.Tensor
    val_dones: torch.Tensor


class RealPolicyRollout(TypedDict):
    """One real rollout from the learned policy."""

    states: list[Position]
    actions: list[int]
    total_return: float
    total_cost: float
    reached_goal: bool
    steps: int


class TinyDreamerMetrics(TypedDict):
    """Final metrics and visual paths."""

    final_world_model_loss: float
    val_next_agent_loss: float
    val_next_agent_accuracy: float
    val_reward_mse: float
    val_cost_loss: float
    val_cost_accuracy: float
    val_done_accuracy: float
    final_imagined_return: float
    real_success_rate: float
    real_mean_return: float
    real_mean_cost: float
    training_curve_path: str
    imagined_return_curve_path: str
    real_rollout_plot_path: str


@dataclass
class TinyDreamerConfig:
    """Settings for the tiny Dreamer-style exercise."""

    state_channels: int = 4
    num_actions: int = 4
    hidden_dim: int = 128
    grid_size: int = 7
    image_size: int = 28
    num_episodes: int = 3000 #1000
    max_steps: int = 18
    batch_size: int = 64
    world_model_epochs: int = 80 #20
    policy_epochs: int = 20
    imagination_horizon: int = 8 #4
    learning_rate: float = 5e-4 #1e-3
    val_fraction: float = 0.2
    cost_weight: float = 3.0 #1.0
    value_num_bins: int = 255
    value_min: float = -20.0
    value_max: float = 20.0
    num_eval_scenes: int = 50 #24
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "tiny_dreamer"

    @property
    def latent_dim(self) -> int:
        """Return the flattened full-state width used by the small model."""
        return self.state_channels * self.grid_size * self.grid_size

    @property
    def full_state_shape(self) -> tuple[int, int, int]:
        """Return the unflattened full-state shape."""
        return (self.state_channels, self.grid_size, self.grid_size)


def symlog(values: torch.Tensor) -> torch.Tensor:
    """Compress signed values while preserving zero and sign."""
    # Where this fits:
    # DreamerV3 uses symlog-style targets so one model can handle reward and
    # value scales that differ greatly across tasks.
    #
    # How this fits into Dreamer:
    # A Dreamer-style agent does not only predict the next state.
    # It also predicts signals such as reward and value.
    #
    # In many environments, those numbers can have very different sizes.
    # One game may give rewards near `1`.
    # Another task may give rewards near `1000` or `-1000`.
    # If the model trains directly on those raw numbers, large targets can
    # dominate the loss and make training unstable.
    #
    # `symlog` is used before training a reward head or value head.
    # It changes a large target into a smaller target:
    #
    # ```text
    # real reward or value -> symlog -> smaller training target
    # ```
    #
    # Example:
    # - raw target `1000` becomes about `6.91`
    # - raw target `10` becomes about `2.40`
    # - raw target `0` stays `0`
    # - raw target `-1000` becomes about `-6.91`
    #
    # The sign still matters, but the size is easier for the neural network
    # to predict.
    #
    # In this exercise, the first simple version trains mostly with raw reward
    # targets. This helper is here for the DreamerV3-style upgrade where the
    # reward or value target is compressed before computing the loss.
    #
    # Formula:
    # `symlog(x) = sign(x) * log(1 + abs(x))`
    #
    # Example:
    # - `symlog(0) = 0`
    # - `symlog(9)` is about `2.30`
    # - `symlog(-9)` is about `-2.30`
    #
    # Method:
    # Use `torch.sign`, `torch.log1p`, and `torch.abs`.
    #
    # Why the formula is split into three parts:
    # - `torch.sign(values)` keeps only the direction:
    #   negative values become `-1`, zero becomes `0`, and positive values
    #   become `1`.
    # - `torch.abs(values)` removes the sign so the log only sees a positive
    #   size, such as `9` instead of `-9`.
    # - `torch.log1p(x)` means `log(1 + x)`. The `+ 1` matters because
    #   `log(0)` is not valid, while `log(1 + 0)` is `0`.
    #
    # Tiny example:
    # - for `values = -9`, `sign = -1`, `abs = 9`, and `log1p(9) = log(10)`
    # - the result is `-1 * log(10)`, which is about `-2.30`
    #
    # So large values are squeezed closer to zero, but their direction is kept.
    
    return torch.sign(values) * torch.log1p(torch.abs(values))

    #raise NotImplementedError("Apply the DreamerV3 symlog transform.")


def symexp(values: torch.Tensor) -> torch.Tensor:
    """Undo `symlog` and return values to the original scale."""
    # Where this fits:
    # If a Dreamer-style model predicts a symlog reward or symlog value, the
    # prediction is still on the compressed scale.
    #
    # Use `symexp` when you want to read that prediction back in the normal
    # reward or value units:
    #
    # ```text
    # model prediction on symlog scale -> symexp -> normal reward or value
    # ```
    #
    # Concrete example:
    # If the reward head predicts about `2.40`, that means the raw reward is
    # about `10`, because `symexp(2.40)` is about `10`.
    #
    # This matters during logging, evaluation, and planning.
    # The model may train on the compressed number, but the agent still cares
    # about the real reward scale when judging imagined futures.
    #
    # Formula:
    # `symexp(x) = sign(x) * (exp(abs(x)) - 1)`
    #
    # Method:
    # Use `torch.sign`, `torch.expm1`, and `torch.abs`.
    #
    # Check:
    # `symexp(symlog(x))` should be close to `x`.
    #
    # This does the opposite of `symlog`.
    # `symlog` used `log(1 + abs(x))` to shrink the value.
    # `symexp` uses `exp(abs(x)) - 1` to grow it back.
    #
    # Why `torch.expm1` is used:
    # - `torch.expm1(x)` means `exp(x) - 1`.
    # - It is usually more accurate than writing `torch.exp(x) - 1`,
    #   especially when `x` is close to zero.
    #
    # Tiny example:
    # - `symlog(-9)` is about `-2.30`
    # - `symexp(-2.30)` takes the sign `-1`, computes `exp(2.30) - 1`,
    #   and gets back about `-9`
    
    return torch.sign(values) * torch.expm1(torch.abs(values))

    #raise NotImplementedError("Undo the DreamerV3 symlog transform.")


def two_hot_targets(
    values: torch.Tensor,
    bin_centers: torch.Tensor,
) -> torch.Tensor:
    """Put target mass on the two value bins nearest each target."""
    # Where this fits:
    # DreamerV3 predicts a distribution over value bins instead of one raw
    # number. Two-hot targets preserve values between bin centers.
    #
    # How this fits into Dreamer:
    # A critic is the model part that predicts "how good this state is".
    # In RL, this "how good" number is called a value.
    #
    # A simple critic can output one number, such as `1.25`.
    # DreamerV3 instead makes the critic output a probability distribution
    # across fixed value bins.
    #
    # That means the model does not say:
    #
    # ```text
    # value = 1.25
    # ```
    #
    # It says something closer to:
    #
    # ```text
    # value is between bin 1.0 and bin 2.0
    # put 75% weight on 1.0
    # put 25% weight on 2.0
    # ```
    #
    # This is called "two-hot" because each target row usually has two
    # non-zero entries. It is like one-hot labels for classification, except
    # the answer can sit between two nearby bins.
    #
    # Every row should sum to `1`.
    # You can read one row as "100% of the target value is spread across these
    # bins." Usually that 100% is split between two nearby bins. At the edge of
    # the bin range, all 100% can go into one bin.
    #
    # Why this helps:
    # - value prediction becomes more like classification over bins
    # - nearby wrong answers are not treated as equally bad as far-away answers
    # - the model can still represent values between bin centers
    #
    # In DreamerV3, the usual flow is:
    #
    # ```text
    # raw value target
    # -> symlog(raw value target)
    # -> two_hot_targets(...)
    # -> train critic logits with cross-entropy
    # ```
    #
    # The critic outputs logits shaped `[batch, num_bins]`.
    # The target from this function is also shaped `[batch, num_bins]`.
    # Cross-entropy then teaches the critic to put probability on the right
    # value bins.
    #
    # Tiny example:
    # - bin centers are `[0, 1, 2]`
    # - target value is `1.25`
    # - target distribution is `[0.0, 0.75, 0.25]`
    #
    # Why those weights are `0.75` and `0.25`:
    # - `1.25` is between `1` and `2`
    # - it is 25% of the way from `1` to `2`
    # - so the upper bin `2` gets `0.25`
    # - the lower bin `1` gets the leftover `0.75`
    #
    # More examples with the same bins:
    #
    # ```text
    # value 0.00 -> [1.00, 0.00, 0.00]
    # value 0.50 -> [0.50, 0.50, 0.00]
    # value 1.00 -> [0.00, 1.00, 0.00]
    # value 1.25 -> [0.00, 0.75, 0.25]
    # value 2.00 -> [0.00, 0.00, 1.00]
    # ```
    #
    # Input:
    # - `values`: `[batch]`
    #   Example: `torch.tensor([0.5, 1.25])`
    # - `bin_centers`: `[num_bins]`, sorted from low to high
    #   Example: `torch.tensor([0.0, 1.0, 2.0])`
    #
    # Output:
    # - `[batch, num_bins]`
    #   Example:
    #   `[[0.5, 0.5, 0.0], [0.0, 0.75, 0.25]]`
    #
    # Steps:
    # 1. Make `clipped_values` by keeping every target value inside the range
    #    covered by `bin_centers`.
    #    Use `values.clamp(min=bin_centers[0], max=bin_centers[-1])`.
    #
    #    Example:
    #    - `values = [-5.0, 0.5, 1.25, 10.0]`
    #    - `bin_centers = [0.0, 1.0, 2.0]`
    #    - `clipped_values = [0.0, 0.5, 1.25, 2.0]`
    #
    # 2. Count the number of bins.
    #    Use `num_bins = len(bin_centers)`.
    #    You need this number to make the output tensor and to keep indices in
    #    range.
    #
    # 3. Find the upper bin index for each clipped value.
    #    Use:
    #    `upper_indices = torch.searchsorted(bin_centers, clipped_values, right=False)`.
    #
    #    This asks: "what is the first bin center at or above this value?"
    #
    #    Example with `bin_centers = [0.0, 1.0, 2.0]`:
    #    - clipped value `0.5` gives upper index `1`
    #    - clipped value `1.25` gives upper index `2`
    #
    # 4. Clamp the upper indices between `1` and `num_bins - 1`.
    #    Use `upper_indices = upper_indices.clamp(min=1, max=num_bins - 1)`.
    #
    #    This is needed because each value must have both:
    #    - a lower bin, using `upper_indices - 1`
    #    - an upper bin, using `upper_indices`
    #
    # 5. Make the lower bin indices.
    #    Use `lower_indices = upper_indices - 1`.
    #
    #    Example:
    #    - if `upper_indices = [1, 2]`
    #    - then `lower_indices = [0, 1]`
    #
    # 6. Look up the real bin-center values for the lower and upper bins.
    #    Use:
    #    - `lower_centers = bin_centers[lower_indices]`
    #    - `upper_centers = bin_centers[upper_indices]`
    #
    # 7. Compute the two weights.
    #    Use:
    #    `upper_weight = (clipped_values - lower_centers) / (upper_centers - lower_centers)`.
    #    Then use:
    #    `lower_weight = 1.0 - upper_weight`.
    #
    #    Example:
    #    - lower center is `1.0`
    #    - upper center is `2.0`
    #    - clipped value is `1.25`
    #    - upper weight is `0.25`
    #    - lower weight is `0.75`
    #
    # 8. Make an all-zero target tensor shaped `[batch, num_bins]`.
    #    Use the same device and dtype as `values`.
    #
    # 9. Write the weights into the target tensor.
    #    Write `lower_weight` at `lower_indices`.
    #    Write `upper_weight` at `upper_indices`.
    #    Use `scatter_` or `scatter_add_`.
    #
    #    The index tensors and weight tensors should be shaped `[batch, 1]`
    #    when you pass them to `scatter_` or `scatter_add_`.
    #
    # Check:
    # - Every output row should sum to `1`.
    #   This means each target row is a complete probability distribution over
    #   the value bins.
    # - A value at the first or last center should put all mass on that center.
    
    num_bins = len(bin_centers)  # scalar Python int, for example `255`
    batch_size = values.shape[0]  # scalar Python int; `values` is shaped `[batch]`

    # keep every target value inside the range covered by bin_centers
    clipped_values = values.clamp(min=bin_centers[0], max=bin_centers[-1])                                  # shape `[batch]`

    # Find the upper bin index for each clipped value
    # what is the first bin center at or above this value?
    # right=False means we want the first bin center at or above the value
    upper_bin_indices = torch.searchsorted(sorted_sequence=bin_centers, input=clipped_values, right=False)  # shape `[batch]`
    # clamp the upper bin indices between `1` and `num_bins - 1`
    upper_bin_indices = upper_bin_indices.clamp(min=torch.tensor(1), max=torch.tensor(num_bins-1))          # shape `[batch]`

    lower_bin_indices = upper_bin_indices - 1       # shape `[batch]`

    lower_centers = bin_centers[lower_bin_indices]  # shape `[batch]`
    upper_centers = bin_centers[upper_bin_indices]  # shape `[batch]`

    upper_weights = (clipped_values - lower_centers) / (upper_centers - lower_centers)  # shape `[batch]`
    lower_weights = 1.0 - upper_weights             # shape `[batch]`

    targets = torch.zeros(size=(batch_size, num_bins), device=values.device, dtype=values.dtype)            # shape `[batch, num_bins]`

    # Shape note:
    # `targets` is 2D, so each write must choose a row and a column.
    # `lower_bin_indices` and `upper_bin_indices` only choose columns.
    # Before using `scatter_`, reshape index and weight tensors from `[batch]`
    # to `[batch, 1]` with `.unsqueeze(1)`.
    row_indices = torch.arange(batch_size, device=values.device)

    targets[row_indices, lower_bin_indices] = lower_weights
    targets[row_indices, upper_bin_indices] = upper_weights

    return targets  # shape `[batch, num_bins]`

    #raise NotImplementedError("Build DreamerV3 two-hot value targets.")


def make_visual_grid_config(config: TinyDreamerConfig) -> VisualGridWorldConfig:
    """Create the shared grid-world config used by this exercise."""
    return VisualGridWorldConfig(
        grid_size=config.grid_size,
        image_size=config.image_size,
        num_episodes=config.num_episodes,
        max_steps=config.max_steps,
        random_state=config.random_state,
    )


def collect_logged_episodes(config: TinyDreamerConfig) -> list[list[LoggedGridStep]]:
    """Collect complete real episodes from the shared grid-world."""
    # Where this fits:
    # This is the real dataset source for the world model.
    #
    # Method:
    # 1. Build `grid_config = make_visual_grid_config(config)`.
    # 2. Collect `episodes = collect_visual_gridworld_episodes(grid_config)`.
    # 3. Check the episode list and every kept episode are not empty.
    # 4. Return the complete episodes without flattening.
    #
    # Checks:
    # - Do not create random latent vectors.
    # - Use `step["full_state"]` and `step["next_full_state"]` as real states.
    
    grid_config = make_visual_grid_config(config)

    episodes = collect_visual_gridworld_episodes(grid_config) # returns list[list[LoggedGridStep]]

    return episodes

    #raise NotImplementedError("Collect complete real grid-world episodes.")


def one_hot_action(action: int, config: TinyDreamerConfig) -> torch.Tensor:
    """Encode one action id as a one-hot action vector."""
    # Goal:
    # Convert action id `0`, `1`, `2`, or `3` into a vector shaped `[4]`.
    #
    # Method:
    # 1. Check `0 <= action < config.num_actions`.
    # 2. Build `encoded = torch.zeros(config.num_actions, dtype=torch.float32)`.
    # 3. Set `encoded[action] = 1.0`.
    # 4. Return `encoded`.

    if not 0 <= action < config.num_actions:
        raise ValueError("Inavlid action id.")

    encoded = torch.zeros(config.num_actions, dtype=torch.float32, device=device)
    encoded[action] = 1.0
    return encoded
    
    #raise NotImplementedError("One-hot encode a grid-world action.")


def build_dreamer_dataset(config: TinyDreamerConfig) -> DreamerDatasetSplits:
    """Build train and validation tensors from real logged transitions."""
    # Where this fits:
    # This function builds the real dataset used to train the world model.
    # The world model learns from real environment steps before the policy is
    # trained with imagined rollouts.
    #
    # One logged step looks like this:
    #
    # ```text
    # current full state + action -> next full state, reward, cost, done
    # ```
    #
    # In the code, one step has fields like:
    # - `step["full_state"]`: current full state, shaped
    #   `[state_channels, grid_size, grid_size]`
    #   The four channels are:
    #   channel 0 = agent position
    #   channel 1 = goal position
    #   channel 2 = all hazard positions
    #   channel 3 = all wall positions
    # - `step["action"]`: action id, one integer such as `0`, `1`, `2`, or `3`
    # - `step["next_full_state"]`: next full state, shaped
    #   `[state_channels, grid_size, grid_size]`
    # - `step["reward"]`: reward number for that step
    # - `step["cost"]`: safety cost number for that step
    # - `step["done"]`: whether the episode ended after that step
    #
    # Goal:
    # Turn many logged steps into tensors that are easy to feed into PyTorch.
    # The train tensors teach the world model.
    # The validation tensors check whether the world model learned useful
    # predictions on episodes it did not train on.
    #
    # Output dictionary:
    # Here `latent` means "flattened full state".
    # With the default config, one flattened full state has width
    # `4 * 7 * 7 = 196`.
    #
    # - `train_latents`: `[num_train_steps, latent_dim]`
    # - `train_actions`: `[num_train_steps, num_actions]`
    # - `train_next_agent_cells`: `[num_train_steps]`
    # - `train_rewards`: `[num_train_steps, 1]`
    # - `train_costs`: `[num_train_steps, 1]`, with `1.0` if any cost happened
    # - `train_dones`: `[num_train_steps, 1]`
    # - `val_latents`: `[num_val_steps, latent_dim]`
    # - `val_actions`: `[num_val_steps, num_actions]`
    # - `val_next_agent_cells`: `[num_val_steps]`
    # - `val_rewards`: `[num_val_steps, 1]`
    # - `val_costs`: `[num_val_steps, 1]`, with `1.0` if any cost happened
    # - `val_dones`: `[num_val_steps, 1]`
    #
    # Steps:
    # 1. Collect complete episodes with `collect_logged_episodes(config)`.
    #    Keep them grouped by episode for now.
    #
    # 2. Shuffle and split whole episodes into train and validation sets.
    #    Use `random.Random(config.random_state)` so the split is repeatable.
    #    Use `config.val_fraction` as the validation percentage.
    #    For example, `0.2` means 20% validation episodes and 80% train
    #    episodes.
    #    Do not split single steps, because one episode must not appear in both
    #    train and validation.
    #
    # 3. Flatten each split separately with `flatten_logged_episodes(...)`.
    #    After this:
    #    - `train_steps` is `list[LoggedGridStep]`
    #    - `val_steps` is `list[LoggedGridStep]`
    #
    # 4. Convert one list of steps into six tensors:
    #    - flattened full states -> latents shaped `[num_steps, latent_dim]`
    #    - one-hot actions -> actions shaped `[num_steps, num_actions]`
    #    - next agent cell ids -> `[num_steps]`
    #    - rewards -> `[num_steps, 1]`
    #    - costs -> `[num_steps, 1]`, using `1.0` if `step["cost"] > 0`
    #    - dones -> `[num_steps, 1]`, using `0.0` for `False` and `1.0` for `True`
    #
    #    Use `step["full_state"].flatten()` for the current latent.
    #    Use `step["next_full_state"][0].flatten().argmax()` for the next
    #    agent cell id.
    #
    # 5. Build those six tensors for train steps and again for validation
    #    steps, then return them with the exact keys listed above.
    #
    # Checks:
    # - Inputs and targets must stay aligned. Row `i` of each tensor must come
    #   from the same logged step.
    # - No episode may appear in both train and validation.
    # - Validation rows must not train the world model or the policy.
    # - `train_latents.shape[0]` should match `train_actions.shape[0]`,
    #   `train_next_agent_cells.shape[0]`, `train_rewards.shape[0]`,
    #   `train_costs.shape[0]`, and `train_dones.shape[0]`.

    episodes = collect_logged_episodes(config)  # # returns list[list[LoggedGridStep]]
    num_episodes = len(episodes)

    shuffled_indices = torch.randperm(num_episodes, generator = torch.Generator().manual_seed(config.random_state))

    num_val = int(num_episodes * config.val_fraction)

    val_eps = [episodes[shuffled_indices[index]] for index in range(0, num_val)]
    train_eps = [episodes[shuffled_indices[index]] for index in range(num_val, len(episodes))]

    val_flat_eps = flatten_logged_episodes(val_eps) #[step for episode in val_eps for step in episode]
    train_flat_eps = flatten_logged_episodes(train_eps) #[step for episode in train_eps for step in episode]

    def build_tensor(steps: list[LoggedGridStep]) -> list[torch.tensor]:
        
        states = []
        actions = []
        next_agent_cells = []
        rewards = []
        costs = []
        dones = []

        for step in steps:
            states.append(step["full_state"].flatten().to(device))
            actions.append(one_hot_action(step["action"], config))
            next_agent_cells.append(step["next_full_state"][0].flatten().argmax().to(device))
            rewards.append(torch.tensor(step["reward"], device=device, dtype=torch.float32).unsqueeze(dim=0))
            costs.append(torch.tensor(float(step["cost"] > 0.0), device=device, dtype=torch.float32).unsqueeze(dim=0))
            dones.append(torch.tensor(step["done"], device=device, dtype=torch.float32).unsqueeze(dim=0))
            
        return [torch.stack(states, dim=0), torch.stack(actions, dim=0), torch.stack(next_agent_cells, dim=0),
                torch.stack(rewards, dim=0), torch.stack(costs, dim=0), torch.stack(dones, dim=0)]

    train_tensors = build_tensor(train_flat_eps)
    val_tensors = build_tensor(val_flat_eps)

    return DreamerDatasetSplits(train_latents=train_tensors[0], train_actions=train_tensors[1], train_next_agent_cells=train_tensors[2], 
                                train_rewards=train_tensors[3], train_costs=train_tensors[4], train_dones=train_tensors[5],
                                val_latents=val_tensors[0], val_actions=val_tensors[1], val_next_agent_cells=val_tensors[2], 
                                val_rewards=val_tensors[3], val_costs=val_tensors[4], val_dones=val_tensors[5])

    #raise NotImplementedError("Build Dreamer-style train and validation tensors.")


class TinyDreamerWorldModel(nn.Module):
    """Predict next agent cell, reward, cost, and done."""

    def __init__(self, config: TinyDreamerConfig) -> None:
        # Goal:
        # Build the learned world model used for imagined rollouts.
        #
        # Method:
        # 1. Call `super().__init__()`.
        # 2. Set `input_dim = config.latent_dim + config.num_actions`.
        #    Here `config.latent_dim` is the flattened full-state width:
        #    `state_channels * grid_size * grid_size`.
        # 3. Build `self.trunk` with two Linear + ReLU blocks.
        # 4. Build heads:
        #    - `next_agent_head`: output `grid_size * grid_size`
        #    - `reward_head`: output `1`
        #    - `cost_head`: output `1` raw logit for "any cost happened"
        #    - `done_head`: output `1`
        #
        # Check:
        # - Do not apply sigmoid in the done head.
        # - Do not apply sigmoid in the cost head.
        super().__init__()

        self.state_dim = config.state_channels * config.grid_size * config.grid_size
        self.input_dim = self.state_dim + config.num_actions

        self.trunk = nn.Sequential(nn.Linear(in_features=self.input_dim, out_features=config.hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(in_features=config.hidden_dim, out_features=config.hidden_dim),
                                    nn.ReLU())
        
        self.next_agent_head = nn.Linear(in_features=config.hidden_dim, out_features=config.grid_size * config.grid_size)
        self.reward_head = nn.Linear(in_features=config.hidden_dim, out_features=1)
        self.cost_head = nn.Linear(in_features=config.hidden_dim, out_features=1)
        self.done_head = nn.Linear(in_features=config.hidden_dim, out_features=1)

        #raise NotImplementedError("Build the tiny Dreamer world model.")

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return next-agent logits, reward, cost logit, and done logit."""
        # Input:
        # - `latents`: flattened full states shaped `[batch_size, latent_dim]`.
        # - `actions`: `[batch_size, num_actions]`.
        #
        # Output:
        # - next-agent logits: `[batch_size, grid_size * grid_size]`
        # - rewards: `[batch_size, 1]`
        # - cost logits: `[batch_size, 1]`
        # - done logits: `[batch_size, 1]`
        #
        # Method:
        # Concatenate latents and actions with `torch.cat(..., dim=-1)`,
        # run the trunk, then run each head.

        inputs = torch.cat([latents, actions], dim=-1)

        features = self.trunk(inputs)

        next_agent_logits = self.next_agent_head(features)
        reward = self.reward_head(features)
        cost_logit = self.cost_head(features)
        done = self.done_head(features)

        return next_agent_logits, reward, cost_logit, done

        #raise NotImplementedError("Run the tiny Dreamer world model.")


class LatentPolicy(nn.Module):
    """Choose actions from flattened full states."""

    def __init__(self, config: TinyDreamerConfig) -> None:
        # Goal:
        # Build the policy trained from imagined rollouts.
        #
        # Method:
        # 1. Call `super().__init__()`.
        # 2. Build `self.net = nn.Sequential(...)`:
        #    - Linear from `config.latent_dim` to `config.hidden_dim`
        #      where `config.latent_dim` is the flattened full-state width
        #    - ReLU
        #    - Linear from `config.hidden_dim` to `config.num_actions`
        #
        # Check:
        # - Return logits, not softmax probabilities.
        super().__init__()

        self.state_dim = config.state_channels * config.grid_size * config.grid_size        

        self.net = nn.Sequential(nn.Linear(in_features=self.state_dim, out_features=config.hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=config.hidden_dim, out_features=config.num_actions))

        #raise NotImplementedError("Build the latent policy.")

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        # Input:
        # - `latents`: flattened full states shaped `[batch_size, latent_dim]`.
        #
        # Output:
        # - action logits: `[batch_size, num_actions]`.

        #full_state=latents, this is the same

        logits = self.net(latents)
        return logits
        #raise NotImplementedError("Run the latent policy.")


def compute_world_model_loss(
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return world-model loss and parts."""
    # Method:
    # 1. Compute next-agent-cell cross-entropy.
    # 2. Compute reward MSE.
    # 3. Compute cost-event loss with binary cross-entropy with logits.
    # 4. Compute done loss with `nn.functional.binary_cross_entropy_with_logits(...)`.
    # 5. Add all four losses.
    # 6. Return `(total_loss, loss_parts)`.

    next_agent_loss = nn.functional.cross_entropy(predictions[0], targets[0].long())
    reward_mse = nn.functional.mse_loss(predictions[1], targets[1])
    cost_loss = nn.functional.binary_cross_entropy_with_logits(predictions[2], targets[2])
    done_loss = nn.functional.binary_cross_entropy_with_logits(predictions[3], targets[3])

    total_loss = next_agent_loss + reward_mse + cost_loss + done_loss
    loss_parts = {
        "next_agent_loss": float(next_agent_loss.detach().cpu().item()),
        "reward_mse": float(reward_mse.detach().cpu().item()),
        "cost_loss": float(cost_loss.detach().cpu().item()),
        "done_loss": float(done_loss.detach().cpu().item()),
    }

    return total_loss, loss_parts

    #raise NotImplementedError("Compute tiny Dreamer world-model loss.")


def train_world_model_one_epoch(model: TinyDreamerWorldModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train the world model for one epoch."""
    # Method:
    # 1. Call `model.train()`.
    # 2. Loop over batches.
    # 3. Move every tensor to `device`.
    # 4. Run the model.
    # 5. Compute loss.
    # 6. Clear gradients, backpropagate, and step the optimizer.
    # 7. Return mean loss weighted by batch size.

    model.train()

    total_loss = 0
    num_samples = 0

    for states, actions, next_agent_cells, rewards, costs, dones in loader:
        states=states.to(device)
        actions = actions.to(device)
        next_agent_cells = next_agent_cells.to(device)
        rewards = rewards.to(device)
        costs = costs.to(device)
        dones = dones.to(device)

        pred_next_agent_logits, pred_rewards, pred_cost_logits, pred_dones = model(states, actions)
        loss, loss_parts = compute_world_model_loss(
            predictions=[pred_next_agent_logits, pred_rewards, pred_cost_logits, pred_dones],
            targets=[next_agent_cells, rewards, costs, dones],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = states.shape[0]
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    mean_loss = total_loss/num_samples

    return mean_loss
    #raise NotImplementedError("Train one tiny Dreamer world-model epoch.")


def evaluate_world_model(model: TinyDreamerWorldModel, splits: DreamerDatasetSplits) -> dict[str, float]:
    """Return validation world-model metrics."""
    # Method:
    # 1. Call `model.eval()`.
    # 2. Use `torch.no_grad()`.
    # 3. Run on validation tensors.
    # 4. Return:
    #    - `val_next_agent_loss`
    #    - `val_next_agent_accuracy`
    #    - `val_reward_mse`
    #    - `val_cost_loss`
    #    - `val_cost_accuracy`
    #    - `val_done_accuracy`

    model.eval()

    with torch.no_grad():
        val_states = splits["val_latents"].to(device)
        val_actions = splits["val_actions"].to(device)
        val_next_agent_cells = splits["val_next_agent_cells"].to(device)
        val_rewards = splits["val_rewards"].to(device)
        val_costs = splits["val_costs"].to(device)
        val_dones = splits["val_dones"].to(device)

        pred_next_agent_logits, pred_rewards, pred_cost_logits, pred_done_logits = model(val_states, val_actions)
        _loss, loss_parts = compute_world_model_loss(
            predictions=(pred_next_agent_logits, pred_rewards, pred_cost_logits, pred_done_logits),
            targets=(val_next_agent_cells, val_rewards, val_costs, val_dones),
        )

        pred_next_agent_cells = pred_next_agent_logits.argmax(dim=-1)
        next_agent_accuracy = (pred_next_agent_cells == val_next_agent_cells).float().mean().item()
        pred_costs = (torch.sigmoid(pred_cost_logits) >= 0.5).float()
        cost_accuracy = (pred_costs == val_costs).float().mean().item()
        pred_dones = (torch.sigmoid(pred_done_logits) >= 0.5).float()
        done_accuracy = (pred_dones == val_dones).float().mean().item()

    return {
        "val_next_agent_loss": loss_parts["next_agent_loss"],
        "val_next_agent_accuracy": float(next_agent_accuracy),
        "val_reward_mse": loss_parts["reward_mse"],
        "val_cost_loss": loss_parts["cost_loss"],
        "val_cost_accuracy": float(cost_accuracy),
        "val_done_accuracy": float(done_accuracy),
    }

    #raise NotImplementedError("Evaluate tiny Dreamer world-model metrics.")


def train_world_model(
    config: TinyDreamerConfig,
) -> tuple[TinyDreamerWorldModel, DreamerDatasetSplits, list[float], list[float], dict[str, float]]:
    """Train the world model from real logged transitions."""
    # Method:
    # 1. Call `set_seed(config.random_state)`.
    # 2. Build `splits = build_dreamer_dataset(config)`.
    # 3. Build a training `TensorDataset` and `DataLoader`.
    # 4. Build `TinyDreamerWorldModel(config).to(device)`.
    # 5. Build `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 6. Train for `config.world_model_epochs`.
    # 7. Track train loss and validation next-agent loss.
    # 8. Return model, splits, histories, and final validation metrics.
    
    set_seed(config.random_state)

    splits = build_dreamer_dataset(config)

    train_ds = TensorDataset(
        splits["train_latents"],
        splits["train_actions"],
        splits["train_next_agent_cells"],
        splits["train_rewards"],
        splits["train_costs"],
        splits["train_dones"],
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    model = TinyDreamerWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate)

    train_losses: list[float] = []
    val_next_agent_losses: list[float] = []
    final_metrics: dict[str, float] = {}

    for epoch in range(1, config.world_model_epochs + 1):
        train_loss = train_world_model_one_epoch(model, train_loader, optimizer)
        final_metrics = evaluate_world_model(model, splits)

        train_losses.append(float(train_loss))
        val_next_agent_losses.append(float(final_metrics["val_next_agent_loss"]))

        print(
            f"epoch {epoch:03d}/{config.world_model_epochs:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_next_agent_loss={final_metrics['val_next_agent_loss']:.4f} "
            f"val_next_agent_accuracy={final_metrics['val_next_agent_accuracy']:.4f} "
            f"val_reward_mse={final_metrics['val_reward_mse']:.4f} "
            f"val_cost_loss={final_metrics['val_cost_loss']:.4f} "
            f"val_cost_accuracy={final_metrics['val_cost_accuracy']:.4f} "
            f"val_done_accuracy={final_metrics['val_done_accuracy']:.4f}"
        )

    return model, splits, train_losses, val_next_agent_losses, final_metrics

    #raise NotImplementedError("Train the tiny Dreamer world model.")


def imagine_rollout(
    world_model: TinyDreamerWorldModel,
    policy_model: LatentPolicy,
    start_latents: torch.Tensor,
    config: TinyDreamerConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Imagine a differentiable rollout inside the world model."""
    # Where this fits:
    # This is how the policy learns without stepping the real environment.
    #
    # At this point, the world model should already be trained.
    # The policy may still be random or weak.
    # That is okay: this function is part of how the policy gets trained.
    #
    # The imagined rollout is not a fixed dataset saved once.
    # During policy training, the code calls this function again and again.
    # Each time:
    # - the current policy chooses actions
    # - the frozen world model predicts what would happen
    # - the policy loss scores those imagined rewards and costs
    # - backprop updates the policy
    #
    # So the loop looks like:
    #
    # ```text
    # weak policy -> imagined futures -> loss -> slightly better policy
    # better policy -> new imagined futures -> loss -> better policy
    # ```
    #
    # The world model is used like a learned practice environment.
    # The real grid-world is not stepped here.
    #
    # Important:
    # The world model should be frozen during policy training.
    # "Frozen" means its weights do not change.
    # Gradients can still pass through its operations to reach the policy.
    # That lets the policy learn which actions the world model predicts will
    # lead to higher reward and lower cost.
    #
    # Method:
    # 1. Start from `start_latents`.
    #    Shape: `[batch_size, latent_dim]`.
    #    In this exercise, each row is one flattened full grid state.
    #    With the default config, `latent_dim = 4 * 7 * 7 = 196`.
    # 2. For each imagined step from `0` to `config.imagination_horizon - 1`:
    #    - run `action_logits = policy(current_latents)`
    #    - build hard Gumbel-Softmax actions so the world model sees one-hot
    #      actions while gradients still reach the policy
    #    - run the world model with those actions
    #    - turn the predicted next-agent logits into a soft agent channel
    #    - copy goal, hazards, and walls from the current full state
    #    - flatten that rebuilt full state and use it as the next latent
    #    - store next latents, rewards, and costs
    # 3. Return lists of imagined latents, rewards, and costs.
    #
    # Check:
    # - Do not use `argmax` here, because policy training needs gradients.

    imagined_states = []
    imagined_rewards = []
    imagined_costs = []

    current_state = start_latents

    for i in range(config.imagination_horizon):
        action_logits = policy_model(current_state)
        #action_probabilities = torch.softmax(action_logits, dim=-1)

        action_probabilities = torch.nn.functional.gumbel_softmax(
            action_logits,
            tau=1.0,
            hard=True,
            dim=-1,
        )

        pred_next_agent_logits, pred_rewards, pred_cost_logits, pred_dones_logit = world_model(current_state, action_probabilities)

        batch_size = current_state.shape[0]

        current_full_state = current_state.view(batch_size, config.state_channels, config.grid_size, config.grid_size)
        # `pred_next_agent_logits` is shaped `[batch_size, grid_size * grid_size]`.
        # With the default grid, that is `[batch_size, 49]`.
        #
        # `torch.softmax(..., dim=-1)` turns the 49 raw cell scores into 49
        # probabilities. Each row sums to `1`, so one row means:
        # "how likely is each grid cell to be the next agent position?"
        #
        # `.view(batch_size, 1, grid_size, grid_size)` reshapes those 49 cell
        # probabilities back into one image-like channel.
        #
        # With the default grid:
        # `[batch_size, 49] -> [batch_size, 1, 7, 7]`
        #
        # The `1` means there is only one channel here: the predicted agent
        # position channel. Goal, hazard, and wall channels are copied below.
        next_agent_channel = torch.softmax(pred_next_agent_logits, dim=-1).view(
            batch_size,
            1,
            config.grid_size,
            config.grid_size,
        )
        next_full_state = torch.cat([next_agent_channel, current_full_state[:, 1:]], dim=1)
        pred_next_states = next_full_state.flatten(start_dim=1)
        pred_cost_probabilities = torch.sigmoid(pred_cost_logits)

        imagined_states.append(pred_next_states)
        imagined_rewards.append(pred_rewards)
        imagined_costs.append(pred_cost_probabilities)

        current_state = pred_next_states

    return imagined_states, imagined_rewards, imagined_costs

    #raise NotImplementedError("Imagine a differentiable rollout.")


def policy_loss_from_imagination(rewards: list[torch.Tensor], costs: list[torch.Tensor], config: TinyDreamerConfig) -> torch.Tensor:
    """Return policy loss from imagined rewards and costs."""
    # Formula:
    # `policy_loss = -mean(sum(reward - cost_weight * cost))`.
    #
    # Where this fits:
    # `imagine_rollout(...)` gives a list of predicted rewards and predicted
    # costs from the frozen world model.
    # This function turns those predictions into one loss for training the
    # policy.
    #
    # Reward and cost are separate signals.
    # They do not need to be exact opposites.
    #
    # Reward means "something we want more of".
    # In this grid-world, reaching the goal gives reward.
    #
    # Cost means "something we want less of".
    # In this grid-world, hazards and walls can create cost.
    #
    # The score for one imagined step is:
    #
    # ```text
    # step_score = reward - cost_weight * cost
    # ```
    #
    # Concrete examples with `cost_weight = 1.0`:
    #
    # ```text
    # reward 1.0, cost 0.0 -> score  1.0
    # reward 1.0, cost 0.5 -> score  0.5
    # reward 1.0, cost 2.0 -> score -1.0
    # reward 0.0, cost 1.0 -> score -1.0
    # ```
    #
    # `cost_weight` controls how much the policy cares about safety cost.
    # A larger value makes the policy avoid costly actions more strongly.
    #
    # The policy wants a high imagined score.
    # But PyTorch optimizers are built to make a loss smaller.
    # So we turn the score into a loss by multiplying it by `-1`.
    #
    # Example:
    # - policy A has imagined score `10`, so its loss is `-10`
    # - policy B has imagined score `3`, so its loss is `-3`
    # - PyTorch prefers `-10` because it is smaller than `-3`
    #
    # This means minimizing `-score` pushes the policy toward a higher score.
    #
    # Method:
    # 1. Check reward and cost lists are not empty and have the same length.
    # 2. Clamp cost at zero with `torch.clamp(cost, min=0.0)`, then compute
    #    `reward - config.cost_weight * safe_cost`.
    # 3. Sum across imagined steps.
    # 4. Take the mean over the batch.
    # 5. Negate it because PyTorch minimizes losses.

    if not rewards:
        raise ValueError("rewards must contain at least one imagined step.")
    if len(rewards) != len(costs):
        raise ValueError("rewards and costs must have the same number of imagined steps.")

    step_scores = []
    for reward, cost in zip(rewards, costs):
        safe_cost = torch.clamp(cost, min=0.0)
        step_scores.append(reward - config.cost_weight * safe_cost)

    rollout_scores = torch.stack(step_scores, dim=0).sum(dim=0)
    policy_loss = -rollout_scores.mean()

    return policy_loss
    #raise NotImplementedError("Compute policy loss from imagined rewards and costs.")


def train_policy_with_imagination(
    world_model: TinyDreamerWorldModel,
    splits: DreamerDatasetSplits,
    config: TinyDreamerConfig,
) -> tuple[LatentPolicy, list[float]]:
    """Train the policy through imagined rollouts."""
    # Method:
    # 1. Put the trained world model in eval mode with `world_model.eval()`.
    #    Then freeze it with `param.requires_grad = False`.
    #    The world model should guide the policy, not keep learning here.
    #
    # 2. Build the policy with `LatentPolicy(config).to(device)`.
    #    Build an optimizer for policy parameters only:
    #    `torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)`.
    #
    # 3. Use `splits["train_latents"]` as real starting states.
    #    Shape: `[num_train_steps, latent_dim]`.
    #    Move this tensor to `device`.
    #
    # 4. Build a DataLoader or loop over batches of start latents.
    #    Each batch should be shaped `[batch_size, latent_dim]`.
    #    You do not need actions, rewards, or next states here because the
    #    world model will imagine those from the start latents and policy.
    #
    # 5. For each policy epoch, loop over the start-latent batches:
    #    - call `imagine_rollout(world_model, policy, start_latents, config)`
    #    - get `imagined_rewards` and `imagined_costs`
    #    - compute `loss = policy_loss_from_imagination(...)`
    #    - call `optimizer.zero_grad()`
    #    - call `loss.backward()`
    #    - call `optimizer.step()`
    #
    # 6. Track the mean imagined return for the epoch.
    #    A simple value to store is `-loss.item()`, because the loss is the
    #    negative imagined score.
    #
    # 7. Return `(policy, imagined_returns)`.
    #
    # Checks:
    # - The real environment is not stepped during policy training.
    # - The world model should not be updated during policy training.

    world_model.eval()
    for param in world_model.parameters():
        param.requires_grad = False

    policy_model = LatentPolicy(config).to(device)
    optimizer = torch.optim.AdamW(params=policy_model.parameters(), lr=config.learning_rate)

    policy_model.train()

    start_states_data = splits["train_latents"].to(device)
    start_ds = TensorDataset(start_states_data)
    start_states_loader = DataLoader(start_ds, batch_size=config.batch_size, shuffle=True)
    imagined_returns: list[float] = []

    for epoch in range(1, config.policy_epochs + 1):
        total_imagined_return = 0.0
        num_samples = 0

        for (start_states,) in start_states_loader:
            start_states = start_states.to(device)

            imagined_states, imagined_rewards, imagined_costs = imagine_rollout(world_model, policy_model, start_states, config)

            loss = policy_loss_from_imagination(imagined_rewards, imagined_costs, config)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = start_states.shape[0]
            total_imagined_return += -float(loss.detach().cpu().item()) * batch_size
            num_samples += batch_size

        mean_imagined_return = total_imagined_return / num_samples
        imagined_returns.append(mean_imagined_return)

        print(
            f"policy epoch {epoch:03d}/{config.policy_epochs:03d} "
            f"imagined_return={mean_imagined_return:.4f}"
        )

    return policy_model, imagined_returns

    #raise NotImplementedError("Train policy through imagined rollouts.")


def choose_policy_action(policy_model: LatentPolicy, latent: torch.Tensor) -> int:
    """Choose one real action for evaluation."""
    # Method:
    # 1. Add a batch dimension to `latent`.
    # 2. Run the policy with `torch.no_grad()`.
    # 3. Use `argmax(dim=-1)` to choose the highest-logit action.
    # 4. Return it as a Python int.
    #
    # This is okay for real evaluation because we are not training here.

    with torch.no_grad():
        latent = latent.to(device)
        action_logits = policy_model(latent.unsqueeze(dim=0))   #add a fake batch dim
        highest_action = torch.argmax(action_logits, dim=-1)

    return highest_action.item()

    #raise NotImplementedError("Choose one policy action for real evaluation.")


def rollout_policy_in_real_scene(policy_model: LatentPolicy, scene: Scene, config: TinyDreamerConfig) -> RealPolicyRollout:
    """Run the learned policy in the real grid-world."""
    # Method:
    # 1. Unpack `scene` into agent, goal, hazards, and walls.
    # 2. Build `grid_config = make_visual_grid_config(config)`.
    # 3. Encode the current real state with `encode_full_grid_state(...)`,
    #    then flatten it before passing it to the policy.
    # 4. Choose an action with `choose_policy_action(...)`.
    # 5. Step the real environment with `step_grid_world(...)`.
    # 6. Track states, actions, total return, total cost, reached goal, and steps.
    # 7. Stop on done or after `config.max_steps`.
    #
    # Check:
    # - Real evaluation rewards and costs must come from `step_grid_world(...)`.
    
    agent, goal, hazards, walls = scene
    grid_config = make_visual_grid_config(config)

    states = [agent]
    actions = []
    total_return = 0.0
    total_cost = 0.0
    reached_goal = False

    for _ in range(config.max_steps):
        current_state = encode_full_grid_state(current_position=agent, goal_position=goal,
                                                hazard_positions=hazards, wall_positions=walls,
                                                config=grid_config,
                                                ).flatten()

        action = choose_policy_action(policy_model, current_state)
        next_agent, reward, cost, done = step_grid_world(agent, action, goal, hazards, walls, grid_config)

        actions.append(action)
        total_return += float(reward)
        total_cost += float(cost)
        states.append(next_agent)

        agent = next_agent
        reached_goal = agent == goal
        if done:
            break

    return {
        "states": states,
        "actions": actions,
        "total_return": total_return,
        "total_cost": total_cost,
        "reached_goal": reached_goal,
        "steps": len(actions),
    }

    #raise NotImplementedError("Roll out learned policy in the real grid-world.")


def build_eval_scenes(config: TinyDreamerConfig) -> list[Scene]:
    """Build held-out real scenes for policy evaluation."""
    # Method:
    # 1. Build `grid_config = make_visual_grid_config(config)`.
    # 2. Build `rng = random.Random(config.random_state + 20_000)`.
    # 3. Call `make_grid_scene(grid_config, rng)` until you have `config.num_eval_scenes`.
    # 4. Return the scenes.
    
    scenes = []

    grid_config = make_visual_grid_config(config)
    rng = random.Random(config.random_state + 20_000)

    for i in range(config.num_eval_scenes):        
        scene = make_grid_scene(grid_config, rng)
        scenes.append(scene)

    return scenes

    #raise NotImplementedError("Build held-out Dreamer evaluation scenes.")


def evaluate_policy_in_real_world(policy_model: LatentPolicy, config: TinyDreamerConfig) -> tuple[dict[str, float], list[RealPolicyRollout], list[Scene]]:
    """Evaluate the learned policy in held-out real scenes."""
    # Method:
    # 1. Build held-out scenes.
    # 2. Roll out the policy in every scene.
    # 3. Compute:
    #    - `real_success_rate`
    #    - `real_mean_return`
    #    - `real_mean_cost`
    # 4. Return metrics, rollouts, and scenes.

    scenes = build_eval_scenes(config)
    rollouts = [rollout_policy_in_real_scene(policy_model, scene, config) for scene in scenes]

    real_success_rate = sum(float(rollout["reached_goal"]) for rollout in rollouts) / len(rollouts)
    real_mean_return = sum(rollout["total_return"] for rollout in rollouts) / len(rollouts)
    real_mean_cost = sum(rollout["total_cost"] for rollout in rollouts) / len(rollouts)

    metrics = {
        "real_success_rate": real_success_rate,
        "real_mean_return": real_mean_return,
        "real_mean_cost": real_mean_cost,
    }

    return metrics, rollouts, scenes

    #raise NotImplementedError("Evaluate tiny Dreamer policy in the real world.")


def save_training_curves(world_losses: list[float], val_next_agent_losses: list[float], config: TinyDreamerConfig) -> str:
    """Save world-model training curves."""
    # Goal:
    # Save `outputs/tiny_dreamer/training_curves.png`.
    #
    # Method:
    # Use Matplotlib with `matplotlib.use("Agg")`.
    # Plot world-model train loss and validation next-agent loss.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "training_curves.png"

    epochs = list(range(1, len(world_losses) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, world_losses, label="train loss")
    ax.plot(epochs, val_next_agent_losses, label="val next-agent loss")
    ax.set_title("World Model Training")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return str(path)

    #raise NotImplementedError("Save tiny Dreamer training curves.")


def save_imagined_return_curve(imagined_returns: list[float], config: TinyDreamerConfig) -> str:
    """Save imagined policy return over training."""
    # Goal:
    # Save `outputs/tiny_dreamer/imagined_return_curve.png`.
    #
    # Method:
    # Plot policy epoch on the x-axis and imagined return on the y-axis.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "imagined_return_curve.png"

    epochs = list(range(1, len(imagined_returns) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, imagined_returns, label="imagined return")
    ax.set_title("Policy Training In Imagination")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean imagined return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return str(path)

    #raise NotImplementedError("Save imagined return curve.")


def save_real_rollout_plot(scene: Scene, rollout: RealPolicyRollout, config: TinyDreamerConfig) -> str:
    """Save one real policy rollout plot."""
    # Goal:
    # Save `outputs/tiny_dreamer/real_policy_rollout.png`.
    #
    # Method:
    # 1. Render the scene with `render_grid_image(...)`.
    # 2. Plot the image.
    # 3. Overlay the policy path.
    # 4. Save and close the plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "real_policy_rollout.png"

    agent, goal, hazards, walls = scene
    grid_config = make_visual_grid_config(config)
    image = render_grid_image(agent, goal, hazards, walls, grid_config)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())

    cell_size = config.image_size / config.grid_size
    rows = [position[0] for position in rollout["states"]]
    cols = [position[1] for position in rollout["states"]]
    xs = [(col + 0.5) * cell_size for col in cols]
    ys = [(row + 0.5) * cell_size for row in rows]

    ax.plot(xs, ys, color="purple", linewidth=2.5, marker="o", markersize=4, label="policy path")
    ax.set_title("Real Policy Rollout")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return str(path)

    #raise NotImplementedError("Save real policy rollout plot.")


def run_tiny_dreamer_demo(config: TinyDreamerConfig) -> TinyDreamerMetrics:
    """Run the full tiny Dreamer-style workflow."""
    # Method:
    # 1. Train the world model with `train_world_model(config)`.
    # 2. Train the policy with `train_policy_with_imagination(...)`.
    # 3. Evaluate the policy in real held-out scenes.
    # 4. Save all three visual files.
    # 5. Return one `TinyDreamerMetrics` dictionary.
    #
    # Checks:
    # - Return real evaluation metrics, not only imagined return.
    # - Do not train the policy before the world model exists.
    world_model, splits, world_losses, val_next_agent_losses, world_metrics = train_world_model(config)
    policy, imagined_returns = train_policy_with_imagination(world_model, splits, config)
    policy_metrics, rollouts, scenes = evaluate_policy_in_real_world(policy, config)

    training_curve_path = save_training_curves(world_losses, val_next_agent_losses, config)
    imagined_return_curve_path = save_imagined_return_curve(imagined_returns, config)
    real_rollout_plot_path = save_real_rollout_plot(scenes[1], rollouts[0], config)

    metrics: TinyDreamerMetrics = {
        "final_world_model_loss": float(world_losses[-1]),
        "val_next_agent_loss": float(world_metrics["val_next_agent_loss"]),
        "val_next_agent_accuracy": float(world_metrics["val_next_agent_accuracy"]),
        "val_reward_mse": float(world_metrics["val_reward_mse"]),
        "val_cost_loss": float(world_metrics["val_cost_loss"]),
        "val_cost_accuracy": float(world_metrics["val_cost_accuracy"]),
        "val_done_accuracy": float(world_metrics["val_done_accuracy"]),
        "final_imagined_return": float(imagined_returns[-1]),
        "real_success_rate": float(policy_metrics["real_success_rate"]),
        "real_mean_return": float(policy_metrics["real_mean_return"]),
        "real_mean_cost": float(policy_metrics["real_mean_cost"]),
        "training_curve_path": training_curve_path,
        "imagined_return_curve_path": imagined_return_curve_path,
        "real_rollout_plot_path": real_rollout_plot_path,
    }

    print("Final tiny Dreamer metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics

    #raise NotImplementedError("Run the tiny Dreamer-style training demo.")


def explain_tiny_dreamer() -> str:
    """Return a short explanation of the Dreamer-style loop."""
    return (
        "This is Dreamer-style because real data trains the world model, then "
        "the policy learns through imagined futures inside that model. The final "
        "policy is still checked in the real environment."
    )


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
    """Run the tiny Dreamer-style scaffold."""
    config = TinyDreamerConfig()
    metrics, ok = report_task_progress("run_tiny_dreamer_demo", run_tiny_dreamer_demo, config)
    if ok:
        print(metrics)
        print(explain_tiny_dreamer())


if __name__ == "__main__":
    main()
