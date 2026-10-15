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
    
    raise NotImplementedError("Apply the DreamerV3 symlog transform.")



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
    
    raise NotImplementedError("Undo the DreamerV3 symlog transform.")



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
    
    raise NotImplementedError("Build DreamerV3 two-hot value targets.")

    # keep every target value inside the range covered by bin_centers

    # Find the upper bin index for each clipped value
    # what is the first bin center at or above this value?
    # right=False means we want the first bin center at or above the value
    # clamp the upper bin indices between `1` and `num_bins - 1`





    # Shape note:
    # `targets` is 2D, so each write must choose a row and a column.
    # `lower_bin_indices` and `upper_bin_indices` only choose columns.
    # Before using `scatter_`, reshape index and weight tensors from `[batch]`
    # to `[batch, 1]` with `.unsqueeze(1)`.





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
    
    raise NotImplementedError("Collect complete real grid-world episodes.")





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

    raise NotImplementedError("One-hot encode a grid-world action.")

    


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

    raise NotImplementedError("Build Dreamer-style train and validation tensors.")










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

        raise NotImplementedError("Build the tiny Dreamer world model.")

        


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

        raise NotImplementedError("Run the tiny Dreamer world model.")






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

        raise NotImplementedError("Build the latent policy.")



    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        # Input:
        # - `latents`: flattened full states shaped `[batch_size, latent_dim]`.
        #
        # Output:
        # - action logits: `[batch_size, num_actions]`.

        #full_state=latents, this is the same

        raise NotImplementedError("Run the latent policy.")


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

    raise NotImplementedError("Compute tiny Dreamer world-model loss.")





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

    raise NotImplementedError("Train one tiny Dreamer world-model epoch.")






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

    raise NotImplementedError("Evaluate tiny Dreamer world-model metrics.")





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
    
    raise NotImplementedError("Train the tiny Dreamer world model.")









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

    raise NotImplementedError("Imagine a differentiable rollout.")






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

    raise NotImplementedError("Compute policy loss from imagined rewards and costs.")





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

    raise NotImplementedError("Train policy through imagined rollouts.")








def choose_policy_action(policy_model: LatentPolicy, latent: torch.Tensor) -> int:
    """Choose one real action for evaluation."""
    # Method:
    # 1. Add a batch dimension to `latent`.
    # 2. Run the policy with `torch.no_grad()`.
    # 3. Use `argmax(dim=-1)` to choose the highest-logit action.
    # 4. Return it as a Python int.
    #
    # This is okay for real evaluation because we are not training here.

    raise NotImplementedError("Choose one policy action for real evaluation.")




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
    
    raise NotImplementedError("Roll out learned policy in the real grid-world.")






def build_eval_scenes(config: TinyDreamerConfig) -> list[Scene]:
    """Build held-out real scenes for policy evaluation."""
    # Method:
    # 1. Build `grid_config = make_visual_grid_config(config)`.
    # 2. Build `rng = random.Random(config.random_state + 20_000)`.
    # 3. Call `make_grid_scene(grid_config, rng)` until you have `config.num_eval_scenes`.
    # 4. Return the scenes.
    
    raise NotImplementedError("Build held-out Dreamer evaluation scenes.")






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

    raise NotImplementedError("Evaluate tiny Dreamer policy in the real world.")






def save_training_curves(world_losses: list[float], val_next_agent_losses: list[float], config: TinyDreamerConfig) -> str:
    """Save world-model training curves."""
    # Goal:
    # Save `outputs/tiny_dreamer/training_curves.png`.
    #
    # Method:
    # Use Matplotlib with `matplotlib.use("Agg")`.
    # Plot world-model train loss and validation next-agent loss.
    raise NotImplementedError("Save tiny Dreamer training curves.")







def save_imagined_return_curve(imagined_returns: list[float], config: TinyDreamerConfig) -> str:
    """Save imagined policy return over training."""
    # Goal:
    # Save `outputs/tiny_dreamer/imagined_return_curve.png`.
    #
    # Method:
    # Plot policy epoch on the x-axis and imagined return on the y-axis.
    raise NotImplementedError("Save imagined return curve.")







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
    raise NotImplementedError("Save real policy rollout plot.")










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
    raise NotImplementedError("Run the tiny Dreamer-style training demo.")







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
