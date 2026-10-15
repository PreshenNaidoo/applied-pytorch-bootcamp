"""
============================================================
EXERCISE 01: GRIDWORLD DYNAMICS MODEL
============================================================

WHAT YOU ARE BUILDING
You are building the first small world model in the bootcamp.

A world model predicts what the world will do next.

This exercise learns:

```text
current state + action -> next state, reward, cost, done
```

Plain meaning:
- current state: what the agent knows now
- action: what the agent does
- next state: what the environment looks like after the action
- reward: whether the action helped the task
- cost: how much safety cost the action produced
- done: whether the episode ended

WHY THIS EXERCISE EXISTS
Later exercises use image world models, latent world models, planning, and
long-horizon drift checks.

This file teaches the smallest useful version first:

```text
state vector + action id -> next state vector and task signals
```

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
robot or game environment           local visual grid-world
camera and sensor state             six-number grid state vector
motor or keyboard action            action id 0, 1, 2, or 3
future sensor state                 next six-number grid state vector
reward and safety signal            reward, cost, and done from the environment
large logged dataset                local logged rollouts from common.embodied_gridworld

WHY THIS IS NOT RANDOM SYNTHETIC DATA
The dataset is generated locally, but it is not random tensors and not fake
labels.

Every row comes from stepping the real project environment in
`common.embodied_gridworld`:

```text
environment state -> action -> environment returns next state, reward, cost, done
```

WHAT IS PROVIDED
`common.embodied_gridworld` provides:
- a grid world with walls, hazards, a goal, and an agent
- a real step rule
- a scripted data-collection policy
- rendered images
- encoded state vectors
- logged rollouts

This exercise uses encoded state vectors first.
Images come later in `world_models/05`, `world_models/08`, and the later video exercises.

WHAT THE MODEL KNOWS
For each training row, the model receives:
- the current state vector
- the action id

WHAT THE MODEL MUST PREDICT
The model predicts:
- the next state vector
- reward
- cost amount
- done

WHAT PROVIDES THE TARGET
The environment provides the targets:
- `step["next_state"]`
- `step["reward"]`
- `step["cost"]`
- `step["done"]`

Do not invent targets with random numbers.

FULL WORKFLOW

```text
1. collect logged rollouts from common.embodied_gridworld
2. split complete episodes into train and validation groups
3. flatten each group and build state + action inputs
4. train a model to predict next state, reward, cost, and done
5. evaluate on held-out transitions
6. save training curves and prediction charts
```

DATASET
Default dataset:
- no download
- use `common.embodied_gridworld`
- call `collect_visual_gridworld_episodes(...)`
- call `flatten_logged_episodes(...)`

One logged row contains:

```text
state: [6]
action: int
reward: float
cost: float
next_state: [6]
done: bool
```

State vector meaning:

```text
index  meaning
-------------------------------
0      agent row, normalized
1      agent column, normalized
2      goal row, normalized
3      goal column, normalized
4      first hazard row, normalized
5      first hazard column, normalized
```

Tiny concrete example:

```text
state = [0.00, 0.00, 1.00, 1.00, 0.50, 0.33]
action = 1
next_state = [0.17, 0.00, 1.00, 1.00, 0.50, 0.33]
reward = -0.01
cost = 0.0
done = False
```

INPUTS AND OUTPUTS
One example:
- state: `[state_dim]`
- action: scalar integer
- model input: `[state_dim + num_actions]`
- next state target: `[state_dim]`
- reward target: `[1]`
- cost target: `[1]`
- done target: `[1]`

One batch:
- inputs: `[batch_size, state_dim + num_actions]`
- next_state_targets: `[batch_size, state_dim]`
- reward_targets: `[batch_size, 1]`
- cost_targets: `[batch_size, 1]`
- done_targets: `[batch_size, 1]`

LOSS FUNCTION
Use one loss per prediction head:

```text
next_state_loss = mse(predicted_next_state, true_next_state)
reward_loss = mse(predicted_reward, true_reward)
cost_loss = mse(predicted_cost, true_cost)
done_loss = binary_cross_entropy_with_logits(done_logit, done_target)

total_loss = next_state_loss + reward_loss + cost_loss + done_loss
```

METRICS TO TRACK
- validation next-state MSE: lower is better
- validation reward MSE: lower is better
- validation cost MSE: lower is better
- validation done accuracy: higher is better
- validation cost precision, recall, and F1: higher is better
- validation done precision, recall, and F1: higher is better
- final training loss: lower is better

VISUALS TO CREATE
Save these files:
- `outputs/gridworld_dynamics/training_curves.png`
- `outputs/gridworld_dynamics/next_state_predictions.png`

TARGET RESULT RANGE
This is a small deterministic environment, so the model should learn quickly.

Useful signs:
- validation next-state MSE becomes small
- done accuracy is clearly better than always guessing one class
- train loss falls over epochs
- prediction plots roughly follow the diagonal

COMMON MISTAKES
- training on random tensors instead of environment transitions
- using current state as the target instead of next state
- shuffling inputs without shuffling targets the same way
- applying sigmoid in the model before `binary_cross_entropy_with_logits`
- reporting only training loss and no validation metrics
- not saving any visual, making it hard to inspect the model

STEP-BY-STEP TODO
1. Collect complete logged episodes from `common.embodied_gridworld`.
2. Split episodes, then build train and validation tensors.
3. Build a small multi-head model.
4. Train the model.
5. Evaluate validation metrics.
6. Save training curves.
7. Save a prediction chart.
8. Explain how this is the first state-based world model.

DEFINITION OF DONE
You are done when:
- data comes from real logged grid-world transitions
- no episode appears in both training and validation
- the model trains end to end after you complete the stubs
- next-state/reward metrics and cost/done precision, recall, and F1 are printed
- training curves are saved
- next-state prediction visuals are saved
- every completed stub keeps its original `raise NotImplementedError(...)` line commented out
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BOOTCAMP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BOOTCAMP_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.embodied_gridworld import  LoggedGridStep, VisualGridWorldConfig, collect_visual_gridworld_episodes, flatten_logged_episodes
from common.reproducibility import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"


# Learning order for this exercise:
# 1. Finish `encode_state_action`.
# 2. Finish `collect_logged_episodes`.
# 3. Finish `build_dynamics_dataset` and check tensor shapes.
# 4. Build the model and run one forward pass.
# 5. Finish the loss and training loop.
# 6. Finish validation metrics.
# 7. Save the two plots.
# 8. Run `main()` and read the metrics.


class DynamicsDatasetSplits(TypedDict):
    """Train and validation tensors for the dynamics model."""

    train_inputs: torch.Tensor
    train_next_states: torch.Tensor
    train_rewards: torch.Tensor
    train_costs: torch.Tensor
    train_dones: torch.Tensor
    val_inputs: torch.Tensor
    val_next_states: torch.Tensor
    val_rewards: torch.Tensor
    val_costs: torch.Tensor
    val_dones: torch.Tensor


class DynamicsMetrics(TypedDict):
    """Final metrics and visual paths."""

    final_train_loss: float
    val_next_state_mse: float
    val_reward_mse: float
    val_cost_mse: float
    val_done_accuracy: float
    val_cost_precision: float
    val_cost_recall: float
    val_cost_f1: float
    val_done_precision: float
    val_done_recall: float
    val_done_f1: float
    training_curve_path: str
    prediction_plot_path: str


@dataclass
class GridworldDynamicsConfig:
    """Settings for the grid-world dynamics model."""

    state_dim: int = 6
    num_actions: int = 4
    num_episodes: int = 256
    max_steps: int = 18
    image_size: int = 28
    batch_size: int = 64
    hidden_dim: int = 128
    epochs: int = 25
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    cost_event_threshold: float = 0.1
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "gridworld_dynamics"


def encode_state_action(state: torch.Tensor, action: int, config: GridworldDynamicsConfig) -> torch.Tensor:
    """Encode one state vector and action id as one model input."""
    # Where this fits:
    # This creates one model input row from one logged transition.
    #
    # Goal:
    # Concatenate the six-number state with a one-hot action vector.
    #
    # Input:
    # - `state`: tensor shaped `[state_dim]`, for example `[0.0, 0.0, 1.0, 1.0, 0.5, 0.33]`.
    # - `action`: integer from `0` to `num_actions - 1`.
    #
    # Output:
    # - tensor shaped `[state_dim + num_actions]`.
    # - with defaults, shape is `[10]`.
    #
    # Method:
    # 1. Check `state.shape == (config.state_dim,)`.
    # 2. Check `0 <= action < config.num_actions`.
    # 3. Build `action_one_hot = torch.zeros(config.num_actions, dtype=torch.float32)`.
    # 4. Set `action_one_hot[action] = 1.0`.
    # 5. Return `torch.cat([state.float(), action_one_hot], dim=0)`.
    #
    # Checks:
    # - The output should contain the original six state numbers.
    # - The last four values should contain exactly one `1.0`.

    raise NotImplementedError("Encode one state-action pair.")



    


def collect_logged_episodes(config: GridworldDynamicsConfig) -> list[list[LoggedGridStep]]:
    """Collect complete real episodes from the shared grid-world."""
    # Where this fits:
    # This is the data source for the whole exercise.
    #
    # Goal:
    # Return complete logged episodes from the real shared bootcamp environment.
    #
    # Method:
    # 1. Build:
    #    `grid_config = VisualGridWorldConfig(image_size=config.image_size, num_episodes=config.num_episodes, max_steps=config.max_steps, random_state=config.random_state)`.
    # 2. Collect:
    #    `episodes = collect_visual_gridworld_episodes(grid_config)`.
    # 3. Check the episode list and every kept episode are not empty.
    # 4. Return `episodes` without flattening them.
    #
    # Checks:
    # - Do not make random tensors here.
    # - The targets must come from the environment rows.
    # Keeping episode boundaries here matters. The dataset builder must split
    # episodes before it turns them into individual transition rows.
    
    raise NotImplementedError("Collect complete logged episodes from common.embodied_gridworld.")

    # If no episode ends early, this is 256 episodes * 18 steps = 4608 rows.



def build_dynamics_dataset(config: GridworldDynamicsConfig) -> DynamicsDatasetSplits:
    """Build train and validation tensors from logged transitions."""
    # Where this fits:
    # This turns environment logs into tensors the model can train on.
    #
    # Goal:
    # Build an episode-level train/validation split and keep inputs aligned
    # with targets.
    #
    # Method:
    # 1. Call `collect_logged_episodes(config)`.
    # 2. Shuffle episode indices once with
    #    `torch.randperm(len(episodes), generator=torch.Generator().manual_seed(config.random_state))`.
    # 3. Split the episode indices by `config.val_fraction`.
    # 4. Use `flatten_logged_episodes(...)` separately on the train episodes
    #    and validation episodes. Never flatten before splitting.
    # 5. For each step in each group, build:
    #    - input from `encode_state_action(...)`
    #    - next state from `step["next_state"]`
    #    - reward, raw environment cost, and done tensors
    # 6. Stack the train tensors and validation tensors separately.
    #
    # Checks:
    # - No episode may appear in both groups.
    # - Validation data should not be used for training.

    raise NotImplementedError("Build train and validation tensors.")






    # Now each step is one row like:
    # {
    #     "state": tensor shape [6],
    #     "action": int,
    #     "next_state": tensor shape [6],
    #     "reward": float,
    #     "cost": float,
    #     "done": bool,
    # }

    # train_inputs: torch.Tensor
    # train_next_states: torch.Tensor
    # train_rewards: torch.Tensor
    # train_costs: torch.Tensor
    # train_dones: torch.Tensor








def binary_precision_recall_f1(
    event_scores: torch.Tensor,
    event_targets: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Return precision, recall, and F1 for one rare binary event."""
    # Where this fits:
    # Cost and done events can be rare. Accuracy alone can look high when a
    # model predicts "no event" for every row.
    #
    # Input:
    # - `event_scores`: one score per validation row.
    #   Shape: `[num_rows, 1]`.
    #   Example:
    #   `[[0.80], [0.10], [0.65]]`
    # - `event_targets`: one true label per validation row.
    #   Shape: `[num_rows, 1]`, the same shape as `event_scores`.
    #   Example:
    #   `[[1.0], [0.0], [1.0]]`
    # - `threshold`: score at or above which an event is predicted.
    #
    # Output:
    # - `{"precision": float, "recall": float, "f1": float}`.
    #   These are plain Python floats, not tensors.
    #
    # Method:
    # 1. Predict an event with `event_scores >= threshold`.
    #    Shape stays `[num_rows, 1]`.
    #    Apply sigmoid before this helper when the model output is a logit.
    # 2. Count true positives, false positives, and false negatives.
    #    Each count should become one scalar value.
    # 3. Compute:
    #    `precision = tp / (tp + fp)`
    #    `recall = tp / (tp + fn)`
    #    `f1 = 2 * precision * recall / (precision + recall)`
    # 4. Use `0.0` when a denominator is zero.
    #
    # Check:
    # - A model that never finds a real event should have recall `0.0`.

    raise NotImplementedError("Compute precision, recall, and F1 for a binary event.")


    # & means both are true
    # ~ means not true
    # .sum() counts how many True values there are
    # .item() turns a one-value tensor into a normal Python number








class GridworldDynamicsModel(nn.Module):
    """Predict next state, reward, cost, and done."""

    def __init__(self, config: GridworldDynamicsConfig) -> None:
        # Where this fits:
        # This is the first state-based world model.
        #
        # Goal:
        # Build one shared trunk with four prediction heads.
        #
        # Method:
        # 1. Call `super().__init__()`.
        # 2. Set `input_dim = config.state_dim + config.num_actions`.
        # 3. Build `self.trunk = nn.Sequential(...)` with two Linear + ReLU blocks.
        # 4. Build heads for next state, reward, cost amount, and done.
        #
        # Checks:
        # - Cost is a normal regression output. Do not apply sigmoid to it.
        # - Done is a raw logit for `binary_cross_entropy_with_logits`.
        super().__init__()

        raise NotImplementedError("Build the grid-world dynamics model.")




    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return next state, reward, cost amount, and done logit."""
        # Goal:
        # Run the input through the shared trunk and four heads.
        #
        # Output order:
        # `(next_state, reward, cost, done_logit)`.

        raise NotImplementedError("Run the grid-world dynamics model.")





def compute_dynamics_loss(
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return total world-model loss and its parts."""
    # Where this fits:
    # This is the training objective. It tells the model what "wrong" means.
    #
    # Method:
    # 1. Compute next-state MSE with `nn.functional.mse_loss(...)`.
    # 2. Compute reward MSE.
    # 3. Compute cost MSE.
    # 4. Compute done loss with:
    #    `nn.functional.binary_cross_entropy_with_logits(pred_done_logit, done)`.
    # 5. Add the four losses.
    # 6. Return `(total_loss, loss_parts)`.

    # `predictions[0]` and `targets[0]` are shaped `[batch_size, 6]`.
    # `mse_loss` uses the mean by default, so it averages over all 6 state
    # values for every row in the batch.
    raise NotImplementedError("Compute dynamics-model loss.")





def train_one_epoch(model: GridworldDynamicsModel, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train for one epoch and return mean total loss."""
    # Where this fits:
    # This is the training loop for one pass over the training split.
    #
    # Method:
    # 1. Call `model.train()`.
    # 2. Loop over batches from the loader.
    #    The loader is built from a `TensorDataset` in `main()`.
    #    Each batch contains five tensors in this order:
    #    - `inputs`: `[batch_size, state_dim + num_actions]`, with defaults `[batch_size, 10]`
    #    - `next_states`: `[batch_size, state_dim]`, with defaults `[batch_size, 6]`
    #    - `rewards`: `[batch_size, 1]`
    #    - `costs`: `[batch_size, 1]`
    #    - `dones`: `[batch_size, 1]`
    # 3. Move each tensor to `device`.
    # 4. Run the model with `predictions = model(inputs)`.
    #    The output is `(pred_next_states, pred_rewards, pred_costs, pred_done_logits)`.
    # 5. Build targets as `(next_states, rewards, costs, dones)`.
    # 6. Compute loss with `compute_dynamics_loss(predictions, targets)`.
    # 7. Call `optimizer.zero_grad()`.
    # 8. Call `loss.backward()`.
    # 9. Call `optimizer.step()`.
    # 10. Return mean loss weighted by batch size.

    raise NotImplementedError("Train one dynamics-model epoch.")





@torch.no_grad()
def evaluate_dynamics_model(
    model: GridworldDynamicsModel,
    loader: DataLoader,
    config: GridworldDynamicsConfig,
) -> dict[str, float]:
    """Return validation metrics without updating the model."""
    # Where this fits:
    # This is the held-out check. It uses a validation `DataLoader`, like
    # `train_one_epoch`, but it only measures. It does not update the model.
    #
    # Key difference from training:
    # - call `model.eval()`
    # - do not call `loss.backward()`
    # - do not call `optimizer.step()`
    #
    # Each validation batch has the same order as the training batch:
    # `inputs, next_states, rewards, costs, dones`
    # - `inputs`: `[batch_size, state_dim + num_actions]`, default second size `10`
    # - `next_states`: `[batch_size, state_dim]`, default second size `6`
    # - `rewards`: `[batch_size, 1]`
    # - `costs`: `[batch_size, 1]`
    # - `dones`: `[batch_size, 1]`
    #
    # Plain-English metric meanings:
    # - next-state MSE: how far the predicted next state is from the real next state
    # - reward MSE: how far the predicted reward is from the real reward
    # - cost MSE: how far the predicted safety cost is from the real safety cost
    # - done accuracy: how often the model correctly predicts episode end or not end
    # - precision: when the model says "event happened", how often it was right
    # - recall: out of all real events, how many the model found
    # - F1: one score that balances precision and recall
    #
    # Output:
    # Return a dictionary with:
    # - `val_next_state_mse`
    # - `val_reward_mse`
    # - `val_cost_mse`
    # - `val_done_accuracy`
    # - `val_cost_precision`, `val_cost_recall`, `val_cost_f1`
    # - `val_done_precision`, `val_done_recall`, `val_done_f1`
    #
    # Method:
    # 1. Call `model.eval()`.
    # 2. Make empty lists for predictions and targets.
    # 3. Loop over batches from `loader`.
    # 4. Move each tensor to `device`.
    # 5. Run:
    #    `pred_next_states, pred_rewards, pred_costs, done_logits = model(inputs)`.
    # 6. Append each prediction and target tensor to its matching list.
    # 7. After the loop, join each list with `torch.cat(..., dim=0)`.
    #    After concatenation, shapes should be:
    #    - `pred_next_states`: `[num_val_rows, state_dim]`
    #    - `pred_rewards`: `[num_val_rows, 1]`
    #    - `pred_costs`: `[num_val_rows, 1]`
    #    - `done_logits`: `[num_val_rows, 1]`
    #    - `val_next_states`: `[num_val_rows, state_dim]`
    #    - `val_rewards`: `[num_val_rows, 1]`
    #    - `val_costs`: `[num_val_rows, 1]`
    #    - `val_dones`: `[num_val_rows, 1]`
    # 8. Compute MSE values with `nn.functional.mse_loss(...)`.
    #    Each MSE should become a Python float with `.item()`.
    # 9. Convert done logits to probabilities:
    #    `done_probs = torch.sigmoid(done_logits)`.
    # 10. Convert done probabilities to predicted labels:
    #    `done_preds = (done_probs >= 0.5).float()`.
    # 11. Compute done accuracy:
    #    `(done_preds == val_dones).float().mean().item()`.
    # 12. For cost event metrics:
    #    - event scores are `pred_costs`
    #    - event targets are `(val_costs > 0.0).float()`
    #    - threshold is `config.cost_event_threshold`
    #    - call `binary_precision_recall_f1(...)`
    # 13. For done event metrics:
    #    - event scores are `done_probs`
    #    - event targets are `val_dones`
    #    - threshold is `0.5`
    #    - call `binary_precision_recall_f1(...)`
    # 14. Return one dictionary with all metric names listed above.
    #
    # Check:
    # - This function should not call `backward()` or `step()`.
    raise NotImplementedError("Evaluate dynamics model on validation data.")











def save_training_curves(
    train_losses: list[float],
    val_next_state_mses: list[float],
    config: GridworldDynamicsConfig,
) -> str:
    """Save train loss and validation next-state MSE curves."""
    # Goal:
    # Save `outputs/gridworld_dynamics/training_curves.png`.
    #
    # Input:
    # - `train_losses`: one float per epoch, so length is usually `config.epochs`.
    # - `val_next_state_mses`: one float per epoch, same length as `train_losses`.
    # - Each value is already a scalar. There is no batch dimension here.
    #
    # Output:
    # - path string for the saved PNG file.
    #
    # Method:
    # Use Matplotlib with `matplotlib.use("Agg")`.
    raise NotImplementedError("Save training curve plot.")











def save_prediction_plot(
    model: GridworldDynamicsModel,
    splits: DynamicsDatasetSplits,
    config: GridworldDynamicsConfig,
) -> str:
    """Save a true-vs-predicted next-state plot."""
    # Goal:
    # Save `outputs/gridworld_dynamics/next_state_predictions.png`.
    #
    # Method:
    # Plot true next-state values on the x-axis and predicted values on the y-axis.
    raise NotImplementedError("Save next-state prediction plot.")















def train_dynamics_model(config: GridworldDynamicsConfig) -> DynamicsMetrics:
    """Train the grid-world dynamics model and save metrics and visuals."""
    # Where this fits:
    # This connects the full exercise.
    #
    # Method:
    # 1. Call `set_seed(config.random_state)`.
    # 2. Build `splits = build_dynamics_dataset(config)`.
    # 3. Build training and validation `TensorDataset` objects.
    # 4. Build a training `DataLoader` with `shuffle=True`.
    # 5. Build a validation `DataLoader` with `shuffle=False`.
    # 6. Build `GridworldDynamicsModel(config).to(device)`.
    # 7. Build `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 8. Train and evaluate for `config.epochs`.
    # 9. Save both visual files.
    # 10. Return `DynamicsMetrics`.
    raise NotImplementedError("Train and evaluate the grid-world dynamics model.")













def explain_first_world_model() -> str:
    """Return a short explanation of what this model learns."""
    return (
        "This is a world model because it learns the environment's next-step "
        "rule from logged transitions. Later exercises replace the clean state "
        "vector with images or latents, but the idea stays the same."
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
    """Run the grid-world dynamics scaffold."""
    config = GridworldDynamicsConfig()
    metrics, ok = report_task_progress("train_dynamics_model", train_dynamics_model, config)
    if ok:
        print(metrics)
        print(explain_first_world_model())


if __name__ == "__main__":
    main()
