"""
============================================================
EXERCISE 07: WORLD MODEL WITH UNCERTAINTY
============================================================

WHAT YOU ARE BUILDING
You are building an ensemble of small world models.

An ensemble means several models learn the same task. If their predictions
disagree, we treat that as uncertainty.

The model task is:

```text
state + action -> next state, reward, cost, done
```

Plain meaning:
- state: what the agent knows now
- action: what the agent does
- next state: what happens after the action
- reward: whether the action helped
- cost: whether the action was unsafe
- done: whether the episode ended
- uncertainty: how much the ensemble disagrees

WHY THIS EXERCISE EXISTS
Earlier Phase 4 exercises train world models that give one answer.
Planning with one answer can be risky.

If a world model is wrong, a planner can choose actions that only look good
inside the model. An ensemble gives a simple warning:

```text
models agree -> lower uncertainty
models disagree -> higher uncertainty
```

REAL SYSTEM VS THIS EXERCISE

real system                         this exercise
------------------------------------------------------------
robot/game environment              shared visual grid-world
sensor state                        six-number grid-world state vector
motor/keyboard action               action id 0, 1, 2, or 3
large learned world models          small PyTorch ensemble
uncertainty warning                 prediction variance across models
risk-aware planning                 score actions with reward - cost - uncertainty

WHY THIS IS NOT RANDOM SYNTHETIC DATA
The data comes from `common.embodied_gridworld`.

Every row is a real logged transition:

```text
state, action -> next_state, reward, cost, done
```

Do not create random target tensors.
Do not use a separate local gridworld in this file.

FULL WORKFLOW

```text
1. collect real logged transitions
2. build train and validation tensors
3. train several world models
4. measure validation prediction quality
5. measure disagreement on held-out and changed maps
6. check whether high disagreement really means larger prediction error
7. use uncertainty to score candidate actions
8. save plots
```

DATASET
Default dataset:
- no download
- use `common.embodied_gridworld`
- call `collect_visual_gridworld_episodes(...)`
- call `flatten_logged_episodes(...)`

One row contains:

```text
state: [6]
action: int
next_state: [6]
reward: float
cost: float
done: bool
```

INPUTS AND OUTPUTS
One model input:
- state: `[6]`
- action one-hot: `[4]`
- input: `[10]`

One model output:
- next state: `[6]`
- reward: `[1]`
- cost: `[1]`
- done logit: `[1]`

UNCERTAINTY RULE
For a batch of next-state predictions:

```text
predictions shape = [ensemble_size, batch_size, state_dim]
uncertainty = mean variance across state dimensions
```

In PyTorch:

```python
uncertainty = stacked_predictions.var(dim=0).mean(dim=1)
```

METRICS TO TRACK
- validation next-state MSE: lower is better
- validation reward MSE: lower is better
- validation cost MSE: lower is better
- validation done accuracy: higher is better
- mean uncertainty: lower means more agreement
- high-uncertainty fraction: lower means fewer warning cases
- uncertainty-error correlation: higher means disagreement is a more useful warning
- low- versus high-uncertainty MSE: high-uncertainty rows should be harder
- unseen-map MSE and uncertainty: show how the ensemble behaves after a map change
- unseen-map uncertainty-error correlation: checks whether disagreement still
  ranks mistakes after the map rules change

VISUALS TO CREATE
Save these files:
- `outputs/world_model_uncertainty/ensemble_training_curves.png`
- `outputs/world_model_uncertainty/uncertainty_histogram.png`
- `outputs/world_model_uncertainty/uncertainty_vs_error.png`
- `outputs/world_model_uncertainty/risk_aware_action_scores.png`

COMMON MISTAKES
- training one model and calling it uncertainty
- giving every ensemble member the exact same training rows in the same order
- using random tensors instead of real transitions
- measuring only training loss
- treating low uncertainty as proof that the prediction is correct
- forgetting that uncertainty is about model disagreement
- reporting disagreement without checking whether it follows real prediction error
- assuming an ensemble must become uncertain whenever the world changes
- not saving a visual that shows where uncertainty is high

STEP-BY-STEP TODO
1. Collect real logged episodes.
2. Split complete episodes, then build train and validation tensors.
3. Build one world model.
4. Train each ensemble member on a different bootstrap sample.
5. Evaluate ensemble mean prediction on validation data.
6. Check whether disagreement follows prediction error.
7. Repeat the check on unseen map layouts.
8. Score actions with reward, cost, and uncertainty.
9. Save metrics and visuals.

DEFINITION OF DONE
You are done when:
- data comes from `common.embodied_gridworld`
- the ensemble has more than one model
- validation metrics are reported
- uncertainty is computed from disagreement between models
- high-uncertainty rows are compared with low-uncertainty rows
- uncertainty-error correlation is reported
- error and uncertainty are measured on unseen maps
- the exercise explains that all ensemble members can agree and still be wrong
- risk-aware action scores are produced
- all four visual files are saved
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

from common.embodied_gridworld import (
    LoggedGridStep,
    VisualGridWorldConfig,
    collect_visual_gridworld_episodes,
    flatten_logged_episodes,
)
from common.reproducibility import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"


# Learning order for this exercise:
# 1. Build the real transition tensors.
# 2. Build one world model.
# 3. Train several models as an ensemble.
# 4. Compute validation metrics and disagreement.
# 5. Build risk-aware action scores.
# 6. Save the four plots.


class UncertaintyDatasetSplits(TypedDict):
    """Train and validation tensors for ensemble world models."""

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
    val_states: torch.Tensor


class UncertaintyEvaluationBatch(TypedDict):
    """One evaluation-only batch with model inputs and real targets."""

    inputs: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor


class UncertaintyMetrics(TypedDict):
    """Final metrics and visual paths."""

    val_next_state_mse: float
    val_reward_mse: float
    val_cost_mse: float
    val_done_accuracy: float
    mean_uncertainty: float
    high_uncertainty_fraction: float
    uncertainty_error_correlation: float
    low_uncertainty_next_state_mse: float
    high_uncertainty_next_state_mse: float
    unseen_map_next_state_mse: float
    unseen_map_mean_uncertainty: float
    unseen_map_uncertainty_error_correlation: float
    training_curve_path: str
    uncertainty_histogram_path: str
    uncertainty_error_plot_path: str
    action_score_plot_path: str


@dataclass
class UncertaintyWorldModelConfig:
    """Settings for the uncertainty world-model exercise."""

    state_dim: int = 6
    num_actions: int = 4
    image_size: int = 28
    num_episodes: int = 320
    max_steps: int = 18
    ensemble_size: int = 4
    hidden_dim: int = 128
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    unseen_map_episodes: int = 80
    unseen_grid_size: int = 9
    unseen_wall_count: int = 8
    unseen_hazard_count: int = 3
    uncertainty_threshold: float = 0.002
    cost_weight: float = 1.0
    uncertainty_weight: float = 2.0
    random_state: int = 42
    output_dir: Path = PROJECT_ROOT / "outputs" / "world_model_uncertainty"


def collect_logged_episodes(config: UncertaintyWorldModelConfig) -> list[list[LoggedGridStep]]:
    """Collect complete real episodes from the shared grid-world."""
    # Where this fits:
    # This is the real dataset source.
    #
    # Method:
    # 1. Build:
    #    `grid_config = VisualGridWorldConfig(image_size=config.image_size, num_episodes=config.num_episodes, max_steps=config.max_steps, random_state=config.random_state)`.
    # 2. Collect:
    #    `episodes = collect_visual_gridworld_episodes(grid_config)`.
    # 3. Check the episode list and every kept episode are not empty.
    # 4. Return complete episodes without flattening them.
    #
    # Checks:
    # - Do not create random targets.
    # - Use the environment-provided `state`, `action`, `next_state`, `reward`, `cost`, and `done`.
    grid_config = VisualGridWorldConfig(
        image_size=config.image_size,
        num_episodes=config.num_episodes,
        max_steps=config.max_steps,
        random_state=config.random_state,
    )
    episodes = collect_visual_gridworld_episodes(grid_config)

    if not episodes:
        raise ValueError("No episodes were collected from the grid-world.")

    complete_episodes = [episode for episode in episodes if episode]
    if len(complete_episodes) != len(episodes):
        raise ValueError("The collected data contains an empty episode.")

    return complete_episodes
    # raise NotImplementedError("Collect complete episodes for uncertainty training.")


def encode_state_action(state: torch.Tensor, action: int, config: UncertaintyWorldModelConfig) -> torch.Tensor:
    """Encode one state and action as one model input."""
    # Goal:
    # Concatenate the six-number state vector with a one-hot action vector.
    #
    # Input:
    # - `state`: `[state_dim]`.
    # - `action`: integer from `0` to `num_actions - 1`.
    #
    # Output:
    # - tensor shaped `[state_dim + num_actions]`.
    #
    # Method:
    # 1. Check `state.shape == (config.state_dim,)`.
    # 2. Build `action_one_hot = torch.zeros(config.num_actions, dtype=torch.float32)`.
    # 3. Set `action_one_hot[action] = 1.0`.
    # 4. Return `torch.cat([state.float(), action_one_hot], dim=0)`.
    if state.shape != (config.state_dim,):
        raise ValueError(f"Expected state shape {(config.state_dim,)}, got {tuple(state.shape)}.")
    if not 0 <= int(action) < config.num_actions:
        raise ValueError(f"Expected action from 0 to {config.num_actions - 1}, got {action}.")

    action_one_hot = torch.zeros(config.num_actions, dtype=torch.float32)
    action_one_hot[int(action)] = 1.0
    return torch.cat([state.float(), action_one_hot], dim=0)
    # raise NotImplementedError("Encode one state-action input.")


def build_uncertainty_dataset(config: UncertaintyWorldModelConfig) -> UncertaintyDatasetSplits:
    """Build train and validation tensors from real transitions."""
    # Method:
    # 1. Call `collect_logged_episodes(config)`.
    # 2. Shuffle and split whole episodes by `config.val_fraction`.
    # 3. Flatten train episodes and validation episodes separately with
    #    `flatten_logged_episodes(...)`.
    # 4. Build model inputs with `encode_state_action(...)` for each group.
    # 5. Stack next states and convert reward, cost, and done to
    #    `[num_examples, 1]`.
    # 6. Also keep `val_states`, because action scoring needs raw current states.
    #
    # Checks:
    # - Inputs and targets must stay aligned.
    # - No episode may appear in both groups.
    episodes = collect_logged_episodes(config)
    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError(f"Expected val_fraction between 0 and 1, got {config.val_fraction}.")

    generator = torch.Generator().manual_seed(config.random_state)
    episode_order = torch.randperm(len(episodes), generator=generator).tolist()
    val_count = max(1, int(len(episodes) * config.val_fraction))
    if val_count >= len(episodes):
        raise ValueError("Need at least one training episode and one validation episode.")

    val_episode_ids = set(episode_order[:val_count])
    train_episodes = [episode for index, episode in enumerate(episodes) if index not in val_episode_ids]
    val_episodes = [episode for index, episode in enumerate(episodes) if index in val_episode_ids]

    train_steps = flatten_logged_episodes(train_episodes)
    val_steps = flatten_logged_episodes(val_episodes)
    if not train_steps or not val_steps:
        raise ValueError("Train and validation splits must both contain at least one transition.")

    def build_tensors(steps: list[LoggedGridStep]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.stack([encode_state_action(step["state"], step["action"], config) for step in steps])
        states = torch.stack([step["state"].float() for step in steps])
        next_states = torch.stack([step["next_state"].float() for step in steps])
        rewards = torch.tensor([step["reward"] for step in steps], dtype=torch.float32).unsqueeze(1)
        costs = torch.tensor([step["cost"] for step in steps], dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor([float(step["done"]) for step in steps], dtype=torch.float32).unsqueeze(1)
        return inputs, states, next_states, rewards, costs, dones

    train_inputs, _, train_next_states, train_rewards, train_costs, train_dones = build_tensors(train_steps)
    val_inputs, val_states, val_next_states, val_rewards, val_costs, val_dones = build_tensors(val_steps)

    return {
        "train_inputs": train_inputs,
        "train_next_states": train_next_states,
        "train_rewards": train_rewards,
        "train_costs": train_costs,
        "train_dones": train_dones,
        "val_inputs": val_inputs,
        "val_next_states": val_next_states,
        "val_rewards": val_rewards,
        "val_costs": val_costs,
        "val_dones": val_dones,
        "val_states": val_states,
    }
    # raise NotImplementedError("Build uncertainty train and validation tensors.")


class EnsembleWorldModel(nn.Module):
    """One world model inside the ensemble."""

    def __init__(self, config: UncertaintyWorldModelConfig) -> None:
        # Goal:
        # Build one model that predicts next state, reward, cost, and done.
        #
        # Method:
        # 1. Call `super().__init__()`.
        # 2. Build a shared trunk with two Linear + ReLU blocks.
        # 3. Build four heads:
        #    - next state head: `config.state_dim`
        #    - reward head: `1`
        #    - cost head: `1`
        #    - done head: `1`
        #
        # Check:
        # - Done head should return raw logits.
        super().__init__()
        input_dim = config.state_dim + config.num_actions
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.reward_head = nn.Linear(config.hidden_dim, 1)
        self.cost_head = nn.Linear(config.hidden_dim, 1)
        self.done_head = nn.Linear(config.hidden_dim, 1)
        # raise NotImplementedError("Build one ensemble world model.")

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return next state, reward, cost, and done logit."""
        # Input:
        # - `inputs`: `[batch_size, state_dim + num_actions]`.
        #
        # Output:
        # - next state: `[batch_size, state_dim]`
        # - reward: `[batch_size, 1]`
        # - cost: `[batch_size, 1]`
        # - done logit: `[batch_size, 1]`
        features = self.trunk(inputs)
        next_state = self.next_state_head(features)
        reward = self.reward_head(features)
        cost = self.cost_head(features)
        done_logit = self.done_head(features)
        return next_state, reward, cost, done_logit
        # raise NotImplementedError("Run one ensemble world model.")


def compute_model_loss(
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Return one model's training loss."""
    # Method:
    # 1. Compute next-state MSE.
    # 2. Compute reward MSE.
    # 3. Compute cost MSE.
    # 4. Compute done BCE with logits.
    # 5. Add all four losses.
    pred_next_states, pred_rewards, pred_costs, pred_done_logits = predictions
    target_next_states, target_rewards, target_costs, target_dones = targets

    next_state_loss = nn.functional.mse_loss(pred_next_states, target_next_states)
    reward_loss = nn.functional.mse_loss(pred_rewards, target_rewards)
    cost_loss = nn.functional.mse_loss(pred_costs, target_costs)
    done_loss = nn.functional.binary_cross_entropy_with_logits(pred_done_logits, target_dones)

    return next_state_loss + reward_loss + cost_loss + done_loss
    # raise NotImplementedError("Compute ensemble model loss.")


def train_one_member(model: EnsembleWorldModel, loader: DataLoader, config: UncertaintyWorldModelConfig, seed: int) -> list[float]:
    """Train one ensemble member and return epoch losses."""
    # Method:
    # 1. Call `torch.manual_seed(seed)`.
    # 2. Build `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 3. Train for `config.epochs`.
    # 4. Move tensors to `device`.
    # 5. Return one mean train loss per epoch.
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    epoch_losses: list[float] = []

    model.train()
    for _ in range(config.epochs):
        batch_losses: list[float] = []
        for inputs, next_states, rewards, costs, dones in loader:
            inputs = inputs.to(device)
            next_states = next_states.to(device)
            rewards = rewards.to(device)
            costs = costs.to(device)
            dones = dones.to(device)

            predictions = model(inputs)
            targets = (next_states, rewards, costs, dones)
            loss = compute_model_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu()))

        if not batch_losses:
            raise ValueError("Training loader produced no batches.")
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    return epoch_losses
    # raise NotImplementedError("Train one ensemble member.")


def train_ensemble(
    config: UncertaintyWorldModelConfig,
    splits: UncertaintyDatasetSplits,
) -> tuple[list[EnsembleWorldModel], list[list[float]]]:
    """Train several world models as an ensemble."""
    # Method:
    # 1. Check `config.ensemble_size > 1`.
    # 2. Build one base `TensorDataset` from the aligned training tensors.
    # 3. For each ensemble member:
    #    - use `torch.randint(0, num_train_rows, (num_train_rows,), generator=generator)`
    #      to draw row indices with replacement
    #    - index every training tensor with those same indices
    #    - build a new `TensorDataset` and shuffled `DataLoader` from that sample
    #    - build `EnsembleWorldModel(config).to(device)` with a different seed
    #    - train it with `train_one_member(...)`
    # 4. Return the models and their loss histories.
    #
    # Why sample with replacement:
    # Each member sees a slightly different version of the real training set.
    # This makes disagreement more useful than training identical copies.
    #
    # Checks:
    # - A bootstrap sample has the same number of rows as the train set.
    # - Repeated row indices are expected.
    # - Never draw rows from the validation episodes.
    if config.ensemble_size <= 1:
        raise ValueError("Ensemble uncertainty needs at least two models.")

    train_tensors = (
        splits["train_inputs"],
        splits["train_next_states"],
        splits["train_rewards"],
        splits["train_costs"],
        splits["train_dones"],
    )
    num_train_rows = train_tensors[0].shape[0]
    if num_train_rows == 0:
        raise ValueError("Training split must contain at least one row.")
    if any(tensor.shape[0] != num_train_rows for tensor in train_tensors):
        raise ValueError("All training tensors must have the same first dimension.")

    models: list[EnsembleWorldModel] = []
    loss_histories: list[list[float]] = []

    for member_index in range(config.ensemble_size):
        member_seed = config.random_state + member_index
        generator = torch.Generator().manual_seed(member_seed)
        row_indices = torch.randint(0, num_train_rows, (num_train_rows,), generator=generator)
        sampled_tensors = tuple(tensor[row_indices] for tensor in train_tensors)
        sampled_dataset = TensorDataset(*sampled_tensors)
        sampled_loader = DataLoader(
            sampled_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
        )

        torch.manual_seed(member_seed)
        model = EnsembleWorldModel(config).to(device)
        member_losses = train_one_member(model, sampled_loader, config, seed=member_seed)
        first_loss = member_losses[0]
        final_loss = member_losses[-1]
        loss_change = first_loss - final_loss
        print(
            f"ensemble member {member_index + 1}/{config.ensemble_size}: "
            f"final_train_loss={final_loss:.4f}, "
            f"loss_change={loss_change:.4f}, "
            f"bootstrap_rows={num_train_rows}"
        )

        models.append(model)
        loss_histories.append(member_losses)

    return models, loss_histories
    # raise NotImplementedError("Train an ensemble of world models.")


@torch.no_grad()
def ensemble_predict(
    models: list[EnsembleWorldModel],
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ensemble mean predictions and next-state uncertainty."""
    # Method:
    # 1. Put each model in eval mode.
    # 2. Collect each model's predictions.
    # 3. Stack next-state predictions into `[ensemble_size, batch_size, state_dim]`.
    # 4. Stack reward, cost, and done-logit predictions too.
    # 5. Average each prediction type across ensemble members.
    # 6. Compute uncertainty with:
    #    `next_state_stack.var(dim=0).mean(dim=1)`.
    #
    # Output:
    # - mean next state
    # - mean reward
    # - mean cost
    # - mean done logit
    # - uncertainty shaped `[batch_size]`
    if len(models) <= 1:
        raise ValueError("Ensemble prediction needs at least two models.")

    model_inputs = inputs.to(device)                                        # `[batch_size, state_dim + num_actions]`
    next_state_predictions: list[torch.Tensor] = []
    reward_predictions: list[torch.Tensor] = []
    cost_predictions: list[torch.Tensor] = []
    done_logit_predictions: list[torch.Tensor] = []

    for model in models:
        model.eval()
        # Each model returns one prediction for every input row.
        next_state, reward, cost, done_logit = model(model_inputs)
        next_state_predictions.append(next_state.cpu())                     # list of `[batch_size, state_dim]` tensors
        reward_predictions.append(reward.cpu())                             # list of `[batch_size, 1]` tensors
        cost_predictions.append(cost.cpu())                                 # list of `[batch_size, 1]` tensors
        done_logit_predictions.append(done_logit.cpu())                     # list of `[batch_size, 1]` tensors

    next_state_stack = torch.stack(next_state_predictions, dim=0)           # `[ensemble_size, batch_size, state_dim]`
    reward_stack = torch.stack(reward_predictions, dim=0)                   # `[ensemble_size, batch_size, 1]`
    cost_stack = torch.stack(cost_predictions, dim=0)                       # `[ensemble_size, batch_size, 1]`
    done_logit_stack = torch.stack(done_logit_predictions, dim=0)           # `[ensemble_size, batch_size, 1]`

    mean_next_state = next_state_stack.mean(dim=0)                          # `[batch_size, state_dim]`
    mean_reward = reward_stack.mean(dim=0)                                  # `[batch_size, 1]`
    mean_cost = cost_stack.mean(dim=0)                                      # `[batch_size, 1]`
    mean_done_logit = done_logit_stack.mean(dim=0)                          # `[batch_size, 1]`

    # Variance here means disagreement across ensemble members for each input row.
    # After `var(dim=0)`: `[batch_size, state_dim]`.
    # After `mean(dim=1)`: `[batch_size]`, one uncertainty score per input row.
    uncertainty = next_state_stack.var(dim=0).mean(dim=1)                   # `[batch_size]`

    return mean_next_state, mean_reward, mean_cost, mean_done_logit, uncertainty
    # raise NotImplementedError("Predict with ensemble uncertainty.")


def build_unseen_map_batch(config: UncertaintyWorldModelConfig) -> UncertaintyEvaluationBatch:
    """Collect evaluation data from larger, more crowded maps."""
    # Where this fits:
    # The normal validation set uses held-out episodes from the training map
    # settings. This batch changes the map size and obstacle counts. It checks
    # whether uncertainty rises when the model sees a different kind of map.
    #
    # Important limit:
    # The six-number state does not include every wall or hazard. The map can
    # therefore change in a way the model input does not reveal. Every ensemble
    # member may agree and still be wrong. Finding that failure is a valid and
    # important result; ensemble disagreement is a warning, not a guarantee.
    #
    # Method:
    # 1. Build:
    #    `unseen_config = VisualGridWorldConfig(`
    #    `    grid_size=config.unseen_grid_size,`
    #    `    image_size=config.image_size,`
    #    `    num_episodes=config.unseen_map_episodes,`
    #    `    max_steps=config.max_steps,`
    #    `    wall_count=config.unseen_wall_count,`
    #    `    hazard_count=config.unseen_hazard_count,`
    #    `    random_state=config.random_state + 10_000,`
    #    `)`.
    # 2. Collect and flatten those episodes. This data is evaluation-only.
    # 3. Encode each `(state, action)` with `encode_state_action(...)`.
    # 4. Stack the real next states and convert reward, cost, and done to
    #    tensors shaped `[num_examples, 1]`.
    # 5. Return an `UncertaintyEvaluationBatch`.
    #
    # Checks:
    # - Do not train any ensemble member on this batch.
    # - `inputs.shape[1] == config.state_dim + config.num_actions`.
    unseen_config = VisualGridWorldConfig(
        grid_size=config.unseen_grid_size,
        image_size=config.image_size,
        num_episodes=config.unseen_map_episodes,
        max_steps=config.max_steps,
        wall_count=config.unseen_wall_count,
        hazard_count=config.unseen_hazard_count,
        random_state=config.random_state + 10_000,
    )
    episodes = collect_visual_gridworld_episodes(unseen_config)
    steps = flatten_logged_episodes(episodes)
    if not steps:
        raise ValueError("No unseen-map transitions were collected.")

    inputs = torch.stack([encode_state_action(step["state"], step["action"], config) for step in steps])
    next_states = torch.stack([step["next_state"].float() for step in steps])
    rewards = torch.tensor([step["reward"] for step in steps], dtype=torch.float32).unsqueeze(1)
    costs = torch.tensor([step["cost"] for step in steps], dtype=torch.float32).unsqueeze(1)
    dones = torch.tensor([float(step["done"]) for step in steps], dtype=torch.float32).unsqueeze(1)

    if inputs.shape[1] != config.state_dim + config.num_actions:
        raise ValueError(f"Expected input width {config.state_dim + config.num_actions}, got {inputs.shape[1]}.")

    return {
        "inputs": inputs,
        "next_states": next_states,
        "rewards": rewards,
        "costs": costs,
        "dones": dones,
    }
    # raise NotImplementedError("Build the unseen-map evaluation batch.")


def summarize_uncertainty_and_error(
    next_state_predictions: torch.Tensor,
    next_state_targets: torch.Tensor,
    uncertainties: torch.Tensor,
) -> dict[str, float]:
    """Check whether larger disagreement follows larger real error."""
    # Input:
    # - predictions: `[num_examples, state_dim]`
    # - targets: `[num_examples, state_dim]`
    # - uncertainties: `[num_examples]`
    #
    # Method:
    # 1. Compute one squared error per row with
    #    `(predictions - targets).pow(2).mean(dim=1)`.
    # 2. Use `torch.corrcoef(torch.stack([uncertainties, row_errors]))[0, 1]`
    #    for the correlation. A positive value means larger disagreement often
    #    appears with larger real error.
    # 3. Find the median uncertainty with `uncertainties.median()`.
    # 4. Split rows into uncertainty at or below the median and above the median.
    # 5. Return `uncertainty_error_correlation`,
    #    `low_uncertainty_next_state_mse`, and
    #    `high_uncertainty_next_state_mse`.
    #
    # Checks:
    # - All three inputs must have the same first dimension.
    # - Handle a zero-variance correlation by returning `0.0`, not NaN.
    # - Usually high-uncertainty MSE should exceed low-uncertainty MSE.
    if next_state_predictions.shape != next_state_targets.shape:
        raise ValueError(
            "Predicted and target next states must have the same shape, "
            f"got {tuple(next_state_predictions.shape)} and {tuple(next_state_targets.shape)}."
        )
    if uncertainties.ndim != 1:
        raise ValueError(f"Expected uncertainties to have shape [num_examples], got {tuple(uncertainties.shape)}.")
    if next_state_predictions.shape[0] != uncertainties.shape[0]:
        raise ValueError("Predictions, targets, and uncertainties must have the same number of rows.")

    # Shapes:
    # - next_state_predictions: `[num_examples, state_dim]`
    # - next_state_targets: `[num_examples, state_dim]`
    # - row_errors: `[num_examples]`
    # This is the mean squared error between the mean predicted next state and
    # the real target next state, computed separately for each row.
    row_errors = (next_state_predictions - next_state_targets).pow(2).mean(dim=1)

    # Standard deviation measures spread. If every uncertainty value is the same,
    # there is no up or down pattern to compare with prediction error.
    uncertainty_std = uncertainties.std(unbiased=False)  # scalar: spread of `[num_examples]` uncertainty scores
    error_std = row_errors.std(unbiased=False)  # scalar: spread of `[num_examples]` row errors
    if float(uncertainty_std) == 0.0 or float(error_std) == 0.0:
        correlation = 0.0
    else:
        # `torch.stack([uncertainties, row_errors])` makes a `[2, num_rows]` tensor:
        # - first row: how much the ensemble disagreed on each example
        # - second row: how wrong the mean prediction was on each example
        # `torch.corrcoef(...)` returns a 2 by 2 table. Entry `[0, 1]` is the
        # correlation between disagreement and real error.
        # Positive means higher disagreement usually comes with higher error.
        # Near 0 means disagreement is not a useful warning on this data.
        # Negative means higher disagreement often comes with lower error.
        # So this correlation is between:
        # - next-state MSE per row
        # - ensemble prediction variance per row
        # In plain words, it asks:
        # "When the ensemble disagrees more, is its next-state prediction also
        # more wrong?"
        stacked_for_correlation = torch.stack([uncertainties, row_errors])  # `[2, num_examples]`
        correlation_value = torch.corrcoef(stacked_for_correlation)[0, 1]  # scalar from a `[2, 2]` table
        correlation = 0.0 if torch.isnan(correlation_value) else float(correlation_value)

    median_uncertainty = uncertainties.median()  # scalar: middle uncertainty value
    low_mask = uncertainties <= median_uncertainty  # `[num_examples]` bool mask
    high_mask = uncertainties > median_uncertainty  # `[num_examples]` bool mask

    # `low_mask` keeps rows with uncertainty at or below the median.
    # This gives the average real error for the lower-uncertainty half.
    # The `else` is a safety fallback in case the mask selects no rows.
    low_mse = row_errors[low_mask].mean() if low_mask.any() else row_errors.mean()  # scalar
    # `high_mask` keeps rows with uncertainty above the median.
    # This gives the average real error for the higher-uncertainty half.
    # The `else` is a safety fallback in case the mask selects no rows.
    high_mse = row_errors[high_mask].mean() if high_mask.any() else row_errors.mean()  # scalar

    return {
        "uncertainty_error_correlation": correlation,
        "low_uncertainty_next_state_mse": float(low_mse),
        "high_uncertainty_next_state_mse": float(high_mse),
    }
    # raise NotImplementedError("Relate ensemble disagreement to real error.")


def evaluate_ensemble(
    models: list[EnsembleWorldModel],
    splits: UncertaintyDatasetSplits,
    config: UncertaintyWorldModelConfig,
) -> dict[str, float]:
    """Evaluate prediction quality and uncertainty on validation data."""
    # Method:
    # 1. Run `ensemble_predict(...)` on validation inputs.
    # 2. Compute next-state, reward, and cost MSE.
    # 3. Compute done accuracy with sigmoid threshold `0.5`.
    # 4. Compute mean uncertainty and high-uncertainty fraction.
    # 5. Call `summarize_uncertainty_and_error(...)` on validation rows.
    # 6. Call `build_unseen_map_batch(config)`.
    # 7. Run `ensemble_predict(...)` on that evaluation-only batch and compute:
    #    - `unseen_map_next_state_mse`
    #    - `unseen_map_mean_uncertainty`
    # 8. Call `summarize_uncertainty_and_error(...)` on the unseen-map
    #    predictions and store its correlation as
    #    `unseen_map_uncertainty_error_correlation`.
    # 9. Return all metrics.
    #
    # Interpretation:
    # A useful warning signal has positive uncertainty-error correlation and
    # higher MSE in the high-uncertainty half. The unseen-map check should often
    # raise both error and uncertainty, but do not assume it always will.
    mean_next_state, mean_reward, mean_cost, mean_done_logit, uncertainties = ensemble_predict(
        models,
        splits["val_inputs"],
    )

    val_next_state_mse = nn.functional.mse_loss(mean_next_state, splits["val_next_states"])
    val_reward_mse = nn.functional.mse_loss(mean_reward, splits["val_rewards"])
    val_cost_mse = nn.functional.mse_loss(mean_cost, splits["val_costs"])
    done_predictions = (torch.sigmoid(mean_done_logit) >= 0.5).float()
    val_done_accuracy = (done_predictions == splits["val_dones"]).float().mean()
    mean_uncertainty = uncertainties.mean()
    high_uncertainty_fraction = (uncertainties > config.uncertainty_threshold).float().mean()

    uncertainty_summary = summarize_uncertainty_and_error(
        mean_next_state,
        splits["val_next_states"],
        uncertainties,
    )

    unseen_batch = build_unseen_map_batch(config)
    unseen_next_state, _, _, _, unseen_uncertainties = ensemble_predict(models, unseen_batch["inputs"])
    unseen_map_next_state_mse = nn.functional.mse_loss(unseen_next_state, unseen_batch["next_states"])
    unseen_map_mean_uncertainty = unseen_uncertainties.mean()
    
    unseen_summary = summarize_uncertainty_and_error(
        unseen_next_state,
        unseen_batch["next_states"],
        unseen_uncertainties,
    )

    metrics = {
        "val_next_state_mse": float(val_next_state_mse),
        "val_reward_mse": float(val_reward_mse),
        "val_cost_mse": float(val_cost_mse),
        "val_done_accuracy": float(val_done_accuracy),
        "mean_uncertainty": float(mean_uncertainty),
        "high_uncertainty_fraction": float(high_uncertainty_fraction),
        "uncertainty_error_correlation": uncertainty_summary["uncertainty_error_correlation"],
        "low_uncertainty_next_state_mse": uncertainty_summary["low_uncertainty_next_state_mse"],
        "high_uncertainty_next_state_mse": uncertainty_summary["high_uncertainty_next_state_mse"],
        "unseen_map_next_state_mse": float(unseen_map_next_state_mse),
        "unseen_map_mean_uncertainty": float(unseen_map_mean_uncertainty),
        "unseen_map_uncertainty_error_correlation": unseen_summary["uncertainty_error_correlation"],
    }

    print(
        "validation metrics: "
        f"next_state_mse={metrics['val_next_state_mse']:.4f}, "
        f"reward_mse={metrics['val_reward_mse']:.4f}, "
        f"cost_mse={metrics['val_cost_mse']:.4f}, "
        f"done_accuracy={metrics['val_done_accuracy']:.3f}"
    )
    print(
        "uncertainty metrics: "
        f"mean={metrics['mean_uncertainty']:.6f}, "
        f"high_fraction={metrics['high_uncertainty_fraction']:.3f}, "
        f"error_correlation={metrics['uncertainty_error_correlation']:.3f}, "
        f"low_mse={metrics['low_uncertainty_next_state_mse']:.4f}, "
        f"high_mse={metrics['high_uncertainty_next_state_mse']:.4f}"
    )
    print(
        "unseen-map metrics: "
        f"next_state_mse={metrics['unseen_map_next_state_mse']:.4f}, "
        f"mean_uncertainty={metrics['unseen_map_mean_uncertainty']:.6f}, "
        f"error_correlation={metrics['unseen_map_uncertainty_error_correlation']:.3f}"
    )

    return metrics
    # raise NotImplementedError("Evaluate ensemble world-model uncertainty.")


def score_actions_with_uncertainty(
    models: list[EnsembleWorldModel],
    state: torch.Tensor,
    config: UncertaintyWorldModelConfig,
) -> dict[int, float]:
    """Score all actions using reward, cost, and uncertainty."""
    # Where this fits:
    # This is the small risk-aware planning check.
    #
    # Formula:
    # `safe_cost = max(predicted_cost, 0.0)`
    # `score = predicted_reward - cost_weight * safe_cost - uncertainty_weight * uncertainty`.
    #
    # Method:
    # 1. For every action id, encode `(state, action)`.
    # 2. Run `ensemble_predict(...)`.
    # 3. Clamp predicted cost at zero before computing the score. This stops a
    #    negative regression mistake from becoming a fake safety bonus.
    # 4. Return a dictionary from action id to score.
    #
    # Check:
    # - Higher score means the action looks better after uncertainty penalty.
    action_scores: dict[int, float] = {}
    for action in range(config.num_actions):
        encoded = encode_state_action(state, action, config).unsqueeze(0)
        _, reward, cost, _, uncertainty = ensemble_predict(models, encoded)
        safe_cost = torch.clamp(cost.squeeze(), min=0.0)
        score = reward.squeeze() - config.cost_weight * safe_cost - config.uncertainty_weight * uncertainty.squeeze()
        action_scores[action] = float(score)

    print("risk-aware action scores:", {action: round(score, 4) for action, score in action_scores.items()})
    return action_scores
    # raise NotImplementedError("Score actions with uncertainty penalty.")


def save_training_curves(loss_histories: list[list[float]], config: UncertaintyWorldModelConfig) -> str:
    """Save ensemble training curves."""
    # Goal:
    # Save `outputs/world_model_uncertainty/ensemble_training_curves.png`.
    #
    # Method:
    # Use Matplotlib with `matplotlib.use("Agg")`.
    # Plot one line per ensemble member.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "ensemble_training_curves.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    for member_index, losses in enumerate(loss_histories):
        ax.plot(range(1, len(losses) + 1), losses, label=f"member {member_index + 1}")
    ax.set_title("Ensemble training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"saved training curves: {path}")
    return str(path)
    # raise NotImplementedError("Save ensemble training curves.")


def save_uncertainty_histogram(uncertainties: torch.Tensor, config: UncertaintyWorldModelConfig) -> str:
    """Save validation uncertainty histogram."""
    # Goal:
    # Save `outputs/world_model_uncertainty/uncertainty_histogram.png`.
    #
    # Method:
    # Plot the uncertainty values from validation examples.
    # Draw a vertical line at `config.uncertainty_threshold`.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "uncertainty_histogram.png"
    uncertainty_values = uncertainties.detach().cpu()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(uncertainty_values.numpy(), bins=30)
    ax.axvline(config.uncertainty_threshold, color="red", linestyle="--", label="threshold")
    ax.set_title("Validation uncertainty")
    ax.set_xlabel("ensemble next-state variance")
    ax.set_ylabel("rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"saved uncertainty histogram: {path}")
    return str(path)
    # raise NotImplementedError("Save uncertainty histogram.")


def save_uncertainty_error_plot(
    next_state_predictions: torch.Tensor,
    next_state_targets: torch.Tensor,
    uncertainties: torch.Tensor,
    config: UncertaintyWorldModelConfig,
) -> str:
    """Save disagreement against the real next-state error."""
    # Goal:
    # Save `outputs/world_model_uncertainty/uncertainty_vs_error.png`.
    #
    # Method:
    # 1. Compute one next-state MSE per row with
    #    `(predictions - targets).pow(2).mean(dim=1)`.
    # 2. Make a scatter plot with uncertainty on the x-axis and real error on
    #    the y-axis.
    # 3. Put the correlation from `summarize_uncertainty_and_error(...)` in the
    #    title.
    #
    # Meaning:
    # A useful warning signal usually slopes upward: more disagreement comes
    # with more real error. The plot can reveal failures hidden by one number.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "uncertainty_vs_error.png"

    predictions = next_state_predictions.detach().cpu()
    targets = next_state_targets.detach().cpu()
    uncertainty_values = uncertainties.detach().cpu()
    row_errors = (predictions - targets).pow(2).mean(dim=1)
    summary = summarize_uncertainty_and_error(predictions, targets, uncertainty_values)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(uncertainty_values.numpy(), row_errors.numpy(), alpha=0.7)
    ax.set_title(f"Uncertainty vs error, corr={summary['uncertainty_error_correlation']:.3f}")
    ax.set_xlabel("ensemble next-state variance")
    ax.set_ylabel("real next-state MSE")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"saved uncertainty-error plot: {path}")
    return str(path)
    # raise NotImplementedError("Save uncertainty against real error.")


def save_action_score_plot(action_scores: dict[int, float], config: UncertaintyWorldModelConfig) -> str:
    """Save risk-aware action score bars."""
    # Goal:
    # Save `outputs/world_model_uncertainty/risk_aware_action_scores.png`.
    #
    # Method:
    # Build a bar chart with action id on the x-axis and risk-aware score on the y-axis.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / "risk_aware_action_scores.png"
    actions = sorted(action_scores)
    scores = [action_scores[action] for action in actions]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(action) for action in actions], scores)
    ax.set_title("Risk-aware action scores")
    ax.set_xlabel("action id")
    ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"saved action score plot: {path}")
    return str(path)
    # raise NotImplementedError("Save risk-aware action score plot.")


def run_uncertainty_demo(config: UncertaintyWorldModelConfig) -> UncertaintyMetrics:
    """Run the full uncertainty exercise."""
    # Method:
    # 1. Call `set_seed(config.random_state)`.
    # 2. Build dataset splits.
    # 3. Train the ensemble.
    # 4. Evaluate held-out episodes and unseen-map episodes.
    # 5. Score all actions for one validation state.
    # 6. Save the four visual files, including uncertainty against real error.
    # 7. Return an `UncertaintyMetrics` dictionary.
    set_seed(config.random_state)
    splits = build_uncertainty_dataset(config)
    print(
        "dataset rows: "
        f"train={splits['train_inputs'].shape[0]}, "
        f"validation={splits['val_inputs'].shape[0]}"
    )

    models, loss_histories = train_ensemble(config, splits)
    metric_values = evaluate_ensemble(models, splits, config)
    mean_next_state, _, _, _, uncertainties = ensemble_predict(models, splits["val_inputs"])
    action_scores = score_actions_with_uncertainty(models, splits["val_states"][0], config)

    training_curve_path = save_training_curves(loss_histories, config)
    uncertainty_histogram_path = save_uncertainty_histogram(uncertainties, config)
    uncertainty_error_plot_path = save_uncertainty_error_plot(
        mean_next_state,
        splits["val_next_states"],
        uncertainties,
        config,
    )
    action_score_plot_path = save_action_score_plot(action_scores, config)

    return {
        **metric_values,
        "training_curve_path": training_curve_path,
        "uncertainty_histogram_path": uncertainty_histogram_path,
        "uncertainty_error_plot_path": uncertainty_error_plot_path,
        "action_score_plot_path": action_score_plot_path,
    }
    # raise NotImplementedError("Run world-model uncertainty demo.")


def explain_uncertainty() -> str:
    """Return a short explanation of ensemble uncertainty."""
    return (
        "This exercise estimates uncertainty by training several world models "
        "and measuring how much their next-state predictions disagree. High "
        "disagreement is a warning that planning through that prediction may be risky."
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
    """Run the world-model uncertainty scaffold."""
    config = UncertaintyWorldModelConfig()
    metrics, ok = report_task_progress("run_uncertainty_demo", run_uncertainty_demo, config)
    if ok:
        print(metrics)
        print(explain_uncertainty())


if __name__ == "__main__":
    main()
