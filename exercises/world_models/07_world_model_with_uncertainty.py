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
    raise NotImplementedError("Collect complete episodes for uncertainty training.")





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
    raise NotImplementedError("Encode one state-action input.")



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
    raise NotImplementedError("Build uncertainty train and validation tensors.")








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
        raise NotImplementedError("Build one ensemble world model.")

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
        raise NotImplementedError("Run one ensemble world model.")


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
    raise NotImplementedError("Compute ensemble model loss.")




def train_one_member(model: EnsembleWorldModel, loader: DataLoader, config: UncertaintyWorldModelConfig, seed: int) -> list[float]:
    """Train one ensemble member and return epoch losses."""
    # Method:
    # 1. Call `torch.manual_seed(seed)`.
    # 2. Build `torch.optim.AdamW(model.parameters(), lr=config.learning_rate)`.
    # 3. Train for `config.epochs`.
    # 4. Move tensors to `device`.
    # 5. Return one mean train loss per epoch.
    raise NotImplementedError("Train one ensemble member.")




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
    raise NotImplementedError("Train an ensemble of world models.")






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
    raise NotImplementedError("Predict with ensemble uncertainty.")





    # Variance here means disagreement across ensemble members for each input row.
    # After `var(dim=0)`: `[batch_size, state_dim]`.
    # After `mean(dim=1)`: `[batch_size]`, one uncertainty score per input row.



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
    raise NotImplementedError("Build the unseen-map evaluation batch.")





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
    raise NotImplementedError("Relate ensemble disagreement to real error.")

    # Shapes:
    # - next_state_predictions: `[num_examples, state_dim]`
    # - next_state_targets: `[num_examples, state_dim]`
    # - row_errors: `[num_examples]`
    # This is the mean squared error between the mean predicted next state and
    # the real target next state, computed separately for each row.

    # Standard deviation measures spread. If every uncertainty value is the same,
    # there is no up or down pattern to compare with prediction error.


    # `low_mask` keeps rows with uncertainty at or below the median.
    # This gives the average real error for the lower-uncertainty half.
    # The `else` is a safety fallback in case the mask selects no rows.
    # `high_mask` keeps rows with uncertainty above the median.
    # This gives the average real error for the higher-uncertainty half.
    # The `else` is a safety fallback in case the mask selects no rows.



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
    raise NotImplementedError("Evaluate ensemble world-model uncertainty.")



    





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
    raise NotImplementedError("Score actions with uncertainty penalty.")



def save_training_curves(loss_histories: list[list[float]], config: UncertaintyWorldModelConfig) -> str:
    """Save ensemble training curves."""
    # Goal:
    # Save `outputs/world_model_uncertainty/ensemble_training_curves.png`.
    #
    # Method:
    # Use Matplotlib with `matplotlib.use("Agg")`.
    # Plot one line per ensemble member.
    raise NotImplementedError("Save ensemble training curves.")






def save_uncertainty_histogram(uncertainties: torch.Tensor, config: UncertaintyWorldModelConfig) -> str:
    """Save validation uncertainty histogram."""
    # Goal:
    # Save `outputs/world_model_uncertainty/uncertainty_histogram.png`.
    #
    # Method:
    # Plot the uncertainty values from validation examples.
    # Draw a vertical line at `config.uncertainty_threshold`.
    raise NotImplementedError("Save uncertainty histogram.")






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
    raise NotImplementedError("Save uncertainty against real error.")







def save_action_score_plot(action_scores: dict[int, float], config: UncertaintyWorldModelConfig) -> str:
    """Save risk-aware action score bars."""
    # Goal:
    # Save `outputs/world_model_uncertainty/risk_aware_action_scores.png`.
    #
    # Method:
    # Build a bar chart with action id on the x-axis and risk-aware score on the y-axis.
    raise NotImplementedError("Save risk-aware action score plot.")






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
    raise NotImplementedError("Run world-model uncertainty demo.")





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
